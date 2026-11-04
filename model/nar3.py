"""System 1 v3: one-pass memory-grammar extractor with open label sets (ontology/grammar.py).

backbone -> proj -> bidirectional adapter -> token states H (L x d). Every span <= MAX_SPAN tokens gets a rep s.

  entity   [none_e(s), q_e(s) . T_k]       over fine entity types k (open, from descriptions) + coarse head (43)
  trigger  [none_t(s), q_t(s) . F_f]       over frames f (FrameNet + memory frames, from descriptions)
  role     [none_r(p), r(p) . R_{f,j}]     for a trigger t of frame f and an argument a (entity, USER or another
                                           trigger): p = pair(t, a); only the roles j of frame f compete
  link     entity mention -> NEW or one of the [STATE] slots
  attrs    per trigger: modality, tense, source, sensitivity, importance, update_of_prior, about_user
  sal      window salience

Label vectors: T/F/R = LayerNorm(W . e(description)) + residual + bias, where e() is the frozen BASE backbone
mean-pooled over the label's description text (computed once, stored as a buffer). A label never seen in training
still gets a meaningful vector from its description; frequent labels refine theirs through the residual.

Partial windows (FrameNet annotations) train only frame identification at gold triggers and roles at those
triggers; nothing in them is used as a negative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.featurize import MAX_SPAN, MAX_STATE, _char_to_token, render
from model.nar import NARMemory, brier, load_backbone, mlp
from ontology.ontology import ATTRIBUTES, ENTITY_LIST, ENTITY_TYPES, FLAGS, USER
from schema.memory_schema import Window

ATTR_KEYS = list(ATTRIBUTES) + ["importance"]
ATTR_VALUES = {**ATTRIBUTES, "importance": ("low", "medium", "high")}
COARSE2ID = {t: i + 1 for i, t in enumerate(ENTITY_LIST)}  # 0 = NONE


def candidate_spans(utt_mask: torch.Tensor, word_start: torch.Tensor | None = None, word_end: torch.Tensor | None = None):
    """Spans <= MAX_SPAN tokens inside one utterance run that start and end on word boundaries."""
    spans = NARMemory.candidate_spans(utt_mask)
    if word_start is None:
        return spans
    ws, we = word_start.tolist(), word_end.tolist()
    return [(s, e) for s, e in spans if ws[s] and we[e]]


N_DIST = 16


def dist_bucket(d: torch.Tensor) -> torch.Tensor:
    """Signed token distance -> 0..N_DIST-2 (log buckets, 7 each side + 0); N_DIST-1 is reserved for USER."""
    a = d.abs().clamp(min=1).float().log2().floor().clamp(max=6).long() + 1
    a = torch.where(d == 0, torch.zeros_like(a), a)
    return torch.where(d < 0, a + 7, a).clamp(max=N_DIST - 2)


# ---------------------------------------------------------------------------------------------------- label sets
class LabelSet:
    """Frames, (frame, role) pairs and fine entity types, with the text each label vector is encoded from."""

    def __init__(self, frames: dict, types: dict[str, str]):
        self.frames = sorted(frames)
        self.frame_id = {f: i for i, f in enumerate(self.frames)}
        self.roles: list[tuple[str, str]] = []
        self.frame_roles: list[list[int]] = []
        for f in self.frames:
            ids = []
            for r in sorted(frames[f]["roles"]):
                ids.append(len(self.roles))
                self.roles.append((f, r))
            self.frame_roles.append(ids)
        self.role_id = {fr: i for i, fr in enumerate(self.roles)}
        self.types = sorted(types)
        self.type_id = {t: i for i, t in enumerate(self.types)}
        self.type_coarse = types
        self.frame_text = [f"{f.replace('_', ' ')}: {frames[f]['def']} Words: "
                           + ", ".join(w.rsplit('.', 1)[0] for w in frames[f]["words"][:12]) for f in self.frames]
        self.role_text = [f"{r.replace('_', ' ')} of {f.replace('_', ' ')}: {frames[f]['roles'][r]}"
                          for f, r in self.roles]
        self.type_text = [f"{t}: a kind of {c.lower()} ({ENTITY_TYPES.get(c, ('', ''))[1]})"
                          for t, c in ((t, types[t]) for t in self.types)]
        self.max_roles = max(len(x) for x in self.frame_roles)

    def to_json(self) -> dict:
        return {"frames": self.frames, "roles": self.roles, "frame_roles": self.frame_roles, "types": self.types,
                "type_coarse": self.type_coarse}

    @classmethod
    def from_json(cls, d: dict, frames: dict) -> "LabelSet":
        return cls({f: frames[f] for f in d["frames"]}, d["type_coarse"])


def norm_type(t: str | None) -> str | None:
    if not t:
        return None
    t = " ".join(t.lower().replace("_", " ").split())[:40]
    if t.endswith("ies") and len(t) > 5:
        t = t[:-3] + "y"
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 4:
        t = t[:-1]
    return t


def build_types(records: list[dict], min_count: int = 2) -> dict[str, str]:
    """Fine types seen >= min_count times in teacher labels (with their majority coarse type) + one fallback per
    coarse type (its lowercased name)."""
    from collections import Counter, defaultdict
    cnt: Counter = Counter()
    coarse: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        for e in r["labels"]["entities"]:
            t = norm_type(e.get("type"))
            if t and e.get("coarse") in ENTITY_TYPES:
                cnt[t] += 1
                coarse[t][e["coarse"]] += 1
    types = {t: coarse[t].most_common(1)[0][0] for t, n in cnt.items() if n >= min_count}
    for c in ENTITY_LIST:
        types.setdefault(c.lower(), c)
    return types


# ---------------------------------------------------------------------------------------------------- features
@dataclass
class Example3:
    input_ids: torch.Tensor
    slot_idx: torch.Tensor
    user_idx: int
    utt_mask: torch.Tensor
    word_start: torch.Tensor | None = None
    word_end: torch.Tensor | None = None
    tok_utt: torch.Tensor | None = None                                        # utterance index per token (-1 outside)
    spk_names: list[str] = field(default_factory=list)                         # other speakers ("S2", ...)
    spk_tok: list[list[int]] = field(default_factory=list)                     # token idx of each speaker's label
    utt_spk: torch.Tensor | None = None                                        # per utterance: 0 USER, k+1 speaker k
    partial: bool = False
    ents: list[tuple[int, int, int, int, int]] = field(default_factory=list)   # ts, te, type id|-1, coarse id|-1, slot
    trgs: list[tuple[int, int, int]] = field(default_factory=list)             # ts, te, frame id
    args: list[tuple[int, int]] = field(default_factory=list)                  # extra (non-entity) argument spans
    roles: list[tuple[int, int, int, int]] = field(default_factory=list)       # trigger i, arg kind, arg i, role pos
                                                                               # kinds: 0 ent, 1 USER, 2 trg, 3 arg, 4 speaker
    attrs: list[dict | None] = field(default_factory=list)                     # per trigger
    salience: float | None = None
    meta: dict = field(default_factory=dict)


def featurize3(rec: dict, tok, labels: LabelSet, max_len: int = 1024) -> Example3 | None:
    """rec = {"window": Window dict, "labels": {...}, "partial": bool}. Arg kinds: 0 entity, 1 USER, 2 trigger,
    3 extra argument span (FrameNet role fillers that are not entities)."""
    w = Window(**{k: v for k, v in rec["window"].items() if k in Window.model_fields})
    lab = rec.get("labels") or {"entities": [], "frames": []}
    partial = bool(rec.get("partial"))
    max_state = MAX_STATE
    while True:
        r = render(w, max_state)
        enc = tok(r.text, return_offsets_mapping=True, add_special_tokens=True)
        if len(enc["input_ids"]) <= max_len or max_state == 0:
            break
        max_state = max(0, max_state - 8)
    ids = enc["input_ids"][:max_len]
    find = _char_to_token([tuple(o) for o in enc["offset_mapping"][:max_len]])

    def tok_at(c):
        sp = find(c, c + 1)
        return sp[0] if sp else 0

    utt_mask = torch.zeros(len(ids), dtype=torch.bool)
    tok_utt = torch.full((len(ids),), -1, dtype=torch.long)
    for ui, u in enumerate(w.utterances):
        cs = r.utt_char_start[ui]
        sp = find(cs, cs + len(u.text))
        if sp:
            utt_mask[sp[0]:sp[1] + 1] = True
            tok_utt[sp[0]:sp[1] + 1] = ui
    offs = [tuple(o) for o in enc["offset_mapping"][:max_len]]
    txt = r.text
    user_lbl = next((k for k, v in w.speakers.items() if v == USER), None)
    spk_names = sorted({u.spk for u in w.utterances} - {user_lbl})
    spk_tok: list[list[int]] = [[] for _ in spk_names]
    utt_spk = []
    for ui, u in enumerate(w.utterances):
        if u.spk == user_lbl:
            utt_spk.append(0)
            continue
        k = spk_names.index(u.spk)
        utt_spk.append(k + 1)
        ce = r.utt_char_start[ui] - len(f"][asr={u.asr_conf:.2f}] ")  # the label sits just before this
        sp = find(ce - len(u.spk), ce)
        if sp and sp[1] < len(ids):
            spk_tok[k].append(sp[1])

    def is_word(c):
        return 0 <= c < len(txt) and (txt[c].isalnum() or txt[c] == "'")

    def first(a, b):  # first non-space char of a token (offsets may include the leading space)
        while a < b and txt[a].isspace():
            a += 1
        return a
    ws = torch.tensor([b > a and not is_word(first(a, b) - 1) for a, b in offs], dtype=torch.bool)
    we = torch.tensor([b > a and not is_word(b) for a, b in offs], dtype=torch.bool)
    ex = Example3(input_ids=torch.tensor(ids), slot_idx=torch.tensor([tok_at(c) for c in r.slot_char], dtype=torch.long),
                  user_idx=tok_at(r.user_char), utt_mask=utt_mask, word_start=ws, word_end=we, tok_utt=tok_utt,
                  partial=partial, spk_names=spk_names, spk_tok=spk_tok, utt_spk=torch.tensor(utt_spk, dtype=torch.long),
                  salience=None if partial else float(lab.get("salience") or 0.5),
                  meta={"window_id": w.window_id, "text": r.text, "offsets": [tuple(o) for o in enc["offset_mapping"][:max_len]],
                        "state_refs": r.state_refs, "utt_char_start": r.utt_char_start})
    slot_of = {ref: k + 1 for k, ref in enumerate(r.state_refs)}

    def span(u, s, e):
        if not (0 <= u < len(w.utterances)):
            return None
        return find(r.utt_char_start[u] + s, r.utt_char_start[u] + e)

    ent_ix, arg_ix = {}, {}
    for e in lab["entities"]:
        sp = span(e["utt"], e["start"], e["end"])
        if not sp:
            continue
        if partial:  # FrameNet role filler: an argument span, not an entity
            arg_ix[e["id"]] = len(ex.args)
            ex.args.append(sp)
            continue
        if sp[1] - sp[0] >= MAX_SPAN:
            continue
        t = labels.type_id.get(norm_type(e.get("type")) or "", labels.type_id.get((e.get("coarse") or "").lower(), -1))
        ent_ix[e["id"]] = len(ex.ents)
        ex.ents.append((sp[0], sp[1], t, COARSE2ID.get(e.get("coarse"), -1) - 1 if e.get("coarse") in COARSE2ID else -1,
                        slot_of.get(e.get("link") or "", 0)))
    trg_ix = {}
    for f in lab["frames"]:
        fid = labels.frame_id.get(f["frame"])
        sp = span(f["utt"], f["start"], f["end"])
        if fid is None or not sp or sp[1] - sp[0] >= MAX_SPAN:
            continue
        trg_ix[f["id"]] = (len(ex.trgs), fid)
        ex.trgs.append((sp[0], sp[1], fid))
        ex.attrs.append(None if partial else {
            **{k: ATTRIBUTES[k].index(f.get(k)) if f.get(k) in ATTRIBUTES[k] else 0 for k in ATTRIBUTES},
            "importance": ("low", "medium", "high").index(f.get("importance", "medium")),
            **{fl: int(bool(f.get(fl))) for fl in FLAGS}})
    for f in lab["frames"]:
        if f["id"] not in trg_ix:
            continue
        ti, fid = trg_ix[f["id"]]
        pos_of = {labels.roles[g][1]: j + 1 for j, g in enumerate(labels.frame_roles[fid])}
        for role, v in f["roles"].items():
            j = pos_of.get(role)
            if j is None:
                continue
            if v == "USER":
                ex.roles.append((ti, 1, 0, j))
            elif v in spk_names:
                ex.roles.append((ti, 4, spk_names.index(v), j))
            elif v in ent_ix:
                ex.roles.append((ti, 0, ent_ix[v], j))
            elif v in trg_ix and trg_ix[v][0] != ti:
                ex.roles.append((ti, 2, trg_ix[v][0], j))
            elif v in arg_ix:
                ex.roles.append((ti, 3, arg_ix[v], j))
    return ex


# ---------------------------------------------------------------------------------------------------- model
class LabelEmbed(nn.Module):
    """Vectors for an open label set: LayerNorm(W e(desc)) + residual, plus a per-label bias."""

    def __init__(self, desc: torch.Tensor, d: int):
        super().__init__()
        self.register_buffer("desc", desc.float())
        self.w = nn.Linear(desc.shape[1], d)
        self.ln = nn.LayerNorm(d)
        self.resid = nn.Parameter(torch.zeros(desc.shape[0], d))
        self.bias = nn.Parameter(torch.zeros(desc.shape[0]))

    def forward(self, idx: torch.Tensor | None = None):
        if idx is None:
            return self.ln(self.w(self.desc)) + self.resid, self.bias
        return self.ln(self.w(self.desc[idx])) + self.resid[idx], self.bias[idx]


class NARMemory3(nn.Module):
    def __init__(self, backbone_name: str, labels: LabelSet, desc: dict[str, torch.Tensor] | None = None,
                 d: int = 512, adapter_layers: int = 2, dtype=torch.float32, speakers: bool = False):
        super().__init__()
        self.labels = labels
        self.speakers = speakers  # other speakers are role fillers (v4); v3 runs have only USER
        self.frame_w = None       # optional per-frame loss weights (training only; set by train3 --frame-weight)
        self.trg_hard = 0         # hard trigger negatives per window (training only)
        self.backbone, hidden = load_backbone(backbone_name, dtype)
        self.hidden = hidden
        self.proj = nn.Linear(hidden, d)
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=4 * d, dropout=0.1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.adapter = nn.TransformerEncoder(layer, adapter_layers) if adapter_layers > 0 else nn.Identity()
        self.width = nn.Embedding(MAX_SPAN, 64)
        sd = 2 * d + 64
        self.d = d
        z = {k: torch.zeros(n, hidden) for k, n in (("frames", len(labels.frames)), ("roles", len(labels.roles)),
                                                     ("types", len(labels.types)))}
        desc = desc or z
        self.type_emb = LabelEmbed(desc["types"], d)
        self.frame_emb = LabelEmbed(desc["frames"], d)
        self.role_emb = LabelEmbed(desc["roles"], d)
        self.ent_q, self.ent_none = mlp(sd, d, d), mlp(sd, d // 2, 1)
        self.coarse_head = mlp(sd, d, len(ENTITY_LIST) + 1)
        self.trg_q, self.trg_none = mlp(sd, d, d), mlp(sd, d // 2, 1)
        self.arg_proj = nn.Linear(sd, d)
        self.kind_emb = nn.Embedding(5 if speakers else 4, d)   # entity, USER, trigger, extra argument, speaker
        self.dist_emb = nn.Embedding(N_DIST, d)
        self.same_utt = nn.Embedding(3 if speakers else 2, d)   # 2: the argument is the speaker of the trigger's line
        self.pair = mlp(3 * d, d, d)
        self.role_none = mlp(d, d // 2, 1)
        self.coref_q, self.coref_k = nn.Linear(d, d), nn.Linear(d, d)
        self.new_slot = nn.Parameter(torch.randn(d) * 0.02)
        self.attr_heads = nn.ModuleDict({k: mlp(sd + d, d // 2, len(v)) for k, v in ATTR_VALUES.items()})
        self.flag_head = mlp(sd + d, d // 2, len(FLAGS))
        self.sal_head = mlp(d, d // 2, 1)
        self.scale = nn.Parameter(torch.tensor(float(d) ** -0.5 * 4))
        self.register_buffer("pos_w", torch.tensor(4.0))        # positives are ~1-2% of candidate spans
        self.register_buffer("temps", torch.ones(4))            # entity, trigger, role, attrs
        self.register_buffer("platt", torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))  # ent a,b ; trg a,b ; role a,b

    # ------------------------------------------------------------------ description vectors
    @torch.no_grad()
    def encode_descriptions(self, tok, texts: list[str], bs: int = 64, max_len: int = 96) -> torch.Tensor:
        """Mean-pooled BASE backbone states of each text (call before fine-tuning)."""
        dev = next(self.parameters()).device
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
            o = self.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hh = (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]).float()
            m = enc["attention_mask"][..., None].float()
            out.append(((hh * m).sum(1) / m.sum(1)).cpu())
        return torch.cat(out)

    def set_descriptions(self, tok):
        lb = self.labels
        for emb, texts in ((self.type_emb, lb.type_text), (self.frame_emb, lb.frame_text), (self.role_emb, lb.role_text)):
            emb.desc.copy_(self.encode_descriptions(tok, texts).to(emb.desc.device))

    # ------------------------------------------------------------------ pieces
    def encode(self, input_ids, attention_mask):
        o = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self.proj((o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]).float())
        if isinstance(self.adapter, nn.TransformerEncoder):
            h = self.adapter(h, src_key_padding_mask=~attention_mask.bool())
        return h

    def span_reps(self, h, spans):
        st = torch.tensor([s for s, _ in spans], device=h.device)
        en = torch.tensor([e for _, e in spans], device=h.device)
        return torch.cat([h[st], h[en], self.width((en - st).clamp(max=MAX_SPAN - 1))], -1)

    def ent_logits(self, s):
        T, b = self.type_emb()
        return torch.cat([self.ent_none(s), self.scale * self.ent_q(s) @ T.T + b], -1)

    def trg_logits(self, s):
        Fm, b = self.frame_emb()
        return torch.cat([self.trg_none(s), self.scale * self.trg_q(s) @ Fm.T + b], -1)

    def role_logits(self, pair_rep, frame_ids: torch.Tensor):
        """pair_rep: (T, A, d); frame_ids: (T,) -> (T, A, 1 + max_roles) logits, padded roles = -inf."""
        dev = pair_rep.device
        R = self.labels.max_roles
        idx = torch.zeros(len(frame_ids), R, dtype=torch.long, device=dev)
        mask = torch.zeros(len(frame_ids), R, dtype=torch.bool, device=dev)
        for i, f in enumerate(frame_ids.tolist()):
            ids = self.labels.frame_roles[f]
            idx[i, :len(ids)] = torch.tensor(ids, device=dev)
            mask[i, :len(ids)] = True
        E, b = self.role_emb(idx.flatten())
        E, b = E.view(len(frame_ids), R, -1), b.view(len(frame_ids), R)
        sc = self.scale * torch.einsum("tad,trd->tar", pair_rep, E) + b[:, None, :]
        sc = sc.masked_fill(~mask[:, None, :], float("-inf"))
        return torch.cat([self.role_none(pair_rep), sc], -1)

    def speaker_reps(self, h, ex: Example3):
        """One argument per other speaker: the mean state over the tokens of their label in each line they speak."""
        if not self.speakers or not ex.spk_names:
            return None
        dev = h.device
        rows = []
        for toks in ex.spk_tok:
            m = h[torch.tensor(toks, device=dev)].mean(0) if toks else h.new_zeros(h.shape[-1])
            rows.append(torch.cat([m, m, self.width.weight[0]], -1))
        return self.arg_proj(torch.stack(rows)) + self.kind_emb.weight[4]

    def pair_reps(self, trg, args, t_pos=None, a_pos=None, tok_utt=None, a_spk=None, utt_spk=None):
        """t_pos / a_pos: token index of each trigger / argument (-1 = USER or a speaker); tok_utt: utterance index per
        token. With speakers: a_spk = speaker id per argument (0 USER, k+1 speaker k, -1 not a speaker) and utt_spk =
        speaker id per utterance; same_utt is 2 where the argument is the speaker of the trigger's line."""
        a = trg[:, None, :].expand(-1, args.shape[0], -1)
        b = args[None, :, :].expand(trg.shape[0], -1, -1)
        x = torch.cat([a, b, a * b], -1)
        out = self.pair(x)
        if t_pos is not None:
            d = a_pos[None, :] - t_pos[:, None]
            bk = dist_bucket(d)
            bk = torch.where((a_pos < 0)[None, :].expand_as(bk), torch.full_like(bk, N_DIST - 1), bk)
            tu = tok_utt[t_pos.clamp(min=0)]
            au = torch.where(a_pos >= 0, tok_utt[a_pos.clamp(min=0)], torch.full_like(a_pos, -2))
            same = (tu[:, None] == au[None, :]).long()
            if a_spk is not None and utt_spk is not None and len(utt_spk):
                ts = utt_spk[tu.clamp(min=0)]
                ts = torch.where(tu >= 0, ts, torch.full_like(ts, -9))
                same = torch.where((a_spk[None, :] >= 0) & (a_spk[None, :] == ts[:, None]), torch.full_like(same, 2), same)
            out = out + self.dist_emb(bk) + self.same_utt(same)
        return out

    # ------------------------------------------------------------------ training
    def loss_one(self, h, ex: Example3, n_neg: int = 768, n_hard: int = 384):
        dev = h.device
        L: dict[str, torch.Tensor] = {}
        cands = None
        if not ex.partial:
            cands = candidate_spans(ex.utt_mask, ex.word_start, ex.word_end)
        # -- entities
        if not ex.partial:
            gold = {(s, e): (t, c) for s, e, t, c, _ in ex.ents}
            hard, rand = [], []
            for c in cands:
                if c in gold:
                    continue
                (hard if any(c[0] <= ge and c[1] >= gs for gs, ge in gold) else rand).append(c)
            hard = [hard[i] for i in torch.randperm(len(hard))[:n_hard].tolist()]
            rand = [rand[i] for i in torch.randperm(len(rand))[:n_neg].tolist()]
            spans = list(gold) + hard + rand
            if spans:
                s = self.span_reps(h, spans)
                el = self.ent_logits(s)
                tgt = torch.tensor([gold[x][0] + 1 if x in gold else 0 for x in spans], device=dev)
                # entity-or-not over all spans; the type only where the gold type is known (-1 -> 0 means unknown)
                is_ent = torch.tensor([x in gold for x in spans], device=dev)
                L["ent_none"] = F.binary_cross_entropy_with_logits(
                    (torch.logsumexp(el[:, 1:], -1) - el[:, 0]), is_ent.float(), pos_weight=self.pos_w)
                typed = is_ent & torch.tensor([x in gold and gold[x][0] >= 0 for x in spans], device=dev)
                if typed.any():
                    L["ent_type"] = F.cross_entropy(el[typed][:, 1:], tgt[typed] - 1)
                cl = self.coarse_head(s)
                ct = torch.tensor([gold[x][1] + 1 if x in gold and gold[x][1] >= 0 else (0 if x not in gold else -100)
                                   for x in spans], device=dev)
                L["coarse"] = F.cross_entropy(cl, ct, ignore_index=-100)
                L["ent_brier"] = ((torch.sigmoid(torch.logsumexp(el[:, 1:], -1) - el[:, 0]) - is_ent.float()) ** 2).mean()
        # -- triggers
        if ex.trgs:
            ts = [(s, e) for s, e, _ in ex.trgs]
            ft = torch.tensor([f for _, _, f in ex.trgs], device=dev)
            if ex.partial:
                tl = self.trg_logits(self.span_reps(h, ts))
                L["frame_fn"] = F.cross_entropy(tl[:, 1:], ft)
            else:
                gold_t = {x: i for i, x in enumerate(ts)}
                hard, rand = [], []
                for c in cands:
                    if c not in gold_t:
                        (hard if any(c[0] <= ge and c[1] >= gs for gs, ge in gold_t) else rand).append(c)
                n_h = self.trg_hard  # spans overlapping a trigger ("coming thursday" around "coming")
                neg = ([hard[i] for i in torch.randperm(len(hard))[:n_h].tolist()] +
                       [rand[i] for i in torch.randperm(len(rand))[:n_neg].tolist()])
                sp = ts + neg
                tl = self.trg_logits(self.span_reps(h, sp))
                is_t = torch.zeros(len(sp), device=dev)
                is_t[:len(ts)] = 1
                fw = None
                if self.frame_w is not None:  # rare frames (most memory frames) weigh more
                    fw = torch.ones(len(sp), device=dev)
                    fw[:len(ts)] = self.frame_w.to(dev)[ft]
                L["trg_none"] = F.binary_cross_entropy_with_logits(torch.logsumexp(tl[:, 1:], -1) - tl[:, 0], is_t,
                                                                   weight=fw, pos_weight=self.pos_w)
                if fw is None:
                    L["frame"] = F.cross_entropy(tl[:len(ts), 1:], ft)
                else:
                    L["frame"] = (F.cross_entropy(tl[:len(ts), 1:], ft, reduction="none") * fw[:len(ts)]).sum() / fw[:len(ts)].sum()
                L["trg_brier"] = ((torch.sigmoid(torch.logsumexp(tl[:, 1:], -1) - tl[:, 0]) - is_t) ** 2).mean()
        elif not ex.partial and cands:
            neg = [cands[i] for i in torch.randperm(len(cands))[:n_neg].tolist()]
            tl = self.trg_logits(self.span_reps(h, neg))
            L["trg_none"] = F.binary_cross_entropy_with_logits(torch.logsumexp(tl[:, 1:], -1) - tl[:, 0],
                                                               torch.zeros(len(neg), device=dev))
        # -- roles, links, attrs
        if ex.trgs:
            trg_s = self.span_reps(h, [(s, e) for s, e, _ in ex.trgs])
            parts, kinds = [], []
            ent_s = self.span_reps(h, [(s, e) for s, e, *_ in ex.ents]) if ex.ents else None
            if ent_s is not None:
                parts.append(self.arg_proj(ent_s) + self.kind_emb.weight[0])
            parts.append(self.arg_proj(self.span_reps(h, [(ex.user_idx, ex.user_idx)])) + self.kind_emb.weight[1])
            parts.append(self.arg_proj(trg_s) + self.kind_emb.weight[2])
            if ex.args:
                parts.append(self.arg_proj(self.span_reps(h, ex.args)) + self.kind_emb.weight[3])
            spk = self.speaker_reps(h, ex)
            n_spk = 0 if spk is None else len(spk)
            if spk is not None:
                parts.append(spk)
            offs = {}
            o = 0
            for k, n in ((0, len(ex.ents)), (1, 1), (2, len(ex.trgs)), (3, len(ex.args)), (4, n_spk)):
                offs[k] = o
                o += n
            A = torch.cat(parts, 0)
            tq = self.arg_proj(trg_s) + self.kind_emb.weight[2]
            pos = ([s for s, *_ in ex.ents] + [-1] + [s for s, _, _ in ex.trgs] + [s for s, _ in ex.args] + [-1] * n_spk)
            a_spk = ([-1] * len(ex.ents) + [0] + [-1] * (len(ex.trgs) + len(ex.args)) + list(range(1, n_spk + 1)))
            sp_kw = {"a_spk": torch.tensor(a_spk, device=dev), "utt_spk": ex.utt_spk.to(dev)} if self.speakers else {}
            pr = self.pair_reps(tq, A, torch.tensor([s for s, _, _ in ex.trgs], device=dev),
                                torch.tensor(pos, device=dev), ex.tok_utt.to(dev), **sp_kw)
            rl = self.role_logits(pr, torch.tensor([f for _, _, f in ex.trgs], device=dev))
            tgt = torch.zeros(rl.shape[:2], dtype=torch.long, device=dev)
            for ti, kind, ai, j in ex.roles:
                if kind == 4 and not n_spk:
                    continue
                tgt[ti, offs[kind] + ai] = j
            for ti in range(len(ex.trgs)):  # a frame is not its own argument
                tgt[ti, offs[2] + ti] = -100
            if ex.partial:  # unlabeled pairs are unknown, not NONE: train only gold pairs + the USER/other-frame rows
                keep = tgt > 0
                if keep.any():
                    L["role_fn"] = F.cross_entropy(rl[keep], tgt[keep])
            else:
                L["role"] = F.cross_entropy(rl.flatten(0, 1), tgt.flatten(), ignore_index=-100,
                                            weight=torch.cat([torch.tensor([0.5], device=dev),
                                                              torch.ones(rl.shape[-1] - 1, device=dev)]))
                if ex.attrs and ex.attrs[0] is not None:
                    pooled = h[ex.utt_mask.to(dev)].mean(0)
                    z = torch.cat([trg_s, pooled[None].expand(len(trg_s), -1)], -1)
                    al = 0
                    for k, head in self.attr_heads.items():
                        al = al + F.cross_entropy(head(z), torch.tensor([a[k] for a in ex.attrs], device=dev))
                    ftg = torch.tensor([[a[f] for f in FLAGS] for a in ex.attrs], device=dev, dtype=torch.float)
                    al = al + F.binary_cross_entropy_with_logits(self.flag_head(z), ftg)
                    L["attrs"] = al / (len(self.attr_heads) + 1)
        if not ex.partial and ex.ents:
            m = self.arg_proj(self.span_reps(h, [(s, e) for s, e, *_ in ex.ents]))
            slots = torch.cat([self.new_slot[None], h[ex.slot_idx.to(dev)]], 0) if len(ex.slot_idx) else self.new_slot[None]
            cl = self.coref_q(m) @ self.coref_k(slots).T / self.d ** 0.5
            L["link"] = F.cross_entropy(cl, torch.tensor([x[4] for x in ex.ents], device=dev))
        if ex.salience is not None:
            sal = torch.sigmoid(self.sal_head(h[ex.utt_mask.to(dev)].mean(0))).squeeze()
            L["sal"] = (sal - ex.salience) ** 2
        return L

    def forward(self, batch, weights: dict | None = None):
        h = self.encode(batch["input_ids"], batch["attention_mask"])
        parts: dict[str, torch.Tensor] = {}
        n: dict[str, int] = {}
        for i, ex in enumerate(batch["examples"]):
            for k, v in self.loss_one(h[i, : len(ex.input_ids)], ex).items():
                parts[k] = parts.get(k, 0) + v
                n[k] = n.get(k, 0) + 1
        parts = {k: v / n[k] for k, v in parts.items()}
        w = weights or {}
        total = sum(v * w.get(k, 1.0) for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}


def collate3(examples, pad_id: int):
    Lm = max(len(e.input_ids) for e in examples)
    ids = torch.full((len(examples), Lm), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), Lm), dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, : len(e.input_ids)] = e.input_ids
        mask[i, : len(e.input_ids)] = 1
    return {"input_ids": ids, "attention_mask": mask, "examples": examples}


__all__ = ["LabelSet", "NARMemory3", "Example3", "featurize3", "collate3", "build_types", "norm_type", "brier"]
