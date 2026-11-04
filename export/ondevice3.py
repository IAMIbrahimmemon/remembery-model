"""System 1 v3 on device: one multi-method ExecuTorch program + the metadata the phone needs.

All neural compute runs inside the .pte; the host (Swift on iOS, Python here for parity) does control flow: render +
tokenize, candidate spans, greedy picking, role selection and compile_memories. Label vectors (types, frames, roles)
depend only on weights, so they are precomputed constants and the description tables are not shipped.

Methods (temperatures and Platt scaling are baked in; C, NE, NT, P are dynamic, one per tensor for XNNPACK):
  encode(input_ids[1, L])                         -> h[L, d]                              L 16..1024
  score(h, starts[C], ends[C])                    -> p_ent, type, type_p, coarse, p_trg, frame, frame_p   (each [C])
  link(h, starts[NE], ends[NE], slot_idx[S], slot_valid[S])   -> slot[NE]                 S = 32 (padded)
  roles(h, t_st[P], t_en[P], a_st[P], a_en[P], a_kind[P], frame[P], tok_utt[L])
                                                  -> role[P], role_p[P]                   one row per (trigger, argument)
  v4 (speakers): roles(..., tok_utt[L], a_spk[P], t_spk[P], spk_idx[Q], spk_seg[Q])
      a_kind 4 = another speaker (a_st = a_en = k); its state is the mean of h over that speaker's label tokens, given
      as a flat list spk_idx (token) / spk_seg (speaker k; NS_MAX = padding) so each tensor keeps one dynamic dim.
      a_spk: 0 USER, k+1 speaker k, -1 otherwise; t_spk: speaker of the trigger's line (-9 outside a line).
  attrs(h, utt_mask[L], t_st[NT], t_en[NT])       -> attrs[NT, 5], flags[NT, 2], sal[1]

Weights stay fp32. On the 89 dev checklist windows, int4 8da4w (D2's recipe) changed the System 1 memories of 86 windows
and fp16 of 44; fp32 changed 3 (pruned vocab). On the iPhone 17, fp32 is also faster than fp16 (docs/ios.md).

    PYTHONPATH=. uv run python -m export.ondevice3 --run runs/v3b --vocab runs/v3b/vocab3.json --out ios/Model
    PYTHONPATH=. uv run python -m export.ondevice3 --run runs/v3b --check-only     # host logic == extract3 (eager)
    PYTHONPATH=. uv run python -m export.ondevice3 --run runs/v4e --ckpt best.pt --vocab runs/v3b/vocab3.json --out ios/Model
    PYTHONPATH=. uv run python -m export.ondevice3 --run runs/v4f --ckpt last.pt --vocab runs/v3b/vocab3.json --out ios/Model_b
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from eval.predict3 import load3
from model.decode3 import compile_memories, extract3
from model.featurize import MAX_SPAN, MAX_STATE
from model.nar3 import ATTR_VALUES, N_DIST, NARMemory3, candidate_spans, dist_bucket, featurize3
from ontology.ontology import ENTITY_LIST, FLAGS

N_SLOTS = MAX_STATE
CHUNK = 1024          # candidate spans per score call
MAX_ITEMS = 64        # extract3's max_items: at most 64 entities and 64 triggers
PAD_MIN = 2           # dynamic dims start at 2 (export specializes 0 and 1); the host pads with dummy spans
NS_MAX = 32           # other speakers per window the roles program can take (v4); row NS_MAX absorbs padding
ATTR_KEYS = list(ATTR_VALUES)


def _platt(p, a, b):
    return torch.sigmoid(a * torch.logit(p.clamp(1e-6, 1 - 1e-6)) + b)


def _span_reps(m: NARMemory3, h, st, en):
    return torch.cat([h[st], h[en], m.width((en - st).clamp(min=0, max=MAX_SPAN - 1))], -1)


class Encode(nn.Module):
    def __init__(self, m: NARMemory3):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        return self.m.encode(input_ids, torch.ones_like(input_ids))[0]


class Score(nn.Module):
    """Entity and trigger heads over a chunk of candidate spans (type and frame vectors are constants)."""

    def __init__(self, m: NARMemory3):
        super().__init__()
        self.m = m
        with torch.no_grad():
            T, tb = m.type_emb()
            Fm, fb = m.frame_emb()
        self.register_buffer("T", T.detach().clone())
        self.register_buffer("tb", tb.detach().clone())
        self.register_buffer("Fm", Fm.detach().clone())
        self.register_buffer("fb", fb.detach().clone())

    def forward(self, h, starts, ends):
        m = self.m
        S = _span_reps(m, h, starts, ends)
        el = torch.cat([m.ent_none(S), m.scale * m.ent_q(S) @ self.T.T + self.tb], -1) / m.temps[0]
        p_ent = _platt(torch.sigmoid(torch.logsumexp(el[:, 1:], -1) - el[:, 0]), m.platt[0], m.platt[1])
        type_p, typ = el[:, 1:].softmax(-1).max(-1)
        coarse = m.coarse_head(S)[:, 1:].argmax(-1)
        tl = torch.cat([m.trg_none(S), m.scale * m.trg_q(S) @ self.Fm.T + self.fb], -1) / m.temps[1]
        p_trg = _platt(torch.sigmoid(torch.logsumexp(tl[:, 1:], -1) - tl[:, 0]), m.platt[2], m.platt[3])
        frame_p, frame = tl[:, 1:].softmax(-1).max(-1)
        return p_ent, typ, type_p, coarse, p_trg, frame, frame_p


class Link(nn.Module):
    def __init__(self, m: NARMemory3):
        super().__init__()
        self.m = m

    def forward(self, h, starts, ends, slot_idx, slot_valid):
        m = self.m
        q = m.arg_proj(_span_reps(m, h, starts, ends))
        slots = torch.cat([m.new_slot[None], h[slot_idx]], 0)
        logits = m.coref_q(q) @ m.coref_k(slots).T / m.d ** 0.5
        valid = torch.cat([torch.ones(1), slot_valid])
        return logits.masked_fill(valid[None] < 0.5, -1e9).argmax(-1)


class Roles(nn.Module):
    """Best role (position within the trigger's frame) and its calibrated p for a flat list of (trigger, argument)
    pairs; one dynamic dimension (P), as XNNPACK requires. Argument kinds: 0 entity, 1 USER (span = the [USER] token),
    2 another trigger, 4 another speaker (v4). The frame -> role table is a constant gathered in the program."""

    def __init__(self, m: NARMemory3):
        super().__init__()
        self.m = m
        lb = m.labels
        R = lb.max_roles
        idx = torch.zeros(len(lb.frames), R, dtype=torch.long)
        mask = torch.zeros(len(lb.frames), R, dtype=torch.bool)
        for f, ids in enumerate(lb.frame_roles):
            idx[f, :len(ids)] = torch.tensor(ids)
            mask[f, :len(ids)] = True
        with torch.no_grad():
            E, b = m.role_emb()
        self.register_buffer("role_idx", idx)
        self.register_buffer("role_mask", mask)
        self.register_buffer("E", E.detach().clone())
        self.register_buffer("b", b.detach().clone())

    def forward(self, h, t_st, t_en, a_st, a_en, a_kind, frame, tok_utt, a_spk=None, t_spk=None, spk_idx=None, spk_seg=None):
        m = self.m
        tq = m.arg_proj(_span_reps(m, h, t_st, t_en)) + m.kind_emb.weight[2]
        user = a_kind == 1
        if m.speakers:
            # nar3.speaker_reps: arg_proj(cat[mean, mean, width[0]]); a kind-4 row is a width-0 span whose ends are the mean
            oh = (spk_seg[:, None] == torch.arange(NS_MAX + 1)[None, :]).to(h.dtype)          # Q x (NS_MAX + 1)
            mean = (oh.T @ h[spk_idx]) / oh.sum(0).clamp(min=1)[:, None]                    # (NS_MAX + 1) x d
            spk = a_kind == 4
            ms = mean[a_st.clamp(max=NS_MAX)]
            sv = torch.where(spk[:, None], ms, h[a_st])
            ev = torch.where(spk[:, None], ms, h[a_en])
            a = m.arg_proj(torch.cat([sv, ev, m.width((a_en - a_st).clamp(min=0, max=MAX_SPAN - 1))], -1)) + m.kind_emb(a_kind)
            user = user | spk
        else:
            a = m.arg_proj(_span_reps(m, h, a_st, a_en)) + m.kind_emb(a_kind)
        pr = m.pair(torch.cat([tq, a, tq * a], -1))                                   # P x d (nar3.pair_reps)
        bk = torch.where(user, torch.full_like(a_st, N_DIST - 1), dist_bucket(a_st - t_st))
        au = torch.where(user, torch.full_like(a_st, -2), tok_utt[a_st])
        same = (tok_utt[t_st] == au).long()
        if m.speakers:
            same = torch.where((a_spk >= 0) & (a_spk == t_spk), torch.full_like(same, 2), same)
        pr = pr + m.dist_emb(bk) + m.same_utt(same)
        idx, mask = self.role_idx[frame], self.role_mask[frame]                         # P x R
        sc = m.scale * torch.einsum("pd,prd->pr", pr, self.E[idx]) + self.b[idx]
        sc = sc.masked_fill(~mask, float("-inf"))
        rp = (torch.cat([m.role_none(pr), sc], -1) / m.temps[2]).softmax(-1)
        p, j = rp[:, 1:].max(-1)
        return j, _platt(p, m.platt[4], m.platt[5])


class Attrs(nn.Module):
    def __init__(self, m: NARMemory3):
        super().__init__()
        self.m = m

    def forward(self, h, utt_mask, t_st, t_en):
        m = self.m
        pooled = (h * utt_mask[:, None]).sum(0) / utt_mask.sum().clamp(min=1.0)
        z = torch.cat([_span_reps(m, h, t_st, t_en), pooled[None].expand(t_st.shape[0], -1)], -1)
        attrs = torch.stack([(m.attr_heads[k](z) / m.temps[3]).argmax(-1) for k in ATTR_KEYS], -1)
        return attrs, torch.sigmoid(m.flag_head(z)), torch.sigmoid(m.sal_head(pooled))


def frame_core(frames: list[str]) -> list[list[str]]:
    """Core roles per frame, as decode3._core (which adds Condition and New_value itself)."""
    from ontology.grammar import load
    g = load()["frames"]
    return [sorted(g[f].get("core", g[f]["roles"])) if f in g else [] for f in frames]


def modules(m: NARMemory3) -> dict:
    return {"encode": Encode(m).eval(), "score": Score(m).eval(), "link": Link(m).eval(), "roles": Roles(m).eval(),
            "attrs": Attrs(m).eval()}


# ---------------------------------------------------------------------- host logic (mirrored in Swift)
def _pad(xs: list[int]) -> list[int]:
    return xs + [xs[-1] if xs else 0] * max(0, PAD_MIN - len(xs))


def host_extract3(run, ex, labels, ent_thresh: float = 0.5, trg_thresh: float = 0.5, role_thresh: float = 0.35,
                  max_items: int = MAX_ITEMS, speakers: bool = False) -> dict:
    """extract3() built only from the five methods. run(name, *tensors) -> tuple of tensors. Greedy picking uses a
    stable sort (ties keep candidate order) so that Swift can reproduce it; extract3's argsort is not stable."""
    L = lambda xs: torch.tensor(xs, dtype=torch.long)  # noqa: E731
    h = run("encode", ex.input_ids[None])[0]
    cands = candidate_spans(ex.utt_mask, ex.word_start, ex.word_end)
    if not cands:
        return {"entities": [], "frames": [], "salience": 0.0}
    outs = []
    for c in range(0, len(cands), CHUNK):
        ch = cands[c:c + CHUNK]
        st, en = _pad([s for s, _ in ch]), _pad([e for _, e in ch])
        outs.append([x[:len(ch)] for x in run("score", h, L(st), L(en))])
    p_ent, typ, type_p, coarse, p_trg, frame, frame_p = (torch.cat([o[k] for o in outs]) for k in range(7))

    def pick(p, thr):
        used = [False] * len(ex.input_ids)
        out = []
        for i in torch.argsort(p, descending=True, stable=True).tolist():
            if p[i] < thr or len(out) >= max_items:
                break
            s, e = cands[i]
            if any(used[s:e + 1]):
                continue
            for k in range(s, e + 1):
                used[k] = True
            out.append(i)
        return sorted(out, key=lambda i: cands[i])

    meta = ex.meta
    text, offs = meta["text"], meta["offsets"]

    def span_text(s, e):
        return text[offs[s][0]:offs[e][1]].strip()

    def where(s):
        c = offs[s][1] - 1
        return max((i for i, st in enumerate(meta["utt_char_start"]) if st <= c), default=0)

    lb = labels
    ents = []
    ei = pick(p_ent, ent_thresh)
    if ei:
        si = torch.zeros(N_SLOTS, dtype=torch.long)
        sv = torch.zeros(N_SLOTS)
        for k, idx in enumerate(ex.slot_idx.tolist()[:N_SLOTS]):
            si[k], sv[k] = idx, 1.0
        link = run("link", h, L(_pad([cands[i][0] for i in ei])), L(_pad([cands[i][1] for i in ei])), si, sv)[0].tolist()
        for k, i in enumerate(ei):
            s, e = cands[i]
            ents.append({"id": f"e{k + 1}", "text": span_text(s, e), "utt": where(s), "type": lb.types[int(typ[i])],
                         "type_p": round(float(type_p[i]), 3), "coarse": ENTITY_LIST[int(coarse[i])],
                         "p": round(float(p_ent[i]), 3),
                         "link": meta["state_refs"][link[k] - 1] if link[k] > 0 else None})
    frames = []
    ti = pick(p_trg, trg_thresh)
    t_st, t_en = _pad([cands[i][0] for i in ti]), _pad([cands[i][1] for i in ti])
    at, fl, sal = run("attrs", h, ex.utt_mask.float(), L(t_st), L(t_en))
    if ti:
        # arguments: entities, USER, triggers, then (v4) other speakers by label; one pair per (trigger, argument other
        # than itself), in extract3's order. (name, start, end, kind, speaker id)
        args = ([(nm, cands[i][0], cands[i][1], 0, -1) for nm, i in zip([e["id"] for e in ents], ei)]
                + [("USER", ex.user_idx, ex.user_idx, 1, 0)]
                + [(f"f{k + 1}", cands[i][0], cands[i][1], 2, -1) for k, i in enumerate(ti)])
        if speakers:
            args += [(nm, k, k, 4, k + 1) for k, nm in enumerate(ex.spk_names[:NS_MAX])]
        pairs = [(k, q) for k in range(len(ti)) for q, a in enumerate(args) if a[0] != f"f{k + 1}"]
        tp = [cands[ti[k]] for k, _ in pairs]
        ap = [args[q] for _, q in pairs]
        xs = [h, L(_pad([t[0] for t in tp])), L(_pad([t[1] for t in tp])), L(_pad([a[1] for a in ap])),
              L(_pad([a[2] for a in ap])), L(_pad([a[3] for a in ap])), L(_pad([int(frame[ti[k]]) for k, _ in pairs])),
              ex.tok_utt]
        if speakers:
            tu = [int(ex.tok_utt[t[0]]) for t in tp]
            t_spk = [int(ex.utt_spk[u]) if u >= 0 else -9 for u in tu]
            sidx = [t for k, toks in enumerate(ex.spk_tok[:NS_MAX]) for t in toks]
            sseg = [k for k, toks in enumerate(ex.spk_tok[:NS_MAX]) for _ in toks]
            sidx, sseg = sidx + [0] * max(0, PAD_MIN - len(sidx)), sseg + [NS_MAX] * max(0, PAD_MIN - len(sseg))
            xs += [L(_pad([a[4] for a in ap])), L(_pad(t_spk)), L(sidx), L(sseg)]
        j, rp = run("roles", *xs)
        for k, i in enumerate(ti):
            s, e = cands[i]
            fid = int(frame[i])
            role_names = [lb.roles[g][1] for g in lb.frame_roles[fid]]
            best: dict[str, list] = {}
            for n, (kk, q) in enumerate(pairs):
                if kk != k:
                    continue
                jj, pr = int(j[n]), float(rp[n])
                if jj < len(role_names) and pr >= role_thresh:
                    best.setdefault(role_names[jj], []).append((pr, args[q][0]))
            roles = {}
            for r, lst in best.items():
                lst.sort(reverse=True)
                roles[r] = [n for _, n in lst[: 2 if r.startswith("Co") else 1]]
            frames.append({"id": f"f{k + 1}", "frame": lb.frames[fid], "frame_p": round(float(frame_p[i]), 3),
                           "trigger": span_text(s, e), "utt": where(s), "p": round(float(p_trg[i]), 3),
                           "roles": roles, **{a: ATTR_VALUES[a][int(at[k, q])] for q, a in enumerate(ATTR_KEYS)},
                           **{f: float(fl[k, q]) > 0.5 for q, f in enumerate(FLAGS)}})
    return {"entities": ents, "frames": frames, "salience": round(float(sal[0]), 3)}


def eager_runner(mods: dict):
    @torch.no_grad()
    def run(name, *xs):
        out = mods[name](*xs)
        return out if isinstance(out, tuple) else (out,)
    return run


def pte_runner(buf: bytes, remap: torch.Tensor | None):
    from executorch.runtime import Runtime
    prog = Runtime.get().load_program(buf)
    methods = {}

    def run(name, *xs):
        if name == "encode" and remap is not None:
            xs = (remap[xs[0]].long(),)
        if name not in methods:
            methods[name] = prog.load_method(name)
        return tuple(methods[name].execute(list(xs)))
    return run


def graph_key(g: dict):
    """Everything parity compares: entity text/type/coarse/link, frames with trigger, roles and attributes."""
    ents = [(e["text"], e["utt"], e["type"], e["coarse"], e["link"]) for e in g["entities"]]
    frames = [(f["frame"], f["trigger"], f["utt"], json.dumps(f["roles"]), *(f[a] for a in ATTR_KEYS), *(f[x] for x in FLAGS))
              for f in g["frames"]]
    return ents, frames


def speakers_of(w: dict):
    user = next((k for k, v in w["speakers"].items() if v == "USER"), "S1")
    return ["ME" if u["spk"] == user else u["spk"] for u in w["utterances"]], [u["text"] for u in w["utterances"]]


def memories(g: dict, w: dict, min_p: float) -> list[str]:
    spk, txt = speakers_of(w)
    return compile_memories(g, min_p=min_p, speakers=spk, utt_text=txt)


def check_windows(checklist: str, n: int | None = None) -> list[dict]:
    """The 89 dev windows that have checklists (data/eval/v3/dev.jsonl carries each window; the probes are among them)."""
    out = [json.loads(line)["window"] for line in open(checklist)]
    return out[:n] if n else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint file in the run (v4 ships v4e best.pt and v4f last.pt)")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--quant", choices=["none", "fp16", "8da4w"], default="none",
                    help="backbone weights; fp32 (none) is shipped: int4 and fp16 change memories on dev (docs/ios.md)")
    ap.add_argument("--out", default="ios/Model")
    ap.add_argument("--checklist", default="data/eval/v3/dev.jsonl")
    ap.add_argument("--probe", default="data/probe/probe_v1.jsonl")
    ap.add_argument("--n-check", type=int, default=None, help="dev windows for the int4 check (default: all 89)")
    ap.add_argument("--n-parity", type=int, default=20, help="dev windows written to parity3.json (plus the probes)")
    ap.add_argument("--s2-lo", type=float, default=0.4)
    ap.add_argument("--s2-hi", type=float, default=0.6)
    ap.add_argument("--check-only", action="store_true", help="only verify host logic (eager) == extract3")
    a = ap.parse_args()

    run = Path(a.run)
    m, tok, labels, th = load3(a.run, ckpt=a.ckpt)
    spk = bool(m.speakers)
    mods = modules(m)
    probes = [json.loads(line) for line in open(a.probe)]
    dev = [w for w in check_windows(a.checklist, a.n_check) if w["window_id"] not in {p["window_id"] for p in probes}]
    feats = {w["window_id"]: featurize3({"window": w, "labels": None}, tok, labels) for w in probes + dev}
    if spk:
        # v4 (decisions 32, 37): the app and the line validator use entity 0.5, trigger 0.5, role 0.35;
        # thresholds3.json is not used. Parity windows: the probes, then meeting windows with 3+ speakers first.
        th = {"ent_thresh": 0.5, "trg_thresh": 0.5, "role_thresh": 0.35}
        kw_app = th
        n_spk = lambda w: len({u["spk"] for u in w["utterances"]})  # noqa: E731
        multi = [w for w in dev if n_spk(w) >= 3]
        dev = multi[:a.n_parity // 2] + [w for w in dev if w not in multi[:a.n_parity // 2]]
    else:
        # the phone extracts with the trigger threshold at System 2's lower bound and stores p >= hi directly
        kw_app = {**th, "trg_thresh": a.s2_lo}

    # 1) host logic over eager fp32 methods must reproduce extract3 (up to argsort ties, counted)
    same_g = same_m = 0
    for w in probes + dev:
        ex = feats[w["window_id"]]
        ref = extract3(m, ex, **th)
        got = host_extract3(eager_runner(mods), ex, labels, **th, speakers=spk)
        sg = graph_key(ref) == graph_key(got)
        sm = memories(ref, w, th["trg_thresh"]) == memories(got, w, th["trg_thresh"])
        same_g += sg
        same_m += sm
        if not sg:
            print("eager host differs from extract3:", w["window_id"], "memories same" if sm else "memories differ")
    print(f"eager host == extract3: graphs {same_g}/{len(feats)}, memories {same_m}/{len(feats)}")
    if a.check_only:
        return

    # 2) export
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    from export.rope_patch import apply as real_rope
    real_rope()
    m_fp = m
    m, *_ = load3(a.run, ckpt=a.ckpt)
    m.backbone.config._attn_implementation = "sdpa"
    remap = None
    vocab_info = {}
    if a.vocab:
        vj = json.loads(Path(a.vocab).read_text())
        keep = vj["keep"]
        vocab_info = {k: vj[k] for k in ("vocab_full", "vocab_kept", "real_dev_token_coverage") if k in vj}
        emb = m.backbone.get_input_embeddings()
        small = nn.Embedding(len(keep), emb.weight.shape[1])
        small.weight.data.copy_(emb.weight.data[torch.tensor(keep)])
        m.backbone.set_input_embeddings(small)
        remap = torch.zeros(len(tok), dtype=torch.int32)  # dropped tokens -> row 0 (as for D2)
        remap[torch.tensor(keep)] = torch.arange(len(keep), dtype=torch.int32)
    mods_q = modules(m)
    for emb in (m.type_emb, m.frame_emb, m.role_emb):  # label vectors are now constants; drop the 76 MB tables
        emb.desc = torch.zeros(0)
    if a.quant == "fp16":
        m.backbone.half()
    elif a.quant == "8da4w":
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig, quantize_
        from torchao.quantization.granularity import PerGroup
        quantize_(m.backbone, Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4, weight_granularity=PerGroup(32)))

    d = m.d
    Ld = torch.export.Dim("L", min=16, max=1024)
    Cd = torch.export.Dim("C", min=PAD_MIN, max=CHUNK)
    Ed = torch.export.Dim("NE", min=PAD_MIN, max=MAX_ITEMS)
    Td = torch.export.Dim("NT", min=PAD_MIN, max=MAX_ITEMS)
    Pd = torch.export.Dim("P", min=PAD_MIN, max=MAX_ITEMS * (2 * MAX_ITEMS + 1))
    ex0 = feats[probes[0]["window_id"]]
    ids0 = ex0.input_ids[None] if remap is None else remap[ex0.input_ids].long()[None]
    i3 = torch.tensor([3, 5, 7])
    i4 = torch.tensor([4, 6, 8, 9])
    with torch.no_grad():
        h0 = mods_q["encode"](ids0)
        roles_x = (h0, i4, i4 + 1, i4 + 2, i4 + 3, torch.tensor([0, 1, 2, 4]), i4 * 10, ex0.tok_utt)
        roles_dyn = [{0: Ld}, {0: Pd}, {0: Pd}, {0: Pd}, {0: Pd}, {0: Pd}, {0: Pd}, {0: Ld}]
        if spk:
            Qd = torch.export.Dim("Q", min=PAD_MIN, max=1024)
            roles_x += (torch.tensor([-1, 0, -1, 1]), torch.tensor([0, 1, 1, -9]), torch.tensor([3, 5, 9]), torch.tensor([0, 0, NS_MAX]))
            roles_dyn += [{0: Pd}, {0: Pd}, {0: Qd}, {0: Qd}]
        t = time.time()
        eps = {
            "encode": torch.export.export(mods_q["encode"], (ids0,), dynamic_shapes=({1: Ld},), strict=False),
            "score": torch.export.export(mods_q["score"], (h0, i3, i3 + 1), dynamic_shapes=({0: Ld}, {0: Cd}, {0: Cd}),
                                         strict=False),
            "link": torch.export.export(mods_q["link"], (h0, i3, i3 + 1, torch.zeros(N_SLOTS, dtype=torch.long),
                                                         torch.zeros(N_SLOTS)),
                                        dynamic_shapes=({0: Ld}, {0: Ed}, {0: Ed}, None, None), strict=False),
            "roles": torch.export.export(mods_q["roles"], roles_x, dynamic_shapes=roles_dyn, strict=False),
            "attrs": torch.export.export(mods_q["attrs"], (h0, ex0.utt_mask.float(), i4, i4 + 1),
                                         dynamic_shapes=({0: Ld}, {0: Ld}, {0: Td}, {0: Td}), strict=False),
        }
    prog = to_edge_transform_and_lower(eps, partitioner=[XnnpackPartitioner()]).to_executorch()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "memnar.pte").write_bytes(prog.buffer)
    if remap is not None:
        (out / "vocab_remap.bin").write_bytes(remap.numpy().astype("<i4").tobytes())
    lb = labels
    meta = {"version": 4 if spk else 3, "run": run.name, "ckpt": a.ckpt, "speakers": spk, "ns_max": NS_MAX, "backbone": json.loads((run / "args.json").read_text())["backbone"], "d": d,
            "max_span": MAX_SPAN, "max_len": 1024, "n_slots": N_SLOTS, "chunk": CHUNK, "max_items": MAX_ITEMS,
            "pad_min": PAD_MIN, "n_dist": N_DIST, **th, "s2_lo": a.s2_lo, "s2_hi": a.s2_hi,
            "frames": lb.frames, "frame_roles": [[lb.roles[g][1] for g in ids] for ids in lb.frame_roles],
            "frame_core": frame_core(lb.frames),
            "types": lb.types, "coarse": ENTITY_LIST, "attr_keys": ATTR_KEYS, "attr_values": ATTR_VALUES,
            "flags": list(FLAGS), "bos_id": tok.bos_token_id, "vocab_pruned": remap is not None, "quant": a.quant,
            **({"vocab": vocab_info} if vocab_info else {})}
    (out / "memnar_meta.json").write_text(json.dumps(meta))
    tok.save_pretrained(out / "tokenizer")
    info = {"MB": round(len(prog.buffer) / 1e6), "export_s": round(time.time() - t)}
    print(json.dumps(info))

    # 3) the .pte through the same host logic vs eager fp32 extract3 on the checklist windows (quantization check)
    runner = pte_runner(prog.buffer, remap)
    agree_app = agree_s1 = 0
    jac, timings, fixtures = [], [], []
    for n, w in enumerate(probes + dev):
        ex = feats[w["window_id"]]
        t0 = time.perf_counter()
        g = host_extract3(runner, ex, labels, **kw_app, speakers=spk)
        timings.append(time.perf_counter() - t0)
        g6 = host_extract3(runner, ex, labels, **th, speakers=spk)
        ref = extract3(m_fp, ex, **kw_app)
        ref6 = extract3(m_fp, ex, **th)
        ma, ra = memories(g, w, 0.0), memories(ref, w, 0.0)
        m6, r6 = memories(g6, w, th["trg_thresh"]), memories(ref6, w, th["trg_thresh"])
        agree_app += ma == ra
        agree_s1 += m6 == r6
        jac.append(len(set(m6) & set(r6)) / max(1, len(set(m6) | set(r6))))
        if m6 != r6:
            print(".pte != fp32:", w["window_id"], "| only .pte:", sorted(set(m6) - set(r6)), "| only fp32:", sorted(set(r6) - set(m6)))
        if n < len(probes) + a.n_parity and spk:
            fixtures.append({"window": w, "input_ids": ex.input_ids.tolist(), "utt_mask": ex.utt_mask.int().tolist(),
                             "word_start": ex.word_start.int().tolist(), "word_end": ex.word_end.int().tolist(),
                             "tok_utt": ex.tok_utt.tolist(), "slot_idx": ex.slot_idx.tolist(), "user_idx": ex.user_idx,
                             "spk_names": ex.spk_names, "spk_tok": ex.spk_tok, "utt_spk": ex.utt_spk.tolist(),
                             "graph": g, "eager_graph_same": graph_key(g) == graph_key(ref)})
        elif n < len(probes) + a.n_parity:
            spk_l, txt = speakers_of(w)
            fixtures.append({"window": w, "input_ids": ex.input_ids.tolist(), "utt_mask": ex.utt_mask.int().tolist(),
                             "word_start": ex.word_start.int().tolist(), "word_end": ex.word_end.int().tolist(),
                             "tok_utt": ex.tok_utt.tolist(), "slot_idx": ex.slot_idx.tolist(), "user_idx": ex.user_idx,
                             "graph": g, "memories_all": memories(g, w, 0.0),
                             "memories_sure": memories(g, w, a.s2_hi),
                             "memories_unsure": [x for x in compile_memories(
                                 {**g, "frames": [f for f in g["frames"] if a.s2_lo <= f["p"] < a.s2_hi]},
                                 speakers=spk_l, utt_text=txt, context=g["frames"]) if x not in memories(g, w, a.s2_hi)],
                             "graph_s1": g6, "memories_s1": m6})
    nd = len(probes) + len(dev)
    info.update({"pte_s_per_window_mac": round(sum(timings) / len(timings), 3),
                 "pte_vs_fp32_windows_identical_s1": f"{agree_s1}/{nd}",
                 "pte_vs_fp32_windows_identical_app": f"{agree_app}/{nd}",
                 "pte_vs_fp32_mean_memory_jaccard_s1": round(sum(jac) / max(1, len(jac)), 3)})
    (out / ("parity4.json" if spk else "parity3.json")).write_text(json.dumps(fixtures))
    (out / "export_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
