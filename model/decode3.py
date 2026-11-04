"""Decode a v3 forward pass into a memory graph, and compile frames into memories.

Entities and triggers are picked greedily (highest p first, no overlap within a layer). Each trigger's roles are
filled from the entities, USER and the other triggers; a role keeps its best filler (two for Co_* roles).
compile_memories() turns each frame with at least one filled role into a readable memory line, e.g.

    Borrowing [has]: Borrower=Mike; Theme=my ladder; Lender=ME; Time=from july
"""

from __future__ import annotations

import re

import torch

from model.nar3 import ATTR_VALUES, Example3, NARMemory3, candidate_spans
from ontology.ontology import ENTITY_LIST, FLAGS


def _platt(p, a, b):
    lg = torch.logit(p.clamp(1e-6, 1 - 1e-6))
    return torch.sigmoid(a * lg + b)


@torch.no_grad()
def extract3(model: NARMemory3, ex: Example3, ent_thresh: float = 0.5, trg_thresh: float = 0.5,
             role_thresh: float = 0.35, max_items: int = 64) -> dict:
    lb = model.labels
    dev = next(model.parameters()).device
    ids = ex.input_ids[None].to(dev)
    h = model.encode(ids, torch.ones_like(ids))[0]
    t_ent, t_trg, t_role, t_attr = model.temps.tolist()
    pa = model.platt.tolist()
    cands = candidate_spans(ex.utt_mask, ex.word_start, ex.word_end)
    if not cands:
        return {"entities": [], "frames": [], "salience": 0.0}
    S = model.span_reps(h, cands)
    el = model.ent_logits(S) / t_ent
    p_ent = _platt(torch.sigmoid(torch.logsumexp(el[:, 1:], -1) - el[:, 0]), pa[0], pa[1])
    tl = model.trg_logits(S) / t_trg
    p_trg = _platt(torch.sigmoid(torch.logsumexp(tl[:, 1:], -1) - tl[:, 0]), pa[2], pa[3])

    def pick(p, thr):
        used = torch.zeros(len(ex.input_ids), dtype=torch.bool)
        out = []
        for i in p.argsort(descending=True).tolist():
            if p[i] < thr or len(out) >= max_items:
                break
            s, e = cands[i]
            if used[s:e + 1].any():
                continue
            used[s:e + 1] = True
            out.append(i)
        return sorted(out, key=lambda i: cands[i])

    meta = ex.meta
    text, offs = meta["text"], meta["offsets"]

    def span_text(s, e):
        return text[offs[s][0]:offs[e][1]].strip()

    def where(s):
        c = offs[s][1] - 1  # last char of the token (its start may be the space before the utterance)
        return max((i for i, st in enumerate(meta["utt_char_start"]) if st <= c), default=0)

    ents = []
    ei = pick(p_ent, ent_thresh)
    if ei:
        coarse = model.coarse_head(S[ei])[:, 1:].argmax(-1).tolist()
        tsoft = el[ei][:, 1:].softmax(-1)
        tp, tix = tsoft.max(-1)
        m = model.arg_proj(S[ei])
        slots = torch.cat([model.new_slot[None], h[ex.slot_idx.to(dev)]], 0) if len(ex.slot_idx) else model.new_slot[None]
        link = (model.coref_q(m) @ model.coref_k(slots).T / model.d ** 0.5).argmax(-1).tolist()
        for k, i in enumerate(ei):
            s, e = cands[i]
            ents.append({"id": f"e{k + 1}", "text": span_text(s, e), "utt": where(s), "type": lb.types[tix[k]],
                         "type_p": round(float(tp[k]), 3), "coarse": ENTITY_LIST[coarse[k]], "p": round(float(p_ent[i]), 3),
                         "link": meta["state_refs"][link[k] - 1] if link[k] > 0 else None})
    frames = []
    ti = pick(p_trg, trg_thresh)
    if ti:
        trg_s = S[ti]
        fsoft = tl[ti][:, 1:].softmax(-1)
        fp, fix = fsoft.max(-1)
        parts = []
        if ei:
            parts.append(model.arg_proj(S[ei]) + model.kind_emb.weight[0])
        parts.append(model.arg_proj(model.span_reps(h, [(ex.user_idx, ex.user_idx)])) + model.kind_emb.weight[1])
        parts.append(model.arg_proj(trg_s) + model.kind_emb.weight[2])
        spk = model.speaker_reps(h, ex) if getattr(model, "speakers", False) else None
        n_spk = 0 if spk is None else len(spk)
        if spk is not None:
            parts.append(spk)
        A = torch.cat(parts, 0)
        names = [e["id"] for e in ents] + ["USER"] + [f"f{k + 1}" for k in range(len(ti))] + list(ex.spk_names[:n_spk])
        tq = model.arg_proj(trg_s) + model.kind_emb.weight[2]
        pos = [cands[i][0] for i in ei] + [-1] + [cands[i][0] for i in ti] + [-1] * n_spk
        sp_kw = {}
        if getattr(model, "speakers", False):
            a_spk = [-1] * len(ei) + [0] + [-1] * len(ti) + list(range(1, n_spk + 1))
            sp_kw = {"a_spk": torch.tensor(a_spk, device=dev), "utt_spk": ex.utt_spk.to(dev)}
        rl = model.role_logits(model.pair_reps(tq, A, torch.tensor([cands[i][0] for i in ti], device=dev),
                                               torch.tensor(pos, device=dev), ex.tok_utt.to(dev), **sp_kw), fix) / t_role
        rp = rl.softmax(-1)                                     # T x A x (1 + R)
        pooled = h[ex.utt_mask.to(dev)].mean(0)
        z = torch.cat([trg_s, pooled[None].expand(len(ti), -1)], -1)
        attrs = {k: [ATTR_VALUES[k][j] for j in (head(z) / t_attr).argmax(-1).tolist()] for k, head in model.attr_heads.items()}
        flags = torch.sigmoid(model.flag_head(z)).tolist()
        for k, i in enumerate(ti):
            s, e = cands[i]
            fid = int(fix[k])
            role_names = [lb.roles[g][1] for g in lb.frame_roles[fid]]
            best: dict[str, list] = {}
            for a in range(len(names)):
                if names[a] == f"f{k + 1}":
                    continue
                j = int(rp[k, a, 1:].argmax())
                pr = float(_platt(rp[k, a, 1 + j], pa[4], pa[5]))
                if j < len(role_names) and pr >= role_thresh:
                    best.setdefault(role_names[j], []).append((pr, names[a]))
            roles = {}
            for r, lst in best.items():
                lst.sort(reverse=True)
                roles[r] = [n for _, n in lst[: 2 if r.startswith("Co") else 1]]
            frames.append({"id": f"f{k + 1}", "frame": lb.frames[fid], "frame_p": round(float(fp[k]), 3),
                           "trigger": span_text(s, e), "utt": where(s), "p": round(float(p_trg[i]), 3),
                           "roles": roles, **{a: attrs[a][k] for a in attrs},
                           **{f: v > 0.5 for f, v in zip(FLAGS, flags[k])}})
    sal = float(torch.sigmoid(model.sal_head(h[ex.utt_mask.to(dev)].mean(0))))
    return {"entities": ents, "frames": frames, "salience": round(sal, 3)}


_CORE = None
_FIRST = {"i", "me", "myself", "we", "us"}
_YOU = re.compile(r"\b(you|your|yours|yourself)\b", re.I)


def _core(frame: str) -> set:
    global _CORE
    if _CORE is None:
        from ontology.grammar import load
        _CORE = {k: set(v.get("core", v["roles"])) | {"Condition", "New_value"} for k, v in load()["frames"].items()}
    return _CORE.get(frame, set())


def _clip(text: str, trigger: str, width: int = 200) -> str:
    """The source line, cut to about width characters around the trigger."""
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    i = max(0, text.lower().find(trigger.lower()))
    a = max(0, min(i - width // 2, len(text) - width))
    return ("…" if a else "") + text[a:a + width] + ("…" if a + width < len(text) else "")


def compile_memories(g: dict, min_p: float = 0.0, require_core: bool = False, min_importance: str = "low",
                     speakers: list[str] | None = None, utt_text: list[str] | None = None,
                     min_roles: int = 1, context: list[dict] | None = None, evidence: bool = False) -> list[str]:
    """One readable line per frame that has at least one filled role (optionally: a core role, an importance
    floor); identical lines are kept once. speakers[i] = "ME" or the speaker label of utterance i: a frame evoked in
    someone else's line is recorded as theirs ("S2 said: ..."). With utt_text, a role the model filled with ME inside
    someone else's line that never says "you"/"your" is given to that speaker ("my husband" in S2's line is S2's).
    context: all frames of the window, for showing a role filled by a frame that g leaves out (eval/s2_v3.py compiles
    only the uncertain band; without it such a filler printed as its raw id, "Message=f11").
    evidence: append the line the frame was evoked in (needs utt_text), as the memory's source."""
    name = {"USER": "ME", **{e["id"]: e["text"] for e in g["entities"]}}
    fr = {f["id"]: f for f in (context or g["frames"])}
    rank = {"low": 0, "medium": 1, "high": 2}
    out = []
    for f in g["frames"]:
        if f["p"] < min_p or not f["roles"]:
            continue
        if sum(len(v) for v in f["roles"].values()) < min_roles:
            continue
        if require_core and not (set(f["roles"]) & _core(f["frame"])):
            continue
        if rank.get(f.get("importance", "medium"), 1) < rank[min_importance]:
            continue
        other = speakers[f["utt"]] if speakers and 0 <= f.get("utt", -1) < len(speakers) and speakers[f["utt"]] != "ME" else None
        name_here = {**name, "USER": other} if other and utt_text and not _YOU.search(utt_text[f["utt"]]) else name
        me = other or "ME"  # first-person pronouns kept as entities mean the speaker of that line
        name_here = {k: (me if k != "USER" and v.lower() in _FIRST else v) for k, v in name_here.items()}

        def show(x, nm=name_here):
            if x in fr:
                return f"[{fr[x]['frame'].replace('_', ' ')}: {fr[x]['trigger']}]"
            return nm.get(x, x)
        roles = "; ".join(f"{r}={' & '.join(show(x) for x in xs)}" for r, xs in f["roles"].items())
        tags = [f.get("modality", "asserted")] if f.get("modality", "asserted") != "asserted" else []
        if f.get("tense") in ("past", "future"):
            tags.append(f["tense"])
        if f.get("update_of_prior"):
            tags.append("changed")
        line = f"{f['frame'].replace('_', ' ')} [{f['trigger']}]: {roles}" + (f" ({', '.join(tags)})" if tags else "")
        if speakers and 0 <= f.get("utt", -1) < len(speakers) and speakers[f["utt"]] != "ME":
            line = f"{speakers[f['utt']]} said: {line}"
        if evidence and utt_text and 0 <= f.get("utt", -1) < len(utt_text):
            line += f' | source: "{_clip(utt_text[f["utt"]], f["trigger"])}"'
        if line not in out:
            out.append(line)
    return out
