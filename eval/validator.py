"""Line validator for format G: keep a candidate source line only if it is worth remembering.

Features come from System 1's own graphs (two pooled models), so the phone computes them for free; the logistic weights
are in model/validator_g2.json (cutoff 0.60). The Swift port is ios/MemNARCore/Sources/MemNARCore/MemoryLines.swift.

    PYTHONPATH=. uv run python -m eval.validator apply --model model/validator_g2.json --windows WINDOWS.jsonl \
        --preds V4E_GRAPHS.jsonl V4F_GRAPHS.jsonl --out MEMORIES.jsonl --records RECORDS.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from eval.pool_formats import speakers_text
from model.decode3 import _clip

RANK = {"low": 0, "medium": 1, "high": 2}
GROUPS = {"PERSON": "who", "GROUP": "who", "ORG": "who", "ROLE": "who", "PET": "who", "DATETIME": "when",
          "DURATION": "when", "RECURRENCE": "when", "PLACE": "where", "VENUE": "where", "ADDRESS": "where",
          "SPOT": "where", "MONEY": "num", "QUANTITY": "num", "CONTACT": "num", "IDENTIFIER": "num",
          "HEALTH": "health", "MEDICATION": "health", "DIET": "health", "EVENT": "event", "APPOINTMENT": "event",
          "TASK": "event", "TRIP": "event", "MILESTONE": "event"}
NAMES = ["bias", "n_models", "n_frames", "max_p", "mean_p", "max_fp", "max_imp", "n_ents", "who", "when", "where",
         "num", "health", "event", "log_words", "is_me", "digits", "question", "only_opinion", "salience", "about_user"]
NAMES2 = NAMES + ["strong", "no_frame", "short", "pronoun_start", "prev_has_ents"]
V1 = {"min_p": 0.5, "ent_lines": False, "clip": 200, "ctx": False, "feat": 1}
_PRON_START = re.compile(r"^\W*(it|that|this|they|them|he|she|those|these|yes|yeah|no|so|and|but|oh)\b", re.I)
_KEY = ("when", "where", "num", "health", "event")


def line_features(u: int, graphs: list[dict], sp, tx, spec: dict = V1) -> tuple[list[float], list[str]] | None:
    """Features of utterance u from the pooled graphs, or None if u is not a candidate. A candidate has a frame with
    p >= min_p and a role, or (ent_lines) an entity of a key kind (time, place, number, health, event)."""
    fs = [(g, f) for g in graphs for f in g["frames"] if f["utt"] == u and f["p"] >= spec["min_p"] and f["roles"]]
    ents = [e for g in graphs for e in g["entities"] if e["utt"] == u]
    grp = {k: 0.0 for k in ("who", "when", "where", "num", "health", "event")}
    for e in ents:
        if e["coarse"] in GROUPS:
            grp[GROUPS[e["coarse"]]] = 1.0
    if not fs and not (spec["ent_lines"] and any(grp[k] for k in _KEY)):
        return None
    names = [f["frame"] for _, f in fs]
    ps = [f["p"] for _, f in fs] or [0.0]
    x = [1.0, float(len({id(g) for g, _ in fs})), float(len(fs)) / len(graphs), max(ps), sum(ps) / len(ps),
         max([f["frame_p"] for _, f in fs] or [0.0]), max([RANK.get(f.get("importance"), 1) for _, f in fs] or [0]) / 2,
         len(ents) / len(graphs), *grp.values(),
         math.log1p(len(tx[u].split())), float(sp[u] == "ME"), float(any(c.isdigit() for c in tx[u])),
         float("?" in tx[u]), float(bool(names) and set(names) == {"Opinion"}),
         sum(g.get("salience", 0.5) for g in graphs) / len(graphs), float(any(f.get("about_user") for _, f in fs))]
    if spec["feat"] >= 2:
        prev_ents = [e for g in graphs for e in g["entities"] if e["utt"] == u - 1]
        x += [float(max(ps) >= 0.5), float(not fs), float(len(tx[u].split()) < 8), float(bool(_PRON_START.search(tx[u]))),
              float(bool(prev_ents))]
    return x, list(dict.fromkeys(n.replace("_", " ") for n in names))


def needs_context(u: int, tx) -> bool:
    return u > 0 and (len(tx[u].split()) < 8 or bool(_PRON_START.search(tx[u])))


def g_line(u, frames, sp, tx, spec: dict = V1) -> str:
    text = _clip(tx[u], "", spec["clip"]) if spec["clip"] else " ".join(tx[u].split())
    line = f"[{', '.join(frames)}] {sp[u]}: \"{text}\"" if frames else f"{sp[u]}: \"{text}\""
    if spec["ctx"] and needs_context(u, tx):
        line += f" (after {sp[u - 1]}: \"{_clip(tx[u - 1], '', 160)}\")"
    return line


def record(w: dict, u: int, graphs: list[dict], sp, tx, line: str) -> dict:
    """The structured memory the glasses store: the rendered line plus its frames, roles, entities and context."""
    from model.formats import _names
    fr = []
    for g in graphs:
        for f in g["frames"]:
            if f["utt"] == u and f["roles"]:
                name = _names(g, sp, tx, f, resolve=True)  # speaker rules, and pronouns resolved to their entity
                fid = {x["id"]: x for x in g["frames"]}
                show = lambda x: f"[{fid[x]['frame'].replace('_', ' ')}: {fid[x]['trigger']}]" if x in fid else name.get(x, x)
                item = {"frame": f["frame"], "trigger": f["trigger"], "p": f["p"], "modality": f.get("modality"),
                        "tense": f.get("tense"), "roles": {r: [show(x) for x in xs] for r, xs in f["roles"].items()}}
                same = next((o for o in fr if (o["frame"], o["trigger"], o["roles"]) == (item["frame"], item["trigger"], item["roles"])), None)
                if same:  # both models found it
                    same["p"] = max(same["p"], item["p"])
                else:
                    fr.append(item)
    ents = {(e["text"], e["coarse"]): e.get("type") for g in graphs for e in g["entities"] if e["utt"] == u}
    return {"window_id": w["window_id"], "date": w.get("date"), "utt": u, "speaker": sp[u], "text": tx[u], "line": line,
            "frames": fr, "entities": [{"text": t, "coarse": c, "type": ty} for (t, c), ty in ents.items()],
            "context": {"speaker": sp[u - 1], "text": tx[u - 1]} if u > 0 else None}


def build(windows, preds, spec: dict = V1):
    for w in windows:
        graphs = [p[w["window_id"]] for p in preds if w["window_id"] in p]
        sp, tx = speakers_text(w)
        for u in range(len(tx)):
            r = line_features(u, graphs, sp, tx, spec)
            if r:
                yield w, u, r[0], g_line(u, r[1], sp, tx, spec)


def _sig(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def score(m, x):
    return _sig(sum(w * (v - mu) / sd for w, v, mu, sd in zip(m["w"], x, m["mu"], m["sd"])))


def apply(a):
    m = json.loads(Path(a.model).read_text())
    t = a.threshold if a.threshold is not None else m["threshold"]
    spec = {**V1, **m.get("spec", {}), **json.loads(a.render)}  # rendering options (clip, ctx) may be changed here
    ws = [r.get("window", r) for r in map(json.loads, open(a.windows))]
    preds = [{json.loads(l)["window_id"]: json.loads(l)["graph"] for l in open(p)} for p in a.preds]
    out, recs, n = {w["window_id"]: [] for w in ws}, [], 0
    for w, u, x, line in build(ws, preds, spec):
        n += 1
        if score(m, x) >= t:
            out[w["window_id"]].append(line)
            if a.records:
                sp, tx = speakers_text(w)
                recs.append(record(w, u, [p[w["window_id"]] for p in preds if w["window_id"] in p], sp, tx, line))
    Path(a.out).write_text("".join(json.dumps({"window_id": k, "memories": v}) + "\n" for k, v in out.items()))
    if a.records:
        Path(a.records).write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(sum(map(len, out.values())), "of", n, "lines kept at", t, spec)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    y = sp.add_parser("apply")
    y.add_argument("--model", required=True)
    y.add_argument("--windows", required=True)
    y.add_argument("--preds", nargs="+", required=True)
    y.add_argument("--out", required=True)
    y.add_argument("--threshold", type=float)
    y.add_argument("--render", default="{}", help='override rendering, e.g. {"clip":0,"ctx":true}')
    y.add_argument("--records", help="also write the structured memory records here")
    a = ap.parse_args()
    apply(a)


if __name__ == "__main__":
    main()
