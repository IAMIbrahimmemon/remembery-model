"""Run the final System 1 on transcript windows: two models, pooled graphs, format G lines, line validator.

This is the system that scores 0.781 (dev) / 0.791 (sealed). Checkpoints come from the Hugging Face model repo
(checkpoints/v4e = best.pt, checkpoints/v4f = last.pt).

    PYTHONPATH=. uv run python memories.py --ckpts CKPT_DIR --windows data/probe/probe_v1.jsonl \
        --out memories.jsonl --records records.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.pool_formats import speakers_text
from eval.predict3 import load3
from eval.validator import V1, build, record, score
from model.decode3 import extract3
from model.nar3 import featurize3

MODELS = [("v4e", "best.pt"), ("v4f", "last.pt")]
CUTOFFS = {"ent_thresh": 0.5, "trg_thresh": 0.5, "role_thresh": 0.35}  # what the validator was trained with


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="folder with v4e/ and v4f/ (args.json, labels3.json, checkpoint)")
    ap.add_argument("--windows", required=True, help="one window per line (see data/probe/probe_v1.jsonl)")
    ap.add_argument("--validator", default="model/validator_g2.json")
    ap.add_argument("--threshold", type=float, help="validator cutoff (default: the tested 0.60)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", help="also write the structured memory records")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    ws = [json.loads(l) for l in open(a.windows)]
    preds = []
    for run, ckpt in MODELS:  # one model in memory at a time
        model, tok, labels, _ = load3(str(Path(a.ckpts) / run), a.device, ckpt)
        preds.append({w["window_id"]: extract3(model, featurize3({"window": w, "labels": None}, tok, labels), **CUTOFFS)
                      for w in ws})
        del model
        print(run, "done")

    m = json.loads(Path(a.validator).read_text())
    t = a.threshold if a.threshold is not None else m["threshold"]
    out, recs = {w["window_id"]: [] for w in ws}, []
    for w, u, x, line in build(ws, preds, {**V1, **m["spec"]}):
        if score(m, x) >= t:
            out[w["window_id"]].append(line)
            sp, tx = speakers_text(w)
            recs.append(record(w, u, [p[w["window_id"]] for p in preds], sp, tx, line))
    Path(a.out).write_text("".join(json.dumps({"window_id": k, "memories": v}) + "\n" for k, v in out.items()))
    if a.records:
        Path(a.records).write_text("".join(json.dumps(r) + "\n" for r in recs))
    for k, v in out.items():
        print(f"\n{k}: {len(v)} memories")
        for line in v:
            print(" ", line)


if __name__ == "__main__":
    main()
