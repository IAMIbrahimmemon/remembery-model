"""Pool the graphs of several System 1 runs and write their memories in each format of model/formats.py.

    PYTHONPATH=. uv run python -m eval.pool_formats --windows data/eval/v3/dev.jsonl \
        --preds runs/eval_v3/v4e_best_ev_dev.jsonl runs/eval_v3/v4f_last_ev_dev.jsonl \
        --formats base A B C E D G --out runs/eval_v3/pool_{fmt}_dev.jsonl
"""

from __future__ import annotations

import argparse
import json

from model.formats import render


def speakers_text(w: dict) -> tuple[list[str], list[str]]:
    user = next((k for k, v in w["speakers"].items() if v == "USER"), "S1")
    return ["ME" if u["spk"] == user else u["spk"] for u in w["utterances"]], [u["text"] for u in w["utterances"]]


def pooled(graphs: list[dict], fmt: str, sp, tx) -> list[str]:
    out = []
    for g in graphs:
        out += render(g, fmt, sp, tx)
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, help="windows, or a memeval set (rows with a 'window')")
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--formats", nargs="+", default=["base"])
    ap.add_argument("--out", required=True, help="path with {fmt}")
    a = ap.parse_args()
    ws = [json.loads(l) for l in open(a.windows)]
    ws = [r.get("window", r) for r in ws]
    preds = [{json.loads(l)["window_id"]: json.loads(l)["graph"] for l in open(p)} for p in a.preds]
    for fmt in a.formats:
        rows = []
        for w in ws:
            sp, tx = speakers_text(w)
            rows.append({"window_id": w["window_id"],
                         "memories": pooled([p[w["window_id"]] for p in preds if w["window_id"] in p], fmt, sp, tx)})
        with open(a.out.format(fmt=fmt), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        print(fmt, sum(len(r["memories"]) for r in rows), "memories")


if __name__ == "__main__":
    main()
