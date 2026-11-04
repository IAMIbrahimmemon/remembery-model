"""Run a v3 System 1 run dir on windows: memories for eval/memeval.py, plus a hand-audit page.

    PYTHONPATH=. uv run python -m eval.predict3 --run runs/v3a --windows runs/eval_v3/dev_windows.jsonl \
        --out runs/eval_v3/v3a_dev.jsonl --audit docs/audits/2026-10-29_v3a.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.decode3 import compile_memories, extract3
from model.nar3 import LabelSet, NARMemory3, featurize3
from ontology.grammar import load as load_grammar


def load3(run: str, device: str = "cpu", ckpt: str = "best.pt"):
    r = Path(run)
    args = json.loads((r / "args.json").read_text())
    labels = LabelSet.from_json(json.loads((r / "labels3.json").read_text()), load_grammar()["frames"])
    tok = AutoTokenizer.from_pretrained(args["backbone"], trust_remote_code=True)
    model = NARMemory3(args["backbone"], labels, d=args["d"], adapter_layers=args["adapter_layers"],
                       speakers=args.get("speakers", False))
    sd = torch.load(r / ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected and all(k in ("platt", "temps", "pos_w") for k in missing), (missing, unexpected)
    th = r / "thresholds3.json"
    kw = json.loads(th.read_text()) if th.exists() else {}
    return model.to(device).eval(), tok, labels, kw


def audit_md(w: dict, g: dict, mems: list[str]) -> str:
    user = next((s for s, v in w["speakers"].items() if v == "USER"), "S1")
    out = [f"## {w['window_id']} ({w.get('scene', '')})", ""]
    out += [f"{i}. **{'Me' if u['spk'] == user else u['spk']}**: {u['text']}" for i, u in enumerate(w["utterances"])]
    out += ["", "Entities: " + "; ".join(f"{e['text']} <{e['type']}/{e['coarse']}> {e['p']:.2f}" for e in g["entities"]), "",
            "Memories:", ""]
    fr = {f["id"]: f for f in g["frames"]}
    for m, f in zip(mems, [f for f in g["frames"] if f["roles"]]):
        out.append(f"- {m}  (p {f['p']:.2f}, frame p {f['frame_p']:.2f}, {f.get('importance')})")
    del fr
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--ent-thresh", type=float)
    ap.add_argument("--trg-thresh", type=float)
    ap.add_argument("--role-thresh", type=float)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-importance", default="low", choices=("low", "medium", "high"),
                    help="drop frames whose predicted importance is below this")
    ap.add_argument("--evidence", action="store_true", help="append each memory's source line")
    a = ap.parse_args()
    model, tok, labels, kw = load3(a.run, a.device, a.ckpt)
    for k in ("ent_thresh", "trg_thresh", "role_thresh"):
        if getattr(a, k) is not None:
            kw[k] = getattr(a, k)
    preds, pages = [], []
    for line in open(a.windows):
        w = json.loads(line)
        ex = featurize3({"window": w, "labels": None}, tok, labels)
        g = extract3(model, ex, **kw)
        user = next((k for k, v in w["speakers"].items() if v == "USER"), "S1")
        mems = compile_memories(g, min_p=kw.get("trg_thresh", 0.0), min_importance=a.min_importance, evidence=a.evidence,
                                speakers=["ME" if u["spk"] == user else u["spk"] for u in w["utterances"]],
                                utt_text=[u["text"] for u in w["utterances"]])
        preds.append({"window_id": w["window_id"], "memories": mems, "graph": g})
        pages.append(audit_md(w, g, mems))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(p) + "\n" for p in preds))
    if a.audit:
        Path(a.audit).parent.mkdir(parents=True, exist_ok=True)
        Path(a.audit).write_text("\n".join(pages))
    print(len(preds), "windows,", sum(len(p["memories"]) for p in preds), "memories", kw)


if __name__ == "__main__":
    main()
