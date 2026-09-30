"""Score prediction files against a claims file.

    python eval/score.py --claims data/zhtabfact_v1/claims.jsonl --pred flash=runs/eval/x__full-table/predictions.jsonl \
        [--pred sql=runs/eval/y__text2sql/predictions.jsonl] [--eval-dir runs/eval] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load(path: Path) -> dict[str, str]:
    out = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        out[r["id"]] = r.get("label", "INVALID")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--pred", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--eval-dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    pairs = [(p.parent.name, p) for p in sorted(Path(args.eval_dir).glob("*/predictions.jsonl"))] if args.eval_dir else []
    for item in args.pred:
        name, _, path = item.partition("=")
        pairs.append((name, Path(path)))
    if not pairs:
        sys.exit("give --pred NAME=PATH or --eval-dir DIR")
    claims = [json.loads(line) for line in open(args.claims, encoding="utf-8")]
    report = {}
    print(f"{'system':40s} {'Acc':>6} {'MacroF1':>8} {'S-F1':>6} {'R-F1':>6} {'N-F1':>6} {'invalid':>7}  macroF1 95%CI")
    for name, path in pairs:
        s = C.score(claims, load(path))
        report[name] = s
        print(f"{name:40s} {s['accuracy']:6.1f} {s['macro_f1']:8.1f} {s['f1']['SUPPORTS']:6.1f} {s['f1']['REFUTES']:6.1f} "
              f"{s['f1']['NEI']:6.1f} {s['invalid']:7d}  {s['macro_f1_ci95']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
