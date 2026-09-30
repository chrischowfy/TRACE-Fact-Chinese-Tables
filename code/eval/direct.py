"""Direct prompting baselines (claim-only / full-table / oracle-evidence / oracle-program).

    export OPENROUTER_API_KEY=...
    python eval/direct.py --claims data/zhtabfact_v1/claims.jsonl --setting full-table \
        --model deepseek/deepseek-v4-flash --base-url https://openrouter.ai/api/v1 \
        --api-key-name OPENROUTER_API_KEY --out runs/eval

Writes <out>/<model>__<setting>/predictions.jsonl (one record per claim, raw output kept) and
run_meta.json with every decoding setting, so each row of the results table is reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--setting", default="full-table", choices=C.SETTINGS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-name", required=True)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--extra", default="{}", help='provider-specific JSON merged into the request, e.g. '
                                                  '\'{"reasoning": {"effort": "low"}}\'')
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = C.api_key(args.api_key_name)
    claims = [json.loads(line) for line in open(args.claims, encoding="utf-8")]
    if args.limit:
        claims = claims[: args.limit]
    run_dir = Path(args.out) / f"{args.model.replace('/', '_')}__{args.setting}"
    meta = {k: v for k, v in vars(args).items() if k != "api_key_name"} | {"system_prompt": C.SYSTEM}
    meta = C.prepare_run(run_dir, meta, args.claims)
    pred_path = run_dir / "predictions.jsonl"
    done = {}
    if pred_path.exists():
        for line in pred_path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("raw_output") is not None:
                done[r["id"]] = r
    pending = [c for c in claims if c["id"] not in done]
    extra = json.loads(args.extra)

    def one(c):
        messages = [{"role": "system", "content": C.SYSTEM}, {"role": "user", "content": C.build_prompt(c, args.setting)}]
        raw, info = C.call_llm(messages, model=args.model, base_url=args.base_url, api_key=key,
                               max_tokens=args.max_tokens, temperature=args.temperature,
                               json_mode=not args.no_json_mode, extra=extra)
        label, err = C.parse_label(raw)
        return {"id": c["id"], "input_sha256": C.input_hash(c), "gold": c["label"], "label": label,
                "parse_error": err, "raw_output": raw, "api": info, "run_fingerprint": meta["run_fingerprint"]}

    records = dict(done)
    with ThreadPoolExecutor(max_workers=args.workers) as ex, pred_path.open("a", encoding="utf-8") as fh:
        for i, fut in enumerate(as_completed([ex.submit(one, c) for c in pending]), 1):
            rec = fut.result()
            records[rec["id"]] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 100 == 0:
                print(f"{i}/{len(pending)}", flush=True)
    report = C.score(claims, {k: v["label"] for k, v in records.items()})
    (run_dir / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n", "accuracy", "macro_f1", "f1", "invalid")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
