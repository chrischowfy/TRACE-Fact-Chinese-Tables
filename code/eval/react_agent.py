"""ReAcTable-style agent with complete observations.

The database, prompt and loop are those of sql_agents.reactable (Zhang et al., 2024): up to --max-turns SELECTs
whose results become queryable tables step1..stepK, then an explicit final label; no final label is INVALID.
One difference: an observation states how many rows a query returned and shows up to --show-rows of them.
sql_agents.reactable shows the first six rows and does not say when rows are cut, which makes the agent treat a
long result as complete; that function is kept unchanged for the runs made with it.

    python eval/react_agent.py --claims data/claims.jsonl --model deepseek-chat \
        --base-url https://api.deepseek.com --api-key-name DEEPSEEK_API_KEY
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import common as C  # noqa: E402
import sql_agents as A  # noqa: E402


def run_sql(conn: sqlite3.Connection, sql: str | None, show_rows: int):
    """((columns, rows shown, rows returned), None) or (None, error); statements are restricted as in sql_agents."""
    if not isinstance(sql, str) or not sql.strip():
        return None, "empty_sql"
    stmt = sql.strip().rstrip(";")
    if A.WRITE_RE.search(stmt) or not stmt.lower().lstrip().startswith(("select", "with")):
        return None, "not_read_only_select"
    try:
        steps = 0

        def limit_steps():
            nonlocal steps
            steps += 1
            return int(steps > 10000)
        conn.set_progress_handler(limit_steps, 1000)
        cur = conn.execute(stmt)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(5000)
        return (cols, rows[:show_rows], len(rows)), None
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return None, f"sql_error: {str(exc)[:120]}"


def reactable(claim: dict, llm, max_turns: int = 5, show_rows: int = 30) -> dict:
    conn, schema = A.build_db(claim)
    history, rec = [], {"id": claim["id"], "gold": claim["label"], "turns": []}
    try:
        for k in range(1, max_turns + 1):
            user = (f"声明：{claim['claim']}\n\n表：\n{schema}\n\n" + ("已执行：\n" + "\n".join(history) + "\n\n" if history else "")
                    + f"第 {k}/{max_turns} 步，只输出 JSON。")
            obj = A._json(llm([{"role": "system", "content": A.RT_SYSTEM}, {"role": "user", "content": user}]))
            rec["turns"].append(obj)
            final = str(obj.get("final_label", "")).upper()
            if final in C.LABELS:
                rec.update(label=final, status="answered")
                return rec
            result, err = run_sql(conn, obj.get("sql"), show_rows)
            if err:
                history.append(f"步骤{k}：{obj.get('sql')} -> 失败（{err}）")
                continue
            cols, rows, total = result
            try:
                conn.execute(f"CREATE TEMP TABLE step{k} AS {obj['sql'].strip().rstrip(';')}")
                saved = f"已存为 step{k}"
            except (sqlite3.Error, sqlite3.Warning) as exc:
                saved = f"未保存中间表：{str(exc)[:120]}"
            shown = f"共 {total} 行" + (f"，只显示前 {len(rows)} 行" if total > len(rows) else "")
            history.append(f"步骤{k}：{obj.get('sql')} -> 列{cols} {shown} 行{rows}（{saved}）")
        rec.update(label="INVALID", status="no_final_label")
        return rec
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-name", required=True)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--show-rows", type=int, default=30)
    ap.add_argument("--extra", default="{}", help="provider-specific JSON merged into each request")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = C.api_key(args.api_key_name)
    claims = [json.loads(line) for line in open(args.claims, encoding="utf-8")]
    if args.limit:
        claims = claims[: args.limit]
    # its own run directory: <model>__reactable holds the runs of sql_agents.reactable
    run_dir = Path(args.out) / f"{args.model.replace('/', '_')}__react-agent"
    meta = C.prepare_run(run_dir, vars(args) | {"agent": "react-agent", "rt_system": A.RT_SYSTEM,
                                                "react_agent_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                         args.claims)
    extra = json.loads(args.extra)

    def one(claim):
        calls = []

        def llm(messages):
            raw, info = C.call_llm(messages, model=args.model, base_url=args.base_url, api_key=key,
                                   max_tokens=args.max_tokens, extra=extra)
            calls.append({"raw_output": raw, "api": info})
            return raw
        rec = reactable(claim, llm, args.max_turns, args.show_rows)
        rec.update(input_sha256=C.input_hash(claim), api_calls=calls, run_fingerprint=meta["run_fingerprint"])
        return rec
    records = {}
    pred_path = run_dir / "predictions.jsonl"
    if pred_path.exists():                       # resume: keep claims whose every call got a response
        for line in pred_path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") and not any((c.get("api") or {}).get("error") for c in r["api_calls"]):
                records[r["id"]] = r
    ids = {c["id"] for c in claims}
    records = {k: v for k, v in records.items() if k in ids}
    pending = [c for c in claims if c["id"] not in records]
    print(f"pending {len(pending)} / done {len(records)}", flush=True)
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in records.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as ex, pred_path.open("a", encoding="utf-8") as fh:
        for fut in as_completed([ex.submit(one, c) for c in pending]):
            rec = fut.result()
            records[rec["id"]] = rec
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    report = C.score(claims, {k: v["label"] for k, v in records.items()})
    report["status"] = {s: sum(1 for r in records.values() if r.get("status") == s)
                        for s in sorted({r.get("status") for r in records.values()})}
    (run_dir / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n", "accuracy", "macro_f1", "f1", "invalid", "status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
