"""Executable-reasoning baselines over an in-memory SQLite copy of each evidence package.

  text2sql   one query: the model returns {"sql": ..., "abstain": bool}; 1 -> SUPPORTS, 0 -> REFUTES,
             abstain=true -> NEI. A query that still fails after one repair round, returns no row, or
             returns a non-boolean value is INVALID (scored wrong) — it is NOT mapped to NEI.
  reactable  ReAcTable-style loop (Zhang et al., 2024): up to --max-turns SELECTs whose results become
             queryable tables step1..stepK, then an explicit final label. No final label -> INVALID.

    python eval/sql_agents.py --agent text2sql --claims data/zhtabfact_v1/claims.jsonl \
        --model deepseek-v4-flash --base-url https://api.deepseek.com/v1 --api-key-name DEEPSEEK_API_KEY
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import common as C  # noqa: E402
from dart_fact.tables import parse_number  # noqa: E402

WRITE_RE = re.compile(r"\b(insert|update|delete|drop|alter|attach|detach|pragma|vacuum|create|replace)\b", re.I)


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def build_db(claim: dict) -> tuple[sqlite3.Connection, str]:
    conn = sqlite3.connect(":memory:")
    lines = []
    for i, t in enumerate(claim["tables"]):
        headers, seen = [], {}
        for h in t["headers"]:
            h = str(h) or "col"
            seen[h] = seen.get(h, 0) + 1
            headers.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
        width = len(headers)
        rows = [list(r) + [""] * (width - len(r)) for r in t["rows"]]
        numeric = [ci for ci in range(width)
                   if sum(parse_number(r[ci]) is not None for r in rows) >= 0.8 * max(1, sum(1 for r in rows if r[ci]))]
        cols = [f"{_q(h)} TEXT" for h in headers] + [f"{_q(headers[ci] + '__num')} REAL" for ci in numeric]
        conn.execute(f"CREATE TABLE t{i} ({', '.join(cols)})")
        for r in rows:
            conn.execute(f"INSERT INTO t{i} VALUES ({','.join('?' * len(cols))})",
                         [str(x) for x in r] + [parse_number(r[ci]) for ci in numeric])
        line = f"t{i}（{t.get('title') or t['table_id']}）列：" + "，".join(_q(h) for h in headers)
        if numeric:
            line += "；数值列：" + "，".join(_q(headers[ci] + "__num") for ci in numeric)
        lines.append(line)
        lines.extend("  例：" + " | ".join(r) for r in rows[:3])
    return conn, "\n".join(lines)


def run_sql(conn: sqlite3.Connection, sql: str | None, max_rows: int = 6):
    if not isinstance(sql, str) or not sql.strip():
        return None, "empty_sql"
    stmt = sql.strip().rstrip(";")
    if WRITE_RE.search(stmt) or not stmt.lower().lstrip().startswith(("select", "with")):
        return None, "not_read_only_select"
    try:
        # Bound expensive generated joins/recursive queries. Interrupted execution is INVALID.
        steps = 0
        def limit_steps():
            nonlocal steps
            steps += 1
            return int(steps > 10000)
        conn.set_progress_handler(limit_steps, 1000)
        cur = conn.execute(stmt)
        cols = [d[0] for d in cur.description or []]
        return (cols, cur.fetchmany(max_rows)), None
    except (sqlite3.Error, sqlite3.Warning) as exc:
        return None, f"sql_error: {str(exc)[:120]}"


def _json(raw: str | None) -> dict:
    if not raw:
        return {}
    for cand in [raw] + re.findall(r"\{.*\}", raw, re.S):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {}


T2S_SYSTEM = ("你用 SQLite 核查中文表格事实声明，只能依据给定的表。写一条 SELECT 查询：声明为真返回 1，为假返回 0。"
              "如果表中没有判定所需的信息，把 abstain 设为 true。列名和表名用双引号，数值比较用 __num 列。只输出 JSON。")


def text2sql(claim: dict, llm) -> dict:
    conn, schema = build_db(claim)
    try:
        return _text2sql(claim, llm, conn, schema)
    finally:
        conn.close()


def _text2sql(claim, llm, conn, schema):
    user = f"声明：{claim['claim']}\n\n表：\n{schema}\n\n只输出 JSON：{{\"sql\": \"SELECT ...\", \"abstain\": false}}"
    obj = _json(llm([{"role": "system", "content": T2S_SYSTEM}, {"role": "user", "content": user}]))
    rec = {"id": claim["id"], "gold": claim["label"], "turns": []}
    for attempt in range(2):
        rec["turns"].append(obj)
        if obj.get("abstain") is True:
            rec.update(label="NEI", status="abstain")
            return rec
        result, err = run_sql(conn, obj.get("sql"))
        if not err:
            rows = result[1]
            value = rows[0][0] if rows and rows[0] else None
            if len(rows) == 1 and len(rows[0]) == 1 and value in (1, 0, 1.0, 0.0, "1", "0"):
                rec.update(label="SUPPORTS" if float(value) == 1 else "REFUTES", status="executed")
            else:
                rec.update(label="INVALID", status="non_boolean_or_empty_result", value=str(value))
            return rec
        if attempt == 0:
            repair = user + f"\n\n上一条 SQL 执行失败（{err}）：{obj.get('sql')}\n请修正后只输出 JSON。"
            obj = _json(llm([{"role": "system", "content": T2S_SYSTEM}, {"role": "user", "content": repair}]))
    rec.update(label="INVALID", status="failed_after_repair")
    return rec


RT_SYSTEM = ("你是表格事实核查 agent。每一步输出一个 JSON：要么 {\"thought\": ..., \"sql\": \"SELECT ...\"} 继续查询"
             "（结果会保存为可继续查询的表 stepK），要么 {\"thought\": ..., \"final_label\": \"SUPPORTS|REFUTES|NEI\"}。"
             "SUPPORTS=表格蕴含声明，REFUTES=表格与声明矛盾，NEI=表格信息不足。只依据表格。")


def reactable(claim: dict, llm, max_turns: int = 5) -> dict:
    conn, schema = build_db(claim)
    try:
        return _reactable(claim, llm, max_turns, conn, schema)
    finally:
        conn.close()


def _reactable(claim, llm, max_turns, conn, schema):
    history, rec = [], {"id": claim["id"], "gold": claim["label"], "turns": []}
    for k in range(1, max_turns + 1):
        user = (f"声明：{claim['claim']}\n\n表：\n{schema}\n\n" + ("已执行：\n" + "\n".join(history) + "\n\n" if history else "")
                + f"第 {k}/{max_turns} 步，只输出 JSON。")
        obj = _json(llm([{"role": "system", "content": RT_SYSTEM}, {"role": "user", "content": user}]))
        rec["turns"].append(obj)
        final = str(obj.get("final_label", "")).upper()
        if final in C.LABELS:
            rec.update(label=final, status="answered")
            return rec
        result, err = run_sql(conn, obj.get("sql"))
        if err:
            history.append(f"步骤{k}：{obj.get('sql')} -> 失败（{err}）")
            continue
        cols, rows = result
        try:
            conn.execute(f"CREATE TEMP TABLE step{k} AS {obj['sql'].strip().rstrip(';')}")
            saved = f"已存为 step{k}"
        except (sqlite3.Error, sqlite3.Warning) as exc:
            saved = f"未保存中间表：{str(exc)[:120]}"
        history.append(f"步骤{k}：{obj.get('sql')} -> 列{cols} 行{rows}（{saved}）")
    rec.update(label="INVALID", status="no_final_label")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=("text2sql", "reactable"), required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-name", required=True)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--extra", default="{}", help="provider-specific JSON merged into each request")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    key = C.api_key(args.api_key_name)
    claims = [json.loads(line) for line in open(args.claims, encoding="utf-8")]
    if args.limit:
        claims = claims[: args.limit]
    run_dir = Path(args.out) / f"{args.model.replace('/', '_')}__{args.agent}"
    meta = C.prepare_run(run_dir, vars(args) | {"t2s_system": T2S_SYSTEM, "rt_system": RT_SYSTEM}, args.claims)

    extra = json.loads(args.extra)

    agent = text2sql if args.agent == "text2sql" else (lambda c, f: reactable(c, f, args.max_turns))
    def one(claim):
        calls = []
        def llm(messages):
            raw, info = C.call_llm(messages, model=args.model, base_url=args.base_url, api_key=key,
                                   max_tokens=args.max_tokens, extra=extra)
            calls.append({"raw_output": raw, "api": info})
            return raw
        rec = agent(claim, llm)
        rec.update(input_sha256=C.input_hash(claim), api_calls=calls, run_fingerprint=meta["run_fingerprint"])
        return rec
    records = {}
    pred_path = run_dir / "predictions.jsonl"
    if pred_path.exists():                       # resume: keep finished claims
        for line in pred_path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("status"):
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
