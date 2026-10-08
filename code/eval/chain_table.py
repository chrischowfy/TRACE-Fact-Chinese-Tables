"""Chain-of-Table-style Chinese multi-table verification (adaptation, not reproduction).

Original method: https://github.com/google-research/chain-of-table
This independent implementation uses typed, deterministic table operations, a
single greedy chain, at most three operations and a final three-way decision.
It retains original tables, adds joins, and does not implement original prompts,
sampling/voting, or free-form column generation. No gold metadata enters prompts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import common as C
from dart_fact.tables import parse_number

SYSTEM = """你使用可执行表格操作核查中文声明。只能依据输入表格，不使用外部知识。
SUPPORTS=表格支持；REFUTES=表格反驳；NEI=给定表格信息不足。未知不是错误，false AND unknown 仍为 REFUTES。
每次只输出一个 JSON。可以输出 {"label":"SUPPORTS|REFUTES|NEI","rationale":"简短依据"} 结束，
或选择一个操作。table 使用 t0、t1 或步骤生成的 s1 等名字；行号和列号从 0 开始。
支持以下操作（用真实参数替换示例，不要包含 op 之外的代码）：
{"op":"select_rows","table":"t0","rows":[0,2]}
{"op":"select_columns","table":"t0","columns":[0,2]}
{"op":"sort","table":"t0","column":1,"numeric":true,"descending":true}
{"op":"group_count","table":"t0","column":1}
{"op":"aggregate","table":"t0","column":1,"method":"sum|mean|min|max|count"}
{"op":"derive","table":"t0","left":1,"right":2,"method":"subtract|add|ratio"}
{"op":"join","table":"t0","other":"t1","left":0,"right":0}
操作由环境执行，不要编造新单元格。原始表始终保留，操作生成新表。筛选造成的信息缺失不等于原表信息不足。
聚合只覆盖给定行，不能擅自当成整个现实总体。比分、日期、区间不能当作单一数值。
""" + C.ROUNDING_NOTE_ZH


def initial_tables(claim):
    return {f"t{i}": {"title": t.get("title", ""), "headers": list(t["headers"]),
                        "rows": copy.deepcopy(t["rows"])} for i, t in enumerate(claim["tables"])}


def integer(value, bound):
    if type(value) is not int or not 0 <= value < bound:
        raise ValueError("index_out_of_range")
    return value


def number(value):
    result = parse_number(value)
    if result is None:
        raise ValueError("not_a_scalar_number")
    return result


def execute(tables, action):
    if action.get("table") not in tables:
        raise ValueError("unknown_table")
    table = tables[action["table"]]
    headers, rows = list(table["headers"]), copy.deepcopy(table["rows"])
    width = len(headers)
    if any(len(r) != width for r in rows):
        raise ValueError("non_rectangular_table")
    op = action.get("op")
    if op == "select_rows":
        indices = action.get("rows")
        if not isinstance(indices, list) or len(indices) != len(set(indices)):
            raise ValueError("invalid_row_selection")
        rows = [rows[integer(i, len(rows))] for i in indices]
    elif op == "select_columns":
        columns = action.get("columns")
        if not isinstance(columns, list) or not columns or len(columns) != len(set(columns)):
            raise ValueError("invalid_column_selection")
        columns = [integer(i, width) for i in columns]
        headers = [headers[i] for i in columns]
        rows = [[r[i] for i in columns] for r in rows]
    elif op == "sort":
        c = integer(action.get("column"), width)
        if type(action.get("numeric", True)) is not bool or type(action.get("descending", False)) is not bool:
            raise ValueError("invalid_sort_flags")
        rows.sort(key=lambda r: number(r[c]) if action.get("numeric", True) else str(r[c]),
                  reverse=action.get("descending", False))
    elif op == "group_count":
        c = integer(action.get("column"), width)
        counts = {}
        for r in rows:
            counts[str(r[c])] = counts.get(str(r[c]), 0) + 1
        headers, rows = [headers[c], "count"], [[k, str(v)] for k, v in counts.items()]
    elif op == "aggregate":
        c = integer(action.get("column"), width)
        method = action.get("method")
        values = [number(r[c]) for r in rows] if method != "count" else rows
        if not values:
            raise ValueError("empty_aggregate")
        functions = {"sum": sum, "mean": lambda xs: sum(xs)/len(xs), "min": min, "max": max, "count": len}
        if method not in functions:
            raise ValueError("unknown_aggregate")
        headers, rows = [f"{method}({headers[c]})"], [[str(functions[method](values))]]
    elif op == "derive":
        a, b = integer(action.get("left"), width), integer(action.get("right"), width)
        functions = {"subtract": lambda x,y:x-y, "add": lambda x,y:x+y, "ratio": lambda x,y:x/y}
        method = action.get("method")
        if method not in functions:
            raise ValueError("unknown_derivation")
        headers.append(f"{method}({headers[a]},{headers[b]})")
        rows = [r + [str(functions[method](number(r[a]), number(r[b])))] for r in rows]
    elif op == "join":
        if action.get("other") not in tables:
            raise ValueError("unknown_join_table")
        other = tables[action["other"]]
        a = integer(action.get("left"), width)
        b = integer(action.get("right"), len(other["headers"]))
        joined = []
        for left in rows:
            for right in other["rows"]:
                if str(left[a]).strip() and str(left[a]).strip() == str(right[b]).strip():
                    joined.append(left + list(right))
                    if len(joined) > 2000:
                        raise ValueError("join_row_limit")
        headers = ["left."+h for h in headers] + ["right."+h for h in other["headers"]]
        rows = joined
    else:
        raise ValueError("unknown_operation")
    return {"title": "Derived " + str(op), "headers": headers, "rows": rows}


def build_prompt(claim, tables, history, final=False):
    payload = {"claim": claim["claim"], "tables": tables, "operation_history": history}
    return json.dumps(payload, ensure_ascii=False) + (
        '\n操作预算已用尽。只输出最终 JSON：{"label":"SUPPORTS|REFUTES|NEI","rationale":"依据"}。'
        if final else "\n选择下一操作或给出最终标签。")


def chain_table(claim, llm, max_operations=3):
    tables, history, calls = initial_tables(claim), [], []
    rec = {"id": claim["id"], "gold": claim["label"], "operations": history, "api_calls": calls}
    for step in range(max_operations + 1):
        raw, api = llm([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": build_prompt(claim, tables, history, step == max_operations)}])
        calls.append({"raw_output": raw, "api": api})
        if api.get("error"):
            return rec | {"label": "INVALID", "status": "api_failure"}
        try:
            obj = json.loads(raw or "")
            if not isinstance(obj, dict):
                raise ValueError("not_an_object")
        except (ValueError, TypeError):
            return rec | {"label": "INVALID", "status": "invalid_json"}
        if "label" in obj:
            label, error = C.parse_label(raw)
            return rec | {"label": label, "status": error or "answered", "rationale": obj.get("rationale", "")}
        if step == max_operations:
            break
        try:
            result = execute(tables, obj)
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError):
            # Execution failure is a protocol error, never an NEI prediction.
            history.append({"action": obj, "error": "invalid_table_operation"})
            return rec | {"label": "INVALID", "status": "invalid_table_operation"}
        name = f"s{step+1}"
        tables[name] = result
        history.append({"action": obj, "output": name, "result": result})
    return rec | {"label": "INVALID", "status": "no_final_label"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--claims", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.workers < 1 or a.limit < 0:
        p.error("invalid workers/limit")
    key = C.api_key("DEEPSEEK_API_KEY")
    all_claims = [json.loads(s) for s in Path(a.claims).read_text().splitlines()]
    claims = all_claims[:a.limit] if a.limit else all_claims
    run = Path(a.out) / (a.model + "__chain-table")
    meta = C.prepare_run(run, vars(a) | {"setting": "chain-table", "system_prompt": SYSTEM,
        "chain_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_operations": 3, "max_tokens": 2048, "temperature": 0,
        "base_url": "https://api.deepseek.com", "extra": '{"thinking":{"type":"disabled"}}'}, a.claims)
    path = run / "predictions.jsonl"
    records = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row["id"] in records:
            raise ValueError("duplicate resumed prediction")
        records[row["id"]] = row
    lookup = {c["id"]:c for c in claims}
    for ident, row in records.items():
        if ident not in lookup or row.get("input_sha256") != C.input_hash(lookup[ident]) or row.get("run_fingerprint") != meta["run_fingerprint"]:
            raise ValueError("resume identity mismatch")
    def one(claim):
        def llm(messages):
            return C.call_llm(messages, model=a.model, base_url="https://api.deepseek.com", api_key=key,
                max_tokens=2048, temperature=0, extra={"thinking":{"type":"disabled"}})
        return chain_table(claim, llm) | {"input_sha256": C.input_hash(claim), "run_fingerprint": meta["run_fingerprint"]}
    pending = [c for c in claims if c["id"] not in records]
    print(f"Chain-table: {len(pending)} pending", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as executor, path.open("a", encoding="utf-8") as output:
        for index, future in enumerate(as_completed([executor.submit(one,c) for c in pending]), 1):
            row = future.result()
            records[row["id"]] = row
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if index % 100 == 0:
                print(f"{index}/{len(pending)}", flush=True)
    report = C.score(claims, {k:v["label"] for k,v in records.items()})
    report["api_failures"] = sum(bool(call["api"].get("error")) for r in records.values() for call in r["api_calls"])
    report["operations_executed"] = sum(sum("result" in op for op in r["operations"]) for r in records.values())
    (run / "score.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:report[k] for k in ("n","accuracy","macro_f1","invalid","api_failures","operations_executed")}), flush=True)


if __name__ == "__main__":
    main()
