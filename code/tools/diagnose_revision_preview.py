"""Read-only source/structure and saved Chain-of-Table output diagnostics."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import chain_table as chain

RELEASE = ROOT / "data/review_repair_r6"
OUT = ROOT.parent / "ccl2026/generated_main4"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_output(raw):
    """Diagnostic only: unwrap explicit, unambiguous formatting, never guess values."""
    raw = raw.strip()
    changes = []
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.S)
    if fence:
        raw = fence[1].strip()
        changes.append("markdown_fence")
    obj = json.loads(raw)
    if isinstance(obj, dict) and set(obj) == {"type", "content"} and obj["type"] == "json_object":
        obj = obj["content"]
        if isinstance(obj, str):
            obj = json.loads(obj)
        changes.append("json_object_envelope")
    if not isinstance(obj, dict):
        raise ValueError("not_object")
    return obj, changes


def source_report(claims):
    tables = {}
    domains = defaultdict(set)
    for c in claims:
        for t in c["tables"]:
            if t["table_id"] in tables:
                assert all(tables[t["table_id"]][k] == t[k] for k in ("headers", "rows", "source")), "Conflicting table identity"
            tables[t["table_id"]] = t
            domains[c["domain"]].add(t["table_id"])
    schemas = Counter(tuple(t["headers"]) for t in tables.values())
    derived = [t for t in tables.values() if t["source"].get("derived_from")]
    raw_ids = {t["source"].get("derived_from", t["table_id"]) for t in tables.values()}
    hosts = Counter(urlparse(t["source"]["url"]).hostname for t in tables.values())
    counts = Counter(c["domain"] for c in claims)
    dimensions = {}
    for name, values in (("rows", [len(t["rows"]) for t in tables.values()]),
                         ("columns", [len(t["headers"]) for t in tables.values()])):
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        dimensions[name] = dict(min=min(values), q1=quartiles[0], median=statistics.median(values),
                                q3=quartiles[2], max=max(values))
    normalized = Counter(tuple(re.sub(r"\s+", "", h).replace("／", "/") for h in t["headers"])
                         for t in tables.values())
    return {"claims_sha256": digest(RELEASE / "claims.jsonl"), "claims": len(claims),
        "source_hosts_package_tables": dict(hosts), "raw_source_tables": len(raw_ids),
        "package_tables": len(tables), "derived_package_tables": len(derived),
        "claims_with_derived_views": sum(any(t["source"].get("derived_from") for t in c["tables"]) for c in claims),
        "domain_claims": [{"domain": d, "claims": n, "percent": round(100*n/len(claims), 1),
                           "distinct_package_tables": len(domains[d])} for d,n in counts.most_common()],
        "package_kind_claims": dict(Counter(c["package_kind"] for c in claims)),
        "dimensions_unique_package_tables": dimensions,
        "exact_header_schemas": len(schemas),
        "tables_in_repeated_exact_schemas": sum(n for n in schemas.values() if n > 1),
        "normalized_header_schemas": len(normalized),
        "normalization": "Remove whitespace and normalize full-width slash only; not semantic schema matching.",
        "top_exact_schemas": [{"headers": list(h), "tables": n} for h,n in schemas.most_common(10)],
        "counting": "Unique table IDs for structure; claims for domain and packaging proportions. Table-domain counts may overlap. Derived views are not independent sources.",
        "selection": "Curated retained tables that support available family-specific bindings and pass executable/semantic screens, not probability sampling of Wikipedia or Chinese tables."}


def chain_report(claims):
    run = ROOT / "runs/review_repair_r6_eval/label_first/full/deepseek-flash__chain-table"
    path = run / "predictions.jsonl"
    rows = list(map(json.loads, path.read_text().splitlines()))
    lookup = {c["id"]: c for c in claims}
    counts, details = Counter(), []
    for r in rows:
        if r["label"] != "INVALID":
            continue
        raw = r["api_calls"][-1]["raw_output"]
        entry = {"id": r["id"], "original_status": r["status"]}
        try:
            obj, changes = normalize_output(raw)
            entry["format_changes"] = changes
            if "label" in obj and "op" not in obj and obj["label"] in ("SUPPORTS", "REFUTES", "NEI"):
                category = "formatted_final_label" if changes else "unhandled_final_label"
                entry["parsed_label"] = obj["label"]
            elif "op" in obj and "label" not in obj:
                tables = chain.initial_tables(lookup[r["id"]])
                for step in r["operations"]:
                    if "result" in step:
                        tables[step["output"]] = step["result"]
                try:
                    chain.execute(tables, obj)
                    category = "formatted_executable_operation" if changes else "unhandled_executable_operation"
                except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError, OverflowError) as exc:
                    category = "operation_execution_error"
                    entry["execution_error"] = str(exc)
            else:
                category = "other_output_schema"
        except (ValueError, TypeError):
            category = "unparseable_json"
        entry["diagnosis"] = category
        counts[category] += 1
        details.append(entry)
    assert len(rows) == 1484 and len(details) == 474
    return {"predictions_sha256": digest(path), "n": len(rows), "invalid": len(details),
        "categories": dict(counts), "records": details,
        "scope": "Offline structural diagnosis of stored outputs, not a rerun or a revised benchmark score. No labels inferred from rationales. Executable operations would require further model calls; original outputs and scores remain unchanged."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--notes", type=Path, default=ROOT.parent / "ccl2026/notes/SOURCE_STRUCTURE_AND_BASELINE.md")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.notes.parent.mkdir(parents=True, exist_ok=True)
    claims = list(map(json.loads, (RELEASE / "claims.jsonl").read_text().splitlines()))
    assert digest(RELEASE / "claims.jsonl") == json.loads((RELEASE / "freeze_manifest.json").read_text())["claims_sha256"]
    source, failures = source_report(claims), chain_report(claims)
    for name, report in (("source_structure_diagnostics", source), ("chain_output_diagnostics", failures)):
        (args.out / (name + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    notes = args.notes
    lines = ["# 来源结构与基线核查", "", "冻结数据与原始预测均未修改。", "", "## 来源与结构", "",
        "所有包内表均源于中文维基百科。领域不等于来源平台；衍生表也不是新增独立事实。", "",
        "|领域|声明数|声明占比|不同包内表数（跨领域可重叠）|", "|---|---:|---:|---:|"]
    lines += [f'|{d["domain"]}|{d["claims"]}|{d["percent"]}%|{d["distinct_package_tables"]}|' for d in source["domain_claims"]]
    lines += ["", f'491 张原始来源表对应 {source["package_tables"]} 张包内表，其中 {source["derived_package_tables"]} 张是衍生视图；119/1484 条声明使用衍生视图。8.0% 是声明比例，不是衍生表比例。',
        "", "行列数（按不同包内表计）：`" + json.dumps(source["dimensions_unique_package_tables"], ensure_ascii=False) + "`。",
        "", f'精确有序表头 {source["exact_header_schemas"]} 种；{source["tables_in_repeated_exact_schemas"]} 张表属于重复表头组。只去空白和统一斜杠后为 {source["normalized_header_schemas"]} 种；此规则不等于语义模式去重。',
        "", "## Chain-of-Table 输出失败", "", "474 条 INVALID 的离线结构诊断：", ""]
    lines += [f"- {k}: {v}" for k,v in failures["categories"].items()]
    lines += ["", "只解开完整代码围栏和显式 json_object/content 封装，不从解释猜标签、不改参数或操作。可执行操作还需续跑才能得到答案；这里没有产生新分数。原始 57.3/69.8 分数保留，其解释必须考虑适配器的格式兼容性。", ""]
    notes.write_text("\n".join(lines))
    print(json.dumps({"source": source, "chain_categories": failures["categories"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
