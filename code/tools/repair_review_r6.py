"""Conservative, reproducible r6 repair; no predictions or LLM calls are read.

The visible claim, not an inherited label or a lost generator intention, is the
semantic target. Ambiguous cases are quarantined, not silently rewritten.
Public pilot programs are reconstructions, never recovered historical programs.
Run without --out to inspect; --out must be new/empty and freezes a new release.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dart_fact.analysis import statistics, surface_baselines
from dart_fact.executor import InvalidProgram
from dart_fact.review_repair import execute
from dart_fact.tables import entity_key, parse_number, profile_table
from nei_semantic_gate import quarantine_reason
from repair_review_release import audit, digest, dump, file_hash, jsonl, repair_record, template

POLICY = "r6_semantic_screen_v1"
ALIASES = {
    "胜场数": ["胜场", "获胜", "赢的场次", "胜了", "赢了", "胜场次数"],
    "负场数": ["负场", "输掉", "输的场次", "负了", "输了"],
    "平局场数": ["平局", "打平"],
    "比赛场次": ["比赛场数", "参赛场次", "场比赛"],
    "进球数": ["进球", "总进球"], "失球数": ["失球"],
    "金牌数": ["金牌"], "银牌数": ["银牌"], "铜牌数": ["铜牌"],
    "助攻数": ["助攻"], "红牌数": ["红牌"], "黄牌数": ["黄牌"],
    "GDP": ["地区生产总值", "国内生产总值"],
    "总计": ["总数", "奖牌总数", "奖牌总量"],
    "总数": ["总计", "奖牌总数", "奖牌总量"],
}
LEGACY_METRICS = {
    "金牌": ["金牌"], "银牌": ["银牌"], "铜牌": ["铜牌"],
    "总数": ["总数", "奖牌总数"], "总计": ["总计", "奖牌总数"],
    "得": ["进球", "总进球"], "进球": ["进球", "总进球"],
    "胜": ["胜场", "获胜场次"], "负": ["负场"],
    "赛": ["比赛场数", "比赛场次", "参赛场次"],
    "胜率": ["胜率"], "夺冠次数": ["夺冠次数"],
    "冠军": ["冠军次数", "冠军数"], "得票": ["得票"],
}
BAD_SURFACE = re.compile(r"某年|某分|一项|某项|观影人数日本|\*|年份最高|年份最低|年分最低|首次登场高于")


def norm(text):
    return re.sub(r"[\s，。；：、,.!?！？;:（）()]+", "", unicodedata.normalize("NFKC", str(text)))


def metric_visible(metric, claim):
    metric = re.sub(r"[（(].*?[）)]", "", metric).strip().rstrip("*")
    names = [metric] + ALIASES.get(metric, [])
    if metric.endswith("数") and len(metric) > 2:
        names.append(metric[:-1])
    return not metric or any(norm(x).lower() in norm(claim).lower() for x in names if x)


def source_map(legacy, path, fetch=False):
    if path.exists():
        return json.loads(path.read_text())
    if not fetch:
        raise ValueError("source map absent; use --resolve-sources to resolve original revision IDs")
    import requests
    revisions = sorted({int(t["revision_id"]) for r in legacy for t in r["tables"]})
    result = {}
    for start in range(0, len(revisions), 40):
        response = requests.get("https://zh.wikipedia.org/w/api.php", params={
            "action": "query", "revids": "|".join(map(str, revisions[start:start+40])),
            "prop": "revisions", "rvprop": "ids", "format": "json"},
            headers={"User-Agent": "TRACE-Fact-revision-audit/1.0"}, timeout=45)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ValueError(payload["error"])
        for page in payload.get("query", {}).get("pages", {}).values():
            for revision in page.get("revisions", []):
                result[str(revision["revid"])] = {
                    "page_id": page["pageid"], "resolved_title": page["title"],
                    "revision_id": revision["revid"], "method": "MediaWiki query by original revid",
                    "endpoint": "https://zh.wikipedia.org/w/api.php"}
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(path, result)
    return result


def convert_table(table, pages):
    revision = str(table["revision_id"])
    if revision not in pages:
        raise ValueError("unresolved_original_revision")
    page = pages[revision]
    identity = {k: table[k] for k in ("title", "headers", "rows", "source_url", "revision_id")}
    return {"table_id": "pilot-" + digest(identity)[:24],
            "title": table["title"], "headers": table["headers"], "rows": table["rows"],
            "source": {"page_title": unquote(urlparse(table["source_url"]).path.split("/wiki/")[-1]),
                       "page_id": page["page_id"], "resolved_title": page["resolved_title"],
                       "revision_id": int(revision), "url": table["source_url"],
                       "revision_url": "https://zh.wikipedia.org/w/index.php?oldid=" + revision,
                       "license": "CC BY-SA 4.0", "legacy_table_id": table["table_id"],
                       "snapshot_status": "public_pilot_embedded_table; revision_identity_resolved"}}


def comparison(claim):
    classes = {"eq": r"相同|一样多|一样|持平|等于|相等", "gt": r"多于|高于|超过|大于", "lt": r"少于|低于|小于"}
    found = [kind for kind, pattern in classes.items() if re.search(pattern, claim)]
    return found[0] if len(found) == 1 else None


def reconstruct_legacy(original, pages):
    """Opt-in deterministic adapter: scalar comparisons and unique extremes only.

    No absent-binding pilot NEI is admitted through this adapter: an old NEI can
    be admitted only if the *visible* tables give a unique decidable program.
    Ambiguous metrics/subjects/scopes, ties and unsupported families stay out.
    """
    claim = original["claim"]
    if BAD_SURFACE.search(claim):
        return None, {"reason": "legacy_ambiguous_or_ill_typed_surface"}
    if re.search(r"[‡†]", claim):
        return None, {"reason": "legacy_footnote_marker_in_entity_surface"}
    sk = original["reasoning_skeleton"]
    extreme = {"SINGLE_TABLE_ARGMAX_ENTITY": "max", "SINGLE_TABLE_ARGMIN_ENTITY": "min"}.get(sk)
    if not extreme and sk not in ("ENTITY_PAIR_METRIC_COMPARE", "SINGLE_TABLE_ENTITY_METRIC_COMPARE"):
        return None, {"reason": "legacy_family_not_uniquely_reconstructable"}
    try:
        tables = [convert_table(t, pages) for t in original["tables"]]
    except ValueError as exc:
        return None, {"reason": str(exc)}
    candidates = []
    for table in tables:
        # General championship pages contain discipline-specific medal tables;
        # generic titles like 'page table 14' have lost that section scope. Do
        # not infer an all-event claim from a men's-doubles table, or a global
        # minimum from an unverified top-ten list.
        if "金牌" in table["headers"] and not re.search(r"(?:19|20)\d{2}", original["topic"]):
            continue
        if extreme == "min":
            continue
        if table["title"] in {"两者", "表格", ""}:
            continue
        prof = profile_table(table)
        if not prof.key or re.search(r"年份|年度|年分|次序|项目|奖项|领域|物理|化学|学科", prof.key.header):
            continue
        # Subject nouns in old templates sometimes call a university a person.
        if "人物" in claim and not re.search(r"姓名|人物|得主|获奖者", prof.key.header):
            continue
        if "城市" in claim and not re.search(r"城市|城巿", prof.key.header):
            continue
        claim_years = set(re.findall(r"(?:19|20)\d{2}", claim))
        table_years = set(re.findall(r"(?:19|20)\d{2}", table["title"]))
        if claim_years and claim_years != table_years:
            continue
        # Only explicit whole-season/year comparisons; no unsupported sub-period.
        if re.search(r"轮|日起|赛季.*起|主场|客场", claim):
            continue
        keys = sorted(set(prof.key_values()), key=len, reverse=True)
        hits = []
        for key in keys:
            if len(key) < 2 or key not in norm(claim) or any(key in earlier for earlier in hits):
                continue
            hits.append(key)
        hits.sort(key=lambda k: norm(claim).index(k))
        if len(hits) != (1 if extreme else 2):
            continue
        for col in prof.columns:
            aliases = LEGACY_METRICS.get(col.header, [])
            if not any(a in claim for a in aliases):
                continue
            if col.header in ("总计", "总数") and not any("金牌" in h for h in table["headers"]):
                continue
            tid = table["table_id"]
            if extreme:
                values = [parse_number(prof.cell(i, col)) for i in range(len(prof.rows))]
                if not values or any(v is None for v in values):
                    continue
                best = (max if extreme == "max" else min)(values)
                if values.count(best) != 1:
                    continue
                ops = [{"op": "ARGEXT", "table": tid, "col": col.header, "mode": extreme, "out": "top"},
                       {"op": "LOOKUP", "table": tid, "key": hits[0], "col": col.header, "out": "claimed_value"},
                       {"op": "COMPARE", "left": "$top", "cmp": "eq", "right": hits[0], "out": "result"}]
                slots = {"mode": extreme, "claimed": hits[0]}
            else:
                cmp = comparison(claim)
                if not cmp:
                    continue
                ops = [{"op": "LOOKUP", "table": tid, "key": hits[0], "col": col.header, "out": "a"},
                       {"op": "LOOKUP", "table": tid, "key": hits[1], "col": col.header, "out": "b"},
                       {"op": "COMPARE", "left": "$a", "cmp": cmp, "right": "$b", "out": "result"}]
                slots = {"e1": hits[0], "e2": hits[1], "cmp": cmp}
            try:
                run = execute(ops, {tid: prof})
            except (InvalidProgram, KeyError, ValueError):
                continue
            if run["label"] == "NEI":
                continue
            candidates.append((table, ops, slots, col.header, run))
    signatures = {digest([t["source"]["revision_id"], t["headers"], t["rows"], ops]): (t, ops, slots, col, run)
                  for t, ops, slots, col, run in candidates}
    if len(signatures) != 1:
        return None, {"reason": "legacy_no_unique_safe_program", "candidate_programs": len(signatures)}
    table, ops, slots, col, run = next(iter(signatures.values()))
    topic = original["topic"]
    domain = "football" if "足球" in topic else "basketball" if "篮球" in topic else "medals" if "金牌" in table["headers"] else "sports_other"
    record = {"id": original["id"], "claim": claim, "label": run["label"],
        "surface": {"kind": "legacy_preserved", "template_claim": claim},
        "evidence_package_id": "r6pkg-" + digest([table])[:24], "tables": [table],
        "evidence_cells": run["cells"], "context_cells": [],
        "program": {"category": "Single-table comparison & ranking",
                    "skeleton_id": "ST_SUPERLATIVE" if extreme else "ST_COMPARE", "operators": ops,
                    "table_ids": [table["table_id"]], "slots": {**slots, "metric_name": next(a for a in LEGACY_METRICS[col] if a in claim),
                    "metric_col": col, "scope": topic, "noun": profile_table(table).key.header}},
        "table_topology": "single", "package_kind": "single",
        "quality_flags": {"program_provenance": "reconstructed_from_public_claim_and_embedded_tables",
                          "original_surface_generation": "not_established; original public wording preserved",
                          "perturbation_axis": "not_established", "label_by_execution": True},
        "domain": domain, "series": topic, "topic": topic, "license": original["license"]}
    return record, {"reason": "legacy_reconstructed", "old_label": original["label"], "new_label": run["label"],
                    "old_num_tables": len(original["tables"]), "new_num_tables": 1,
                    "surface_changed": False, "program_recovered": False}


def shortcut_witness(row):
    """Prove a strict same-group competitor without a rank/category-name bridge.

    This does not rename tables in a fixed program (which could mix years or
    entity types). It uses raw keys, identical group memberships and one metric.
    """
    p = row["program"]
    if row["label"] != "REFUTES" or p["skeleton_id"] not in {"RJ_RANK_TOP", "CAT_ARGEXT"}:
        return None
    tabs = {t["table_id"]: t for t in row["tables"]}
    join = next(o for o in p["operators"] if o["op"] == "JOIN")
    vals = next(o for o in p["operators"] if o["op"] == "VALUES")
    ent, met = tabs[join["table"]], tabs[vals["table"]]
    ep, mp = profile_table(ent), profile_table(met)
    if not ep.key or not mp.key or vals["col"] not in met["headers"]:
        return None
    fk, mc = ent["headers"].index(join["fk_col"]), met["headers"].index(vals["col"])
    groups, values = collections.defaultdict(set), collections.defaultdict(set)
    for cells in ent["rows"]:
        groups[entity_key(cells[ep.key.index])].add(entity_key(cells[fk]))
    for cells in met["rows"]:
        value = parse_number(cells[mc])
        if value is not None:
            values[entity_key(cells[mp.key.index])].add(value)
    claimed = entity_key(p["slots"]["claimed"])
    if len(groups[claimed]) != 1 or len(values[claimed]) != 1:
        return None
    cv = next(iter(values[claimed]))
    mode = p["slots"].get("mode", "max")
    for other in sorted(groups):
        if other == claimed or groups[other] != groups[claimed] or len(values[other]) != 1:
            continue
        ov = next(iter(values[other]))
        if (ov > cv if mode == "max" else ov < cv):
            return {"reason": "alternative_same_group_refutation", "claimed": claimed,
                    "claimed_value": cv, "competitor": other, "competitor_value": ov,
                    "group_key": next(iter(groups[claimed])),
                    "sufficient_table_ids": sorted({ent["table_id"], met["table_id"]})}
    return None


def semantic_issues(row):
    slots = row["program"]["slots"]
    noun, metric = slots.get("noun", ""), slots.get("metric_name", "")
    issues = []
    if BAD_SURFACE.search(row["claim"]):
        issues.append({"reason": "ill_typed_or_ambiguous_surface"})
    if not metric_visible(metric, row["claim"]):
        issues.append({"reason": "metric_surface_requires_adjudication", "metric": metric})
    if (noun == "城市" and metric == "长度" or
        noun == "球员" and metric in {"控球率", "主场观众人数"} or
        noun == "公司" and metric == "跑道数量" or
        noun == "州" and metric in {"文创收入", "志愿者人数"}):
        issues.append({"reason": "metric_subject_type_ambiguous", "noun": noun, "metric": metric})
    if noun == "城市" and any("希腊岛屿" in t["source"]["page_title"] for t in row["tables"]) and metric:
        issues.append({"reason": "island_metric_attributed_to_capital_city"})
    if row["program"]["skeleton_id"] == "RJ_RANK_TOP" and not re.search(r"(?:中|里|内|当中)", row["claim"].split("赛季")[-1].replace("中超", "")):
        issues.append({"reason": "within_team_extreme_scope_not_explicit"})
    if row["label"] == "NEI" and row["quality_flags"].get("perturbation_axis") == "join_key":
        issues.append({"reason": "missing_membership_closed_world_ambiguity"})
    old = quarantine_reason(row)
    if old:
        issues.append({"reason": old["rule"], **old})
    witness = shortcut_witness(row)
    if witness:
        issues.append(witness)
    # Absence of a strict witness does not certify these structurally risky
    # negative families. Keep unresolved cases out pending semantic adjudication.
    elif row["label"] == "REFUTES" and row["program"]["skeleton_id"] in {"RJ_RANK_TOP", "CAT_ARGEXT"}:
        issues.append({"reason": "negative_bridge_necessity_unresolved"})
    # Some country rows are mistaken for aggregate rows by the legacy profiler.
    scan = {"ARGEXT", "RANK", "COUNT_ABOVE", "SUM", "FILTER"}
    for op in row["program"]["operators"]:
        if op["op"] not in scan or "table" not in op:
            continue
        table = next(t for t in row["tables"] if t["table_id"] == op["table"])
        prof = profile_table(table)
        if not prof.key:
            continue
        if noun in {"国家", "国家或地区"} and any(i not in prof.row_map and entity_key(c[prof.key.index]) == "中国大陆" for i,c in enumerate(table["rows"])):
            issues.append({"reason": "country_row_filtered_as_aggregate", "table_id": table["table_id"]})
    return issues


def correct_visible_gdp(original):
    row = copy.deepcopy(original)
    changed = None
    if row["id"] == "review-repair-r5-01419":
        if row["claim"] != "浙江各地级市中，台州市的地区生产总值最低。":
            raise ValueError("reviewed correction target changed")
        if "GDP (本币)" not in row["tables"][0]["headers"]:
            raise ValueError("GDP correction source changed")
        for op in row["program"]["operators"]:
            if op.get("col") == "长度":
                op["col"] = "GDP (本币)"
        row["program"]["slots"].update(metric_col="GDP (本币)", metric_name="GDP", unit="百万元")
        row["quality_flags"].update(perturbation_axis="not_established", nei_absence_verified=False)
        changed = {"reason": "visible_claim_metric_corrected", "old_metric": "长度", "new_metric": "GDP (本币)",
                   "basis": "台州242645 > 舟山64432; all three GDP representations give the same counterexample",
                   "surface_changed": False}
    # Synchronize stale slots with numeric operators, without changing operators.
    numeric_cols = {op["col"] for op in row["program"]["operators"] if op["op"] in {"LOOKUP", "VALUES", "ARGEXT", "RANK", "COUNT_ABOVE"} and "col" in op}
    if len(numeric_cols) == 1 and "metric_col" in row["program"]["slots"]:
        row["program"]["slots"]["metric_col"] = next(iter(numeric_cols))
    return row, changed


def build(legacy, r5, pages):
    rows, changes, quarantine = [], [], []
    seen_claims, seen_semantics = {}, {}
    for origin, originals in (("public_pilot", legacy), ("r5", r5)):
        for original in originals:
            log = {"origin": origin, "original_id": original["id"], "original_sha256": digest(original)}
            if origin == "public_pilot":
                row, detail = reconstruct_legacy(original, pages)
                log.update(detail)
            else:
                row, correction = correct_visible_gdp(original)
                log.update(reason="retained", correction=correction)
            if row is None:
                changes.append(log)
                quarantine.append({**log, "original": original})
                continue
            repaired, execution_change = repair_record(row)
            if repaired is None:
                log.update(execution_change)
                changes.append(log)
                quarantine.append({**log, "original": original})
                continue
            issues = semantic_issues(repaired)
            if issues:
                log.update(reason="semantic_quarantine", issues=issues)
                changes.append(log)
                quarantine.append({**log, "original": original})
                continue
            # Canonicalize table references by immutable source snapshot content.
            mapping = {t["table_id"]: digest({k:t[k] for k in ("headers", "rows")} | {
                "page_id":t["source"]["page_id"], "revision_id":t["source"]["revision_id"]}) for t in repaired["tables"]}
            ops = [{k: mapping.get(v,v) if k == "table" else v for k,v in op.items()} for op in repaired["program"]["operators"]]
            semantic = digest(ops)
            textkey = norm(repaired["claim"])
            duplicate = seen_claims.get(textkey) or seen_semantics.get(semantic)
            if duplicate:
                log.update(reason="duplicate", retained_id=duplicate)
                changes.append(log)
                continue
            repaired["id"] = f"review-repair-r6-{len(rows)+1:05d}"
            repaired["quality_flags"].update(semantic_screen=POLICY,
                semantic_screen_scope="targeted deterministic checks; not independent human validation")
            rows.append(repaired)
            seen_claims[textkey] = seen_semantics[semantic] = repaired["id"]
            log.update(repaired_id=repaired["id"], old_label=original["label"], new_label=repaired["label"],
                       final_num_tables=len(repaired["tables"]), execution_change=execution_change)
            changes.append(log)
    return rows, changes, quarantine


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy", type=Path, default=ROOT.parent / "trace-fact-A298/data/small/claims.jsonl")
    p.add_argument("--r5", type=Path, default=ROOT / "data/review_repair_r5/claims.jsonl")
    p.add_argument("--source-map", type=Path, default=ROOT / "runs/review_repair_r6_source_pages.json")
    p.add_argument("--resolve-sources", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.out and a.out.exists() and any(a.out.iterdir()):
        raise SystemExit("output must be new/empty; never overwrite a frozen release")
    legacy = [json.loads(x) for x in a.legacy.read_text().splitlines() if x.strip()]
    r5 = [json.loads(x) for x in a.r5.read_text().splitlines() if x.strip()]
    pages = source_map(legacy, a.source_map, a.resolve_sources)
    rows, changes, quarantine = build(legacy, r5, pages)
    validation = audit(rows, json.loads((ROOT / "schemas/claim.schema.json").read_text()))
    summary = {"records": len(rows), "retained_by_origin": dict(collections.Counter(c["origin"] for c in changes if "repaired_id" in c)),
               "labels": dict(collections.Counter(r["label"] for r in rows)),
               "label_corrections": sum(c.get("old_label") != c.get("new_label") for c in changes if "repaired_id" in c),
               "quarantine_records": len(quarantine), "dispositions": dict(collections.Counter(c["reason"] for c in changes)),
               "issue_counts": dict(collections.Counter(i["reason"] for c in changes for i in c.get("issues", []))),
               "validation": validation, "human_validation_status": "not_completed",
               "per_instance_model_correctness_used_for_selection": False,
               "model_informed_rule_development": True,
               "source_registry_recovered": False}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not validation["passed"]:
        raise SystemExit("structural audit failed; release not written")
    if not a.out:
        return
    a.out.mkdir(parents=True, exist_ok=True)
    jsonl(a.out / "claims.jsonl", rows)
    jsonl(a.out / "changes.jsonl", changes)
    jsonl(a.out / "semantic_quarantine.jsonl", quarantine)
    dump(a.out / "validation.json", validation)
    dump(a.out / "build_summary.json", summary)
    dump(a.out / "source_revision_resolution.json", pages)
    # Include immutable input bytes in the self-contained release, not secrets/cache.
    (a.out / "input_public_pilot.jsonl").write_bytes(a.legacy.read_bytes())
    (a.out / "input_r5.jsonl").write_bytes(a.r5.read_bytes())
    st = statistics(rows)
    dump(a.out / "stats.json", {"statistics": st, "surface_baselines": surface_baselines(rows)})
    tables = {t["table_id"]:t for r in rows for t in r["tables"]}
    headers = collections.Counter(tuple(t["headers"]) for t in tables.values())
    dump(a.out / "table_structure.json", {"tables":len(tables), "distinct_exact_header_schemas":len(headers),
        "top_repeated_schemas":[{"headers":list(h), "table_count":n} for h,n in headers.most_common(10)],
        "claims_with_derived_views":sum(any(t["source"].get("derived_from") for t in r["tables"]) for r in rows)})
    unique_pages = {(t["source"]["page_id"], t["source"]["revision_id"]):t["source"] for t in tables.values()}
    jsonl(a.out / "source_pages.jsonl", [{k:s[k] for k in ("page_id", "revision_id", "page_title", "url", "revision_url", "license")} for _,s in sorted(unique_pages.items())])
    packages = {r["evidence_package_id"]:{"package_id":r["evidence_package_id"], "kind":r["package_kind"], "table_ids":[t["table_id"] for t in r["tables"]]} for r in rows}
    jsonl(a.out / "packages.jsonl", [packages[k] for k in sorted(packages)])
    registry = []
    for sid in sorted({r["program"]["skeleton_id"] for r in rows}):
        rr = [r for r in rows if r["program"]["skeleton_id"] == sid]
        variants = {digest(template(r["program"]["operators"])):template(r["program"]["operators"]) for r in rr}
        registry.append({"skeleton_id":sid, "category":rr[0]["program"]["category"],
                         "provenance_status":"current_executable_inventory_not_historical_source_derivations",
                         "source_support":[], "historical_source_record_ids_available":False,
                         "program_templates":list(variants.values()), "instances":len(rr),
                         "labels":dict(collections.Counter(r["label"] for r in rr))})
    jsonl(a.out / "skeleton_registry.jsonl", registry)
    dump(a.out / "review_comparison.json", {"reviewed":json.loads((ROOT / "tools/repro_spec.json").read_text())["overall"], "repaired":st,
        "note":"Public pilot recovery plus reconstructed candidates; not recovery of the full reviewed artifact."})
    code_names = ["tools/repair_review_r6.py", "tools/repair_review_release.py", "tools/nei_semantic_gate.py"]
    code_names += [str(f.relative_to(ROOT)) for f in sorted((ROOT / "src/dart_fact").glob("*.py"))]
    code = {name:file_hash(ROOT / name) for name in code_names}
    dump(a.out / "freeze_manifest.json", {"release":"review-repair-r6", "date":"2026-09-20",
        "claims_sha256":file_hash(a.out / "claims.jsonl"), "code_sha256":code,
        "inputs_sha256":{"input_public_pilot.jsonl":file_hash(a.legacy), "input_r5.jsonl":file_hash(a.r5)},
        "artifacts_sha256":{f.name:file_hash(f) for f in sorted(a.out.iterdir()) if f.is_file()},
        "evaluation_status":"not_run", "human_validation_status":"not_completed",
        "historical_source_derivations":"unavailable", "reporting_code_policy":"separately hashed in reporting manifest; not construction inputs"})


if __name__ == "__main__":
    main()
