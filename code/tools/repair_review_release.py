"""Repair a newly rebuilt candidate and export an auditable, separate release.

No model predictions enter this process. Reject invalid/shortcut examples;
recompute labels and dependency evidence; keep a record-by-record change log.
The reconstructed registry is explicitly NOT the lost source-derived registry.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dart_fact.analysis import statistics, surface_baselines
from dart_fact.executor import InvalidProgram
from dart_fact.review_repair import POLICY, execute
from dart_fact.tables import parse_number, profile_table
try:
    from jsonschema import Draft202012Validator as SchemaValidator
except ImportError:
    # This schema uses only draft-7-compatible constraints and JSON Pointer refs.
    from jsonschema import Draft7Validator as SchemaValidator


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def template(operators):
    """Delexicalize the reconstructed program, preserving variable dependencies."""
    tables, slots = {}, {}
    def convert(key, value):
        if key in ("op", "out", "cmp", "mode", "order") or (isinstance(value, str) and value.startswith("$")):
            return value
        if isinstance(value, list):
            return [convert(key, x) for x in value]
        if key == "table":
            return tables.setdefault(value, f"T{len(tables) + 1}")
        token = (key, json.dumps(value, sort_keys=True, ensure_ascii=False))
        return slots.setdefault(token, f"<{key}_{len(slots) + 1}>")
    return [{k: convert(k, v) for k, v in op.items()} for op in operators]


def repair_record(original):
    row = copy.deepcopy(original)
    operators = row["program"]["operators"]
    bound = {op["table"] for op in operators if "table" in op}
    unused = [t["table_id"] for t in row["tables"] if t["table_id"] not in bound]
    if unused:
        # Remove unbound distractors, not tables read by a redundant branch.
        row["tables"] = [t for t in row["tables"] if t["table_id"] in bound]
        row["evidence_package_id"] = "repair-" + digest(row["tables"])[:24]
    profiles = {t["table_id"]: profile_table(t) for t in row["tables"]}
    try:
        run = execute(operators, profiles)
        ablations = {}
        if run["label"] != "NEI":
            for tid in profiles:
                ablations[tid] = execute(operators, {k: p for k, p in profiles.items() if k != tid})["label"]
            if any(label != "NEI" for label in ablations.values()):
                return None, {"reason": "table_not_necessary", "ablations": ablations}
    except (InvalidProgram, KeyError, TypeError, ValueError) as exc:
        return None, {"reason": "invalid_program", "detail": str(exc)}
    row["label"] = run["label"]
    row["evidence_cells"] = run["cells"] if run["label"] != "NEI" else []
    row["context_cells"] = run["cells"] if run["label"] == "NEI" else []
    declared = sorted({op["table"] for op in operators if "table" in op})
    row["program"]["table_ids"] = declared
    row["table_topology"] = "single" if len(declared) == 1 else "two_table" if len(declared) == 2 else "three_plus_table"
    row["quality_flags"].update(execution_policy=POLICY, evidence_policy="operator_dependency_trace",
                                table_ablation_labels=ablations, label_by_execution=True)
    if row["label"] == "NEI":
        row["quality_flags"]["nei_missing_binding"] = run["missing"]
    else:
        row["quality_flags"].pop("nei_missing_binding", None)
        row["quality_flags"]["tables_read"] = len(run["tables_used"])
    return row, {"reason": "retained", "old_label": original["label"], "new_label": row["label"],
                 "removed_unbound_tables": unused,
                 "old_evidence_cells": len(original["evidence_cells"]),
                 "new_evidence_cells": len(row["evidence_cells"])}


def audit(rows, schema):
    issues, ids, texts, tables_seen, packages_seen = [], set(), set(), {}, {}
    validator = SchemaValidator(schema)
    counts = collections.Counter()
    for row in rows:
        rid = row["id"]
        def fail(kind, detail=""):
            issues.append({"id": rid, "kind": kind, "detail": detail})
        for error in validator.iter_errors(row):
            fail("schema", f"{list(error.absolute_path)}: {error.message}")
        if rid in ids: fail("duplicate_id")
        if row["claim"] in texts: fail("duplicate_claim")
        ids.add(rid)
        texts.add(row["claim"])
        tables = {t["table_id"]: t for t in row["tables"]}
        if len(tables) != len(row["tables"]): fail("duplicate_table_id_in_package")
        for tid, table in tables.items():
            identity = digest({k: table[k] for k in ("headers", "rows")})
            if tables_seen.setdefault(tid, identity) != identity: fail("table_id_collision", tid)
        ph = digest(row["tables"])
        if packages_seen.setdefault(row["evidence_package_id"], ph) != ph: fail("package_id_collision")
        profiles = {tid: profile_table(t) for tid, t in tables.items()}
        try:
            run = execute(row["program"]["operators"], profiles)
            if run["label"] != row["label"]: fail("execution_label")
            cells = row["context_cells"] if row["label"] == "NEI" else row["evidence_cells"]
            if cells != run["cells"]: fail("execution_evidence")
            if row["label"] == "NEI" and row["evidence_cells"]: fail("nei_has_decisive_evidence")
            if row["label"] != "NEI":
                if not cells: fail("missing_evidence")
                for tid in tables:
                    if execute(row["program"]["operators"], {k:p for k,p in profiles.items() if k != tid})["label"] != "NEI":
                        fail("table_not_necessary", tid)
                counts["decidable"] += 1
                counts["decidable_multitable"] += int(len(tables) > 1)
            for cell in cells:
                table = tables[cell["table_id"]]
                raw = table["rows"][cell["row"]][table["headers"].index(cell["col"])]
                if raw != cell["raw_value"]: fail("raw_cell_mismatch")
                value = cell["normalized_value"]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    counts["numeric_dependency_cells"] += 1
                    if parse_number(raw) != value: fail("numeric_normalization")
        except (InvalidProgram, KeyError, TypeError, ValueError, IndexError) as exc:
            fail("execution_or_cell_error", str(exc))
        counts["records"] += 1
    return {"passed": not issues, "counts": dict(counts), "issue_count": len(issues), "issues": issues,
            "scope": "Structural/execution checks; not independent human semantic validation",
            "historical_source_registry_recovered": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--release", default="review-repair-r5")
    ap.add_argument("--semantic-nei-gate", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("output must be empty; use a new version instead of overwriting a frozen release")
    originals = [json.loads(line) for line in args.candidate.read_text().splitlines() if line.strip()]
    rows, changes, quarantine = [], [], []
    for original in originals:
        from nei_semantic_gate import quarantine_reason
        semantic_issue = quarantine_reason(original) if args.semantic_nei_gate else None
        if semantic_issue:
            repaired, change = None, {"reason":"semantic_nei_quarantine", "detail":semantic_issue}
            quarantine.append({"candidate":original, "issue":semantic_issue})
        else:
            repaired, change = repair_record(original)
        change["candidate_id"] = original["id"]
        if repaired is not None:
            repaired["id"] = f"{args.release}-{len(rows)+1:05d}"
            change["repaired_id"] = repaired["id"]
            rows.append(repaired)
        changes.append(change)
    if not rows: raise SystemExit("no valid rows")
    schema = json.loads((ROOT / "schemas/claim.schema.json").read_text())
    report = audit(rows, schema)
    if not report["passed"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit("audit failed; no repaired release written")
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl(args.out / "claims.jsonl", rows)
    jsonl(args.out / "changes.jsonl", changes)
    if args.semantic_nei_gate:
        jsonl(args.out / "semantic_quarantine.jsonl", quarantine)
    dump(args.out / "validation.json", report)
    st = statistics(rows)
    dump(args.out / "stats.json", {"statistics": st, "surface_baselines": surface_baselines(rows)})
    unique_tables = {t["table_id"]:t for r in rows for t in r["tables"]}
    schemas = collections.Counter(tuple(t["headers"]) for t in unique_tables.values())
    dimensions = sorted((len(t["rows"]), len(t["headers"])) for t in unique_tables.values())
    dump(args.out / "table_structure.json", {
        "unit": "unique table IDs as represented in packages, including derived views",
        "source_type": "Chinese Wikipedia only; not a representative sample of all Chinese tables",
        "tables": len(unique_tables), "distinct_exact_header_schemas": len(schemas),
        "top_repeated_schemas": [{"headers":list(h), "table_count":n} for h,n in schemas.most_common(10)],
        "row_count_min_median_max": [min(x[0] for x in dimensions), sorted(x[0] for x in dimensions)[len(dimensions)//2], max(x[0] for x in dimensions)],
        "column_count_min_median_max": [min(x[1] for x in dimensions), sorted(x[1] for x in dimensions)[len(dimensions)//2], max(x[1] for x in dimensions)],
        "claims_with_derived_views": sum(any(t["source"].get("derived_from") for t in r["tables"]) for r in rows)})
    pages, packages = {}, {}
    for row in rows:
        packages[row["evidence_package_id"]] = {"package_id": row["evidence_package_id"],
            "kind": row["package_kind"], "table_ids": [t["table_id"] for t in row["tables"]]}
        for table in row["tables"]:
            source = table["source"]
            pages[(source["page_id"], source["revision_id"])] = {k:source[k] for k in
                ("page_id", "revision_id", "page_title", "url", "revision_url", "license") if k in source}
    jsonl(args.out / "source_pages.jsonl", [pages[k] for k in sorted(pages)])
    jsonl(args.out / "packages.jsonl", [packages[k] for k in sorted(packages)])
    registry = []
    for sid in sorted({r["program"]["skeleton_id"] for r in rows}):
        rr = [r for r in rows if r["program"]["skeleton_id"] == sid]
        variants = {digest(template(r["program"]["operators"])): template(r["program"]["operators"]) for r in rr}
        registry.append({"skeleton_id": sid, "category": rr[0]["program"]["category"],
            "provenance_status": "reconstructed_from_current_builders_not_original_source_registry",
            "source_support": [], "historical_source_record_ids_available": False,
            "builder": "src/dart_fact/families.py", "builder_sha256": file_hash(ROOT / "src/dart_fact/families.py"),
            "package_kinds": sorted({r["package_kind"] for r in rr}),
            "program_templates": list(variants.values()), "instances": len(rr),
            "labels": dict(collections.Counter(r["label"] for r in rr)),
            "example_record_ids": [r["id"] for r in rr[:3]]})
    jsonl(args.out / "skeleton_registry.jsonl", registry)
    reviewed = json.loads((ROOT / "tools/repro_spec.json").read_text())
    dump(args.out / "review_comparison.json", {"reviewed": reviewed["overall"], "repaired": st,
        "note": "Different records and provenance; numerical proximity is not evidence of recovering the reviewed data."})
    reasons = collections.Counter(c["reason"] for c in changes)
    dump(args.out / "build_summary.json", {"claims": len(rows), "candidate_claims": len(originals),
        "labels": st["labels"], "surface": dict(collections.Counter(r["surface"]["kind"] for r in rows)),
        "rejects": {k:v for k,v in reasons.items() if k != "retained"},
        "label_corrections": sum(c.get("old_label") != c.get("new_label") for c in changes if c["reason"] == "retained"),
        "evidence_policy": POLICY,
        "model_predictions_used_for_selection": bool(args.semantic_nei_gate),
        "model_informed_rule_development": bool(args.semantic_nei_gate),
        "per_instance_model_correctness_used_for_selection": False,
        "semantic_nei_gate": bool(args.semantic_nei_gate),
        "semantic_gate_scope": "Conservative aliases and type/scope ambiguity; not exhaustive human semantic validation"})
    code = {str(p.relative_to(ROOT)): file_hash(p) for folder in ("src/dart_fact", "tools")
            for p in sorted((ROOT / folder).glob("*.py"))}
    dump(args.out / "freeze_manifest.json", {"release": args.release, "date": "2026-09-20",
        "candidate_sha256": file_hash(args.candidate), "candidate_path": str(args.candidate),
        "claims_sha256": file_hash(args.out / "claims.jsonl"), "code_sha256": code,
        "artifacts_sha256": {p.name: file_hash(p) for p in sorted(args.out.iterdir()) if p.is_file()},
        "evaluation_status": "not_run", "human_validation_status": "not_completed",
        "historical_source_derivations": "unavailable"})
    print(json.dumps({"claims":len(rows), "rejected":len(originals)-len(rows), "statistics":st,
                      "validation": report["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
