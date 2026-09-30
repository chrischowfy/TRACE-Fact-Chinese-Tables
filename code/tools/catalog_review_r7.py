"""Build a fresh, offline candidate inventory before r7 coverage selection.

The frozen r6 files, historical candidate caches and model predictions are not
modified. Tables are stored once, separately from compact candidate records.
This inventory is not a validated benchmark release.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from dart_fact import pipeline
from pool_census import pool
from repair_review_release import digest, repair_record
from repair_review_r6 import semantic_issues


SPORT_DOMAINS = {"football", "football_intl", "basketball", "medals"}


def register_unlisted(cache: Path) -> list[dict]:
    """Register cached pages that no SERIES entry names, for this process only.

    `build_packages` only visits registered titles.  Classification is by title
    (`register_cached.classify`); sports pages are not added because sport is
    already over-represented and entity-NEI is banned there.  The registry module
    is not edited; every addition is returned for the inventory report."""
    from dart_fact import source_adapters
    from dart_fact.acquire import to_simplified
    from register_cached import classify
    known = {to_simplified(t) for s in source_adapters.SERIES for t in s["titles"]}
    added = []
    for path in sorted(cache.glob("*.json")):
        try:
            page = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(page, dict) or not page.get("title"):
            continue
        variants = [v for v in [page["title"], to_simplified(page["title"]), page.get("resolved_title"),
                                to_simplified(page.get("resolved_title") or "")] if v]
        if any(to_simplified(v) in known for v in variants):
            continue
        # The cache file is named by the title it was requested under.
        title = next((v for v in variants if hashlib.sha1(v.encode("utf-8")).hexdigest()[:16] == path.stem), None)
        if not title or source_adapters.SENSITIVE_TITLE_RE.search(to_simplified(title)):
            continue
        cls = classify(to_simplified(title))
        if not cls or cls[0] in SPORT_DOMAINS:
            continue
        domain, series = cls
        domain = "infrastructure" if domain == "architecture" else domain
        source_adapters.SERIES.append({"series": series, "domain": domain, "titles": [title]})
        known.add(to_simplified(title))
        added.append({"title": title, "domain": domain, "series": series, "file": path.name})
    return added


def inventory(cache: Path, out: Path, register: bool = False):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Inventory output must be new/empty")
    registered = register_unlisted(cache) if register else []
    if register:
        print(f"Registered {len(registered)} unlisted cached pages for this run", flush=True)
    print("Generating fresh candidates from pinned local page caches", flush=True)
    items = pool(str(cache))
    print(f"Initial executable drafts: {len(items)}", flush=True)
    if not items:
        raise ValueError("No executable candidates; --cache must point to the pages directory")
    rejected = Counter()
    for name, gate in pipeline.POST_GATES:
        kept = []
        for item in items:
            if gate(item):
                rejected[name] += 1
            else:
                kept.append(item)
        items = kept
    print(f"After original post-gates: {len(items)}", flush=True)
    rows, tables, seen = [], {}, set()
    for index, item in enumerate(items):
        pipeline._restate_scope_and_periods(item)
        row = pipeline.to_record(index, item)
        row, change = repair_record(row)
        if row is None:
            rejected[change["reason"]] += 1
            continue
        issues = semantic_issues(row)
        if issues:
            rejected[issues[0]["reason"]] += 1
            continue
        identity = digest({"operators": row["program"]["operators"],
                           "tables": [t["table_id"] for t in row["tables"]]})
        if identity in seen:
            rejected["duplicate_program_binding"] += 1
            continue
        seen.add(identity)
        row["id"] = "r7cand-" + identity[:24]
        row["quality_flags"]["candidate_origin"] = "fresh_cached_generation"
        row["quality_flags"]["contrast_base"] = item["draft"].base_id
        for table in row["tables"]:
            tid = table["table_id"]
            # Period metadata may differ in packages; evidence rows/headers may not.
            if tid in tables and (tables[tid]["headers"], tables[tid]["rows"]) != (table["headers"], table["rows"]):
                raise ValueError("Conflicting candidate table identity: " + tid)
            tables[tid] = table
        row["table_refs"] = [{"table_id": t["table_id"],
                              **({"period": t["period"]} if "period" in t else {})}
                             for t in row.pop("tables")]
        rows.append(row)
        if index and index % 2000 == 0:
            print(f"Replayed {index} candidates; kept {len(rows)}", flush=True)
    grouped = defaultdict(Counter)
    for row in rows:
        grouped[row["program"]["category"]][row["label"]] += 1
    report = {"status": "candidate_inventory_not_release", "records": len(rows),
              "labels": dict(Counter(r["label"] for r in rows)),
              "tables_per_record": dict(Counter(len(r["table_refs"]) for r in rows)),
              "group_by_label": {g: dict(c) for g, c in sorted(grouped.items())},
              "domains": dict(Counter(r["domain"] for r in rows)),
              "topics": len({r["topic"] for r in rows}), "unique_package_tables": len(tables),
              "raw_source_tables": len({t["source"].get("derived_from", t["table_id"]) for t in tables.values()}),
              "rejected": dict(rejected), "model_api_calls": 0,
              "cache_dir": str(cache), "registered_unlisted_pages": registered}
    out.mkdir(parents=True, exist_ok=True)
    for name, records in [("candidates.jsonl", rows), ("tables.jsonl", [tables[k] for k in sorted(tables)])]:
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in records:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/cache/pages")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--register-unlisted", action="store_true",
                        help="register cached pages missing from SERIES for this run (recorded in the report)")
    args = parser.parse_args()
    inventory(args.cache, args.out, args.register_unlisted)


if __name__ == "__main__":
    main()
