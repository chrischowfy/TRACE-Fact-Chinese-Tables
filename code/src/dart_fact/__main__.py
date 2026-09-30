"""Command line: ``python -m dart_fact {acquire,build,stats,verify}``.

  acquire  fetch every seed page (source_adapters.SERIES) into the page cache
  build    packages -> executed, gated, realized claims (claims.jsonl + build_summary.json)
  stats    dataset statistics and table-free artifact baselines for a claims file
  verify   re-execute every program in a claims file and check label / evidence consistency
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .acquire import extract_tables, fetch_page
from .analysis import statistics, surface_baselines
from .executor import InvalidProgram, execute
from .pipeline import build
from .source_adapters import SERIES
from .tables import profile_table


def cmd_acquire(args: argparse.Namespace) -> None:
    inventory = []
    for series in SERIES:
        for title in series["titles"]:
            try:
                page = fetch_page(title, args.cache)
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR {title}: {exc}", file=sys.stderr)
                continue
            n = len(extract_tables(page)) if page else 0
            inventory.append({"series": series["series"], "title": title, "exists": bool(page), "tables": n})
            print(f"{'OK ' if page else 'MISSING'} {title} tables={n}")
    Path(args.cache, "..", "seed_inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")


def cmd_build(args: argparse.Namespace) -> None:
    from .frozen import find_manifest, replay
    manifest = find_manifest(args.cache)
    if manifest is not None and not args.regenerate:
        if args.quota or args.naturalize or args.naturalize_limit is not None:
            raise SystemExit(
                "the cache contains a frozen release manifest; these options require --regenerate"
            )
        summary = replay(args.cache, args.out, manifest)
        print(f"replayed frozen release manifest: {manifest}")
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return

    quota = domain_cap = None
    table_target = 0
    four_table = 0
    if args.quota:
        spec = json.loads(Path(args.quota).read_text(encoding="utf-8"))
        table = spec.get("family_by_label", spec)
        quota = {fam: {lab: cells[lab] for lab in ("SUPPORTS", "REFUTES", "NEI")}
                 for fam, cells in table.items() if not fam.startswith("_")}
        # the published composition also fixes how much of the corpus one domain may contribute
        four_table = int(spec.get("table_count", {}).get("four_or_more_tables", 0) or 0)
        # once this many distinct source tables are covered, stop preferring packages that add a new
        # one: without a stop the spread objective overshoots the table target and inflates the
        # evidence-package count, and both are checked exactly
        table_target = int(spec.get("overall", {}).get("raw_tables", 0) or 0)
        football = spec.get("topics", {}).get("football_claims")
        if football:
            domain_cap = {"football": int(football * 0.6), "football_intl": int(football * 0.4)}
    summary = build(args.cache, args.out, seed=args.seed, cache_only=True, naturalize_api=args.naturalize,
                    naturalize_limit=args.naturalize_limit, quota=quota, domain_cap=domain_cap,
                    max_per_package=args.max_per_package, max_per_page=args.max_per_page,
                    metric_nei_budget=args.metric_nei_budget, four_table_target=four_table,
                    table_target=table_target)
    print(json.dumps({k: v for k, v in summary.items() if k != "rejects"}, ensure_ascii=False, indent=1))


def cmd_resolve(args: argparse.Namespace) -> None:
    """Resolve entity names and linked article titles to canonical zh-Wikipedia articles (network)."""
    from .packages import PAGE_LINKS, build_packages
    from .resolve import resolve_titles
    names: set[str] = set()
    for pkg in build_packages(args.cache, cache_only=True):
        for t in pkg["tables"]:
            prof = profile_table(t)
            if prof.key:
                names.update(prof.key_values())
    for pairs in PAGE_LINKS.values():
        for pair in pairs:
            text, _, link = pair.partition("\t")
            names.update((text, link))
    print(f"{len(names)} names to resolve")
    n = resolve_titles(sorted(names), Path(args.cache).parent / "redirects.json")
    print(f"resolved {n} new names")


def cmd_stats(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in open(args.claims, encoding="utf-8")]
    report = {"statistics": statistics(rows), "surface_baselines": surface_baselines(rows)}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


def cmd_verify(args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in open(args.claims, encoding="utf-8")]
    bad = 0
    for r in rows:
        profiles = {t["table_id"]: profile_table(t) for t in r["tables"]}
        try:
            runner = execute
            if r.get("quality_flags", {}).get("execution_policy") == "review_repair_v2":
                from .review_repair import execute as review_execute
                runner = review_execute
            run = runner(r["program"]["operators"], profiles)
        except InvalidProgram as exc:
            print(f"{r['id']}: invalid program: {exc}")
            bad += 1
            continue
        cells = run["cells"] if r["label"] != "NEI" else []
        context_ok = r["label"] != "NEI" or run["cells"] == r.get("context_cells", [])
        if run["label"] != r["label"] or cells != r["evidence_cells"] or not context_ok:
            print(f"{r['id']}: label {r['label']} -> {run['label']} / evidence match {cells == r['evidence_cells']}")
            bad += 1
    print(f"verified {len(rows)} claims, {bad} inconsistent")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser(prog="dart_fact")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--cache", default="runs/cache/pages")
    b = sub.add_parser("build")
    b.add_argument("--cache", default="runs/cache/pages")
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=20260913)
    b.add_argument("--naturalize", action="store_true",
                   help="call the LLM for groups missing from runs/cache/naturalize.json (needs DEEPSEEK_API_KEY)")
    b.add_argument("--naturalize-limit", type=int, default=None, help="pilot: only this many sibling groups")
    b.add_argument("--max-per-package", type=int, default=29,
                   help="cap on claims from one evidence package (the published build reports 29)")
    b.add_argument("--max-per-page", type=int, default=44,
                   help="cap on claims from one source page (the published build reports 44)")
    b.add_argument("--metric-nei-budget", type=int, default=0,
                   help="instances the second pass may lose in exchange for restricting metric-slot "
                        "NEI to words that also occur in the released decidable claims")
    b.add_argument("--quota", default=None,
                   help="JSON with family_by_label targets (e.g. tools/repro_spec.json): select exactly "
                        "that many instances per reasoning family and label instead of a global target")
    b.add_argument("--regenerate", action="store_true",
                   help="ignore a frozen release manifest next to the cache and run candidate generation")
    r = sub.add_parser("resolve")
    r.add_argument("--cache", default="runs/cache/pages")
    s = sub.add_parser("stats")
    s.add_argument("--claims", required=True)
    s.add_argument("--out")
    v = sub.add_parser("verify")
    v.add_argument("--claims", required=True)
    args = ap.parse_args()
    {"acquire": cmd_acquire, "build": cmd_build, "resolve": cmd_resolve, "stats": cmd_stats,
     "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
