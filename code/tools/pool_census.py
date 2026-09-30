"""Census of the candidate pool: how many instances *could* be sampled per family x label.

The sampler picks ~1.6k records out of ~13k valid drafts under per-package / per-page / per-domain
caps.  Before changing quotas it matters whether a target cell (e.g. 'Real-join rank-bridge NEI = 47')
is reachable at all.  This runs the build up to the candidate pool, applies every post-sampling gate
to the whole pool, and reports the ceiling per cell against tools/repro_spec.json.

    PYTHONPATH=src python tools/pool_census.py --cache runs/cache
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dart_fact import pipeline  # noqa: E402
from dart_fact.packages import build_packages  # noqa: E402
from dart_fact.families import BUILDERS, Context  # noqa: E402
from dart_fact.realize import realize  # noqa: E402

# the post-sampling gates, in the order build() applies them
GATES = [
    ("nei_metric_present_on_page", pipeline._metric_on_page),
    ("hub_profile_metric_nei_implausible_value", pipeline._hp_metric_nei),
    ("attribute_value_with_change_marker", pipeline._changing_attribute),
    ("multi_table_single_table_shortcut", pipeline._single_table_shortcut),
    ("period_pair_scope_mismatch", pipeline._period_scope_mismatch),
    ("slice_scope_for_rank_or_category", pipeline._slice_rank_or_category),
    ("split_pair_unscoped_stage_tables", pipeline._unscoped_split_pair),
    ("count_over_partial_list", pipeline._count_over_partial_list),
    ("negative_value_in_count_metric", pipeline._negative_count),
    ("nei_metric_derivable_from_package", pipeline._derivable_metric_nei),
    ("nei_rank_of_absent_entity", pipeline._rank_absent_entity),
    ("non_member_unit_in_extreme_or_rank", pipeline._non_member_units),
    ("incomplete_metric_name", pipeline._incomplete_metric_name),
    ("aggregate_row_in_ranking", pipeline._aggregate_row_in_ranking),
    ("rank_differs_from_table_rank_column", pipeline._rank_column_mismatch),
    ("superlative_over_subset_table", pipeline._superlative_over_subset),
    ("rank_bridge_detail_not_top_list", pipeline._rank_bridge_detail_incomplete),
    ("nei_metric_confusable_with_header", pipeline._confusable_metric_nei),
    ("join_filter_open_entity_set", pipeline._join_filter_open_set),
    ("category_count_with_duplicate_members", pipeline._duplicate_members),
    ("nei_entity_in_sports_table", pipeline._sports_entity_nei),
    ("nei_entity_admin_suffix_variant", pipeline._admin_suffix_alias),
    ("nei_entity_in_membership_scoped_table", pipeline._membership_scoped_entity_nei),
    ("nei_entity_below_ranked_list_cutoff", pipeline._nei_below_list_cutoff),
    ("extreme_over_one_of_several_tables", pipeline._extreme_over_sibling_table),
    ("extreme_over_ranked_list_continuation", pipeline._ranked_list_continuation),
    ("extreme_over_prc_table_with_partial_region_set", pipeline._prc_region_set_unclear),
    ("join_key_nei_value_readable_on_page", pipeline._join_key_value_readable),
]


def pool(cache_dir: str, memo: str | None = None) -> list[dict]:
    """Candidate pool.  `memo` caches it on disk: rebuilding costs minutes, and the gate-tracing tools
    re-read it many times.  The cache is keyed by the caller; delete the file after changing a builder."""
    if memo and pathlib.Path(memo).exists():
        import pickle
        with open(memo, "rb") as fh:
            return pickle.load(fh)
    items = _pool_uncached(cache_dir)
    if memo:
        import pickle
        pathlib.Path(memo).parent.mkdir(parents=True, exist_ok=True)
        with open(memo, "wb") as fh:
            pickle.dump(items, fh)
    return items


def _pool_uncached(cache_dir: str) -> list[dict]:
    from dart_fact.resolve import load as load_redirects
    load_redirects(pathlib.Path(cache_dir).parent / "redirects.json")
    packages = build_packages(cache_dir, cache_only=True)
    ctx = Context(packages, seed=20260913)
    kept = []
    for pkg in packages:
        try:
            drafts = BUILDERS[pkg["kind"]](pkg, ctx)
        except Exception:  # noqa: BLE001
            continue
        profiles = pipeline._profiles(pkg)
        for d in drafts:
            run, _reason = pipeline.check_draft(d, pkg, profiles)
            if run is None:
                continue
            claim = realize(d.skeleton, d.slots, d.base_id)
            if pipeline.LEAK_RE.search(claim) or len(claim) > 90 or len(re.findall(r"[A-Za-z]", claim)) > 6:
                continue
            kept.append({"draft": d, "run": run, "claim": claim, "pkg": pkg})
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/cache")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    kept = pool(args.cache)
    print(f"valid drafts before gates: {len(kept)}")

    survivors = list(kept)
    for name, test in GATES:
        before = len(survivors)
        survivors = [i for i in survivors if not test(i)]
        if before - len(survivors):
            print(f"  gate {name:<46} -{before - len(survivors)}")
    print(f"pool after all gates:      {len(survivors)}")

    spec = json.loads((pathlib.Path(__file__).with_name("repro_spec.json")).read_text(encoding="utf-8"))
    cells: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    pages: dict[tuple[str, str], set] = collections.defaultdict(set)
    for i in survivors:
        g, lab = i["draft"].group, i["draft"].label
        cells[g][lab] += 1
        pages[(g, lab)].add(i["pkg"]["page_title"])

    print()
    print(f"{'family':<36} {'label':<9} {'want':>5} {'pool':>6} {'pages':>6}  status")
    short = []
    for fam, want in spec["family_by_label"].items():
        if fam.startswith("_"):
            continue
        for lab in ("SUPPORTS", "REFUTES", "NEI"):
            have = cells.get(fam, {}).get(lab, 0)
            npages = len(pages.get((fam, lab), ()))
            ok = have >= want[lab]
            print(f"{fam[:36]:<36} {lab:<9} {want[lab]:>5} {have:>6} {npages:>6}  {'ok' if ok else 'SHORT by %d' % (want[lab]-have)}")
            if not ok:
                short.append((fam, lab, want[lab] - have))

    print()
    if short:
        print("cells that cannot be filled from the current pool:")
        for fam, lab, gap in sorted(short, key=lambda x: -x[2]):
            print(f"  {fam} / {lab}: need {gap} more")
    else:
        print("every target cell is reachable by re-quota alone")

    if args.json:
        args.json.write_text(json.dumps(
            {"pool_total": len(survivors),
             "cells": {g: dict(c) for g, c in cells.items()},
             "short": [{"family": f, "label": l, "gap": g} for f, l, g in short]},
            ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
