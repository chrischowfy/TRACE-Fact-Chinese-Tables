"""Dataset-level statistics and artifact checks (stdlib only).

``surface_baselines`` answers reviewer R2's template-bias question directly: how well can a label be
predicted WITHOUT tables — from the claim text (character n-gram naive Bayes, cross-validated with
source pages kept disjoint across folds), from the skeleton id alone, or from the comparison word.
"""
from __future__ import annotations

import collections
import math
import random
import re
from typing import Any

LABELS = ("SUPPORTS", "REFUTES", "NEI")


def metrics(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    f1 = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = round(100 * (2 * pr * rc / (pr + rc) if pr + rc else 0.0), 1)
    return {"n": len(pairs), "accuracy": round(100 * sum(g == p for g, p in pairs) / max(1, len(pairs)), 1),
            "macro_f1": round(sum(f1.values()) / 3, 1), "f1": f1}


def _grams(text: str, n_max: int = 3) -> list[str]:
    s = re.sub(r"\d+(?:\.\d+)?", "0", text)
    return [s[i:i + n] for n in range(1, n_max + 1) for i in range(len(s) - n + 1)]


def source_groups(rows):
    """Keep all claims sharing any source page in one connected component.

    Topic/caption strings are not source-page identities. Multi-page packages
    also connect their pages and must not bridge train and test folds.
    """
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    pages_per_row = [sorted({str(t["source"]["page_id"]) for t in r["tables"]}) for r in rows]
    if any(not pages for pages in pages_per_row):
        raise ValueError("source-disjoint evaluation requires source pages for every record")
    for pages in pages_per_row:
        for page in pages:
            a, b = find(pages[0]), find(page)
            parent[max(a, b)] = min(a, b)
    return [find(pages[0]) for pages in pages_per_row]


def claim_only_nb(rows: list[dict[str, Any]], folds: int = 5, seed: int = 13) -> dict[str, Any]:
    row_groups = source_groups(rows)
    groups = sorted(set(row_groups))
    if len(groups) < 2:
        raise ValueError("at least two independent source components are required")
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % folds for i, g in enumerate(groups)}
    pairs = []
    for k in range(folds):
        train = [r for r,g in zip(rows, row_groups) if fold_of[g] != k]
        test = [r for r,g in zip(rows, row_groups) if fold_of[g] == k]
        if not train or not test:
            continue
        prior = collections.Counter(r["label"] for r in train)
        counts = {lab: collections.Counter() for lab in LABELS}
        for r in train:
            counts[r["label"]].update(_grams(r["claim"]))
        vocab = len(set().union(*counts.values()))
        totals = {lab: sum(counts[lab].values()) for lab in LABELS}
        for r in test:
            scores = {}
            for lab in LABELS:
                s = math.log((prior[lab] + 1) / (len(train) + 3))
                for g in _grams(r["claim"]):
                    s += math.log((counts[lab][g] + 1) / (totals[lab] + vocab))
                scores[lab] = s
            pairs.append((r["label"], max(scores, key=scores.get)))
    return metrics(pairs) | {"grouping": "connected_source_page_ids", "source_components": len(groups),
                             "folds": folds, "seed": seed}


def _cue(claim: str) -> str:
    for word, tag in (("相同", "eq"), ("一样", "eq"), ("持平", "eq"), ("多于", "gt"), ("高于", "gt"), ("少于", "lt"),
                      ("低于", "lt"), ("最多", "max"), ("最高", "max"), ("最少", "min"), ("最低", "min"),
                      ("增加", "up"), ("减少", "down"), ("排", "rank"), ("共有", "count"), ("一共", "count")):
        if word in claim:
            return tag
    return "none"


def majority_by(rows: list[dict[str, Any]], key) -> dict[str, Any]:
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        table[key(r)][r["label"]] += 1
    major = {k: c.most_common(1)[0][0] for k, c in table.items()}
    return metrics([(r["label"], major[key(r)]) for r in rows])


def surface_baselines(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = collections.Counter(r["label"] for r in rows).most_common(1)[0][0]
    return {
        "majority": metrics([(r["label"], overall) for r in rows]),
        "skeleton_only": majority_by(rows, lambda r: r["program"]["skeleton_id"]),
        "comparison_word_only": majority_by(rows, lambda r: _cue(r["claim"])),
        "domain_only": majority_by(rows, lambda r: r["domain"]),
        "claim_only_char_ngram_nb_page_disjoint_cv": claim_only_nb(rows),
    }


def statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tables = {(t["table_id"]) for r in rows for t in r["tables"]}
    raw_tables = {t["table_id"] for r in rows for t in r["tables"] if not t["source"].get("derived_from")}
    derived_from = {t["source"]["derived_from"] for r in rows for t in r["tables"] if t["source"].get("derived_from")}
    pages = {t["source"]["page_id"] for r in rows for t in r["tables"]}
    decidable = [r for r in rows if r["label"] != "NEI"]
    topics = collections.Counter(r["topic"] for r in rows)
    return {
        "claims": len(rows),
        "labels": dict(collections.Counter(r["label"] for r in rows)),
        "evidence_packages": len({r["evidence_package_id"] for r in rows}),
        "source_pages": len(pages),
        "distinct_tables_in_packages": len(tables),
        "raw_wikipedia_tables": len(raw_tables | derived_from),
        "avg_evidence_cells_decidable": round(sum(len(r["evidence_cells"]) for r in decidable) / max(1, len(decidable)), 2),
        "skeletons": len({r["program"]["skeleton_id"] for r in rows}),
        "groups": len({r["program"]["category"] for r in rows}),
        "group_by_label": {g: dict(collections.Counter(r["label"] for r in rows if r["program"]["category"] == g))
                           for g in sorted({r["program"]["category"] for r in rows})},
        "skeleton_by_label": {s: dict(collections.Counter(r["label"] for r in rows if r["program"]["skeleton_id"] == s))
                              for s in sorted({r["program"]["skeleton_id"] for r in rows})},
        "topology": dict(collections.Counter(r["table_topology"] for r in rows)),
        "package_kind": dict(collections.Counter(r["package_kind"] for r in rows)),
        "domain": dict(collections.Counter(r["domain"] for r in rows)),
        "domain_by_label": {d: dict(collections.Counter(r["label"] for r in rows if r["domain"] == d))
                            for d in sorted({r["domain"] for r in rows})},
        "topics": len(topics), "largest_topic": topics.most_common(1)[0] if topics else None,
        "derived_tables_share": round(sum(1 for r in rows if any(t["source"].get("derived_from") for t in r["tables"]))
                                      / max(1, len(rows)), 3),
        "nei_axis": dict(collections.Counter((r["quality_flags"].get("perturbation_axis") or "") for r in rows
                                             if r["label"] == "NEI")),
        "refute_axis": dict(collections.Counter((r["quality_flags"].get("perturbation_axis") or "") for r in rows
                                                if r["label"] == "REFUTES")),
    }
