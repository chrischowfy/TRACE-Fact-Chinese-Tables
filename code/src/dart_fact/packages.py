"""Evidence packages: which tables a claim is verified against.

Package topologies (``kind``):
  single              one table                                          -> single-table comparison & ranking
  split_pair          two same-schema tables with disjoint entities      -> two-entity cross-table comparison
  category_decomp     one flat table with a category column, decomposed  -> category decomposition
                      into category / membership / metric tables (synthetic ids, no new facts)
  rank_detail         ranking table + detail table whose foreign key      -> real-join rank–bridge
                      points at the ranking table's entities
  hub_profile         two tables keyed by the same entities, one with     -> hub-profile conjunction
                      text attributes, one with metrics
  join_filter         attribute table with a repeated attribute + metric  -> join-filter aggregation
                      table keyed by the same entities
  category_partner    category decomposition + a second real table of the page (four tables)
  period_pair         two snapshots of the same entities (two season      -> cross-period growth
                      pages, or two year columns split into two tables)
"""
from __future__ import annotations

import hashlib
import itertools
import re
from typing import Any

from .acquire import extract_tables, fetch_page, to_simplified
from .source_adapters import NON_SPORTS_METRIC_RE, SENSITIVE_CELL_RE, SENSITIVE_TITLE_RE, SERIES, clean_name, metric_info
from .tables import PERIOD_HEADER_RE, TableProfile, entity_key, join_key, parse_number, profile_table, usable


PAGE_TEXT: dict[str, list[str]] = {}     # page title -> every normalized string on the page (all tables)
PAGE_HEADERS: dict[str, list[str]] = {}  # page title -> every column header on the page (all tables)
PAGE_LINKS: dict[str, list[str]] = {}
PAGE_TABLE_SHAPES: dict[str, list[tuple[str, tuple[str, ...]]]] = {}  # page -> (key header, headers) per table    # page title -> "cell text<TAB>linked article" pairs from all tables
# page -> per table: (set of key values, set of column headers).  A join-key NEI claims a metric for an
# entity that has no row in the partner table; this index proves the value is not stated by some *other*
# table of the same page either, which is what makes the instance genuinely undecidable.
PAGE_KEY_SETS: dict[str, list[tuple[frozenset[str], frozenset[str]]]] = {}


def _pid(*parts: str) -> str:
    return "pkg_" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def merge_split_tables(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge runs of consecutive tables with identical headers and disjoint entity keys (split scorer lists)."""
    merged: list[dict[str, Any]] = []
    for t in tables:
        prev = merged[-1] if merged else None
        if prev and prev["headers"] == t["headers"]:
            pp, tp = profile_table(prev), profile_table(t)
            if pp.key and tp.key and pp.key.header == tp.key.header and not (set(pp.key_values()) & set(tp.key_values())):
                if _continues(pp, tp):
                    prev["rows"] = prev["rows"] + t["rows"]
                    prev["source"].setdefault("merged_table_indexes", [prev["source"]["table_index"]])
                    prev["source"]["merged_table_indexes"].append(t["source"]["table_index"])
                    prev["table_id"] = re.sub(r"(m\d+)?$", "", prev["table_id"]) + f"m{len(prev['source']['merged_table_indexes'])}"
                    continue
        merged.append(dict(t, source=dict(t["source"])))
    return merged


def _continues(a: TableProfile, b: TableProfile) -> bool:
    """True when b looks like the continuation of a ranked list (rank column keeps increasing)."""
    ra, rb = a.rank(), b.rank()
    if not ra or not rb:
        return False
    try:
        last = max(float(a.cell(i, ra)) for i in range(len(a.rows)))
        first = min(float(b.cell(i, rb)) for i in range(len(b.rows)))
    except ValueError:
        return False
    return first >= last


def page_tables(title: str, cache_dir: str, cache_only: bool = False) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    page = fetch_page(title, cache_dir, cache_only=cache_only)
    if not page:
        return None, []
    return page, merge_split_tables(extract_tables(page))


def _metric_headers(prof: TableProfile) -> set[str]:
    return {c.header for c in prof.metrics() if metric_info(c.header, unit=c.unit, is_rate=c.is_rate)}


def _overlap(a: TableProfile, b: TableProfile) -> float:
    ka, kb = {join_key(k) for k in a.key_values()}, {join_key(k) for k in b.key_values()}
    return len(ka & kb) / max(1, min(len(ka), len(kb)))


def _fk_column(detail: TableProfile, ranking: TableProfile):
    """Column of `detail` (not its key) whose values point at `ranking` entities."""
    targets = {join_key(k) for k in ranking.key_values()}
    best = None
    for col in detail.columns:
        if detail.key and col.index == detail.key.index or col.kind == "metric":
            continue
        vals = [join_key(detail.cell(r, col)) for r in range(len(detail.rows))]
        vals = [v for v in vals if v]
        if len(vals) < 5:
            continue
        hit = sum(1 for v in vals if v in targets) / len(vals)
        distinct = len(set(vals))
        if hit >= 0.8 and distinct >= 3 and distinct < len(vals) and (best is None or hit > best[1]):
            best = (col, hit)
    return best[0] if best else None


def _decompose_category(prof: TableProfile, cat, page: dict[str, Any]) -> list[dict[str, Any]]:
    base = prof.table
    metrics = [c for c in prof.metrics() if metric_info(c.header, unit=c.unit, is_rate=c.is_rate)]
    names = sorted({entity_key(prof.cell(r, cat)) for r in range(len(prof.rows))})
    ids = {n: f"类别{i + 1:02d}" for i, n in enumerate(names)}
    kh, ch = prof.key.header, cat.header
    src = dict(base["source"], derived_from=base["table_id"], derivation="category_decomposition")
    # A source table with two category columns decomposes twice; without the column in the id both
    # decompositions emit different tables under the same table_id, so the second silently overwrites
    # the first everywhere tables are counted or looked up by id.
    suffix = hashlib.sha1(ch.encode("utf-8")).hexdigest()[:4]
    t_cat = {"table_id": f"{base['table_id']}_cat{suffix}", "title": f"{scope_title(page)}：{ch}",
             "headers": [ch, "类别编号"], "rows": [[n, ids[n]] for n in names], "source": src}
    t_ent = {"table_id": f"{base['table_id']}_ent{suffix}", "title": f"{scope_title(page)}：{kh}所属{ch}",
             "headers": [kh, "类别编号"],
             "rows": [[prof.cell(r, prof.key), ids[entity_key(prof.cell(r, cat))]] for r in range(len(prof.rows))],
             "source": src}
    t_met = {"table_id": f"{base['table_id']}_met{suffix}", "title": f"{scope_title(page)}：{kh}数据",
             "headers": [kh] + [m.header for m in metrics],
             "rows": [[prof.cell(r, prof.key)] + [prof.cell(r, m) for m in metrics] for r in range(len(prof.rows))],
             "source": src}
    return [t_cat, t_ent, t_met]


def _decompose_category_partner(prof: TableProfile, cat, page: dict[str, Any],
                                partners: list[tuple[dict[str, Any], TableProfile]]
                                ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Four tables: the three-way category decomposition of one table, plus a second *real* table of the
    same page keyed by the same entities and carrying a metric the first one does not.

    A claim then has to pick the category's members, rank them on the first metric, and read a second
    metric of the winner somewhere else - the only topology in the benchmark that binds four tables.
    The three decomposed tables carry no facts the source table did not already state; the partner is
    an ordinary table of the page.
    """
    out: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    t_cat, t_ent, t_met = _decompose_category(prof, cat, page)
    mine = {join_key(prof.cell(r, prof.key)) for r in range(len(prof.rows))}
    own = {clean_name(h) for h in prof.table["headers"]}
    for t2, p2 in partners:
        if t2["table_id"] == prof.table["table_id"] or not p2.key:
            continue
        theirs = {join_key(p2.cell(r, p2.key)) for r in range(len(p2.rows))}
        if not mine or not theirs:
            continue
        if len(mine & theirs) / max(1, min(len(mine), len(theirs))) < 0.6:
            continue
        extra = [c for c in p2.metrics()
                 if metric_info(c.header, unit=c.unit, is_rate=c.is_rate) and clean_name(c.header) not in own]
        # the partner must not also carry the ranked metric, or three of the four tables would be enough
        if not extra or any(clean_name(h) in own for h in _metric_headers(p2)):
            continue
        out.append((t2, [t_cat, t_ent, t_met, t2]))
    return out


def _split_periods(prof: TableProfile, page: dict[str, Any]) -> list[tuple[int, int, list[dict[str, Any]]]]:
    """Two-snapshot packages from a table whose metric columns are years (2023 / 2022年GDP …)."""
    groups: dict[str, list] = {}
    for c in prof.metrics():
        if c.period and "p" not in c.header.lower() and "预" not in c.header and "初步" not in c.header:
            groups.setdefault(c.period_rest, []).append(c)
    out = []
    for rest, cols in groups.items():
        cols = sorted(cols, key=lambda c: c.period)
        name = clean_name(rest) if rest else ""
        if not name or not NON_SPORTS_METRIC_RE.search(name):
            found = NON_SPORTS_METRIC_RE.findall(page["title"]) or NON_SPORTS_METRIC_RE.findall(prof.table.get("title") or "")
            name = max(found, key=len) if found else ""
        if not name:
            continue
        if "人均" in page["title"] and "人均" not in name:
            name = "人均" + name
        for a, b in zip(cols, cols[1:]):
            if b.period - a.period > 5:
                continue
            tabs = []
            for c in (a, b):
                src = dict(prof.table["source"], derived_from=prof.table["table_id"], derivation="period_split",
                           period=c.period)
                # two metric groups on one table share a year, so the metric has to be in the id too
                tabs.append({"table_id": f"{prof.table['table_id']}_y{c.period}"
                                         f"{hashlib.sha1(name.encode('utf-8')).hexdigest()[:4]}",
                             "title": f"{scope_title(page)}（{c.period}年）", "period": c.period,
                             "headers": [prof.key.header, name],
                             "rows": [[prof.cell(r, prof.key), prof.cell(r, c)] for r in range(len(prof.rows))],
                             "source": src})
            out.append((a.period, b.period, tabs))
    return out


SPORTS = ("football", "football_intl", "basketball")


def slice_scope(scope: str, table: dict[str, Any], domain: str) -> str:
    caption = table.get("title") or ""
    if caption and "（表" not in caption and len(caption) <= 16 and caption not in scope:
        return f"{scope}{caption}"
    standings = any(h in ("胜", "负", "积分", "分", "胜率") for h in table["headers"])
    if not standings:
        return scope
    if domain == "football_intl":
        return f"{scope}小组赛"
    if domain in SPORTS:
        return f"{scope}的分组赛"
    return scope


def scope_title(page: dict[str, Any]) -> str:
    title = re.sub(r"（[^）]*）|\([^)]*\)", "", to_simplified(page["title"])).strip()
    if re.search(r"列表$|排名$", title):
        return re.sub(r"列表$|排名$", "", title) + "数据"
    return re.sub(r"小组赛$", "", title)


def _year_of(title: str) -> int | None:
    m = re.search(r"((?:19|20)\d{2})年", title)
    return int(m.group(1)) if m else None


def build_packages(cache_dir: str, cache_only: bool = False) -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    season_tables: dict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {}
    for series in SERIES:
        for title in series["titles"]:
            if SENSITIVE_TITLE_RE.search(to_simplified(title)):
                continue
            try:
                page, tables = page_tables(title, cache_dir, cache_only)
            except Exception:  # noqa: BLE001 - network or parse failure: skip the page
                continue
            if not page:
                continue
            meta = {"series": series["series"], "domain": series["domain"], "page_title": to_simplified(page["title"]),
                    "scope": scope_title(page)}
            all_page_headers = sorted({h for t in tables for h in t["headers"]})   # before any filtering
            all_page_links = sorted({lk for t in tables for lk in t.get("cell_links", [])})
            tables = [t for t in tables if not any(SENSITIVE_CELL_RE.search(c) for r in t["rows"] for c in r)
                      and not SENSITIVE_CELL_RE.search(t.get("title", ""))]
            profs = [(t, profile_table(t)) for t in tables]
            header_counts: dict[tuple, int] = {}
            for t, _ in profs:
                header_counts[tuple(t["headers"])] = header_counts.get(tuple(t["headers"]), 0) + 1
            for t, _ in profs:
                t["is_slice"] = header_counts[tuple(t["headers"])] >= 2
            good = [(t, p) for t, p in profs if usable(p)]
            page_text = sorted({entity_key(x) for t in tables for x in [t.get("title", "")] + t["headers"] + [c for r in t["rows"] for c in r] if x})
            PAGE_TEXT[meta["page_title"]] = page_text
            PAGE_HEADERS[meta["page_title"]] = all_page_headers
            PAGE_LINKS[meta["page_title"]] = all_page_links
            PAGE_TABLE_SHAPES[meta["page_title"]] = [(p.key.header if p.key else "", tuple(t["headers"])) for t, p in profs]
            PAGE_KEY_SETS[meta["page_title"]] = [
                (frozenset(join_key(p.cell(r, p.key)) for r in range(len(p.rows))), frozenset(t["headers"]))
                for t, p in profs if p.key]
            year = _year_of(page["title"])
            for t, p in good:
                if t.get("is_slice") and meta["domain"] in ("football", "basketball"):
                    continue      # domestic-league stage/group tables: the scope cannot be stated reliably
                single_meta = dict(meta, scope=slice_scope(meta["scope"], t, meta["domain"])) if t.get("is_slice") else meta
                packages.append(dict(single_meta, package_id=_pid(t["table_id"]), kind="single", tables=[t],
                                     is_slice=bool(t.get("is_slice"))))
                for cat in p.categories():
                    n_cat = len({entity_key(p.cell(r, cat)) for r in range(len(p.rows))})
                    if 2 <= n_cat <= 12 and len(p.rows) >= 6:
                        packages.append(dict(meta, package_id=_pid(t["table_id"], cat.header), kind="category_decomp",
                                             tables=_decompose_category(p, cat, page), category=cat.header,
                                             is_slice=bool(t.get("is_slice"))))
                        for t2, tabs4 in _decompose_category_partner(p, cat, page, good):
                            packages.append(dict(meta, package_id=_pid(t["table_id"], cat.header, t2["table_id"], "cp"),
                                                 kind="category_partner", tables=tabs4, category=cat.header,
                                                 partner_table=t2["table_id"], is_slice=bool(t.get("is_slice"))))
                for y1, y2, tabs in _split_periods(p, page):
                    packages.append(dict(meta, package_id=_pid(t["table_id"], str(y1), str(y2)), kind="period_pair",
                                         tables=tabs, periods=[y1, y2], period_source="columns"))
                if year:
                    season_tables.setdefault(series["series"], []).append((year, t, meta))
            for (ta, pa), (tb, pb) in itertools.combinations(good, 2):
                if (pa.key.header == pb.key.header and _metric_headers(pa) & _metric_headers(pb) and _overlap(pa, pb) == 0
                        and meta["domain"] not in ("football", "basketball", "medals")):
                    if ta["headers"] == tb["headers"]:
                        standings = any(h in ("胜", "负", "积分", "分", "胜率") for h in ta["headers"])
                        packages.append(dict(meta, scope=meta["scope"] + ({"football_intl": "小组赛"}.get(meta["domain"], "的分组赛")
                                                                         if meta["domain"] in SPORTS and standings else ""),
                                             package_id=_pid(ta["table_id"], tb["table_id"]), kind="split_pair",
                                             tables=[ta, tb]))
            with_key = [(t, p) for t, p in profs if p.key and len(p.rows) >= 4]
            for (ta, pa), (tb, pb) in itertools.permutations(with_key, 2):
                standings = any(h in ("胜", "负", "积分", "分", "胜率") for h in ta["headers"])
                detail_rank = pb.rank()
                detail_complete = not tb.get("is_slice") and (
                    detail_rank is None or min((parse_number(pb.cell(r, detail_rank)) or 1) for r in range(len(pb.rows))) == 1)
                if (usable(pa) and usable(pb) and pa.rank() and not ta.get("is_slice") and detail_complete
                        and (standings or meta["domain"] not in ("football", "football_intl", "basketball"))):
                    fk = _fk_column(pb, pa)
                    if fk is not None and pa.key.header != pb.key.header:
                        packages.append(dict(meta, package_id=_pid(ta["table_id"], tb["table_id"], "rd"),
                                             kind="rank_detail", tables=[ta, tb], fk_col=fk.header))
                if usable(pb) and _overlap(pa, pb) >= 0.7 and pa.key.header != "" and ta["headers"] != tb["headers"]:
                    attrs = [c for c in pa.columns if c.kind in ("text", "key", "category")
                             and c.index != pa.key.index and 2 <= _avg_len(pa, c) <= 12]
                    if attrs:
                        packages.append(dict(meta, package_id=_pid(ta["table_id"], tb["table_id"], "hp"),
                                             kind="hub_profile", tables=[ta, tb], attr_cols=[c.header for c in attrs]))
                    cats = [c for c in pa.columns if c.kind == "category" and c.index != pa.key.index]
                    if cats:
                        packages.append(dict(meta, package_id=_pid(ta["table_id"], tb["table_id"], "jf"),
                                             kind="join_filter", tables=[ta, tb], filter_cols=[c.header for c in cats]))
    # cross-season snapshots: same series, consecutive editions, same key header, shared metrics, overlapping keys
    for series, items in season_tables.items():
        items.sort(key=lambda x: x[0])
        years = sorted({y for y, _, _ in items})
        for y1, y2 in zip(years, years[1:]):
            for (_, ta, meta_a), (_, tb, meta_b) in itertools.product(
                    [i for i in items if i[0] == y1], [i for i in items if i[0] == y2]):
                pa, pb = profile_table(ta), profile_table(tb)
                if pa.key.header != pb.key.header or not (_metric_headers(pa) & _metric_headers(pb)):
                    continue
                ov = _overlap(pa, pb)
                if ov < 0.5 or (pa.rank() is None) != (pb.rank() is None):
                    continue
                t1, t2 = dict(ta, period=y1), dict(tb, period=y2)
                packages.append({"package_id": _pid(ta["table_id"], tb["table_id"], "pp"), "kind": "period_pair",
                                 "series": series, "domain": meta_a["domain"],
                                 "page_title": f"{meta_a['page_title']}；{meta_b['page_title']}",
                                 "scope": re.sub(r"^(?:(19|20)\d{2}年至)?(19|20)\d{2}(年|–\d{2}赛季)", "", meta_b["scope"]).strip() or meta_b["scope"],
                                 "tables": [t1, t2], "periods": [y1, y2], "period_source": "pages"})
    return _dedup(packages)


def _avg_len(prof: TableProfile, col) -> float:
    vals = [entity_key(prof.cell(r, col)) for r in range(len(prof.rows))]
    vals = [v for v in vals if v]
    return sum(len(v) for v in vals) / len(vals) if vals else 0.0


def _dedup(packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for p in packages:
        if p["package_id"] in seen:
            continue
        seen.add(p["package_id"])
        out.append(p)
    return out
