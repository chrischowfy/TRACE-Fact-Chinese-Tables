"""End-to-end build: packages -> drafts -> execution + quality gates -> realization -> sampling -> records.

Quality gates (a draft is kept only if all pass; each outcome is written to ``quality_flags``):
  label_by_execution    executing the program yields exactly the intended label
  tie_free / numeric    enforced by the executor (InvalidProgram -> dropped)
  table_necessity       multi-table groups: the program reads >=2 tables and removing any read table
                        makes it non-executable; single-table group: exactly one table read
  nei_absence           NEI: every substituted string occurs nowhere in the package (titles, headers, cells)
  nei_not_decidable     NEI: the program fails at a missing binding, not earlier/later by accident
  surface_clean         no placeholder / missingness wording, no Latin-heavy or overlong claim
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any

from .executor import InvalidProgram, execute
from .families import BUILDERS, Context, Draft
from .packages import build_packages
from .realize import realize
from .tables import entity_key, has_cjk, join_key, profile_table

LEAK_RE = re.compile(r"某|缺失|不存在|无法判断|证据不足|未知|NEI|null|表格|表\d|列\d")
MULTI_TABLE_GROUPS = {"Two-entity cross-table comparison", "Category decomposition", "Real-join rank–bridge",
                      "Hub-profile (multi-table)", "Cross-period growth", "Join-filter aggregation"}


def _profiles(pkg: dict[str, Any]) -> dict[str, Any]:
    return {t["table_id"]: profile_table(t) for t in pkg["tables"]}


def check_draft(d: Draft, pkg: dict[str, Any], profiles: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    try:
        run = execute(d.operators, profiles)
    except InvalidProgram as exc:
        return None, f"invalid_program:{exc}"
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"program_error:{type(exc).__name__}"
    if run["label"] != d.label:
        return None, f"label_mismatch:{d.label}->{run['label']}"
    flags: dict[str, Any] = {"label_by_execution": True, "tie_free": True, "strict_numeric": True}
    if d.label != "NEI":
        used = run["tables_used"]
        if d.group in MULTI_TABLE_GROUPS:
            if len(used) < 2:
                return None, "multi_table_not_used"
            for tid in used:
                reduced = {k: v for k, v in profiles.items() if k != tid}
                try:
                    ablated = execute(d.operators, reduced)
                except InvalidProgram:
                    ablated = {"label": "NEI"}
                if ablated["label"] != "NEI":
                    return None, "table_not_necessary"
            flags["table_necessity"] = f"all {len(used)} tables required"
        elif len(used) != 1:
            return None, "single_table_group_uses_multiple_tables"
    else:
        texts = [join_key(x) for x in _package_strings(pkg)]
        for s in d.absent:
            js = join_key(s)
            if any(js and (js in t or (len(t) >= 2 and t in js)) for t in texts):
                return None, "nei_absent_string_present"
        flags["nei_absence"] = "verified" if d.absent else "natural_missing_binding"
        flags["nei_missing_binding"] = run["missing"]
        flags["nei_axis"] = d.axis
    return run, ""


def _package_strings(pkg: dict[str, Any]) -> list[str]:
    out = []
    for t in pkg["tables"]:
        out.append(t.get("title", ""))
        out.extend(t["headers"])
        for row in t["rows"]:
            out.extend(row)
    return [entity_key(x) for x in out if x]


def _metric_on_page(item: dict[str, Any]) -> bool:
    from .families import _metric_norm
    from .packages import PAGE_HEADERS
    from .source_adapters import clean_name
    d = item["draft"]
    if d.label != "NEI" or d.axis != "metric":
        return False
    nm = _metric_norm(d.slots.get("metric_name", ""))
    for title in str(item["pkg"].get("page_title", "")).split("；"):
        for h in PAGE_HEADERS.get(title, []):
            hn = _metric_norm(clean_name(h))
            if nm and (nm == hn or (len(hn) >= 2 and (nm in hn or hn in nm))):
                return True
    return False


def _hp_metric_nei(item: dict[str, Any]) -> bool:
    """HP_AND metric-NEI has to state a value for a column the package does not carry.

    An invented number can land at a magnitude that reads as plainly false, which would make the
    instance refutable rather than unreadable, so only a value copied from a real row of that same
    metric elsewhere in the benchmark is admitted.
    """
    d = item["draft"]
    return d.skeleton == "HP_AND" and d.label == "NEI" and d.axis == "metric" and d.value_source != "corpus"


def _changing_attribute(item: dict[str, Any]) -> bool:
    """Attributes such as 主教练 'A→B' describe a change during the season, not one value."""
    s = item["draft"].slots
    if any(isinstance(s.get(k), str) and re.search(r"→|->|⇒|；", s[k]) for k in ("attr", "cat", "e", "e1", "e2", "claimed", "member")):
        return True
    return any(re.search(r"→|->|⇒", str(c.get("raw_value", ""))) for c in item["run"]["cells"])


def _single_table_shortcut(item: dict[str, Any]) -> bool:
    """Multi-table instance whose metric column (by normalized name) is also present in another package table
    holding the same entities: the claim would be decidable from one table, so it is not multi-table."""
    from .families import _metric_norm
    from .source_adapters import clean_name
    d = item["draft"]
    if d.group not in MULTI_TABLE_GROUPS or d.group in ("Two-entity cross-table comparison", "Cross-period growth"):
        return False
    metric_reads = {(op["table"], op["col"]) for op in d.operators if op["op"] in ("LOOKUP", "VALUES") and "col" in op}
    if not metric_reads:
        return False
    tables = {t["table_id"]: t for t in item["pkg"]["tables"]}
    profiles = {tid: profile_table(t) for tid, t in tables.items()}
    for tid, col in metric_reads:
        target = _metric_norm(clean_name(col))
        mine = {join_key(k) for k in profiles[tid].key_values()} if profiles[tid].key else set()
        for other_id, other in tables.items():
            if other_id == tid or not profiles[other_id].key or not mine:
                continue
            theirs = {join_key(k) for k in profiles[other_id].key_values()}
            same_entities = len(mine & theirs) / max(1, min(len(mine), len(theirs))) >= 0.5
            if same_entities and any(_metric_norm(clean_name(h)) == target for h in other["headers"]):
                return True
    return False


def _period_scope_mismatch(item: dict[str, Any]) -> bool:
    """Two snapshots must describe the same scope: no group slice against a full table, similar row counts."""
    pkg = item["pkg"]
    if pkg["kind"] != "period_pair":
        return False
    t1, t2 = pkg["tables"]
    if t1.get("is_slice") or t2.get("is_slice"):
        return True
    n1, n2 = len(t1["rows"]), len(t2["rows"])
    return min(n1, n2) / max(1, max(n1, n2)) < 0.7


def _slice_rank_or_category(item: dict[str, Any]) -> bool:
    """Ranks and category extremes over one group of a split table would silently change the claim's scope."""
    pkg = item["pkg"]
    if pkg["kind"] == "rank_detail":
        return bool(pkg["tables"][0].get("is_slice"))
    if pkg["kind"] == "category_decomp":
        return bool(pkg.get("is_slice"))
    return False


STANDINGS_HEADERS = ("胜", "负", "积分", "分", "胜率", "和", "平")


def _unscoped_split_pair(item: dict[str, Any]) -> bool:
    """Same-schema split tables in sport pages are stages or groups; only standings tables carry a group scope
    (小组赛/分组赛) in the claim. Player lists split by stage, and group tables of multi-sport events, are dropped."""
    pkg = item["pkg"]
    if pkg["kind"] != "split_pair" and not (pkg["kind"] == "single" and pkg.get("is_slice")):
        return False
    if pkg["domain"] == "medals":
        return True
    if pkg["domain"] not in ("football", "football_intl", "basketball"):
        return False
    headers = pkg["tables"][0]["headers"]
    return not any(h in STANDINGS_HEADERS for h in headers)


def _count_over_partial_list(item: dict[str, Any]) -> bool:
    from .source_adapters import PARTIAL_SERIES
    return (item["draft"].skeleton in ("CAT_COUNT", "JF_COUNT_GT", "ST_COUNT_GT", "JF_SUM", "CAT_SUM")
            and item["pkg"].get("series") in PARTIAL_SERIES)


COUNT_METRIC_RE = re.compile(r"金牌|银牌|铜牌|奖牌|胜|负|平|和|进球|失球|得|失|赛|场|次数|人数|数量|车站|层数|座")
NET_METRIC_RE = re.compile(r"净|差|变化|增|涨|跌")


def _negative_count(item: dict[str, Any]) -> bool:
    """A count-like metric column holding negative numbers is a change table (e.g. medal reallocations)."""
    from .tables import parse_number
    d = item["draft"]
    tables = {t["table_id"]: t for t in item["pkg"]["tables"]}
    for op in d.operators:
        col, tid = op.get("col"), op.get("table")
        if not col or tid not in tables or not COUNT_METRIC_RE.search(col) or NET_METRIC_RE.search(col):
            continue
        t = tables[tid]
        if col in t["headers"]:
            ci = t["headers"].index(col)
            if any((parse_number(r[ci]) or 0) < 0 for r in t["rows"] if ci < len(r)):
                return True
    return False


def _derivable_metric_nei(item: dict[str, Any]) -> bool:
    """Population is derivable from GDP and per-capita GDP; growth from two period columns."""
    d = item["draft"]
    if d.label != "NEI" or d.axis != "metric":
        return False
    name = d.slots.get("metric_name", "")
    headers = "|".join(h for t in item["pkg"]["tables"] for h in t["headers"])
    if "人口" in name and "GDP" in headers and "人均" in headers:
        return True
    if re.search(r"增长|增幅|增速", name) and len(re.findall(r"(19|20)\d{2}", headers)) >= 2:
        return True
    return False


def _rank_absent_entity(item: dict[str, Any]) -> bool:
    """'X ranks k-th' with X absent: rank k is held by another listed entity, so the claim reads as refutable."""
    d = item["draft"]
    return d.label == "NEI" and d.skeleton == "ST_RANK"


NON_MEMBER_UNIT_RE = re.compile(r"示范区|新区|开发区|管理区|管委会|林区|经济区|合作区")
COUNTY_RE = re.compile(r"县$|县级市$|直管")


def _non_member_units(item: dict[str, Any]) -> bool:
    """Extremes/ranks 'among all cities' are ill-defined when the table also lists zones that are not cities."""
    d = item["draft"]
    if d.skeleton not in ("ST_SUPERLATIVE", "ST_RANK", "CAT_ARGEXT", "JF_ARGEXT", "CAT_COUNT", "JF_COUNT_GT"):
        return False
    for t in item["pkg"]["tables"]:
        if any(NON_MEMBER_UNIT_RE.search(str(r[0]) if r else "") for r in t["rows"]):
            return True
        keys = profile_table(t).key_values()
        counties = sum(bool(COUNTY_RE.search(k)) for k in keys)
        if 0 < counties < len(keys):      # 省辖市 listed together with 省直管县
            return True
    return False


def _incomplete_metric_name(item: dict[str, Any]) -> bool:
    """Metric names whose measured quantity lives only in a parent header (人均 / 年增幅 without GDP)."""
    return item["draft"].slots.get("metric_name", "") in ("人均", "年增幅", "增幅", "比重", "年增幅实际", "年增幅名义")


AGG_SUFFIX_RE = re.compile(r"(全省|全市|全区|全国|全州|全盟|总计|合计)$")
EXTREME_SKELETONS = ("ST_SUPERLATIVE", "ST_RANK", "CAT_ARGEXT", "JF_ARGEXT", "CAT_COUNT", "JF_COUNT_GT")
RANK_HEADER_RE = re.compile(r"位次|排名|名次|^名$|^#$")


def _aggregate_row_in_ranking(item: dict[str, Any]) -> bool:
    """Suffix-form totals (河北全省) survive the aggregate filter: never name them, never rank against them."""
    d = item["draft"]
    if any(AGG_SUFFIX_RE.search(entity_key(str(d.slots[k]))) for k in ("e", "e1", "e2", "claimed", "member")
           if d.slots.get(k)):
        return True
    if d.skeleton not in EXTREME_SKELETONS:
        return False
    for t in item["pkg"]["tables"]:
        prof = profile_table(t)
        if prof.key and any(AGG_SUFFIX_RE.search(k) for k in prof.key_values()):
            return True
    return False


def _rank_column_mismatch(item: dict[str, Any]) -> bool:
    """The table's own rank column (位次/排名) must agree with the rank the program computes for the entity."""
    from .tables import parse_number
    d = item["draft"]
    if d.skeleton != "ST_RANK" or d.label == "NEI":
        return False
    computed = next((s["value"] for s in item["run"]["trace"] if s["op"] == "RANK"), None)
    t = item["pkg"]["tables"][0]
    prof = profile_table(t)
    row = prof.find_row(str(d.slots.get("e", "")))
    if row is None or computed is None:
        return False
    base = re.sub(r"[（(].*", "", str(d.slots.get("metric_col", ""))).strip()
    for c in prof.columns:
        if c.index == (prof.key.index if prof.key else -1) or not RANK_HEADER_RE.search(c.header):
            continue
        qual = re.sub(r"位次|排名|名次|#|[\s()（）]", "", c.header)
        if qual and base and qual not in base and base not in qual:
            continue                      # rank of another measure (位次 (人均))
        v = parse_number(prof.cell(row, c))
        if v is not None and int(v) != int(computed):
            return True
    return False


def _superlative_over_subset(item: dict[str, Any]) -> bool:
    """Sports standings with a group column or fewer than 8 teams (a group, the third-placed teams, a play-off)
    do not support 'most/least among all teams of the tournament'."""
    d = item["draft"]
    pkg = item["pkg"]
    if d.skeleton not in ("ST_SUPERLATIVE", "ST_RANK") or pkg["domain"] not in ("football", "football_intl", "basketball"):
        return False
    t = pkg["tables"][0]
    return len(t["rows"]) < 8 or any(re.search(r"组别|小组|分组", h) for h in t["headers"])


def _rank_bridge_detail_incomplete(item: dict[str, Any]) -> bool:
    """RJ_RANK_TOP needs a ranked leaders list (rank column starting at 1); a flat member list may omit a team's
    top member."""
    from .tables import parse_number
    if item["draft"].skeleton != "RJ_RANK_TOP":
        return False
    prof = profile_table(item["pkg"]["tables"][1])
    ranks = [c for c in prof.columns if c.kind == "rank" or RANK_HEADER_RE.search(c.header)]
    if not ranks:
        return True
    vals = [parse_number(prof.cell(r, ranks[0])) for r in range(len(prof.rows))]
    vals = [v for v in vals if v is not None]
    return not vals or min(vals) != 1


def _confusable_metric_nei(item: dict[str, Any]) -> bool:
    """A metric-NEI whose name contains a one-character header of the page (得分 vs 分, 失球 vs 失) is read as that
    column."""
    from .packages import PAGE_HEADERS
    d = item["draft"]
    if d.label != "NEI" or d.axis != "metric":
        return False
    name = str(d.slots.get("metric_name", ""))
    headers = {h.strip() for title in str(item["pkg"].get("page_title", "")).split("；") for h in PAGE_HEADERS.get(title, [])}
    headers |= {h.strip() for t in item["pkg"]["tables"] for h in t["headers"]}
    return any(len(h) == 1 and has_cjk(h) and h in name for h in headers)


def _join_filter_open_set(item: dict[str, Any]) -> bool:
    """'Among all X with attribute A' is only closed if every entity of the metric table is listed in the attribute
    table; otherwise an unlisted entity (锦州港) may hold A and change the extreme or the count."""
    d = item["draft"]
    if d.skeleton not in ("JF_ARGEXT", "JF_COUNT_GT"):
        return False
    tabs = {op.get("table") for op in d.operators if op.get("table")}
    attr_tid = next((op["table"] for op in d.operators if op["op"] == "FILTER"), None)
    profs = {t["table_id"]: profile_table(t) for t in item["pkg"]["tables"] if t["table_id"] in tabs}
    if attr_tid not in profs:
        return False
    attr_keys = {join_key(k) for k in profs[attr_tid].key_values()}
    for tid, prof in profs.items():
        if tid != attr_tid and prof.key and not {join_key(k) for k in prof.key_values()} <= attr_keys:
            return True
    return False


def _duplicate_members(item: dict[str, Any]) -> bool:
    """Counts and extremes over a member list that repeats an entity (朝鲜 listed twice) are ambiguous."""
    if item["draft"].skeleton not in ("CAT_COUNT", "CAT_ARGEXT", "JF_COUNT_GT", "JF_ARGEXT"):
        return False
    for t in item["pkg"]["tables"]:
        keys = profile_table(t).key_values()
        if len(keys) != len(set(keys)):
            return True
    return False


SPORTS_DOMAINS = ("football", "football_intl", "basketball")
ADMIN_SUFFIX_RE = re.compile(r"(特别行政区|自治州|自治区|自治县|地区|新区|市|县|区|省|州|盟)$")


def _sports_entity_nei(item: dict[str, Any]) -> bool:
    """League and tournament tables list every participant, so an absent team reads as 'did not take part' (0 wins),
    and club names change with sponsors (沈阳飞豹 = 辽宁衡业飞豹): no entity-NEI in sports tables, as for medals."""
    d = item["draft"]
    return d.label == "NEI" and d.axis == "entity" and item["pkg"]["domain"] in SPORTS_DOMAINS


MEMBERSHIP_SCOPE_RE = re.compile(
    r"非洲|亚洲|欧洲|北美洲|南美洲|大洋洲|拉丁美洲|中美洲|加勒比|^洲$|洲国家|"
    r"语国家|阿拉伯世界|英联邦|欧盟|东盟|独立国家联合体|不结盟|经济合作|成员国|加盟")


def _membership_scoped_entity_nei(item: dict[str, Any]) -> bool:
    """A page whose entity set is a membership class (the countries of Africa, the Portuguese-speaking
    countries, the EU member states) lists that class in full, so an entity absent from it is absent
    because it is not a member.  'X belongs to East Africa and has population N' is then false in the
    world rather than unreadable - the same reason league tables get no entity-NEI."""
    d = item["draft"]
    if d.label != "NEI" or d.axis != "entity":
        return False
    text = str(item["pkg"].get("page_title", "")) + " " + str(item["pkg"].get("scope", ""))
    return bool(MEMBERSHIP_SCOPE_RE.search(text))


def _admin_suffix_alias(item: dict[str, Any]) -> bool:
    """阿勒泰地区 vs 阿勒泰市: the same place name with another administrative suffix is not an absent entity."""
    from .packages import PAGE_TEXT
    d = item["draft"]
    if d.label != "NEI" or d.axis != "entity" or not d.absent:
        return False
    stem = ADMIN_SUFFIX_RE.sub("", entity_key(d.absent[0]))
    if len(stem) < 2:
        return False
    page = [x for title in str(item["pkg"].get("page_title", "")).split("；") for x in PAGE_TEXT.get(title, [])]
    page += _package_strings(item["pkg"])
    return any(ADMIN_SUFFIX_RE.sub("", x) == stem for x in page)


def _nei_below_list_cutoff(item: dict[str, Any]) -> bool:
    """Top-N and threshold lists (largest ports, skyscrapers, grossing films) ranked by the compared metric: an absent
    entity is below the cut-off, so 'listed X is higher than absent Y' is inferable (佛山港 is not in the top 20)."""
    from .source_adapters import PARTIAL_SERIES
    from .tables import parse_number
    d = item["draft"]
    if d.label != "NEI" or d.axis != "entity" or item["pkg"].get("series") not in PARTIAL_SERIES:
        return False
    tables = {t["table_id"]: t for t in item["pkg"]["tables"]}
    for op in d.operators:
        if op.get("table") not in tables or not op.get("col"):
            continue
        prof = profile_table(tables[op["table"]])
        col = prof.column(op["col"])
        if col is None:
            continue
        vals = [parse_number(prof.cell(r, col)) for r in range(len(prof.rows))]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 5 and (all(a >= b for a, b in zip(vals, vals[1:])) or all(a <= b for a, b in zip(vals, vals[1:]))):
            return True
    return False


def _extreme_over_sibling_table(item: dict[str, Any]) -> bool:
    """'Largest among the page's X' over one of several tables of the same kind (the Korean table of an East-Asian city
    list): same key header and same metric (ignoring year/unit qualifiers) in another table of the page."""
    from .packages import PAGE_TABLE_SHAPES
    d, pkg = item["draft"], item["pkg"]
    if d.skeleton not in ("ST_SUPERLATIVE", "ST_RANK") or pkg["kind"] != "single":
        return False
    prof = profile_table(pkg["tables"][0])
    if not prof.key:
        return False
    base = lambda h: re.sub(r"[（(].*$|\s|\d{4}年?|_\d+$", "", str(h))  # noqa: E731
    metric = base(d.slots.get("metric_col", ""))
    same = sum(1 for key, headers in PAGE_TABLE_SHAPES.get(pkg["page_title"], [])
               if key == prof.key.header and any(base(h) == metric for h in headers))
    return same >= 2


def _ranked_list_continuation(item: dict[str, Any]) -> bool:
    """A table whose running number starts after 1 (深圳摩天大楼 表2 starts at 29) continues a list split elsewhere."""
    from .tables import parse_number
    if item["draft"].skeleton not in EXTREME_SKELETONS:
        return False
    for t in item["pkg"]["tables"]:
        for ci in range(len(t["headers"])):
            vals = [parse_number(r[ci]) if ci < len(r) else None for r in t["rows"]]
            ints = [int(v) for v in vals if v is not None and float(v).is_integer()]
            if (len(ints) >= 5 and len(ints) >= 0.8 * len(t["rows"]) and 1 < ints[0] < 1000 and ints[-1] > ints[0]
                    and all(0 <= y - x <= 1 for x, y in zip(ints, ints[1:]))):
                return True
    return False


def _prc_region_set_unclear(item: dict[str, Any]) -> bool:
    """Extremes and ranks over 中华人民共和国 tables that list 香港/澳门 but not 台湾 have no scope that states the
    compared set exactly; they are dropped (tables without 港澳台 rows are restated as 中国大陆 instead)."""
    d, pkg = item["draft"], item["pkg"]
    if d.skeleton not in EXTREME_SKELETONS or not str(d.slots.get("scope", "")).startswith("中华人民共和国"):
        return False
    cells = " ".join(str(c) for t in pkg["tables"] for r in t["rows"] for c in r)
    return bool(re.search(r"香港|澳门", cells)) and "台湾" not in cells


def _join_key_value_readable(item: dict[str, Any]) -> bool:
    """A join-key NEI claims a metric for an entity that has no row in the partner table.  It is only
    undecidable if no *other* table of the same page states that value: for a hub-profile instance the
    entity must not key a row in another table carrying the metric, and for a rank-bridge instance the
    metric must appear on exactly one table of the page."""
    from .packages import PAGE_KEY_SETS
    d = item["draft"]
    if d.label != "NEI" or d.axis != "join_key":
        return False
    col = d.slots.get("metric_col")
    if not col:
        return True
    tables = PAGE_KEY_SETS.get(str(item["pkg"].get("page_title", "")), [])
    if d.skeleton == "HP_AND":
        # the entity has no row in the profile table; drop it if it keys a row in any other table of the
        # page that carries the same metric, because then the value is on the page after all
        entity = join_key(str(d.slots.get("e", "")))
        return any(col in headers and entity in keys for keys, headers in tables)
    # rank-bridge: the team ranked k has no member row in the detail table.  The value is recoverable only
    # from another per-member table that carries both the metric and the same foreign key to the ranking.
    fk = str(item["pkg"].get("fk_col", ""))
    return sum(1 for _keys, headers in tables if col in headers and fk in headers) > 1


OUTSIDE_MAINLAND_RE = re.compile(r"香港|澳门|台湾")
SEASON_PAGE_RE = re.compile(r"(\d{4})年至(\d{4})年")


def _restate_scope_and_periods(item: dict[str, Any]) -> bool:
    """Text-only corrections applied after sampling: tables of 中华人民共和国 pages that list no 港澳台 rows cover
    中国大陆; periods taken from season pages are seasons (2012至2013赛季), not calendar years."""
    d, pkg = item["draft"], item["pkg"]
    changed = False
    sc = str(d.slots.get("scope", ""))
    if sc.startswith("中华人民共和国") and not any(OUTSIDE_MAINLAND_RE.search(str(c)) for t in pkg["tables"]
                                                  for r in t["rows"] for c in r):
        d.slots["scope"] = "中国大陆" + sc[len("中华人民共和国"):]
        changed = True
    if d.skeleton in ("CP_DIRECTION", "CP_DIFF") and pkg.get("period_source") == "pages":
        seasons = [SEASON_PAGE_RE.search(t) for t in str(pkg["page_title"]).split("；")]
        if seasons and all(seasons):
            labels = {int(m.group(1)): f"{m.group(1)}至{m.group(2)}赛季" for m in seasons}
            for k in ("y1", "y2"):
                if d.slots.get(k) in labels:
                    d.slots[k] = labels[d.slots[k]]
                    changed = True
    if changed:
        item["claim"] = realize(d.skeleton, d.slots, d.base_id)
    return changed


def topology(n_tables: int) -> str:
    return {1: "single", 2: "two_table"}.get(n_tables, "three_plus_table")


# Gates applied to every instance after it has been realized.  They are stable: whether an instance
# passes depends only on the instance, never on which other instances were drawn, so they can be run
# either before or after selection without changing the verdict on any instance.
POST_GATES: list[tuple[str, Any]] = [
    ("nei_metric_present_on_page", _metric_on_page),
    ("hub_profile_metric_nei_implausible_value", _hp_metric_nei),
    ("attribute_value_with_change_marker", _changing_attribute),
    ("multi_table_single_table_shortcut", _single_table_shortcut),
    ("period_pair_scope_mismatch", _period_scope_mismatch),
    ("slice_scope_for_rank_or_category", _slice_rank_or_category),
    ("split_pair_unscoped_stage_tables", _unscoped_split_pair),
    ("count_over_partial_list", _count_over_partial_list),
    ("negative_value_in_count_metric", _negative_count),
    ("nei_metric_derivable_from_package", _derivable_metric_nei),
    ("nei_rank_of_absent_entity", _rank_absent_entity),
    ("non_member_unit_in_extreme_or_rank", _non_member_units),
    ("incomplete_metric_name", _incomplete_metric_name),
    ("aggregate_row_in_ranking", _aggregate_row_in_ranking),
    ("rank_differs_from_table_rank_column", _rank_column_mismatch),
    ("superlative_over_subset_table", _superlative_over_subset),
    ("rank_bridge_detail_not_top_list", _rank_bridge_detail_incomplete),
    ("nei_metric_confusable_with_header", _confusable_metric_nei),
    ("join_filter_open_entity_set", _join_filter_open_set),
    ("category_count_with_duplicate_members", _duplicate_members),
    ("nei_entity_in_sports_table", _sports_entity_nei),
    ("nei_entity_admin_suffix_variant", _admin_suffix_alias),
    ("nei_entity_in_membership_scoped_table", _membership_scoped_entity_nei),
    ("nei_entity_below_ranked_list_cutoff", _nei_below_list_cutoff),
    ("extreme_over_one_of_several_tables", _extreme_over_sibling_table),
    ("extreme_over_ranked_list_continuation", _ranked_list_continuation),
    ("extreme_over_prc_table_with_partial_region_set", _prc_region_set_unclear),
    ("join_key_nei_value_readable_on_page", _join_key_value_readable),
]


def build(cache_dir: str, out_dir: str, *, seed: int = 20260913, max_per_package: int = 9,
          max_per_page: int = 36, nei_share: float = 0.22, cache_only: bool = False,
          naturalize_api: bool = False, naturalize_limit: int | None = None,
          quota: dict[str, dict[str, int]] | None = None,
          domain_cap: dict[str, int] | None = None,
          metric_nei_budget: int = 0, four_table_target: int = 0,
          table_target: int = 0) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from .resolve import load as load_redirects
    load_redirects(Path(cache_dir).parent / "redirects.json")
    packages = build_packages(cache_dir, cache_only=cache_only)
    (out / "packages.jsonl").write_text("\n".join(json.dumps({k: v for k, v in p.items() if k != "tables"} |
                                                             {"table_ids": [t["table_id"] for t in p["tables"]]},
                                                             ensure_ascii=False) for p in packages), encoding="utf-8")
    two_pass: dict[str, Any] = {}

    def generate(ctx: Context) -> tuple[list[dict[str, Any]], collections.Counter]:
        rejects: collections.Counter = collections.Counter()
        kept: list[dict[str, Any]] = []
        for pkg in packages:
            try:
                drafts = BUILDERS[pkg["kind"]](pkg, ctx)
            except Exception as exc:  # noqa: BLE001 - a broken package must not stop the build
                rejects[f"builder_error:{pkg['kind']}:{type(exc).__name__}"] += 1
                continue
            profiles = _profiles(pkg)
            for d in drafts:
                run, reason = check_draft(d, pkg, profiles)
                if run is None:
                    rejects[reason.split(":")[0] + ":" + d.skeleton] += 1
                    continue
                claim = realize(d.skeleton, d.slots, d.base_id)
                if LEAK_RE.search(claim) or len(claim) > 90 or len(re.findall(r"[A-Za-z]", claim)) > 6:
                    rejects[f"surface:{d.skeleton}"] += 1
                    continue
                kept.append({"draft": d, "run": run, "claim": claim, "pkg": pkg})
        return kept, rejects

    def select(kept: list[dict[str, Any]], rejects: collections.Counter):
        if quota is None:
            records = sample(kept, seed=seed, max_per_package=max_per_package, max_per_page=max_per_page,
                             nei_share=nei_share)
            for name, test in POST_GATES:
                before = len(records)
                records = [item for item in records if not test(item)]
                rejects[name] = before - len(records)
            return records, {}
        # quota mode: gate the whole candidate pool first, so that selecting an exact per-cell count
        # cannot be undone by a later gate.  The gates are stable, so this yields the same verdicts.
        pool = kept
        for name, test in POST_GATES:
            before = len(pool)
            pool = [item for item in pool if not test(item)]
            rejects[name] = before - len(pool)
        return sample_to_quota(pool, quota, seed=seed, max_per_package=max_per_package,
                               max_per_page=max_per_page, domain_cap=domain_cap,
                               four_table_target=four_table_target, table_target=table_target)

    ctx = Context(packages, seed=seed)
    kept, rejects = generate(ctx)
    records, shortfall = select(kept, rejects)

    # Second pass, to keep the metric word out of the label.  A metric-slot NEI names a column the tables
    # do not carry; if that word occurs in no decidable claim either, its mere presence marks the instance
    # as NEI and a classifier fitted on the claims alone can read the label off the surface.  Pass one
    # fixes which metric words the released SUPPORTS/REFUTES claims actually use; pass two rebuilds with
    # the borrowing restricted to exactly those words.  Kept only if it does not cost coverage.
    if quota is not None:
        attested = {str(item["draft"].slots.get(k, "")) for item in records if item["draft"].label != "NEI"
                    for k in ("metric_name", "metric_col")}
        attested.discard("")
        ent_attested = {str(item["draft"].slots.get(k, "")) for item in records if item["draft"].label != "NEI"
                        for k in ("e", "e1", "e2", "claimed", "member")}
        ent_attested.discard("")
        if attested:
            ctx2 = Context(packages, seed=seed)
            ctx2.attested_metrics = attested
            ctx2.attested_entities = ent_attested
            kept2, rejects2 = generate(ctx2)
            records2, shortfall2 = select(kept2, rejects2)
            two_pass = {"attested_words": len(attested),
                        "attested_entities": len(ent_attested),
                        "shortfall_pass1": sum(shortfall.values()),
                        "shortfall_pass2": sum(shortfall2.values())}
            # a few missing instances are worth far more than a metric word that marks its own label
            if sum(shortfall2.values()) - sum(shortfall.values()) <= metric_nei_budget:
                kept, rejects, records, shortfall = kept2, rejects2, records2, shortfall2
                two_pass["used"] = True
            else:
                two_pass["used"] = False
    rejects["restated_scope_or_season"] = sum(_restate_scope_and_periods(item) for item in records)
    # slot-locked LLM paraphrase of the surviving claims (labels, slots and instance set are already fixed)
    from .naturalize import naturalize
    nat = naturalize(records, Path(cache_dir).parent / "naturalize.json", call_api=naturalize_api,
                     limit=naturalize_limit)
    # paraphrases can merge claims that differed only in template wording (e.g. GDP in local currency vs USD):
    # keep the first of identical surfaces; drop all of them if their labels disagree
    by_claim: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in records:
        by_claim[item["claim"]].append(item)
    before = len(records)
    records = [item for item in records
               if len({x["draft"].label for x in by_claim[item["claim"]]}) == 1 and by_claim[item["claim"]][0] is item]
    rejects["duplicate_surface_after_paraphrase"] = before - len(records)
    rows = [to_record(i, item) for i, item in enumerate(records)]
    with (out / "claims.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "packages": len(packages), "packages_by_kind": dict(collections.Counter(p["kind"] for p in packages)),
        "drafts_kept_before_sampling": len(kept), "rejects": dict(rejects.most_common()),
        "claims": len(rows), "labels": dict(collections.Counter(r["label"] for r in rows)),
        "naturalization": nat,
        "surface": dict(collections.Counter(r["surface"]["kind"] for r in rows)),
        "by_group": {g: dict(collections.Counter(r["label"] for r in rows if r["program"]["category"] == g))
                     for g in sorted({r["program"]["category"] for r in rows})},
        "quota_shortfall": shortfall,
        "metric_nei_two_pass": two_pass,
    }
    (out / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


DOMAIN_SHARE_CAP = {"football": 0.20, "football_intl": 0.12, "basketball": 0.10, "medals": 0.12}
DEFAULT_DOMAIN_SHARE_CAP = 0.25


def sample(kept: list[dict[str, Any]], *, seed: int, max_per_package: int, max_per_page: int,
           nei_share: float, target: int = 2400) -> list[dict[str, Any]]:
    """Choose sibling sets (S/R/N of one fact stay together).

    Groups are visited round-robin starting from the scarcest, so multi-table families are not crowded out by
    single-table ones; per-package, per-page and per-domain caps limit topical concentration; duplicate
    surfaces are dropped; finally NEI members are thinned to the target share.
    """
    rng = random.Random(seed)
    by_base: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in kept:
        by_base[item["draft"].base_id + item["pkg"]["package_id"]].append(item)
    by_group: dict[str, list[list[dict[str, Any]]]] = collections.defaultdict(list)
    for sibs in by_base.values():
        by_group[sibs[0]["draft"].group].append(sibs)
    for g in by_group:
        rng.shuffle(by_group[g])
    order = sorted(by_group, key=lambda g: len(by_group[g]))
    caps = {d: int(share * target) for d, share in DOMAIN_SHARE_CAP.items()}
    chosen, seen_claims = [], set()
    per_pkg: collections.Counter = collections.Counter()
    per_page: collections.Counter = collections.Counter()
    per_domain: collections.Counter = collections.Counter()
    cursors = {g: 0 for g in order}
    progress = True
    while progress and len(chosen) < target:
        progress = False
        for g in order:
            while cursors[g] < len(by_group[g]):
                sibs = by_group[g][cursors[g]]
                cursors[g] += 1
                pkg = sibs[0]["pkg"]
                sibs = [x for x in sibs if x["claim"] not in seen_claims]
                dom_cap = caps.get(pkg["domain"], int(DEFAULT_DOMAIN_SHARE_CAP * target))
                if (not sibs or per_pkg[pkg["package_id"]] + len(sibs) > max_per_package
                        or per_page[pkg["page_title"]] + len(sibs) > max_per_page
                        or per_domain[pkg["domain"]] + len(sibs) > dom_cap):
                    continue
                chosen.extend(sibs)
                seen_claims.update(x["claim"] for x in sibs)
                per_pkg[pkg["package_id"]] += len(sibs)
                per_page[pkg["page_title"]] += len(sibs)
                per_domain[pkg["domain"]] += len(sibs)
                progress = True
                break
    nei = [c for c in chosen if c["draft"].label == "NEI"]
    dec = [c for c in chosen if c["draft"].label != "NEI"]
    cap = int(nei_share / (1 - nei_share) * len(dec))
    if len(nei) > cap:
        rng.shuffle(nei)
        nei = nei[:cap]
    final = dec + nei
    final.sort(key=lambda c: (c["pkg"]["page_title"], c["draft"].base_id, c["draft"].label))
    return final


def _round_robin_by_page(groups: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Reorder sibling groups so consecutive picks come from different source pages.

    Without this the sampler drains whichever pages happen to be listed first - in practice the sports
    seasons, which are by far the most productive - and fills its quota before ever touching the rest of
    the corpus.  Cycling across pages spreads the draw over topics and keeps any single domain from
    dominating a family."""
    by_page: dict[str, list[list[dict[str, Any]]]] = collections.defaultdict(list)
    for g in groups:
        by_page[g[0]["pkg"]["page_title"]].append(g)
    order = sorted(by_page, key=lambda p: (-len(by_page[p]), p))
    out: list[list[dict[str, Any]]] = []
    while any(by_page[p] for p in order):
        for p in order:
            if by_page[p]:
                out.append(by_page[p].pop())
    return out


def sample_to_quota(kept: list[dict[str, Any]], quota: dict[str, dict[str, int]], *, seed: int,
                    max_per_package: int, max_per_page: int,
                    domain_cap: dict[str, int] | None = None,
                    four_table_target: int = 0,
                    table_target: int = 0) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select instances so that each (reasoning family, label) cell hits an exact target count.

    `sample()` fills a single global target and lets the family mix fall out of availability, which makes
    the mix an accident of how many candidates each builder happens to produce.  Reproducing a published
    composition needs the opposite: the per-cell counts are given, and sampling must hit them.

    Sibling sets (the SUPPORTS/REFUTES/NEI variants of one grounded fact) are still visited together so a
    contrast set is taken as a unit wherever both of its cells are still open; tight cells are served first
    so an abundant family cannot exhaust a page budget that a scarce one needs.  Returns the selection and
    the per-cell shortfall.
    """
    rng = random.Random(seed)
    by_base: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in kept:
        by_base[item["draft"].base_id + item["pkg"]["package_id"]].append(item)
    by_family: dict[str, list[list[dict[str, Any]]]] = collections.defaultdict(list)
    for sibs in by_base.values():
        by_family[sibs[0]["draft"].group].append(sibs)
    for fam in by_family:
        rng.shuffle(by_family[fam])
        by_family[fam] = _round_robin_by_page(by_family[fam])
    # The published composition fixes the table-count distribution as well as the family mix, and the
    # four-table topology exists on only a couple of pages.  Visiting those sibling sets first - but only
    # as many as the target needs - keeps the rest of the family unbiased.
    front_loaded: set[int] = set()
    if four_table_target > 0:
        # largest sibling sets first, so the greedy fill lands on the target rather than under it: the
        # sampler may still decline a member whose cell is already full, hence "about", not "exactly"
        budget = four_table_target
        picked: set[int] = set()
        four = sorted(((fam, sibs) for fam, groups in by_family.items() for sibs in groups
                       if len(sibs[0]["pkg"]["tables"]) >= 4), key=lambda x: -len(x[1]))
        for _fam, sibs in four:
            if budget <= 0:
                break
            if len(sibs) <= budget:
                picked.add(id(sibs))
                budget -= len(sibs)
        for fam, groups in by_family.items():
            by_family[fam] = ([g for g in groups if id(g) in picked]
                              + [g for g in groups if id(g) not in picked])
        front_loaded = picked

    need = {fam: dict(cells) for fam, cells in quota.items()}
    supply = {fam: collections.Counter(s["draft"].label for sibs in groups for s in sibs)
              for fam, groups in by_family.items()}
    # Tightest family first: the one whose candidate pool is smallest relative to what it must deliver.
    # Counting candidates alone misses the constraint that actually binds, which is per-page capacity:
    # two-entity lives on 25 pages and single-table on 219, so by raw counts single looks tighter, gets
    # served first, and spends the page budget on the handful of non-football pages two-entity depends
    # on.  Two-entity then falls through to the enforce_domain=False pass where the football cap no
    # longer applies - the single largest source of the football over-representation.
    def tightness(fam: str) -> float:
        want = sum(need.get(fam, {}).values())
        if not want:
            return float("inf")
        have = sum(supply.get(fam, {}).values())
        pages = len({g[0]["pkg"]["page_title"] for g in by_family.get(fam, ())})
        return min(have / want, pages * max_per_page / want)

    # A domain cap is a budget, and whichever family is served first spends it - even a family with
    # thousands of alternatives.  A family that can fill its whole quota from uncapped domains therefore
    # leaves the capped ones alone on the first pass, so the budget stays with the families that have no
    # alternative (the rank-bridge shape exists almost only on football pages).
    capped = set(domain_cap or {})
    self_sufficient = {
        fam: sum(len(g) for g in groups if g[0]["pkg"]["domain"] not in capped) >= 1.2 * sum(quota.get(fam, {}).values())
        for fam, groups in by_family.items()}

    chosen: list[dict[str, Any]] = []
    seen_claims: set[str] = set()
    per_pkg: collections.Counter = collections.Counter()
    per_page: collections.Counter = collections.Counter()
    per_domain: collections.Counter = collections.Counter()
    caps = domain_cap or {}
    used_pages: set[str] = set()
    used_tables: set[str] = set()
    used_pkgs: set[str] = set()

    def coverage_tier(pkg: dict[str, Any], sibs: list[dict[str, Any]]) -> int:
        """How much *new* source material this package brings.  Sweeping the tiers in order makes a
        family drain fresh topics first, then fresh tables, and take a package that adds neither only
        last - which is what the released build was short of (223 topics of 257 reachable, 468 raw
        tables of ~790).  A package that adds no new table is worse than re-opening one already used,
        because it costs an evidence package for nothing.

        The four-table front-loading above works by reordering `by_family`, which a tier sweep would
        undo - the ordering only matters inside a tier.  Those sibling sets therefore stay in the first
        tier whatever else they cover, or the exact four-table target silently drops."""
        if id(sibs) in front_loaded:
            return 0
        if pkg["page_title"] not in used_pages:
            base = 0
        elif (not table_target or len(used_tables) < table_target) and \
                any(t["table_id"] not in used_tables for t in pkg["tables"]):
            base = 1
        else:
            base = 2 if pkg["package_id"] in used_pkgs else 3
        # a capped domain is the last resort inside each tier, not interleaved with the rest: a family
        # with plenty of uncapped candidates would otherwise still spend capped budget on a fresh page
        return base * 2 + (1 if pkg["domain"] in capped else 0)

    for fam in sorted(need, key=tightness):
        # Three passes over the domain caps.  The first honours them; without a later pass a cap on an
        # abundant domain would leave a cell short even though candidates exist.  But the relief pass
        # used to drop the cap entirely, which is how the capped domain ended up 38 claims over target:
        # the pass exists to fill cells the cap starved, not to abandon the cap.  So relax it by half
        # first, and only then lift it, which still guarantees no shortfall.
        for slack, tier in itertools.product((1.0, 1.5, None), range(8)):
            enforce_domain = slack is not None
            for sibs in by_family.get(fam, []):
                if not any(need[fam].get(s["draft"].label, 0) > 0 for s in sibs):
                    continue
                pkg = sibs[0]["pkg"]
                # the tier partitions the candidates of this pass rather than filtering any out: every
                # candidate is still visited before the pass ends, so a cell with no surplus behaves
                # exactly as it did before and the tight NEI cells are unaffected
                if coverage_tier(pkg, sibs) != tier:
                    continue
                # a sibling set can hold two variants of one label (an entity-NEI and a metric-NEI of the
                # same base); count them against the cell as they are taken, or the cell overshoots
                room = dict(need[fam])
                take = []
                for s in sibs:
                    lab = s["draft"].label
                    if room.get(lab, 0) > 0 and s["claim"] not in seen_claims:
                        take.append(s)
                        room[lab] -= 1
                if not take:
                    continue
                if (per_pkg[pkg["package_id"]] + len(take) > max_per_package
                        or per_page[pkg["page_title"]] + len(take) > max_per_page):
                    continue
                if enforce_domain and pkg["domain"] in caps and \
                        (self_sufficient.get(fam) or
                         per_domain[pkg["domain"]] + len(take) > int(caps[pkg["domain"]] * slack)):
                    continue
                chosen.extend(take)
                seen_claims.update(s["claim"] for s in take)
                per_pkg[pkg["package_id"]] += len(take)
                per_page[pkg["page_title"]] += len(take)
                per_domain[pkg["domain"]] += len(take)
                used_pages.add(pkg["page_title"])
                used_pkgs.add(pkg["package_id"])
                used_tables.update(t["table_id"] for t in pkg["tables"])
                for s in take:
                    need[fam][s["draft"].label] -= 1
                if not any(v > 0 for v in need[fam].values()):
                    break
            if not any(v > 0 for v in need[fam].values()):
                break

    shortfall = {f"{fam}/{lab}": n for fam, cells in need.items() for lab, n in cells.items() if n > 0}
    chosen.sort(key=lambda c: (c["pkg"]["page_title"], c["draft"].base_id, c["draft"].label))
    return chosen, shortfall


def to_record(idx: int, item: dict[str, Any]) -> dict[str, Any]:
    d, run, pkg = item["draft"], item["run"], item["pkg"]
    used = run["tables_used"] if d.label != "NEI" else sorted({op["table"] for op in d.operators if "table" in op})
    tables = [t for t in pkg["tables"]]
    return {
        "id": f"zhtf-{idx + 1:05d}",
        "claim": item["claim"],
        "label": d.label,
        "surface": {"kind": item.get("surface", "template"), "template_claim": item.get("template_claim", item["claim"])},
        "evidence_package_id": pkg["package_id"],
        "tables": [{"table_id": t["table_id"], "title": t.get("title"), "headers": t["headers"], "rows": t["rows"],
                    **({"period": t["period"]} if "period" in t else {}), "source": t["source"]} for t in tables],
        "evidence_cells": run["cells"] if d.label != "NEI" else [],
        "context_cells": run["cells"] if d.label == "NEI" else [],
        "program": {"category": d.group, "skeleton_id": d.skeleton, "operators": d.operators,
                    "table_ids": used, "slots": _slots_public(d.slots)},
        "table_topology": topology(len(used)),
        "package_kind": pkg["kind"],
        "quality_flags": {**{k: v for k, v in _flags(d, run).items()}},
        "domain": pkg["domain"], "series": pkg["series"], "topic": pkg["page_title"],
        "license": "CC BY-SA 4.0",
    }


def _flags(d: Draft, run: dict[str, Any]) -> dict[str, Any]:
    flags = {"label_by_execution": True, "tie_free": True, "strict_numeric_cells": True,
             "perturbation_axis": d.axis or None}
    if d.label == "NEI":
        flags["nei_missing_binding"] = run["missing"]
        flags["nei_absence_verified"] = bool(d.absent) or "natural"
    else:
        flags["tables_read"] = len(run["tables_used"])
    return flags


def _slots_public(slots: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in slots.items() if k not in ("numeric_surface_ok",)}
