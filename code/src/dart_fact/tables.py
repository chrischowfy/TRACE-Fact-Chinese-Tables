"""Cell normalization and column typing.

Design rules that come straight out of the audit of the review-time release (audit/AUDIT_REPORT.md):
  * a cell is numeric only if it carries exactly ONE numeric token (``1：3``, ``15-11, 15-11``,
    ``5.4公里（3.3英里）`` and dates are rejected instead of being cut to their first number);
  * year/date, ordinal/rank, identifier and score columns are never metrics;
  * entity keys drop annotations such as ``（冠）`` or ``(第 1-13 轮)`` but keep the raw cell;
  * aggregate rows (合计/总计/全国…) are removed before any argmax/argmin/rank.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CJK_RE = re.compile(r"[一-鿿]")
PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
MARK_RE = re.compile(r"[†‡*＊§#※^]+")
NUM_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")
STRICT_NUM_RE = re.compile(
    r"^(?P<sign>[+\-−–])?\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>%|‰|[a-zA-Z²³/]{0,6}|[一-鿿]{0,3})$")
DATE_RE = re.compile(r"\d{1,4}\s*[年月日]|\d{4}[-/.]\d{1,2}|\d{1,2}:\d{2}")
AGGREGATE_ROW_RE = re.compile(
    r"^(合计|总计|总和|总数|全国|全国合计|中国大陆|大陆|全省|全市|全区|全州|全盟|全县|平均|小计|其他|其它|总体|世界|全球|"
    r"东部|中部|西部|东北|total)$|(总计|合计|小计|平均值|其他|其它)$|^(全省|全市|全国|全区)", re.I)

FORBIDDEN_METRIC_RE = re.compile(
    r"年份|年度|年代|日期|时间|届|赛季|编号|序号|代码|排名|名次|^名$|^#$|^No\.?$|次序|顺位|位次|邮编|电话|"
    r"比分|比数|成绩|坐标|经度|纬度|海拔高度范围|成立|创建|出生|逝世|生卒|首次|开幕|建成|任期|号码|车号|背号|列\d+$|"
    r"轮次|^轮|周次|期别|场序|赛程|得票|投票|票数|排序|顺序|等级|星级|邮政|区号|面值|汇率|合人民币")
RANK_HEADER_RE = re.compile(r"^(排名|名次|名|#|位次|排位|序)$")
KEY_HINT_RE = re.compile(
    r"球队|队伍|俱乐部|球员|姓名|名称|国家|地区|省|市|城市|车手|车队|学校|大学|机场|公司|企业|电影|影片|作品|"
    r"河流|山峰|建筑|代表团|运动员|选手|项目|区|县|州|政区")
PERIOD_HEADER_RE = re.compile(r"^(?P<year>(?:19|20)\d{2})年?(?P<rest>.*)$")
KEY_SUFFIXES = ("足球俱乐部", "篮球俱乐部", "俱乐部", "足球队", "篮球队", "代表团", "队")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


NAVBOX_RE = re.compile(r"查\s*论\s*编|查\s*论\s*编\s*$|\bv\s*t\s*e\b", re.I)


def entity_key(raw: Any) -> str:
    """Display/lookup form of an entity cell: annotations and marks removed, whitespace collapsed."""
    text = str(raw or "")
    prev = None
    while prev != text:
        prev, text = text, PAREN_RE.sub("", text)
    text = MARK_RE.sub("", NAVBOX_RE.sub("", text))
    text = re.sub(r"\s*\((?:H|C|Q|A|R|P|X|E)\)\s*$", "", text)
    return re.sub(r"\s+", "", text).strip("，,、；;:：")


def join_key(raw: Any) -> str:
    """Looser form used only to match the same entity across tables (武汉三镇足球俱乐部 ~ 武汉三镇)."""
    key = entity_key(raw)
    for suffix in KEY_SUFFIXES:
        if key.endswith(suffix) and len(key) - len(suffix) >= 2:
            return key[: -len(suffix)]
    return key


def parse_number(raw: Any) -> float | None:
    """Strict numeric parse: exactly one numeric token, no dates, no scores, no ranges."""
    text = str(raw or "").strip()
    if not text:
        return None
    text = MARK_RE.sub("", text)
    text = re.sub(r"(?<=\d)[,，](?=\d{3}\b)", "", text)          # thousands separators only
    if DATE_RE.search(text):
        return None
    if len(NUM_TOKEN_RE.findall(text)) != 1:                    # count tokens BEFORE dropping spaces
        return None
    text = text.replace(" ", "")
    m = STRICT_NUM_RE.match(text)
    if not m:
        return None
    value = float(m.group("num"))
    if m.group("sign") and m.group("sign") != "+":
        value = -value
    return value


def format_number(value: float, raw: Any = None) -> str:
    """Render a value the way the source wrote it (decimal places), without thousands separators."""
    raw_text = str(raw or "")
    decimals = 0
    m = re.search(r"\d+\.(\d+)", raw_text.replace(",", ""))
    if m:
        decimals = len(m.group(1))
    elif value != int(value):
        decimals = min(2, len(repr(value).split(".")[1]))
    text = f"{value:.{decimals}f}"
    return text


@dataclass
class Column:
    index: int
    header: str
    kind: str                      # key | metric | rank | category | text | other
    numeric_ratio: float
    unique_ratio: float
    reason: str = ""
    period: int | None = None      # 2022 for headers like "2022年" / "2022年GDP"
    period_rest: str = ""
    unit: str = ""
    is_rate: bool = False


@dataclass
class TableProfile:
    table: dict[str, Any]
    rows: list[list[str]]                              # data rows after aggregate-row removal
    row_map: list[int]                                 # data row -> original row index
    columns: list[Column] = field(default_factory=list)
    key: Column | None = None

    @property
    def table_id(self) -> str:
        return self.table["table_id"]

    def metrics(self) -> list[Column]:
        return [c for c in self.columns if c.kind == "metric"]

    def categories(self) -> list[Column]:
        return [c for c in self.columns if c.kind == "category"]

    def rank(self) -> Column | None:
        return next((c for c in self.columns if c.kind == "rank"), None)

    def column(self, header: str) -> Column | None:
        return next((c for c in self.columns if c.header == header), None)

    def key_values(self) -> list[str]:
        return [entity_key(r[self.key.index]) for r in self.rows] if self.key else []

    def find_rows(self, key: str, *, loose: bool = False) -> list[int]:
        """Data-row indexes whose key equals ``key`` (exact entity_key; join_key when loose)."""
        if not self.key:
            return []
        norm = join_key if loose else entity_key
        target = norm(key)
        return [i for i, r in enumerate(self.rows) if norm(r[self.key.index]) == target]

    def find_row(self, key: str, *, loose: bool = False) -> int | None:
        hits = self.find_rows(key, loose=loose)
        return hits[0] if len(hits) == 1 else None

    def cell(self, row: int, col: Column) -> str:
        r = self.rows[row]
        return r[col.index] if col.index < len(r) else ""


_PROFILE_CACHE: dict[tuple, TableProfile] = {}


def profile_table(table: dict[str, Any]) -> TableProfile:
    """Profile a table; memoized on (table_id, shape) since ids are unique per content."""
    key = (table.get("table_id"), len(table["rows"]), len(table["headers"]), table.get("period"))
    cached = _PROFILE_CACHE.get(key)
    if cached is not None and cached.table is table:
        return cached
    prof = _profile_table(table)
    _PROFILE_CACHE[key] = prof
    return prof


def _profile_table(table: dict[str, Any]) -> TableProfile:
    """Two passes: type columns on all rows to find the key and rank columns, then drop aggregate rows
    (key value 全区/全国/合计…, or a non-numeric rank where the other rows are ranked) and type again."""
    headers = [str(h) for h in table["headers"]]
    width = len(headers)
    raw_rows = [list(r) + [""] * (width - len(r)) for r in table["rows"]]
    first = _type_columns(table, raw_rows, list(range(len(raw_rows))))
    keep, row_map = [], []
    rank = first.rank()
    ranked = [i for i, r in enumerate(raw_rows) if rank and parse_number(r[rank.index]) is not None]
    for i, r in enumerate(raw_rows):
        key_text = entity_key(r[first.key.index]) if first.key else next((entity_key(c) for c in r if c and has_cjk(c)), "")
        if AGGREGATE_ROW_RE.search(key_text or ""):
            continue
        if rank and len(ranked) >= 0.8 * len(raw_rows) and i not in ranked:
            continue
        keep.append(r)
        row_map.append(i)
    return _type_columns(table, keep, row_map)


def _type_columns(table: dict[str, Any], keep: list[list[str]], row_map: list[int]) -> TableProfile:
    headers = [str(h) for h in table["headers"]]
    prof = TableProfile(table=table, rows=keep, row_map=row_map)
    n = len(keep)
    for ci, h in enumerate(headers):
        vals = [r[ci] for r in keep if r[ci] not in ("", "—", "-", "–", "N/A", "n/a")]
        nonempty = len(vals)
        nums = [parse_number(v) for v in vals]
        numeric_ratio = sum(x is not None for x in nums) / nonempty if nonempty else 0.0
        keys = [entity_key(v) for v in vals]
        unique_ratio = len(set(keys)) / nonempty if nonempty else 0.0
        col = Column(ci, h, "other", round(numeric_ratio, 3), round(unique_ratio, 3))
        pm = PERIOD_HEADER_RE.match(h)
        if pm:
            col.period, col.period_rest = int(pm.group("year")), pm.group("rest").strip(" ：:（）()")
        if n == 0 or nonempty < max(3, int(0.8 * n)):
            col.reason = "sparse"
        elif numeric_ratio >= 0.95:
            ints = [x for x in nums if x is not None]
            yearish = sum(1 for x in ints if float(x).is_integer() and 1800 <= x <= 2100) / len(ints)
            ordinal = sorted(ints) == [float(k) for k in range(1, len(ints) + 1)]
            if RANK_HEADER_RE.match(h) or (ordinal and not has_cjk(h.replace("数", ""))):
                col.kind = "rank"
            elif FORBIDDEN_METRIC_RE.search(h) and col.period is None:
                col.reason = "forbidden_header"
                if RANK_HEADER_RE.search(h) or "排名" in h or "名次" in h:
                    col.kind = "rank"
            elif yearish >= 0.8:
                col.reason = "year_values"
            elif not has_cjk(h) and col.period is None:
                col.reason = "non_chinese_header"
            elif ordinal and len(ints) >= 5:
                col.kind = "rank"
                col.reason = "ordinal_sequence"
            elif (sum(1 for v in vals if str(v).strip()[:1] in "+−–-" and parse_number(v) not in (None, 0.0)) > 0.3 * nonempty
                  and not re.search(r"净|差|变化|增|涨|跌|盈亏", h)):
                col.reason = "signed_change_values"
            else:
                col.kind = "metric"
                col.is_rate = "率" in h or sum("%" in v for v in vals) >= 0.8 * nonempty
                units = set()
                for v in vals:
                    t = re.sub(r"(?<=\d)[,，](?=\d{3}\b)", "", MARK_RE.sub("", str(v))).replace(" ", "")
                    m = STRICT_NUM_RE.match(t)
                    if m:
                        units.add(m.group("unit"))
                col.unit = units.pop() if len(units) == 1 else ""
                if len(set(x for x in ints)) <= 1:
                    col.kind, col.reason = "other", "constant"
                elif len(units) > 1 or (len(units) == 0 and col.unit == "" and False):
                    col.kind, col.reason = "other", "mixed_units"
                elif re.search(r"_\d+$", h):
                    col.kind, col.reason = "other", "duplicate_header"
        elif numeric_ratio <= 0.2:
            avg_len = sum(len(k) for k in keys) / nonempty
            distinct = len(set(keys))
            if unique_ratio >= 0.95 and 2 <= avg_len <= 24 and sum(has_cjk(k) for k in keys) >= 0.8 * nonempty:
                col.kind = "key"
            elif (2 <= distinct <= max(2, n // 2) and sum(1 for k in set(keys) if keys.count(k) >= 2) >= 2
                  and avg_len <= 12 and has_cjk(h) and not re.search(r"[＼\\]", h)
                  and sum(1 for k in keys if re.search(r"\d", k)) <= 0.2 * nonempty):
                col.kind = "category"
            else:
                col.kind = "text"
        prof.columns.append(col)
    keys = [c for c in prof.columns if c.kind == "key"]
    if keys:
        keys.sort(key=lambda c: (not KEY_HINT_RE.search(c.header), c.index))
        prof.key = keys[0]
    return prof


def usable(prof: TableProfile, min_rows: int = 4) -> bool:
    return prof.key is not None and len(prof.rows) >= min_rows and bool(prof.metrics())
