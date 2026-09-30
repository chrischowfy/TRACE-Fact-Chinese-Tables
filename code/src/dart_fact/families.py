"""Instance builders for the seven reasoning-family groups.

Each builder turns an evidence package into *bases* (one grounded fact) and, for each base, sibling
drafts with the SAME skeleton, slots layout and surface template:
  SUPPORTS  the grounded program returns True;
  REFUTES   exactly one semantic slot is perturbed (entity, number, rank, category member, direction)
            so that the same program returns False;
  NEI       exactly one required binding is replaced by one that is absent from the whole package
            (entity from another edition, metric the tables do not carry, missing join partner,
            missing period), while every other slot stays grounded.
The intended label is only a request: ``pipeline`` executes every program and keeps a draft only if
execution returns that label.
"""
from __future__ import annotations

import collections
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .source_adapters import (ATTR_HEADER_RE, CATEGORY_HEADER_RE, NEI_METRICS, NON_PERIOD_METRICS, PERSON_NOUNS, PROVINCES,
                              clean_name, entity_noun as _entity_noun, metric_info, only_max, province_core, refine_noun)
from .resolve import REDIRECTS
from .tables import AGGREGATE_ROW_RE, TableProfile, entity_key, format_number, has_cjk, join_key, parse_number, profile_table

GROUPS = {
    "single": "Single-table comparison & ranking",
    "split_pair": "Two-entity cross-table comparison",
    "category_decomp": "Category decomposition",
    "rank_detail": "Real-join rank–bridge",
    "hub_profile": "Hub-profile (multi-table)",
    "period_pair": "Cross-period growth",
    "join_filter": "Join-filter aggregation",
    # four tables, but the same question the cross-period family asks, so it stays in that family
    # four tables, but the question it asks is a category decomposition, so it stays in that family
    "category_partner": "Category decomposition",
}
SPORTS_DOMAINS = ("football", "football_intl", "basketball")
LATIN_RE = re.compile(r"[A-Za-z]")
UNIT_IN_HEADER_RE = re.compile(
    r"[（(]\s*(十亿|百亿|百万|千万|亿|万|千)?\s*(人民币|欧元|日元|英镑|港元|港币|新台币|美元|元|标准箱|标箱|人|吨|平方公里|公里|千米|米|km²|km)\s*[)）]")


@dataclass
class Draft:
    skeleton: str
    group: str
    label: str
    base_id: str
    package_id: str
    operators: list[dict[str, Any]]
    slots: dict[str, Any]
    axis: str = ""                     # refutation / NEI axis
    absent: list[str] = field(default_factory=list)   # strings that must NOT occur in the package (NEI)
    value_source: str = ""             # "corpus" when a stated number was copied from a real row of the
                                       # same metric elsewhere in the benchmark rather than invented


def _lex_items():
    from .source_adapters import MEDAL_ONLY, METRIC_LEXICON
    return list(METRIC_LEXICON.items()) + list(MEDAL_ONLY.items())


GENERIC_KEY_RE = re.compile(r"^(名称|名字|列\d+|项目|条目)$")


def entity_noun(key_header: str, domain: str = "", prof: TableProfile | None = None) -> str:
    noun = _entity_noun(key_header, domain)
    if noun == "对象" and prof is not None and GENERIC_KEY_RE.match(clean_name(key_header)):
        # a bare 名称 column says nothing about what is listed; the table's own caption and the page
        # title usually do ("安徽省自然保护区列表" -> 保护区), and that is what the claim must call them
        from .source_adapters import ENTITY_NOUNS
        title = f"{prof.table.get('title', '')}{prof.table.get('source', {}).get('page_title', '')}"
        for pattern, cand in ENTITY_NOUNS:
            if cand != "对象" and re.search(pattern, title):
                noun = cand
                break
    return refine_noun(noun, prof.key_values()) if prof is not None and prof.key else noun


_SYNONYMS = [(r"本币|美元|人民币|购买力平价|国际元|汇率|名义|实际", ""), (r"参赛次数|出席次数", "场"),
             (r"地区生产总值|国内生产总值|生产总值", "GDP"), (r"冠军", "金牌"), (r"亚军", "银牌"), (r"季军", "铜牌"),
             (r"楼层", "层数"), (r"增速|增长率|增幅|增长%|增长", "增幅"), (r"实际|名义|含普调|年均|年", ""),
             (r"^.{0,6}人口$", "人口"), (r"场次|场数", "场"), (r"得$|进球", "进球"), (r"失$|失球", "失球")]


def _metric_norm(name: str) -> str:
    """Coarse metric identity used to keep NEI metrics truly absent (synonyms, 人均 and units ignored)."""
    n = re.sub(r"（.*|\(.*|/.*|%|\s", "", name)
    n = n.replace("人均", "")
    for pattern, repl in _SYNONYMS:
        n = re.sub(pattern, repl, n)
    return re.sub(r"(数量|总数|总量|总额|数)$", "", n).strip() or name


# metrics that can be computed from columns the package does have: never used as "absent"
_DERIVABLE = [("胜差", ("胜", "负")), ("积分", ("胜", "负")), ("净胜球", ("得", "失")), ("奖牌总数", ("金牌", "银牌")),
              ("人均", ("GDP", "人口")), ("密度", ("人口", "面积")), ("胜率", ("胜", "负")), ("比赛场次", ("胜", "负"))]


def _decimals(raw: Any) -> int:
    m = re.search(r"\d+\.(\d+)", str(raw).replace(",", ""))
    return len(m.group(1)) if m else 0


def _bid(*parts: Any) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def good_entity(name: str) -> bool:
    return (2 <= len(name) <= 16 and has_cjk(name) and len(LATIN_RE.findall(name)) <= 2
            and not AGGREGATE_ROW_RE.search(name)
            and not re.search(r"[^一-鿿A-Za-z·•・\-]", name))


def metric_slot(prof: TableProfile, col, domain: str = "") -> dict[str, Any] | None:
    info = metric_info(col.header, unit=col.unit, is_rate=col.is_rate, domain=domain)
    if not info:
        return None
    name, style, measure = info
    context = f"{prof.table.get('title', '')}{prof.table.get('source', {}).get('page_title', '')}"
    other_headers = "|".join(str(h) for h in prof.table.get("headers", []) if h != col.header)
    if ("人均" in context and "人均" not in name and re.match(r"(GDP|生产总值|地区生产总值|可支配收入|收入|产值)$", name)
            and not re.search(r"百万|亿|万", col.header) and "人均" not in other_headers):
        name = "人均" + name
    unit = measure
    if col.unit and has_cjk(col.unit):          # unit written in the cells (亿美元) beats the header (美元)
        unit = col.unit
    elif col.unit in ("%",):
        unit = "%"
    if not unit:
        m = UNIT_IN_HEADER_RE.search(col.header)
        if m:
            unit = (m.group(1) or "") + {"人民币": "元"}.get(m.group(2), m.group(2))
        elif col.is_rate:
            unit = "%" if col.unit == "%" else ""
        elif col.unit and has_cjk(col.unit):
            unit = col.unit
    bare_currency = unit in ("美元", "元", "人民币", "国际元") and domain in ("economy", "world_stats")
    return {"metric_col": col.header, "metric_name": name, "style": style, "unit": unit,
            "numeric_surface_ok": (bool(unit) or style == "count") and not bare_currency}


def rows_with_values(prof: TableProfile, col) -> list[tuple[str, float, str]]:
    out, seen = [], set()
    counts: dict[str, int] = {}
    for r in range(len(prof.rows)):
        jk = join_key(prof.cell(r, prof.key))
        counts[jk] = counts.get(jk, 0) + 1
    for r in range(len(prof.rows)):
        name = entity_key(prof.cell(r, prof.key))
        if counts.get(join_key(name), 0) != 1:
            continue
        value = parse_number(prof.cell(r, col))
        if value is None or not good_entity(name) or join_key(name) in seen:
            continue
        seen.add(join_key(name))
        out.append((name, value, prof.cell(r, col)))
    return out


def value_dict(value: float, raw: str, unit: str, magnitude: bool = False) -> dict[str, Any]:
    """Surface form of a number. Signs are kept (净胜球数 −5), except for explicit magnitudes such as the
    amount in 'X比Y多d' or '增加了d', whose direction is carried by the verb."""
    text = format_number(abs(value), raw)
    if value < 0 and not magnitude:
        text = "-" + text
    return {"value": value, "text": text, "unit": unit}



class Context:
    """Cross-package resources: entity pools for NEI substitution, package text for absence checks."""

    def __init__(self, packages: list[dict[str, Any]], seed: int = 20260913) -> None:
        self.rng = random.Random(seed)
        self.pools: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for pkg in packages:
            if pkg["kind"] != "single":
                continue
            prof = profile_table(pkg["tables"][0])
            if not prof.key:
                continue
            key = (pkg["domain"], entity_noun(prof.key.header, pkg["domain"], prof))
            if key[1] == "对象":
                continue
            for name in prof.key_values():
                if good_entity(name) and ((key[1] == "省级行政区") == (province_core(name) in PROVINCES)
                                          or key[1] not in ("省级行政区", "城市")):
                    self.pools.setdefault(key, []).append((name, pkg["series"]))
        for key in self.pools:
            self.pools[key] = sorted(set(self.pools[key]))
        self._metric_use: collections.Counter = collections.Counter()
        # when set, a substituted slot prefers one of these words: values that also occur in the released
        # decidable claims, so the substitution cannot mark the label by its presence (pipeline.build pass two)
        self.attested_metrics: set[str] | None = None
        self.attested_entities: set[str] | None = None
        self.categories: dict[tuple[str, str], list[str]] = {}
        for pkg in packages:
            if pkg["kind"] == "category_decomp":
                names = [entity_key(r[0]) for r in pkg["tables"][0]["rows"]]
                self.categories.setdefault((pkg["domain"], pkg["category"]), []).extend(n for n in names if n)
        for key in self.categories:
            self.categories[key] = sorted(set(self.categories[key]))
        # real metric names per domain: a metric-slot NEI borrows a metric that IS verifiable elsewhere in the
        # benchmark, so the metric word itself carries no label signal
        # ...and, with each metric, real observed values of it.  A metric-slot NEI has to state some
        # number; inventing one risks a magnitude that reads as plainly false rather than as unreadable,
        # so the number is copied from an actual row of that same metric elsewhere in the benchmark.
        # Every table of every package, not only single-table ones: the invariant a metric-slot NEI needs
        # is that the borrowed name is a real, readable metric of this kind of entity *somewhere* in the
        # benchmark, which a detail or profile table establishes just as well as a standalone list.  Keying
        # by the table's own entity noun keeps player metrics out of the pool for teams and vice versa.
        self.metric_pool: dict[str, dict[str, tuple[str, str, str, frozenset, tuple]]] = {}
        for pkg in packages:
            for table in pkg["tables"]:
                prof = profile_table(table)
                if not prof.key:
                    continue
                noun = entity_noun(prof.key.header, pkg["domain"], prof)
                if noun == "对象":
                    continue
                for col in prof.metrics():
                    ms = metric_slot(prof, col, pkg["domain"])
                    if not ms or col.period is not None:
                        continue
                    headers = frozenset(h for h, (n, _, _) in _lex_items() if n == ms["metric_name"]) | {col.header}
                    slot = self.metric_pool.setdefault((pkg["domain"], noun), {})
                    prev = slot.get(ms["metric_name"])
                    samples = prev[4] if prev else ()
                    if len(samples) < 40:
                        samples = samples + tuple((v, raw) for _n, v, raw in rows_with_values(prof, col)[:20])
                    slot[ms["metric_name"]] = (ms["metric_name"], ms["style"], ms["unit"],
                                               (prev[3] | headers) if prev else headers, samples)

    @staticmethod
    def package_text(pkg: dict[str, Any]) -> list[str]:
        """Strings of the package AND of every table on its source page(s): an NEI substitute must not occur
        anywhere on the page, so the missing binding is not merely left out of the package."""
        from .packages import PAGE_TEXT
        texts = []
        for t in pkg["tables"]:
            texts.append(t.get("title", ""))
            texts.extend(t["headers"])
            for row in t["rows"]:
                texts.extend(row)
        out = [entity_key(x) for x in texts if x]
        for title in str(pkg.get("page_title", "")).split("；"):
            out.extend(PAGE_TEXT.get(title, []))
        return out

    def absent_entity(self, pkg: dict[str, Any], domain: str, noun: str) -> str | None:
        if domain == "medals":
            return None          # medal tables list only medal winners: absence reads as zero, not as missing
        if noun in PERSON_NOUNS:
            return None          # transliteration variants (艾克森 / 埃尔克森) make absence of a person unreliable
        texts = self.package_text(pkg)
        page_canon = self.page_canonicals(pkg)
        prefixes = {t[:2] for t in texts if len(t) >= 3} if noun == "球队" else set()
        pool = self.pools.get((domain, noun), [])
        same = [n for n, s in pool if s == pkg.get("series")]
        other = [n for n, s in pool if s != pkg.get("series")]
        for candidates in (same, other):
            cands = candidates[:]
            self.rng.shuffle(cands)
            if self.attested_entities is not None:
                cands.sort(key=lambda n: n not in self.attested_entities)
            for name in cands[:200]:
                jk = join_key(name)
                if noun == "球队" and name[:2] in prefixes:
                    continue     # same city prefix: likely the same club under another sponsor name
                canon = REDIRECTS.get(name)
                if canon is None or (canon and canon in page_canon):
                    continue     # unresolved name, or an alias of an entity on the page (乔治亚 = 格鲁吉亚)
                if not any(jk in t or (len(t) >= 2 and t in name) for t in texts):
                    return name
        return None

    @staticmethod
    def page_canonicals(pkg: dict[str, Any]) -> set[str]:
        """Canonical articles of every linked or named entity on the package's source page(s)."""
        from .packages import PAGE_LINKS, PAGE_TEXT
        out: set[str] = set()
        for title in str(pkg.get("page_title", "")).split("；"):
            for pair in PAGE_LINKS.get(title, []):
                text, _, link = pair.partition("\t")
                for s in (link, text):
                    c = REDIRECTS.get(s)
                    out.add(c if c else s)
            for text in PAGE_TEXT.get(title, []):
                c = REDIRECTS.get(text)
                if c:
                    out.add(c)
        return out

    def absent_metric(self, pkg: dict[str, Any], domain: str, noun: str = "") -> tuple[str, str, str, tuple] | None:
        """A metric name that is verifiable elsewhere in the benchmark but readable nowhere on this page.

        The page — not just the package — is the unit: a name carried by any table of the source page
        would be readable by a model that is shown the page, so those candidates are skipped here and the
        next one is tried, instead of being generated and then rejected downstream.  Returns the name, its
        style and unit, and real observed values of it.
        """
        from .packages import PAGE_HEADERS
        headers = {h for t in pkg["tables"] for h in t["headers"]}
        for title in str(pkg.get("page_title", "")).split("；"):
            headers |= {h for h in PAGE_HEADERS.get(title, [])}
        singles = {h.strip() for h in headers if len(h.strip()) == 1 and has_cjk(h.strip())}
        names_here = set()
        for t in pkg["tables"]:
            prof = profile_table(t)
            for col in prof.metrics():
                ms = metric_slot(prof, col, domain)
                if ms:
                    names_here.add(ms["metric_name"])
        options = sorted(self.metric_pool.get((domain, noun), {}).values(), key=lambda o: o[0])
        self.rng.shuffle(options)
        # Then the same entity type in any other domain.  What makes a borrowed metric name safe is that
        # it is a real, readable property *of this kind of entity*; which page happens to carry it is an
        # artefact of how the corpus was collected.  Keeping the domain-local names first keeps the
        # wording idiomatic for the topic when the corpus offers a choice.
        # Not every metric of an entity type survives the move to another domain: a CBA team with a
        # 平局场数 is not an unreadable fact but a false one, because basketball has no draws.  A metric
        # that several domains independently record for this entity type (球场容量, 主场观众人数) is a
        # property of the entity; one that only a single domain records is a property of that sport or
        # that statistical tradition, and does not travel.  So the corpus itself decides.
        # 'football' and 'football_intl' are one sport, so a metric attested in both is still attested
        # by a single tradition and must not count as transferable
        def tradition(d: str) -> str:
            return "football" if d.startswith("football") else d
        seen_in: dict[str, set[str]] = {}
        for (_d, n), slot in self.metric_pool.items():
            if n == noun:
                for name in slot:
                    seen_in.setdefault(name, set()).add(tradition(_d))
        wider = sorted((v for (_d, n), slot in self.metric_pool.items()
                        if n == noun and tradition(_d) != tradition(domain)
                        for v in slot.values() if len(seen_in.get(v[0], ())) >= 2), key=lambda o: o[0])
        self.rng.shuffle(wider)
        tier = {o[0]: 0 for o in options}
        for o in wider:
            tier.setdefault(o[0], 1)
        options = options + wider
        # Fall back to the curated per-domain vocabulary only when the corpus offers nothing.  Preferring
        # corpus-attested names keeps the metric word itself free of label signal - it is a word that is
        # readable elsewhere in the benchmark - but some corners of the corpus carry a single metric for an
        # entity type (every football squad table lists 进球 and nothing else), leaving no name to borrow.
        curated = [(n, s, u, frozenset(), ()) for n, s, u in NEI_METRICS.get(domain, [])]
        self.rng.shuffle(curated)
        for o in curated:
            tier.setdefault(o[0], 2)
        options = options + curated
        # Least-used name first.  Without this one convenient word ('奖金') wins the filters on hundreds of
        # packages and ends up in a sixth of all NEI while appearing in no decidable claim at all, which is
        # a lexical tell a hypothesis-only classifier can learn.  Ties keep the shuffled order above.
        # Balance first, then preference: a domain-local corpus name still wins over a borrowed or
        # curated one at equal usage, so the idiomatic wording survives the spreading.
        # Attested first: a word that also occurs in the released decidable claims cannot mark the label
        # by its presence alone.  It is a preference, not a filter - a cell that would otherwise go short
        # still falls back to an unattested name, and the build reports how many did.
        att = self.attested_metrics
        options.sort(key=lambda o: (0 if att is not None and o[0] in att else 1,
                                    self._metric_use[o[0]], tier.get(o[0], 3)))
        here_norm = {_metric_norm(n) for n in names_here} | {_metric_norm(clean_name(h)) for h in headers}
        for name, style, unit, source_headers, samples in options:
            if any(s in name for s in singles):
                continue     # 得分 against a page whose header is 分: the name reads as that column
            core = re.sub(r"(数|数量|人数|次数|场数)$", "", name)
            if name in names_here or source_headers & headers or _metric_norm(name) in here_norm:
                continue
            if any(_metric_norm(name) in h or h in _metric_norm(name) for h in here_norm if len(h) >= 2):
                continue
            if name in ("指数", "密度", "数值") or len(_metric_norm(name)) < 2:
                continue
            joined = "|".join(headers)
            if any(key in name and all(part in joined for part in parts) for key, parts in _DERIVABLE):
                continue
            if any(core and (core in h or (len(h) >= 2 and h in name)) for h in headers):
                continue
            self._metric_use[name] += 1
            return name, style, unit, samples
        return None


# =========================================================================== builders
def build_single(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    t = pkg["tables"][0]
    prof = profile_table(t)
    tid, group, rng = t["table_id"], GROUPS["single"], ctx.rng
    noun = entity_noun(prof.key.header, pkg["domain"], prof)
    out: list[Draft] = []
    if noun == "对象":
        return out
    metrics = [(c, metric_slot(prof, c, pkg["domain"])) for c in prof.metrics() if c.period is None]
    metrics = [(c, ms) for c, ms in metrics if ms][:3]
    for col, ms in metrics:
        vals = rows_with_values(prof, col)
        if len(vals) < 4:
            continue
        base_slots = dict(ms, scope=pkg["scope"], noun=noun, table=tid)
        # ---- pairwise comparison
        pairs = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1:] if a[1] != b[1]]
        rng.shuffle(pairs)
        for (hi, lo) in [(a, b) if a[1] > b[1] else (b, a) for a, b in pairs[:2]]:
            bid = _bid(pkg["package_id"], "cmp", col.header, hi[0], lo[0])
            cmp = rng.choice(["gt", "lt"])
            for label in ("SUPPORTS", "REFUTES"):
                first_hi = (cmp == "gt") == (label == "SUPPORTS")
                e1, e2 = (hi[0], lo[0]) if first_hi else (lo[0], hi[0])
                out.append(Draft("ST_COMPARE", group, label, bid, pkg["package_id"],
                                 _cmp_ops(tid, tid, e1, e2, col.header, cmp), dict(base_slots, e1=e1, e2=e2, cmp=cmp),
                                 axis="entity_order" if label == "REFUTES" else ""))
            absent = ctx.absent_entity(pkg, pkg["domain"], noun)
            if absent:
                out.append(Draft("ST_COMPARE", group, "NEI", bid, pkg["package_id"],
                                 _cmp_ops(tid, tid, hi[0], absent, col.header, cmp),
                                 dict(base_slots, e1=hi[0], e2=absent, cmp=cmp), axis="entity", absent=[absent]))
        eq_pairs = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1:] if a[1] == b[1]]
        if eq_pairs:
            a, b = rng.choice(eq_pairs)
            c = rng.choice([x for x in vals if x[1] != a[1]])
            bid = _bid(pkg["package_id"], "eq", col.header, a[0], b[0])
            out.append(Draft("ST_COMPARE", group, "SUPPORTS", bid, pkg["package_id"],
                             _cmp_ops(tid, tid, a[0], b[0], col.header, "eq"), dict(base_slots, e1=a[0], e2=b[0], cmp="eq")))
            out.append(Draft("ST_COMPARE", group, "REFUTES", bid, pkg["package_id"],
                             _cmp_ops(tid, tid, a[0], c[0], col.header, "eq"), dict(base_slots, e1=a[0], e2=c[0], cmp="eq"),
                             axis="entity"))
        # ---- superlatives (skip when the extreme is tied)
        for mode in (() if pkg.get("is_slice") else
                     ("max",) if only_max(pkg["series"], noun) else ("max", "min")):
            ext = (max if mode == "max" else min)(v for _, v, _ in vals)
            winners = [x for x in vals if x[1] == ext]
            if len(winners) != 1:
                continue
            others = sorted([x for x in vals if x[1] != ext], key=lambda x: abs(x[1] - ext))
            distractor = others[0] if rng.random() < 0.5 else rng.choice(others)
            bid = _bid(pkg["package_id"], "sup", col.header, mode)
            for label, claimed in (("SUPPORTS", winners[0][0]), ("REFUTES", distractor[0])):
                out.append(Draft("ST_SUPERLATIVE", group, label, bid, pkg["package_id"],
                                 _sup_ops(tid, col.header, mode, claimed), dict(base_slots, mode=mode, claimed=claimed),
                                 axis="entity" if label == "REFUTES" else ""))
            nm = ctx.absent_metric(pkg, pkg["domain"], noun)
            if nm:
                claimed = rng.choice([winners[0][0], distractor[0]])
                out.append(Draft("ST_SUPERLATIVE", group, "NEI", bid, pkg["package_id"],
                                 _sup_ops(tid, nm[0], mode, claimed),
                                 dict(base_slots, metric_col=nm[0], metric_name=nm[0], style=nm[1], unit=nm[2],
                                      mode=mode, claimed=claimed), axis="metric", absent=[nm[0]]))
        # ---- rank position (unique values only; not over partial lists)
        uniq = [x for x in vals if sum(1 for y in vals if y[1] == x[1]) == 1]
        if len(vals) >= 5 and uniq and not only_max(pkg["series"], noun) and not pkg.get("is_slice"):
            e = rng.choice(uniq)
            k = 1 + sum(1 for y in vals if y[1] > e[1])
            wrong = k + rng.choice([-1, 1]) if 1 < k < len(vals) else (k + 1 if k == 1 else k - 1)
            bid = _bid(pkg["package_id"], "rank", col.header, e[0])
            for label, kk in (("SUPPORTS", k), ("REFUTES", wrong)):
                out.append(Draft("ST_RANK", group, label, bid, pkg["package_id"], _rank_ops(tid, e[0], col.header, kk),
                                 dict(base_slots, e=e[0], k=kk), axis="rank" if label == "REFUTES" else ""))
            absent = ctx.absent_entity(pkg, pkg["domain"], noun)
            if absent:
                out.append(Draft("ST_RANK", group, "NEI", bid, pkg["package_id"], _rank_ops(tid, absent, col.header, k),
                                 dict(base_slots, e=absent, k=k), axis="entity", absent=[absent]))
        # ---- count over a threshold.  Only over a table that lists its whole population: on a top-N
        # list "how many exceed T" counts the listed rows, not the real ones.
        if (ms["numeric_surface_ok"] and len(vals) >= 6 and not pkg.get("is_slice")
                and not only_max(pkg["series"], noun)):
            ordered = sorted(v for _, v, _ in vals)
            # a threshold near the top keeps the counted set small: the evidence is the rows above it,
            # and "37 of 80 exceed X" would both bloat the evidence and be tedious to check
            threshold = ordered[max(0, int(len(ordered) * 0.8) - 1)]
            n = sum(1 for v in ordered if v > threshold)
            if 2 <= n <= 8:
                raw_t = next(raw for _, v, raw in vals if v == threshold)
                bid = _bid(pkg["package_id"], "cntgt", col.header)
                for label, nn in (("SUPPORTS", n), ("REFUTES", n + 1 if n + 1 < len(ordered) else n - 1)):
                    out.append(Draft("ST_COUNT_GT", group, label, bid, pkg["package_id"],
                                     _st_count_ops(tid, col.header, threshold, nn),
                                     dict(base_slots, threshold=value_dict(threshold, raw_t, ms["unit"]), n=nn),
                                     axis="number" if label == "REFUTES" else ""))
        # ---- difference (only when numbers can be rendered with a unit)
        if ms["numeric_surface_ok"] and not ms["style"] == "rate" and len(pairs) > 2:
            hi, lo = (pairs[2][0], pairs[2][1]) if pairs[2][0][1] > pairs[2][1][1] else (pairs[2][1], pairs[2][0])
            out.extend(_diff_drafts("ST_DIFF", group, pkg, ctx, tid, tid, hi, lo, col.header, base_slots, noun))
    return out


def _diff_drafts(skeleton, group, pkg, ctx, t1, t2, hi, lo, header, base_slots, noun) -> list[Draft]:
    rng = ctx.rng
    d = round(hi[1] - lo[1], 6)
    if d <= 0:
        return []
    decimals = max(_decimals(hi[2]), _decimals(lo[2]))
    step = max(1.0, round(abs(d) * rng.choice([0.2, 0.3, 0.5]))) if decimals == 0 else round(max(abs(d) * 0.3, 10 ** -decimals), decimals)
    wrong = round(d + rng.choice([-1, 1]) * step, decimals)
    if wrong <= 0:
        wrong = round(d + step, decimals)
    raw_fmt = "1." + "0" * decimals if decimals else "1"
    bid = _bid(pkg["package_id"], skeleton, header, hi[0], lo[0])
    out = []
    for label, dd in (("SUPPORTS", d), ("REFUTES", wrong)):
        out.append(Draft(skeleton, group, label, bid, pkg["package_id"], _diff_ops(t1, t2, hi[0], lo[0], header, dd),
                         dict(base_slots, e1=hi[0], e2=lo[0], diff=value_dict(dd, raw_fmt, base_slots["unit"], magnitude=True),
                              table2=t2), axis="number" if label == "REFUTES" else ""))
    absent = ctx.absent_entity(pkg, pkg["domain"], noun)
    if absent:
        out.append(Draft(skeleton, group, "NEI", bid, pkg["package_id"], _diff_ops(t1, t2, hi[0], absent, header, d),
                         dict(base_slots, e1=hi[0], e2=absent, diff=value_dict(d, raw_fmt, base_slots["unit"], magnitude=True), table2=t2),
                         axis="entity", absent=[absent]))
    return out


def build_split_pair(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    ta, tb = pkg["tables"]
    pa, pb = profile_table(ta), profile_table(tb)
    group, rng = GROUPS["split_pair"], ctx.rng
    noun = entity_noun(pa.key.header, pkg["domain"], pa)
    out: list[Draft] = []
    if noun == "对象":
        return out
    shared = [c for c in pa.metrics() if pb.column(c.header) and metric_slot(pa, c, pkg["domain"])][:2]
    for col in shared:
        ms = metric_slot(pa, col, pkg["domain"])
        va, vb = rows_with_values(pa, col), rows_with_values(pb, pb.column(col.header))
        pairs = [(a, b) for a in va for b in vb if a[1] != b[1]]
        if not pairs:
            continue
        rng.shuffle(pairs)
        base_slots = dict(ms, scope=pkg["scope"], noun=noun)
        for a, b in pairs[:2]:
            bid = _bid(pkg["package_id"], "xcmp", col.header, a[0], b[0])
            cmp = rng.choice(["gt", "lt"])
            truth_gt = a[1] > b[1]
            for label in ("SUPPORTS", "REFUTES"):
                # keep entity order (e1 from table A); REFUTES flips which entity is higher by swapping the
                # partner for another entity of table B on the other side of e1's value
                if label == "SUPPORTS":
                    partner, use_cmp = b, ("gt" if truth_gt else "lt")
                else:
                    flip = [x for x in vb if (x[1] > a[1]) == truth_gt and x[1] != a[1]]
                    if not flip:
                        continue
                    partner, use_cmp = rng.choice(flip), ("gt" if truth_gt else "lt")
                if cmp != use_cmp and label == "SUPPORTS":
                    use_cmp = "gt" if truth_gt else "lt"
                out.append(Draft("XT_COMPARE", group, label, bid, pkg["package_id"],
                                 _cmp_ops(ta["table_id"], tb["table_id"], a[0], partner[0], col.header, use_cmp),
                                 dict(base_slots, e1=a[0], e2=partner[0], cmp=use_cmp),
                                 axis="entity" if label == "REFUTES" else ""))
            absent = ctx.absent_entity(pkg, pkg["domain"], noun)
            if absent:
                out.append(Draft("XT_COMPARE", group, "NEI", bid, pkg["package_id"],
                                 _cmp_ops(ta["table_id"], tb["table_id"], a[0], absent, col.header, cmp),
                                 dict(base_slots, e1=a[0], e2=absent, cmp=cmp), axis="entity", absent=[absent]))
            # The second binding this family needs is the metric.  Generated alongside the entity variant
            # rather than as a fallback: entity substitution is unsafe on most of these pages (medal
            # tables, person names, ranked lists with a cutoff) and is rejected downstream, so a base whose
            # entity NEI dies still contributes one, and the family stops depending on five pages.
            nm = ctx.absent_metric(pkg, pkg["domain"], noun)
            if nm:
                out.append(Draft("XT_COMPARE", group, "NEI", bid, pkg["package_id"],
                                 _cmp_ops(ta["table_id"], tb["table_id"], a[0], b[0], nm[0], cmp),
                                 dict(base_slots, e1=a[0], e2=b[0], cmp=cmp, metric_name=nm[0],
                                      style=nm[1], unit=nm[2]), axis="metric", absent=[nm[0]]))
        if ms["numeric_surface_ok"] and ms["style"] != "rate":
            a, b = pairs[-1]
            hi, lo, t1, t2 = (a, b, ta, tb) if a[1] > b[1] else (b, a, tb, ta)
            out.extend(_diff_drafts("XT_DIFF", group, pkg, ctx, t1["table_id"], t2["table_id"], hi, lo, col.header,
                                    dict(base_slots), noun))
    return out


def build_category(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    t_cat, t_ent, t_met = pkg["tables"]
    pc, pe, pm = profile_table(t_cat), profile_table(t_ent), profile_table(t_met)
    if not (pc.key and pe.key and pm.key):
        return []
    group, rng = GROUPS["category_decomp"], ctx.rng
    cat_col = pkg["category"]
    noun = entity_noun(pm.key.header, pkg["domain"], pm)
    out: list[Draft] = []
    if noun == "对象" or not CATEGORY_HEADER_RE.search(cat_col):
        return out
    members: dict[str, list[str]] = {}
    ids = {entity_key(r[1]): entity_key(r[0]) for r in t_cat["rows"]}
    for r in t_ent["rows"]:
        members.setdefault(ids.get(entity_key(r[1]), ""), []).append(entity_key(r[0]))
    cats = [c for c, ms in members.items() if c and len(ms) >= 2 and 2 <= len(c) <= 12 and has_cjk(c) and not re.search(r"\d", c)]
    rng.shuffle(cats)
    all_cats = set(members)
    for col in [c for c in pm.metrics() if metric_slot(pm, c, pkg["domain"])][:2]:
        ms = metric_slot(pm, col, pkg["domain"])
        values = {n: v for n, v, _ in rows_with_values(pm, col)}
        base_slots = dict(ms, scope=pkg["scope"], noun=noun, cat_col=clean_name(cat_col))
        for cat in cats[:3]:
            mem = [m for m in members[cat] if m in values]
            if len(mem) < 2 or len(mem) != len(members[cat]):
                continue
            mode = rng.choice(["max", "min"])
            ext = (max if mode == "max" else min)(values[m] for m in mem)
            winners = [m for m in mem if values[m] == ext]
            if len(winners) != 1:
                continue
            outside = [n for n in values if n not in mem]
            distractors = [m for m in mem if m != winners[0]]
            bid = _bid(pkg["package_id"], "catarg", col.header, cat, mode)
            wrong = rng.choice(distractors) if (rng.random() < 0.6 or not outside) else rng.choice(outside)
            for label, claimed in (("SUPPORTS", winners[0]), ("REFUTES", wrong)):
                out.append(Draft("CAT_ARGEXT", group, label, bid, pkg["package_id"],
                                 _cat_arg_ops(t_cat["table_id"], t_ent["table_id"], t_met["table_id"], cat, col.header, mode, claimed),
                                 dict(base_slots, cat=cat, mode=mode, claimed=claimed),
                                 axis="entity" if label == "REFUTES" else ""))
            # NEI keeps the category and claimed member grounded and asks about a metric the package lacks.
            # (An absent category is NOT used: the claimed member's real category would contradict the claim.)
            nm = ctx.absent_metric(pkg, pkg["domain"], noun)
            if nm:
                out.append(Draft("CAT_ARGEXT", group, "NEI", bid, pkg["package_id"],
                                 _cat_arg_ops(t_cat["table_id"], t_ent["table_id"], t_met["table_id"], cat, nm[0], mode, winners[0]),
                                 dict(base_slots, metric_name=nm[0], style=nm[1], unit=nm[2], cat=cat, mode=mode,
                                      claimed=winners[0]), axis="metric", absent=[nm[0]]))
            # total over the same category.  Additive metrics only, and the category must be small
            # enough that the sum is a stated fact rather than an arithmetic exercise.
            if ms["numeric_surface_ok"] and ms["style"] != "rate" and 2 <= len(mem) <= 6:
                total = round(sum(values[m] for m in mem), 6)
                if total > 0 and len({values[m] for m in mem}) >= 2:
                    raw_t = next(raw for n_, _v, raw in rows_with_values(pm, col) if n_ == mem[0])
                    sbid = _bid(pkg["package_id"], "catsum", col.header, cat)
                    off = max(abs(min(values[m] for m in mem)), 1.0)
                    for label, tv in (("SUPPORTS", total), ("REFUTES", round(total + off, 6))):
                        out.append(Draft("CAT_SUM", group, label, sbid, pkg["package_id"],
                                         _cat_sum_ops(t_cat["table_id"], t_ent["table_id"], t_met["table_id"],
                                                      cat, col.header, tv),
                                         dict(base_slots, cat=cat, total=value_dict(tv, raw_t, ms["unit"])),
                                         axis="number" if label == "REFUTES" else ""))
    for cat in cats[:3]:
        n = len(members[cat])
        bid = _bid(pkg["package_id"], "catcount", cat)
        base_slots = dict(scope=pkg["scope"], noun=noun, cat_col=clean_name(cat_col), metric_name="", style="count", unit="")
        for label, nn in (("SUPPORTS", n), ("REFUTES", n + rng.choice([-1, 1]) if n > 2 else n + 1)):
            out.append(Draft("CAT_COUNT", group, label, bid, pkg["package_id"],
                             _cat_count_ops(t_cat["table_id"], t_ent["table_id"], cat, nn),
                             dict(base_slots, cat=cat, n=nn), axis="number" if label == "REFUTES" else ""))
    return out


def _absent_category(ctx: Context, pkg: dict[str, Any], present: set[str]) -> str | None:
    """A category value of the same attribute taken from another package of the same domain."""
    pool = ctx.categories.get((pkg["domain"], pkg["category"]), [])
    texts = ctx.package_text(pkg)
    cands = [c for c in pool if c not in present and not any(c in t for t in texts)]
    return ctx.rng.choice(cands) if cands else None


def build_rank_detail(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    t_rank, t_det = pkg["tables"]
    pr, pd = profile_table(t_rank), profile_table(t_det)
    rank_col, fk = pr.rank(), pd.column(pkg["fk_col"])
    if not rank_col or not fk:
        return []
    group, rng = GROUPS["rank_detail"], ctx.rng
    rank_noun, member_noun = entity_noun(pr.key.header, pkg["domain"], pr), entity_noun(pd.key.header, pkg["domain"], pd)
    out: list[Draft] = []
    if "对象" in (rank_noun, member_noun) or rank_noun == member_noun:
        return out
    ranks = {}
    for r in range(len(pr.rows)):
        k = parse_number(pr.cell(r, rank_col))
        if k is not None and float(k).is_integer():
            ranks.setdefault(int(k), []).append(entity_key(pr.cell(r, pr.key)))
    ranks = {k: v[0] for k, v in ranks.items() if len(v) == 1}
    team_of = {entity_key(pd.cell(r, pd.key)): entity_key(pd.cell(r, fk)) for r in range(len(pd.rows))}
    rank_of_team = {join_key(t): k for k, t in ranks.items()}
    # every metric the detail table carries, not just the first: a squad table lists goals, assists and
    # appearances, and each is a separate grounded fact about the same join
    for col in [c for c in pd.metrics() if metric_slot(pd, c, pkg["domain"])][:3]:
        ms = metric_slot(pd, col, pkg["domain"])
        vals = {n: v for n, v, _ in rows_with_values(pd, col)}
        base_slots = dict(ms, scope=pkg["scope"], noun=member_noun, rank_noun=rank_noun)
        ks = sorted(ranks)
        rng.shuffle(ks)
        made = 0
        for k in ks:
            team = ranks[k]
            mem = [p for p, tm in team_of.items() if join_key(tm) == join_key(team) and p in vals]
            if len(mem) < 2:
                continue
            best = max(vals[p] for p in mem)
            winners = [p for p in mem if vals[p] == best]
            if len(winners) != 1:
                continue
            others = [p for p in mem if p != winners[0]]
            bid = _bid(pkg["package_id"], "rjtop", k, col.header)
            for label, claimed in (("SUPPORTS", winners[0]), ("REFUTES", rng.choice(others))):
                out.append(Draft("RJ_RANK_TOP", group, label, bid, pkg["package_id"],
                                 _rj_top_ops(t_rank["table_id"], t_det["table_id"], rank_col.header, fk.header, k, col.header, claimed),
                                 dict(base_slots, k=k, claimed=claimed), axis="entity" if label == "REFUTES" else ""))
            nm = ctx.absent_metric(pkg, pkg["domain"], member_noun)
            if nm:
                out.append(Draft("RJ_RANK_TOP", group, "NEI", bid, pkg["package_id"],
                                 _rj_top_ops(t_rank["table_id"], t_det["table_id"], rank_col.header, fk.header, k, nm[0], winners[0]),
                                 dict(base_slots, metric_name=nm[0], style=nm[1], k=k, claimed=winners[0]),
                                 axis="metric", absent=[nm[0]]))
            made += 1
            if made >= 8:
                break     # one base per ranked team; football squads carry a single metric, so the
                          # ranks themselves are the only axis left to vary
        # Value form of the same join: the highest metric value *inside* the team ranked k.  When the
        # detail table is a partial leaders list, a ranked team with no row in it makes that value
        # unreadable, which is the family's only sound NEI (substituting a player name is unsafe -
        # transliterations vary - and substituting the team would be refuted by the ranking table).
        teams_in_detail = {join_key(t) for t in team_of.values()}
        detail_partial = len(teams_in_detail) < len(ranks)
        raw_of = {n: raw for n, _v, raw in rows_with_values(pd, col)}
        made = 0
        for k in ks:
            if made >= 3:
                break
            team = ranks[k]
            mem = [p for p, tm in team_of.items() if join_key(tm) == join_key(team) and p in vals]
            bid = _bid(pkg["package_id"], "rjtopval", k, col.header)
            if len(mem) >= 2:
                best = max(vals[p] for p in mem)
                winners = [p for p in mem if vals[p] == best]
                others = sorted({vals[p] for p in mem if vals[p] != best})
                if len(winners) != 1 or not others:
                    continue
                for label, v in (("SUPPORTS", best), ("REFUTES", others[-1])):
                    out.append(Draft("RJ_RANK_TOPVAL", group, label, bid, pkg["package_id"],
                                     _rj_topval_ops(t_rank["table_id"], t_det["table_id"], rank_col.header,
                                                    fk.header, k, col.header, v),
                                     dict(base_slots, k=k, value=value_dict(v, raw_of[winners[0]], ms["unit"])),
                                     axis="number" if label == "REFUTES" else ""))
                made += 1
            elif not mem and detail_partial and ms["numeric_surface_ok"]:
                observed = sorted(set(vals.values()))
                # a column that only ever holds one or two small values (停赛场次 0/1) makes a degenerate
                # claim: "the highest is 1" is guessable without the table
                if len(observed) < 3 or max(observed) < 3:
                    continue
                # the claim states a *maximum*, so a plausible value comes from the upper half of the
                # column; a bottom value would read as implausible independently of the tables
                v = rng.choice(observed[len(observed) // 2:])
                src = next(p for p in vals if vals[p] == v)
                out.append(Draft("RJ_RANK_TOPVAL", group, "NEI", bid, pkg["package_id"],
                                 _rj_topval_ops(t_rank["table_id"], t_det["table_id"], rank_col.header,
                                                fk.header, k, col.header, v),
                                 dict(base_slots, k=k, value=value_dict(v, raw_of[src], ms["unit"])),
                                 axis="join_key", absent=[]))
                made += 1
    players = [p for p in team_of if good_entity(p)]
    rng.shuffle(players)
    made = 0
    for p in players:
        k = rank_of_team.get(join_key(team_of[p]))
        base_slots = dict(scope=pkg["scope"], rank_noun=rank_noun, member=p, metric_name="", style="count", unit="")
        bid = _bid(pkg["package_id"], "rjrank", p)
        if k is None:
            # a member whose team is missing from the ranking table is NOT used as NEI: team names differ across
            # tables (sponsor renames) and the ranking may be split into groups, so "missing" is unreliable
            continue
        if made >= 5:
            continue
        wrong = k + rng.choice([-1, 1]) if k > 1 else k + 1
        for label, kk in (("SUPPORTS", k), ("REFUTES", wrong)):
            out.append(Draft("RJ_MEMBER_RANK", group, label, bid, pkg["package_id"],
                             _rj_rank_ops(t_rank["table_id"], t_det["table_id"], rank_col.header, fk.header, p, kk),
                             dict(base_slots, k=kk), axis="rank" if label == "REFUTES" else ""))
        absent = ctx.absent_entity(pkg, pkg["domain"], entity_noun(pd.key.header, pkg["domain"], pd))
        if absent:
            out.append(Draft("RJ_MEMBER_RANK", group, "NEI", bid, pkg["package_id"],
                             _rj_rank_ops(t_rank["table_id"], t_det["table_id"], rank_col.header, fk.header, absent, k),
                             dict(base_slots, member=absent, k=k), axis="entity", absent=[absent]))
        made += 1
    return out


def build_hub_profile(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    th, tm = pkg["tables"]
    ph, pm = profile_table(th), profile_table(tm)
    group, rng = GROUPS["hub_profile"], ctx.rng
    noun = entity_noun(ph.key.header, pkg["domain"], ph)
    out: list[Draft] = []
    metrics = [c for c in pm.metrics() if metric_slot(pm, c, pkg["domain"]) and metric_slot(pm, c, pkg["domain"])["numeric_surface_ok"]]
    if not metrics:
        return out
    if noun == "对象":
        return out
    for attr_header in [h for h in pkg["attr_cols"] if ATTR_HEADER_RE.search(h)][:2]:
        acol = ph.column(attr_header)
        col = rng.choice(metrics)
        ms = metric_slot(pm, col, pkg["domain"])
        vals = {join_key(n): (n, v, raw) for n, v, raw in rows_with_values(pm, col)}
        attrs = {entity_key(ph.cell(r, ph.key)): entity_key(ph.cell(r, acol)) for r in range(len(ph.rows))}
        attrs = {e: a for e, a in attrs.items() if good_entity(e) and a and 2 <= len(a) <= 14 and not re.search(r"\d", a)}
        distinct_attr = sorted(set(attrs.values()))
        if len(distinct_attr) < 3:
            continue
        # four grounded entities per (package, attribute column), not two: each is a separate fact, and
        # absent_metric is drawn again per entity, so the borrowed metric name varies across them
        ents = [e for e in attrs if join_key(e) in vals]
        rng.shuffle(ents)
        base_slots = dict(ms, scope=pkg["scope"], noun=noun, attr_col=clean_name(attr_header))
        for e in ents[:4]:
            name, v, raw = vals[join_key(e)]
            a = attrs[e]
            bid = _bid(pkg["package_id"], "hp", attr_header, col.header, e)
            out.append(Draft("HP_AND", group, "SUPPORTS", bid, pkg["package_id"],
                             _hp_ops(th["table_id"], tm["table_id"], e, attr_header, a, col.header, v),
                             dict(base_slots, e=e, attr=a, value=value_dict(v, raw, ms["unit"]))))
            if rng.random() < 0.5:
                wrong_a = rng.choice([x for x in distinct_attr if x != a])
                out.append(Draft("HP_AND", group, "REFUTES", bid, pkg["package_id"],
                                 _hp_ops(th["table_id"], tm["table_id"], e, attr_header, wrong_a, col.header, v),
                                 dict(base_slots, e=e, attr=wrong_a, value=value_dict(v, raw, ms["unit"])), axis="attribute"))
            else:
                other_vals = sorted({x[1] for x in vals.values() if x[1] != v})
                if not other_vals:
                    continue
                wv = min(other_vals, key=lambda x: abs(x - v))
                out.append(Draft("HP_AND", group, "REFUTES", bid, pkg["package_id"],
                                 _hp_ops(th["table_id"], tm["table_id"], e, attr_header, a, col.header, wv),
                                 dict(base_slots, e=e, attr=a, value=value_dict(wv, raw, ms["unit"])), axis="number"))
            nm = ctx.absent_metric(pkg, pkg["domain"], noun)
            # a unit is required here and nowhere else: HP_AND is the only skeleton that states a bare
            # number next to the metric name, and '奖金为14.5' reads as malformed rather than as a claim
            if nm and nm[2] and nm[3]:
                # the number comes from a real row of this metric elsewhere in the benchmark, drawn from
                # the middle of its observed range, so the claim fails on the missing column and not on a
                # magnitude a reader could dismiss without consulting any table
                obs = sorted(nm[3])
                mv, mraw = obs[len(obs) // 3: (2 * len(obs)) // 3 + 1][
                    rng.randrange(len(obs[len(obs) // 3: (2 * len(obs)) // 3 + 1]))]
                out.append(Draft("HP_AND", group, "NEI", bid, pkg["package_id"],
                                 _hp_ops(th["table_id"], tm["table_id"], e, attr_header, a, nm[0], mv),
                                 dict(base_slots, e=e, attr=a, metric_name=nm[0], style=nm[1], unit=nm[2],
                                      value=value_dict(mv, mraw, nm[2])),
                                 axis="metric", absent=[nm[0]], value_source="corpus"))
            # Comparative form of the same hub fact.  Two grounded entities, no stated magnitude, so a
            # metric-slot NEI here can use a name the corpus never puts a number to (the curated
            # per-domain vocabulary), which the value form cannot.
            partner = next((x for x in ents if x != e and join_key(x) in vals
                            and vals[join_key(x)][1] != v), None)
            if partner is not None:
                pv = vals[join_key(partner)][1]
                cbid = _bid(pkg["package_id"], "hpcmp", attr_header, col.header, e)
                for label, cmp in (("SUPPORTS", "gt" if v > pv else "lt"),
                                   ("REFUTES", "lt" if v > pv else "gt")):
                    out.append(Draft("HP_CMP", group, label, cbid, pkg["package_id"],
                                     _hp_cmp_ops(th["table_id"], tm["table_id"], e, attr_header, a,
                                                 partner, col.header, cmp),
                                     dict(base_slots, e1=e, e2=partner, attr=a, cmp=cmp),
                                     axis="direction" if label == "REFUTES" else ""))
                nc = ctx.absent_metric(pkg, pkg["domain"], noun)
                if nc:
                    out.append(Draft("HP_CMP", group, "NEI", cbid, pkg["package_id"],
                                     _hp_cmp_ops(th["table_id"], tm["table_id"], e, attr_header, a,
                                                 partner, nc[0], "gt" if v > pv else "lt"),
                                     dict(base_slots, e1=e, e2=partner, attr=a, metric_name=nc[0],
                                          style=nc[1], unit=nc[2], cmp="gt" if v > pv else "lt"),
                                     axis="metric", absent=[nc[0]]))
            # entity NEI: both stated bindings are real values read off this page, but they are asserted
            # of an entity that appears on neither table, so neither conjunct can be looked up.  This is
            # the plain "entity" binding of the NEI definition and does not depend on the profile table
            # being partial, unlike the join-key form below.
            ae = ctx.absent_entity(pkg, pkg["domain"], noun)
            if ae:
                out.append(Draft("HP_AND", group, "NEI", bid, pkg["package_id"],
                                 _hp_ops(th["table_id"], tm["table_id"], ae, attr_header, a, col.header, v),
                                 dict(base_slots, e=ae, attr=a, value=value_dict(v, raw, ms["unit"])),
                                 axis="entity", absent=[ae], value_source="corpus"))
        # join-key NEI: the hub row (and its attribute) is grounded, but the entity has no row in the
        # profile table, so the metric cannot be read.  Only for amount/rate metrics, where a missing row
        # never means zero (unlike medal counts), and only when the profile table is visibly partial.
        if ms["style"] in ("amount", "rate") and pkg["domain"] != "medals" and len(pm.rows) < len(ph.rows):
            missing = [e for e in attrs if join_key(e) not in vals and good_entity(e)]
            rng.shuffle(missing)
            observed = sorted(vals.values(), key=lambda x: x[1])
            for e in missing[:2]:
                if not observed:
                    break
                a = attrs[e]
                # take the surface form from a real row so the number is rendered the way the column
                # renders it (decimals, thousands separators), not as a bare repr
                _name, v, raw = rng.choice(observed)
                bid = _bid(pkg["package_id"], "hpjk", attr_header, col.header, e)
                out.append(Draft("HP_AND", group, "NEI", bid, pkg["package_id"],
                                 _hp_ops(th["table_id"], tm["table_id"], e, attr_header, a, col.header, v),
                                 dict(base_slots, e=e, attr=a, value=value_dict(v, raw, ms["unit"])),
                                 axis="join_key", absent=[]))
    return out


def build_join_filter(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    th, tm = pkg["tables"]
    ph, pm = profile_table(th), profile_table(tm)
    group, rng = GROUPS["join_filter"], ctx.rng
    noun = entity_noun(ph.key.header, pkg["domain"], ph)
    out: list[Draft] = []
    if noun == "对象":
        return out
    # more than one filter column (分区 and 国家 partition the same table differently) and more than two
    # groups per column: each is a separate closed set, and this family survives on only three pages
    for fcol_header in [h for h in pkg["filter_cols"] if CATEGORY_HEADER_RE.search(h)][:2]:
        fcol = ph.column(fcol_header)
        groups: dict[str, list[str]] = {}
        for r in range(len(ph.rows)):
            groups.setdefault(entity_key(ph.cell(r, fcol)), []).append(entity_key(ph.cell(r, ph.key)))
        for col in [c for c in pm.metrics() if metric_slot(pm, c, pkg["domain"])][:2]:
            ms = metric_slot(pm, col, pkg["domain"])
            vals = {join_key(n): (n, v, raw) for n, v, raw in rows_with_values(pm, col)}
            base_slots = dict(ms, scope=pkg["scope"], noun=noun, attr_col=clean_name(fcol_header))
            opts = [(a, mem) for a, mem in groups.items() if a and len(mem) >= 2 and not re.search(r"\d", a)]
            rng.shuffle(opts)
            for attr, mem in opts[:4]:
                if not all(join_key(m) in vals for m in mem):
                    continue      # incomplete join (renames / sliced tables): no instance, never NEI
                series = [(m, vals[join_key(m)][1]) for m in mem]
                mode = rng.choice(["max", "min"])
                ext = (max if mode == "max" else min)(v for _, v in series)
                winners = [m for m, v in series if v == ext]
                if len(winners) == 1:
                    bid = _bid(pkg["package_id"], "jfarg", attr, col.header, mode)
                    wrong = rng.choice([m for m, _ in series if m != winners[0]])
                    for label, claimed in (("SUPPORTS", winners[0]), ("REFUTES", wrong)):
                        out.append(Draft("JF_ARGEXT", group, label, bid, pkg["package_id"],
                                         _jf_arg_ops(th["table_id"], tm["table_id"], fcol_header, attr, col.header, mode, claimed),
                                         dict(base_slots, attr=attr, mode=mode, claimed=claimed),
                                         axis="entity" if label == "REFUTES" else ""))
                    nm = ctx.absent_metric(pkg, pkg["domain"], noun)
                    if nm:
                        out.append(Draft("JF_ARGEXT", group, "NEI", bid, pkg["package_id"],
                                         _jf_arg_ops(th["table_id"], tm["table_id"], fcol_header, attr, nm[0], mode, winners[0]),
                                         dict(base_slots, metric_name=nm[0], style=nm[1], attr=attr, mode=mode,
                                              claimed=winners[0]), axis="metric", absent=[nm[0]]))
                if ms["numeric_surface_ok"] and len(series) >= 3:
                    values = sorted(v for _, v in series)
                    t = values[len(values) // 2]
                    raw_t = vals[join_key(series[0][0])][2]
                    n = sum(1 for v in values if v > t)
                    bid = _bid(pkg["package_id"], "jfcnt", attr, col.header)
                    for label, nn in (("SUPPORTS", n), ("REFUTES", n + 1 if n < len(values) - 1 or n == 0 else n - 1)):
                        out.append(Draft("JF_COUNT_GT", group, label, bid, pkg["package_id"],
                                         _jf_count_ops(th["table_id"], tm["table_id"], fcol_header, attr, col.header, t, nn),
                                         dict(base_slots, attr=attr, threshold=value_dict(t, raw_t, ms["unit"]), n=nn),
                                         axis="number" if label == "REFUTES" else ""))
                # total over the filtered set.  Only for additive metrics: summing a rate or an average
                # is meaningless, and a sum is only checkable when the filtered set is closed (gated).
                if ms["numeric_surface_ok"] and ms["style"] != "rate" and 2 <= len(series) <= 8:
                    total = round(sum(v for _, v in series), 6)
                    raw_t = vals[join_key(series[0][0])][2]
                    if total > 0 and len({v for _, v in series}) >= 2:
                        bid = _bid(pkg["package_id"], "jfsum", attr, col.header)
                        off = max(abs(min(v for _, v in series)), 1.0)
                        for label, tv in (("SUPPORTS", total), ("REFUTES", round(total + off, 6))):
                            out.append(Draft("JF_SUM", group, label, bid, pkg["package_id"],
                                             _jf_sum_ops(th["table_id"], tm["table_id"], fcol_header, attr, col.header, tv),
                                             dict(base_slots, attr=attr, total=value_dict(tv, raw_t, ms["unit"])),
                                             axis="number" if label == "REFUTES" else ""))
    return out


def build_period_pair(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    t1, t2 = pkg["tables"]
    p1, p2 = profile_table(t1), profile_table(t2)
    if not (p1.key and p2.key):
        return []
    y1, y2 = pkg["periods"]
    group, rng = GROUPS["period_pair"], ctx.rng
    noun = entity_noun(p2.key.header, pkg["domain"], p2)
    out: list[Draft] = []
    shared = [c for c in p2.metrics() if p1.column(c.header) and metric_slot(p2, c, pkg["domain"])
              and not NON_PERIOD_METRICS.search(c.header)][:2]
    for col in shared:
        ms = metric_slot(p2, col, pkg["domain"])
        v1 = {join_key(n): (n, v, raw) for n, v, raw in rows_with_values(p1, p1.column(col.header))}
        v2 = {join_key(n): (n, v, raw) for n, v, raw in rows_with_values(p2, col)}
        both = [k for k in v2 if k in v1 and v1[k][1] != v2[k][1]]
        if len(both) < 3:
            continue
        ups = sum(1 for k in both if v2[k][1] > v1[k][1])
        balanced = min(ups, len(both) - ups) >= 0.3 * len(both)
        base_slots = dict(ms, scope=pkg["scope"], noun=noun, y1=y1, y2=y2)
        rng.shuffle(both)
        for k in both[:3 if balanced else 1]:
            name = v2[k][0]
            up = v2[k][1] > v1[k][1]
            bid = _bid(pkg["package_id"], "cpdir", col.header, k)
            if balanced:
                for label, direction in (("SUPPORTS", "up" if up else "down"), ("REFUTES", "down" if up else "up")):
                    out.append(Draft("CP_DIRECTION", group, label, bid, pkg["package_id"],
                                     _cp_dir_ops(t1["table_id"], t2["table_id"], name, col.header, direction),
                                     dict(base_slots, e=name, direction=direction), axis="direction" if label == "REFUTES" else ""))
                nm = ctx.absent_metric(pkg, pkg["domain"], noun)
                if nm and rng.random() < 0.5:
                    direction = rng.choice(["up", "down"])
                    out.append(Draft("CP_DIRECTION", group, "NEI", bid, pkg["package_id"],
                                     _cp_dir_ops(t1["table_id"], t2["table_id"], name, nm[0], direction),
                                     dict(base_slots, metric_name=nm[0], style=nm[1], unit=nm[2], e=name, direction=direction),
                                     axis="metric", absent=[nm[0]]))
            if ms["numeric_surface_ok"] and ms["style"] != "rate":
                d = round(v2[k][1] - v1[k][1], 6)
                raw = v2[k][2]
                decimals = max(_decimals(raw), _decimals(v1[k][2]))
                step = max(1.0, round(abs(d) * 0.3)) if decimals == 0 else round(max(abs(d) * 0.3, 10 ** -decimals), decimals)
                wrong = round(abs(d) + rng.choice([-1, 1]) * step, decimals)
                wrong = wrong if wrong > 0 else round(abs(d) + step, decimals)
                bid2 = _bid(pkg["package_id"], "cpdiff", col.header, k)
                direction = "up" if d > 0 else "down"
                for label, mag in (("SUPPORTS", abs(d)), ("REFUTES", wrong)):
                    signed = mag if d > 0 else -mag
                    out.append(Draft("CP_DIFF", group, label, bid2, pkg["package_id"],
                                     _cp_diff_ops(t1["table_id"], t2["table_id"], name, col.header, signed),
                                     dict(base_slots, e=name, direction=direction,
                                          diff=value_dict(mag, "1." + "0" * decimals if decimals else "1", ms["unit"], magnitude=True)),
                                     axis="number" if label == "REFUTES" else ""))
    return out


def _has_name_variant(name: str, others: list[str]) -> bool:
    """True if `name` probably denotes an entity listed under another name (sponsor renames: 北京国安 ~ 北京中赫国安)."""
    core = join_key(name)
    tail = core[-2:] if len(core) >= 3 else core
    head = core[:2]
    for other in others:
        o = join_key(other)
        if not o:
            continue
        if core in o or o in core or (len(core) >= 3 and tail in o) or (len(o) >= 3 and o[-2:] in core and head == o[:2]):
            return True
    return False


def build_category_partner(pkg: dict[str, Any], ctx: Context) -> list[Draft]:
    """Category winner on one metric, plus that winner's value for a metric held by a fourth table."""
    t_cat, t_ent, t_met, t_par = pkg["tables"]
    pc, pe, pm, pp = (profile_table(t) for t in pkg["tables"])
    if not (pc.key and pe.key and pm.key and pp.key):
        return []
    group, rng = GROUPS["category_partner"], ctx.rng
    noun = entity_noun(pm.key.header, pkg["domain"], pm)
    if noun == "对象":
        return []
    own = {clean_name(h) for h in t_met["headers"]}
    m1s = [c for c in pm.metrics() if metric_slot(pm, c, pkg["domain"])]
    m2s = [c for c in pp.metrics() if metric_slot(pp, c, pkg["domain"]) and clean_name(c.header) not in own]
    if not m1s or not m2s:
        return []
    col, col2 = m1s[0], rng.choice(m2s)
    ms, ms2 = metric_slot(pm, col, pkg["domain"]), metric_slot(pp, col2, pkg["domain"])
    vals = {n: v for n, v, _ in rows_with_values(pm, col)}
    pvals = {join_key(n): v for n, v, _ in rows_with_values(pp, col2)}
    # read the decomposed tables by position, as build_category does: profile_table may pick the
    # synthetic 类别编号 column as the key, and the claim must name the real category, not its id
    ids = {entity_key(r[1]): entity_key(r[0]) for r in t_cat["rows"]}
    members_of: dict[str, list[str]] = {}
    for r in t_ent["rows"]:
        members_of.setdefault(ids.get(entity_key(r[1]), ""), []).append(entity_key(r[0]))
    out: list[Draft] = []
    if not CATEGORY_HEADER_RE.search(pkg["category"]):
        return out
    base_slots = dict(ms, scope=pkg["scope"], noun=noun, cat_col=clean_name(pkg["category"]),
                      metric2=ms2["metric_name"], style2=ms2["style"])
    cats = [c for c, mm in members_of.items()
            if c and len(mm) >= 3 and 2 <= len(c) <= 12 and has_cjk(c) and not re.search(r"\d", c)]
    rng.shuffle(cats)
    made = 0
    for cat in cats:
        if made >= 2:
            break
        mem = [m for m in members_of[cat] if m in vals]
        if len(mem) < 3 or len(mem) != len(members_of[cat]):
            continue
        mode = rng.choice(["max", "min"])
        ext = (max if mode == "max" else min)(vals[m] for m in mem)
        winners = [m for m in mem if vals[m] == ext]
        if len(winners) != 1 or join_key(winners[0]) not in pvals:
            continue
        w = winners[0]
        wv = pvals[join_key(w)]
        # the comparison partner is any other entity of the fourth table, inside or outside the category
        others = [n for n, v in pvals.items() if n != join_key(w) and v != wv]
        if not others:
            continue
        raw_of = {join_key(n): n for n, _v, _r in rows_with_values(pp, col2)}
        other = raw_of[rng.choice(others)]
        cmp = "gt" if wv > pvals[join_key(other)] else "lt"
        bid = _bid(pkg["package_id"], "catpar", cat, col.header, col2.header)
        out.append(Draft("CAT_PARTNER", group, "SUPPORTS", bid, pkg["package_id"],
                         _cat_partner_ops(t_cat["table_id"], t_ent["table_id"], t_met["table_id"],
                                          t_par["table_id"], cat, col.header, mode, w, col2.header, other, cmp),
                         dict(base_slots, cat=cat, mode=mode, claimed=w, e2=other, cmp=cmp)))
        if rng.random() < 0.5:
            wrong = rng.choice([m for m in mem if m != w])
            out.append(Draft("CAT_PARTNER", group, "REFUTES", bid, pkg["package_id"],
                             _cat_partner_ops(t_cat["table_id"], t_ent["table_id"], t_met["table_id"],
                                              t_par["table_id"], cat, col.header, mode, wrong, col2.header, other, cmp),
                             dict(base_slots, cat=cat, mode=mode, claimed=wrong, e2=other, cmp=cmp),
                             axis="entity"))
        else:
            flip = "lt" if cmp == "gt" else "gt"
            out.append(Draft("CAT_PARTNER", group, "REFUTES", bid, pkg["package_id"],
                             _cat_partner_ops(t_cat["table_id"], t_ent["table_id"], t_met["table_id"],
                                              t_par["table_id"], cat, col.header, mode, w, col2.header, other, flip),
                             dict(base_slots, cat=cat, mode=mode, claimed=w, e2=other, cmp=flip),
                             axis="direction"))
        nm = ctx.absent_metric(pkg, pkg["domain"], noun)
        if nm:
            out.append(Draft("CAT_PARTNER", group, "NEI", bid, pkg["package_id"],
                             _cat_partner_ops(t_cat["table_id"], t_ent["table_id"], t_met["table_id"],
                                              t_par["table_id"], cat, col.header, mode, w, nm[0], other, cmp),
                             dict(base_slots, cat=cat, mode=mode, claimed=w, e2=other, cmp=cmp,
                                  metric2=nm[0], style2=nm[1]), axis="metric", absent=[nm[0]]))
        made += 1
    return out


BUILDERS = {"category_partner": build_category_partner, "single": build_single, "split_pair": build_split_pair, "category_decomp": build_category,
            "rank_detail": build_rank_detail, "hub_profile": build_hub_profile, "join_filter": build_join_filter,
            "period_pair": build_period_pair}


# =========================================================================== operator templates
def _cmp_ops(t1, t2, e1, e2, col, cmp):
    return [{"op": "LOOKUP", "table": t1, "key": e1, "col": col, "out": "a"},
            {"op": "LOOKUP", "table": t2, "key": e2, "col": col, "out": "b"},
            {"op": "COMPARE", "left": "$a", "cmp": cmp, "right": "$b", "out": "result"}]


def _sup_ops(t, col, mode, claimed):
    return [{"op": "ARGEXT", "table": t, "col": col, "mode": mode, "out": "top"},
            {"op": "LOOKUP", "table": t, "key": claimed, "col": col, "out": "claimed_value"},
            {"op": "COMPARE", "left": "$top", "cmp": "eq", "right": claimed, "out": "result"}]


def _rank_ops(t, e, col, k):
    return [{"op": "RANK", "table": t, "key": e, "col": col, "order": "desc", "out": "rank"},
            {"op": "COMPARE", "left": "$rank", "cmp": "eq", "right": k, "out": "result"}]


def _diff_ops(t1, t2, e1, e2, col, d):
    return [{"op": "LOOKUP", "table": t1, "key": e1, "col": col, "out": "a"},
            {"op": "LOOKUP", "table": t2, "key": e2, "col": col, "out": "b"},
            {"op": "SUB", "left": "$a", "right": "$b", "out": "diff"},
            {"op": "COMPARE", "left": "$diff", "cmp": "eq", "right": d, "out": "result"}]


def _cat_arg_ops(tc, te, tm, cat, col, mode, claimed):
    return [{"op": "ATTR", "table": tc, "key": cat, "col": "类别编号", "out": "cid"},
            {"op": "JOIN", "table": te, "fk_col": "类别编号", "eq": "$cid", "out": "members"},
            {"op": "VALUES", "table": tm, "keys": "$members", "col": col, "out": "vals"},
            {"op": "ARGEXT_OF", "keys": "$members", "values": "$vals", "mode": mode, "out": "top"},
            {"op": "COMPARE", "left": "$top", "cmp": "eq", "right": claimed, "out": "result"}]


def _st_count_ops(t, col, threshold, n):
    return [{"op": "COUNT_ABOVE", "table": t, "col": col, "threshold": threshold, "out": "n"},
            {"op": "COMPARE", "left": "$n", "cmp": "eq", "right": n, "out": "result"}]


def _cat_sum_ops(tc, te, tm, cat, col, total):
    return [{"op": "ATTR", "table": tc, "key": cat, "col": "类别编号", "out": "cid"},
            {"op": "JOIN", "table": te, "fk_col": "类别编号", "eq": "$cid", "out": "members"},
            {"op": "VALUES", "table": tm, "keys": "$members", "col": col, "out": "vals"},
            {"op": "SUM", "values": "$vals", "out": "total"},
            {"op": "COMPARE", "left": "$total", "cmp": "eq", "right": total, "out": "result"}]


def _cat_partner_ops(tc, te, tm, tp, cat, col, mode, claimed, col2, other, cmp):
    """Four tables: category -> members -> ranked on one metric -> the winner compared with another
    entity on a metric only the fourth table carries.  Stated as a comparison rather than as a value,
    so the claim never has to render a magnitude whose unit the source header does not spell out."""
    return [{"op": "ATTR", "table": tc, "key": cat, "col": "类别编号", "out": "cid"},
            {"op": "JOIN", "table": te, "fk_col": "类别编号", "eq": "$cid", "out": "members"},
            {"op": "VALUES", "table": tm, "keys": "$members", "col": col, "out": "vals"},
            {"op": "ARGEXT_OF", "keys": "$members", "values": "$vals", "mode": mode, "out": "top"},
            {"op": "COMPARE", "left": "$top", "cmp": "eq", "right": claimed, "out": "c1"},
            {"op": "LOOKUP", "table": tp, "key": claimed, "col": col2, "out": "a"},
            {"op": "LOOKUP", "table": tp, "key": other, "col": col2, "out": "b"},
            {"op": "COMPARE", "left": "$a", "cmp": cmp, "right": "$b", "out": "c2"},
            {"op": "AND", "args": ["$c1", "$c2"], "out": "result"}]


def _cat_count_ops(tc, te, cat, n):
    return [{"op": "ATTR", "table": tc, "key": cat, "col": "类别编号", "out": "cid"},
            {"op": "JOIN", "table": te, "fk_col": "类别编号", "eq": "$cid", "out": "members"},
            {"op": "COUNT", "values": "$members", "out": "n"},
            {"op": "COMPARE", "left": "$n", "cmp": "eq", "right": n, "out": "result"}]


def _rj_top_ops(tr, td, rank_col, fk, k, col, claimed):
    return [{"op": "AT_RANK", "table": tr, "rank_col": rank_col, "k": k, "out": "team"},
            {"op": "JOIN", "table": td, "fk_col": fk, "eq": "$team", "out": "members"},
            {"op": "VALUES", "table": td, "keys": "$members", "col": col, "out": "vals"},
            {"op": "ARGEXT_OF", "keys": "$members", "values": "$vals", "mode": "max", "out": "top"},
            {"op": "COMPARE", "left": "$top", "cmp": "eq", "right": claimed, "out": "result"}]


def _rj_topval_ops(tr, td, rank_col, fk, k, col, value):
    return [{"op": "AT_RANK", "table": tr, "rank_col": rank_col, "k": k, "out": "team"},
            {"op": "JOIN", "table": td, "fk_col": fk, "eq": "$team", "out": "members"},
            {"op": "VALUES", "table": td, "keys": "$members", "col": col, "out": "vals"},
            {"op": "EXTVAL", "values": "$vals", "mode": "max", "out": "top"},
            {"op": "COMPARE", "left": "$top", "cmp": "eq", "right": value, "out": "result"}]


def _rj_rank_ops(tr, td, rank_col, fk, member, k):
    return [{"op": "ATTR", "table": td, "key": member, "col": fk, "out": "team"},
            {"op": "LOOKUP", "table": tr, "key": "$team", "col": rank_col, "out": "rank"},
            {"op": "COMPARE", "left": "$rank", "cmp": "eq", "right": k, "out": "result"}]


def _hp_ops(th, tm, e, attr_col, attr, col, value):
    return [{"op": "ATTR", "table": th, "key": e, "col": attr_col, "out": "attr"},
            {"op": "COMPARE", "left": "$attr", "cmp": "eq", "right": attr, "out": "c1"},
            {"op": "LOOKUP", "table": tm, "key": e, "col": col, "out": "v"},
            {"op": "COMPARE", "left": "$v", "cmp": "eq", "right": value, "out": "c2"},
            {"op": "AND", "args": ["$c1", "$c2"], "out": "result"}]


def _hp_cmp_ops(th, tm, e1, attr_col, attr, e2, col, cmp):
    """Hub attribute of e1, and e1's metric compared against another entity's, rather than against a
    stated number.  The comparative form is what lets a metric with no observed value in the corpus be
    used for a metric-slot NEI: the claim never has to name a magnitude."""
    return [{"op": "ATTR", "table": th, "key": e1, "col": attr_col, "out": "attr"},
            {"op": "COMPARE", "left": "$attr", "cmp": "eq", "right": attr, "out": "c1"},
            {"op": "LOOKUP", "table": tm, "key": e1, "col": col, "out": "a"},
            {"op": "LOOKUP", "table": tm, "key": e2, "col": col, "out": "b"},
            {"op": "COMPARE", "left": "$a", "cmp": cmp, "right": "$b", "out": "c2"},
            {"op": "AND", "args": ["$c1", "$c2"], "out": "result"}]


def _jf_arg_ops(th, tm, fcol, attr, col, mode, claimed):
    return [{"op": "FILTER", "table": th, "col": fcol, "eq": attr, "out": "members"},
            {"op": "VALUES", "table": tm, "keys": "$members", "col": col, "out": "vals"},
            {"op": "ARGEXT_OF", "keys": "$members", "values": "$vals", "mode": mode, "out": "top"},
            {"op": "COMPARE", "left": "$top", "cmp": "eq", "right": claimed, "out": "result"}]


def _jf_sum_ops(th, tm, fcol, attr, col, total):
    return [{"op": "FILTER", "table": th, "col": fcol, "eq": attr, "out": "members"},
            {"op": "VALUES", "table": tm, "keys": "$members", "col": col, "out": "vals"},
            {"op": "SUM", "values": "$vals", "out": "total"},
            {"op": "COMPARE", "left": "$total", "cmp": "eq", "right": total, "out": "result"}]


def _jf_count_ops(th, tm, fcol, attr, col, t, n):
    return [{"op": "FILTER", "table": th, "col": fcol, "eq": attr, "out": "members"},
            {"op": "VALUES", "table": tm, "keys": "$members", "col": col, "out": "vals"},
            {"op": "COUNT_GT", "values": "$vals", "threshold": t, "out": "n"},
            {"op": "COMPARE", "left": "$n", "cmp": "eq", "right": n, "out": "result"}]


def _cp_dir_ops(t1, t2, e, col, direction):
    return [{"op": "LOOKUP", "table": t1, "key": e, "col": col, "out": "before"},
            {"op": "LOOKUP", "table": t2, "key": e, "col": col, "out": "after"},
            {"op": "COMPARE", "left": "$after", "cmp": "gt" if direction == "up" else "lt", "right": "$before", "out": "result"}]


def _cp_diff_ops(t1, t2, e, col, signed):
    return [{"op": "LOOKUP", "table": t1, "key": e, "col": col, "out": "before"},
            {"op": "LOOKUP", "table": t2, "key": e, "col": col, "out": "after"},
            {"op": "SUB", "left": "$after", "right": "$before", "out": "change"},
            {"op": "COMPARE", "left": "$change", "cmp": "eq", "right": signed, "out": "result"}]
