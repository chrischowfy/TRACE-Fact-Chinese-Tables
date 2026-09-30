"""Binders for source-admitted skeleton cards that the family builders do not instantiate.

The registry (tools/induce_skeletons.py) admits core cards induced from TabFact, WTQ, TAT-QA and
MultiModalQA programs.  Some admitted cards have no binder among the r6 family builders; this
module binds them to the package topologies the paper describes (category decomposition,
join-filter pairs, period snapshots).  Every record is checked against the registry when it is
built: the program's AOL signature must be the card named below and must be admitted for the
package topology.  No table value is generated; refuting numbers are real values of the same
column or a documented wrong-base computation, and every label comes from execution.

  CAT_EXTVAL / JF_EXTVAL   CMP_EQ(EXTVAL(FILTER(ROWS,C1,EQ,S1),C2),N1)          TabFact max/min, MMQA
  CAT_AVG / JF_AVG         CMP_EQ(AVG(FILTER(ROWS,C1,EQ,S1),C2),N1)             TabFact/WTQ avg, MMQA
  CAT_COUNT_GT             CMP_EQ(COUNT(FILTERS(COND(C1,EQ,S1),COND(C2,ORD,N1),ROWS)),N2)   TabFact
  CAT_SUM                  CMP_EQ(N1,SUM(FILTER(ROWS,C1,EQ,S1),C2))             TabFact/WTQ sum
  CP_PCT                   CMP_EQ(N1,PCT_CHANGE(HOP(..S1..,C2),HOP(..S1..,C3)))  TAT-QA derivations
"""
from __future__ import annotations

from collections import defaultdict
import copy
import math
import re

from repair_review_release import digest, repair_record
from r7_extensions import category_parts, name_leaks_category, replay, scoped_title
from dart_fact import aol
from dart_fact.families import _decimals, entity_noun, metric_slot, rows_with_values, value_dict
from dart_fact.realize import _cmp, _num, _year
from dart_fact.tables import entity_key, format_number, join_key, parse_number, profile_table

CARDS = {
    'CAT_EXTVAL': 'CMP_EQ(EXTVAL(FILTER(ROWS,C1,EQ,S1),C2),N1)',
    'JF_EXTVAL': 'CMP_EQ(EXTVAL(FILTER(ROWS,C1,EQ,S1),C2),N1)',
    'CAT_AVG': 'CMP_EQ(AVG(FILTER(ROWS,C1,EQ,S1),C2),N1)',
    'JF_AVG': 'CMP_EQ(AVG(FILTER(ROWS,C1,EQ,S1),C2),N1)',
    'CAT_COUNT_GT': 'CMP_EQ(COUNT(FILTERS(COND(C1,EQ,S1),COND(C2,ORD,N1),ROWS)),N2)',
    'CAT_SUM': 'CMP_EQ(N1,SUM(FILTER(ROWS,C1,EQ,S1),C2))',
    'CP_PCT': 'CMP_EQ(N1,PCT_CHANGE(HOP(FILTER(ROWS,C1,EQ,S1),C2),HOP(FILTER(ROWS,C1,EQ,S1),C3)))',
    'JF_COUNT_GT': 'CMP_EQ(COUNT(FILTERS(COND(C1,EQ,S1),COND(C2,ORD,N1),ROWS)),N2)',
    'JF_SUM': 'CMP_EQ(N1,SUM(FILTER(ROWS,C1,EQ,S1),C2))',
    # mean of one entity over 3 / 4 period snapshots (TAT-QA "average ... from 2017 to 2019")
    'CP_AVG3': 'CMP_EQ(AVG_OF(HOP(FILTER(ROWS,C1,EQ,S1),C2),HOP(FILTER(ROWS,C1,EQ,S1),C3),HOP(FILTER(ROWS,C1,EQ,S1),C4)),N1)',
    'CP_AVG4': 'CMP_EQ(AVG_OF(HOP(FILTER(ROWS,C1,EQ,S1),C2),HOP(FILTER(ROWS,C1,EQ,S1),C3),HOP(FILTER(ROWS,C1,EQ,S1),C4),HOP(FILTER(ROWS,C1,EQ,S1),C5)),N1)',
}
GROUP = {'CAT': 'Category decomposition', 'JF': 'Join-filter aggregation', 'CP': 'Cross-period growth'}
KIND = {'CAT': 'category_decomp', 'JF': 'join_filter', 'CP': 'period_pair'}
REGISTRY = None      # set by the driver: dart_fact.skeleton_registry.Registry


# A sum is a stated fact only for extensive quantities (population, area, output ...); positions,
# heights, altitudes, ordinals and intensive measures do not add up.  A mean of a position or an
# ordinal (station chainage, rank, year) is not a property of the group either.
NON_ADDITIVE_RE = re.compile(r'海拔|高度|高程|里程|深度|楼层|层数|温度|气温|速度|时速|排名|名次|年份|年代|人均|密度|率|比|'
                             r'平均|均价|指数|寿命|长度|跨度|坡度|纬度|经度')
POSITION_RE = re.compile(r'里程|排名|名次|年份|年代|编号|序号|纬度|经度')


def metric_named_by_header(ms, header: str) -> bool:
    """The claim's metric name must be read off the column header (metric_slot may add 人均 on
    per-capita pages); lexicon renames such as 容量 -> 球场容量 are not trusted."""
    squash = lambda x: re.sub(r'\s', '', str(x))  # noqa: E731
    return squash(ms['metric_name']).replace('人均', '') in squash(header)


def is_subject_row(name: str, title: str) -> bool:
    """The page's own subject (德国 in 德国各州人口) is a total row, not a listed entity."""
    return len(name) >= 2 and entity_key(name) in title


def _pick(base: str, n: int) -> int:
    return int(digest(base)[:8], 16) % n


def card_record(parent, tables, skeleton, claim, ops, slots, base, axis, intended):
    family = skeleton.split('_')[0]
    tables = [{k: t[k] for k in ('table_id', 'title', 'headers', 'rows', 'source', 'period') if k in t} for t in tables]
    parent = {k: parent[k] for k in ('domain', 'series', 'topic')}
    row = {'id': 'r8new-' + digest([skeleton, ops, [t['table_id'] for t in tables]])[:24],
           'claim': claim, 'label': intended, 'surface': {'kind': 'template', 'template_claim': claim},
           'evidence_package_id': 'r8pkg-' + digest(tables)[:24], 'tables': tables,
           'evidence_cells': [], 'context_cells': [],
           'program': {'category': GROUP[family], 'skeleton_id': skeleton, 'operators': ops,
                       'table_ids': [t['table_id'] for t in tables], 'slots': slots},
           'table_topology': 'two_table' if len(tables) == 2 else 'three_plus_table',
           'package_kind': 'category_partner' if skeleton == 'CAT_ARGEXT_ATTR' else KIND[family],
           'quality_flags': {'candidate_origin': 'r8_source_card_binder', 'contrast_base': base,
                             'skeleton_provenance': 'source_admitted_card_new_binder'},
           'domain': parent['domain'], 'series': parent['series'], 'topic': parent['topic'],
           'license': 'CC BY-SA 4.0'}
    row, _ = repair_record(row)
    if not row or row['label'] != intended:
        return None
    res = REGISTRY.admit_program(row['program']['operators'], row['tables'])
    expected = CARDS[skeleton + str(len(tables))] if skeleton == 'CP_AVG' else CARDS[skeleton]
    if not res['admitted'] or res['signature'] != expected:
        return None
    row['quality_flags']['perturbation_axis'] = axis if intended == 'REFUTES' else None
    row['quality_flags']['skeleton_card'] = {k: res[k] for k in ('card_id', 'core_card', 'signature', 'topology')}
    return row


def _eligible_category(cat: str, key_values: list[str]) -> bool:
    return (2 <= len(cat) <= 12 and bool(re.search(r'[一-鿿]', cat)) and not re.search(r'\d|[/、；]', cat)
            and not name_leaks_category(cat, key_values))


def nice_threshold(low: float, high: float) -> float | None:
    """The roundest number strictly between two adjacent member values."""
    if high <= low:
        return None
    top = int(math.floor(math.log10(max(abs(high), abs(low), 1e-9)))) + 1
    for exp in range(top, -4, -1):
        step = 10.0 ** exp
        t = math.floor(high / step) * step
        if t >= high:
            t -= step
        if low < t < high:
            return round(t, max(0, -exp))
    return None


def _member_aggregates(parent, tabs, prefix_ops, cat, members, pm, col, values, raw_of, scope, cat_col, noun, family,
                       other_means, metric_table=None):
    """EXTVAL / AVG / COUNT_GT / SUM drafts for one category of one metric column."""
    ms = metric_slot(pm, col, parent['domain'])
    if not ms or not ms['numeric_surface_ok'] or not metric_named_by_header(ms, col.header):
        return
    if any(is_subject_row(n, parent['topic']) for n in members):
        return
    m, style, unit = ms['metric_name'], ms['style'], ms['unit']
    # the metric read from a table with its own caption (2024年中国90城市统计) is named with that caption
    mt = f"“{metric_table}”中的{m}" if metric_table else m
    vals = [values[n] for n in members]
    base_slots = dict(ms, scope=scope, noun=noun, cat=cat, cat_col=cat_col)
    if metric_table:
        base_slots['metric_table'] = metric_table
    tail = [{'op': 'VALUES', 'table': tabs[-1]['table_id'], 'keys': '$members', 'col': col.header, 'out': 'vals'}]
    raw0 = raw_of[members[0]]
    # extreme value
    mode = ['max', 'min'][_pick(digest([cat, col.header, 'ext']), 2)]
    ext = (max if mode == 'max' else min)(vals)
    others = sorted({v for v in vals if v != ext}, reverse=(mode == 'max'))
    if others:
        base = digest([family, 'ext', [t['table_id'] for t in tabs], col.header, cat, mode])
        # 最高海拔最高为 -> 最高海拔的最大值为
        w = ('的最大值' if mode == 'max' else '的最小值') if re.match(r'最高|最低|最大|最小|最多|最少', m) else _cmp(style, mode)
        for label, v, axis in (('SUPPORTS', ext, None), ('REFUTES', others[0], 'number')):
            claimed = raw_of[next(n for n in members if values[n] == v)]
            vd = value_dict(v, claimed, unit)
            claim = f"{scope}，{cat_col}为{cat}的{noun}中，{mt}{w}为{_num(vd)}。"
            ops = prefix_ops + tail + [{'op': 'EXTVAL', 'values': '$vals', 'mode': mode, 'out': 'ext'},
                                       {'op': 'COMPARE', 'left': '$ext', 'cmp': 'eq', 'right': v, 'out': 'result'}]
            yield family + '_EXTVAL', claim, ops, dict(base_slots, mode=mode, value=vd), base, axis, label
    # mean over the members, at the precision of the column plus one decimal
    if len(members) >= 2 and len(set(vals)) >= 2 and not POSITION_RE.search(m + col.header):
        digits = min(2, max(_decimals(raw_of[n]) for n in members) + 1)
        mean = round(sum(vals) / len(vals), digits)
        wrong = next((round(x, digits) for x in other_means
                      if abs(round(x, digits) - mean) >= max(10 ** -digits, 0.05 * abs(mean))), None)
        if wrong is not None:
            base = digest([family, 'avg', [t['table_id'] for t in tabs], col.header, cat])
            for label, v, axis in (('SUPPORTS', mean, None), ('REFUTES', wrong, 'number')):
                text = format(v, f'.{digits}f').rstrip('0').rstrip('.') if digits else str(int(v))
                claim = f"{scope}，{cat_col}为{cat}的{noun}，{mt}的平均值为{text}{unit}。"
                ops = prefix_ops + tail + [{'op': 'AVG', 'values': '$vals', 'out': 'mean'},
                                           {'op': 'ROUND', 'value': '$mean', 'digits': digits, 'out': 'mean_r'},
                                           {'op': 'COMPARE', 'left': '$mean_r', 'cmp': 'eq', 'right': v, 'out': 'result'}]
                yield family + '_AVG', claim, ops, dict(base_slots, value={'value': v, 'text': text, 'unit': unit},
                                                        digits=digits), base, axis, label
    # count above a round threshold
    if len(members) >= 3 and style != 'rate':
        s = sorted(vals)
        k = 1 + _pick(digest([cat, col.header, 'cnt']), len(s) - 1)
        thr = nice_threshold(s[k - 1], s[k])
        if thr is not None:
            n = sum(v > thr for v in vals)
            wrong = n + (1 if n < len(vals) - 1 else -1)
            base = digest([family, 'cnt', [t['table_id'] for t in tabs], col.header, cat, thr])
            thr_text = format(thr, 'f').rstrip('0').rstrip('.')
            for label, nn, axis in (('SUPPORTS', n, None), ('REFUTES', wrong, 'number')):
                claim = f"{scope}，{cat_col}为{cat}的{noun}中，{mt}超过{thr_text}{unit}的有{nn}个。"
                ops = prefix_ops + tail + [{'op': 'COUNT_GT', 'values': '$vals', 'threshold': thr, 'out': 'n'},
                                           {'op': 'COMPARE', 'left': '$n', 'cmp': 'eq', 'right': nn, 'out': 'result'}]
                yield family + '_COUNT_GT', claim, ops, dict(base_slots, threshold=thr, n=nn), base, axis, label
    # total of a small category of an additive metric (the families.py rule)
    if style != 'rate' and 2 <= len(members) <= 6 and not NON_ADDITIVE_RE.search(m + col.header):
        total = round(sum(vals), 6)
        if total > 0:
            off = max(abs(min(vals)), 1.0)
            base = digest([family, 'sum', [t['table_id'] for t in tabs], col.header, cat])
            for label, tv, axis in (('SUPPORTS', total, None), ('REFUTES', round(total + off, 6), 'number')):
                vd = value_dict(tv, raw0, unit)
                claim = f"{scope}，{cat_col}为{cat}的{noun}的{mt}合计为{_num(vd)}。"
                ops = prefix_ops + tail + [{'op': 'SUM', 'values': '$vals', 'out': 'total'},
                                           {'op': 'COMPARE', 'left': '$total', 'cmp': 'eq', 'right': tv, 'out': 'result'}]
                yield family + '_SUM', claim, ops, dict(base_slots, total=vd), base, axis, label


def category_card_candidates(parent):
    tabs = parent['tables'][:3]
    parts = category_parts(tabs)
    if parts is None:
        return
    cats, members = parts
    tc, te, tm = tabs
    pm = profile_table(tm)
    if not pm.key:
        return
    noun = entity_noun(pm.key.header, parent['domain'], pm)
    scope = scoped_title(parent)
    cat_col = tc['headers'][0]
    eligible = [c for c, code in cats.items() if len(members[code]) >= 2 and _eligible_category(c, pm.key_values())]
    for col in pm.metrics():
        rows = rows_with_values(pm, col)
        values = {n: v for n, v, _ in rows}
        raw_of = {n: raw for n, _, raw in rows}
        usable = [c for c in eligible if all(n in values for n in members[cats[c]])]
        means = {c: sum(values[n] for n in members[cats[c]]) / len(members[cats[c]]) for c in usable}
        for cat in usable:
            prefix = [{'op': 'ATTR', 'table': tc['table_id'], 'key': cat, 'col': '类别编号', 'out': 'cid'},
                      {'op': 'JOIN', 'table': te['table_id'], 'fk_col': '类别编号', 'eq': '$cid', 'out': 'members'}]
            other = [means[c] for c in sorted(usable) if c != cat]
            for sk, claim, ops, slots, base, axis, label in _member_aggregates(
                    parent, tabs, prefix, cat, members[cats[cat]], pm, col, values, raw_of, scope, cat_col, noun,
                    'CAT', other):
                row = card_record(parent, tabs, sk, claim, ops, slots, base, axis, label)
                if row:
                    yield row


def join_filter_card_candidates(package):
    th, tm = [{k: t[k] for k in ('table_id', 'title', 'headers', 'rows', 'source') if k in t} for t in package['tables']]
    ph, pm = profile_table(th), profile_table(tm)
    if not ph.key or not pm.key:
        return
    parent = {'domain': package['domain'], 'series': package['series'], 'topic': package['page_title'],
              'tables': [th, tm]}
    noun = entity_noun(pm.key.header, package['domain'], pm)
    scope = scoped_title(parent)
    # Tables with their own captions (a 2020 list and a 2024 list of one page) are named: the member set is the
    # first table's rows and the metric is the second table's column, so the page title alone is ambiguous.
    own = lambda t: t['title'] if t.get('title') and not t['title'].startswith(package['page_title']) else None  # noqa: E731
    member_table, metric_table = own(th), own(tm)
    if member_table:
        scope = f"《{package['page_title']}》“{member_table}”所列条目中"
    for fcol in package['filter_cols']:
        if fcol in tm['headers'] or fcol not in th['headers']:
            continue
        fc = th['headers'].index(fcol)
        members = defaultdict(list)
        for cells in ph.rows:
            members[entity_key(cells[fc])].append(entity_key(cells[ph.key.index]))
        for col in pm.metrics():
            if any(h == col.header for h in th['headers']):
                continue
            rows = rows_with_values(pm, col)
            values = {join_key(n): v for n, v, _ in rows}
            raw_of = {join_key(n): raw for n, _, raw in rows}
            usable = [c for c, keys in members.items() if len(keys) >= 2 and _eligible_category(c, pm.key_values())
                      and all(join_key(k) in values for k in keys)]
            means = {c: sum(values[join_key(k)] for k in members[c]) / len(members[c]) for c in usable}
            for cat in usable:
                prefix = [{'op': 'FILTER', 'table': th['table_id'], 'col': fcol, 'eq': cat, 'out': 'members'}]
                keys = [join_key(k) for k in members[cat]]
                other = [means[c] for c in sorted(usable) if c != cat]
                for sk, claim, ops, slots, base, axis, label in _member_aggregates(
                        parent, [th, tm], prefix, cat, keys, pm, col, values, raw_of, scope, fcol, noun, 'JF', other,
                        metric_table):
                    if member_table:
                        slots = dict(slots, member_table=member_table)
                    row = card_record(parent, [th, tm], sk, claim, ops, slots, base, axis, label)
                    if row:
                        yield row


def period_card_candidates(package):
    t1, t2 = [{k: t[k] for k in ('table_id', 'title', 'headers', 'rows', 'source', 'period') if k in t}
              for t in package['tables']]
    p1, p2 = profile_table(t1), profile_table(t2)
    if not (p1.key and p2.key):
        return
    y1, y2 = package['periods']
    parent = {'domain': package['domain'], 'series': package['series'], 'topic': package['page_title']}
    sc = package['scope']
    for col in p2.metrics():
        if not p1.column(col.header):
            continue
        ms = metric_slot(p2, col, package['domain'])
        if not ms or ms['style'] == 'rate' or re.search(r'率|比|指数|排名|名次', ms['metric_name']) \
                or not metric_named_by_header(ms, col.header):
            continue
        v1 = {join_key(n): (n, v) for n, v, _ in rows_with_values(p1, p1.column(col.header))}
        v2 = {join_key(n): (n, v) for n, v, _ in rows_with_values(p2, col)}
        keys = sorted((k for k in v2 if k in v1 and v1[k][1] > 0 and v1[k][1] != v2[k][1]),
                      key=lambda k: digest([package['package_id'], col.header, k]))
        keys = [k for k in keys if not is_subject_row(v2[k][0], package['page_title'] + package['scope'])]
        for k in keys[:3]:
            name, before, after = v2[k][0], v1[k][1], v2[k][1]
            pct = round((after - before) / before * 100, 1)
            if abs(pct) < 1 or abs(pct) > 400:
                continue
            wrong_base = round((after - before) / after * 100, 1) if after > 0 else None
            wrong = wrong_base if wrong_base is not None and abs(wrong_base - pct) >= 1.0 and wrong_base * pct > 0 \
                else round(pct * 1.5, 1)
            base = digest(['CP', 'pct', package['package_id'], col.header, k])
            for label, v, axis in (('SUPPORTS', pct, None), ('REFUTES', wrong, 'number')):
                verb = '增长' if v > 0 else '下降'
                text = format(abs(v), '.1f').rstrip('0').rstrip('.')
                claim = f"在{sc}中，{name}的{ms['metric_name']}从{_year(y1)}到{_year(y2)}{verb}了{text}%。"
                ops = [{'op': 'LOOKUP', 'table': t1['table_id'], 'key': name, 'col': col.header, 'out': 'before'},
                       {'op': 'LOOKUP', 'table': t2['table_id'], 'key': name, 'col': col.header, 'out': 'after'},
                       {'op': 'PCT_CHANGE', 'left': '$after', 'right': '$before', 'digits': 1, 'out': 'pct'},
                       {'op': 'COMPARE', 'left': '$pct', 'cmp': 'eq', 'right': v, 'out': 'result'}]
                slots = dict(ms, scope=sc, e=name, y1=y1, y2=y2, pct={'value': v, 'text': text, 'unit': '%'},
                             error='wrong_base' if axis and v == wrong_base else ('scaled' if axis else None))
                row = card_record(parent, [t1, t2], 'CP_PCT', claim, ops, slots, base, axis, label)
                if row:
                    yield row


def _numeric_targets(row, run, old: float) -> list[float]:
    """Cell values that would make the stated number hold (or fail) for this program."""
    env = {t['out']: t['value'] for t in run['trace'] if t.get('out')}
    last = row['program']['operators'][-1]
    right = last.get('right')
    if not isinstance(right, (int, float)) or isinstance(right, bool):
        return []
    cur = env.get(str(last['left']).lstrip('$'))
    if not isinstance(cur, (int, float)):
        return []
    n = len(env.get('vals') or []) or 1
    k = next((len(o['args']) for o in row['program']['operators'] if o['op'] == 'AVG_OF'), 1)
    out = [right, old + (right - cur), old + (right - cur) * n, old + (right - cur) * k]
    if 'before' in env and 'after' in env:
        out += [env['before'] * (1 + right / 100), env['after'] / (1 + right / 100) if right != -100 else None]
    return [x for x in out if x is not None]


def table_witnesses(row):
    """For every table, a completion of that table alone that flips SUPPORTS/REFUTES.

    Edits are one cell (a solved or extreme number, another value of the column) or a swap of two
    values throughout one column (category codes).  They certify dependence on each table under
    the declared relational/numeric model; they are not a semantic or human validation."""
    run = replay(row)
    base = run['label']
    if base not in {'SUPPORTS', 'REFUTES'}:
        return None
    compared = str(row['program']['operators'][-1].get('left', '')).lstrip('$')
    value_of = lambda r: next((t['value'] for t in r['trace'] if t.get('out') == compared), None)  # noqa: E731
    base_value = value_of(run)
    out = []
    for idx, table in enumerate(row['tables']):
        prof = profile_table(table)
        key = prof.key.index if prof.key else -1
        cells = [c for c in row['evidence_cells'] if c['table_id'] == table['table_id']]
        found = fallback = None
        for cell in cells:
            ci = table['headers'].index(cell['col'])
            if ci == key:
                continue
            column = [r[ci] for r in table['rows'] if ci < len(r)]
            numbers = [v for v in (parse_number(x) for x in column) if v is not None]
            old = table['rows'][cell['row']][ci]
            edits = []
            old_num = parse_number(old)
            if old_num is not None and numbers:
                hi, lo = max(numbers), min(numbers)
                for x in [hi + abs(hi) + 1, lo - abs(lo) - 1] + _numeric_targets(row, run, old_num):
                    edits.append([(cell['row'], ci, format(round(x, 6), 'f').rstrip('0').rstrip('.'))])
            else:
                for other in sorted({x for x in column if x != old}):
                    edits.append([(cell['row'], ci, other)])
                    edits.append([(r, ci, other if v == old else old) for r, v in enumerate(column)
                                  if v in (old, other)])
            for edit in edits:
                mutated = copy.deepcopy(table)
                for r, c, v in edit:
                    mutated['rows'][r][c] = v
                tables = list(row['tables'])
                tables[idx] = mutated
                try:
                    changed = replay(row, tables)
                except Exception:        # noqa: BLE001 - an edit that breaks the program is not a witness
                    continue
                label = changed['label']
                if label not in {'SUPPORTS', 'REFUTES'}:
                    continue
                record = {'table_id': table['table_id'],
                          'edits': [{'row': r, 'col': table['headers'][c], 'before': table['rows'][r][c], 'after': v}
                                    for r, c, v in edit]}
                if label != base:
                    found = dict(record, kind='label_flip', opposite_label=label)
                    break
                if fallback is None and value_of(changed) != base_value:
                    # e.g. a refuting number no single-table edit can make true: the table still
                    # determines the compared value
                    fallback = dict(record, kind='compared_value_change', label=label,
                                    value_before=base_value, value_after=value_of(changed))
            if found:
                break
        if not (found or fallback):
            return None
        out.append(found or fallback)
    return out


NOTE_HEADER_RE = re.compile(r'备注|注释|附注|参考|来源|资料|图片|图像|照片|网站|链接|说明|简介|编号|序号|代码|坐标')
CARDS['CAT_ARGEXT_ATTR'] = 'CMP_EQ(HOP(ARGEXT(FILTER(ROWS,C1,EQ,S1),C2),C3),S2)'


def _text_attributes(table, prof, own):
    """Short categorical text columns of a partner table that the category source does not carry."""
    out = []
    metric_headers = {c.header for c in prof.metrics()}
    for ci, header in enumerate(table['headers']):
        if prof.key and ci == prof.key.index or header in metric_headers or NOTE_HEADER_RE.search(header):
            continue
        if re.sub(r'\s', '', header) in own or re.search(r'_\d+$', header):
            continue
        values = [entity_key(r[ci]) for r in prof.rows if ci < len(r)]
        good = [v for v in values if 2 <= len(v) <= 12 and re.search(r'[一-鿿]', v)
                and not re.search(r'\d', v) and parse_number(v) is None]
        if len(good) >= 0.8 * len(values) and 2 <= len(set(good)) < len(good):
            out.append(header)
    return out


def partner_attribute_candidates(parent, partners):
    """Four tables: category dictionary, membership, metric (the extreme member) and a second real
    table of the page that holds a text attribute of that member."""
    tabs = parent['tables'][:3]
    parts = category_parts(tabs)
    if parts is None:
        return
    cats, members = parts
    tc, te, tm = tabs
    pm = profile_table(tm)
    if not pm.key:
        return
    noun = entity_noun(pm.key.header, parent['domain'], pm)
    scope = scoped_title(parent)
    cat_col = tc['headers'][0]
    own = {re.sub(r'\s', '', h) for h in parent['source_headers']}
    mine = {join_key(k) for k in pm.key_values()}
    eligible = [c for c, code in cats.items() if len(members[code]) >= 2 and _eligible_category(c, pm.key_values())]
    for t2, p2 in partners:
        if not p2.key or t2['table_id'] == parent['source_table_id']:
            continue
        theirs = {join_key(p2.cell(r, p2.key)): r for r in range(len(p2.rows))}
        if len(mine & set(theirs)) < 0.6 * min(len(mine), len(theirs)):
            continue
        partner = {k: t2[k] for k in ('table_id', 'title', 'headers', 'rows', 'source') if k in t2}
        for attr in _text_attributes(t2, p2, own):
            ai = t2['headers'].index(attr)
            attr_of = {k: entity_key(p2.rows[r][ai]) for k, r in theirs.items() if ai < len(p2.rows[r])}
            for col in [c for c in pm.metrics() if metric_slot(pm, c, parent['domain'])][:2]:
                ms = metric_slot(pm, col, parent['domain'])
                if not metric_named_by_header(ms, col.header):
                    continue
                values = {n: v for n, v, _ in rows_with_values(pm, col)}
                for cat in eligible:
                    mem = members[cats[cat]]
                    if any(is_subject_row(n, parent['topic']) for n in mem):
                        continue
                    # the extreme is taken in the metric table, so every member needs a value there;
                    # the partner is only read for the winner (and supplies the refuting attribute)
                    if any(n not in values for n in mem):
                        continue
                    mode = ['max', 'min'][_pick(digest([cat, col.header, attr, 'attr']), 2)]
                    ext = (max if mode == 'max' else min)(values[n] for n in mem)
                    winners = [n for n in mem if values[n] == ext]
                    if len(winners) != 1 or join_key(winners[0]) not in attr_of:
                        continue
                    truth = attr_of[join_key(winners[0])]
                    wrong = sorted({attr_of[join_key(n)] for n in mem if join_key(n) in attr_of} - {truth, ''})
                    if not truth or not wrong:
                        continue
                    wrong = wrong[_pick(digest([cat, attr, 'wrong']), len(wrong))]
                    w = _cmp(ms['style'], mode)
                    base = digest(['CAT', 'attr', [t['table_id'] for t in tabs], partner['table_id'], col.header, attr, cat])
                    for label, claimed, axis in (('SUPPORTS', truth, None), ('REFUTES', wrong, 'entity_or_value')):
                        claim = f"{scope}，{cat_col}为{cat}的{noun}中，{ms['metric_name']}{w}者的{attr}是{claimed}。"
                        ops = [{'op': 'ATTR', 'table': tc['table_id'], 'key': cat, 'col': '类别编号', 'out': 'cid'},
                               {'op': 'JOIN', 'table': te['table_id'], 'fk_col': '类别编号', 'eq': '$cid', 'out': 'members'},
                               {'op': 'VALUES', 'table': tm['table_id'], 'keys': '$members', 'col': col.header, 'out': 'vals'},
                               {'op': 'ARGEXT_OF', 'keys': '$members', 'values': '$vals', 'mode': mode, 'out': 'top'},
                               {'op': 'ATTR', 'table': partner['table_id'], 'key': '$top', 'col': attr, 'out': 'attr'},
                               {'op': 'COMPARE', 'left': '$attr', 'cmp': 'eq', 'right': claimed, 'out': 'result'}]
                        slots = dict(ms, scope=scope, noun=noun, cat=cat, cat_col=cat_col, mode=mode,
                                     attr_col=attr, claimed=claimed)
                        row = card_record(parent, tabs + [partner], 'CAT_ARGEXT_ATTR', claim, ops, slots, base, axis, label)
                        if row:
                            yield row


def period_window_candidates(packages):
    """Means of one entity over 3 or 4 consecutive period snapshots of the same source table.

    The snapshots are the pipeline's own period-split tables (packages._split_periods); pairs that
    share a source table and metric are chained into windows, so every table is one real period."""
    chains = defaultdict(dict)
    meta = {}
    for pkg in packages:
        if pkg['kind'] != 'period_pair':
            continue
        t1, t2 = pkg['tables']
        key = (t1['source'].get('derived_from'), t1['headers'][1])
        for t in (t1, t2):
            chains[key][t['period']] = {k: t[k] for k in ('table_id', 'title', 'headers', 'rows', 'source', 'period') if k in t}
        meta[key] = pkg
    for key, by_period in chains.items():
        pkg = meta[key]
        periods = sorted(by_period)
        for size in (3, 4):
            for i in range(len(periods) - size + 1):
                window = periods[i:i + size]
                if any(b - a > 5 for a, b in zip(window, window[1:])):
                    continue
                tables = [by_period[p] for p in window]
                yield from _period_mean(pkg, tables, window)


def _period_mean(pkg, tables, window):
    profs = [profile_table(t) for t in tables]
    if not all(p.key for p in profs):
        return
    col = profs[-1].column(tables[-1]['headers'][1])
    ms = metric_slot(profs[-1], col, pkg['domain']) if col else None
    if not ms or ms['style'] == 'rate' or not ms['numeric_surface_ok'] or POSITION_RE.search(ms['metric_name']) \
            or re.search(r'率|比|指数|排名|名次', ms['metric_name']):
        return
    series = [{join_key(n): (n, v, raw) for n, v, raw in rows_with_values(p, p.column(t['headers'][1]))}
              for p, t in zip(profs, tables)]
    common = sorted(set.intersection(*[set(s) for s in series]), key=lambda k: digest([window, k]))
    means = {k: sum(s[k][1] for s in series) / len(series) for k in common}
    parent = {'domain': pkg['domain'], 'series': pkg['series'], 'topic': pkg['page_title']}
    sc, m, unit = pkg['scope'], ms['metric_name'], ms['unit']
    for k in common[:3]:
        name = series[-1][k][0]
        digits = min(2, max(_decimals(s[k][2]) for s in series) + 1)
        mean = round(means[k], digits)
        wrong = next((round(means[o], digits) for o in common
                      if o != k and abs(round(means[o], digits) - mean) >= max(10 ** -digits, 0.05 * abs(mean))), None)
        if wrong is None or len(set(round(s[k][1], 6) for s in series)) < 2:
            continue
        base = digest(['CP', 'avg', [t['table_id'] for t in tables], k])
        for label, v, axis in (('SUPPORTS', mean, None), ('REFUTES', wrong, 'number')):
            text = format(v, f'.{digits}f').rstrip('0').rstrip('.') if digits else str(int(v))
            claim = f"在{sc}中，{name}{_year(window[0])}至{_year(window[-1])}的{m}平均为{text}{unit}。"
            ops = [{'op': 'LOOKUP', 'table': t['table_id'], 'key': name, 'col': t['headers'][1], 'out': f'v{i}'}
                   for i, t in enumerate(tables)]
            ops += [{'op': 'AVG_OF', 'args': [f'$v{i}' for i in range(len(tables))], 'out': 'mean'},
                    {'op': 'ROUND', 'value': '$mean', 'digits': digits, 'out': 'mean_r'},
                    {'op': 'COMPARE', 'left': '$mean_r', 'cmp': 'eq', 'right': v, 'out': 'result'}]
            slots = dict(ms, scope=sc, e=name, periods=window, value={'value': v, 'text': text, 'unit': unit},
                         digits=digits)
            row = card_record(parent, tables, 'CP_AVG', claim, ops, slots, base, axis, label)
            if row:
                yield row


def verify_witnesses(row, witnesses) -> bool:
    """Replay stored witnesses: one completion per table, each still flipping the label or
    changing the compared value against the record's current tables."""
    if not witnesses or {w['table_id'] for w in witnesses} != {t['table_id'] for t in row['tables']}:
        return False
    run = replay(row)
    compared = str(row['program']['operators'][-1].get('left', '')).lstrip('$')
    value_of = lambda r: next((t['value'] for t in r['trace'] if t.get('out') == compared), None)  # noqa: E731
    for w in witnesses:
        idx = next(i for i, t in enumerate(row['tables']) if t['table_id'] == w['table_id'])
        mutated = copy.deepcopy(row['tables'][idx])
        for e in w['edits']:
            ci = mutated['headers'].index(e['col'])
            if mutated['rows'][e['row']][ci] != e['before']:
                return False
            mutated['rows'][e['row']][ci] = e['after']
        tables = list(row['tables'])
        tables[idx] = mutated
        try:
            changed = replay(row, tables)
        except Exception:        # noqa: BLE001
            return False
        if changed['label'] not in {'SUPPORTS', 'REFUTES'}:
            return False
        if w['kind'] == 'label_flip' and changed['label'] == run['label']:
            return False
        if w['kind'] == 'compared_value_change' and value_of(changed) == value_of(run):
            return False
    return True
