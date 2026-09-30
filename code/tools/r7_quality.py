"""Conservative, dataset-wide rules motivated by documented semantic issues.

These checks do not use benchmark model predictions and do not confer human
validation. Rejections are retained for author review, not silently relabelled.
"""
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

from dart_fact.tables import entity_key, join_key, parse_number, profile_table
from r7_extensions import category_parts, name_leaks_category

# The merged r7 cache (tools/merge_cache_r7.py) is a superset of runs/cache/pages in which the
# build cache wins every collision; without it, pages from discovery caches would go unchecked.
_ROOT = Path(__file__).resolve().parents[1]
PAGE_CACHE = next(p for p in [_ROOT / 'runs/cache_r7/pages', _ROOT / 'runs/cache/pages'] if p.exists())
# Titles registered in source_adapters.SERIES at import time, i.e. before any runtime
# registration (catalog_review_r7.py --register-unlisted).  Series-specific NEI gates
# (partial top-N lists, sports tables) were written for these pages only.
from dart_fact.acquire import to_simplified
from dart_fact.source_adapters import SERIES as _SERIES
STATIC_TITLES = frozenset(to_simplified(t) for s in _SERIES for t in s['titles'])
SCOPE_RELATIVE_METRIC_RE = re.compile(r'占|比重|份额|百分比')
ORDER_STATISTIC_SKELETONS = {'ST_SUPERLATIVE', 'ST_RANK', 'ST_COUNT_GT', 'CAT_ARGEXT', 'CAT_COUNT', 'CAT_SUM',
                             'JF_ARGEXT', 'JF_COUNT_GT', 'RJ_RANK_TOP', 'RJ_RANK_TOPVAL', 'RJ_MEMBER_RANK',
                             # r8 binders of source-admitted cards (tools/r8_cards.py)
                             'CAT_EXTVAL', 'CAT_AVG', 'CAT_COUNT_GT', 'JF_EXTVAL', 'JF_AVG', 'CAT_ARGEXT_ATTR', 'JF_SUM'}
# Aggregates over a table's member set (mean, argmax, count, sum): if the page has a
# second table of the same kind (another year's list), an unqualified scope is ambiguous.
MEMBER_AGGREGATE_SKELETONS = {'CAT_MEAN_COMPARE', 'CAT_LINKED_COMPARE', 'JF_MEAN_COMPARE',
                              'CAT_EXTVAL', 'CAT_AVG', 'CAT_COUNT_GT', 'JF_EXTVAL', 'JF_AVG', 'CAT_ARGEXT_ATTR', 'JF_SUM'}
SIBLING_RULE_SKELETONS = MEMBER_AGGREGATE_SKELETONS | {'CAT_ARGEXT', 'CAT_COUNT', 'CAT_SUM', 'JF_ARGEXT', 'JF_COUNT_GT'}
_metric_base = lambda h: re.sub(r'[（(].*$|\s|\d{4}年?|_\d+$', '', str(h))  # noqa: E731


@lru_cache(maxsize=1)
def _page_files():
    index = {}
    for path in sorted(PAGE_CACHE.glob('*.json')):
        try:
            page = json.loads(path.read_text())
        except ValueError:
            continue
        if isinstance(page, dict) and isinstance(page.get('pageid'), int):
            index.setdefault((page['pageid'], page.get('revid')), path)
    return index


@lru_cache(maxsize=1024)
def page_table_shapes(page_id, revision_id):
    """(table id, key header, headers, profile) of every table on the pinned cached revision; None if not cached."""
    from dart_fact.acquire import extract_tables
    path = _page_files().get((int(page_id), int(revision_id)))
    if path is None:
        return None
    shapes = []
    for table in extract_tables(json.loads(path.read_text())):
        prof = profile_table(table)
        shapes.append((table['table_id'], prof.key.header if prof.key else None, tuple(table['headers']), prof))
    return tuple(shapes)


def _value(cell):
    number = parse_number(cell)
    return number if number is not None else entity_key(cell)


def _column_map(prof, header):
    col = prof.column(header)
    if col is None or not prof.key:
        return None
    return {join_key(prof.cell(r, prof.key)): _value(prof.cell(r, col)) for r in range(len(prof.rows))}


def _raw_ids(table):
    source = table['source']
    ids = {source.get('derived_from', table['table_id'])}
    ids |= {f"zw{source.get('page_id')}r{source.get('revision_id')}t{int(k):02d}"
            for k in source.get('merged_table_indexes') or ([source['table_index']] if 'table_index' in source else [])}
    return ids


def conflicting_sibling_tables(table, col, role='values'):
    """Other tables of the source page that answer the same member-set question differently.

    A sibling has the same key header and a column of the same kind (years/units ignored).
    role='values': the table only supplies values; a sibling conflicts if it states other
    values for a shared entity (another year's list).  role='universe': the table also
    defines the compared member set (category decomposition of one raw table), so a
    different entity set conflicts too (a 12-row excerpt vs the page's full list).
    role='filter': the table defines membership through a category column; a sibling
    conflicts if shared entities are assigned differently or the per-category member counts
    differ (robust to one club listed under a sponsor name in one table and a common name
    in another).  The record's own raw table is skipped."""
    from collections import Counter
    shapes = page_table_shapes(table['source'].get('page_id'), table['source'].get('revision_id'))
    prof = profile_table(table)
    used = _column_map(prof, col)
    if shapes is None or used is None:
        return 0
    own, base, conflicts = _raw_ids(table), _metric_base(col), 0
    for table_id, key, headers, other in shapes:
        if table_id in own or key != prof.key.header:
            continue
        for header in headers:
            if _metric_base(header) != base:
                continue
            values = _column_map(other, header)
            if values is None:
                continue
            differs = any(values[k] != used[k] for k in used if k in values)
            if role == 'universe':
                differs = differs or set(values) != set(used)
            elif role == 'filter':
                differs = differs or Counter(values.values()) != Counter(used.values())
            if differs:
                conflicts += 1
                break
    return conflicts


ADMIN_SUFFIX = re.compile(r'(壮族自治区|回族自治区|维吾尔自治区|自治区|特别行政区|自治州|地区|省|市|盟)$')


def _core(value):
    return ADMIN_SUFFIX.sub('', entity_key(value))


def grouping_readable_in_metric_table(row):
    """A category value readable in a non-key column of a table the metric is read from.

    Administrative suffixes are ignored on both sides (安徽 vs 安徽省)."""
    slots = row['program']['slots']
    forms = lambda v: {f for f in (entity_key(v), _core(v)) if len(f) >= 2}  # noqa: E731
    cats = {f for k in ('cat', 'cat_a', 'cat_b') if slots.get(k) for f in forms(slots[k])}
    read = {o['table'] for o in row['program']['operators'] if o['op'] in {'VALUES', 'LOOKUP'} and 'table' in o}
    for table in row['tables']:
        if table['table_id'] not in read:
            continue
        prof = profile_table(table)
        key = prof.key.index if prof.key else -1
        for cells in table['rows']:
            if any(i != key and forms(cell) & cats for i, cell in enumerate(cells)):
                return True
    return False


def metric_families(value):
    groups = {
        'sport': r'胜|负|平局|积分|进球|失球|助攻|红牌|黄牌|比赛|场次|得分|篮板|控球|观众',
        'medal': r'金牌|银牌|铜牌|奖牌|夺冠|冠军|亚军|季军',
        'economy': r'GDP|生产总值|收入|销售|营业|资产|出口|进口|贸易|财政|支出|工资|所得|产值',
        'population': r'人口|生育|出生|死亡|寿命|识字|失业',
        'geography': r'面积|海拔|高度|深度|水深|长度|蓄水|库容|流量|流域',
        'transport': r'旅客|客运|货运|吞吐|客流|载客|列车|车站|线路|速度|里程|机场|航班|跑道',
        'building': r'楼层|层数|高度|建筑面积|电梯',
        'energy': r'发电|电量|功率|容量|装机|机组',
        'culture': r'票房|票价|参观|游客|观众|博物馆|文物|藏品|展品',
        'education': r'学生|教师|在校|教授|论文|学科|经费',
        'production': r'产量|生产量|产能|收获|耕地|种植',
    }
    return {name for name, pattern in groups.items() if re.search(pattern, value, re.I)}


# Rules added after the R8 blind model pre-audit (runs/r8_pre_audit): each generalises one kind of defect a
# human reader would reject.  They are opt-in (issues(row, audit_rules=True)) so earlier builds replay unchanged.
METRIC_SYNONYM_CLASSES = [r'深度|最深|水深', r'长度|总长|全长|里程', r'营业收入|营收|销售额', r'海拔|标高|高程', r'楼层|层数']
IDENTIFIER_RE = re.compile(r'名称|编号|代码|代号|序号|号码|ID\b', re.I)
CONTINENT_RE = re.compile(r'^(亚洲|非洲|欧洲|北美洲|南美洲|美洲|大洋洲|南极洲|中东|加勒比)$')
CATEGORY_HEADER_RE = re.compile(r'^(地区|区域|洲|大洲|所属大洲|所属地区|分区|类型|分类|类别)$')
NOUN_CONFLICT_HEADER = {'城市': re.compile(r'^(行政区|属地|州|省|省份|省级行政区|国家|国家/地区|国家或地区|主权国家|地区)$')}
UNIVERSAL_SKELETONS = ORDER_STATISTIC_SKELETONS | {'CAT_AVG', 'CAT_EXTVAL', 'JF_AVG', 'JF_EXTVAL', 'JF_SUM',
                                                   'JF_ARGEXT', 'JF_COUNT_GT', 'CAT_ARGEXT_ATTR'}


def _metric_core(text):
    text = re.sub(r'[（(][^）)]*[）)]', '', str(text))
    return re.sub(r'\s+|（|）|\(|\)', '', text)


def _metric_stem(text):
    return re.sub(r'\d{4}|[\s()（）]|百万|亿|万|人民币|美元|CNY|USD|元|年', '', str(text))


def audit_issues(row):
    out = []
    p, q, s, sk = row['program'], row['quality_flags'], row['program']['slots'], row['program']['skeleton_id']
    ops = p['operators']
    byid = {t['table_id']: t for t in row['tables']}
    if row['label'] == 'NEI' and q.get('perturbation_axis') == 'metric':
        # 最深深度 against a 最大深度（m） column is the same quantity under another name, not missing evidence.
        claimed = _metric_core(s.get('metric_name', ''))
        headers = [_metric_core(h) for t in row['tables'] for h in t['headers']]
        head = claimed[-2:]
        if (len(head) == 2 and any(head in h for h in headers)) or any(
                re.search(c, claimed) and any(re.search(c, h) for h in headers) for c in METRIC_SYNONYM_CLASSES):
            out.append('substituted_metric_synonymous_with_header')
    if sk in UNIVERSAL_SKELETONS and s.get('scope'):
        # 马来亚州属 is one table of the page 马来西亚...: an extreme or count over its rows is not a
        # statement about the page's whole scope unless the claim names the table's own caption.
        for t in row['tables']:
            title, page = t.get('title') or '', t['source']['page_title']
            own_caption = title and not title.startswith(page) and not title.startswith(s['scope'])
            flat = lambda x: re.sub(r'[\s“”"《》]', '', unicodedata.normalize('NFKC', x))  # noqa: E731
            if own_caption and '：' not in title and s['scope'] not in title and flat(title) not in flat(row['claim']):
                out.append('claim_scope_broader_than_table_caption')
                break
    key_headers = []
    for op in ops:
        t = byid.get(op.get('table'))
        if t is None:
            continue
        prof = profile_table(t)
        if op['op'] == 'ATTR' or '类别编号' in t['headers'] and op['op'] != 'JOIN':
            continue                       # the category dictionary of a decomposition, not an entity table
        if op['op'] == 'JOIN' and t['headers']:
            key_headers.append(t['headers'][0])
        elif prof.key is not None:
            key_headers.append(prof.key.header)
            others = [c for c in prof.columns if c.kind == 'key' and c.index != prof.key.index]
            if CATEGORY_HEADER_RE.match(prof.key.header.strip()) and others:
                # a region column that happens to be unique in a small table is not the entity column
                out.append('entity_key_is_category_column')
    if sk in UNIVERSAL_SKELETONS and s.get('noun') == '国家' and '国家' in row['claim'] and key_headers \
            and all(re.search(r'属地|领地|地区', h) for h in key_headers):
        # "17个国家" counted over a 国家和属地 column that also lists British territories and Spanish cities
        out.append('entity_noun_narrower_than_key_header')
    if any(s.get(k) == '分节' for k in ('cat_col', 'attr_col')):
        # 分节 is the page-section column added by table extraction, not an attribute a reader can name
        out.append('section_heading_used_as_category')
    removed = q.get('nei_removed_table')
    if removed and {o['op'] for o in ops if o.get('table') == removed} & {'ATTR', 'JOIN', 'FILTER', 'AT_RANK'}:
        # a missing-table NEI (r9) may only lack the table that supplies the values: a missing ranking
        # can be re-derived from the detail rows, and missing membership is often readable in names
        # (台北捷运 -> 台湾)
        out.append('missing_table_membership_or_rank_derivable')
    if removed:
        rest = [t for t in row['tables'] if t['table_id'] != removed]
        for o in (o for o in ops if o.get('table') == removed):
            if str(o.get('key', '')).startswith('$') and any(o.get('col') in t['headers'] for t in rest):
                # the key is only known at run time (a team, a country), so a same-named column left in
                # the package (a company 排名 next to the country 排名 that was removed) invites misreading
                out.append('missing_table_removed_column_ambiguous')
                break
    if IDENTIFIER_RE.search(str(s.get('metric_col', ''))):
        out.append('identifier_column_as_metric')        # 19号线 compared as a quantity
    if s.get('noun') in {'国家', '国家或地区', '国家和地区'}:
        for t in row['tables']:
            prof = profile_table(t)
            names = prof.key_values() if prof.key else []
            if names and sum(bool(CONTINENT_RE.match(n)) for n in names) * 2 >= len(names):
                out.append('continent_rows_called_countries')    # 亚洲、非洲 … counted as 国家和地区
                break
    pattern = NOUN_CONFLICT_HEADER.get(s.get('noun'))
    entity_values = [str(s[k]) for k in ('claimed', 'e1', 'e2', 'member') if s.get(k)]
    if (pattern and s['noun'] in row['claim'] and key_headers and all(pattern.match(h.strip()) for h in key_headers)
            and not (entity_values and all(v.endswith('市') for v in entity_values))):
        # 加利福尼亚州 called a 城市 because the page-level noun was 城市 while the rows are 行政区
        out.append('entity_noun_not_supported_by_key_header')
    for k in ('metric_name', 'metric_col', 'attr_col', 'cat_col'):
        v = str(s.get(k, ''))
        if re.search(r'[\u4e00-\u9fff]\d+$', _metric_core(v)) and re.sub(r'\s', '', _metric_core(v)) in re.sub(r'\s', '', row['claim']):
            # 人口1: a footnote number fused to a header, copied into the claim
            out.append('footnote_digit_in_claim_header')
            break
    if sk.startswith(('JF_', 'CAT_')) and not re.search(r'\d{4}年', row['claim']):
        values = [o for o in ops if o['op'] in {'VALUES', 'LOOKUP'} and o.get('table') in byid]
        for o in values:
            stem = _metric_stem(o['col'])
            if len(stem) < 2:
                continue
            for t in row['tables']:
                if t['table_id'] != o['table'] and any(_metric_stem(h) == stem for h in t['headers']):
                    # GDP read from the 2024 table while the 2020 member table also has GDP: which year?
                    out.append('metric_period_unspecified_across_tables')
                    break
    return out


def issues(row, audit_rules=False):
    out = audit_issues(row) if audit_rules else []
    p, q = row['program'], row['quality_flags']
    s, sk = p['slots'], p['skeleton_id']
    claim = row['claim']
    names = '|'.join(t['source']['page_title'] for t in row['tables'])
    if re.search(r'航线|航空线',names) and row['package_kind'] == 'period_pair':
        out.append('route_endpoints_and_cross_period_identity_require_review')
    if sk in {'CAT_COUNT','JF_COUNT_GT','ST_COUNT_GT'} and re.search(r'\d+场比赛.*场次',claim):
        out.append('entity_count_unit_mismatch')
    if s.get('noun') == '水电站' and '发电站列表' in names and not re.search(r'水力|水电',s.get('cat','')):
        out.append('mixed_power_plant_types_called_hydropower')
    if sk == 'RJ_RANK_TOPVAL' and '达到' in claim:
        out.append('threshold_versus_equality_ambiguous')
    if sk == 'CAT_PARTNER':
        out.append('explicit_winner_conjunction_requires_semantic_necessity_review')
    if sk.startswith('CAT_') and sk not in {'CAT_MEAN_COMPARE','CAT_LINKED_COMPARE'}:
        # The category dictionary necessarily names the category; it is not an
        # entity-name shortcut. Inspect only the entity/metric tables.
        if s.get('cat') and name_leaks_category(s['cat'],[entity_key(c[0]) for t in row['tables'][1:] for c in t['rows']]):
            out.append('category_visible_in_entity_names')
        if s.get('cat') and any(s['cat'] != entity_key(c[0]) and s['cat'] in entity_key(c[0])
                               for t in row['tables'] if t['headers']==[s.get('cat_col'),'类别编号'] for c in t['rows']):
            out.append('joint_or_composite_category_requires_scope_review')
    if sk in MEMBER_AGGREGATE_SKELETONS:
        # 所在地_2: a colspan duplicate, not a header a reader can resolve.
        if any(re.search(r'_\d+$', str(s.get(k, ''))) for k in ('cat_col', 'attr_col', 'metric_col')):
            out.append('duplicate_colspan_header')
        if grouping_readable_in_metric_table(row):
            out.append('grouping_attribute_readable_in_metric_table')
    if sk in SIBLING_RULE_SKELETONS:
        # Neither the member-defining table nor the metric table may have a conflicting sibling.
        byid = {t['table_id']: t for t in row['tables']}
        membership = set().union(*[_raw_ids(byid[o['table']]) for o in p['operators']
                                   if o['op'] in {'ATTR', 'JOIN'} and o.get('table') in byid] or [set()])
        for op in p['operators']:
            if op['op'] in {'VALUES', 'FILTER'} and op.get('table') in byid:   # a missing-table NEI lacks one
                table = byid[op['table']]
                role = 'filter' if op['op'] == 'FILTER' else 'universe' if _raw_ids(table) & membership else 'values'
                if conflicting_sibling_tables(table, op['col'], role):
                    out.append('aggregate_over_one_of_several_tables')
                    break
    runtime_page = any(t['source']['page_title'] not in STATIC_TITLES for t in row['tables'])
    if runtime_page and row['label'] == 'NEI' and q.get('perturbation_axis') != 'missing_table':
        # Series-specific NEI gates (medal closed world, partial top-N lists, sports) and the
        # domain used by them are unknown for pages classified by title at run time.
        out.append('nei_on_runtime_registered_page')
    if runtime_page and sk in ORDER_STATISTIC_SKELETONS and '所列' not in claim:
        # Whether a table is a top-N excerpt is recorded per series; without it an extreme,
        # rank or count over 'X数据' may silently generalise a partial list.
        out.append('order_statistic_on_runtime_registered_page')
    if re.search(r'注\s*\d|\[\d+\]|\[注', claim):
        # a footnote marker copied from a column header (里程注1) is not part of the metric's name
        out.append('footnote_marker_in_claim')
    if any(re.search(r'^[—\-－–·、]|[—\-－–·、]$', str(s.get(k, ''))) for k in ('cat', 'cat_a', 'cat_b', 'attr')):
        out.append('ill_formed_category_value')
    if row['label'] == 'NEI' and q.get('perturbation_axis') == 'entity':
        # 'Share of the province's GDP' is defined relative to the table's scope: an entity
        # outside that scope has a determinable (zero/undefined) value, not missing evidence.
        if SCOPE_RELATIVE_METRIC_RE.search(s.get('metric_name', '') + '|'.join(
                o.get('col', '') for o in p['operators'])):
            out.append('scope_relative_metric_entity_nei')
    if row['label'] == 'NEI' and q.get('perturbation_axis') == 'metric':
        required = metric_families(s.get('metric_name',''))
        present = metric_families('|'.join(h for t in row['tables'] for h in t['headers']))
        if not required or not (required & present):
            out.append('missing_metric_context_compatibility_unverified')
    if sk in {'ST_DIFF','XT_DIFF','CP_DIFF','CAT_SUM','CAT_EXTVAL','CAT_AVG','JF_EXTVAL','JF_AVG','JF_SUM','CP_AVG'}:
        number = s.get('diff',s.get('total',s.get('value',{})))
        if isinstance(number,dict):
            unit = number.get('unit','')
            if unit and unit not in claim:
                out.append('stated_difference_unit_omitted')
            # A bare real number copied from a scaled header must not lose its scale.
            headers = '|'.join(o.get('col','') for o in p['operators'])
            if any(scale in headers and scale not in claim for scale in ['万','亿']):
                out.append('stated_difference_scale_omitted')
    if sk == 'ST_COMPARE':
        reads = [o for o in p['operators'] if o['op'] == 'LOOKUP']
        if len(reads) == 2 and reads[0]['table'] == reads[1]['table']:
            table = next(t for t in row['tables'] if t['table_id'] == reads[0]['table'])
            metric = s.get('metric_name','')
            # Multiple representations are potentially rounded/defined differently.
            matching = [h for h in table['headers'] if metric and metric in h]
            if len(matching)>1 and reads[0]['col'] not in claim:
                prof=profile_table(table)
                hits=[prof.find_rows(o['key']) for o in reads]
                if all(len(h)==1 for h in hits):
                    signs=[]
                    for header in matching:
                        col=prof.column(header)
                        vals=[parse_number(prof.cell(h[0],col)) for h in hits]
                        if all(v is not None for v in vals):
                            signs.append((vals[0]>vals[1])-(vals[0]<vals[1]))
                    if len(set(signs))>1:
                        out.append('unspecified_unit_representations_disagree')
    if row['label'] == 'REFUTES' and len(row['tables'])>1:
        metric = s.get('metric_name','')
        nonnegative = bool(re.search(r'金牌|银牌|铜牌|奖牌|冠军|亚军|季军|胜场|负场|平局|进球|失球|场次|次数|数量|人数',metric)) and not re.search(r'净|差|增|变化',metric)
        if nonnegative and sk in {'XT_COMPARE','XT_DIFF','CP_DIRECTION','CP_DIFF'}:
            reads=[o for o in p['operators'] if o['op']=='LOOKUP']
            if len(reads)==2:
                values=[]
                for op in reads:
                    table=next(t for t in row['tables'] if t['table_id']==op['table'])
                    prof=profile_table(table); hits=prof.find_rows(op['key'])
                    values.append(parse_number(prof.cell(hits[0],prof.column(op['col']))) if len(hits)==1 else None)
                if all(v is not None for v in values):
                    env={o['out']:v for o,v in zip(reads,values)}
                    last=p['operators'][-1]
                    if sk in {'XT_COMPARE','CP_DIRECTION'}:
                        left=env.get(last['left'].lstrip('$')); right=env.get(last['right'].lstrip('$'))
                        if (last['cmp']=='gt' and left==0) or (last['cmp']=='lt' and right==0):
                            out.append('nonnegative_comparison_single_table_shortcut')
                    else:
                        sub=next(o for o in p['operators'] if o['op']=='SUB')
                        left=env[sub['left'].lstrip('$')]; right=env[sub['right'].lstrip('$')]
                        wanted=last['right']
                        if isinstance(wanted,(int,float)) and (wanted>left or wanted < -right):
                            out.append('nonnegative_difference_single_table_shortcut')
        if sk in {'JF_COUNT_GT','CAT_COUNT_GT'}:
            values_op=next(o for o in p['operators'] if o['op']=='VALUES')
            count_op=next(o for o in p['operators'] if o['op']=='COUNT_GT')
            table=next(t for t in row['tables'] if t['table_id']==values_op['table'])
            col=table['headers'].index(values_op['col'])
            values=[parse_number(c[col]) for c in table['rows']]
            n=p['operators'][-1]['right']
            if all(v is not None for v in values) and n>sum(v>count_op['threshold']+1e-9 for v in values):
                out.append('subset_count_global_upper_bound_shortcut')
    return sorted(set(out))
