"""Dependent comparison programs over existing, attributed Chinese tables.

These are new construction schemas, not recovered historical skeleton cards.
No table values are generated. Counterfactual tables are diagnostic witnesses
only and are never substituted for the actual evidence in a released record.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import itertools
import re

from repair_review_release import digest, repair_record
from dart_fact.families import metric_slot, rows_with_values, _metric_norm
from dart_fact.executor import InvalidProgram
from dart_fact.review_repair import execute
from dart_fact.tables import entity_key, join_key, parse_number, profile_table


def replay(row, tables=None):
    return execute(row["program"]["operators"],
                   {t["table_id"]: profile_table(t) for t in (tables or row["tables"])})


def category_parts(tables):
    if len(tables) < 3:
        return None
    tc, te, tm = tables[:3]
    if (len(tc["headers"]) != 2 or len(te["headers"]) != 2
            or tc["headers"][1] != "类别编号" or te["headers"][1] != "类别编号"):
        return None
    cats = {entity_key(r[0]): entity_key(r[1]) for r in tc["rows"]}
    members = defaultdict(list)
    for name, code in te["rows"]:
        members[entity_key(code)].append(entity_key(name))
    return cats, members


def name_leaks_category(cat, names):
    aliases = {"水力": ["水电"], "核能": ["核电"], "火力": ["火电"],
               "风力": ["风电"], "地级市": ["市"], "自治州": ["自治州"],
               "地区": ["地区"], "直辖市": ["直辖市"]}
    terms = [cat] + aliases.get(cat, [])
    return any(any(term in name for term in terms) for name in names)


def scoped_title(row):
    # Explicitly bound to the listed snapshot, not all real-world entities.
    return "《" + row["tables"][0]["source"]["page_title"] + "》所列条目中"


def fresh_record(parent, tables, skeleton, claim, operators, slots, base):
    row = {"id": "r7new-" + digest([skeleton, operators, [t["table_id"] for t in tables]])[:24],
           "claim": claim, "label": "SUPPORTS", "surface": {"kind": "template", "template_claim": claim},
           "evidence_package_id": "r7pkg-" + digest(tables)[:24], "tables": tables,
           "evidence_cells": [], "context_cells": [],
           "program": {"category": "Join-filter aggregation" if skeleton.startswith('JF_') else "Category decomposition", "skeleton_id": skeleton,
                       "operators": operators, "table_ids": [t["table_id"] for t in tables], "slots": slots},
           "table_topology": "two_table" if len(tables)==2 else "three_plus_table",
           "package_kind": "join_filter" if skeleton.startswith('JF_') else "category_partner" if len(tables) == 4 else "category_decomp",
           "quality_flags": {"candidate_origin": "r7_dependent_comparison",
                             "contrast_base": base, "skeleton_provenance": "new_implemented_schema_not_historical_recovery"},
           "domain": parent["domain"], "series": parent["series"], "topic": parent["topic"],
           "license": "CC BY-SA 4.0"}
    row, _ = repair_record(row)
    if row and row['label'] == 'NEI':
        # A profiling/binding failure in a fully observed source is not a new NEI case.
        return None
    if row:
        row["quality_flags"]["perturbation_axis"] = "direction" if row["label"] == "REFUTES" else None
    return row


def group_mean_candidates(parent):
    tabs = parent["tables"][:3]
    parts = category_parts(tabs)
    if parts is None:
        return
    cats, members = parts
    tc, te, tm = tabs
    pm = profile_table(tm)
    if not pm.key:
        return
    eligible = [cat for cat, code in cats.items() if 2 <= len(cat) <= 12
                and re.search(r'[\u4e00-\u9fff]', cat) and not re.search(r'\d|[/、；]', cat)
                and len(members[code]) >= 2 and not name_leaks_category(cat, pm.key_values())]
    for col in pm.metrics():
        ms = metric_slot(pm, col, parent["domain"])
        if not ms:
            continue
        values = {name: value for name, value, _ in rows_with_values(pm, col)}
        for left, right in itertools.combinations(sorted(eligible), 2):
            a, b = members[cats[left]], members[cats[right]]
            if any(name not in values for name in a + b):
                continue
            if abs(sum(values[n] for n in a)/len(a) - sum(values[n] for n in b)/len(b)) < 1e-6:
                continue
            prefix = []
            for suffix, cat in [('a', left), ('b', right)]:
                prefix += [{"op": "ATTR", "table": tc['table_id'], "key": cat, "col": "类别编号", "out": 'cid_'+suffix},
                           {"op": "JOIN", "table": te['table_id'], "fk_col": "类别编号", "eq": '$cid_'+suffix, "out": 'members_'+suffix},
                           {"op": "VALUES", "table": tm['table_id'], "keys": '$members_'+suffix, "col": col.header, "out": 'values_'+suffix},
                           {"op": "AVG", "values": '$values_'+suffix, "out": 'mean_'+suffix}]
            base = digest([tc['table_id'], tm['table_id'], col.header, left, right])
            for cmp, word in [('gt', '高于'), ('lt', '低于')]:
                ops = prefix + [{"op": "COMPARE", "left": "$mean_a", "cmp": cmp, "right": "$mean_b", "out": "result"}]
                claim = f"{scoped_title(parent)}，{tc['headers'][0]}为{left}的条目，其{col.header}平均值{word}{right}类条目。"
                slots = dict(ms, scope=scoped_title(parent), noun='条目', cat_a=left, cat_b=right,
                             cat_col=tc['headers'][0], cmp=cmp)
                row = fresh_record(parent, tabs, 'CAT_MEAN_COMPARE', claim, ops, slots, base)
                if row:
                    yield row


def mean_witnesses(row):
    """Find explicit changed-table completions with an opposite executable label.

    This certifies non-determinacy in the declared relational/numeric model;
    it is not a general natural-language semantic or human validation claim.
    """
    if row['program']['skeleton_id'] != 'CAT_MEAN_COMPARE':
        return None
    tc, te, tm = row['tables']
    cats, members = category_parts(row['tables'])
    slots = row['program']['slots']; ca, cb = cats[slots['cat_a']], cats[slots['cat_b']]
    mutations = []
    a = copy.deepcopy(tc)
    for cells in a['rows']:
        if entity_key(cells[0]) == slots['cat_a']: cells[1] = cb
        elif entity_key(cells[0]) == slots['cat_b']: cells[1] = ca
    mutations.append(a)
    b = copy.deepcopy(te)
    for cells in b['rows']:
        if entity_key(cells[1]) == ca: cells[1] = cb
        elif entity_key(cells[1]) == cb: cells[1] = ca
    mutations.append(b)
    pm = profile_table(tm)
    ci = tm['headers'].index(slots['metric_col'])
    key_col = pm.key.index
    numeric = [(i, float(v)) for i, cells in enumerate(tm['rows'])
               if (v := parse_number(cells[ci])) is not None]
    if not numeric:
        return None
    lo, hi = min(numeric, key=lambda pair: pair[1])[0], max(numeric, key=lambda pair: pair[1])[0]
    # Reverse the actual ordering; copy whole metric vectors to preserve within-row identities.
    original = replay(row)
    left_mean = next(s['value'] for s in original['trace'] if s['out'] == 'mean_a')
    right_mean = next(s['value'] for s in original['trace'] if s['out'] == 'mean_b')
    donors = {ca: lo if left_mean > right_mean else hi, cb: hi if left_mean > right_mean else lo}
    c = copy.deepcopy(tm)
    for i, cells in enumerate(c['rows']):
        code = ca if entity_key(cells[key_col]) in members[ca] else cb if entity_key(cells[key_col]) in members[cb] else None
        if code:
            c['rows'][i] = [cells[j] if j == key_col else tm['rows'][donors[code]][j] for j in range(len(cells))]
    mutations.append(c)
    witnesses = []
    for idx, mutation in enumerate(mutations):
        modified = list(row['tables']); modified[idx] = mutation
        run = replay(row, modified)
        if run['label'] not in {'SUPPORTS', 'REFUTES'} or run['label'] == row['label']:
            return None
        edits = [{'row': r, 'col': mutation['headers'][c], 'before': old, 'after': new}
                 for r, (old_row, new_row) in enumerate(zip(row['tables'][idx]['rows'], mutation['rows']))
                 for c, (old, new) in enumerate(zip(old_row, new_row)) if old != new]
        witnesses.append({'table_id': mutation['table_id'], 'opposite_label': run['label'], 'edits': edits})
    return witnesses


def join_mean_candidates(parent):
    if parent['package_kind'] not in {'join_filter','hub_profile'} or len(parent['tables'])!=2:
        return
    filters=[op for op in parent['program']['operators'] if op['op']=='FILTER']
    reads=[op for op in parent['program']['operators'] if op['op']=='VALUES']
    if parent['package_kind']=='hub_profile':
        filters=[op for op in parent['program']['operators'] if op['op']=='ATTR'][:1]
        reads=[op for op in parent['program']['operators'] if op['op']=='LOOKUP'][:1]
    if len(filters)!=1 or len(reads)!=1: return
    byid={t['table_id']:t for t in parent['tables']};th,tm=byid[filters[0]['table']],byid[reads[0]['table']]
    if th['table_id']==tm['table_id']: return
    fcol,mcol=filters[0]['col'],reads[0]['col']
    if fcol in tm['headers'] or any(_metric_norm(h)==_metric_norm(mcol) for h in th['headers']): return
    ph,pm=profile_table(th),profile_table(tm)
    if not ph.key or not pm.key: return
    if fcol not in th['headers'] or pm.column(mcol) is None: return
    members=defaultdict(list);fc=th['headers'].index(fcol)
    for cells in ph.rows: members[entity_key(cells[fc])].append(entity_key(cells[ph.key.index]))
    values={join_key(n):v for n,v,_ in rows_with_values(pm,pm.column(mcol))}
    eligible=[cat for cat,keys in members.items() if 2<=len(cat)<=12 and len(keys)>=2
              and not name_leaks_category(cat,pm.key_values()) and all(join_key(k) in values for k in keys)]
    for a,b in itertools.combinations(sorted(eligible),2):
        if abs(sum(values[join_key(n)] for n in members[a])/len(members[a])-sum(values[join_key(n)] for n in members[b])/len(members[b]))<1e-6: continue
        prefix=[]
        for suffix,cat in [('a',a),('b',b)]:
            prefix.extend([{'op':'FILTER','table':th['table_id'],'col':fcol,'eq':cat,'out':'members_'+suffix},
                           {'op':'VALUES','table':tm['table_id'],'keys':'$members_'+suffix,'col':mcol,'out':'values_'+suffix},
                           {'op':'AVG','values':'$values_'+suffix,'out':'mean_'+suffix}])
        base=digest([th['table_id'],tm['table_id'],fcol,mcol,a,b])
        for cmp,word in [('gt','高于'),('lt','低于')]:
            ops=prefix+[{'op':'COMPARE','left':'$mean_a','cmp':cmp,'right':'$mean_b','out':'result'}]
            claim=f'{scoped_title(parent)}，{fcol}为{a}的条目，其“{mcol}”平均值{word}{b}类条目。'
            slots={'scope':scoped_title(parent),'noun':'条目','metric_name':mcol,'metric_col':mcol,'attr_col':fcol,'cat_a':a,'cat_b':b,'cmp':cmp}
            row=fresh_record(parent,[th,tm],'JF_MEAN_COMPARE',claim,ops,slots,base)
            if row: yield row


def linked_witnesses(row):
    """Opposite-label completions for every table in a dependent four-table chain."""
    if row['program']['skeleton_id']!='CAT_LINKED_COMPARE':
        return None
    tc,te,tm,tp=row['tables']; slots=row['program']['slots']
    cats,members=category_parts(row['tables']);code=cats[slots['cat']]
    original=replay(row)
    if original['label'] not in {'SUPPORTS','REFUTES'}:
        return None
    winner=next(step['value'] for step in original['trace'] if step['out']=='top')
    def flips(index,table):
        altered=list(row['tables']);altered[index]=table
        try:
            label=replay(row,altered)['label']
        except (ValueError,TypeError,KeyError,InvalidProgram):
            return False
        return label in {'SUPPORTS','REFUTES'} and label!=row['label']
    alternatives=[]
    for other,other_code in cats.items():
        if other==slots['cat']: continue
        modified=copy.deepcopy(tc)
        for cells in modified['rows']:
            if entity_key(cells[0])==slots['cat']: cells[1]=other_code
            elif entity_key(cells[0])==other: cells[1]=code
        if flips(0,modified):
            alternatives.append((modified,other_code));break
    if not alternatives: return None
    cat_change,other_code=alternatives[0]
    membership_change=copy.deepcopy(te)
    for cells in membership_change['rows']:
        if entity_key(cells[1])==code: cells[1]=other_code
        elif entity_key(cells[1])==other_code: cells[1]=code
    if not flips(1,membership_change): return None
    pm=profile_table(tm);pp=profile_table(tp)
    metric_change=None
    hit=pm.find_rows(winner)
    if len(hit)!=1: return None
    winrow=pm.row_map[hit[0]]
    for candidate in members[code]:
        if candidate==winner: continue
        positions=pm.find_rows(candidate)
        if len(positions)!=1: continue
        candrow=pm.row_map[positions[0]]
        modified=copy.deepcopy(tm)
        for col in range(len(tm['headers'])):
            if col!=pm.key.index:
                modified['rows'][winrow][col],modified['rows'][candrow][col]=tm['rows'][candrow][col],tm['rows'][winrow][col]
        if flips(2,modified): metric_change=modified;break
    if metric_change is None: return None
    hits=[pp.find_rows(name) or pp.find_rows(name,loose=True) for name in (winner,slots['e2'])]
    if any(len(hit)!=1 for hit in hits): return None
    i,j=[pp.row_map[hit[0]] for hit in hits]
    partner_change=copy.deepcopy(tp)
    for col in range(len(tp['headers'])):
        if col!=pp.key.index:
            partner_change['rows'][i][col],partner_change['rows'][j][col]=tp['rows'][j][col],tp['rows'][i][col]
    if not flips(3,partner_change): return None
    witnesses=[]
    for index,modified in enumerate([cat_change,membership_change,metric_change,partner_change]):
        edits=[{'row':r,'col':modified['headers'][c],'before':a,'after':b}
               for r,(old,new) in enumerate(zip(row['tables'][index]['rows'],modified['rows']))
               for c,(a,b) in enumerate(zip(old,new)) if a!=b]
        witnesses.append({'table_id':modified['table_id'],
                          'opposite_label':'REFUTES' if row['label']=='SUPPORTS' else 'SUPPORTS','edits':edits})
    return witnesses


def linked_candidates(parent):
    """Four-table reference chain without an independently falsifiable conjunction."""
    if len(parent['tables']) != 4 or category_parts(parent['tables']) is None:
        return
    tc, te, tm, tp = parent['tables']; cats, members = category_parts(parent['tables'])
    pm, pp = profile_table(tm), profile_table(tp)
    if not pm.key or not pp.key:
        return
    # Expose only attributed key/metric columns in this explicit derived view.
    # This avoids repeating the category-name mapping in the partner table.
    for col in pm.metrics()[:2]:
        ms = metric_slot(pm, col, parent['domain'])
        if not ms:
            continue
        vals = {n:v for n,v,_ in rows_with_values(pm,col)}
        for col2 in pp.metrics():
            ms2 = metric_slot(pp,col2,parent['domain'])
            if not ms2 or col2.header in tm['headers']:
                continue
            pvals = {join_key(n):(n,v) for n,v,_ in rows_with_values(pp,col2)}
            projection = {'table_id':'r7view-'+digest([tp['table_id'],pp.key.header,col2.header])[:24],
                          'title':tp['title'], 'headers':[pp.key.header,col2.header],
                          'rows':[[r[pp.key.index],r[col2.index]] for r in tp['rows']],
                          'source':dict(tp['source'],derived_from=tp['source'].get('derived_from',tp['table_id']),
                                        derivation='category_decomposition',view_role='key_metric_projection')}
            for cat, code in sorted(cats.items()):
                mem = members[code]
                if (len(mem)<3 or not 2<=len(cat)<=12 or re.search(r'\d|[/、；]',cat)
                        or name_leaks_category(cat,pm.key_values()) or any(n not in vals for n in mem)):
                    continue
                best=max(vals[n] for n in mem); winners=[n for n in mem if vals[n]==best]
                if len(winners)!=1 or join_key(winners[0]) not in pvals:
                    continue
                w=winners[0]; wv=pvals[join_key(w)][1]
                others=sorted(((n,v) for n,v in pvals.values() if n!=w and v!=wv),key=lambda x:(abs(x[1]-wv),x[0]))
                made=0
                for other, ov in others:
                    # Both sides of the final comparison must be possible among candidate winners.
                    alternate=[pvals[join_key(n)][1] for n in mem if join_key(n) in pvals]
                    if not alternate or not min(alternate)<ov<max(alternate):
                        continue
                    if made>=3: break
                    made+=1
                    base=digest([tc['table_id'],projection['table_id'],col.header,cat,other])
                    for cmp,word in [('gt','高于'),('lt','低于')]:
                        ops=[{'op':'ATTR','table':tc['table_id'],'key':cat,'col':'类别编号','out':'cid'},
                             {'op':'JOIN','table':te['table_id'],'fk_col':'类别编号','eq':'$cid','out':'members'},
                             {'op':'VALUES','table':tm['table_id'],'keys':'$members','col':col.header,'out':'vals'},
                             {'op':'ARGEXT_OF','keys':'$members','values':'$vals','mode':'max','out':'top'},
                             {'op':'LOOKUP','table':projection['table_id'],'key':'$top','col':col2.header,'out':'a'},
                             {'op':'LOOKUP','table':projection['table_id'],'key':other,'col':col2.header,'out':'b'},
                             {'op':'COMPARE','left':'$a','cmp':cmp,'right':'$b','out':'result'}]
                        claim=f"{scoped_title(parent)}，{tc['headers'][0]}为{cat}且{col.header}最高的条目，其{col2.header}{word}{other}。"
                        slots=dict(ms,scope=scoped_title(parent),noun='条目',cat=cat,cat_col=tc['headers'][0],metric2=col2.header,e2=other,cmp=cmp)
                        row=fresh_record(parent,[tc,te,tm,projection],'CAT_LINKED_COMPARE',claim,ops,slots,base)
                        if row and linked_witnesses(row):
                            yield row
