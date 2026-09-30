"""Coverage-constrained selection; inputs contain no model evaluation results.

The solver selects already screened records. Constraints cannot manufacture
new evidence, change labels or relax failed quality gates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from prepare_review_r7 import read_jsonl, hydrate
from repair_review_release import audit, digest, file_hash
from dart_fact.analysis import statistics, surface_baselines
from r7_extensions import mean_witnesses, linked_witnesses


def select(pool,out,targets_path,diagnose=False):
    from r7_runtime import enable
    enable()
    import numpy as np
    import scipy
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new/empty')
    spec=json.loads(targets_path.read_text())
    targets=spec['targets']
    release=spec.get('release',{'name':'review-repair-r7-candidate','id_prefix':'r7'})
    tables={t['table_id']:t for t in read_jsonl(pool/'tables.jsonl')}
    records=read_jsonl(pool/'candidates.jsonl')
    # New dependent skeletons are eligible only with an explicit witness per table
    # (r7: opposite-label completions; r8 source-card binders: tools/r8_cards.table_witnesses).
    unwitnessed=Counter()
    def proof_for(full):
        sk=full['program']['skeleton_id']
        q=full['quality_flags']
        binder=q.get('r7_origin')=='r8_source_card_binder' or \
            (q.get('r7_origin')=='r9_hard_variant' and q.get('parent_origin')=='r8_source_card_binder' and full['label']!='NEI')
        if binder:
            from r8_cards import table_witnesses, verify_witnesses
            stored=full['quality_flags'].get('necessity_witnesses')
            # witnesses found at binding time are replayed here, not searched for again
            proof=stored if stored and verify_witnesses(full,stored) else table_witnesses(full)
            return proof,'single_table_completions_label_flip_or_compared_value'
        if sk in {'CAT_MEAN_COMPARE','CAT_LINKED_COMPARE'}:
            return (mean_witnesses(full) if sk=='CAT_MEAN_COMPARE' else linked_witnesses(full)),'single_table_relational_completions'
        return True,None
    def witnessed(row):
        q=row['quality_flags']
        binder=q.get('r7_origin')=='r8_source_card_binder' or \
            (q.get('r7_origin')=='r9_hard_variant' and q.get('parent_origin')=='r8_source_card_binder' and row['label']!='NEI')
        if not binder and row['program']['skeleton_id'] not in {'CAT_MEAN_COMPARE','CAT_LINKED_COMPARE'}:
            return True
        ok=bool(proof_for(hydrate(row,tables))[0])
        if not ok: unwitnessed[row['program']['skeleton_id']]+=1
        return ok
    records=[r for r in records if witnessed(r)]
    n=len(records)
    if not n: raise ValueError('No candidates')
    index=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(records):
        for field,key in [('label',row['label']),('group',row['program']['category']),
                          ('skeleton',row['program']['skeleton_id']),('topic',row['topic']),
                          ('package',row['evidence_package_id']),('domain',row['domain']),
                          ('text',''.join(row['claim'].split()))]:
            index[field][key].append(i)
        raw=set()
        native=set();pages=set()
        for ref in row['table_refs']:
            t=tables[ref['table_id']]
            raw.add(t['source'].get('derived_from',t['table_id']))
            source=t['source'];pages.add(str(source['page_id']))
            positions=source.get('merged_table_indexes') or ([source['table_index']] if 'table_index' in source else [])
            if positions:
                native.update((str(source['page_id']),str(source['revision_id']),str(k)) for k in positions)
            else:
                native.add((str(source['page_id']),str(source['revision_id']),t['source'].get('derived_from',t['table_id'])))
        for key in raw: index['raw'][key].append(i)
        for key in native: index['native_raw'][key].append(i)
        for key in pages: index['source_page'][key].append(i)
        index['raw_family'][(tuple(sorted(raw)),row['program']['category'])].append(i)
        for k in (2,3,4):
            if len(row['table_refs'])>=k: index['table_min'][k].append(i)
    matrix_r,matrix_c,matrix_v,lower,upper,names=[],[],[],[],[],[]
    objective=[(0.0 if r['quality_flags']['r7_origin']=='r6' else 1.0)+int(digest(r['id'])[:6],16)/16**6*1e-4 for r in records]
    def constrain(name,coeff,lo=-np.inf,hi=np.inf):
        rid=len(lower);lower.append(lo);upper.append(hi);names.append(name)
        for j,v in coeff.items():
            matrix_r.append(rid);matrix_c.append(j);matrix_v.append(v)
    def count(name,ids,lo=-np.inf,hi=np.inf):
        constrain(name,{i:1 for i in ids},lo,hi)
    count('claims',range(n),targets['claims'],targets['claims'])
    count('SUPPORTS',index['label']['SUPPORTS'],700,820)
    count('REFUTES',index['label']['REFUTES'],700,820)
    count('NEI',index['label']['NEI'],np.ceil(targets['claims']*targets['nei_share_min']),320)
    count('multitable',index['table_min'][2],np.ceil(targets['claims']*targets['multitable_share_min']))
    count('three_plus',index['table_min'][3],targets['three_plus_table_min'])
    count('four_plus',index['table_min'][4],targets['four_plus_table_min'])
    count('football_combined',index['domain']['football']+index['domain']['football_intl'],hi=int(targets['claims']*targets['football_combined_share_max']))
    group_min=targets.get('group_min',{'Single-table comparison & ranking':500,'Two-entity cross-table comparison':250,
               'Category decomposition':165,'Real-join rank–bridge':140,
               'Hub-profile (multi-table)':50,'Cross-period growth':49,'Join-filter aggregation':32})
    for group,minimum in group_min.items(): count('group:'+group,index['group'][group],minimum)
    for sk,ids in index['skeleton'].items(): count('skeleton:'+sk,ids,1)
    for topic,ids in index['topic'].items(): count('topic_cap:'+topic,ids,hi=targets['largest_topic_max'])
    for pkg,ids in index['package'].items(): count('package_cap:'+pkg,ids,hi=targets.get('package_claims_max',10))
    for key,ids in index['raw_family'].items(): count('family_cap:'+digest(key)[:12],ids,hi=targets['largest_package_family_max'])
    for domain,ids in index['domain'].items(): count('domain_cap:'+domain,ids,hi=int(targets['claims']*0.22))
    for key,ids in index['text'].items():
        if len(ids)>1: count('unique_text:'+digest(key)[:12],ids,hi=1)
    difficulty=spec.get('difficulty')
    hard_kind={}
    if difficulty:
        # R9: structural difficulty (tools/r9_hardness.py) — near-miss numbers, runner-up distractors,
        # missing-table NEI, and a cap on two-cell comparisons / plain superlatives. No model output.
        from r9_hardness import SIMPLE, hardness
        for i,row in enumerate(records):
            if row['label']!='SUPPORTS':
                hard_kind[i]=hardness(hydrate(row,tables))
        simple=[i for i,r in enumerate(records) if r['program']['skeleton_id'] in SIMPLE]
        count('simple_max',simple,hi=int(targets['claims']*difficulty['simple_share_max']))
        hard_refutes=[i for i,k in hard_kind.items() if k and records[i]['label']=='REFUTES']
        count('hard_refutes',hard_refutes,int(np.ceil(targets['claims']*difficulty['hard_refutes_min_share'])))
        missing=[i for i,k in hard_kind.items() if k=='missing_table']
        count('nei_missing_table',missing,int(np.ceil(targets['claims']*difficulty['nei_missing_table_min_share'])))
        for i,k in hard_kind.items():
            if k: objective[i]-=difficulty.get('hard_bonus',0.0)
        # Hard records must not become a label shortcut: balance labels within each skeleton and
        # group, and keep NEI (and its missing-table share) near the paper's proportions.
        nei=index['label']['NEI']
        count('nei_max',nei,hi=int(targets['claims']*difficulty['nei_share_max']))
        share=difficulty['missing_table_share_of_nei_max']
        constrain('missing_table_share_of_nei',{**{i:-share for i in nei},**{i:1-share for i in missing}},hi=0)
        cap=difficulty['group_nei_share_max']
        for group,ids in index['group'].items():
            constrain('group_nei_share:'+group,{i:(1-cap if records[i]['label']=='NEI' else -cap) for i in ids},hi=0)
        ratio=difficulty['skeleton_sr_ratio_max']
        for sk,ids in index['skeleton'].items():
            sup=[i for i in ids if records[i]['label']=='SUPPORTS'];ref=[i for i in ids if records[i]['label']=='REFUTES']
            if sup and ref:
                constrain('sr_balance:R/S:'+sk,{**{i:1 for i in ref},**{i:-ratio for i in sup}},hi=0)
                constrain('sr_balance:S/R:'+sk,{**{i:1 for i in sup},**{i:-ratio for i in ref}},hi=0)
    if len(index['skeleton'])<targets['skeletons_min']: raise ValueError('Too few distinct executable skeletons')
    for field,minimum in [('topic',targets['topics_min']),('source_page',targets['topics_min']),('raw',targets['raw_source_tables_min']),('native_raw',targets['raw_source_tables_min']),('package',targets['evidence_packages_min'])]:
        indicators=[]
        for key,ids in sorted(index[field].items()):
            var=len(objective);objective.append(-0.0001);indicators.append(var)
            constrain('coverage:'+field+':'+str(key),{**{i:-1 for i in ids},var:1},hi=0)
        count('distinct_'+field,indicators,minimum)
    # Report every lower bound the pool cannot reach by itself before asking the solver.
    shortfalls=[]
    by_row=defaultdict(float)
    for r,v in zip(matrix_r,matrix_v):
        if v>0: by_row[r]+=v
    for i,name in enumerate(names):
        if name.startswith('coverage:') or np.isneginf(lower[i]): continue
        if by_row[i]<lower[i]: shortfalls.append({'constraint':name,'required':float(lower[i]),'pool_maximum':by_row[i]})
    if shortfalls:
        raise RuntimeError('Pool cannot meet targets: '+json.dumps(shortfalls,ensure_ascii=False))
    if diagnose:
        return diagnose_conflicts(objective,matrix_r,matrix_c,matrix_v,lower,upper,names)
    size=len(objective)
    mat=coo_matrix((np.array(matrix_v,dtype=float),(matrix_r,matrix_c)),shape=(len(lower),size)).tocsc()
    print(json.dumps({'candidates':n,'variables':size,'constraints':len(lower),'solver':'scipy.optimize.milp','version':scipy.__version__}),flush=True)
    result=milp(np.array(objective),integrality=np.ones(size),bounds=Bounds(np.zeros(size),np.ones(size)),
                constraints=LinearConstraint(mat,np.array(lower),np.array(upper)),
                options={'time_limit':180,'mip_rel_gap':0.005,'disp':True})
    if result.x is None:
        raise RuntimeError('No feasible selection: '+result.message)
    chosen=np.rint(result.x).astype(int)
    activity=mat@chosen
    violations=[names[i] for i,v in enumerate(activity) if v<lower[i]-1e-6 or v>upper[i]+1e-6]
    if violations: raise ValueError('Selection violates constraints: '+str(violations))
    rows=[hydrate(records[i],tables) for i in range(n) if chosen[i]]
    if difficulty:
        kinds=[hard_kind.get(i) for i in range(n) if chosen[i]]
        for row,k in zip(rows,kinds): row['quality_flags']['structural_hardness']=k
    rows.sort(key=lambda r:(r['topic'],r['program']['skeleton_id'],r['claim'],r['id']))
    lineage=[]
    for i,row in enumerate(rows,1):
        previous=row['id'];row['id']=f"{release['id_prefix']}-{i:05d}"
        lineage.append({'id':row['id'],'candidate_id':previous,'r6_id':row['quality_flags'].get('previous_record_id'),
                        'claim_tables_label_unchanged_from_r6':row['quality_flags']['r7_origin']=='r6',
                        'origin':row['quality_flags']['r7_origin']})
    print('Replaying selected programs, schema, evidence and table ablations',flush=True)
    validation=audit(rows,json.loads((ROOT/'schemas/claim.schema.json').read_text()))
    if not validation['passed']:
        print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)
        raise ValueError('Selected records failed structural validation')
    witnesses=[]
    for row in rows:
        proof,method=proof_for(row)
        row['quality_flags'].pop('necessity_witnesses',None)   # published once, in necessity_witnesses.jsonl
        if method is None: continue
        if not proof: raise ValueError('Missing counterfactual witness: '+row['id'])
        witnesses.append({'id':row['id'],'method':method,
                          'scope':'declared relational/numeric model; not independent human validation',
                          'witnesses':proof})
    st=statistics(rows)
    extras={'multitable_records':sum(len(r['tables'])>=2 for r in rows),
            'three_plus_records':sum(len(r['tables'])>=3 for r in rows),
            'four_plus_records':sum(len(r['tables'])>=4 for r in rows),
            'football_combined_records':sum(r['domain'] in {'football','football_intl'} for r in rows),
            'origin_counts':dict(Counter(r['quality_flags']['r7_origin'] for r in rows)),
            'counterfactual_checked_records':len(witnesses)}
    print(json.dumps({'statistics':st,'additional_checks':extras},ensure_ascii=False),flush=True)
    baseline=surface_baselines(rows)
    out.mkdir(parents=True,exist_ok=True)
    def jsonl(name,items):
        with (out/name).open('w') as fh:
            for value in items: fh.write(json.dumps(value,ensure_ascii=False,sort_keys=True)+'\n')
    def write(name,value): (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    jsonl('claims.jsonl',rows);jsonl('lineage.jsonl',lineage);jsonl('necessity_witnesses.jsonl',witnesses)
    used={t['table_id']:t for r in rows for t in r['tables']}
    jsonl('table_snapshots.jsonl',[used[k] for k in sorted(used)])
    plan=[]
    for row in rows:
        item=copy.deepcopy(row)
        item['table_refs']=[{'table_id':t['table_id'],**({'period':t['period']} if 'period' in t else {})} for t in item.pop('tables')]
        item.pop('evidence_cells');item.pop('context_cells');plan.append(item)
    jsonl('selection.jsonl',plan)
    write('stats.json',{'statistics':st,'additional_checks':extras,'surface_baselines':baseline})
    write('validation.json',validation)
    write('selection_report.json',{'solver_status':int(result.status),'message':result.message,'unwitnessed_excluded':dict(unwitnessed),
          'scipy_version':scipy.__version__,'objective':'retain eligible r6 records, then deterministic tie-break; no model scores',
          'constraints':[{ 'name':name,'lower':None if np.isneginf(lower[i]) else float(lower[i]),
                          'upper':None if np.isposinf(upper[i]) else float(upper[i]),'achieved':int(activity[i])}
                         for i,name in enumerate(names) if not name.startswith(('coverage:','unique_text:','topic_cap:','package_cap:','family_cap:'))]})
    write('snapshot_manifest.json',{'release':release['name'],'status':'candidate_pending_semantic_and_human_validation',
          'claims_sha256':file_hash(out/'claims.jsonl'),'source_release_preserved':'review-repair-r6',
          'human_validation':'not_completed','evaluation_status':'not_run_on_'+release['id_prefix'],
          'historical_artifact_recovered':False,'historical_skeleton_derivations_recovered':False,
          'counterfactual_scope':release.get('counterfactual_scope','new CAT_MEAN_COMPARE and CAT_LINKED_COMPARE records; other table ablations are fixed-program checks'),
          'skeleton_registry':release.get('skeleton_registry'),
          'artifacts_sha256':{p.name:file_hash(p) for p in sorted(out.iterdir()) if p.is_file()},
          'input_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in [pool/'candidates.jsonl',pool/'tables.jsonl',targets_path]},
          'code_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in [Path(__file__),ROOT/'tools/r7_extensions.py',ROOT/'tools/r7_quality.py',ROOT/'tools/prepare_review_r7.py']+[ROOT/f for f in release.get('extra_code',[])]}})
    print(json.dumps({'output':str(out),'claims':len(rows),'status':'candidate_built','claims_sha256':file_hash(out/'claims.jsonl')}),flush=True)


def diagnose_conflicts(objective,matrix_r,matrix_c,matrix_v,lower,upper,names):
    """Elastic relaxation: the least total violation that makes the targets jointly satisfiable.

    Coverage-link rows stay hard; every other bound gets a penalised slack. Writes nothing."""
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    rows,cols,vals=list(matrix_r),list(matrix_c),list(matrix_v)
    obj=list(objective);integral=[1]*len(obj);slacks=[]
    for i,name in enumerate(names):
        if name.startswith(('coverage:','unique_text:')): continue
        for sign,finite in [(1,not np.isneginf(lower[i])),(-1,not np.isposinf(upper[i]))]:
            if finite:
                rows.append(i);cols.append(len(obj));vals.append(sign)
                slacks.append((len(obj),name,'below_lower' if sign>0 else 'above_upper'))
                obj.append(1000.0);integral.append(0)
    size=len(obj)
    mat=coo_matrix((np.array(vals,dtype=float),(rows,cols)),shape=(len(lower),size)).tocsc()
    ub=np.ones(size);ub[[s[0] for s in slacks]]=np.inf
    result=milp(np.array(obj),integrality=np.array(integral),bounds=Bounds(np.zeros(size),ub),
                constraints=LinearConstraint(mat,np.array(lower),np.array(upper)),
                options={'time_limit':300,'mip_rel_gap':0.01,'disp':False})
    if result.x is None:
        raise RuntimeError('Elastic model failed: '+result.message)
    report=[{'constraint':name,'direction':d,'violation':round(float(result.x[j]),2)}
            for j,name,d in slacks if result.x[j]>1e-6]
    print(json.dumps({'status':result.message,'violations':report},ensure_ascii=False,indent=1),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--targets',type=Path,default=ROOT/'tools/r7_targets.json')
    parser.add_argument('--diagnose',action='store_true',help='report the least joint violation of the targets; writes nothing')
    args=parser.parse_args();select(args.pool.resolve(),args.out.resolve(),args.targets.resolve(),args.diagnose)
