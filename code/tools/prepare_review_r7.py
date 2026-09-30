"""Screen old and fresh candidates, add dependent programs, and record lineage.

This writes a candidate pool, not a measured or human-validated release.

With --registry (r8), every candidate must use a skeleton card admitted by the source-induced
registry (tools/induce_skeletons.py) for its table topology; the r7 dependent-comparison
generators are not run because their programs have no admitted card.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from repair_review_release import digest, file_hash
from r7_extensions import group_mean_candidates, linked_candidates, join_mean_candidates, category_parts
from r7_quality import issues


def read_jsonl(path):
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def binding(row):
    return digest({'operators':row['program']['operators'],
                   'tables':[t['table_id'] for t in row['tables']]})


def hydrate(compact,tables):
    row=dict(compact)
    row['tables']=[]
    for ref in row.pop('table_refs'):
        table={k:v for k,v in tables[ref['table_id']].items() if k!='period'}
        if 'period' in ref: table['period']=ref['period']
        row['tables'].append(table)
    return row


def prepare(inventory,out,extra_inventories=(),registry_dir=None,audit_rules=False):
    from r7_runtime import enable
    enable()
    registry=None
    if registry_dir:
        from dart_fact.skeleton_registry import Registry
        registry=Registry.load(registry_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new/empty')
    tables={t['table_id']:t for t in read_jsonl(inventory/'tables.jsonl')}
    fresh=[hydrate(r,tables) for r in read_jsonl(inventory/'candidates.jsonl')]
    all_tables=dict(tables)
    for extra in extra_inventories:
        extra_tables={t['table_id']:t for t in read_jsonl(extra/'tables.jsonl')}
        all_tables.update(extra_tables)
        fresh.extend(hydrate(r,extra_tables) for r in read_jsonl(extra/'candidates.jsonl'))
    old=read_jsonl(ROOT/'data/review_repair_r6/claims.jsonl')
    print(f'Loaded r6={len(old)}, fresh={len(fresh)}',flush=True)
    freeze=json.loads((ROOT/'data/review_repair_r6/freeze_manifest.json').read_text())
    assert file_hash(ROOT/'data/review_repair_r6/claims.jsonl')==freeze['claims_sha256']
    parents={}
    for row in old+fresh:
        if category_parts(row['tables']):
            parents.setdefault(tuple(t['table_id'] for t in row['tables']),row)
    print(f'Category parent packages={len(parents)}',flush=True)
    rows, rejected, counts, seen = [], [], Counter(), set()
    card_counts=Counter()
    def admit(row,origin):
        key=binding(row)
        if key in seen:
            counts['duplicate_program_binding']+=1
            return
        found=issues(row,audit_rules=audit_rules)
        card=None
        if registry is not None and not found:
            removed=row['quality_flags'].get('nei_removed_table')
            # a missing-table NEI (r9) keeps its parent's program: admit it on the parent's full package
            card=registry.admit_program(row['program']['operators'],
                                        row['tables']+([all_tables[removed]] if removed else []))
            card_counts[(row['program']['skeleton_id'],card['admitted'],card['reason'])]+=1
            if not card['admitted']:
                found=['no_source_admitted_skeleton_card']
        if found:
            counts.update(found)
            rejected.append({'candidate_id':row['id'],'binding_sha256':key,'origin':origin,'issues':found})
            return
        seen.add(key)
        row=copy.deepcopy(row)
        row['quality_flags'].update(r7_quality_rules='r7_quality_v1+r8_audit_v5' if audit_rules else 'r7_quality_v1',r7_origin=origin,
                                    previous_record_id=row['id'] if origin=='r6' else None)
        if card is not None:
            support=(row['quality_flags'].get('skeleton_card') or {}).get('nei_evidence_support')
            row['quality_flags']['skeleton_card']={k:card[k] for k in ('card_id','core_card','signature','topology')}
            if support: row['quality_flags']['skeleton_card']['nei_evidence_support']=support
        rows.append(row)
    for row in old:
        admit(row,'r6')
    new_skeletons={'CAT_MEAN_COMPARE','CAT_LINKED_COMPARE','JF_MEAN_COMPARE'}
    for index,row in enumerate(fresh):
        origin=row['quality_flags'].get('candidate_origin','')
        admit(row,'r8_source_card_binder' if origin.startswith('r8_') else 'r9_hard_variant' if origin.startswith('r9_') else
              'r7_dependent_comparison' if row['program']['skeleton_id'] in new_skeletons else 'fresh_cached_generation')
        if index and index%2000==0:
            print(f'Screened fresh={index}, kept={len(rows)}',flush=True)
    mean_parents=set()
    if registry is not None:
        parents={}   # r7 dependent comparisons have no source-admitted card
    for index,parent in enumerate(parents.values()):
        key=tuple(t['table_id'] for t in parent['tables'][:3])
        if key not in mean_parents:
            mean_parents.add(key)
            for row in group_mean_candidates(parent):
                admit(row,'r7_dependent_comparison')
        for row in linked_candidates(parent):
            admit(row,'r7_dependent_comparison')
        if index and index%20==0:
            print(f'Extended category parents={index}/{len(parents)}, kept={len(rows)}',flush=True)
    join_parents={}
    for row in old+fresh:
        if row['package_kind'] in {'join_filter','hub_profile'}:
            key=digest([row['evidence_package_id'],[(o.get('table'),o.get('col')) for o in row['program']['operators'] if o['op'] in {'FILTER','VALUES','ATTR','LOOKUP'}]])
            join_parents.setdefault(key,row)
    for parent in (join_parents.values() if registry is None else []):
        for row in join_mean_candidates(parent): admit(row,'r7_dependent_comparison')
    # One surface must not carry two labels anywhere in the pool (different member sets or
    # table versions behind identical wording): drop every such record, keep the evidence.
    # A missing-table NEI (r9) repeats its parent's wording on purpose — its label differs because a
    # table is absent, not because the wording is ambiguous — so it does not create a conflict, but it
    # is dropped with its wording when the ordinary records of that wording conflict.  At most one
    # record per wording is selected (select_review_r7 unique_text).
    surfaces=defaultdict(set)
    for row in rows:
        if row['quality_flags'].get('perturbation_axis')!='missing_table':
            surfaces[''.join(row['claim'].split())].add(row['label'])
    conflicting={k for k,labels in surfaces.items() if len(labels)>1}
    kept=[]
    for row in rows:
        if ''.join(row['claim'].split()) in conflicting:
            counts['same_surface_conflicting_labels']+=1
            rejected.append({'candidate_id':row['id'],'binding_sha256':binding(row),
                             'origin':row['quality_flags']['r7_origin'],'issues':['same_surface_conflicting_labels']})
        else:
            kept.append(row)
    rows=kept
    # One evidence_package_id must name one table content (the release validator checks this).
    # Where two contents share an id (a table with and without a period tag), r6 keeps its id and
    # every other content gets a content-derived id; package caps then count each content once.
    contents,r6_contents=defaultdict(set),defaultdict(set)
    for row in rows:
        contents[row['evidence_package_id']].add(digest(row['tables']))
        if row['quality_flags']['r7_origin']=='r6': r6_contents[row['evidence_package_id']].add(digest(row['tables']))
    for row in rows:
        pid=row['evidence_package_id']
        if len(contents[pid])>1 and row['quality_flags']['r7_origin']!='r6':
            if digest(row['tables']) not in r6_contents[pid]:
                row['evidence_package_id']='pkgfix-'+digest(row['tables'])[:24]
                counts['package_id_reassigned']+=1
    out.mkdir(parents=True,exist_ok=True)
    used_tables={}
    with (out/'candidates.jsonl').open('w') as fh:
        for row in rows:
            compact=dict(row)
            for t in row['tables']:
                prior=used_tables.get(t['table_id'])
                if prior and (prior['headers'],prior['rows'])!=(t['headers'],t['rows']):
                    raise ValueError('Table identity collision')
                used_tables[t['table_id']]=t
            compact['table_refs']=[{'table_id':t['table_id'],**({'period':t['period']} if 'period' in t else {})} for t in compact.pop('tables')]
            fh.write(json.dumps(compact,ensure_ascii=False)+'\n')
    for name,data in [('tables.jsonl',[used_tables[k] for k in sorted(used_tables)]),('quarantine.jsonl',rejected)]:
        with (out/name).open('w') as fh:
            for row in data: fh.write(json.dumps(row,ensure_ascii=False)+'\n')
    report={'status':'screened_candidate_pool_not_human_validated_release','records':len(rows),
            'by_origin':dict(Counter(r['quality_flags']['r7_origin'] for r in rows)),
            'by_label':dict(Counter(r['label'] for r in rows)),
            'tables_per_record':dict(Counter(len(r['tables']) for r in rows)),
            'skeletons':dict(Counter(r['program']['skeleton_id'] for r in rows)),
            'unique_raw_tables':len({t['source'].get('derived_from',t['table_id']) for t in used_tables.values()}),
            'topics':len({r['topic'] for r in rows}), 'rejections':dict(counts),
            'r6_claims_sha256':freeze['claims_sha256'],'model_api_calls':0,
            'registry':None if registry is None else {'path':str(registry_dir),
                'core_cards_sha256':file_hash(registry_dir/'core_cards.jsonl'),
                'admission':[{'skeleton_id':k[0],'admitted':k[1],'reason':k[2],'candidates':v}
                             for k,v in sorted(card_counts.items(),key=lambda kv:(kv[0][0],-kv[1]))]},
            'input_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in [inventory/'candidates.jsonl',inventory/'tables.jsonl',Path(__file__),ROOT/'tools/r7_quality.py',ROOT/'tools/r7_extensions.py']+[d/name for d in extra_inventories for name in ['candidates.jsonl','tables.jsonl']]}}
    (out/'pool_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--extra-inventory',type=Path,nargs='*',default=[])
    parser.add_argument('--registry',type=Path,help='source-induced skeleton registry; gate every candidate on it')
    parser.add_argument('--audit-rules',action='store_true',help='also apply the rules added after the R8 model pre-audit (r7_quality.audit_issues)')
    args=parser.parse_args()
    prepare(args.inventory.resolve(),args.out.resolve(),[p.resolve() for p in args.extra_inventory],
            args.registry.resolve() if args.registry else None,args.audit_rules)
