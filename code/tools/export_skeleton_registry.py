"""Write the skeleton registry of a selected release: every skeleton the release instantiates, the
source-induced card it binds, the topology, and the verified source programs that support it.

    python tools/export_skeleton_registry.py --release runs/review_repair_r8_select_v1 \
        --registry runs/skeleton_registry_v1

The output (skeleton_registry.jsonl next to claims.jsonl) is derived from claims.jsonl and the
registry; it is not listed in the release's snapshot manifest and changes no record.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dart_fact.skeleton_registry import Registry, card_id


def main(release: Path, registry_dir: Path, pool_tables: Path | None = None) -> None:
    reg = Registry.load(registry_dir)
    # a missing-table NEI (r9) is admitted on its parent's full package; the removed table is in the pool
    extra = {t['table_id']: t for t in map(json.loads, pool_tables.open(encoding='utf-8'))} if pool_tables else {}
    by_card = defaultdict(lambda: {'records': 0, 'labels': Counter(), 'skeleton_ids': Counter(), 'category': set()})
    with (release / 'claims.jsonl').open(encoding='utf-8') as fh:
        for line in fh:
            row = json.loads(line)
            removed = row['quality_flags'].get('nei_removed_table')
            res = reg.admit_program(row['program']['operators'], row['tables'] + ([extra[removed]] if removed else []))
            if not res['admitted']:
                raise ValueError(f"{row['id']} uses a card the registry does not admit: {res['reason']}")
            entry = by_card[(row['program']['skeleton_id'], res['card_id'])]
            entry.update(res={k: res[k] for k in ('signature', 'topology', 'core_card', 'outer_signature',
                                                  'inner_signatures', 'inner_support') if k in res})
            entry['records'] += 1
            entry['labels'][row['label']] += 1
            entry['category'].add(row['program']['category'])
    out = []
    for (skeleton, cid), e in sorted(by_card.items()):
        res = e['res']
        core_sig = res.get('outer_signature') or res['signature']
        core = reg.cores[core_sig]
        item = {'skeleton_id': skeleton, 'card_id': cid, 'category': sorted(e['category']),
                'signature': res['signature'], 'topology': res['topology'],
                'core_card': core['card_id'], 'core_signature': core_sig,
                'core_support': core['support'], 'core_support_total': core['support_total'],
                'source_examples': core['examples'][:10], 'refute_axes': core['refute_axes'],
                'nei_axes': core['nei_axes'], 'instances': e['records'], 'labels': dict(e['labels']),
                'provenance_status': 'induced_from_verified_source_programs_strict_whole_program_match',
                'historical_card_recovered': False}
        if res.get('inner_signatures'):
            item['bridge'] = {'inner_signatures': res['inner_signatures'], 'inner_fragment_support': res['inner_support'],
                              'composition_support': reg.compositions['bridge_verified'],
                              'composition_examples': reg.compositions['examples'].get('bridge', [])[:10]}
        if res['topology'] not in {'single_table', 'category_decomposition', 'bridge'}:
            item['multi_table_composition_support'] = reg.compositions['multi_table_verified']
            item['multi_table_examples'] = reg.compositions['examples'].get('multi_table', [])[:10]
        out.append(item)
    with (release / 'skeleton_registry.jsonl').open('w', encoding='utf-8') as fh:
        for item in out:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    print(json.dumps({'skeleton_ids': len({i['skeleton_id'] for i in out}), 'cards': len({i['card_id'] for i in out}),
                      'core_cards': len({i['core_card'] for i in out}),
                      'records': sum(i['instances'] for i in out)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=ROOT / 'runs/skeleton_registry_v1')
    parser.add_argument('--pool-tables', type=Path, help='tables.jsonl of the pool (needed for missing-table NEI)')
    args = parser.parse_args()
    main(args.release, args.registry, args.pool_tables)
