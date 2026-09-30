"""Camera-ready R9 release: the R9 candidate minus records quarantined after human annotation.

Nothing is relabeled or rewritten: the listed records are removed, every other record, table and program is
byte-identical, and quarantine.jsonl keeps the removed records with the reason. Registry instance counts,
stats and validation are recomputed; the manifest records the parent claims sha256 so that predictions made
on the parent release can be scored on this subset after a per-record input-hash check.

    runs/r7_build_env/bin/python -B tools/quarantine_r9.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from repair_review_release import audit, file_hash

OWN_GOAL = ('The player table is the 乌龙球 (own-goal) sub-table of the 射手榜 section of the source page; the '
            'caption kept only its number, so RJ_RANK_TOPVAL computed the top scorer of a team from own goals.')
QUARANTINE = {
    'r9-00242': {'source': 'zh.wikipedia oldid=91334776, table 22 (zw5071906r91334776t21)'},
    'r9-00243': {'source': 'zh.wikipedia oldid=91334776, table 22 (zw5071906r91334776t21)'},
    'r9-00357': {'source': 'zh.wikipedia oldid=93333248, table 40 (zw7268449r93333248t39)'},
    'r9-00358': {'source': 'zh.wikipedia oldid=93333248, table 40 (zw7268449r93333248t39)'},
}
FOUND = ('human annotation 2026-09-29, 300-record sample item 228 (r9-00358) marked UNCERTAIN by annotator 1; '
         'the other three share the defect and were found by checking every use of the own-goal tables')


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def dump(path, rows):
    with Path(path).open('w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--release', type=Path, default=ROOT / 'runs/review_repair_r9_select_v4_nat')
    p.add_argument('--out', type=Path, default=ROOT / 'runs/review_repair_r9_release')
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out} exists and is not empty')
    claims = load(a.release / 'claims.jsonl')
    ids = {c['id'] for c in claims}
    if not set(QUARANTINE) <= ids:
        raise SystemExit('quarantine ids missing from the release: ' + ', '.join(sorted(set(QUARANTINE) - ids)))
    kept = [c for c in claims if c['id'] not in QUARANTINE]
    removed = [c for c in claims if c['id'] in QUARANTINE]
    validation = audit(kept, json.loads((ROOT / 'schemas/claim.schema.json').read_text()))
    if not validation['passed']:
        raise SystemExit('validation failed: ' + json.dumps(validation['issues'][:5], ensure_ascii=False))
    a.out.mkdir(parents=True, exist_ok=True)
    dump(a.out / 'claims.jsonl', kept)
    for name in ('selection.jsonl', 'lineage.jsonl', 'paraphrase_log.jsonl', 'necessity_witnesses.jsonl'):
        dump(a.out / name, [r for r in load(a.release / name) if r['id'] not in QUARANTINE])
    shutil.copy2(a.release / 'table_snapshots.jsonl', a.out / 'table_snapshots.jsonl')
    # registry: instance and label counts of each card over the kept records
    by_card = defaultdict(Counter)
    for c in kept:
        by_card[c['quality_flags']['skeleton_card']['card_id']][c['label']] += 1
    registry = []
    for card in load(a.release / 'skeleton_registry.jsonl'):
        n = by_card.get(card['card_id'], Counter())
        if sum(n.values()):
            registry.append({**card, 'instances': sum(n.values()), 'labels': dict(sorted(n.items()))})
    dump(a.out / 'skeleton_registry.jsonl', registry)
    dump(a.out / 'quarantine.jsonl', [
        {'id': c['id'], 'label': c['label'], 'claim': c['claim'], 'skeleton_id': c['program']['skeleton_id'],
         'reason': OWN_GOAL, 'source': QUARANTINE[c['id']]['source'], 'found_by': FOUND, 'record': c}
        for c in removed])
    from dart_fact.analysis import statistics, surface_baselines
    parent_stats = json.loads((a.release / 'stats.json').read_text())
    (a.out / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    (a.out / 'stats.json').write_text(json.dumps(
        {'statistics': statistics(kept), 'surface_baselines': surface_baselines(kept),
         'naturalize': parent_stats.get('naturalize'),
         'quarantine': {'removed': len(removed), 'labels': dict(Counter(c['label'] for c in removed))}},
        ensure_ascii=False, indent=2) + '\n')
    manifest = json.loads((a.release / 'snapshot_manifest.json').read_text())
    manifest.update(
        release='trace-fact-r9-camera-ready', status='camera-ready release',
        claims_sha256=file_hash(a.out / 'claims.jsonl'),
        human_validation='two graduate-student annotators (native Chinese speakers), 300-record stratified sample; '
                         'independent judgments, see Appendix A.2 of the paper',
        quarantine={'parent_release': manifest['release'], 'parent_claims_sha256': file_hash(a.release / 'claims.jsonl'),
                    'removed_ids': sorted(QUARANTINE), 'reason': OWN_GOAL, 'found_by': FOUND,
                    'file': 'quarantine.jsonl',
                    'scoring': 'predictions on the parent release are scored on the kept records after a '
                               'per-record input_sha256 check; the kept records are byte-identical'})
    manifest['artifacts_sha256'] = {q.name: file_hash(q) for q in sorted(a.out.iterdir())
                                    if q.is_file() and q.name != 'snapshot_manifest.json'}
    (a.out / 'snapshot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'out': str(a.out.relative_to(ROOT)), 'records': len(kept),
                      'labels': dict(Counter(c['label'] for c in kept)), 'removed': sorted(QUARANTINE),
                      'claims_sha256': manifest['claims_sha256'], 'registry_cards': len(registry)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
