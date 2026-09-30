"""Induce the skeleton registry from the reasoning sources and check Chinese programs against it.

    python tools/induce_skeletons.py --sources runs/sources --out runs/skeleton_registry_v1 \
        --check data/review_repair_r6/claims.jsonl runs/review_repair_r7_select_v3/claims.jsonl

Writes source_records.jsonl (every parsed source instance with its provenance result), the
registry (core_cards.jsonl, fragments.json, compositions.json), and, for each checked Chinese
file, the admission result of every skeleton/signature/topology it uses.  Offline; the sources
must first be fetched with tools/sources_fetch.py.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dart_fact import aol
from dart_fact.skeleton_registry import MIN_SUPPORT, Registry, to_json
from dart_fact.source_programs import PARSERS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_programs(path: Path):
    """Claims files carry tables inline; candidate pools reference tables.jsonl next to them."""
    tables = {}
    if path.is_dir():
        with (path / 'tables.jsonl').open(encoding='utf-8') as fh:
            for line in fh:
                t = json.loads(line)
                tables[t['table_id']] = t
        path = path / 'candidates.jsonl'
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            tabs = r.get('tables') or [tables[x['table_id']] for x in r['table_refs']]
            yield r, tabs


def main(sources: Path, out: Path, check: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    records, per_source = [], {}
    with (out / 'source_records.jsonl').open('w', encoding='utf-8') as fh:
        for name, parser in PARSERS.items():
            n = Counter()
            for r in parser(sources / name):
                r = dict(r, tree=to_json(r['tree']) if r.get('tree') is not None else None)
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
                records.append(r)
                n['parsed'] += 1
                n['verified'] += r['verified']
                if r.get('composition'):
                    n[f"composition:{r['composition']}:{'verified' if r['verified'] else 'unverified'}"] += 1
            per_source[name] = dict(n)
            print(name, dict(n), flush=True)
    reg = Registry.induce(records)
    reg.save(out)
    admitted = [c for c in reg.cores.values() if c['admitted']]
    report = {'status': 'source_induced_skeleton_registry', 'min_support': MIN_SUPPORT,
              'sources_manifest_sha256': sha256(sources / 'manifest.json'),
              'per_source': per_source, 'core_cards': len(reg.cores), 'admitted_core_cards': len(admitted),
              'admitted_core_support_by_source': dict(sum((Counter(c['support']) for c in admitted), Counter())),
              'bridge_compositions_verified': reg.bridge_support(),
              'multi_table_compositions_verified': reg.multi_table_support(),
              'nei_evidence_sets': sum(reg.compositions['nei_evidence'].values()), 'checks': {}}
    for path in check:
        by_skeleton = defaultdict(Counter)
        results = {}
        for r, tabs in load_programs(path):
            res = reg.admit_program(r['program']['operators'], tabs)
            key = (res['signature'], res['topology'])
            results[key] = res
            by_skeleton[r['program']['skeleton_id']][key] += 1
        rows = []
        for sk in sorted(by_skeleton):
            for key, n in by_skeleton[sk].most_common():
                res = results[key]
                rows.append({'skeleton_id': sk, 'records': n, 'signature': key[0], 'topology': key[1],
                             'admitted': res['admitted'], 'reason': res['reason'], 'card_id': res['card_id'],
                             'core_support': res.get('core_support') or (reg.cores.get(res.get('outer_signature') or key[0]) or {}).get('support'),
                             'inner_support': res.get('inner_support')})
        total = sum(r['records'] for r in rows)
        ok = sum(r['records'] for r in rows if r['admitted'])
        report['checks'][str(path)] = {'records': total, 'admitted_records': ok, 'rows': rows}
        print(f'\n== {path}: {ok}/{total} records use an admitted card')
        for r in rows:
            print(f"  {'OK ' if r['admitted'] else 'NO '} {r['skeleton_id']:20s} {r['records']:5d} {r['topology']:36s} "
                  f"{r['signature']}  <- {r['core_support']} {r['reason'] if not r['admitted'] else ''}")
    (out / 'induction_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sources', type=Path, default=ROOT / 'runs/sources')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check', type=Path, nargs='*', default=[])
    args = parser.parse_args()
    main(args.sources, args.out, args.check)
