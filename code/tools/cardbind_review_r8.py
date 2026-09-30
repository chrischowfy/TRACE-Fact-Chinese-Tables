"""Bind source-admitted skeleton cards without an r6 binder to every cached package.

Category decompositions are built exactly as in catmean_review_r7.py (same page selection,
`packages._decompose_category`, 2-12 categories, >=6 rows); join-filter and period-pair packages
come from the pipeline's own `build_packages` (unlisted cached pages registered for this process,
as in catalog_review_r7.py --register-unlisted).  Every record is checked against the registry
(`r8_cards.card_record`), screened with the r7 and r6 rules, and must have a single-table
opposite-label witness for each of its tables.  No web/API calls; no values are generated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from dart_fact.acquire import extract_tables, to_simplified
from dart_fact.packages import _decompose_category, build_packages
from dart_fact.skeleton_registry import Registry
from dart_fact.source_adapters import SERIES, SENSITIVE_CELL_RE, SENSITIVE_TITLE_RE
from dart_fact.tables import entity_key, profile_table
from catalog_review_r7 import register_unlisted
from register_cached import classify
from repair_review_release import digest, file_hash
from repair_review_r6 import semantic_issues
import r8_cards
from r7_quality import issues

SPORT_DOMAINS = {'football', 'football_intl', 'basketball', 'medals'}


def build(cache, registry, out, partner_only=False):
    from r7_runtime import enable
    enable()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new/empty')
    r8_cards.REGISTRY = Registry.load(registry)
    rows, tables, rejects, seen = [], {}, Counter(), set()
    generated, by_origin = Counter(), Counter()

    def admit(row, origin):
        generated[(origin, row['program']['skeleton_id'])] += 1
        key = digest([row['program']['operators'], row['program']['table_ids']])
        if key in seen:
            rejects['duplicate_program_binding'] += 1
            return
        seen.add(key)
        found = issues(row) + [i['reason'] for i in semantic_issues(row)]
        witnesses = None if found else r8_cards.table_witnesses(row)
        if not found and not witnesses:
            found = ['no_opposite_label_witness']
        if witnesses:
            row['quality_flags']['necessity_witnesses'] = witnesses
        if found:
            rejects.update(found)
            by_origin.update((origin, f) for f in found)
            return
        row['quality_flags'].update(candidate_origin=origin)
        for t in row['tables']:
            prior = tables.get(t['table_id'])
            if prior and (prior['headers'], prior['rows']) != (t['headers'], t['rows']):
                raise ValueError('Conflicting table identity: ' + t['table_id'])
            tables[t['table_id']] = t
        row['table_refs'] = [{'table_id': t['table_id'], **({'period': t['period']} if 'period' in t else {})}
                             for t in row.pop('tables')]
        rows.append(row)

    meta = {to_simplified(t): (s['domain'], s['series']) for s in SERIES for t in s['titles']}
    for index, path in enumerate(sorted(cache.glob('*.json'))):
        page = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(page, dict) or not page.get('html'):
            continue
        title = to_simplified(page['title'])
        if SENSITIVE_TITLE_RE.search(title):
            continue
        domain_series = meta.get(title) or classify(title)
        if not domain_series or domain_series[0] in SPORT_DOMAINS:
            continue
        domain, series = domain_series
        domain = 'infrastructure' if domain == 'architecture' else domain
        page_tables = [t for t in extract_tables(page)
                       if not any(SENSITIVE_CELL_RE.search(c) for r in t['rows'] for c in r)]
        profiles = [(t, profile_table(t)) for t in page_tables]
        for table, prof in profiles:
            if not prof.key:
                continue
            for cat in prof.categories():
                n_cat = len({entity_key(prof.cell(r, cat)) for r in range(len(prof.rows))})
                if not (2 <= n_cat <= 12 and len(prof.rows) >= 6):
                    continue
                parent = {'tables': _decompose_category(prof, cat, page), 'domain': domain,
                          'series': series, 'topic': title, 'source_headers': table['headers'],
                          'source_table_id': table['table_id']}
                made = (r8_cards.partner_attribute_candidates(parent, profiles) if partner_only
                        else r8_cards.category_card_candidates(parent))
                for row in made:
                    row['quality_flags']['source_cache_sha256'] = file_hash(path)
                    admit(row, 'r8_category_partner_card_binder' if partner_only else 'r8_category_card_binder')
        if index and index % 200 == 0:
            print(f'Cached pages examined={index}, candidates={len(rows)}', flush=True)
    registered = [] if partner_only else register_unlisted(cache)
    packages = [] if partner_only else build_packages(str(cache), cache_only=True)
    kinds = Counter(p['kind'] for p in packages)
    print(f'Registered {len(registered)} unlisted pages; packages={dict(kinds)}', flush=True)
    for package in packages:
        if package['domain'] in SPORT_DOMAINS:
            continue
        if package['kind'] == 'join_filter':
            for row in r8_cards.join_filter_card_candidates(package):
                admit(row, 'r8_join_filter_card_binder')
        elif package['kind'] == 'period_pair':
            for row in r8_cards.period_card_candidates(package):
                admit(row, 'r8_period_card_binder')
    for row in r8_cards.period_window_candidates([p for p in packages if p['domain'] not in SPORT_DOMAINS]):
        admit(row, 'r8_period_window_card_binder')
    out.mkdir(parents=True, exist_ok=True)
    for name, items in [('candidates.jsonl', rows), ('tables.jsonl', [tables[k] for k in sorted(tables)])]:
        with (out / name).open('w', encoding='utf-8') as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + '\n')
    report = {'status': 'supplemental_source_card_candidates_not_release', 'records': len(rows),
              'skeletons': dict(Counter(r['program']['skeleton_id'] for r in rows)),
              'labels': dict(Counter(r['label'] for r in rows)),
              'tables_per_record': dict(Counter(len(r['table_refs']) for r in rows)),
              'topics': len({r['topic'] for r in rows}),
              'packages': len({r['evidence_package_id'] for r in rows}),
              'rejections': dict(rejects), 'registered_unlisted_pages': len(registered),
              'generated_by_origin_skeleton': {f'{o}|{k}': v for (o, k), v in sorted(generated.items())},
              'rejections_by_origin': {f'{o}|{k}': v for (o, k), v in sorted(by_origin.items())},
              'package_kinds': dict(kinds), 'registry': str(registry),
              'registry_core_cards_sha256': file_hash(registry / 'core_cards.jsonl'),
              'cache_dir': str(cache), 'api_calls': 0}
    (out / 'cardbind_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs/cache_r7/pages')
    parser.add_argument('--registry', type=Path, default=ROOT / 'runs/skeleton_registry_v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--partner-only', action='store_true',
                        help='only the four-table category + partner-attribute card (CAT_ARGEXT_ATTR)')
    args = parser.parse_args()
    build(args.cache.resolve(), args.registry.resolve(), args.out.resolve(), args.partner_only)
