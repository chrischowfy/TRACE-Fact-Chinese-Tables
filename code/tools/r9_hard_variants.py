"""R9 hard variants: structurally harder REFUTES / NEI derived from the screened R8 candidate pool.

    python tools/r9_hard_variants.py --pool runs/review_repair_r8_pool_v7 --out runs/review_repair_r9_hard_v1

From every screened SUPPORTS candidate of a numeric or argmax skeleton, at most one REFUTES is made
by changing only the stated fact to a close wrong one (tools/r9_hardness.py):
  near_miss_number   the stated number moved by 5-12% (one unit for counts and ranks), at the claim's
                     own precision;
  runner_up_value    an extreme value replaced by the second-best value of the same set;
  runner_up_entity   an argmax/argmin entity replaced by the second-ranked entity.
These are the 'number' / 'entity_or_value' refutation axes the source cards already admit.
From every screened multi-table SUPPORTS/REFUTES candidate, at most one NEI is made by removing a
table the program needs (missing_table).  The claim is unchanged; its support is the FEVEROUS NEI
evidence sets of the registry (relevant but insufficient evidence; >= 3 required), and a table is
not removed when what it supplies can still be read from the remaining tables.
Every record is re-executed (labels by execution), keeps its parent's admitted skeleton card, and is
screened again by tools/prepare_review_r7.py; binder-derived REFUTES carry per-table witnesses.
No model output is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from dart_fact.skeleton_registry import Registry
from dart_fact.tables import entity_key
from repair_review_release import digest, repair_record
import r8_cards
from r9_hardness import ARGMAX, EXTREME_VALUE, ranking, env_before_compare

NUMBER_SLOTS = ('value', 'diff', 'total', 'pct')
COUNTS = {'ST_COUNT_GT', 'CAT_COUNT', 'CAT_COUNT_GT', 'JF_COUNT_GT'}
FACTORS = (1.05, 0.95, 1.08, 0.92, 1.12, 0.88)
AGGREGATE_NAME = re.compile(r'合计|总计|总数|全国|全球|世界|其他|其它|平均')


def hydrate(c, tables):
    row = copy.deepcopy(c)
    row['tables'] = [{**tables[r['table_id']], **({'period': r['period']} if 'period' in r else {})}
                     for r in row.pop('table_refs')]
    return row


def compact(row):
    row = copy.deepcopy(row)
    row['table_refs'] = [{'table_id': t['table_id'], **({'period': t['period']} if 'period' in t else {})}
                         for t in row.pop('tables')]
    return row


def replace_once(text, old, new):
    """Replace the number/name `old` in the claim if it occurs exactly once as a whole token."""
    pattern = re.compile(r'(?<![\d.])' + re.escape(old) + r'(?![\d.])' if re.fullmatch(r'[\d.]+', old) else re.escape(old))
    hits = pattern.findall(text)
    return pattern.sub(new, text, count=1) if len(hits) == 1 else None


def decimals(text):
    return len(text.split('.')[1]) if '.' in text else 0


def fmt(x, d):
    return format(x, f'.{d}f')


def near_numbers(row):
    """(kind, new value, new text, slot key, extra) candidates for a numeric SUPPORTS record, best first."""
    s, sk = row['program']['slots'], row['program']['skeleton_id']
    base = digest(row['id'])
    if sk in COUNTS and isinstance(s.get('n'), int):
        n = s['n']
        steps = [1, -1] if int(base[:2], 16) % 2 else [-1, 1]
        return [('near_miss_number', n + d, str(n + d), 'n', {}) for d in steps if n + d >= 1]
    if sk == 'RJ_MEMBER_RANK' and isinstance(s.get('k'), int):
        k = s['k']
        return [('near_miss_number', k + d, str(k + d), 'k', {}) for d in (1, -1) if k + d >= 1]
    key = next((k for k in NUMBER_SLOTS if isinstance(s.get(k), dict)), None)
    if key is None:
        return []
    v, text = float(s[key]['value']), s[key]['text']
    d = decimals(text)
    out = []
    if sk in EXTREME_VALUE:
        pairs, _ = ranking(row)
        distinct = []
        for _, x in pairs:
            if x not in distinct:
                distinct.append(x)
        if len(distinct) >= 2 and abs(distinct[0] - v) < 1e-9:
            second = distinct[1]
            if abs(round(second, d) - second) < 1e-9:
                out.append(('runner_up_value', second, fmt(second, d), key, {}))
    order = sorted(FACTORS, key=lambda f: digest([base, f]))
    for f in order:
        new = round(v * f, d)
        if d == 0 and abs(v) < 40:
            new = v + (1 if f > 1 else -1) * max(1, round(abs(v) * abs(f - 1)))
        if new == v or (v > 0 and new <= 0) or abs(new - v) < 10 ** -d:
            continue
        out.append(('near_miss_number', new, fmt(new, d), key, {'relative_error': round(abs(new - v) / abs(v), 3)}))
    return out


def runner_up(row):
    pairs, _ = ranking(row)
    if len(pairs) < 2 or pairs[1][1] == pairs[0][1]:
        return None
    name = pairs[1][0]
    title = row['tables'][0]['source'].get('page_title', '')
    if AGGREGATE_NAME.search(name) or r8_cards.is_subject_row(name, title):
        return None
    return name


def derived(parent, claim, ops, slots, kind, extra, label, tables=None):
    row = copy.deepcopy(parent)
    row['claim'] = claim
    row['surface'] = {'kind': 'template', 'template_claim': claim}
    row['program']['operators'] = ops
    row['program']['slots'] = slots
    if tables is not None:
        row['tables'] = tables
        row['evidence_package_id'] = 'r9pkg-' + digest(tables)[:24]
    q = row['quality_flags']
    parent_origin = q.get('r7_origin')
    for k in ('necessity_witnesses', 'semantic_screen', 'semantic_screen_scope', 'nei_missing_binding'):
        q.pop(k, None)
    q.update(candidate_origin='r9_hard_variant', parent_origin=parent_origin,
             hard_variant={'kind': kind, 'parent_id': parent['id'], **extra},
             perturbation_axis={'missing_table': 'missing_table', 'runner_up_entity': 'entity_or_value'}.get(kind, 'number'))
    row['id'] = 'r9hv-' + digest([kind, ops, [t['table_id'] for t in row['tables']], claim])[:24]
    row['label'] = label
    fixed, why = repair_record(row)
    if not fixed or fixed['label'] != label:
        return None, (why or {}).get('reason', 'label_changed')
    if tables is not None:
        fixed['table_topology'] = ('single' if len(tables) == 1 else 'two_table' if len(tables) == 2
                                   else 'three_plus_table')
        fixed['evidence_package_id'] = 'r9pkg-' + digest(tables)[:24]
    return fixed, None


def removed_readable(parent, tid):
    """Is what table `tid` supplies still readable from the other tables of the package?"""
    ops = parent['program']['operators']
    slots = parent['program']['slots']
    removed = next(t for t in parent['tables'] if t['table_id'] == tid)
    rest = [t for t in parent['tables'] if t['table_id'] != tid]
    cells = lambda t: {entity_key(c) for r in t['rows'] for c in r}  # noqa: E731
    for op in (o for o in ops if o.get('table') == tid):
        for t in rest:
            other_period = t.get('period') and removed.get('period') and t['period'] != removed['period']
            if op['op'] == 'LOOKUP' and op.get('col') in t['headers'] and not other_period \
                    and entity_key(op.get('key', '')) in cells(t):
                return 'lookup_value_in_other_table'
            if op['op'] in {'VALUES', 'ARGEXT', 'EXTVAL'} and op.get('col') in t['headers'] and not other_period:
                return 'metric_column_in_other_table'
            if op['op'] in {'ATTR', 'JOIN', 'FILTER'}:
                cat = slots.get('cat') or slots.get('attr')
                if cat and '类别编号' not in t['headers'][:1] and entity_key(cat) in cells(t):
                    return 'category_value_in_other_table'
            if op['op'] == 'AT_RANK' and op.get('col') in t['headers']:
                return 'rank_column_in_other_table'
    return None


def main(pool: Path, out: Path, registry_dir: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new/empty')
    reg = Registry.load(registry_dir)
    r8_cards.REGISTRY = reg
    nei_support = reg.compositions.get('nei_evidence', {})
    tables = {t['table_id']: t for t in map(json.loads, (pool / 'tables.jsonl').open(encoding='utf-8'))}
    cands = [json.loads(l) for l in (pool / 'candidates.jsonl').open(encoding='utf-8')]
    rows, used_tables, stats, rejects = [], {}, Counter(), Counter()

    def keep(row, parent):
        card = reg.admit_program(row['program']['operators'],
                                 parent['tables'] if row['label'] == 'NEI' else row['tables'])
        if not card['admitted'] or card['signature'] != parent['quality_flags']['skeleton_card']['signature']:
            rejects['card_changed'] += 1
            return
        row['quality_flags']['skeleton_card'] = {k: card[k] for k in ('card_id', 'core_card', 'signature', 'topology')}
        if row['label'] == 'NEI':
            source = 'feverous_multi_table' if len(row['tables']) >= 2 else 'feverous_single_table'
            if nei_support.get(source, 0) < 3:
                rejects['no_nei_evidence_support'] += 1
                return
            row['quality_flags']['skeleton_card']['nei_evidence_support'] = {source: nei_support[source]}
        elif row['quality_flags'].get('parent_origin') == 'r8_source_card_binder':
            witnesses = r8_cards.table_witnesses(row)
            if not witnesses:
                rejects['no_witness'] += 1
                return
            row['quality_flags']['necessity_witnesses'] = witnesses
        for t in parent['tables']:
            used_tables[t['table_id']] = {k: v for k, v in t.items() if k != 'period'}
        rows.append(compact(row))
        stats[row['quality_flags']['hard_variant']['kind']] += 1

    for c in cands:
        parent = hydrate(c, tables)
        sk, label = parent['program']['skeleton_id'], parent['label']
        ops, slots = parent['program']['operators'], parent['program']['slots']
        try:
            # 1. close REFUTES from SUPPORTS
            if label == 'SUPPORTS':
                made = False
                for kind, new, text, key, extra in near_numbers(parent):
                    old = str(slots[key]['text']) if isinstance(slots[key], dict) else str(slots[key])
                    claim = replace_once(parent['claim'], old, text)
                    if claim is None:
                        rejects['number_not_unique_in_claim'] += 1
                        break
                    new_ops = copy.deepcopy(ops)
                    new_ops[-1]['right'] = int(new) if isinstance(ops[-1]['right'], int) else float(new)
                    new_slots = copy.deepcopy(slots)
                    if isinstance(slots[key], dict):
                        new_slots[key] = {**slots[key], 'value': float(new), 'text': text}
                    else:
                        new_slots[key] = int(new)
                    row, why = derived(parent, claim, new_ops, new_slots, kind, extra, 'REFUTES')
                    if row:
                        keep(row, parent)
                        made = True
                        break
                    rejects[why] += 1
                if not made and sk in ARGMAX and isinstance(slots.get('claimed'), str):
                    name = runner_up(parent)
                    claim = replace_once(parent['claim'], slots['claimed'], name) if name else None
                    if claim:
                        new_ops = copy.deepcopy(ops)
                        new_ops[-1]['right'] = name
                        row, why = derived(parent, claim, new_ops, {**slots, 'claimed': name}, 'runner_up_entity',
                                           {}, 'REFUTES')
                        if row:
                            keep(row, parent)
                        else:
                            rejects[why] += 1
            # 2. missing-table NEI from multi-table SUPPORTS / REFUTES
            if label in {'SUPPORTS', 'REFUTES'} and len(parent['tables']) >= 2:
                ablations = parent['quality_flags'].get('table_ablation_labels') or {}
                order = sorted((tid for tid, l in ablations.items() if l == 'NEI'), key=lambda t: digest([c['id'], t]))
                for tid in order:
                    why = removed_readable(parent, tid)
                    if why:
                        rejects[why] += 1
                        continue
                    rest = [t for t in parent['tables'] if t['table_id'] != tid]
                    row, why = derived(parent, parent['claim'], copy.deepcopy(ops), copy.deepcopy(slots),
                                       'missing_table', {'removed_table': tid}, 'NEI', tables=rest)
                    if row:
                        row['quality_flags']['nei_removed_table'] = tid
                        keep(row, parent)
                        break
                    rejects[why] += 1
        except Exception as exc:  # noqa: BLE001 - a parent that cannot be re-run yields no variant
            rejects['error:' + type(exc).__name__] += 1
    out.mkdir(parents=True)
    with (out / 'candidates.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    with (out / 'tables.jsonl').open('w', encoding='utf-8') as fh:
        for tid in sorted(used_tables):
            fh.write(json.dumps(used_tables[tid], ensure_ascii=False, sort_keys=True) + '\n')
    summary = {'status': 'supplemental_hard_variant_candidates_not_release', 'parents': len(cands),
               'records': len(rows), 'by_kind': dict(stats), 'labels': dict(Counter(r['label'] for r in rows)),
               'by_skeleton': dict(Counter(r['program']['skeleton_id'] for r in rows)),
               'rejects': dict(rejects.most_common()), 'nei_evidence_support': dict(nei_support), 'api_calls': 0}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=ROOT / 'runs/skeleton_registry_v1')
    a = parser.parse_args()
    main(a.pool, a.out, a.registry)
