"""Blind DeepSeek-V4-Pro pre-audit of a stratified sample of an R8 release (model opinion, not a human judgment).

    python tools/model_pre_audit_r8.py sample --release runs/review_repair_r8_select_v2_nat
    DEEPSEEK_API_KEY=... python tools/model_pre_audit_r8.py run --limit 20     # pilot; later runs resume
    DEEPSEEK_API_KEY=... python tools/model_pre_audit_r8.py run
    python tools/model_pre_audit_r8.py report
    # top-up: audit only records a rebuilt release adds or rewords (not in the audited release)
    python tools/model_pre_audit_r8.py sample --name topup_v3 --release runs/review_repair_r8_select_v3_nat \
        --exclude runs/review_repair_r8_select_v2_nat --total 60

The model sees only the claim and the full tables (model_pre_audit.payload): no label, program, skeleton,
origin or evidence hint.  The label stage runs in thinking mode so that arithmetic slips of the auditor do
not dominate the disagreements (the r6 pre-audit ran without thinking).  After the label is fixed, a second
non-thinking turn checks the dependency cells, as in the r6 protocol.  The sample is drawn before any call
and is stratified by origin x group x label; it is not a population accuracy estimate, and nothing here
feeds selection or edits the release.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
sys.path.insert(0, str(ROOT / 'tools'))
import common as C
from complete_model_pre_audit import relaxed_validate
from diagnose_revision_preview import normalize_output
from model_pre_audit import SYSTEM, payload

OUT = ROOT / 'runs/r8_pre_audit'
MODEL = 'deepseek-v4-pro'
SEED = 'r8-pre-audit-v1'
ALLOCATION = {'r6': 100, 'fresh_cached_generation': 120, 'r8_source_card_binder': 80}
EVIDENCE_PROMPT = ('保持刚才标签不变。现在检查以下依赖提示是否相关且足以核验；NEI需检查完整表格的缺失信息。'
                   '提示为空或不确定用U；忽略同表矛盾用0。提示不要求最小，不应仅因含竞争项判0。'
                   '只输出JSON：evidence_ok(1/0/U), evidence_note(中文)。提示：')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_key(record_id: str) -> str:
    return hashlib.sha1((SEED + record_id).encode()).hexdigest()


def allocate(sizes: dict, total: int) -> dict:
    """Largest remainder over strata, at least one item per non-empty stratum."""
    n = sum(sizes.values())
    base = {k: max(1, int(total * v / n)) for k, v in sizes.items()}
    rest = sorted(sizes, key=lambda k: (-(total * sizes[k] / n - int(total * sizes[k] / n)), k))
    i = 0
    while sum(base.values()) < total:
        k = rest[i % len(rest)]
        if base[k] < sizes[k]:
            base[k] += 1
        i += 1
    while sum(base.values()) > total:
        k = max((k for k in base if base[k] > 1), key=lambda k: (base[k] / sizes[k], k))
        base[k] -= 1
    return base


def binding(row):
    # program, tables and wording: a record whose sentence changed is audited again
    return hashlib.sha256(json.dumps([row['program']['operators'], [t['table_id'] for t in row['tables']], row['claim']],
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sample(release: Path, total: int = 0, exclude: Path | None = None) -> None:
    manifest = json.loads((release / 'snapshot_manifest.json').read_text())
    if digest(release / 'claims.jsonl') != manifest['claims_sha256']:
        raise SystemExit('claims.jsonl differs from its snapshot manifest')
    rows = [json.loads(l) for l in (release / 'claims.jsonl').open(encoding='utf-8')]
    allocation = ALLOCATION
    if exclude is not None:
        seen = {binding(json.loads(l)) for l in (exclude / 'claims.jsonl').open(encoding='utf-8')}
        rows = [r for r in rows if binding(r) not in seen]
        allocation = allocate(Counter(r['quality_flags'].get('r7_origin') for r in rows), min(total, len(rows)))
    items = []
    for origin, total in allocation.items():
        strata = defaultdict(list)
        for r in rows:
            if r['quality_flags'].get('r7_origin') == origin:
                strata[(r['program']['category'], r['label'])].append(r)
        quota = allocate({k: len(v) for k, v in strata.items()}, total)
        for k, members in sorted(strata.items()):
            for r in sorted(members, key=lambda r: order_key(r['id']))[:quota[k]]:
                items.append({'id': r['id'], 'origin': origin, 'group': k[0], 'label': r['label'],
                              'surface': r['surface']['kind'], 'tables': len(r['tables']),
                              'skeleton_id': r['program']['skeleton_id']})
    items.sort(key=lambda x: order_key(x['id']))
    for n, x in enumerate(items, 1):
        x['item'] = n
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'sample.jsonl').open('w', encoding='utf-8') as fh:
        for x in items:
            fh.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + '\n')
    meta = {'release': str(release.relative_to(ROOT)), 'claims_sha256': manifest['claims_sha256'], 'seed': SEED,
            'allocation': allocation, 'excluded_release': str(exclude.relative_to(ROOT)) if exclude else None, 'strata': 'origin x group x label, largest remainder, >=1 per stratum',
            'items': len(items), 'code_sha256': digest(Path(__file__))}
    (OUT / 'sample_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'items': len(items), 'by_origin': Counter(x['origin'] for x in items),
                      'by_label': Counter(x['label'] for x in items),
                      'by_surface': Counter(x['surface'] for x in items),
                      'by_tables': Counter(x['tables'] for x in items)}, ensure_ascii=False))


def load_sample():
    meta = json.loads((OUT / 'sample_meta.json').read_text())
    release = ROOT / meta['release']
    if digest(release / 'claims.jsonl') != meta['claims_sha256']:
        raise SystemExit('release changed after the sample was drawn')
    claims = {r['id']: r for r in map(json.loads, (release / 'claims.jsonl').open(encoding='utf-8'))}
    items = [json.loads(l) for l in (OUT / 'sample.jsonl').open(encoding='utf-8')]
    return meta, claims, items


def run(limit: int, workers: int) -> None:
    meta, claims, items = load_sample()
    key = C.api_key('DEEPSEEK_API_KEY')
    label_extra = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high'}
    config = {'model': MODEL, 'system': SYSTEM, 'claims_sha256': meta['claims_sha256'],
              'sample_sha256': digest(OUT / 'sample.jsonl'), 'protocol_version': 'r8_v1_thinking_label_stage',
              'label_stage': {'max_tokens': 16384, 'temperature': 0, **label_extra},
              'evidence_stage': {'max_tokens': 2048, 'temperature': 0, 'thinking': {'type': 'disabled'}},
              'human': False}
    fp = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run_dir = OUT / MODEL
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / 'run_meta.json'
    if meta_path.exists():
        assert json.loads(meta_path.read_text()) == config, 'resume configuration changed'
    else:
        meta_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    path = run_dir / 'judgments.jsonl'
    old = [json.loads(l) for l in path.open(encoding='utf-8')] if path.exists() else []
    assert all(r['run_fingerprint'] == fp for r in old)
    done = {r['item'] for r in old if r['status'] == 'complete'}
    pending = [x for x in items[:limit or None] if x['item'] not in done]

    def one(x):
        claim = claims[x['id']]
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': json.dumps(payload(claim, x['item']), ensure_ascii=False)}]
        rec = {'item': x['item'], 'id': x['id'], 'run_fingerprint': fp, 'human': False, 'status': 'incomplete',
               'input_sha256': hashlib.sha256(messages[-1]['content'].encode()).hexdigest(), 'api_calls': []}
        raw, api = C.call_llm(messages, model=MODEL, base_url='https://api.deepseek.com', api_key=key,
                              max_tokens=16384, temperature=0, extra=label_extra, timeout=600)
        rec['api_calls'].append({'stage': 'label', 'raw_output': raw, 'api': api})
        try:
            assert not api.get('error')
            j, _ = normalize_output(raw)
            rec['workflow_warnings'] = relaxed_validate(j, claim)
        except (ValueError, TypeError, KeyError, AssertionError, AttributeError):
            rec['status'] = 'invalid_label_output'
            return rec
        rec['label_stage_judgment'] = j
        cells = claim['evidence_cells'] or claim['context_cells']
        index = {t['table_id']: f't{i}' for i, t in enumerate(claim['tables'])}
        hints = [{'table': index[c['table_id']], 'row': c['row'], 'column': c['col'], 'value': c['raw_value']}
                 for c in cells]
        messages += [{'role': 'assistant', 'content': raw},
                     {'role': 'user', 'content': EVIDENCE_PROMPT + json.dumps(hints, ensure_ascii=False)}]
        raw2, api2 = C.call_llm(messages, model=MODEL, base_url='https://api.deepseek.com', api_key=key,
                                max_tokens=2048, temperature=0, extra={'thinking': {'type': 'disabled'}})
        rec['api_calls'].append({'stage': 'evidence', 'raw_output': raw2, 'api': api2})
        try:
            assert not api2.get('error')
            ev, _ = normalize_output(raw2)
            assert str(ev['evidence_ok']) in ('1', '0', 'U')
            rec['judgment'] = j | {'evidence_ok': str(ev['evidence_ok']), 'evidence_note': ev.get('evidence_note', '')}
            rec['status'] = 'complete'
        except (ValueError, TypeError, KeyError, AssertionError, AttributeError):
            rec['status'] = 'invalid_evidence_output'
        return rec

    print(f'{len(pending)} items pending ({len(done)} complete)', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, path.open('a', encoding='utf-8') as fh:
        for n, fut in enumerate(as_completed([pool.submit(one, x) for x in pending]), 1):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fh.flush()
            print(f"{n}/{len(pending)} item={rec['item']} {rec['status']}", flush=True)


def report() -> None:
    meta, claims, items = load_sample()
    by_item = {x['item']: x for x in items}
    latest = {}
    for r in map(json.loads, (OUT / MODEL / 'judgments.jsonl').open(encoding='utf-8')):
        if r['status'] == 'complete' or r['item'] not in latest:
            latest[r['item']] = r          # a complete retry supersedes an earlier failed attempt
    done = [r for r in latest.values() if r['status'] == 'complete']
    usage = Counter()
    for r in map(json.loads, (OUT / MODEL / 'judgments.jsonl').open(encoding='utf-8')):
        for call in r['api_calls']:
            for k in ('prompt_tokens', 'completion_tokens'):
                usage[k] += (call['api'].get('usage') or {}).get(k, 0)
    table = defaultdict(lambda: Counter())
    disagreements, uncertain, low_nat, evidence_bad, necessity_zero = [], [], [], [], []
    for r in sorted(done, key=lambda r: r['item']):
        x, j = by_item[r['item']], r['judgment']
        j = j | {'needs_all_tables': str(j['needs_all_tables'])}   # the model returns 1 or "1"
        model_label = j['ann_label']
        row = {'item': x['item'], 'id': x['id'], 'origin': x['origin'], 'group': x['group'],
               'skeleton_id': x['skeleton_id'], 'surface': x['surface'], 'tables': x['tables'],
               'gold': x['label'], 'model': model_label, 'naturalness': j['naturalness'],
               'claim': claims[x['id']]['claim'], 'note': j['note'], 'issues': j['issues'],
               'evidence_ok': j['evidence_ok'], 'needs_all_tables': j['needs_all_tables']}
        for dim in ('all', 'origin', 'group', 'label', 'surface', 'tables'):
            key = 'all' if dim == 'all' else f"{dim}={x['label' if dim == 'label' else dim]}"
            c = table[key]
            c['n'] += 1
            c['uncertain'] += model_label == 'UNCERTAIN'
            c['definite'] += model_label != 'UNCERTAIN'
            c['agree'] += model_label == x['label']
            c['naturalness_sum'] += j['naturalness']
        if model_label == 'UNCERTAIN':
            uncertain.append(row)
        elif model_label != x['label']:
            disagreements.append(row)
        if j['naturalness'] <= 2:
            low_nat.append(row)
        if j['evidence_ok'] == '0':
            evidence_bad.append(row)
        if j['needs_all_tables'] == '0':
            necessity_zero.append(row)
    summary = {k: {'n': c['n'], 'definite': c['definite'], 'uncertain': c['uncertain'],
                   'agreement_on_definite_pct': round(100 * c['agree'] / c['definite'], 1) if c['definite'] else None,
                   'mean_naturalness': round(c['naturalness_sum'] / c['n'], 2)} for k, c in sorted(table.items())}
    out = {'sample_items': len(items), 'complete': len(done), 'failed_items': sorted(set(latest) - {r['item'] for r in done}),
           'usage': dict(usage), 'summary': summary,
           'naturalness_hist': dict(Counter(r['judgment']['naturalness'] for r in done)),
           'evidence_ok_hist': dict(Counter(r['judgment']['evidence_ok'] for r in done)),
           'needs_all_tables_hist': dict(Counter(str(r['judgment']['needs_all_tables']) for r in done)),
           'disagreements': disagreements, 'uncertain': uncertain, 'naturalness_le2': low_nat,
           'evidence_ok_0': evidence_bad, 'needs_all_tables_0': necessity_zero,
           'limitations': 'Model opinion only (not human validation); stratified sample, not a population estimate; '
                          'disagreements need adjudication against the tables before any conclusion.'}
    (OUT / 'report.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('sample_items', 'complete', 'failed_items', 'usage')}, ensure_ascii=False))
    for k in ('all', 'origin=r6', 'origin=fresh_cached_generation', 'origin=r8_source_card_binder'):
        print(k, summary.get(k))
    print({k: len(out[k]) for k in ('disagreements', 'uncertain', 'naturalness_le2', 'evidence_ok_0', 'needs_all_tables_0')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['sample', 'run', 'report'])
    parser.add_argument('--release', type=Path, default=ROOT / 'runs/review_repair_r8_select_v2_nat')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--name', default='', help='sub-directory of runs/r8_pre_audit for a separate sample')
    parser.add_argument('--exclude', type=Path, help='sample only records whose binding is not in this release')
    parser.add_argument('--total', type=int, default=0)
    a = parser.parse_args()
    OUT = OUT / a.name if a.name else OUT
    if a.action == 'sample':
        sample(a.release.resolve(), a.total, a.exclude.resolve() if a.exclude else None)
    elif a.action == 'run':
        run(a.limit, a.workers)
    else:
        report()
