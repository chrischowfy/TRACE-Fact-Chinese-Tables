"""Slot-locked LLM paraphrase of the template claims of a selected r8 candidate.

The same method as src/dart_fact/naturalize.py (rewrite model on a placeholder sentence, surface
checks, independent judge against an unambiguous program gloss, sibling claims share wording),
with one change: the placeholder sentence is recovered from the record itself.  Every fact-bearing
slot value (entities, category, numbers, years, comparison / extreme / change words) must occur
exactly once in the template claim and is replaced by a placeholder; the group is paraphrased only
if refilling the placeholders reproduces every sibling's template claim exactly.  Records kept from
r6 are never touched.  Labels, programs, tables and evidence are unchanged.

    python tools/naturalize_review_r8.py --release runs/review_repair_r8_select_v2 --dry-run
    DEEPSEEK_API_KEY=... python tools/naturalize_review_r8.py --release ... --out ... --call-api
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from dart_fact import naturalize as N
from dart_fact import realize as R
from dart_fact.source_adapters import comparison_words
from repair_review_release import audit, file_hash

ENTITY_KEYS = ('e', 'e1', 'e2', 'claimed', 'member', 'cat', 'attr', 'k', 'n')
NUMBER_KEYS = ('value', 'diff', 'threshold', 'total', 'pct')
CONTEXT_KEYS = ('cat_col', 'attr_col', 'rank_noun', 'metric_name', 'noun')
CAPTION_KEYS = ('member_table', 'metric_table')   # table captions named in a claim; locked verbatim


def _extreme_word(s):
    return '最大' if s.get('mode', 'max') == 'max' else '最小'


def gloss(skeleton: str, s: dict) -> str:
    """Program gloss for the judge; r8 binder skeletons added to naturalize.gloss."""
    sc, m, noun = re.sub(r'中$', '', s.get('scope', '')), s.get('metric_name', ''), s.get('noun', '对象')
    if s.get('metric_table'):
        m = f"“{s['metric_table']}”中的{m}"
    col, val = s.get('attr_col', s.get('cat_col', '')), s.get('attr', s.get('cat', ''))
    num = lambda k: R._num(s[k]) if isinstance(s[k], dict) else str(s[k])  # noqa: E731
    if skeleton in ('CAT_EXTVAL', 'JF_EXTVAL'):
        return f"在{sc}中，{col}为{val}的所有{noun}里，{m}数值的{_extreme_word(s)}值等于{num('value')}。"
    if skeleton in ('CAT_AVG', 'JF_AVG'):
        return f"在{sc}中，{col}为{val}的所有{noun}的{m}数值的算术平均值（按句中精度）等于{num('value')}。"
    if skeleton in ('CAT_COUNT_GT', 'JF_COUNT_GT'):
        return f"在{sc}中，{col}为{val}的{noun}里，{m}超过{_threshold_text(s)}的共有{s['n']}个。"
    if skeleton in ('CAT_SUM', 'JF_SUM'):
        return f"在{sc}中，{col}为{val}的所有{noun}的{m}数值相加，总和等于{num('total')}。"
    if skeleton == 'CP_PCT':
        v = s['pct']['value']
        return (f"在{sc}中，{s['e']}的{m}从{R._year(s['y1'])}到{R._year(s['y2'])}的变化率"
                f"（后值减前值再除以前值，按句中精度）为{'增长' if v > 0 else '下降'}{s['pct']['text']}%。")
    if skeleton == 'CAT_ARGEXT_ATTR':
        return (f"在{sc}中，{col}为{val}的所有{noun}里，{m}数值{_extreme_word(s)}的那一个，"
                f"其{s['attr_col']}是{s['claimed']}。")
    return N.gloss(skeleton, s)


def _threshold_text(s):
    t = s['threshold']
    if isinstance(t, dict):
        return R._num(t)
    return format(float(t), 'f').rstrip('0').rstrip('.')


def slot_strings(row: dict) -> list[tuple[str, list[str]]]:
    """(placeholder, candidate surface strings) for every fact-bearing slot of a record."""
    s = row['program']['slots']
    sk = row['program']['skeleton_id']
    style = s.get('style', 'amount')
    words = comparison_words(style)
    # captions first: they may contain digits or the metric name (2020年人均GDP超万美元地级市)
    out = [(k, [str(s[k])]) for k in CAPTION_KEYS if s.get(k)]
    for k in ENTITY_KEYS:
        if k in s and not isinstance(s[k], dict):
            out.append((k, [str(s[k])]))
    for k in ('y1', 'y2'):
        if k in s:
            out.append((k, [R._year(s[k])]))
    for k in NUMBER_KEYS:
        if k not in s:
            continue
        v = s[k]
        if k == 'pct':
            out.append((k, [v['text'] + '%']))
        elif isinstance(v, dict):
            out.append((k, [R._num(v), v.get('text', '')]))
        else:
            out.append((k, [_threshold_text(s)]))
    if s.get('cmp') in ('gt', 'lt', 'eq'):
        out.append(('cmp', [words.get(s['cmp'], '')]))
    if 'mode' in s:
        out.append(('mode', ['的最大值' if s['mode'] == 'max' else '的最小值', words[s['mode']]]))
    if 'direction' in s:
        out.append(('verb', [R._verb(s['direction'])]))
        out.append(('more', [R._more(style, s['direction'])]))
    if sk == 'CP_PCT':
        out.append(('verb', ['增长' if s['pct']['value'] > 0 else '下降']))
    for k in CONTEXT_KEYS:
        if s.get(k):
            out.append((k, [str(s[k])]))
    return out


def placeholder(row: dict) -> tuple[str, dict[str, str]] | None:
    """Recover the placeholder sentence of a template claim; None if any fact slot is ambiguous."""
    text = row['surface'].get('template_claim') or row['claim']
    values: dict[str, str] = {}
    for key, options in slot_strings(row):
        if key in values:
            continue
        hit = [o for o in options if o and text.count(o) == 1]
        if not hit:
            if key in CONTEXT_KEYS or key in ('more',):
                continue          # not stated in this wording
            return None
        values[key] = hit[0]
        text = text.replace(hit[0], f'〔{key}〕')
    if N.fill(text, values) != (row['surface'].get('template_claim') or row['claim']):
        return None
    if not any(k in values for k in ENTITY_KEYS + NUMBER_KEYS):
        return None
    return text, values


def plan(rows: list[dict]):
    groups, skipped = defaultdict(list), Counter()
    for i, row in enumerate(rows):
        if row['quality_flags'].get('r7_origin') == 'r6' or row['surface'].get('kind') != 'template':
            continue
        got = placeholder(row)
        if got is None:
            skipped[row['program']['skeleton_id']] += 1
            continue
        tpl, values = got
        base = row['quality_flags'].get('contrast_base') or row['id']
        groups[(base, tpl)].append((i, values))
    return groups, skipped


def run(release: Path, out: Path | None, cache_path: Path, call_api: bool, dry_run: bool, workers: int) -> None:
    rows = [json.loads(l) for l in (release / 'claims.jsonl').open(encoding='utf-8')]
    groups, skipped = plan(rows)
    members = sum(len(v) for v in groups.values())
    est = {'template_records_eligible': sum(1 for r in rows if r['quality_flags'].get('r7_origin') != 'r6'
                                            and r['surface'].get('kind') == 'template'),
           'groups': len(groups), 'records_in_groups': members, 'no_placeholder_by_skeleton': dict(skipped),
           # one rewrite call per group (+1 retry), one judge call per sibling per tried candidate
           'estimated_calls': {'rewrite': f'{len(groups)}-{2 * len(groups)}', 'judge': f'{members}-{6 * members}'}}
    print(json.dumps(est, ensure_ascii=False, indent=1))
    if dry_run:
        return
    cache = N.Cache(cache_path)
    stats, reasons = Counter(), Counter()
    new_rows = copy.deepcopy(rows)

    def work(gkey):
        base, tpl = gkey
        items = groups[gkey]
        values = [v for _, v in items]
        example = values[next((j for j, (i, _) in enumerate(items) if rows[i]['label'] == 'SUPPORTS'), 0)]
        locked = N.lock_template(tpl, values) if all(set(v) == set(values[0]) for v in values) else tpl
        rkey = N._key('rewrite', N.REWRITE_MODEL, locked, example)
        out1 = cache.get(rkey)
        if out1 is None and call_api:
            out1 = N._call(N.REWRITE_MODEL, N.REWRITE_SYSTEM, N.rewrite_request(locked, example), 600)
            if out1 is None:
                return gkey, None, 'api_unavailable'
            cache.put(rkey, out1)

        def attempt(output, why):
            for cand in [c.strip() for c in (output or {}).get('candidates', []) if isinstance(c, str)]:
                fail = N.surface_ok(locked, cand, values)
                if fail:
                    why.append(fail)
                    continue
                ok = True
                for (i, vals) in items:
                    a = gloss(rows[i]['program']['skeleton_id'], rows[i]['program']['slots'])
                    b = N.fill(cand, vals)
                    jkey = N._key('judge', N.JUDGE_MODEL, a, b)
                    verdict = cache.get(jkey)
                    if verdict is None and call_api:
                        verdict = N._call(N.JUDGE_MODEL, N.JUDGE_SYSTEM, N.judge_request(a, b), 200)
                        if verdict is not None:     # a failed call is not cached, so a rerun retries it
                            cache.put(jkey, verdict)
                    if not (isinstance(verdict, dict) and verdict.get('same') is True):
                        ok = False
                        break
                if ok:
                    return cand
                why.append('judge')
            return None

        why: list[str] = []
        cand = attempt(out1, why)
        if cand is None and out1 is not None:
            rkey2 = N._key('rewrite_retry', N.REWRITE_MODEL, locked, example)
            out2 = cache.get(rkey2)
            if out2 is None and call_api:
                out2 = N._call(N.REWRITE_MODEL, N.REWRITE_SYSTEM, N.rewrite_request(locked, example) + N.RETRY_NOTE, 600)
                if out2 is None:
                    return gkey, None, 'api_unavailable'
                cache.put(rkey2, out2)
            cand = attempt(out2, why)
        return gkey, cand, 'ok' if cand else (','.join(why) or 'no_candidates')

    import concurrent.futures as cf
    results = {}
    with cf.ThreadPoolExecutor(max_workers=workers if call_api else 1) as pool:
        for n, (gkey, cand, why) in enumerate(pool.map(work, list(groups)), 1):
            results[gkey] = (cand, why)
            if call_api and n % 50 == 0:
                cache.save()
                print(f'naturalize {n}/{len(groups)}', flush=True)
    if call_api:
        cache.save()
    changed = []
    for gkey, (cand, why) in results.items():
        if cand is None:
            stats['template_groups'] += 1
            reasons[why.split(',')[0]] += 1
            continue
        stats['paraphrased_groups'] += 1
        for i, vals in groups[gkey]:
            row = new_rows[i]
            # the claim schema allows only kind/template_claim; method and models go to paraphrase_log.jsonl
            row['surface'] = {'kind': 'llm_paraphrase', 'template_claim': rows[i]['claim']}
            row['claim'] = N.fill(cand, vals)
            changed.append(row['id'])
    from r7_quality import audit_issues
    for i, row in enumerate(new_rows):
        # a paraphrase must not introduce a quality-rule issue its template did not have (a dropped caption)
        if row['id'] in changed and set(audit_issues(row)) - set(audit_issues(rows[i])):
            new_rows[i] = rows[i]
            stats['reverted_audit_rule'] += 1
    texts = Counter(''.join(r['claim'].split()) for r in new_rows)
    dup = {t for t, c in texts.items() if c > 1}
    for i, row in enumerate(new_rows):
        if ''.join(row['claim'].split()) in dup and row['id'] in changed:
            new_rows[i] = rows[i]           # a paraphrase must not collide with another claim
            stats['reverted_duplicate'] += 1
    kept = {r['id'] for r in new_rows if r['id'] in set(changed) and r['surface']['kind'] == 'llm_paraphrase'}
    stats['records_paraphrased'] = len(kept)
    print(json.dumps({'stats': dict(stats), 'fallback_reasons': dict(reasons)}, ensure_ascii=False, indent=1))
    if out is None:
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new/empty')
    validation = audit(new_rows, json.loads((ROOT / 'schemas/claim.schema.json').read_text()))
    if not validation['passed']:
        raise ValueError('paraphrased release failed validation: ' + json.dumps(validation['issues'][:5], ensure_ascii=False))
    out.mkdir(parents=True)
    by_id = {r['id']: r for r in new_rows}
    for name in ('lineage.jsonl', 'necessity_witnesses.jsonl', 'table_snapshots.jsonl', 'skeleton_registry.jsonl'):
        if (release / name).exists():
            shutil.copy2(release / name, out / name)
    with (out / 'claims.jsonl').open('w', encoding='utf-8') as fh:
        for r in new_rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    with (release / 'selection.jsonl').open(encoding='utf-8') as src, (out / 'selection.jsonl').open('w', encoding='utf-8') as fh:
        for line in src:
            item = json.loads(line)
            item['claim'], item['surface'] = by_id[item['id']]['claim'], by_id[item['id']]['surface']
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    with (out / 'paraphrase_log.jsonl').open('w', encoding='utf-8') as fh:
        for r in new_rows:
            if r['id'] in kept:
                fh.write(json.dumps({'id': r['id'], 'method': 'slot_locked_paraphrase_r8',
                                     'prompt_version': N.PROMPT_VERSION, 'rewrite_model': N.REWRITE_MODEL,
                                     'judge_model': N.JUDGE_MODEL, 'template_claim': r['surface']['template_claim'],
                                     'claim': r['claim']}, ensure_ascii=False, sort_keys=True) + '\n')
    from dart_fact.analysis import statistics, surface_baselines
    (out / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n')
    (out / 'stats.json').write_text(json.dumps({'statistics': statistics(new_rows),
                                                'surface_baselines': surface_baselines(new_rows),
                                                'naturalize': {'stats': dict(stats), 'fallback_reasons': dict(reasons)}},
                                               ensure_ascii=False, indent=2) + '\n')
    manifest = json.loads((release / 'snapshot_manifest.json').read_text())
    manifest.update(release=manifest['release'] + '-paraphrased', claims_sha256=file_hash(out / 'claims.jsonl'),
                    surface='slot-locked LLM paraphrase of non-r6 template claims; labels, programs, tables unchanged',
                    paraphrase_source_release_claims_sha256=file_hash(release / 'claims.jsonl'),
                    paraphrase_cache_sha256=file_hash(cache_path),
                    paraphrase={'method': 'slot_locked_paraphrase_r8', 'prompt_version': N.PROMPT_VERSION,
                                'rewrite_model': N.REWRITE_MODEL, 'judge_model': N.JUDGE_MODEL,
                                'records': len(kept), 'log': 'paraphrase_log.jsonl'})
    manifest['artifacts_sha256'] = {p.name: file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / 'snapshot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'claims_sha256': manifest['claims_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs/cache/naturalize_r8.json')
    parser.add_argument('--call-api', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    a = parser.parse_args()
    run(a.release, a.out, a.cache, a.call_api, a.dry_run, a.workers)
