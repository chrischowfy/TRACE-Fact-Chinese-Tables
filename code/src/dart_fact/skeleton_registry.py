"""Source-induced skeleton registry and the admission check applied to Chinese programs.

Induction (paper, Program Skeletons and Reasoning Families): source programs, derivations and
templates are parsed and delexicalized (``source_programs``); every verified program whose root
is a verification judgement becomes support for the *core card* of its AOL signature.  Multi-hop
bridges and multi-table evidence are compositions: they are supported by the sources whose
annotations record them (MultiModalQA Compose/Compare/Intersect, HybridQA traces, FEVEROUS
evidence sets).

Admission of a Chinese program with AOL signature s and table topology t:
  core    no bridge: >= MIN_SUPPORT verified source programs with signature s
          bridge: the outer program (holes filled) is an admitted core, every inner program occurs
          as a sub-expression of >= MIN_SUPPORT verified source programs, and >= MIN_SUPPORT verified
          cross-source bridge compositions exist
  topology  single_table / category_decomposition need nothing further (the decomposition is the
          package construction of the paper, adding no fact); every other multi-table topology needs
          >= MIN_SUPPORT verified multi-table evidence compositions
MIN_SUPPORT is fixed before any Chinese program is checked.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import aol

MIN_SUPPORT = 3
PROGRAM_SOURCES = ('tabfact', 'wtq', 'tatqa', 'multimodalqa')
OP_WHITELIST = {'ROWS', 'COL', 'STR', 'NUM', 'SUBQ', 'FILTER', 'FILTERS', 'COND', 'HOP', 'ARGEXT', 'NTH_ARGEXT',
                'ORDER', 'NEXT', 'EXTVAL', 'COUNT', 'SUM', 'AVG', 'MODE', 'RANK_OF', 'SUB', 'ADD', 'DIV', 'MUL',
                'PCT_CHANGE', 'AVG_OF', 'CMP_EQ', 'CMP_NEQ', 'CMP_ORD', 'AND', 'ONLY', 'ALL', 'MOST',
                'INTERSECT', 'UNION', 'EXCEPT', 'TEXTQ', 'PROJ'}
NO_EXTRA_TOPOLOGY = {'single_table', 'category_decomposition'}


def card_id(signature: str) -> str:
    return 'SK_' + hashlib.sha1(signature.encode('utf-8')).hexdigest()[:10]


def to_json(tree: Any) -> Any:
    return [to_json(x) for x in tree] if isinstance(tree, tuple) else tree


def from_json(tree: Any) -> Any:
    return tuple(from_json(x) for x in tree) if isinstance(tree, list) else tree


def schema_ok(tree: tuple) -> tuple[bool, str]:
    ops = {s[0] for s in aol.subtrees(tree)}
    bad = ops - OP_WHITELIST
    if bad:
        return False, 'operators outside whitelist: ' + ','.join(sorted(bad))
    if tree[0] not in aol.BOOL_ROOTS:
        return False, 'root is not a verification judgement'
    if ops & {'SUBQ', 'TEXTQ', 'INTERSECT'}:
        return False, 'composition with a non-table sub-question'
    if 'COL' not in ops:
        return False, 'no table column'
    return True, 'ok'


def refute_axes(tree: tuple) -> list[str]:
    ops = {s[0] for s in aol.subtrees(tree)}
    axes = []
    if any(s[0] in {'CMP_EQ', 'CMP_NEQ'} and any(c[0] == 'STR' for c in s[1:]) for s in aol.subtrees(tree)):
        axes.append('entity_or_value')
    if any(s[0] in {'CMP_EQ', 'CMP_NEQ'} and any(c[0] == 'NUM' for c in s[1:]) for s in aol.subtrees(tree)):
        axes.append('number')
    if 'CMP_ORD' in ops:
        axes.append('comparison_direction')
    if ops & {'ARGEXT', 'EXTVAL'}:
        axes.append('extreme_direction')
    if any(s[0] in {'FILTER', 'COND'} for s in aol.subtrees(tree)):
        axes.append('filter_value')
    return axes


def nei_axes(tree: tuple, topology: str | None = None) -> list[str]:
    axes = ['metric']
    if any(s[0] in {'FILTER', 'COND'} and (s[3] if s[0] == 'FILTER' else s[2]) == 'EQ' for s in aol.subtrees(tree)):
        axes.append('entity_or_condition')
    if topology == 'co_keyed':
        axes.append('period_or_partner_table')
    if topology == 'bridge':
        axes.append('join_partner')
    return axes


class Registry:
    def __init__(self, cores: dict[str, dict], fragments: Counter, compositions: dict[str, Any]):
        self.cores = cores
        self.fragments = fragments
        self.compositions = compositions

    # ------------------------------------------------------------------ building
    @classmethod
    def induce(cls, records: Iterable[dict]) -> 'Registry':
        cores: dict[str, dict] = {}
        fragments: Counter = Counter()
        comp = {'bridge_verified': Counter(), 'bridge_trace_only': Counter(), 'multi_table_verified': Counter(),
                'nei_evidence': Counter(), 'examples': defaultdict(list)}
        for r in records:
            if not r['verified']:
                continue
            src, c = r['source'], r.get('composition')
            if src == 'multimodalqa' and c == 'bridge':
                comp['bridge_verified'][r.get('qtype')] += 1
                _keep(comp['examples']['bridge'], r)
            if src == 'hybridqa' and c == 'bridge':
                comp['bridge_trace_only']['hybridqa_passage_hop'] += 1
            if src == 'multimodalqa' and c == 'parallel':
                comp['multi_table_verified'][r.get('qtype')] += 1
                _keep(comp['examples']['multi_table'], r)
            if src == 'feverous' and c == 'multi_table' and r.get('label') in {'SUPPORTS', 'REFUTES'}:
                comp['multi_table_verified']['feverous_multi_table_evidence'] += 1
                _keep(comp['examples']['multi_table'], r)
            if src == 'feverous' and r.get('label') == 'NOT ENOUGH INFO':
                comp['nei_evidence'][f"feverous_{c}"] += 1
            if src not in PROGRAM_SOURCES or r.get('tree') is None:
                continue
            tree = from_json(r['tree'])
            for f in r.get('fragments') or aol.fragment_signatures(tree):
                fragments[f] += 1
            ok, why = schema_ok(tree)
            if not ok:
                continue
            sig = r['signature']
            core = cores.setdefault(sig, {'card_id': card_id(sig), 'signature': sig, 'tree': r['tree'],
                                          'support': Counter(), 'examples': []})
            core['support'][src] += 1
            if len(core['examples']) < 25:
                core['examples'].append({'source': src, 'example_id': r['example_id'], 'split': r['split'],
                                         'text': r['text'][:200], 'annotation': str(r['raw'])[:300]})
        for sig, core in cores.items():
            tree = from_json(core['tree'])
            n = sum(core['support'].values())
            core['support'] = dict(core['support'])
            core['support_total'] = n
            core['admitted'] = n >= MIN_SUPPORT
            core['ops'] = sorted({s[0] for s in aol.subtrees(tree)} - {'ROWS', 'COL', 'STR', 'NUM'})
            core['refute_axes'] = refute_axes(tree)
            core['nei_axes'] = nei_axes(tree)
        comp = {k: (dict(v) if isinstance(v, Counter) else {kk: vv for kk, vv in v.items()}) for k, v in comp.items()}
        return cls(cores, fragments, comp)

    # ------------------------------------------------------------------ admission
    def bridge_support(self) -> int:
        return sum(self.compositions['bridge_verified'].values())

    def multi_table_support(self) -> int:
        return sum(self.compositions['multi_table_verified'].values())

    def admit(self, tree: tuple, topology: str) -> dict[str, Any]:
        sig = aol.signature(tree)
        out = {'signature': sig, 'topology': topology, 'card_id': card_id(sig) + '@' + topology}
        outer, inners = aol.split_bridges(tree)
        if inners:
            outer_sig = aol.signature(aol.fill_holes(outer))
            core = self.cores.get(outer_sig)
            inner_sigs = [aol.signature(i) for i in inners]
            out.update(core_card=card_id(outer_sig), outer_signature=outer_sig, inner_signatures=inner_sigs,
                       inner_support=[self.fragments.get(s, 0) for s in inner_sigs],
                       bridge_support=self.bridge_support())
            if not core or not core['admitted']:
                return dict(out, admitted=False, reason='bridge outer program has no admitted core card')
            if any(self.fragments.get(s, 0) < MIN_SUPPORT for s in inner_sigs):
                return dict(out, admitted=False, reason='bridge inner program not supported by source fragments')
            if self.bridge_support() < MIN_SUPPORT:
                return dict(out, admitted=False, reason='no verified bridge composition')
        else:
            core = self.cores.get(sig)
            out['core_card'] = card_id(sig)
            if not core:
                return dict(out, admitted=False, reason='signature not found in any verified source program')
            if not core['admitted']:
                return dict(out, admitted=False, reason=f"core support {core['support_total']} < {MIN_SUPPORT}")
        if topology not in NO_EXTRA_TOPOLOGY and topology != 'bridge' and self.multi_table_support() < MIN_SUPPORT:
            return dict(out, admitted=False, reason='no verified multi-table composition')
        out['core_support'] = core['support']
        return dict(out, admitted=True, reason='admitted')

    def admit_program(self, operators: list[dict], tables: list[dict]) -> dict[str, Any]:
        tree = aol.abstract_program(operators, tables)
        return self.admit(tree, aol.topology(operators, tables, tree))

    # ------------------------------------------------------------------ persistence
    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        with (out / 'core_cards.jsonl').open('w', encoding='utf-8') as fh:
            for sig in sorted(self.cores, key=lambda s: (-self.cores[s]['support_total'], s)):
                fh.write(json.dumps(self.cores[sig], ensure_ascii=False) + '\n')
        (out / 'fragments.json').write_text(json.dumps(dict(self.fragments.most_common()), ensure_ascii=False) + '\n')
        (out / 'compositions.json').write_text(json.dumps(self.compositions, ensure_ascii=False, indent=1) + '\n')
        (out / 'registry_params.json').write_text(json.dumps({'min_support': MIN_SUPPORT,
                                                              'program_sources': PROGRAM_SOURCES}, indent=1) + '\n')

    @classmethod
    def load(cls, path: Path) -> 'Registry':
        cores = {}
        with (path / 'core_cards.jsonl').open(encoding='utf-8') as fh:
            for line in fh:
                c = json.loads(line)
                cores[c['signature']] = c
        fragments = Counter(json.loads((path / 'fragments.json').read_text(encoding='utf-8')))
        compositions = json.loads((path / 'compositions.json').read_text(encoding='utf-8'))
        return cls(cores, fragments, compositions)


def _keep(bucket: list, r: dict, limit: int = 25) -> None:
    if len(bucket) < limit:
        bucket.append({'source': r['source'], 'example_id': r['example_id'], 'split': r['split'],
                       'text': r['text'][:200], 'qtype': r.get('qtype'), 'label': r.get('label')})
