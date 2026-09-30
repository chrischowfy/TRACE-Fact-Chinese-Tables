"""Structural hardness of a record (R9).  No model output is used anywhere here.

A record is *hard* when the table itself makes the decision close:
  near_miss_number   REFUTES whose stated number is within 15% of the executed value (or one unit
                     away for integer counts/ranks) — the number has to be computed, not estimated;
  runner_up_value    REFUTES whose stated extreme is the second-best value of the same set;
  runner_up_entity   REFUTES of an argmax/argmin claim that names the second-ranked entity;
  missing_table      NEI whose package lacks one table the program needs (the claim itself is an
                     ordinary decidable-looking claim; FEVEROUS NEI evidence sets are the source).
These flags feed selection constraints (tools/r9_targets.json); they are not labels.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dart_fact.executor import Execution, execute
from dart_fact.tables import parse_number, profile_table

ARGMAX = {'ST_SUPERLATIVE', 'CAT_ARGEXT', 'JF_ARGEXT', 'RJ_RANK_TOP'}
EXTREME_VALUE = {'CAT_EXTVAL', 'JF_EXTVAL', 'RJ_RANK_TOPVAL'}
SIMPLE = {'ST_COMPARE', 'XT_COMPARE', 'CP_DIRECTION', 'ST_SUPERLATIVE'}
NEAR = 0.15


def profiles(row):
    return {t['table_id']: profile_table(t) for t in row['tables']}


def env_before_compare(row):
    """Values bound by every operator except the final comparison."""
    ex = Execution(profiles(row))
    ex.run(row['program']['operators'][:-1])
    return ex.env


def ranking(row):
    """[(entity, value), ...] best first, for the set an argmax/extreme claim ranges over."""
    ops = row['program']['operators']
    mode = next((o.get('mode') for o in ops if o['op'] in {'ARGEXT', 'ARGEXT_OF', 'EXTVAL'} and o.get('mode')),
                row['program']['slots'].get('mode', 'max'))
    run = execute(ops, profiles(row))
    trace = {t['out']: t['value'] for t in run['trace']}
    if 'members' in trace and 'vals' in trace:
        pairs = list(zip(trace['members'], trace['vals']))
    else:
        op = next((o for o in ops if o['op'] == 'ARGEXT'), None)
        if op is None:
            return [], mode
        table = next(t for t in row['tables'] if t['table_id'] == op['table'])
        prof = profile_table(table)
        col = prof.column(op['col'])
        if prof.key is None or col is None:
            return [], mode
        pairs = [(r[prof.key.index], parse_number(r[col.index])) for r in table['rows']]
    pairs = [(str(e), v) for e, v in pairs if isinstance(v, (int, float))]
    pairs.sort(key=lambda p: -p[1] if mode == 'max' else p[1])
    return pairs, mode


def hardness(row):
    """The hardness kind of a hydrated record, or None."""
    q, sk = row['quality_flags'], row['program']['skeleton_id']
    if q.get('hard_variant'):
        return q['hard_variant']['kind']
    if row['label'] == 'NEI':
        return 'missing_table' if q.get('perturbation_axis') == 'missing_table' else None
    if row['label'] != 'REFUTES':
        return None
    last = row['program']['operators'][-1]
    try:
        if q.get('perturbation_axis') == 'number' and isinstance(last.get('right'), (int, float)) \
                and isinstance(last.get('left'), str):
            true = env_before_compare(row).get(last['left'].lstrip('$'))
            if isinstance(true, (int, float)) and true:
                gap = abs(last['right'] - true)
                if gap / abs(true) <= NEAR or (float(true).is_integer() and gap <= 1):
                    return 'near_miss_number'
        if sk in ARGMAX | EXTREME_VALUE:
            pairs, _ = ranking(row)
            values = [v for _, v in pairs]
            claimed = last.get('right')
            if sk in EXTREME_VALUE and len(set(values)) >= 2:
                second = sorted(set(values), key=values.index)[1]
                if isinstance(claimed, (int, float)) and abs(claimed - second) < 1e-9:
                    return 'runner_up_value'
            if sk in ARGMAX and len(pairs) >= 2 and pairs[1][1] != pairs[0][1]:
                if str(claimed) == pairs[1][0]:
                    return 'runner_up_entity'
    except Exception:  # noqa: BLE001 - a record whose program cannot be re-run is simply not flagged
        return None
    return None


if __name__ == '__main__':
    from collections import Counter
    release = Path(sys.argv[1])
    rows = [json.loads(l) for l in (release / 'claims.jsonl').open(encoding='utf-8')]
    print(json.dumps({'hardness': Counter(hardness(r) for r in rows),
                      'simple_share': round(sum(r['program']['skeleton_id'] in SIMPLE for r in rows) / len(rows), 3)},
                     ensure_ascii=False, default=str))
