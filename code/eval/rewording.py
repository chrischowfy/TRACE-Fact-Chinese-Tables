"""The final wording pass of the R13 release (tools/build_review_r13.py): the record of what changed, and the way back.

rewording.jsonl has one line per changed record: the claim before and after, the cell values that were restated
(``values``: previous -> current) and the input hash (eval/common.input_hash) of the previous and of the current
record.  ``earlier`` rebuilds the previous record from the current one, so a prediction that was requested on the
previous wording can still be checked against the exact input it answered.  Standard library only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

HASHED = ('claim', 'tables', 'program', 'evidence_cells', 'context_cells')     # eval/common.input_hash


def input_hash(record):
    payload = {k: record.get(k) for k in HASHED}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def swap(value, mapping):
    """Every string of a JSON value that equals a key of ``mapping`` is replaced; nothing is matched inside a string."""
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [swap(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: swap(v, mapping) for k, v in value.items()}
    return value


def earlier(record, row):
    """The record as it was before the pass: the previous claim, and the restated values written the previous way."""
    old = copy.deepcopy(record)
    old['claim'] = row['claim_before']
    back = {after: before for before, after in (row.get('values') or {}).items()}
    if back:
        for field in ('tables', 'program', 'evidence_cells', 'context_cells'):
            old[field] = swap(old[field], back)
    return old


EARLIER = {}      # id -> line of rewording.jsonl; set by the tools that read the predictions of an R13 evaluation


def answers(prediction, record, current_hash):
    """Is the stored prediction a response to this record?  Either to its current input, or - for a row marked
    earlier_wording (tools/carry_over_r13.py) - to the record as it was before the wording pass."""
    if prediction.get('input_sha256') == current_hash:
        return not prediction.get('earlier_wording')
    row = EARLIER.get(record['id'])
    return bool(row) and prediction.get('earlier_wording') is True and prediction.get('input_sha256') == row['input_sha256_before']


def claim_answered(prediction, record):
    """The claim text the prediction answered."""
    return EARLIER[record['id']]['claim_before'] if prediction.get('earlier_wording') else record['claim']


def load(release):
    """id -> line of rewording.jsonl; empty for a release without the file."""
    path = Path(release) / 'rewording.jsonl'
    if not path.is_file():
        return {}
    return {r['id']: r for r in map(json.loads, path.read_text(encoding='utf-8').splitlines())}


def earlier_hashes(release):
    """id -> input hash of the record before the pass."""
    return {i: r['input_sha256_before'] for i, r in load(release).items()}
