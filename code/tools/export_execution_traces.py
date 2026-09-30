"""Per-step execution traces of a release: replay every program on its own tables and keep each operator's output.

    runs/r7_build_env/bin/python -B tools/export_execution_traces.py --release runs/review_repair_r9_release --out traces.jsonl

One line per record, in release order:
  {"id", "label", "steps": [{"step", "op", "out", "value"}, ...], "missing": null | {"what", "detail"}}
step i is program.operators[i]. In an NEI record the step whose binding is absent, and every step that depends on it,
has the value {"missing": table|entity|column, "detail": ...}, and "missing" gives the first absence. Aborts unless
every replayed label equals the released label. The release files are not modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))


def plain(value):
    """JSON-ready value; integral floats become ints, containers are converted recursively."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (list, tuple, set)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return value


def traces(claims):
    from r7_runtime import enable
    enable()
    from r7_extensions import replay
    for c in claims:
        out = replay(c)
        ops = c['program']['operators']
        steps = out['trace']
        if out['label'] != c['label']:
            raise SystemExit(f"{c['id']}: replayed {out['label']} != released {c['label']}")
        if len(steps) > len(ops) or any(s['op'] != o['op'] or s['out'] != o.get('out') for s, o in zip(steps, ops)):
            raise SystemExit(f"{c['id']}: trace does not follow program.operators")
        yield {'id': c['id'], 'label': c['label'],
               'steps': [{'step': i, 'op': s['op'], 'out': s['out'], 'value': plain(s['value'])}
                         for i, s in enumerate(steps)],
               'missing': plain(out['missing'])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--release', type=Path, default=ROOT / 'runs/review_repair_r9_release')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    claims = [json.loads(x) for x in (a.release / 'claims.jsonl').read_text(encoding='utf-8').splitlines()]
    n = 0
    with a.out.open('w', encoding='utf-8') as fh:
        for row in traces(claims):
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            n += 1
    print(json.dumps({'records': n, 'out': str(a.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
