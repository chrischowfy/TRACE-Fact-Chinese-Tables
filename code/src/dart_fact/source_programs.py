"""Parse the reasoning sources named in the paper into AOL trees, with provenance checks.

Every parsed record is executed against its own source table (or checked against its own
annotation) and is kept as skeleton support only when it reproduces the dataset's gold result:

  TabFact       human-annotated LPA programs (bootstrap/bootstrap.json), label must be True
  WTQ           annotated lambda-DCS formulas (data/annotated-all.examples), denotation == target
  TAT-QA        arithmetic / comparison derivations over table cells, value == answer
  MultiModalQA  template pseudo-language of table sub-questions and their composition type;
                table part executed on MMQA_tables with the dataset's intermediate answers
  HybridQA      traced answer nodes (table cell -> linked passage) as bridge structure
  FEVEROUS      evidence sets (cell / sentence ids) as multi-table and insufficiency structure

Questions are turned into verification form ``CMP_EQ(program, answer)``; comparison questions
("which is higher: A or B") into ``CMP_ORD``.  No source instance becomes a target claim.
"""
from __future__ import annotations

import ast
import csv
import datetime as _dt
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from . import aol

NUM_RE = re.compile(r'-?\d[\d,]*\.?\d*|-?\.\d+')
MONTHS = {m: i + 1 for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                          'august', 'september', 'october', 'november', 'december'])}


def _record(source, example_id, split, text, raw, tree, verified, detail, **extra) -> dict[str, Any]:
    rec = {'source': source, 'example_id': example_id, 'split': split, 'text': text, 'raw': raw,
           'tree': tree, 'verified': bool(verified), 'verify_detail': detail}
    rec.update(extra)
    if tree is not None:
        rec['signature'] = aol.signature(tree)
        outer, inners = aol.split_bridges(tree)
        rec['bridge'] = bool(inners)
        rec['outer_signature'] = aol.signature(aol.fill_holes(outer)) if inners else rec['signature']
        rec['fragments'] = sorted(aol.fragment_signatures(tree))
    return rec


# =========================================================================== TabFact
def _tf_parse(s: str) -> Any:
    """``func{a; b}`` -> nested lists [func, args...]; literals stay strings."""
    pos = 0

    def parse():
        nonlocal pos
        m = re.match(r'\s*([a-z_]+)\{', s[pos:])
        if m:
            name = m.group(1)
            pos += m.end()
            args = []
            while True:
                args.append(parse())
                while pos < len(s) and s[pos] == ' ':
                    pos += 1
                if s[pos] == ';':
                    pos += 1
                    continue
                if s[pos] == '}':
                    pos += 1
                    return [name] + args
                raise ValueError('bad program at ' + s[pos:pos + 20])
        depth, start = 0, pos
        while pos < len(s):
            ch = s[pos]
            if ch == '{':
                depth += 1
            elif ch == '}':
                if depth == 0:
                    break
                depth -= 1
            elif ch == ';' and depth == 0:
                break
            pos += 1
        return s[start:pos].strip()

    out = parse()
    return out


def _num(text: Any) -> float | None:
    if isinstance(text, (int, float)):
        return float(text)
    t = str(text).strip().lower()
    d = _date(t)
    if d is not None and not re.fullmatch(r'-?[\d,.]+', t):
        return d
    m = NUM_RE.search(t.replace('−', '-'))
    if not m:
        return None
    try:
        return float(m.group().replace(',', ''))
    except ValueError:
        return None


def _date(t: str) -> float | None:
    m = re.search(r'(\d{1,2}) ([a-z]+) (\d{4})', t) or re.search(r'([a-z]+) (\d{1,2}) , (\d{4})', t)
    if not m:
        return None
    a, b, c = m.groups()
    day, mon = (a, b) if a.isdigit() else (b, a)
    if mon not in MONTHS:
        return None
    try:
        return float(_dt.date(int(c), MONTHS[mon], int(day)).toordinal())
    except ValueError:
        return None


def _tf_str_eq(a: str, b: str) -> bool:
    a, b = str(a).strip().lower(), str(b).strip().lower()
    return a == b or (len(b) > 0 and b in a) or (len(a) > 0 and a in b)


class _TabFactTable:
    def __init__(self, path: Path):
        with path.open(encoding='utf-8') as fh:
            rows = [line.rstrip('\n').split('#') for line in fh]
        self.header = [h.strip().lower() for h in rows[0]]
        self.rows = [r for r in rows[1:] if len(r) == len(self.header)]

    def col(self, name: str) -> int:
        name = name.strip().lower()
        if name in self.header:
            return self.header.index(name)
        raise KeyError(name)


def _tf_eval(node: Any, table: _TabFactTable) -> Any:
    if isinstance(node, str):
        return node if node != 'all_rows' else list(range(len(table.rows)))
    f, args = node[0], node[1:]
    ev = lambda x: _tf_eval(x, table)
    cell = lambda r, c: table.rows[r][table.col(c)]
    if f.startswith('filter_'):
        rows = ev(args[0])
        if f == 'filter_all':
            return rows
        c, v = args[1], args[2]
        kind = f[len('filter_'):]
        out = []
        for r in rows:
            x = cell(r, c)
            if kind in {'eq', 'str_eq'}:
                nv, nx = _num(v), _num(x)
                ok = (nv is not None and nx is not None and abs(nv - nx) < 1e-6 and re.fullmatch(r'[-\d,.]+', v.strip())) \
                    or _tf_str_eq(x, v)
            elif kind in {'not_eq', 'not_str_eq'}:
                ok = not _tf_str_eq(x, v)
            else:
                nv, nx = _num(v), _num(x)
                if nv is None or nx is None:
                    ok = False
                else:
                    ok = {'greater': nx > nv, 'less': nx < nv, 'greater_eq': nx >= nv, 'less_eq': nx <= nv}[kind]
            if ok:
                out.append(r)
        return out
    if f in {'hop', 'str_hop', 'num_hop'}:
        rows = ev(args[0])
        rows = rows if isinstance(rows, list) else [rows]
        if not rows:
            raise ValueError('hop on empty rows')
        return cell(rows[0], args[1])
    if f in {'argmax', 'argmin'}:
        rows = ev(args[0])
        vals = [(r, _num(cell(r, args[1]))) for r in rows]
        vals = [(r, v) for r, v in vals if v is not None]
        if not vals:
            raise ValueError('argext without numbers')
        best = (max if f == 'argmax' else min)(v for _, v in vals)
        return [r for r, v in vals if v == best][:1]
    if f in {'max', 'min', 'sum', 'avg'}:
        rows = ev(args[0])
        vals = [_num(cell(r, args[1])) for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            raise ValueError('aggregate without numbers')
        return {'max': max, 'min': min, 'sum': sum, 'avg': lambda v: sum(v) / len(v)}[f](vals)
    if f == 'count':
        return len(ev(args[0]))
    if f == 'only':
        return len(ev(args[0])) == 1
    if f == 'diff':
        a, b = _num(ev(args[0])), _num(ev(args[1]))
        return a - b
    if f in {'eq', 'round_eq', 'str_eq', 'not_eq', 'not_str_eq'}:
        a, b = ev(args[0]), ev(args[1])
        na, nb = _num(a), _num(b)
        if na is not None and nb is not None and (isinstance(a, (int, float)) or isinstance(b, (int, float))
                                                  or re.fullmatch(r'[-\d,.]+', str(b).strip())):
            digits = len(str(b).split('.')[1]) if '.' in str(b) else 0
            same = abs(na - nb) < 1e-6 or abs(round(na, digits) - nb) < 1e-6 or \
                (f == 'round_eq' and abs(na - nb) <= 0.05 * max(1.0, abs(nb)))
        else:
            same = _tf_str_eq(a, b)
        return same if f in {'eq', 'round_eq', 'str_eq'} else not same
    if f in {'greater', 'less'}:
        a, b = _num(ev(args[0])), _num(ev(args[1]))
        return a > b if f == 'greater' else a < b
    if f == 'and':
        return all(ev(a) for a in args)
    if f.startswith('all_') or f.startswith('most_'):
        rows = ev(args[0])
        kind = f.split('_', 1)[1]
        v = args[2]
        hits = []
        for r in rows:
            x = cell(r, args[1])
            if kind in {'eq', 'str_eq'}:
                hits.append(_tf_str_eq(x, v))
            elif kind in {'not_eq', 'not_str_eq'}:
                hits.append(not _tf_str_eq(x, v))
            else:
                nx, nv = _num(x), _num(v)
                hits.append(nx is not None and nv is not None and
                            {'greater': nx > nv, 'less': nx < nv, 'greater_eq': nx >= nv, 'less_eq': nx <= nv}[kind])
        return all(hits) if f.startswith('all_') else sum(hits) * 2 >= len(hits)
    raise ValueError('unsupported tabfact function ' + f)


def _tf_const(v: str) -> tuple:
    return ('NUM', float(v.replace(',', ''))) if re.fullmatch(r'-?[\d,]*\.?\d+', v.strip()) else ('STR', v.strip().lower())


def _tf_tree(node: Any) -> tuple:
    if isinstance(node, str):
        if node == 'all_rows':
            return ('ROWS',)
        return _tf_const(node)
    f, a = node[0], node[1:]
    col = lambda x: ('COL', x.strip().lower())
    if f.startswith('filter_'):
        base = _tf_tree(a[0])
        if f == 'filter_all':
            return base
        kind = f[len('filter_'):]
        cmp = 'EQ' if kind in {'eq', 'str_eq'} else 'NEQ' if kind.startswith('not') else 'ORD'
        return ('FILTER', base, col(a[1]), cmp, _tf_const(a[2]))
    if f in {'hop', 'str_hop', 'num_hop'}:
        return ('HOP', _tf_tree(a[0]), col(a[1]))
    if f in {'argmax', 'argmin'}:
        return ('ARGEXT', _tf_tree(a[0]), col(a[1]))
    if f in {'max', 'min'}:
        return ('EXTVAL', _tf_tree(a[0]), col(a[1]))
    if f in {'sum', 'avg'}:
        return (f.upper(), _tf_tree(a[0]), col(a[1]))
    if f == 'count':
        return ('COUNT', _tf_tree(a[0]))
    if f == 'only':
        return ('ONLY', _tf_tree(a[0]))
    if f == 'diff':
        return ('SUB', _tf_tree(a[0]), _tf_tree(a[1]))
    if f in {'eq', 'round_eq', 'str_eq'}:
        return ('CMP_EQ', _tf_tree(a[0]), _tf_tree(a[1]))
    if f in {'not_eq', 'not_str_eq'}:
        return ('CMP_NEQ', _tf_tree(a[0]), _tf_tree(a[1]))
    if f in {'greater', 'less'}:
        return ('CMP_ORD', _tf_tree(a[0]), _tf_tree(a[1]))
    if f == 'and':
        return ('AND',) + tuple(_tf_tree(x) for x in a)
    if f.startswith('all_') or f.startswith('most_'):
        kind = f.split('_', 1)[1]
        cmp = 'EQ' if kind in {'eq', 'str_eq'} else 'NEQ' if kind.startswith('not') else 'ORD'
        return ('ALL' if f.startswith('all_') else 'MOST', _tf_tree(a[0]), col(a[1]), cmp, _tf_const(a[2]))
    raise ValueError('unsupported tabfact function ' + f)


def tabfact(root: Path) -> Iterator[dict]:
    boot = json.loads((root / 'bootstrap/bootstrap.json').read_text(encoding='utf-8'))
    for table_name in sorted(boot):
        table = _TabFactTable(root / 'data/all_csv' / table_name)
        for i, entry in enumerate(boot[table_name]):
            statement, program = entry[0], entry[4]
            body, _, expected = program.rpartition('=')
            eid = f'{table_name}#{i}'
            try:
                parsed = _tf_parse(body)
                tree = aol.rewrite(_tf_tree(parsed))
            except Exception as exc:          # noqa: BLE001 - unparseable annotations are reported, not fixed
                yield _record('tabfact', eid, 'bootstrap', statement, program, None, False, f'parse: {exc}')
                continue
            try:
                result = _tf_eval(parsed, table)
                ok = bool(result) == (expected.strip() == 'True')
                detail = 'executes to annotated label' if ok else f'executes to {result}, annotated {expected}'
            except Exception as exc:          # noqa: BLE001
                ok, detail = False, f'execution: {exc}'
            yield _record('tabfact', eid, 'bootstrap', statement, program, tree, ok, detail,
                          annotation='human semantic parse (TabFact bootstrap)', source_table=table_name)


# =========================================================================== WTQ
def _sexp(s: str) -> Any:
    tokens = re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+', s)
    pos = 0

    def walk():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == '(':
            out = []
            while tokens[pos] != ')':
                out.append(walk())
            pos += 1
            return out
        return tok

    return walk()


def _wtq_examples(path: Path) -> Iterator[dict]:
    text = path.read_text(encoding='utf-8')
    for block in re.split(r'\n#+ ex \d+ #+\n', text)[1:]:
        ex = _sexp(block)
        fields = {x[0]: x[1:] for x in ex[1:] if isinstance(x, list)}
        yield {'id': fields['id'][0], 'utterance': fields['utterance'][0].strip('"'),
               'table': fields['context'][0][2], 'target': [d[1].strip('"') for d in fields['targetValue'][0][1:]],
               'formula': fields.get('targetFormula', [None])[0]}


class _WTQTable:
    def __init__(self, path: Path):
        with path.open(encoding='utf-8') as fh:
            rows = list(csv.DictReader(fh, delimiter='\t', quoting=csv.QUOTE_NONE))
        self.cols: dict[str, int] = {}
        self.cells: dict[tuple[int, int], dict] = {}
        n = 0
        for r in rows:
            ri, ci = int(r['row']), int(r['col'])
            if ri == -1:
                self.cols[r['id'].replace('fb:row.row.', '')] = ci
                continue
            self.cells[(ri, ci)] = r
            n = max(n, ri + 1)
        self.n = n

    def cell_id(self, r, c):
        cell = self.cells.get((r, c))
        return cell['id'].replace('fb:cell.', '') if cell else None

    def number(self, r, c, prop='number'):
        cell = self.cells.get((r, c))
        if not cell:
            return None
        v = cell.get(prop) or ''
        if prop == 'date':
            m = re.fullmatch(r'(-?\d+|xx+)-(\d+|xx)-(\d+|xx)', v)
            if not m or 'x' in m.group(1):
                return None
            y, mo, d = m.groups()
            return float(int(y) * 10000 + (int(mo) if 'x' not in mo else 0) * 100 + (int(d) if 'x' not in d else 0))
        try:
            return float(v.split('|')[0]) if v else None
        except ValueError:
            return None


class _Unsupported(Exception):
    pass


def _wtq_eval(f: Any, t: _WTQTable) -> Any:
    """Denotation: ('rows', set) | ('cells', list of (r, c)) | ('vals', list) ."""
    if isinstance(f, str):
        if f.startswith('c.'):
            return ('ent', f[2:])
        raise _Unsupported(f)
    op = f[0]
    if op == '@type' and f[1] == '@row':
        return ('rows', set(range(t.n)))
    if isinstance(op, str) and op.startswith('r.'):              # rows whose column value is in the arg
        c = t.cols.get(op[2:])
        if c is None:
            raise _Unsupported('column ' + op)
        arg = f[1]
        if isinstance(arg, list) and arg[0] in {'@p.num', '@p.date', '@p.num2'}:
            prop = {'@p.num': 'number', '@p.date': 'date', '@p.num2': 'num2'}[arg[0]]
            cond = arg[1]
            if isinstance(cond, list) and cond[0] in {'>', '<', '>=', '<='}:
                v = _wtq_scalar(cond[1])
                cmp = {'>': lambda a: a > v, '<': lambda a: a < v, '>=': lambda a: a >= v, '<=': lambda a: a <= v}[cond[0]]
            else:
                v = _wtq_scalar(cond)
                cmp = lambda a: abs(a - v) < 1e-9
            return ('rows', {r for r in range(t.n) if (x := t.number(r, c, prop)) is not None and cmp(x)})
        if isinstance(arg, list) and arg[0] == '@p.part':
            raise _Unsupported('@p.part')
        den = _wtq_eval(arg, t)
        if den[0] == 'ent':
            return ('rows', {r for r in range(t.n) if t.cell_id(r, c) == den[1]})
        if den[0] == 'ents':
            return ('rows', {r for r in range(t.n) if t.cell_id(r, c) in den[1]})
        if den[0] == 'not':
            return ('rows', {r for r in range(t.n) if t.cell_id(r, c) not in den[1]})
        if den[0] == 'cells':
            ids = {t.cell_id(*rc) for rc in den[1]}
            return ('rows', {r for r in range(t.n) if t.cell_id(r, c) in ids})
        raise _Unsupported('r. arg')
    if isinstance(op, str) and op.startswith('!r.'):
        c = t.cols.get(op[3:])
        if c is None:
            raise _Unsupported('column ' + op)
        den = _wtq_eval(f[1], t)
        if den[0] != 'rows':
            raise _Unsupported('!r. on non-rows')
        return ('cells', [(r, c) for r in sorted(den[1]) if (r, c) in t.cells])
    if op in {'@!p.num', '@!p.date', '@!p.num2'}:
        prop = {'@!p.num': 'number', '@!p.date': 'date', '@!p.num2': 'num2'}[op]
        den = _wtq_eval(f[1], t)
        if den[0] != 'cells':
            raise _Unsupported('p on non-cells')
        vals = [t.number(r, c, prop) for r, c in den[1]]
        return ('vals', [v for v in vals if v is not None])
    if op == 'or':
        a, b = _wtq_eval(f[1], t), _wtq_eval(f[2], t)
        if a[0] == 'ent' and b[0] == 'ent':
            return ('ents', {a[1], b[1]})
        if a[0] == b[0] == 'rows':
            return ('rows', a[1] | b[1])
        raise _Unsupported('or')
    if op == 'and':
        a, b = _wtq_eval(f[1], t), _wtq_eval(f[2], t)
        if a[0] == b[0] == 'rows':
            return ('rows', a[1] & b[1])
        if a[0] == 'not' and b[0] == 'cells':
            return ('cells', [rc for rc in b[1] if t.cell_id(*rc) not in a[1]])
        if b[0] == 'not' and a[0] == 'cells':
            return ('cells', [rc for rc in a[1] if t.cell_id(*rc) not in b[1]])
        raise _Unsupported('and')
    if op == '!=':
        den = _wtq_eval(f[1], t)
        if den[0] == 'ent':
            return ('not', {den[1]})
        raise _Unsupported('!=')
    if op in {'count', 'sum', 'avg', 'max', 'min'}:
        den = _wtq_eval(f[1], t)
        if op == 'count':
            if den[0] == 'rows':
                return ('vals', [float(len(den[1]))])
            if den[0] == 'cells':
                return ('vals', [float(len({t.cell_id(*rc) for rc in den[1]}))])
            raise _Unsupported('count')
        if den[0] != 'vals' or not den[1]:
            raise _Unsupported(op)
        v = den[1]
        return ('vals', [{'sum': sum, 'avg': lambda x: sum(x) / len(x), 'max': max, 'min': min}[op](v)])
    if op in {'-', '+'}:
        a, b = _wtq_eval(f[1], t), _wtq_eval(f[2], t)
        if a[0] != 'vals' or b[0] != 'vals' or len(a[1]) != 1 or len(b[1]) != 1:
            raise _Unsupported('arith')
        return ('vals', [a[1][0] - b[1][0] if op == '-' else a[1][0] + b[1][0]])
    if op in {'argmax', 'argmin'}:
        base = _wtq_eval(f[3], t)
        key = f[4]
        pick = max if op == 'argmax' else min
        if key == '@index':
            if base[0] != 'rows' or not base[1]:
                raise _Unsupported('argmax index')
            return ('rows', {pick(base[1])})
        # (reverse (lambda x (@!p.num (!r.col (var x)))))  or  count of rows per value
        if isinstance(key, list) and key[0] == 'reverse' and key[1][0] == 'lambda':
            body = key[1][2]
            if base[0] == 'rows' and body[0] in {'@!p.num', '@!p.date'} and body[1][0].startswith('!r.'):
                c = t.cols.get(body[1][0][3:])
                prop = 'number' if body[0] == '@!p.num' else 'date'
                vals = [(r, t.number(r, c, prop)) for r in base[1]]
                vals = [(r, v) for r, v in vals if v is not None]
                if not vals:
                    raise _Unsupported('argmax without numbers')
                best = pick(v for _, v in vals)
                return ('rows', {r for r, v in vals if v == best})
            if base[0] == 'cells' and body[0] == 'count' and body[1][0].startswith('r.'):
                c = t.cols.get(body[1][0][2:])
                counts = defaultdict(int)
                for r in range(t.n):
                    counts[t.cell_id(r, c)] += 1
                ids = {t.cell_id(*rc) for rc in base[1]}
                best = pick(counts[i] for i in ids)
                return ('ents_named', {i for i in ids if counts[i] == best})
        raise _Unsupported('argmax form')
    if op == '@!next':
        den = _wtq_eval(f[1], t)
        return ('rows', {r + 1 for r in den[1] if r + 1 < t.n})
    raise _Unsupported(str(op))


def _wtq_scalar(x: Any) -> float:
    if isinstance(x, list) and x[0] == 'date':
        y, m, d = (int(v) for v in x[1:4])
        return float(y * 10000 + max(m, 0) * 100 + max(d, 0))
    if isinstance(x, list) and x[0] == 'number':
        return float(x[1])
    return float(x)


def _wtq_tree(f: Any) -> tuple:
    if isinstance(f, str):
        if f.startswith('c.'):
            return ('STR', f[2:])
        raise _Unsupported(f)
    op = f[0]
    if op == '@type' and f[1] == '@row':
        return ('ROWS',)
    if isinstance(op, str) and op.startswith('r.'):
        c = ('COL', op[2:])
        arg = f[1]
        if isinstance(arg, list) and arg[0] in {'@p.num', '@p.date', '@p.num2'}:
            cond = arg[1]
            if isinstance(cond, list) and cond[0] in {'>', '<', '>=', '<='}:
                return ('FILTER', ('ROWS',), c, 'ORD', ('NUM', _wtq_scalar(cond[1])))
            return ('FILTER', ('ROWS',), c, 'EQ', ('NUM', _wtq_scalar(cond)))
        if isinstance(arg, list) and arg[0] == '!=':
            return ('FILTER', ('ROWS',), c, 'NEQ', _wtq_tree(arg[1]))
        if isinstance(arg, list) and arg[0] == 'or':
            return ('FILTER', ('ROWS',), c, 'IN', ('STR', '|'.join(sorted(str(x) for x in arg[1:]))))
        value = _wtq_tree(arg)
        if value[0] == 'HOP':      # rows whose value equals a value read elsewhere
            return ('FILTER', ('ROWS',), c, 'EQ', value)
        return ('FILTER', ('ROWS',), c, 'EQ', value)
    if isinstance(op, str) and op.startswith('!r.'):
        return ('HOP', _wtq_tree(f[1]), ('COL', op[3:]))
    if op in {'@!p.num', '@!p.date', '@!p.num2'}:
        return _wtq_tree(f[1])
    if op == 'and':
        a, b = _wtq_tree(f[1]), _wtq_tree(f[2])
        if a[0] == 'FILTER' and a[1] == ('ROWS',) and b[0] == 'FILTER':
            return ('FILTER', b, a[2], a[3], a[4])
        if b[0] == 'FILTER' and b[1] == ('ROWS',) and a[0] == 'FILTER':
            return ('FILTER', a, b[2], b[3], b[4])
        return ('INTERSECT', a, b)
    if op == 'or':
        return ('UNION', _wtq_tree(f[1]), _wtq_tree(f[2]))
    if op == '!=':
        return ('EXCEPT', _wtq_tree(f[1]))
    if op == 'count':
        inner = _wtq_tree(f[1])
        return ('COUNT', inner)
    if op in {'sum', 'avg', 'max', 'min'}:
        inner = _wtq_tree(f[1])
        if inner[0] == 'HOP':
            return ({'sum': 'SUM', 'avg': 'AVG', 'max': 'EXTVAL', 'min': 'EXTVAL'}[op], inner[1], inner[2])
        raise _Unsupported(op + ' over non-column')
    if op in {'-', '+'}:
        return ('SUB' if op == '-' else 'ADD', _wtq_tree(f[1]), _wtq_tree(f[2]))
    if op in {'argmax', 'argmin'}:
        base, key = _wtq_tree(f[3]), f[4]
        if key == '@index':
            return ('ORDER', base)
        if isinstance(key, list) and key[0] == 'reverse' and key[1][0] == 'lambda':
            body = key[1][2]
            if body[0] in {'@!p.num', '@!p.date'} and body[1][0].startswith('!r.'):
                return ('ARGEXT', base, ('COL', body[1][0][3:]))
            if body[0] == 'count' and body[1][0].startswith('r.') and base[0] == 'HOP':
                return ('MODE', base[1], base[2])
        raise _Unsupported('argmax form')
    if op in {'@!next', '@next'}:
        return ('NEXT', _wtq_tree(f[1]))
    raise _Unsupported(str(op))


def _wtq_answer_matches(den: Any, target: list[str], t: _WTQTable) -> bool:
    def norm(x):
        x = str(x).strip().lower().replace('_', ' ')
        return re.sub(r'\s+', ' ', x)

    def as_num(x):
        try:
            return float(str(x).replace(',', ''))
        except ValueError:
            return None

    if den[0] == 'vals':
        got = den[1]
        if len(got) != len(target):
            return False
        return all(as_num(g) is not None and as_num(x) is not None and abs(float(g) - as_num(x)) < 1e-6
                   for g, x in zip(sorted(got), sorted(target, key=lambda y: as_num(y) or 0)))
    if den[0] == 'cells':
        got = sorted({norm(t.cells[rc]['content']) for rc in den[1]})
        return got == sorted({norm(x) for x in target})
    if den[0] in {'ent', 'ents', 'ents_named'}:
        ids = {den[1]} if den[0] == 'ent' else den[1]
        contents = sorted({norm(c['content']) for c in t.cells.values() if c['id'].replace('fb:cell.', '') in ids})
        return contents == sorted({norm(x) for x in target})
    return False


def _wtq_answer_const(target: list[str]) -> tuple:
    if len(target) == 1:
        try:
            return ('NUM', float(target[0].replace(',', '')))
        except ValueError:
            return ('STR', target[0].lower())
    return ('STR', '|'.join(sorted(x.lower() for x in target)))


def wtq(root: Path) -> Iterator[dict]:
    for ex in _wtq_examples(root / 'data/annotated-all.examples'):
        if ex['formula'] is None:
            continue
        raw = json.dumps(ex['formula'])
        try:
            body = _wtq_tree(ex['formula'])
            tree = aol.rewrite(('CMP_EQ', body, _wtq_answer_const(ex['target'])))
        except (_Unsupported, IndexError, ValueError, TypeError) as exc:
            yield _record('wtq', ex['id'], 'annotated-all', ex['utterance'], raw, None, False, f'parse: {exc}')
            continue
        table = _WTQTable(root / (ex['table'].replace('-csv/', '-tagged/').replace('csv/', 'tagged/').replace('.csv', '.tagged')))
        try:
            den = _wtq_eval(ex['formula'], table)
            ok = _wtq_answer_matches(den, ex['target'], table)
            detail = 'denotation equals target' if ok else f'denotation {str(den)[:80]} != {ex["target"]}'
        except (_Unsupported, KeyError, ValueError, TypeError) as exc:
            ok, detail = False, f'execution: {exc}'
        yield _record('wtq', ex['id'], 'annotated-all', ex['utterance'], raw, tree, ok, detail,
                      annotation='annotated lambda-DCS logical form', source_table=ex['table'])


# =========================================================================== TAT-QA
def _tat_cell_value(text: str) -> float | None:
    t = text.strip().replace('$', '').replace('%', '').replace(',', '').replace('€', '').replace('£', '')
    neg = t.startswith('(') and t.endswith(')')
    t = t.strip('()').strip()
    if not re.fullmatch(r'-?\d+(\.\d+)?', t):
        return None
    v = float(t)
    return -v if neg else v


def _tat_parse_expr(derivation: str) -> ast.AST:
    expr = derivation.replace('[', '(').replace(']', ')').replace('×', '*').replace('x', '*').replace('÷', '/')
    expr = expr.replace('$', '').replace('%', '').replace('€', '').replace('£', '')
    expr = re.sub(r'(?<=\d),(?=\d{3})', '', expr)
    if not re.fullmatch(r'[\d\s.+\-*/()]+', expr):
        raise ValueError('non-arithmetic derivation')
    return ast.parse(expr.strip(), mode='eval').body


def _tat_locate(table: list[list[str]]) -> dict[float, list[tuple[int, int]]]:
    where = defaultdict(list)
    for r, row in enumerate(table):
        for c, cell in enumerate(row):
            v = _tat_cell_value(cell)
            if v is not None and c > 0:
                where[abs(v)].append((r, c))
    return where


def _tat_headers(table: list[list[str]]) -> tuple[list[str], int]:
    """Column labels from the leading rows that carry no numbers in value columns."""
    header_rows = 0
    for row in table:
        if any(_tat_cell_value(c) is not None for c in row[1:]) and not all(
                re.fullmatch(r'(19|20)\d\d', c.strip()) for c in row[1:] if c.strip()):
            break
        header_rows += 1
    labels = [' | '.join(table[r][c].strip() for r in range(header_rows) if c < len(table[r]) and table[r][c].strip())
              for c in range(max(len(r) for r in table))]
    return labels, header_rows


def _tat_tree_and_value(node: ast.AST, table, where, labels, used) -> tuple[tuple, float]:
    if isinstance(node, ast.Constant):
        v = float(node.value)
        spots = where.get(abs(v), [])
        if spots:
            r, c = _choose(spots, used)
            used.append((r, c))
            cell = _tat_cell_value(table[r][c])
            row_label = table[r][0].strip()
            return ('HOP', ('FILTER', ('ROWS',), ('COL', '__row_label__'), 'EQ', ('STR', row_label.lower())),
                    ('COL', labels[c].lower() or f'col{c}')), cell
        return ('NUM', v), v
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        t, v = _tat_tree_and_value(node.operand, table, where, labels, used)
        return t, -v
    if isinstance(node, ast.BinOp):
        lt, lv = _tat_tree_and_value(node.left, table, where, labels, used)
        rt, rv = _tat_tree_and_value(node.right, table, where, labels, used)
        if isinstance(node.op, ast.Sub):
            return ('SUB', lt, rt), lv - rv
        if isinstance(node.op, ast.Add):
            return ('ADD', lt, rt), lv + rv
        if isinstance(node.op, ast.Div):
            return ('DIV', lt, rt), lv / rv
        if isinstance(node.op, ast.Mult):
            return ('MUL', lt, rt), lv * rv
    raise ValueError('unsupported expression')


def _choose(spots, used):
    """Prefer a cell sharing a row or column with cells already used by the derivation."""
    for r, c in spots:
        if any(r == ur or c == uc for ur, uc in used):
            return r, c
    return spots[0]


def _answer_number(answer: Any) -> float | None:
    if isinstance(answer, list):
        answer = answer[0] if len(answer) == 1 else None
    if answer is None:
        return None
    try:
        return float(str(answer).replace(',', ''))
    except ValueError:
        return None


def tatqa(root: Path) -> Iterator[dict]:
    for split in ['train', 'dev']:
        docs = json.loads((root / f'dataset_raw/tatqa_dataset_{split}.json').read_text(encoding='utf-8'))
        for doc in docs:
            table = doc['table']['table']
            if not table or len(table) < 2:
                continue
            where = _tat_locate(table)
            labels, _ = _tat_headers(table)
            for q in doc['questions']:
                if q['answer_from'] != 'table':
                    continue
                deriv = (q.get('derivation') or '').strip()
                if q['answer_type'] == 'arithmetic':
                    try:
                        used: list = []
                        tree, value = _tat_tree_and_value(_tat_parse_expr(deriv), table, where, labels, used)
                    except (ValueError, SyntaxError, ZeroDivisionError, KeyError, IndexError) as exc:
                        yield _record('tatqa', q['uid'], split, q['question'], deriv, None, False, f'parse: {exc}')
                        continue
                    gold = _answer_number(q['answer'])
                    located = sum(1 for s in aol.subtrees(tree) if s[0] == 'HOP')
                    ok = gold is not None and located >= 2 and (
                        abs(value - gold) <= 0.01 + 0.005 * abs(gold) or abs(value * 100 - gold) <= 0.01 + 0.005 * abs(gold))
                    tree = aol.rewrite(('CMP_EQ', tree, ('NUM', gold if gold is not None else 0.0)))
                    detail = ('derivation over located table cells equals answer' if ok else
                              f'value {value} vs answer {q["answer"]}, located cells {located}')
                    yield _record('tatqa', q['uid'], split, q['question'], deriv, tree, ok, detail,
                                  annotation='arithmetic derivation', scale=q.get('scale'))
                elif q.get('req_comparison') and re.fullmatch(r'\$?[\d,.()]+\s*[<>]\s*\$?[\d,.()]+', deriv):
                    a, op, b = re.match(r'(.+?)\s*([<>])\s*(.+)', deriv).groups()
                    va, vb = _tat_cell_value(a), _tat_cell_value(b)
                    if va is None or vb is None or not where.get(abs(va)) or not where.get(abs(vb)):
                        yield _record('tatqa', q['uid'], split, q['question'], deriv, None, False, 'comparison operands not in table')
                        continue
                    used: list = []
                    ra, ca = _choose(where[abs(va)], used); used.append((ra, ca))
                    rb, cb = _choose(where[abs(vb)], used)
                    look = lambda r, c: ('HOP', ('FILTER', ('ROWS',), ('COL', '__row_label__'), 'EQ',
                                                 ('STR', table[r][0].strip().lower())), ('COL', labels[c].lower() or f'col{c}'))
                    tree = ('CMP_ORD', look(ra, ca), look(rb, cb))
                    holds = (va > vb) if op == '>' else (va < vb)
                    answers = [str(x).strip().lower() for x in (q['answer'] if isinstance(q['answer'], list) else [q['answer']])]
                    winner = labels[ca].lower() if holds else labels[cb].lower()
                    alt = table[ra][0].strip().lower() if holds else table[rb][0].strip().lower()
                    ok = any(ans and (ans in winner or ans in alt) for ans in answers)
                    yield _record('tatqa', q['uid'], split, q['question'], deriv, aol.rewrite(tree), ok,
                                  'comparison consistent with answer' if ok else f'answer {answers} not the compared winner',
                                  annotation='comparison derivation')


# =========================================================================== MultiModalQA
_B = r'\[([^\[\]]*)\]'
MMQA_TABLE_TEMPLATES = [
    ('lookup', re.compile(r'^In \[.*\] of \[.*\], what was the ' + _B + r'\(s\) when the ' + _B + r' was ' + _B + r'\s*$')),
    ('lookup2', re.compile(r'^What was the ' + _B + r'\(s\), in \[.*\] of \[.*\], when the ' + _B + r' was ' + _B +
                           r' and the ' + _B + r' was ' + _B + r'\s*$')),
    ('which2', re.compile(r'^In which ' + _B + r's, in \[.*\] of \[.*\], the ' + _B + r' was ' + _B +
                          r' and the ' + _B + r' was ' + _B + r'\s*$')),
    ('verify', re.compile(r'^In \[.*\] of \[.*\], was ' + _B + r'\s+the ' + _B + r'\(s\) when the ' + _B + r' was ' + _B + r'\?\s*$')),
    ('compare', re.compile(r'^In \[.*\] of \[.*\], which ' + _B + r' has (higher|lower|most recent|earlier) ' + _B +
                           r' : (\{\{?.*?\}?\}|\[[^\[\]]*\]) or (\{\{?.*?\}?\}|\[[^\[\]]*\])\?\s*$')),
    ('ext', re.compile(r'^In \[.*\] of \[.*\], what was the (MOST RECENT|EARLIEST) ' + _B + r'\(s\) when the ' + _B + r' was ' + _B + r'\s*$')),
    ('avg', re.compile(r'^In \[.*\] of \[.*\], what was the AVERAGE ' + _B + r'\(s\) when the ' + _B + r' was ' + _B + r'\s*$')),
    ('ext2', re.compile(r'^what was the (MOST RECENT|EARLIEST) ' + _B + r'\(s\), in \[.*\] of \[.*\], when the ' + _B +
                        r' was ' + _B + r' and the ' + _B + r' was ' + _B + r'\s*$')),
    ('of_sub', re.compile(r'^In \[.*\] of \[.*\], what was the ' + _B + r'\(s\) of \{the ' + _B + r' when the ' + _B +
                          r' was ' + _B + r'\},\s*$')),
    ('compose_lookup', re.compile(r'^What was the ' + _B + r'\(s\), in \[.*\] of \[.*\], when the ' + _B + r' was \{.*\}\s*$')),
    ('intersect', re.compile(r'^\{.*\} and was the ' + _B + r'\(s\), in \[.*\] of \[.*\], when the ' + _B + r' was ' + _B + r'\s*$')),
    ('text_of_table', re.compile(r'^.*\{the ' + _B + r'\(s\) in the \[.*\] of \[.*\] when the ' + _B + r' was ' + _B + r'\}.*$')),
]


def _mm_norm(x: str) -> str:
    return re.sub(r'\s+', ' ', str(x)).strip().lower()


def _mm_table(t: dict) -> tuple[list[str], list[list[str]]]:
    header = [_mm_norm(h['column_name']) for h in t['table']['header']]
    rows = [[_mm_norm(c['text']) for c in r] for r in t['table']['table_rows']]
    return header, rows


def _mm_select(header, rows, conds):
    out = []
    for r in rows:
        ok = True
        for c, v in conds:
            c = _mm_norm(c)
            if c not in header:
                raise KeyError(c)
            if r[header.index(c)] != _mm_norm(v):
                ok = False
        if ok:
            out.append(r)
    return out


def _mm_val(x: str) -> float | None:
    m = re.search(r'\d{4}', x) if re.search(r'[a-z]', x) else None
    if m:
        return float(m.group())
    return _num(x)


def _mm_parse(qtype: str, pseudo: str) -> tuple[str, tuple, list] | None:
    """(template, tree, check) where check lets the table part be executed."""
    lookup = lambda key_col, key, col: ('HOP', ('FILTER', ('ROWS',), ('COL', _mm_norm(key_col)), 'EQ', key), ('COL', _mm_norm(col)))
    s = lambda v: ('STR', _mm_norm(v))
    for name, rx in MMQA_TABLE_TEMPLATES:
        m = rx.match(pseudo.strip())
        if not m:
            continue
        g = m.groups()
        if name == 'lookup':
            return name, lookup(g[1], s(g[2]), g[0]), ['select', g[0], [(g[1], g[2])]]
        if name in {'lookup2', 'which2'}:
            tree = ('HOP', ('FILTER', ('FILTER', ('ROWS',), ('COL', _mm_norm(g[1])), 'EQ', s(g[2])),
                            ('COL', _mm_norm(g[3])), 'EQ', s(g[4])), ('COL', _mm_norm(g[0])))
            return name, tree, ['select', g[0], [(g[1], g[2]), (g[3], g[4])]]
        if name == 'verify':
            return name, ('CMP_EQ', lookup(g[2], s(g[3]), g[1]), s(g[0])), ['verify', g[1], [(g[2], g[3])], g[0]]
        if name == 'compare':
            key, how, metric, a, b = g
            side = lambda x: ('SUBQ', 'q') if x.startswith('{') else s(x.strip('[]'))
            tree = ('CMP_ORD', lookup(key, side(a), metric), lookup(key, side(b), metric))
            return name, tree, ['compare', key, metric, how, a, b]
        if name == 'ext':
            return name, ('EXTVAL', ('FILTER', ('ROWS',), ('COL', _mm_norm(g[2])), 'EQ', s(g[3])), ('COL', _mm_norm(g[1]))), \
                ['ext', g[0], g[1], [(g[2], g[3])]]
        if name == 'avg':
            return name, ('AVG', ('FILTER', ('ROWS',), ('COL', _mm_norm(g[1])), 'EQ', s(g[2])), ('COL', _mm_norm(g[0]))), \
                ['avg', g[0], [(g[1], g[2])]]
        if name == 'ext2':
            tree = ('EXTVAL', ('FILTER', ('FILTER', ('ROWS',), ('COL', _mm_norm(g[2])), 'EQ', s(g[3])),
                               ('COL', _mm_norm(g[4])), 'EQ', s(g[5])), ('COL', _mm_norm(g[1])))
            return name, tree, ['ext', g[0], g[1], [(g[2], g[3]), (g[4], g[5])]]
        if name == 'of_sub':
            inner = lookup(g[2], s(g[3]), g[1])        # the [Season] when the [Club] was [X]
            tree = ('HOP', ('FILTER', ('ROWS',), ('COL', _mm_norm(g[1])), 'EQ', inner), ('COL', _mm_norm(g[0])))
            return name, tree, ['of_sub', g[0], g[1], g[2], g[3]]
        if name == 'compose_lookup':
            return name, lookup(g[1], ('SUBQ', 'q'), g[0]), ['select_sub', g[0], g[1]]
        if name == 'intersect':
            return name, ('INTERSECT', ('SUBQ', 'q'), lookup(g[1], s(g[2]), g[0])), ['select', g[0], [(g[1], g[2])]]
        if name == 'text_of_table':
            return name, ('TEXTQ', lookup(g[1], s(g[2]), g[0])), ['select_inner', g[0], [(g[1], g[2])]]
    return None


def _mm_check(check: list, header, rows, answers: list[str], inter: list) -> bool:
    ans = {_mm_norm(a) for a in answers}
    kind = check[0]
    if kind == 'select':
        got = {r[header.index(_mm_norm(check[1]))] for r in _mm_select(header, rows, check[2])}
        return bool(got) and (ans <= got or got <= ans or bool(ans & got))
    if kind == 'select_inner':          # inner TableQ of Compose(TextQ, TableQ): check its intermediate answer
        got = {r[header.index(_mm_norm(check[1]))] for r in _mm_select(header, rows, check[2])}
        mids = {_mm_norm(a['answer']) for grp in inter for a in grp}
        return bool(got & mids)
    if kind == 'verify':
        got = {r[header.index(_mm_norm(check[1]))] for r in _mm_select(header, rows, check[2])}
        truth = _mm_norm(check[3]) in got
        return ('yes' in ans) == truth
    if kind == 'select_sub':            # Compose(TableQ, TextQ): outer filter value = text answer
        mids = [_mm_norm(a['answer']) for grp in inter for a in grp if a.get('modality') == 'text']
        for mid in mids:
            got = {r[header.index(_mm_norm(check[1]))] for r in _mm_select(header, rows, [(check[2], mid)])}
            if got and (ans & got):
                return True
        return False
    if kind == 'of_sub':
        _, col, via, key_col, key = check
        vals = {r[header.index(_mm_norm(via))] for r in _mm_select(header, rows, [(key_col, key)])}
        got = {r[header.index(_mm_norm(col))] for r in rows if r[header.index(_mm_norm(via))] in vals}
        return bool(ans & got)
    if kind in {'ext', 'avg'}:
        col = check[2] if kind == 'ext' else check[1]
        conds = check[3] if kind == 'ext' else check[2]
        cells = [r[header.index(_mm_norm(col))] for r in _mm_select(header, rows, conds)]
        vals = [(c, _mm_val(c)) for c in cells if _mm_val(c) is not None]
        if not vals:
            return False
        if kind == 'avg':
            target = sum(v for _, v in vals) / len(vals)
            return any((x := _num(a)) is not None and abs(x - target) <= 0.01 * max(1, abs(target)) for a in ans)
        pick = max if check[1] == 'MOST RECENT' else min
        best = pick(v for _, v in vals)
        return any(c in ans for c, v in vals if v == best)
    if kind == 'compare':
        _, key, metric, how, a, b = check
        mids = [_mm_norm(x['answer']) for grp in inter for x in grp]
        names = []
        for side in (a, b):
            if side.startswith('{'):
                names.append([m for m in mids if _mm_select(header, rows, [(key, m)])][:1])
            else:
                names.append([_mm_norm(side.strip('[]'))])
        if not all(names):
            return False
        va, vb = (_mm_val(_mm_select(header, rows, [(key, n[0])])[0][header.index(_mm_norm(metric))]) for n in names)
        if va is None or vb is None or va == vb:
            return False
        higher = how in {'higher', 'most recent'}
        winner = names[0][0] if ((va > vb) == higher) else names[1][0]
        return winner in ans
    return False


def multimodalqa(root: Path) -> Iterator[dict]:
    tables = {}
    with gzip.open(root / 'dataset/MMQA_tables.jsonl.gz', 'rt', encoding='utf-8') as fh:
        for line in fh:
            t = json.loads(line)
            tables[t['id']] = t
    for split in ['train', 'dev']:
        with gzip.open(root / f'dataset/MMQA_{split}.jsonl.gz', 'rt', encoding='utf-8') as fh:
            for line in fh:
                q = json.loads(line)
                qtype = q['metadata']['type']
                if 'Image' in qtype or 'Table' not in qtype:
                    continue
                pseudo = q['metadata']['pseudo_language_question']
                parsed = _mm_parse(qtype, pseudo)
                composition = ('bridge' if qtype.startswith('Compose') or (qtype.startswith('Compare') and 'Compose' in qtype)
                               else 'parallel' if qtype.startswith(('Intersect', 'Compare')) else
                               'within_table_two_hop' if parsed and parsed[0] == 'of_sub' else 'single')
                if parsed is None:
                    yield _record('multimodalqa', q['qid'], split, q['question'], pseudo, None, False,
                                  'template not parsed', composition=composition, qtype=qtype)
                    continue
                template, body, check = parsed
                answers = [a['answer'] for a in q['answers']]
                table = tables.get(q['metadata']['table_id'])
                try:
                    header, rows = _mm_table(table)
                    ok = _mm_check(check, header, rows, answers, q['metadata'].get('intermediate_answers') or [])
                    detail = 'table part reproduces answer' if ok else 'table part does not reproduce answer'
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    ok, detail = False, f'execution: {exc}'
                if template in {'verify', 'compare'}:
                    tree = body
                elif template in {'intersect', 'text_of_table'}:
                    tree = body
                else:
                    ans = answers[0] if len(answers) == 1 else '|'.join(sorted(map(str, answers)))
                    tree = ('CMP_EQ', body, _tf_const(str(ans)) if re.fullmatch(r'-?[\d,]*\.?\d+', str(ans)) else ('STR', _mm_norm(ans)))
                yield _record('multimodalqa', q['qid'], split, q['question'], pseudo, aol.rewrite(tree), ok, detail,
                              composition=composition, qtype=qtype, template=template,
                              annotation='template pseudo-language + composition type')


# =========================================================================== HybridQA
def hybridqa(root: Path) -> Iterator[dict]:
    for split in ['train', 'dev']:
        data = json.loads((root / f'released_data/{split}.traced.json').read_text(encoding='utf-8'))
        for q in data:
            nodes = q.get('answer-node') or []
            kinds = {n[3] for n in nodes}
            if len(nodes) == 1 and kinds == {'passage'} and nodes[0][2]:
                comp, detail, ok = 'bridge', 'answer traced to a passage linked from one table cell', True
            elif len(nodes) == 1 and kinds == {'table'}:
                comp, detail, ok = 'single', 'answer traced to one table cell', True
            else:
                comp, detail, ok = 'ambiguous', f'{len(nodes)} traced nodes {sorted(kinds)}', False
            yield _record('hybridqa', q['question_id'], split, q['question'],
                          json.dumps(nodes, ensure_ascii=False), None, ok, detail, composition=comp,
                          annotation='dataset answer trace (hop structure only; no program)')


# =========================================================================== FEVEROUS
_EV = re.compile(r'^(.*)_(cell|header_cell|table_caption|sentence|item|section|title)_?([\d_]*)$')


def feverous(root: Path) -> Iterator[dict]:
    for split, name in [('train', 'feverous_train_challenges.jsonl'), ('dev', 'feverous_dev_challenges.jsonl')]:
        with (root / name).open(encoding='utf-8') as fh:
            for line in fh:
                x = json.loads(line)
                if not x.get('claim'):
                    continue
                for k, ev in enumerate(x.get('evidence') or []):
                    tables, sentences, bad = set(), set(), 0
                    for eid in ev.get('content', []):
                        m = _EV.match(eid)
                        if not m:
                            bad += 1
                            continue
                        page, kind, idx = m.groups()
                        if kind in {'cell', 'header_cell', 'table_caption'}:
                            tables.add((page, idx.split('_')[0]))
                        elif kind == 'sentence':
                            sentences.add((page, idx))
                    if not tables:
                        continue
                    comp = ('multi_table' if len(tables) >= 2 else 'table_text' if sentences else 'single_table')
                    ok = bad == 0 and x['label'] in {'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'}
                    yield _record('feverous', f"{x['id']}#{k}", split, x['claim'], json.dumps(sorted(tables)), None, ok,
                                  'evidence ids parsed' if ok else f'{bad} unparsed evidence ids', composition=comp,
                                  label=x['label'], challenge=x.get('challenge'), n_tables=len(tables),
                                  n_pages=len({p for p, _ in tables}),
                                  annotation='human evidence set (structure only; no program)')


PARSERS = {'tabfact': tabfact, 'wtq': wtq, 'tatqa': tatqa, 'multimodalqa': multimodalqa,
           'hybridqa': hybridqa, 'feverous': feverous}
