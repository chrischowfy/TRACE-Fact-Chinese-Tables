"""Abstract operator language (AOL) shared by source programs and Chinese programs.

Skeleton induction compares programs from very different formalisms (TabFact LPA programs, WTQ
lambda-DCS, TAT-QA derivations, MultiModalQA templates, and this package's executable JSON programs).
All of them are converted into one expression tree over a single *universal relation*:

  leaves   ('ROWS',)  ('COL', name)  ('STR', v)  ('NUM', v)  ('SUBQ', tag)
  rows     ('FILTER', rows, col, cmp, value)      cmp in EQ | NEQ | ORD | IN
           ('FILTERS', rows, ('COND', col, cmp, value), ...)   a conjunction of selections
  row      ('ARGEXT', rows, col)                   extreme row; max/min is a slot, not structure
           ('NTH_ARGEXT', rows, col, n)            ('ORDER', rows)   first/last/next by table order
  value    ('HOP', row_or_rows, col)   ('EXTVAL', rows, col)   ('COUNT', rows)   ('SUM', rows, col)
           ('AVG', rows, col)   ('MODE', rows, col)   ('RANK_OF', rows, col)
           ('SUB', a, b)   ('ADD', a, b, ...)   ('DIV', a, b)   ('PCT_CHANGE', new, old)   ('AVG_OF', a, b, ...)
  bool     ('CMP_EQ', a, b)   ('CMP_NEQ', a, b)   ('CMP_ORD', a, b)   ('AND', a, b, ...)
           ('ONLY', rows)   ('ALL', rows, col, cmp, value)

Delexicalization numbers columns C1.., string constants S1.., numbers N1.. in first-occurrence order
after commutative children are put in canonical order, so two programs share a signature exactly when
they apply the same operators to the same pattern of columns and constants.  Comparator direction,
extreme direction and the sign of a difference are slots (refutation axes), not structure.
"""
from __future__ import annotations

from itertools import permutations, product
from typing import Any

COMMUTATIVE = {'CMP_EQ', 'CMP_NEQ', 'CMP_ORD', 'AND', 'ADD', 'AVG_OF', 'SUB', 'FILTERS'}
LEAVES = {'ROWS', 'COL', 'STR', 'NUM', 'SUBQ'}
BOOL_ROOTS = {'CMP_EQ', 'CMP_NEQ', 'CMP_ORD', 'AND', 'ONLY', 'ALL'}


def rows() -> tuple:
    return ('ROWS',)


def col(name: str) -> tuple:
    return ('COL', str(name))


def const(value: Any) -> tuple:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ('NUM', float(value))
    return ('STR', str(value))


def lookup(key_col: str, key: Any, column: str) -> tuple:
    return ('HOP', ('FILTER', rows(), col(key_col), 'EQ', key if isinstance(key, tuple) else const(key)), col(column))


# ------------------------------------------------------------------ rewriting
def rewrite(t: tuple) -> tuple:
    """Bottom-up normalization of equivalent forms."""
    if t[0] in LEAVES:
        return t
    t = (t[0],) + tuple(rewrite(c) if isinstance(c, tuple) else c for c in t[1:])
    op = t[0]
    # (a - b) / b  ->  PCT_CHANGE(a, b)
    if op == 'DIV' and isinstance(t[1], tuple) and t[1][0] == 'SUB' and t[1][2] == t[2]:
        return ('PCT_CHANGE', t[1][1], t[2])
    # (a + b + ...) / k  ->  AVG_OF(a, b, ...) when k is the operand count
    if op == 'DIV' and isinstance(t[1], tuple) and t[1][0] == 'ADD' and t[2][0] == 'NUM' \
            and abs(t[2][1] - (len(t[1]) - 1)) < 1e-9:
        return ('AVG_OF',) + t[1][1:]
    # re-selecting the row of an entity that was itself selected by key is a hop on that row:
    # HOP(FILTER(ROWS, K, EQ, HOP(x, K')), C) with K, K' the same key  ->  HOP(x, C)
    if op == 'HOP' and t[1][0] == 'FILTER' and t[1][1] == ('ROWS',) and t[1][3] == 'EQ':
        inner = t[1][4]
        if inner[0] == 'HOP' and inner[2] == t[1][2] and inner[1][0] in {'ARGEXT', 'NTH_ARGEXT', 'ORDER'}:
            return ('HOP', inner[1], t[2])
    # a chain of selections is a set of conditions; their nesting order is not structure
    if op == 'FILTER' and t[1][0] in {'FILTER', 'FILTERS'}:
        inner = t[1]
        conds = [('COND', inner[2], inner[3], inner[4])] if inner[0] == 'FILTER' else list(inner[2:])
        return ('FILTERS', inner[1]) + tuple(conds + [('COND', t[2], t[3], t[4])])
    if op in {'ADD', 'AND'}:        # flatten nested associative ops
        flat = []
        for c in t[1:]:
            flat.extend(c[1:] if c[0] == op else [c])
        return (op,) + tuple(flat)
    return t


# ------------------------------------------------------------------ canonical signature
def _shape(t: Any) -> str:
    if not isinstance(t, tuple):
        return str(t)
    if t[0] in {'COL', 'STR', 'NUM'}:
        return t[0][0]
    if t[0] in {'ROWS', 'SUBQ'}:
        return t[0]
    kids = [_shape(c) for c in t[1:]]
    if t[0] in COMMUTATIVE:
        kids.sort()
    return t[0] + '(' + ','.join(kids) + ')'


def _render(t: Any, names: dict) -> str:
    if not isinstance(t, tuple):
        return str(t)
    kind = t[0]
    if kind in {'COL', 'STR', 'NUM'}:
        prefix = {'COL': 'C', 'STR': 'S', 'NUM': 'N'}[kind]
        # equal numbers are a coincidence of the instance, never a structural co-reference
        key = (kind, t[1]) if kind != 'NUM' else (kind, len(names))
        if key not in names:
            names[key] = f'{prefix}{sum(1 for k in names if k[0] == kind) + 1}'
        return names[key]
    if kind in {'ROWS', 'SUBQ'}:
        return kind
    return kind + '(' + ','.join(_render(c, names) for c in t[1:]) + ')'


def _orderings(t: Any, budget: list[int]):
    """All child orderings of commutative nodes whose children tie on shape (bounded)."""
    if not isinstance(t, tuple) or t[0] in LEAVES:
        yield t
        return
    kid_options = [list(_orderings(c, budget)) for c in t[1:]]
    first = True
    for kids in product(*kid_options):
        if t[0] in COMMUTATIVE:
            kids = sorted(kids, key=_shape)
            groups: list[list] = []
            for k in kids:
                if groups and _shape(groups[-1][0]) == _shape(k):
                    groups[-1].append(k)
                else:
                    groups.append([k])
            variants = product(*[permutations(g) for g in groups])
        else:
            variants = [[[k] for k in kids]]
        for perm in variants:
            # the budget bounds the search, but every node yields at least its sorted ordering
            if not first and budget[0] <= 0:
                return
            first = False
            budget[0] -= 1
            yield (t[0],) + tuple(x for g in perm for x in g)


def signature(t: tuple) -> str:
    """Delexicalized canonical form (lexicographically least over commutative orderings)."""
    best = None
    for variant in _orderings(t, [5000]):
        s = _render(variant, {})
        if best is None or s < best:
            best = s
    return best


def shape(t: tuple) -> str:
    return _shape(t)


# ------------------------------------------------------------------ decomposition
def subtrees(t: Any):
    if isinstance(t, tuple):
        yield t
        for c in t[1:]:
            yield from subtrees(c)


def is_constant(t: Any) -> bool:
    return isinstance(t, tuple) and t[0] in {'STR', 'NUM', 'SUBQ'}


def _value_slot(s: tuple) -> int | None:
    """Index of the comparison value in a selection node (FILTER or one COND of a FILTERS chain)."""
    return 4 if s[0] == 'FILTER' else 3 if s[0] == 'COND' else None


def bridges(t: tuple) -> list[tuple]:
    """Selections whose comparison value is computed by another sub-program (multi-hop dependency)."""
    return [s for s in subtrees(t) if _value_slot(s) and not is_constant(s[_value_slot(s)])]


def split_bridges(t: tuple) -> tuple[tuple, list[tuple]]:
    """Replace every bridge value by SUBQ; return (outer, [inner programs])."""
    inners: list[tuple] = []

    def walk(x):
        if not isinstance(x, tuple) or x[0] in LEAVES:
            return x
        i = _value_slot(x)
        if i and not is_constant(x[i]):
            inner, more = split_bridges(x[i])
            inners.append(inner)
            inners.extend(more)
            return tuple(walk(c) if j < i else c for j, c in enumerate(x[:i])) + (('SUBQ', 'q'),) + x[i + 1:]
        return (x[0],) + tuple(walk(c) for c in x[1:])

    return walk(t), inners


def fill_holes(t: tuple) -> tuple:
    """A bridge outer with its holes filled by a string constant (the equivalent single-hop program)."""
    if not isinstance(t, tuple):
        return t
    if t[0] == 'SUBQ':
        return ('STR', '__hole__')
    return (t[0],) + tuple(fill_holes(c) for c in t[1:])


def fragment_signatures(t: tuple) -> set[str]:
    """Signatures of every non-leaf sub-expression (used for fragment support)."""
    return {signature(s) for s in subtrees(t) if s[0] not in LEAVES}


# ------------------------------------------------------------------ Chinese programs
SYNTHETIC_KEY_HEADERS = {'类别编号'}


def _table_facts(tables: list[dict]) -> dict[str, dict]:
    from .tables import join_key, profile_table
    facts = {}
    for t in tables:
        prof = profile_table(t)
        key = prof.key.header if prof.key else t['headers'][0]
        ki = t['headers'].index(key)
        facts[t['table_id']] = {'key': key, 'headers': tuple(t['headers']),
                                'keys': {join_key(r[ki]) for r in t['rows'] if ki < len(r)},
                                'derivation': (t.get('source') or {}).get('derivation')}
    return facts


def relation_layout(tables: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Map each table to a relation id and a key group.

    Same-schema tables with disjoint entities are one relation (row partition, e.g. split tables);
    tables sharing entities are joined on their key (one key group) but stay separate relations, so
    a column of the same name in two period snapshots remains two columns.
    """
    facts = _table_facts(tables)
    ids = list(facts)
    rel = {t: t for t in ids}
    grp = {t: t for t in ids}

    def find(m, x):
        while m[x] != x:
            m[x] = m[m[x]]
            x = m[x]
        return x

    for a, b in [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]:
        fa, fb = facts[a], facts[b]
        shared = fa['keys'] & fb['keys']
        if fa['headers'] == fb['headers'] and not shared:
            rel[find(rel, a)] = find(rel, b)
            grp[find(grp, a)] = find(grp, b)
        # co-keyed tables share most entities; a few coincidental names (a region and a country both
        # called 南非) do not make a category dictionary an entity table
        elif shared and len(shared) >= 0.5 * min(len(fa['keys']), len(fb['keys'])):
            grp[find(grp, a)] = find(grp, b)
    return {t: find(rel, t) for t in ids}, {t: find(grp, t) for t in ids}


def abstract_program(operators: list[dict], tables: list[dict]) -> tuple:
    """Convert an executable JSON program into an AOL tree over the universal relation."""
    facts = _table_facts(tables)
    rel, grp = relation_layout(tables)

    key_of: dict[tuple, tuple] = {}      # column -> key column of its relation (for ARGEXT_OF winners)

    def colid(table_id: str, header: str) -> tuple:
        key = ('COL', 'KEY@' + grp[table_id])
        c = key if header == facts[table_id]['key'] else ('COL', rel[table_id] + '.' + header)
        key_of[c] = key
        return c

    def scalar(v):
        if isinstance(v, str) and not v.startswith('$'):
            try:
                return const(float(v))
            except ValueError:
                return const(v)
        return arg(v)

    def is_synthetic(table_id: str, header: str) -> bool:
        return header in SYNTHETIC_KEY_HEADERS and facts[table_id]['derivation'] == 'category_decomposition'

    env: dict[str, tuple] = {}

    def arg(v):
        if isinstance(v, str) and v.startswith('$'):
            return env[v[1:]]
        return const(v)

    def rows_of(v):
        """Key lists are row sets of the universal relation."""
        return arg(v)

    last = None
    for op in operators:
        k = op['op']
        t = op.get('table')
        if k in {'LOOKUP', 'ATTR'}:
            node = ('HOP', ('FILTER', rows(), colid(t, facts[t]['key']), 'EQ', arg(op['key'])), colid(t, op['col']))
        elif k == 'ARGEXT':
            base = rows_of(op['keys']) if op.get('keys') else rows()
            node = ('HOP', ('ARGEXT', base, colid(t, op['col'])), colid(t, facts[t]['key']))
        elif k == 'RANK':
            node = ('RANK_OF', ('FILTER', rows(), colid(t, facts[t]['key']), 'EQ', arg(op['key'])), colid(t, op['col']))
        elif k == 'AT_RANK':
            node = ('HOP', ('FILTER', rows(), colid(t, op['rank_col']), 'EQ', scalar(op['k'])),
                    colid(t, facts[t]['key']))
        elif k == 'FILTER':
            node = ('FILTER', rows(), colid(t, op['col']), 'EQ', arg(op['eq']))
        elif k == 'JOIN':
            value = arg(op['eq'])
            if is_synthetic(t, op['fk_col']) and value[0] == 'HOP' and value[1][0] == 'FILTER':
                # dictionary lookup name -> synthetic code -> members: the code carries no fact, so the
                # pair is the selection on the category name it encodes
                node = ('FILTER', rows(), value[1][2], 'EQ', value[1][4])
            else:
                node = ('FILTER', rows(), colid(t, op['fk_col']), 'EQ', value)
        elif k == 'VALUES':
            node = ('PROJ', rows_of(op['keys']), colid(t, op['col']))
        elif k == 'COUNT_GT':
            p = arg(op['values'])
            node = ('COUNT', ('FILTER', p[1], p[2], 'ORD', scalar(op['threshold'])))
        elif k == 'COUNT_ABOVE':
            node = ('COUNT', ('FILTER', rows(), colid(t, op['col']), 'ORD', scalar(op['threshold'])))
        elif k == 'COUNT':
            v = arg(op['values'])
            node = ('COUNT', v[1] if v[0] == 'PROJ' else v)
        elif k in {'SUM', 'AVG'}:
            p = arg(op['values'])
            node = (k, p[1], p[2])
        elif k == 'EXTVAL':
            p = arg(op['values'])
            node = ('EXTVAL', p[1], p[2])
        elif k == 'ARGEXT_OF':
            p = arg(op['values'])
            # the winner is the key of the projected rows
            node = ('HOP', ('ARGEXT', arg(op['keys']), p[2]), key_of[p[2]])
        elif k == 'SUB':
            node = ('SUB', arg(op['left']), arg(op['right']))
        elif k == 'SUB_VALUES':
            a, b = arg(op['left']), arg(op['right'])
            c = ('COL', f"({a[2][1]})-({b[2][1]})")
            key_of[c] = key_of[a[2]]
            node = ('PROJ', a[1], c)
        elif k == 'COMPARE':
            kind = {'eq': 'CMP_EQ', 'ne': 'CMP_NEQ', 'gt': 'CMP_ORD', 'lt': 'CMP_ORD'}[op['cmp']]
            node = (kind, arg(op['left']), arg(op['right']))
        elif k == 'AND':
            node = ('AND',) + tuple(arg(a) for a in op['args'])
        elif k == 'ROUND':
            node = arg(op['value'])          # the stated precision is presentation, not structure
        elif k == 'AVG_OF':
            node = ('AVG_OF',) + tuple(arg(a) for a in op['args'])
        elif k == 'PCT_CHANGE':
            node = ('PCT_CHANGE', arg(op['left']), arg(op['right']))
        else:
            raise ValueError(f'unsupported operator {k}')
        if op.get('out'):
            env[op['out']] = node
        last = node
    return rewrite(last)


def topology(operators: list[dict], tables: list[dict], tree: tuple) -> str:
    """How the tables of a program are connected, as the card's required table topology."""
    used = {op['table'] for op in operators if op.get('table')}
    if len(used) <= 1:
        return 'single_table'
    if bridges(tree):
        return 'bridge'
    facts = _table_facts([t for t in tables if t['table_id'] in used])
    if any(f['derivation'] == 'category_decomposition' for f in facts.values()):
        return 'category_decomposition' if len(used) <= 3 else 'category_decomposition_plus_partner'
    rel, _ = relation_layout([t for t in tables if t['table_id'] in used])
    if len(set(rel.values())) == 1:
        return 'row_partition'
    return 'co_keyed'
