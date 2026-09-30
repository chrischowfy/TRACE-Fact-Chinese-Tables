"""Executable verification programs.

A program is a JSON list of operators over the tables of an evidence package. Executing it returns
True (SUPPORTS), False (REFUTES) or Missing (NEI: a required binding — entity, metric column,
period table, join partner — is absent from the package). Every cell the program reads is recorded,
which is where ``evidence_cells`` come from. Ties in ARGMAX/ARGMIN/RANK and unparseable numeric
cells raise ``InvalidProgram``: such instances are dropped, never labeled.

Operators (fields in brackets are optional):
  LOOKUP        table, key, col, out                 value of `col` in the row whose key == key
  ATTR          table, key, col, out                 text attribute (entity_key form)
  ARGEXT        table, col, mode(max|min), [keys], out   key of the unique extreme row (optionally among keys)
  RANK          table, key, col, order(desc|asc), out    1-based competition rank; ties invalid
  AT_RANK       table, rank_col, k, out              key of the row whose rank cell == k (unique)
  FILTER        table, col, eq, out                  keys whose `col` (entity_key) == eq
  JOIN          table, fk_col, eq, out               keys of `table` whose fk (join_key) == join_key(eq)
  VALUES        table, keys, col, out                list of values for keys (missing key -> Missing)
  COUNT_GT      values, threshold, out               number of values > threshold
  SUM / AVG     values, out
  ARGEXT_OF     keys, values, mode, out              key of the unique extreme among paired lists
  SUB           left, right, out                     left - right
  COMPARE       left, cmp(gt|lt|eq|ne), right, out   numbers (tolerance 1e-9) or entity strings (join_key)
  AND           args, out
  COUNT         values, out
  ROUND         value, digits, out                   value rounded to the precision the claim states
  PCT_CHANGE    left, right, digits, out             (left - right) / right * 100, rounded; right > 0
  AVG_OF        args, out                            mean of scalar operands (one entity over several periods)
Arguments that are strings starting with "$" refer to earlier outputs.
"""
from __future__ import annotations

from typing import Any

from .tables import TableProfile, entity_key, join_key, parse_number


class Missing(Exception):
    """A required binding is not grounded in the package."""

    def __init__(self, what: str, detail: str = "") -> None:
        super().__init__(f"{what}: {detail}")
        self.what = what
        self.detail = detail


class InvalidProgram(Exception):
    """Program cannot be executed deterministically (tie, unparseable cell, bad reference)."""


EPS = 1e-9


class Execution:
    def __init__(self, profiles: dict[str, TableProfile]) -> None:
        self.profiles = profiles
        self.env: dict[str, Any] = {}
        self.cells: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.tables_used: set[str] = set()

    # ----------------------------------------------------------------- helpers
    def _arg(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            name = value[1:]
            if name not in self.env:
                raise InvalidProgram(f"unbound reference {value}")
            return self.env[name]
        return value

    def _table(self, table_id: str) -> TableProfile:
        prof = self.profiles.get(table_id)
        if prof is None:
            raise Missing("table", table_id)
        self.tables_used.add(table_id)
        return prof

    def _col(self, prof: TableProfile, header: str):
        col = prof.column(header)
        if col is None:
            raise Missing("column", f"{prof.table_id}.{header}")
        return col

    def _record(self, prof: TableProfile, row: int, col, value: Any) -> None:
        raw = prof.cell(row, col)
        cell = {"table_id": prof.table_id, "row": prof.row_map[row], "col": col.header,
                "raw_value": raw, "normalized_value": value}
        if cell not in self.cells:
            self.cells.append(cell)

    def _row(self, prof: TableProfile, key: str) -> int:
        hits = prof.find_rows(key) or prof.find_rows(key, loose=True)
        if len(hits) > 1:
            raise InvalidProgram(f"ambiguous entity {key} in {prof.table_id}")
        if not hits:
            raise Missing("entity", f"{prof.table_id}:{key}")
        idx = hits[0]
        self._record(prof, idx, prof.key, entity_key(prof.cell(idx, prof.key)))
        return idx

    def _num(self, prof: TableProfile, row: int, col) -> float:
        value = parse_number(prof.cell(row, col))
        if value is None:
            raise InvalidProgram(f"non-numeric cell {prof.table_id}[{row}].{col.header}")
        self._record(prof, row, col, value)
        return value

    def _column_values(self, prof: TableProfile, col, keys: list[str] | None):
        rows = range(len(prof.rows)) if keys is None else [self._row(prof, k) for k in keys]
        out = []
        for r in rows:
            v = parse_number(prof.cell(r, col))
            if v is None:
                raise InvalidProgram(f"non-numeric cell in {prof.table_id}.{col.header}")
            out.append((r, v))
        return out

    # ----------------------------------------------------------------- operators
    def run(self, operators: list[dict[str, Any]]) -> Any:
        result = None
        for op in operators:
            result = self._step(op)
            if op.get("out"):
                self.env[op["out"]] = result
            self.trace.append({"op": op["op"], "out": op.get("out"), "value": _jsonable(result)})
        return result

    def _step(self, op: dict[str, Any]) -> Any:
        kind = op["op"]
        if kind == "LOOKUP":
            prof = self._table(op["table"])
            col = self._col(prof, op["col"])
            return self._num(prof, self._row(prof, self._arg(op["key"])), col)
        if kind == "ATTR":
            prof = self._table(op["table"])
            col = self._col(prof, op["col"])
            row = self._row(prof, self._arg(op["key"]))
            value = entity_key(prof.cell(row, col))
            if not value:
                raise Missing("attribute", f"{prof.table_id}.{col.header}")
            self._record(prof, row, col, value)
            return value
        if kind == "ARGEXT":
            prof = self._table(op["table"])
            col = self._col(prof, op["col"])
            keys = self._arg(op["keys"]) if op.get("keys") else None
            pairs = self._column_values(prof, col, keys)
            if len(pairs) < 2:
                raise InvalidProgram("argext over <2 rows")
            best = (max if op["mode"] == "max" else min)(v for _, v in pairs)
            winners = [r for r, v in pairs if abs(v - best) <= EPS]
            if len(winners) != 1:
                raise InvalidProgram("tie at extreme")
            row = winners[0]
            self._record(prof, row, col, best)
            self._record(prof, row, prof.key, entity_key(prof.cell(row, prof.key)))
            return entity_key(prof.cell(row, prof.key))
        if kind == "RANK":
            prof = self._table(op["table"])
            col = self._col(prof, op["col"])
            row = self._row(prof, self._arg(op["key"]))
            mine = self._num(prof, row, col)
            values = [v for _, v in self._column_values(prof, col, None)]
            if sum(1 for v in values if abs(v - mine) <= EPS) != 1:
                raise InvalidProgram("tie at rank")
            better = sum(1 for v in values if (v > mine + EPS if op.get("order", "desc") == "desc" else v < mine - EPS))
            return better + 1
        if kind == "AT_RANK":
            prof = self._table(op["table"])
            col = self._col(prof, op["rank_col"])
            k = self._arg(op["k"])
            rows = [r for r in range(len(prof.rows)) if parse_number(prof.cell(r, col)) == float(k)]
            if not rows:
                raise Missing("rank", f"{prof.table_id}:{k}")
            if len(rows) != 1:
                raise InvalidProgram("shared rank")
            self._record(prof, rows[0], col, float(k))
            self._record(prof, rows[0], prof.key, entity_key(prof.cell(rows[0], prof.key)))
            return entity_key(prof.cell(rows[0], prof.key))
        if kind == "FILTER":
            prof = self._table(op["table"])
            col = self._col(prof, op["col"])
            target = entity_key(self._arg(op["eq"]))
            rows = [r for r in range(len(prof.rows)) if entity_key(prof.cell(r, col)) == target]
            if not rows:
                raise Missing("filter_value", f"{prof.table_id}.{col.header}={target}")
            for r in rows:
                self._record(prof, r, col, target)
                self._record(prof, r, prof.key, entity_key(prof.cell(r, prof.key)))
            return [entity_key(prof.cell(r, prof.key)) for r in rows]
        if kind == "JOIN":
            prof = self._table(op["table"])
            col = self._col(prof, op["fk_col"])
            target = join_key(self._arg(op["eq"]))
            rows = [r for r in range(len(prof.rows)) if join_key(prof.cell(r, col)) == target]
            if not rows:
                raise Missing("join_partner", f"{prof.table_id}.{col.header}={target}")
            for r in rows:
                self._record(prof, r, col, entity_key(prof.cell(r, col)))
                self._record(prof, r, prof.key, entity_key(prof.cell(r, prof.key)))
            return [entity_key(prof.cell(r, prof.key)) for r in rows]
        if kind == "VALUES":
            prof = self._table(op["table"])
            col = self._col(prof, op["col"])
            keys = self._arg(op["keys"])
            out = []
            for key in keys:
                row = self._row(prof, key)
                out.append(self._num(prof, row, col))
            return out
        if kind == "COUNT_GT":
            return sum(1 for v in self._arg(op["values"]) if v > float(self._arg(op["threshold"])) + EPS)
        if kind == "COUNT":
            return len(self._arg(op["values"]))
        if kind == "SUM":
            return round(sum(self._arg(op["values"])), 6)
        if kind == "AVG":
            vals = self._arg(op["values"])
            return round(sum(vals) / len(vals), 6)
        if kind == "ARGEXT_OF":
            keys, vals = self._arg(op["keys"]), self._arg(op["values"])
            if len(keys) != len(vals) or len(keys) < 2:
                raise InvalidProgram("argext_of needs >=2 paired items")
            best = (max if op["mode"] == "max" else min)(vals)
            winners = [k for k, v in zip(keys, vals) if abs(v - best) <= EPS]
            if len(winners) != 1:
                raise InvalidProgram("tie at extreme")
            return winners[0]
        if kind == "COUNT_ABOVE":
            prof = self._table(op["table"])
            col = self._col(prof, op["col"])
            threshold = float(self._arg(op["threshold"]))
            pairs = self._column_values(prof, col, None)
            if len(pairs) < 3:
                raise InvalidProgram("count over <3 rows")
            hits = [(r, v) for r, v in pairs if v > threshold + EPS]
            # like ARGEXT, the recorded evidence is the rows that carry the answer, not the whole column
            for row, value in hits:
                self._record(prof, row, col, value)
                self._record(prof, row, prof.key, entity_key(prof.cell(row, prof.key)))
            return len(hits)
        if kind == "EXTVAL":
            vals = self._arg(op["values"])
            if len(vals) < 2:
                raise InvalidProgram("extval needs >=2 values")
            return (max if op["mode"] == "max" else min)(vals)
        if kind == "SUB_VALUES":
            # elementwise, so that a growth ranking inside a category can be argmax'd; the evidence is
            # already recorded by the two VALUES reads that produced the lists
            a, b = self._arg(op["left"]), self._arg(op["right"])
            if len(a) != len(b) or len(a) < 2:
                raise InvalidProgram("sub_values needs two paired lists of >=2")
            return [round(float(x) - float(y), 6) for x, y in zip(a, b)]
        if kind == "SUB":
            return round(float(self._arg(op["left"])) - float(self._arg(op["right"])), 6)
        if kind == "COMPARE":
            left, right, cmp = self._arg(op["left"]), self._arg(op["right"]), op["cmp"]
            if isinstance(left, str) or isinstance(right, str):
                same = join_key(left) == join_key(right)
                if cmp == "eq":
                    return same
                if cmp == "ne":
                    return not same
                raise InvalidProgram("ordering comparison on strings")
            left, right = float(left), float(right)
            return {"gt": left > right + EPS, "lt": left < right - EPS,
                    "eq": abs(left - right) <= EPS, "ne": abs(left - right) > EPS}[cmp]
        if kind == "AND":
            return all(bool(self._arg(a)) for a in op["args"])
        if kind == "AVG_OF":
            vals = [float(self._arg(a)) for a in op["args"]]
            if len(vals) < 2:
                raise InvalidProgram("avg_of needs >=2 operands")
            return round(sum(vals) / len(vals), 6)
        if kind == "ROUND":
            return round(float(self._arg(op["value"])), int(op["digits"]))
        if kind == "PCT_CHANGE":
            new, old = float(self._arg(op["left"])), float(self._arg(op["right"]))
            if old <= EPS:
                raise InvalidProgram("percentage change needs a positive base")
            return round((new - old) / old * 100, int(op.get("digits", 1)))
        raise InvalidProgram(f"unknown operator {kind}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def execute(operators: list[dict[str, Any]], profiles: dict[str, TableProfile]) -> dict[str, Any]:
    """Run a program; returns {label, result, missing, cells, trace, tables_used}. Raises InvalidProgram."""
    ex = Execution(profiles)
    try:
        result = ex.run(operators)
    except Missing as miss:
        return {"label": "NEI", "result": None, "missing": {"what": miss.what, "detail": miss.detail},
                "cells": ex.cells, "trace": ex.trace, "tables_used": sorted(ex.tables_used)}
    if not isinstance(result, bool):
        raise InvalidProgram("program must end in a boolean")
    return {"label": "SUPPORTS" if result else "REFUTES", "result": result, "missing": None,
            "cells": ex.cells, "trace": ex.trace, "tables_used": sorted(ex.tables_used)}
