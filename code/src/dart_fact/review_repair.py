"""Versioned repairs for the reviewed task; legacy v1.0 execution stays unchanged.

Unlike the legacy early-stop executor, a missing conjunct does not conceal a
known false conjunct. Evidence includes the values scanned for an extreme,
ranking or count, and the predicates scanned for membership. This is an
operator-dependency trace, not a claim of minimal evidence.
"""
from __future__ import annotations

from .executor import Execution, InvalidProgram, Missing
from .tables import entity_key, parse_number

POLICY = "review_repair_v2"


class ReviewExecution(Execution):
    def _arg(self, value):
        result = super()._arg(value)
        if isinstance(result, Missing):
            raise result
        return result

    def _column_values(self, prof, col, keys):
        pairs = super()._column_values(prof, col, keys)
        for row, value in pairs:
            self._record(prof, row, col, value)
            self._record(prof, row, prof.key, entity_key(prof.cell(row, prof.key)))
        return pairs

    def _step(self, op):
        if op["op"] == "AND":
            values = []
            for argument in op["args"]:
                try:
                    value = self._arg(argument)
                except Missing as miss:
                    value = miss
                if not isinstance(value, (bool, Missing)):
                    raise InvalidProgram("AND requires Boolean or missing inputs")
                values.append(value)
            if any(value is False for value in values):
                return False
            for value in values:
                if isinstance(value, Missing):
                    raise value
            return True
        if op["op"] in ("FILTER", "JOIN", "AT_RANK"):
            prof = self._table(op["table"])
            field = {"FILTER": "col", "JOIN": "fk_col", "AT_RANK": "rank_col"}[op["op"]]
            col = self._col(prof, op[field])
            for row in range(len(prof.rows)):
                raw = prof.cell(row, col)
                value = parse_number(raw) if op["op"] == "AT_RANK" else None
                self._record(prof, row, col, value if value is not None else entity_key(raw))
        return super()._step(op)

    def run(self, operators):
        result = None
        for op in operators:
            try:
                result = self._step(op)
            except Missing as miss:
                result = miss
            if op.get("out"):
                self.env[op["out"]] = result
            value = ({"missing": result.what, "detail": result.detail}
                     if isinstance(result, Missing) else result)
            self.trace.append({"op": op["op"], "out": op.get("out"), "value": value})
        return result


def execute(operators, profiles):
    ex = ReviewExecution(profiles)
    result = ex.run(operators)
    if isinstance(result, Missing):
        label, missing, result_value = "NEI", {"what": result.what, "detail": result.detail}, None
    elif isinstance(result, bool):
        label, missing, result_value = ("SUPPORTS" if result else "REFUTES"), None, result
    else:
        raise InvalidProgram("program must end in a Boolean or a missing binding")
    return {"label": label, "result": result_value, "missing": missing, "cells": ex.cells,
            "trace": ex.trace, "tables_used": sorted(ex.tables_used)}
