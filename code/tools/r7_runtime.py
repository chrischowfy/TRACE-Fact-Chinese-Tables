"""Process-local acceleration with unchanged profile and evidence semantics.

No frozen source file is edited. Profile cache keys include actual table bytes;
counterfactual tables cannot accidentally reuse an unchanged table's profile.
"""
from functools import lru_cache
import json
import sys


def enable():
    from dart_fact import tables, executor
    if getattr(tables.profile_table,'_r7_cache',False):
        return
    original=tables.profile_table
    @lru_cache(maxsize=2048)
    def cached(serialized):
        return original(json.loads(serialized))
    def profile(table):
        # All metadata is included because key selection can depend on it.
        return cached(json.dumps(table,ensure_ascii=False,sort_keys=True,separators=(',',':')))
    profile._r7_cache=True
    for name in ['dart_fact.tables','dart_fact.packages','dart_fact.families','dart_fact.pipeline',
                 'repair_review_release','repair_review_r6','r7_extensions','r7_quality']:
        module=sys.modules.get(name)
        if module and getattr(module,'profile_table',None) is original:
            module.profile_table=profile
    def record(self,prof,row,col,value):
        raw=prof.cell(row,col)
        key=(prof.table_id,prof.row_map[row],col.header,raw,value)
        if not hasattr(self,'_r7_cell_set'):
            self._r7_cell_set={(c['table_id'],c['row'],c['col'],c['raw_value'],c['normalized_value']) for c in self.cells}
        if key not in self._r7_cell_set:
            self._r7_cell_set.add(key)
            self.cells.append({'table_id':prof.table_id,'row':prof.row_map[row],'col':col.header,
                               'raw_value':raw,'normalized_value':value})
    executor.Execution._record=record
