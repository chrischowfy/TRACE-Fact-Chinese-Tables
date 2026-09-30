"""Conservative semantic quarantine, not automatic alias-based relabelling.

Rules were developed after an r4 pilot/full-run inspection exposed 总计/总数.
They apply to every candidate independently of its model prediction. A flagged
record is excluded pending semantic adjudication, not assigned the model label.
"""
import re


def base_header(value):
    value = re.sub(r"[（(].*?[）)]", "", str(value))
    return re.sub(r"[\s*]", "", value).lower()


def quarantine_reason(row):
    if row["label"] != "NEI" or row.get("quality_flags", {}).get("perturbation_axis") != "metric":
        return None
    metric = base_header(row["program"].get("slots", {}).get("metric_name", ""))
    headers = {base_header(h) for t in row["tables"] for h in t["headers"]}
    if re.fullmatch(r"线路\d+", metric):
        return {"rule":"categorical_route_identifier_as_numeric_metric", "metric":metric}
    aliases = {
        "总计": {"总数", "合计"},
        "总数": {"总计", "合计"},
        "设计最高时速": {"设计速度", "设计时速"},
        "最大深度": {"最大水深"},
        "最大水深": {"最大深度"},
        "乘客总量": {"客运", "旅客吞吐量", "客运量", "乘客总数"},
    }
    overlap = headers & aliases.get(metric, set())
    if overlap:
        return {"rule":"missing_metric_has_contextual_header_alias", "metric":metric, "headers":sorted(overlap)}
    # Plausible equivalence is enough to invalidate a confident absence claim;
    # it is NOT enough to relabel without checking source definitions/units.
    ambiguous = {"地上楼层": {"楼层数", "楼层"}, "销售额": {"收入", "营业收入"}}
    overlap = headers & ambiguous.get(metric, set())
    if overlap:
        return {"rule":"missing_metric_scope_ambiguous_with_present_header", "metric":metric, "headers":sorted(overlap)}
    return None
