"""Deterministic Chinese surface realization.

Every skeleton has several paraphrase templates. The template is chosen from the *base* id (shared by
the SUPPORTS / REFUTES / NEI siblings built from the same grounded fact), never from the label, so a
surface pattern cannot predict the label. Entity names and numbers are inserted verbatim from slots.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .source_adapters import comparison_words


def _pick(base_id: str, options: list[str]) -> str:
    h = int(hashlib.sha1(base_id.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def _cmp(style: str, cmp: str) -> str:
    return comparison_words(style)[cmp]


def _num(v: dict[str, Any]) -> str:
    return f"{v['text']}{v.get('unit', '')}"


def _year(v: Any) -> str:
    """2012 -> 2012年; season labels such as 2012至2013赛季 are kept as they are."""
    return f"{v}年" if str(v).isdigit() else str(v)


def _verb(direction: str) -> str:
    return "增加" if direction == "up" else "减少"


def _more(style: str, direction: str = "up") -> str:
    return comparison_words(style)["more" if direction == "up" else "less"]


def realize(skeleton: str, s: dict[str, Any], base_id: str) -> str:
    sc = s.get("scope", "")
    m, style = s.get("metric_name", ""), s.get("style", "amount")
    noun = s.get("noun", "对象")
    if skeleton in ("ST_COMPARE", "XT_COMPARE"):
        e1, e2, cmp = s["e1"], s["e2"], s["cmp"]
        if cmp == "eq":
            return _pick(base_id, [f"在{sc}中，{e1}的{m}与{e2}相同。", f"{sc}，{e1}和{e2}的{m}一样。",
                                   f"在{sc}里，{e1}与{e2}的{m}持平。"])
        w = _cmp(style, cmp)
        return _pick(base_id, [f"在{sc}中，{e1}的{m}{w}{e2}。", f"{sc}，{e1}的{m}{w}{e2}。",
                               f"就{m}而言，{sc}中{e1}{w}{e2}。"])
    if skeleton == "ST_SUPERLATIVE":
        w = _cmp(style, s["mode"])
        e = s["claimed"]
        return _pick(base_id, [f"在{sc}中，{m}{w}的{noun}是{e}。", f"{sc}，{e}是{m}{w}的{noun}。",
                               f"在{sc}的所有{noun}中，{e}的{m}{w}。"])
    if skeleton == "ST_RANK":
        e, k = s["e"], s["k"]
        return _pick(base_id, [f"在{sc}中，{e}的{m}按从高到低排在第{k}位。",
                               f"{sc}，按{m}从高到低排序，{e}位列第{k}。"])
    if skeleton in ("ST_DIFF", "XT_DIFF"):
        e1, e2, d = s["e1"], s["e2"], s["diff"]
        more = _more(style)
        return _pick(base_id, [f"在{sc}中，{e1}的{m}比{e2}{more}{_num(d)}。",
                               f"{sc}，{e1}的{m}比{e2}{more}出{_num(d)}。"])
    if skeleton == "CAT_ARGEXT":
        w = _cmp(style, s["mode"])
        return _pick(base_id, [f"在{sc}中，{s['cat_col']}为{s['cat']}的{noun}里，{m}{w}的是{s['claimed']}。",
                               f"{sc}，在所有{s['cat_col']}为{s['cat']}的{noun}中，{s['claimed']}的{m}{w}。"])
    if skeleton == "ST_COUNT_GT":
        return _pick(base_id, [f"在{sc}中，{m}超过{_num(s['threshold'])}的{noun}有{s['n']}个。",
                               f"{sc}里，共有{s['n']}个{noun}的{m}高于{_num(s['threshold'])}。"])
    if skeleton == "CAT_SUM":
        return _pick(base_id, [f"在{sc}中，{s['cat_col']}为{s['cat']}的{noun}的{m}合计为{_num(s['total'])}。",
                               f"{sc}，{s['cat_col']}属于{s['cat']}的{noun}，{m}加总为{_num(s['total'])}。"])
    if skeleton == "CAT_COUNT":
        return _pick(base_id, [f"在{sc}中，{s['cat_col']}为{s['cat']}的{noun}共有{s['n']}个。",
                               f"{sc}里，{s['cat_col']}属于{s['cat']}的{noun}一共{s['n']}个。"])
    if skeleton == "RJ_RANK_TOP":
        w = _cmp(style, "max")
        return _pick(base_id, [f"在{sc}中，排名第{s['k']}的{s['rank_noun']}里，{m}{w}的{noun}是{s['claimed']}。",
                               f"{sc}，排在第{s['k']}位的{s['rank_noun']}中，{s['claimed']}是{m}{w}的{noun}。"])
    if skeleton == "RJ_RANK_TOPVAL":
        w = _cmp(style, "max")
        return _pick(base_id, [f"在{sc}中，排名第{s['k']}的{s['rank_noun']}里，{noun}的{m}{w}为{_num(s['value'])}。",
                               f"{sc}，排在第{s['k']}位的{s['rank_noun']}中，{m}{w}的{noun}达到{_num(s['value'])}。"])
    if skeleton == "RJ_MEMBER_RANK":
        return _pick(base_id, [f"在{sc}中，{s['member']}所在的{s['rank_noun']}排名第{s['k']}。",
                               f"{sc}，{s['member']}效力的{s['rank_noun']}位列第{s['k']}名。"])
    if skeleton == "HP_AND":
        return _pick(base_id, [f"在{sc}中，{s['e']}的{s['attr_col']}是{s['attr']}，并且其{m}为{_num(s['value'])}。",
                               f"{sc}，{s['attr_col']}为{s['attr']}的{s['e']}，{m}是{_num(s['value'])}。"])
    if skeleton == "HP_CMP":
        w = _cmp(style, s["cmp"])
        return _pick(base_id, [f"在{sc}中，{s['attr_col']}为{s['attr']}的{s['e1']}，{m}{w}{s['e2']}。",
                               f"{sc}，{s['e1']}的{s['attr_col']}是{s['attr']}，且其{m}{w}{s['e2']}。"])
    if skeleton == "CAT_PARTNER":
        w = _cmp(style, s["mode"])
        w2 = _cmp(s.get("style2", "amount"), s["cmp"])
        return _pick(base_id, [f"在{sc}中，{s['cat_col']}为{s['cat']}的{noun}里，{m}{w}的是{s['claimed']}，其{s['metric2']}{w2}{s['e2']}。",
                               f"{sc}，{s['cat_col']}是{s['cat']}的所有{noun}中{m}{w}的{s['claimed']}，{s['metric2']}{w2}{s['e2']}。"])
    if skeleton == "JF_ARGEXT":
        w = _cmp(style, s["mode"])
        return _pick(base_id, [f"在{sc}中，{s['attr_col']}为{s['attr']}的{noun}里，{m}{w}的是{s['claimed']}。",
                               f"{sc}，{s['attr_col']}是{s['attr']}的所有{noun}中，{s['claimed']}的{m}{w}。"])
    if skeleton == "JF_COUNT_GT":
        return _pick(base_id, [f"在{sc}中，{s['attr_col']}为{s['attr']}的{noun}里，{m}超过{_num(s['threshold'])}的有{s['n']}个。",
                               f"{sc}，{s['attr_col']}是{s['attr']}的{noun}中共有{s['n']}个{m}高于{_num(s['threshold'])}。"])
    if skeleton == "JF_SUM":
        return _pick(base_id, [f"在{sc}中，{s['attr_col']}为{s['attr']}的所有{noun}的{m}合计为{_num(s['total'])}。",
                               f"{sc}，{s['attr_col']}是{s['attr']}的{noun}，{m}加起来是{_num(s['total'])}。"])
    if skeleton == "CP_DIRECTION":
        verb = _verb(s["direction"])
        return _pick(base_id, [f"在{sc}中，与{_year(s['y1'])}相比，{s['e']}{_year(s['y2'])}的{m}有所{verb}。",
                               f"在{sc}中，{s['e']}的{m}从{_year(s['y1'])}到{_year(s['y2'])}{verb}了。",
                               f"{sc}，{s['e']}{_year(s['y2'])}的{m}比{_year(s['y1'])}{_more(style, s['direction'])}。"])
    if skeleton == "CP_DIFF":
        verb = _verb(s["direction"])
        return _pick(base_id, [f"在{sc}中，与{_year(s['y1'])}相比，{s['e']}{_year(s['y2'])}的{m}{verb}了{_num(s['diff'])}。",
                               f"在{sc}中，{s['e']}的{m}从{_year(s['y1'])}到{_year(s['y2'])}{verb}了{_num(s['diff'])}。"])
    raise KeyError(skeleton)
