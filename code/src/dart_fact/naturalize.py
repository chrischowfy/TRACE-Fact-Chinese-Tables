"""Slot-locked LLM paraphrase of template claims.

A language model never sees or changes a label, an entity, a number or a comparison word. For every group of
sibling instances (same base, same template) the template is rendered with placeholders for every slot that carries
a fact (entities, metric, scope, numbers, comparison/superlative/change words). The model rewrites that placeholder
sentence; each sibling's claim is then obtained by filling the placeholders with its own slot values, so the
SUPPORTS / REFUTES / NEI siblings of one fact keep sharing their wording.

A candidate rewrite is accepted only if
  * every placeholder occurs exactly as often as in the template, and no other bracket, Latin letter, digit,
    negation or missingness word is introduced;
  * filling it reproduces a fluent claim of bounded length for every sibling;
  * an independent judge call confirms, for every sibling, that the rewrite states exactly the same fact as the
    template claim (same entities, scope, metric, direction, rank order and numbers; nothing added or removed).
Otherwise the group keeps its template claim. All model outputs are cached (``runs/cache/naturalize.json``), so a
build with the cache is deterministic; ``python -m dart_fact naturalize`` fills the cache.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any

from . import realize as R
from .source_adapters import comparison_words

PROMPT_VERSION = "slot_locked_paraphrase_v4"
REWRITE_MODEL = "deepseek-v4-pro"
JUDGE_MODEL = "deepseek-flash"
BASE_URL = "https://api.deepseek.com/v1"

TEXT_SLOTS = ("scope", "e", "e1", "e2", "claimed", "member", "metric_name", "metric2", "cat", "attr", "k", "n", "y1", "y2")
NUM_SLOTS = ("value", "diff", "threshold", "total")
ALWAYS_LOCKED = ("e", "e1", "e2", "claimed", "member", "cat", "attr", "k", "n", "y1", "y2") + NUM_SLOTS
TOKEN_RE = re.compile(r"〔([a-z0-9_]+)〕")
NEGATION_RE = re.compile(r"[不没未非无否]")
LEAK_RE = re.compile(r"某|缺失|不存在|无法|证据|未知|NEI|null|表格|表中|表\d|列\d|数据显示|根据|据统计|资料")

REWRITE_SYSTEM = (
    "你是中文体育新闻与百科编辑。把一句机械、模板化的中文事实陈述改写成地道、简洁、像真人写的一句话。\n"
    "句中形如〔e1〕〔diff〕〔cmp〕的记号是占位符，会被填入名称、数字或比较词，示例取值只帮助理解。\n"
    "硬性要求：\n"
    "1. 每个占位符原样保留，出现次数与原句相同；不得修改、拆开、加引号，也不得把示例取值写进句子。\n"
    "2. 事实含义完全不变：统计范围（赛事、年份、地区、类别）、指标、比较方向、最高/最低、排名方向与名次、增减方向都不能变。\n"
    "3. 不得添加或删除信息：不新增名称、数字、年份、单位、原因、评价、否定词或程度词（如“仅”“远超”“约”）。\n"
    "4. 要真正改写：去掉“在……中”“就……而言”“……数据”“按从高到低”这类模板腔，统计范围放在句首或自然地嵌入，不要放在句末；"
    "可用通行的中文简称（如“中超”“中甲”“冬奥会”），可把“胜场数多于”说成“赢的场次比……多”。\n"
    "5. 只用简体中文，不出现英文字母、括注，不提“表格”“数据显示”“根据”，不提信息是否充分。\n"
    "6. 原句有两个并列断言时（如“主教练是……，并且进球数为……”），改写后仍是两个并列分句，不得把属性和数值合成一个动作（如“在某球场踢了N场”）。\n"
    "示例（仅示范风格）：\n"
    "  原句：在2016年中国足球超级联赛中，〔e1〕的胜场数多于〔e2〕。 改写：2016赛季中超，〔e1〕赢下的比赛比〔e2〕多。\n"
    "  原句：在河南各地级市地区生产总值数据中，GDP最高的城市是〔claimed〕。 改写：河南各地级市中，GDP最高的是〔claimed〕。\n"
    "  原句：在中国足球超级联赛中，与〔y1〕相比，〔e〕〔y2〕的平局场数〔verb〕了〔diff〕。 改写：〔e〕在中超的平局场数，〔y2〕比〔y1〕〔verb〕了〔diff〕。\n"
    "  原句：在2021年世界乒乓球锦标赛中，〔e〕的金牌数按从高到低排在第〔k〕位。 改写：2021年世乒赛上，〔e〕的金牌数排名第〔k〕。\n"
    '只输出 JSON：{"candidates": ["改写1", "改写2", "改写3"]}，按自然程度从高到低排列，三句措辞彼此不同。'
)
JUDGE_SYSTEM = (
    "你是严格的中文语义审校。给你一条“标准断言”（由程序生成，含义精确）和一句中文句子。"
    "判断句子的字面意思是否恰好表达这条断言，不多也不少。逐项核对：\n"
    "(1) 归属：每个数值、属性、排名属于的对象与断言相同（例如“球队的比赛场次”不能变成“球场举办的比赛”）；\n"
    "(2) 范围：赛事、项目、年份、地区、类别与断言相同，不变宽、不变窄、不产生歧义"
    "（如“世界羽毛球锦标赛”不能写成泛指的“世锦赛”；通行且无歧义的简称如“中超”“欧洲杯”可以）；\n"
    "(3) 指标、关系（大于/小于/相等/最大/最小/名次/增减）与数字相同；\n"
    "(4) 句子没有额外断言（例如另外声称某对象“最高”、排名、总数或原因），也没有遗漏断言的任何部分。\n"
    '只输出 JSON：{"attribution_ok": bool, "scope_ok": bool, "relation_ok": bool, "no_extra_or_missing": bool, '
    '"same": bool, "reason": "不超过30字"}。四项都为 true 时 same 才为 true。'
)


def gloss(skeleton: str, s: dict[str, Any]) -> str:
    """Unambiguous statement of what the program asserts (used only as the judge's reference)."""
    sc, m, noun = s.get("scope", ""), s.get("metric_name", ""), s.get("noun", "对象")
    ext = {"max": "最大", "min": "最小"}.get(s.get("mode", "max"), "最大")
    num = lambda k: R._num(s[k])  # noqa: E731
    if skeleton in ("ST_COMPARE", "XT_COMPARE"):
        rel = {"gt": "大于", "lt": "小于", "eq": "等于"}[s["cmp"]]
        return f"在{sc}中，{s['e1']}的{m}数值{rel}{s['e2']}的{m}数值。"
    if skeleton == "ST_SUPERLATIVE":
        return f"在{sc}列出的所有{noun}中，{m}数值{ext}的是{s['claimed']}。"
    if skeleton == "ST_RANK":
        return f"在{sc}列出的所有{noun}中，按{m}数值从大到小排序，{s['e']}排在第{s['k']}位。"
    if skeleton in ("ST_DIFF", "XT_DIFF"):
        return f"在{sc}中，{s['e1']}的{m}数值减去{s['e2']}的{m}数值等于{num('diff')}（前者比后者大{num('diff')}）。"
    if skeleton == "CAT_ARGEXT":
        return f"在{sc}中，{s['cat_col']}为{s['cat']}的所有{noun}里，{m}数值{ext}的是{s['claimed']}。"
    if skeleton == "CAT_COUNT":
        return f"在{sc}中，{s['cat_col']}为{s['cat']}的{noun}共有{s['n']}个。"
    if skeleton == "RJ_RANK_TOP":
        return f"在{sc}中，排名第{s['k']}位的{s['rank_noun']}的所有{noun}里，{m}数值最大的是{s['claimed']}。"
    if skeleton == "RJ_RANK_TOPVAL":
        return (f"在{sc}中，排名第{s['k']}位的{s['rank_noun']}的所有{noun}里，"
                f"{m}数值最大的那个数值是{num('value')}。")
    if skeleton == "RJ_MEMBER_RANK":
        return f"在{sc}中，{s['member']}所属的{s['rank_noun']}排名第{s['k']}位。"
    if skeleton == "ST_COUNT_GT":
        return f"在{sc}列出的所有{noun}中，{m}数值超过{num('threshold')}的共有{s['n']}个。"
    if skeleton == "CAT_SUM":
        return (f"在{sc}中，{s['cat_col']}为{s['cat']}的所有{noun}的{m}数值相加，"
                f"总和等于{num('total')}。")
    if skeleton == "JF_SUM":
        return (f"在{sc}中，{s['attr_col']}为{s['attr']}的所有{noun}的{m}数值相加，"
                f"总和等于{num('total')}。")
    if skeleton == "HP_AND":
        return (f"在{sc}中，同时断言两件事：①{s['e']}的{s['attr_col']}是{s['attr']}；"
                f"②{s['e']}本身的{m}是{num('value')}。")
    if skeleton == "HP_CMP":
        rel = "大于" if s["cmp"] == "gt" else "小于"
        return (f"在{sc}中，同时断言两件事：①{s['e1']}的{s['attr_col']}是{s['attr']}；"
                f"②{s['e1']}的{m}数值{rel}{s['e2']}的{m}数值。")
    if skeleton == "CAT_PARTNER":
        rel = "大于" if s["cmp"] == "gt" else "小于"
        return (f"在{sc}中，先取出{s['cat_col']}为{s['cat']}的全部{noun}，其中{m}数值{ext}的是{s['claimed']}；"
                f"并且{s['claimed']}的{s['metric2']}数值{rel}{s['e2']}的{s['metric2']}数值。")
    if skeleton == "JF_ARGEXT":
        return f"在{sc}中，{s['attr_col']}为{s['attr']}的所有{noun}里，{m}数值{ext}的是{s['claimed']}。"
    if skeleton == "JF_COUNT_GT":
        return f"在{sc}中，{s['attr_col']}为{s['attr']}的{noun}里，{m}超过{num('threshold')}的共有{s['n']}个。"
    if skeleton == "CP_DIRECTION":
        rel = "大于" if s["direction"] == "up" else "小于"
        return f"在{sc}中，{s['e']}的{m}在{R._year(s['y2'])}的数值{rel}{R._year(s['y1'])}的数值。"
    if skeleton == "CP_DIFF":
        return f"在{sc}中，{s['e']}的{m}在{R._year(s['y2'])}比{R._year(s['y1'])}{R._verb(s['direction'])}了{num('diff')}。"
    raise KeyError(skeleton)


RETRY_NOTE = (
    "\n上一次的改写没有通过检查。请特别注意：不要使用英文字母或英文缩写（如CBA、GDP以外的缩写）；"
    "不要增删“最”“不”“没”等字；赛事、项目、年份写全称或无歧义的中文简称（不要写泛指的“世锦赛”）；"
    "占位符一个都不能少；主语与数值的归属不能改变。"
)


# ----------------------------------------------------------------------------- placeholder rendering
def _patched_realize(skeleton: str, slots: dict[str, Any], base_id: str) -> str:
    ph = dict(slots)
    for k in TEXT_SLOTS:
        if k in ph:
            ph[k] = f"〔{k}〕"
    for k in NUM_SLOTS:
        if isinstance(ph.get(k), dict):
            ph[k] = {"text": f"〔{k}〕", "unit": ""}
    saved = (R._cmp, R._verb, R._more)
    R._cmp = lambda style, c: "〔cmp〕" if c in ("gt", "lt") else "〔mode〕" if c in ("max", "min") else comparison_words(style)[c]
    R._verb = lambda d: "〔verb〕"
    R._more = lambda style, d="up": "〔more〕"
    try:
        return R.realize(skeleton, ph, base_id)
    finally:
        R._cmp, R._verb, R._more = saved


def slot_values(skeleton: str, slots: dict[str, Any]) -> dict[str, str]:
    style = slots.get("style", "amount")
    words = comparison_words(style)
    out = {k: str(slots[k]) for k in TEXT_SLOTS if k in slots}
    for k in ("y1", "y2"):
        if k in slots:
            out[k] = R._year(slots[k])
    out.update({k: R._num(slots[k]) for k in NUM_SLOTS if isinstance(slots.get(k), dict)})
    if slots.get("cmp") in ("gt", "lt"):
        out["cmp"] = words[slots["cmp"]]
    out["mode"] = words[slots.get("mode", "max")]
    out["verb"] = R._verb(slots.get("direction", "up"))
    out["more"] = R._more(style, slots.get("direction", "up"))
    return out


def fill(template: str, values: dict[str, str]) -> str:
    return TOKEN_RE.sub(lambda m: values[m.group(1)], template)


def lock_template(template: str, sibling_values: list[dict[str, str]]) -> str:
    """Keep placeholders for entities, numbers and every slot whose value differs between siblings; write the
    shared value of all other slots (scope, metric, comparison word) into the sentence the model may rephrase."""
    def sub(m: re.Match) -> str:
        k = m.group(1)
        vals = {v.get(k) for v in sibling_values}
        return m.group(0) if k in ALWAYS_LOCKED or len(vals) > 1 else sibling_values[0][k]
    return TOKEN_RE.sub(sub, template)


def placeholder_template(skeleton: str, slots: dict[str, Any], base_id: str) -> str | None:
    """Placeholder sentence, or None if filling it does not reproduce the template claim exactly."""
    tpl = _patched_realize(skeleton, slots, base_id)
    try:
        ok = fill(tpl, slot_values(skeleton, slots)) == R.realize(skeleton, slots, base_id)
    except KeyError:
        ok = False
    return tpl if ok else None


# ----------------------------------------------------------------------------- candidate checks
def _outside_tokens(text: str) -> str:
    return TOKEN_RE.sub("", text)


def surface_ok(template: str, cand: str, sibling_values: list[dict[str, str]]) -> str:
    """Return '' if the candidate passes the surface checks, else the failed check."""
    if sorted(TOKEN_RE.findall(cand)) != sorted(TOKEN_RE.findall(template)):
        return "placeholders"
    rest, trest = _outside_tokens(cand), _outside_tokens(template)
    if re.search(r"[〔〕\[\]【】{}<>]", rest):
        return "brackets"
    if sorted(re.findall(r"[A-Za-z]+", rest)) != sorted(re.findall(r"[A-Za-z]+", trest)):
        return "latin"
    if sorted(re.findall(r"\d", rest)) != sorted(re.findall(r"\d", trest)):
        return "digits"
    if len(NEGATION_RE.findall(rest)) != len(NEGATION_RE.findall(trest)):
        return "negation"
    if trest.count("最") != rest.count("最"):
        return "superlative_word"
    if re.search(r"并且|且|同时", trest) and not re.search(r"并且|且|同时|，", rest):
        return "conjunction"
    for vals in sibling_values:
        claim = fill(cand, vals)
        if LEAK_RE.search(_outside_tokens(cand)) or len(claim) > 100 or not claim.endswith("。"):
            return "leak_or_length"
    return ""


# ----------------------------------------------------------------------------- API with cache
class Cache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        with self.lock:
            self.data[key] = value

    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)


def _key(*parts: Any) -> str:
    return hashlib.sha1(json.dumps([PROMPT_VERSION, *parts], ensure_ascii=False).encode("utf-8")).hexdigest()


def _call(model: str, system: str, user: str, max_tokens: int) -> dict[str, Any] | None:
    import requests
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        # Without this the run looks successful: every request goes out with an empty bearer token, is
        # rejected, is retried four times, and the empty result is written to the cache - so a later run
        # with a valid key reads the poisoned entry and still produces nothing.
        raise RuntimeError("DEEPSEEK_API_KEY is not set; refusing to run the paraphraser with no key")
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "max_tokens": max_tokens, "temperature": 0.0, "response_format": {"type": "json_object"},
               "thinking": {"type": "disabled"}}
    for attempt in range(4):
        try:
            r = requests.post(BASE_URL + "/chat/completions", json=payload, timeout=120,
                              headers={"Authorization": f"Bearer {key}"})
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + random.random())
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception:  # noqa: BLE001 - network / JSON errors are retried, then treated as no output
            time.sleep(2 ** attempt + random.random())
    return None


def rewrite_request(template: str, example: dict[str, str]) -> str:
    shown = {f"〔{k}〕": v for k, v in example.items() if f"〔{k}〕" in template}
    return f"原句：{template}\n占位符示例取值：{json.dumps(shown, ensure_ascii=False)}"


def judge_request(reference: str, sentence: str) -> str:
    return f"标准断言：{reference}\n句子：{sentence}"


# ----------------------------------------------------------------------------- main entry
def groups_of(items: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for i, item in enumerate(items):
        d = item["draft"]
        tpl = placeholder_template(d.skeleton, d.slots, d.base_id)
        if tpl is None or "〔" not in tpl:
            continue
        groups.setdefault((d.base_id, tpl), []).append(i)
    return groups


def naturalize(items: list[dict[str, Any]], cache_path: str | Path, *, call_api: bool = False,
               workers: int = 16, limit: int | None = None) -> dict[str, Any]:
    """Replace item['claim'] by an accepted paraphrase (in place); returns counts. Without call_api only cached
    model outputs are used, so the result is deterministic."""
    cache = Cache(cache_path)
    groups = groups_of(items)
    if limit is not None:
        groups = dict(list(groups.items())[:: max(1, len(groups) // limit)][:limit])
    stats: dict[str, Any] = {"groups": len(groups), "paraphrased_groups": 0, "template_groups": 0}

    def work(gkey: tuple[str, str]) -> tuple[tuple[str, str], str | None, str]:
        _, tpl = gkey
        idx = groups[gkey]
        drafts = [items[i]["draft"] for i in idx]
        values = [slot_values(d.skeleton, d.slots) for d in drafts]
        example = values[next((j for j, d in enumerate(drafts) if d.label == "SUPPORTS"), 0)]
        tpl = lock_template(tpl, values)
        rkey = _key("rewrite", REWRITE_MODEL, tpl, example)
        out = cache.get(rkey)
        if out is None and call_api:
            out = _call(REWRITE_MODEL, REWRITE_SYSTEM, rewrite_request(tpl, example), 600)
            if out is None:      # transport failure, not an empty answer: never cache it
                return gkey, None, "api_unavailable"
            cache.put(rkey, out)
        def try_candidates(output: Any, reasons: list[str]) -> str | None:
            for cand in [c.strip() for c in (output or {}).get("candidates", []) if isinstance(c, str)]:
                fail = surface_ok(tpl, cand, values)
                if fail:
                    reasons.append(fail)
                    continue
                same_all = True
                for d, vals in zip(drafts, values):
                    a, b = gloss(d.skeleton, d.slots), fill(cand, vals)
                    jkey = _key("judge", JUDGE_MODEL, a, b)
                    verdict = cache.get(jkey)
                    if verdict is None and call_api:
                        verdict = _call(JUDGE_MODEL, JUDGE_SYSTEM, judge_request(a, b), 200) or {"same": False, "reason": "no_output"}
                        cache.put(jkey, verdict)
                    if not (isinstance(verdict, dict) and verdict.get("same") is True):
                        same_all = False
                        break
                if same_all:
                    return cand
                reasons.append("judge")
            return None

        reasons: list[str] = []
        cand = try_candidates(out, reasons)
        if cand is None and out is not None:
            # one retry with the failed checks spelled out
            rkey2 = _key("rewrite_retry", REWRITE_MODEL, tpl, example)
            out2 = cache.get(rkey2)
            if out2 is None and call_api:
                out2 = _call(REWRITE_MODEL, REWRITE_SYSTEM, rewrite_request(tpl, example) + RETRY_NOTE, 600)
                if out2 is None:
                    return gkey, None, "api_unavailable"
                cache.put(rkey2, out2)
            cand = try_candidates(out2, reasons)
        if cand is not None:
            return gkey, cand, "ok"
        return gkey, None, ",".join(reasons) or "no_candidates"

    results: dict[tuple[str, str], tuple[str | None, str]] = {}
    if call_api:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for n, (gkey, cand, why) in enumerate(pool.map(work, list(groups)), 1):
                results[gkey] = (cand, why)
                if n % 100 == 0:
                    cache.save()
                    print(f"naturalize {n}/{len(groups)}", flush=True)
        cache.save()
    else:
        results = {g: work(g)[1:] for g in groups}
    reasons: dict[str, int] = {}
    for gkey, (cand, why) in results.items():
        vals_idx = groups[gkey]
        if cand is None:
            stats["template_groups"] += 1
            reasons[why] = reasons.get(why, 0) + 1
            for i in vals_idx:
                items[i]["surface"] = "template"
            continue
        stats["paraphrased_groups"] += 1
        for i in vals_idx:
            d = items[i]["draft"]
            items[i]["template_claim"] = items[i]["claim"]
            items[i]["claim"] = fill(cand, slot_values(d.skeleton, d.slots))
            items[i]["surface"] = "llm_paraphrase"
    stats["fallback_reasons"] = reasons
    return stats
