"""Register cached-but-unused pages into the page registry.

`build_packages` walks `SERIES` and never looks at a page that is not registered there, so a page that
discovery fetched but whose SERIES block was never pasted into `source_adapters.py` is invisible to the
whole pipeline.  787 pages are cached and only 380 produce evidence packages; ~453 of the rest are
simply unregistered, among them pages with 20-48 usable tables (`每年度入选世界遗产列表`,
`各国最高建筑物列表`, `北京地铁车站列表`, `全球机场货运量列表`).  That is the cheapest supply the
reproduction has left: distinct topics, raw tables and non-football sources all come from it.

This classifies each unregistered page by title, judges it by actually building packages for it (the
registry is injected temporarily, the same trick `discover_jf2.py` needs), and prints a SERIES block
for the pages that yield packages.  It does not edit `source_adapters.py`: the series names are an
editorial choice about how the corpus is described in the paper.

    PYTHONPATH=src python tools/register_cached.py --cache runs/cache/pages
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import pathlib
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dart_fact import source_adapters  # noqa: E402
from dart_fact.acquire import to_simplified  # noqa: E402
from dart_fact.packages import build_packages  # noqa: E402
from dart_fact.source_adapters import SENSITIVE_TITLE_RE, SERIES  # noqa: E402

# (pattern, domain, series).  Ordered: the first match wins, so put the specific before the generic.
# Domains reuse the sixteen already in the registry; a page that matches nothing is reported but not
# registered, because an unclassified page would land in a domain cap it does not belong to.
CLASSIFY: list[tuple[str, str, str]] = [
    (r"世界遗产", "culture", "世界遗产"),
    (r"博物馆|美术馆|图书馆", "culture", "文化设施"),
    (r"国家公园|风景名胜|自然保护区|森林公园", "geography", "保护地"),
    (r"最高建筑|摩天|大楼|大厦|建筑物", "architecture", "高层建筑"),
    (r"桥梁|大桥|隧道|桥隧", "infrastructure", "桥隧"),
    (r"景区|风景区|旅游景点|名胜", "culture", "旅游景区"),
    (r"屋苑|住宅|楼盘|小区", "architecture", "住宅"),
    (r"电台|广播|电视台", "culture", "广播电视"),
    (r"证券交易所|交易所|基金|股票", "economy", "金融市场"),
    (r"电车|轻轨", "transport", "轨道交通"),
    (r"长河|水系", "geography", "自然地理"),
    (r"学校列表|院校", "education", "学校"),
    (r"旗帜|徽章", "culture", "国家象征"),
    (r"水库|水电站|大坝", "energy", "水利"),
    (r"核电|电站|发电|装机|风电|光伏", "energy", "电力"),
    (r"机场|航空|航线", "transport", "航空"),
    (r"地铁|轨道交通|车站|铁路|高铁|线路", "transport", "轨道交通"),
    (r"港口|吞吐量|集装箱", "transport", "港口"),
    (r"高速公路|公路", "infrastructure", "公路"),
    (r"大学|高校|学院|双一流|985|211", "education", "高校"),
    (r"票房|电影|影片|影院", "film", "票房"),
    (r"电视|收视|综艺", "culture", "电视"),
    (r"企业|公司|集团|银行|市值|财富|500强", "economy", "企业"),
    (r"生产总值|GDP|人均|收入|经济|财政|税收|贸易|出口|进口", "economy", "宏观经济"),
    (r"人口|人口密度|城市化|出生|生育|年龄", "demographics", "人口"),
    (r"行政区|省级|地级|县级|直辖市|自治区|首都|前首都", "demographics", "行政区划"),
    (r"语言|官方语言|文字", "culture", "语言"),
    (r"河流|湖泊|岛屿|山峰|海拔|面积|地形|沙漠|冰川", "geography", "自然地理"),
    (r"指数|排名|排行", "world_stats", "国际指数"),
    (r"国旗|国徽|国歌", "culture", "国家象征"),
    (r"专利|科研|论文|研究|科学|太空|火箭|卫星", "science", "科技"),
    (r"奥林匹克|奥运|亚运|全运|奖牌", "medals", "综合运动会"),
    (r"篮球|CBA|NBA", "basketball", "篮球"),
    (r"足球|联赛|世界杯|欧洲杯|亚洲杯", "football", "足球"),
    (r"各国|世界|全球|国家和地区", "world_stats", "国家指标"),
]


# Titles that a CLASSIFY pattern would otherwise pull in but that sit next to the exclusions the paper
# already applies (military, ethnic, religious, political).  Kept here rather than widened into
# SENSITIVE_TITLE_RE, because broadening the project's policy regex is an author's call, not a
# side effect of a registration pass.
SKIP_RE = re.compile(r"警察|公安|武警|国防|情报|侦察|社会主义|首辅|宇航员|航天局")


def classify(title: str) -> tuple[str, str] | None:
    if SKIP_RE.search(title):
        return None
    for pattern, domain, series in CLASSIFY:
        if re.search(pattern, title):
            return domain, series
    return None


@contextlib.contextmanager
def registered(title: str, domain: str, series_name: str):
    entry = {"series": series_name, "domain": domain, "titles": [title]}
    source_adapters.SERIES.append(entry)
    try:
        yield
    finally:
        source_adapters.SERIES.remove(entry)


def packages_for(title: str, path: pathlib.Path, domain: str, series_name: str) -> collections.Counter:
    """Build packages for one page in isolation, so a page cannot be credited with a shape that only
    exists because another page supplied the second table."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="reg-"))
    try:
        shutil.copy2(path, tmp / path.name)
        with registered(title, domain, series_name):
            return collections.Counter(p["kind"] for p in build_packages(str(tmp), cache_only=True))
    except Exception:  # noqa: BLE001
        return collections.Counter()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/cache/pages", type=pathlib.Path)
    ap.add_argument("--packages", default="runs/repro_final2/packages.jsonl", type=pathlib.Path,
                    help="packages of the current build; pages appearing here are already in use")
    ap.add_argument("--min-packages", type=int, default=1)
    ap.add_argument("--out", default="runs/logs/register_cached.json", type=pathlib.Path)
    args = ap.parse_args()

    known = {to_simplified(t) for s in SERIES for t in s["titles"]}
    in_use = {to_simplified(json.loads(line)["page_title"])
              for line in args.packages.open(encoding="utf-8")} if args.packages.exists() else set()

    accepted: list[dict] = []
    stat = collections.Counter()
    for path in sorted(args.cache.glob("*.json")):
        try:
            title = to_simplified(json.loads(path.read_text(encoding="utf-8")).get("title", ""))
        except Exception:  # noqa: BLE001
            stat["unreadable"] += 1
            continue
        if not title:
            stat["no title"] += 1
            continue
        if title in known or title in in_use:
            stat["already registered or in use"] += 1
            continue
        if SENSITIVE_TITLE_RE.search(title):
            stat["excluded as sensitive"] += 1
            continue
        cls = classify(title)
        if cls is None:
            stat["unclassified"] += 1
            print(f"  ? unclassified: {title}", file=sys.stderr)
            continue
        domain, series_name = cls
        kinds = packages_for(title, path, domain, series_name)
        if sum(kinds.values()) < args.min_packages:
            stat["yields no package"] += 1
            continue
        stat["accepted"] += 1
        accepted.append({"title": title, "domain": domain, "series": series_name,
                         "packages": sum(kinds.values()), "kinds": dict(kinds)})
        print(f"    + {title[:34]:<36}{sum(kinds.values()):>5} pkgs  {dict(kinds)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(accepted, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    for k, n in stat.most_common():
        print(f"  {n:5d}  {k}")
    print(f"\n  new packages unlocked: {sum(a['packages'] for a in accepted)}")
    kinds = collections.Counter()
    for a in accepted:
        kinds.update(a["kinds"])
    print(f"  by kind: {dict(kinds.most_common())}")
    print(f"  by domain: {dict(collections.Counter(a['domain'] for a in accepted).most_common())}")

    print("\n    # ---- cached pages registered by tools/register_cached.py")
    by_series: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for a in accepted:
        by_series[(a["series"], a["domain"])].append(a["title"])
    for (series_name, domain), titles in sorted(by_series.items()):
        items = ", ".join(f'"{t}"' for t in sorted(titles))
        print(f'    {{"series": "{series_name}", "domain": "{domain}", "titles": [{items}]}},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
