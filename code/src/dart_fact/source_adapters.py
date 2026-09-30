"""Source registry: which Chinese Wikipedia pages feed the benchmark, and domain lexicons.

Pages are grouped into *series* (e.g. one league across seasons) so that cross-period packages can
pair consecutive editions. The lexicon maps terse wiki headers (得/失/分…) to readable metric names,
picks comparison words (多于/高于), and lists plausible-but-absent metrics per domain for NEI.
Politically sensitive and casualty-related pages are deliberately not included.
"""
from __future__ import annotations

import re
from typing import Any


def _years(a: int, b: int, step: int = 1) -> list[int]:
    return list(range(a, b + 1, step))


SERIES: list[dict[str, Any]] = [
    # ---- sports: football leagues (season pages share schema across years)
    {"series": "中超", "domain": "football", "titles": [f"{y}年中国足球超级联赛" for y in _years(2012, 2024)]},
    {"series": "中甲", "domain": "football", "titles": [f"{y}年中国足球甲级联赛" for y in _years(2014, 2024)]},
    {"series": "中乙", "domain": "football", "titles": [f"{y}年中国足球乙级联赛" for y in _years(2017, 2024)]},
    {"series": "世界杯小组赛", "domain": "football_intl",
     "titles": [f"{y}年国际足联世界杯小组赛" for y in (2002, 2006, 2010, 2014, 2018, 2022)]
     + [f"{y}年国际足联世界杯" for y in (1998, 2002, 2006, 2010, 2014, 2018, 2022)]},
    {"series": "欧洲杯", "domain": "football_intl",
     "titles": [f"{y}年欧洲足球锦标赛" for y in (2004, 2008, 2012, 2016, 2020, 2024)]},
    {"series": "亚洲杯", "domain": "football_intl",
     "titles": [f"{y}年亚洲杯足球赛" for y in (2004, 2007, 2011, 2015, 2019, 2023)]},
    # ---- sports: basketball
    {"series": "CBA", "domain": "basketball",
     "titles": [f"{y}年至{y + 1}年中国男子篮球职业联赛" for y in _years(2012, 2024)]},
    {"series": "NBA", "domain": "basketball",
     "titles": [f"{y}–{str(y + 1)[2:]}赛季NBA" for y in _years(2012, 2023)]},
    # ---- multi-sport medal tables
    {"series": "夏奥奖牌榜", "domain": "medals",
     "titles": [f"{y}年夏季奥林匹克运动会奖牌榜" for y in _years(1984, 2024, 4)]},
    {"series": "冬奥奖牌榜", "domain": "medals",
     "titles": [f"{y}年冬季奥林匹克运动会奖牌榜" for y in (1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022)]},
    {"series": "亚运奖牌榜", "domain": "medals",
     "titles": [f"{y}年亚洲运动会奖牌榜" for y in (1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022)]},
    {"series": "全运会奖牌榜", "domain": "medals",
     "titles": [f"中华人民共和国第{n}届运动会奖牌榜" for n in ("九", "十", "十一", "十二", "十三", "十四")]},
    {"series": "世锦赛羽毛球", "domain": "medals", "titles": [f"{y}年世界羽毛球锦标赛" for y in _years(2005, 2023)]},
    {"series": "世乒赛", "domain": "medals", "titles": [f"{y}年世界乒乓球锦标赛" for y in _years(2005, 2024)]},
    # ---- economy & geography (China)
    {"series": "省级GDP", "domain": "economy", "titles": [
        "中华人民共和国省级行政区地区生产总值列表", "中华人民共和国省级行政区人均地区生产总值列表",
        "中华人民共和国各省级行政区人口列表", "中华人民共和国省级行政区土地面积列表",
        "中华人民共和国省级行政区人类发展指数列表", "中华人民共和国主要城市地区生产总值列表",
        "中国地级市人均地区生产总值列表", "中国城市人口列表", "中华人民共和国城市人口列表"]},
    {"series": "机场", "domain": "transport", "titles": [
        "中华人民共和国机场旅客吞吐量列表", "中华人民共和国机场货邮吞吐量列表",
        "中国大陆机场旅客吞吐量排名", "世界最繁忙机场列表", "世界最繁忙货运机场列表"]},
    {"series": "建筑", "domain": "architecture", "titles": [
        "中国最高建筑列表", "世界最高建筑列表", "世界最高电视塔列表", "中国大陆最高建筑列表", "香港最高建筑物列表"]},
    {"series": "地理", "domain": "geography", "titles": [
        "中国河流列表", "中国湖泊列表", "世界最长河流列表", "世界最大湖泊列表", "中国岛屿列表",
        "各国面积列表", "各国人口列表", "各国人口密度列表", "各国国内生产总值列表", "各国人均国内生产总值列表",
        "日本都道府县人口列表", "日本都道府县面积列表", "美国各州人口列表", "美国各州面积列表",
        "韩国行政区划人口列表", "印度各邦人口列表"]},
    {"series": "电影", "domain": "film", "titles": [
        "中国内地最高电影票房收入列表", "香港最高電影票房收入列表", "南韓最高電影票房收入列表",
        "美國和加拿大最高電影票房收入列表", "印度最高電影票房收入列表", "全球最高跨媒體製作收入列表"]},
    # (全球最高電影票房收入導演列表 / 最高電影票房演員列表 removed: their tables report North-American box office)
    # ---- added 2026-09-13 from API search (tools: runs/logs/discover.json)
    {"series": "地级市GDP", "domain": "economy", "titles": [
        f"{p}各地级市地区生产总值列表" for p in ("广东", "江苏", "辽宁", "河南", "山东", "安徽", "河北", "陕西", "山西", "广西", "福建")]
        + ["湖南各市州地区生产总值列表", "中國地級市人均地區生產總值列表", "中国省级行政区人均可支配收入列表"]},
    {"series": "中国城市人口", "domain": "economy", "titles": [
        "中華人民共和國城市人口排名", "中華人民共和國城市城區常住人口排名", "中国城市凝聚区人口排名"]},
    {"series": "世界城市人口", "domain": "world_stats", "titles": [
        "東亞城市人口列表", "世界城市地區人口排序列表", "世界城市市域人口排序列表", "歐洲城市人口列表",
        "美國城市人口排序列表", "韓國城市人口排序列表", "按人口排列的加拿大城市列表", "澳大利亚城市人口列表",
        "美国人口最多县份列表"]},
    {"series": "各国统计", "domain": "world_stats", "titles": [
        "国家陆地面积列表", "各國世界遺產數列表", "各国家和地区人口密度列表", "各国家和地区人口列表",
        "各国名义国内生产总值列表", "各国人均名义国内生产总值列表", "各国人均实质国内生产总值列表", "各国森林面积列表",
        "各国出口额列表", "各国家和地区生育率列表", "歐洲國家面積列表", "歐洲國家和地區列表",
        "亞洲國家人口列表", "非洲国家人口列表", "美洲国家人口列表", "欧洲联盟国家人口列表",
        "面积超过40万平方公里的世界各国一级行政区列表", "德國各州人口列表", "美國各州人口密度列表", "印度行政区人口列表"]},
    {"series": "历史国家人口", "domain": "world_stats", "titles": ["1913年國家人口列表", "1930年國家人口列表", "1950年國家人口列表"]},
    {"series": "山川湖泊", "domain": "geography", "titles": [
        "世界高山列表", "八千米以上山峰列表", "世界长河列表", "日本湖泊列表", "湖泊深度列表", "安徽省自然保护区列表"]},
    {"series": "交通能源", "domain": "infrastructure", "titles": [
        "地鐵列表", "中運量系統列表", "全球機場客運量列表", "中华人民共和国吞吐量最大港口列表", "世界貨櫃吞吐量最大港口列表",
        "世界貨物吞吐量最大港口列表", "中国最高水坝列表", "世界水电站列表", "中华人民共和国水电站列表"]},
    {"series": "摩天大楼", "domain": "infrastructure", "titles": [
        "中华人民共和国摩天大楼列表", "上海市摩天大樓列表", "北京摩天大樓列表", "深圳摩天大楼列表", "最高建築及結構物列表"]},
    # ---- added 2026-09-16 for the reproduction build (tools/discover_pages.py -> runs/logs/seed_candidates.json)
    {"series": "媒体", "domain": "culture", "titles": ["世界参观人数最多的美术馆列表", "广东省博物馆列表", "按区域划分参观最多的博物馆列表", "英国参观人数最多的博物馆列表"]},
    {"series": "世界人口", "domain": "demographics", "titles": ["人口老龄化", "人类发展指数", "各国人口预期寿命列表", "各国人均名目国民总所得列表"]},
    {"series": "中国人口", "domain": "demographics", "titles": ["中华人民共和国各省级行政区总和生育率表", "中华人民共和国各省级行政区预期寿命列表"]},
    {"series": "产业", "domain": "economy", "titles": ["上海市", "天津市经济", "财富世界500强"]},
    {"series": "地区经济", "domain": "economy", "titles": ["云南省经济", "各国储蓄率列表", "安徽各地级市地区各项经济指标列表", "广东省经济", "江西各地级市地区生产总值列表", "浙江各地级市地区生产总值列表", "浙江省", "湖南各县市地区生产总值列表", "西班牙最高电影票房收入列表"]},
    {"series": "贸易", "domain": "economy", "titles": ["世界各国和地区面积列表", "各国人均实质国民总所得列表", "各国实质国内生产总值列表", "各国死刑列表"]},
    {"series": "金融", "domain": "economy", "titles": ["上海证券交易所综合股价指数", "沪深300"]},
    {"series": "高校", "domain": "education", "titles": ["ESI大学排名", "中华人民共和国教育部直属高等学校列表", "中国大陆高等学校列表", "国际乒乓球总会世界排名", "澳门高等院校列表"]},
    {"series": "票房", "domain": "film", "titles": ["义大利最高电影票房收入列表", "墨西哥最高电影票房收入列表", "巴西最高电影票房收入列表", "日本最高电影票房收入列表"]},
    {"series": "地形", "domain": "geography", "titles": ["各国土地灌溉面积列表", "山", "岛屿", "岛屿列表", "岛屿列表 (按密度)", "岛屿面积列表", "希腊岛屿列表", "浙江省岛屿列表", "美国岛屿面积列表", "香港岛屿列表"]},
    {"series": "水文", "domain": "geography", "titles": ["湖泊", "珠江", "陆地边界长度列表"]},
    {"series": "航天", "domain": "science", "titles": ["中国运载火箭发射列表", "印度运载火箭发射列表", "长征系列运载火箭", "长征系列运载火箭发射列表"]},
    {"series": "航空", "domain": "transport", "titles": ["伦敦的机场", "全球城市机场系统列表", "全球机场国际客量列表", "全球机场客运吞吐量列表 (2000-2009)", "全球机场客运吞吐量列表 (2010-2019)", "全球机场货运量列表", "北京大兴国际机场", "北京首都国际机场", "广州白云国际机场", "成都天府国际机场", "沈阳桃仙国际机场"]},
    {"series": "铁路", "domain": "transport", "titles": ["上海地铁", "上海地铁车站列表", "中华人民共和国铁路运输", "中华人民共和国高速铁路", "中华人民共和国高铁线路列表", "中国城市轨道交通系统", "中国铁路线路列表", "京广高速铁路", "京沪高速铁路", "佛山地铁", "北京地铁", "广州地铁车站列表", "慕尼黑地铁车站列表", "成都地铁", "新加坡地铁", "苏州地铁车站列表", "西班牙高速铁路"]},
    {"series": "指标", "domain": "world_stats", "titles": ["世界旅游排名", "世界遗产", "中华人民共和国国内生产总值", "各国4G LTE渗透率列表", "各国二氧化碳排放量列表", "各国互联网主机数列表", "各国互联网使用者数目列表", "各国各产业国内生产总值列表", "各国宽带互联网使用者数目列表", "各国智慧型手机普及率列表", "各国氧化铝产量列表", "各国温室气体排放量列表", "各国移民数列表", "欧洲国家互联网用户数列表", "温室气体排放"]},
    # ---- added 2026-09-16, shape-targeted pass (tools/discover_shapes.py): split_pair / category_decomp
    {"series": "两级行政区", "domain": "demographics", "titles": ["中华人民共和国地级行政区列表"]},
    {"series": "洲际", "domain": "demographics", "titles": ["北美洲国家和地区列表", "欧洲", "法语国家和地区列表", "英语国家和地区列表", "荷兰语国家和地区列表", "葡萄牙语国家和地区列表", "阿拉伯语国家和地区列表", "非洲"]},
    {"series": "两级行政区", "domain": "economy", "titles": ["云南省"]},
    {"series": "企业", "domain": "economy", "titles": ["中华人民共和国最大公司列表", "中国移动", "福布斯全球企业2000强"]},
    {"series": "分区", "domain": "education", "titles": ["世界大学学术排名", "武汉大学", "泰晤士高等教育世界大学排名"]},
    {"series": "分区", "domain": "energy", "titles": ["德国核能"]},
    {"series": "分区", "domain": "film", "titles": ["2010年韩国电影作品列表", "2012年韩国电影作品列表", "2013年韩国电影作品列表", "2014年韩国电影作品列表", "2016年韩国电影作品列表", "2017年韩国电影作品列表", "2020年韩国电影作品列表", "2022年韩国电影作品列表"]},
    {"series": "分区", "domain": "infrastructure", "titles": ["广州市摩天大楼列表", "摩天大楼", "武汉摩天大楼列表", "长江"]},
    {"series": "洲际", "domain": "world_stats", "titles": ["各国最高建筑物列表", "没有世界遗产的国家列表"]},
    # ---- added 2026-09-16, hub/join-filter targeted pass (tools/discover_hub.py): the hub-profile and
    # join-filter shapes existed almost only on sports pages, where an absent entity reads as "did not
    # take part" rather than as unknown, so those two families had no sound entity-NEI to draw on
    {"series": "国家指标", "domain": "world_stats", "titles": ["各国人类发展指数列表", "联合国地理区划列表", "阿拉伯世界"]},
    {"series": "建筑", "domain": "infrastructure", "titles": ["东南亚摩天大楼列表", "阿联酋摩天大楼列表"]},
    {"series": "水利", "domain": "infrastructure", "titles": ["中国大陆湖泊列表"]},
    {"series": "电力", "domain": "energy", "titles": ["发电站列表"]},
    {"series": "高校", "domain": "education", "titles": ["中华人民共和国教育", "普通高等学校招生全国统一考试安排列表"]},
    # ---- added 2026-09-17, topic-diversity pass (tools/discover_pages.py --topics)
    {"series": "世界人口", "domain": "demographics", "titles": ["各国人口年龄结构列表", "各国家和地区年龄中位数列表"]},
    {"series": "中国人口", "domain": "demographics", "titles": ["云南省县级行政区列表", "浙江省经济", "湖北省各市州人口列表"]},
    {"series": "产业", "domain": "economy", "titles": ["华为", "福特汽车", "联想集团", "通用汽车"]},
    {"series": "地区经济", "domain": "economy", "titles": ["中国医疗保障制度", "中国电信", "中国自由贸易试验区", "广东省", "长沙经济"]},
    {"series": "地形", "domain": "geography", "titles": ["海岸线", "荒漠面积列表"]},
    {"series": "基础教育", "domain": "education", "titles": ["世界经济自由度", "发展援助提供国列表"]},
    {"series": "媒体", "domain": "culture", "titles": ["世界百大报纸列表", "各国人均烟草消费量列表", "各国人均酒精消费量列表", "新世纪福音战士", "香港报纸列表"]},
    {"series": "建筑", "domain": "infrastructure", "titles": ["JJ20世界巡回演唱会", "中国大陆大型体育场列表", "杭州奥体中心体育馆", "欧洲体育场列表"]},
    {"series": "指标", "domain": "world_stats", "titles": ["中华人民共和国岛屿列表", "中国地理", "中国大陆最低工资标准", "优良国家指数", "各国人均肉类消费列表", "各国关税税率列表", "各国医生数量列表", "各国医院床位数量列表", "各国国债列表", "各国土地使用情况列表", "各国大麦产量列表", "各国捕捞与养殖水产品产量列表", "各国最低工资列表", "各国汽车产量列表", "各国淡水消耗量列表", "各国煤产量列表", "各国番茄产量列表", "各国粗粮产量列表", "各国苹果产量列表", "各国营养不良人口比例列表", "各国谷物产量列表", "各国钢产量列表", "各国铁矿石产量列表", "各国黄金产量列表", "国家住宅自有率列表", "大韩民国", "工资", "希腊", "新加坡医院列表", "森林", "社会进步指数", "粮食安全", "美国各州人均汽车拥有量列表", "美国各州研发支出列表", "美国州份和领地列表", "美国首府列表", "英国", "青岛市", "马来西亚"]},
    {"series": "桥隧", "domain": "infrastructure", "titles": ["世界公路隧道列表", "六横公路大桥", "香港跨海大桥列表"]},
    {"series": "水文", "domain": "geography", "titles": ["中华人民共和国世界遗产列表"]},
    {"series": "电力", "domain": "energy", "titles": ["中华人民共和国太阳能发电", "印度风力发电", "各国可再生能源电力产量列表", "太阳能光电产业成长", "离岸风力发电", "风能"]},
    {"series": "票房", "domain": "film", "titles": ["变形金刚电影系列", "哆啦A梦电影作品", "漫威电影宇宙系列电影", "詹姆斯·邦德系列电影"]},
    {"series": "科研", "domain": "science", "titles": ["诺贝尔和平奖得主列表"]},
    {"series": "航天", "domain": "science", "titles": ["R-7系列火箭", "中国内燃机车列表", "北斗卫星导航系统", "欧洲歌唱大赛"]},
    {"series": "航空", "domain": "transport", "titles": ["全球城市", "协和式客机", "印度航空公司列表", "最繁忙客运航空线列表", "桃园—香港航空路线"]},
    {"series": "资源", "domain": "energy", "titles": ["中华人民共和国能源政策", "各国天然气储量列表", "各国天然气出口量列表", "各国天然气进口量列表"]},
    {"series": "金融", "domain": "economy", "titles": ["恒生指数"]},
    {"series": "铁路", "domain": "transport", "titles": ["上海轨道交通11号线", "各国公路里程列表", "国家铁路里程列表", "深圳地铁"]},
    {"series": "高校", "domain": "education", "titles": ["东南大学", "中国海洋大学", "各国主权财富基金列表", "各国生态足迹列表", "欧洲国家人口列表", "芬兰大学列表"]},
]

DOMAIN_OF_SERIES = {s["series"]: s["domain"] for s in SERIES}
# series whose tables list only the top of a population (tallest buildings, busiest airports, scorers):
# "lowest"/"fewest" over such a table would be false in the world, so only max-direction claims are built.
PERSON_NOUNS = {"球员", "导演", "演员", "车手", "运动员"}
PARTIAL_SERIES = {"机场", "建筑", "地理", "电影", "中国城市人口", "世界城市人口", "山川湖泊", "交通能源", "摩天大楼"}
MEMBER_NOUNS = {"球员"}


def only_max(series: str, noun: str) -> bool:
    return series in PARTIAL_SERIES or noun in MEMBER_NOUNS

# --------------------------------------------------------------------------- lexicon
# header -> (display name, style, measure word). style: count | amount | rate
METRIC_LEXICON: dict[str, tuple[str, str, str]] = {
    "得": ("进球数", "count", "个"), "进": ("进球数", "count", "个"), "进球": ("进球数", "count", "个"),
    "失": ("失球数", "count", "个"), "失球": ("失球数", "count", "个"),
    "净": ("净胜球数", "count", "个"), "差": ("净胜球数", "count", "个"), "净胜球": ("净胜球数", "count", "个"),
    "分": ("积分", "count", "分"), "积分": ("积分", "count", "分"),
    "赛": ("比赛场次", "count", "场"), "场": ("比赛场次", "count", "场"), "场次": ("比赛场次", "count", "场"),
    "胜": ("胜场数", "count", "场"), "负": ("负场数", "count", "场"), "平": ("平局场数", "count", "场"),
    "和": ("平局场数", "count", "场"), "胜率": ("胜率", "rate", ""),
    "金牌": ("金牌数", "count", "枚"), "金": ("金牌数", "count", "枚"), "银牌": ("银牌数", "count", "枚"),
    "银": ("银牌数", "count", "枚"), "铜牌": ("铜牌数", "count", "枚"), "铜": ("铜牌数", "count", "枚"),
    "奖牌总数": ("奖牌总数", "count", "枚"),
    "容量": ("球场容量", "amount", "人"), "容纳人数": ("球场容量", "amount", "人"),
    "点球": ("点球进球数", "count", "个"), "助攻": ("助攻数", "count", "次"),
    "冠军": ("冠军数", "count", "个"), "亚军": ("亚军数", "count", "个"), "季军": ("季军数", "count", "个"),
    "已赛": ("比赛场次", "count", "场"), "出席次数": ("参赛次数", "count", "次"), "得分": ("得分", "count", "分"),
    "失分": ("失分", "count", "分"),
    "乘客总量": ("乘客总量", "amount", "人次"), "人次": ("观影人次", "amount", "人次"),
    "观影人次估计": ("观影人次", "amount", "人次"), "股值": ("市值", "amount", ""),
}
# headers that read as superlatives or are too vague to name a metric
PERIOD_ONLY_RE = re.compile(r"^(19|20)\d{2}\s*(年|\(p\)|（p）|\(e\))?$")
AMBIGUOUS_METRIC_RE = re.compile(r"^(最高|最低|最少|最多|平均|变化|数量|数值|数据|其他|备注|总数_\d|总计_\d|列\d+)$")

ENTITY_NOUNS: list[tuple[str, str]] = [
    (r"球员|射手|运动员|选手", "球员"), (r"球队|队伍|俱乐部|队名", "球队"), (r"车手", "车手"),
    (r"代表团|国家.*地区|国家／地区|国家/地区|国家或地区|国家 / 地区", "国家或地区"), (r"^国家$|国家", "国家"),
    (r"地级行政区|地级市|城市|^市$|市州|^行政区$", "城市"), (r"省级行政区|省份|省区|^省$|^地区$|行政区", "省级行政区"),
    (r"机场", "机场"), (r"建筑|大厦|大楼", "建筑"), (r"电视塔|塔", "电视塔"), (r"河流|河名|^河$", "河流"),
    (r"湖泊|湖名|湖沼", "湖泊"), (r"山峰|山名|^山", "山峰"), (r"港口", "港口"), (r"水电站|电站", "水电站"),
    (r"水坝|大坝", "水坝"), (r"线路", "线路"), (r"导演", "导演"), (r"演员", "演员"), (r"系列", "系列作品"), (r"岛屿|岛名", "岛屿"), (r"都道府县", "都道府县"), (r"州", "州"), (r"邦", "邦"),
    (r"电影|影片|片名", "影片"), (r"大学|高校|学校|院校|校名", "高校"), (r"^NOC$|奥委会", "国家或地区"),
    # key headers the topic-diversity pass turned up: without them 199 usable single-table packages,
    # across 137 pages, resolved to 对象 and every builder skipped them
    (r"铁路线|线路名称|线名", "线路"), (r"站名|车站", "车站"), (r"保护区", "保护区"),
    (r"型号|车型|机型", "型号"), (r"区划", "行政区"), (r"公园", "公园"), (r"博物馆", "博物馆"),
    # before the catch-all below: '公司名称(中文)' matches 名称 first and would resolve to 对象, which makes
    # every builder skip the table (Fortune-500 style country->company rank bridges, among others)
    (r"公司|企业|集团|银行", "公司"),
    (r"名称|名字", "对象"),
]

ATTR_HEADER_RE = re.compile(r"^(城市|主场|主场馆|球场|主教练|队长|省份|省|大洲|所属.{0,4}|地区|国家|联盟|分区|赛区|首都|省会|车队|制造商|区域|组别|所在地|所在城市|主场城市)$")
CATEGORY_HEADER_RE = re.compile(r"分区|赛区|组别|小组|大洲|^洲$|省份|^省$|地区|区域|所属|类型|类别|联盟|分节|级别|国家|城市|车队|制造商|分部")

# plausible metrics per domain used for metric-slot NEI (must be absent from the package headers)
NEI_METRICS: dict[str, list[tuple[str, str, str]]] = {
    "football": [("助攻数", "count", "次"), ("黄牌数", "count", "张"), ("红牌数", "count", "张"),
                 ("射门次数", "count", "次"), ("控球率", "rate", ""), ("主场观众人数", "amount", "人")],
    "football_intl": [("黄牌数", "count", "张"), ("射门次数", "count", "次"), ("控球率", "rate", ""),
                      ("角球数", "count", "个")],
    "basketball": [("场均得分", "amount", "分"), ("场均篮板", "amount", "个"), ("三分命中率", "rate", ""),
                   ("主场胜场数", "count", "场")],
    "medals": [("参赛运动员人数", "count", "人"), ("参赛项目数", "count", "项"), ("第四名次数", "count", "次")],
    "economy": [("财政收入", "amount", ""), ("进出口总额", "amount", ""), ("社会消费品零售总额", "amount", ""),
                ("城镇化率", "rate", "")],
    "transport": [("航线数量", "count", "条"), ("跑道数量", "count", "条"), ("航站楼面积", "amount", "")],
    "architecture": [("电梯数量", "count", "部"), ("建筑面积", "amount", ""), ("停车位数量", "count", "个")],
    "geography": [("年平均降水量", "amount", ""), ("森林覆盖率", "rate", ""), ("平均海拔", "amount", "")],
    "film": [("观影人次", "amount", ""), ("排片场次", "count", "场"), ("豆瓣评分", "amount", "")],
    "demographics": [("出生率", "rate", ""), ("死亡率", "rate", ""), ("男女性别比", "amount", ""),
                     ("户籍人口", "amount", "人"), ("常住外来人口", "amount", "人")],
    "education": [("师生比", "amount", ""), ("图书馆藏书量", "amount", "册"), ("博士点数量", "count", "个"),
                  ("留学生人数", "amount", "人")],
    "energy": [("发电小时数", "amount", "小时"), ("输电损耗率", "rate", ""), ("机组数量", "count", "台")],
    "culture": [("年度展览场次", "count", "场"), ("文创收入", "amount", ""), ("志愿者人数", "amount", "人")],
    "science": [("有效载荷质量", "amount", "千克"), ("轨道倾角", "amount", "度"), ("在轨寿命", "amount", "年")],
}


MEDAL_ONLY = {"总计": ("奖牌总数", "count", "枚"), "总数": ("奖牌总数", "count", "枚"), "合计": ("奖牌总数", "count", "枚"),
              "金": ("金牌数", "count", "枚"), "银": ("银牌数", "count", "枚"), "铜": ("铜牌数", "count", "枚")}
SPORTS_ONLY = {"得", "失", "净", "差", "分", "赛", "场", "胜", "负", "平", "和", "进"}
NON_PERIOD_METRICS = re.compile(r"容量|容纳|面积|长度|高度|海拔")


NON_SPORTS_METRIC_RE = re.compile(
    r"人口|面积|生产总值|GDP|收入|吞吐量|高度|长度|票房|人数|密度|增长率|增速|里程|装机容量|产量|海拔|流域|楼层|层数|"
    r"席位|金牌|银牌|铜牌|奖牌|总数|总计|旅客|货邮|起降|架次|指数|预期寿命|消费|支出|产值|投资|出口|进口|座位|容量|车站|线路|"
    r"深度|坝高|库容|客运|遗产|生育率|发电量|楼层|排放|总额|吨|标准箱|长|高|面积|世界遗产|数目|"
    # added with the topic-diversity pass: this whitelist, not the lexicon, is what decides whether a
    # column can be named at all, so a page about sales, passengers or reserves produced no claims
    r"乘客|人次|销售|利润|营收|营业|市值|股值|储量|装机|运量|载客|学生|教职工|藏书|床位|专利|保有量|"
    r"用水|耕地|粮食|发电|风电|光伏|机队|航线|隧道|桥梁|水库|工资|税率|失业|识字|肥胖|中位数|年龄")


SENSITIVE_CELL_RE = re.compile(r"中华民国|中華民國")
SENSITIVE_TITLE_RE = re.compile(
    r"军|武装|战争|战役|伤亡|死亡|遇难|灾|民族|宗教|台湾|臺灣|西藏|新疆.*(事件|冲突)|选举|政党|抗议|"
    # the topic-diversity search surfaced an air-disaster page, a terrorism index and a thriller;
    # casualty and crime content is out of scope for the benchmark whatever its table shape
    r"空难|坠机|海难|事故|恐怖|袭击|杀人|凶杀|犯罪|谋杀|疫情|病死|自杀|气旋|飓风|台风|地震|海啸")


SEMANTIC_PAREN_RE = re.compile(r"增幅|增速|增长|比重|占比|人均|变化|名义|实际")


def clean_name(header: str) -> str:
    """Readable column name: drop parentheses, years, units, estimate markers and duplicate suffixes.
    Parentheses that change the meaning of the column (GDP（增幅/%）, GDP（人均）) are kept as words."""
    def paren(m: re.Match) -> str:
        inner = m.group(0)[1:-1]
        words = SEMANTIC_PAREN_RE.findall(inner)
        return "".join(words)
    name = re.sub(r"[（(][^）)]*[）)]", paren, header)
    name = re.sub(r"_\d+$", "", name)
    name = re.sub(r"(19|20)\d{2}(–\d{2,4})?年?", "", name)
    name = re.sub(r"估计|统计|数据|初步数|（.*$|\(.*$", "", name)
    name = re.sub(r"\s*(百万|千万|亿|万)?\s*(美元|人民币|元|国际元|平方公里|km2|km²|公里|千米|米|m)$", "", name.strip())
    name = re.sub(r"\s+", "", name)
    return name or header


def metric_info(header: str, *, unit: str = "", is_rate: bool = False, domain: str = "") -> tuple[str, str, str] | None:
    """(display name, style, measure word) for a metric header; None if the header is unusable."""
    h = header.strip()
    if domain not in ("football", "football_intl", "basketball", "medals", "") and not NON_SPORTS_METRIC_RE.search(h):
        return None
    if AMBIGUOUS_METRIC_RE.match(h) or PERIOD_ONLY_RE.match(h):
        return None
    if h in MEDAL_ONLY:
        return MEDAL_ONLY[h] if domain == "medals" else None
    if h in SPORTS_ONLY and domain not in ("football", "football_intl", "basketball", ""):
        return None
    if h in METRIC_LEXICON:
        return METRIC_LEXICON[h]
    base = clean_name(h)
    if base in METRIC_LEXICON:
        return METRIC_LEXICON[base]
    if len(base) == 1:
        return None
    if is_rate or "率" in base or unit in ("%", "‰"):
        return (base, "rate", "")
    if re.search(r"(数|次数|人数|个数|场数|座数|枚|场次|次|场)$", base):
        return (base, "count", "")
    return (base, "amount", "")


PROVINCES = {"北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏", "浙江", "安徽", "福建",
             "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏",
             "陕西", "甘肃", "青海", "宁夏", "新疆", "香港", "澳门", "台湾"}


def province_core(name: str) -> str:
    return re.sub(r"(省|市|自治区|壮族自治区|回族自治区|维吾尔自治区|特别行政区)$", "", name)


def refine_noun(noun: str, keys: list[str]) -> str:
    """Disambiguate 省级行政区 vs 城市 from the entities themselves (a header like 地区 is used for both)."""
    if noun not in ("省级行政区", "城市") or not keys:
        return noun
    share = sum(1 for k in keys if province_core(k) in PROVINCES) / len(keys)
    return "省级行政区" if share >= 0.7 else "城市"


def entity_noun(key_header: str, domain: str = "") -> str:
    if domain in ("football", "football_intl", "basketball") and re.search(r"^姓名$|^名字$", key_header):
        return "球员"
    if domain in ("world_stats", "medals", "geography") and re.search(r"^地区$|国家", key_header):
        return "国家或地区"
    for pattern, noun in ENTITY_NOUNS:
        if re.search(pattern, key_header):
            return noun
    return "对象"


def comparison_words(style: str) -> dict[str, str]:
    if style == "count":
        return {"gt": "多于", "lt": "少于", "eq": "与…相同", "max": "最多", "min": "最少", "more": "多", "less": "少"}
    return {"gt": "高于", "lt": "低于", "eq": "与…相同", "max": "最高", "min": "最低", "more": "高", "less": "低"}


def scope_phrase(page_title: str) -> str:
    title = re.sub(r"（[^）]*）|\([^)]*\)", "", page_title).strip()
    title = re.sub(r"(列表|排名|奖牌榜|小组赛)$", lambda m: "奖牌榜" if m.group(1) == "奖牌榜" else "", title)
    return title
