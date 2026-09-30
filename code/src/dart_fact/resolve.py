"""Wikipedia redirect resolution used to keep NEI substitutes from being aliases of entities in the tables.

``REDIRECTS`` maps a name to its canonical zh-Wikipedia article title ("" when no article exists). It is filled
from ``runs/cache/redirects.json`` by :func:`load`; :func:`resolve_titles` fills that cache through the MediaWiki
API (50 titles per request, ~1 request/s). Names missing from the cache are treated as unresolved and are never
used as NEI substitutes, so a build stays conservative and deterministic without network access.
"""
from __future__ import annotations

import json
from pathlib import Path

REDIRECTS: dict[str, str] = {}
CACHE = Path("runs/cache/redirects.json")


def load(path: str | Path = CACHE) -> None:
    p = Path(path)
    if p.exists():
        REDIRECTS.update(json.loads(p.read_text(encoding="utf-8")))


def _post(params: dict) -> dict:
    import time
    import requests
    from . import acquire
    for attempt in range(acquire.MAX_RETRIES):
        wait = acquire.REQUEST_DELAY - (time.time() - acquire._last_request)
        if wait > 0:
            time.sleep(wait)
        acquire._last_request = time.time()
        try:
            resp = requests.post(acquire.API_URL, data=params, headers={"User-Agent": acquire.USER_AGENT}, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 ** (attempt + 2))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            time.sleep(2 ** (attempt + 1))
    return {}


def resolve_titles(names: list[str], path: str | Path = CACHE) -> int:
    load(path)
    todo = sorted({n for n in names if n and n not in REDIRECTS and len(n) <= 60 and "|" not in n
                   and "页面不存在" not in n and "頁面不存在" not in n})
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        data = _post({"action": "query", "titles": "|".join(batch), "redirects": "1", "converttitles": "1",
                      "format": "json", "formatversion": "2"}).get("query", {})
        step = {}
        for key in ("normalized", "converted", "redirects"):
            for m in data.get(key, []):
                step[m["from"]] = m["to"]
        existing = {pg["title"] for pg in data.get("pages", []) if not pg.get("missing") and not pg.get("invalid")}
        for name in batch:
            t = name
            for _ in range(4):
                if t in step and step[t] != t:
                    t = step[t]
                else:
                    break
            REDIRECTS[name] = t if t in existing else ""
        if i // 50 % 20 == 0:
            Path(path).write_text(json.dumps(REDIRECTS, ensure_ascii=False), encoding="utf-8")
    Path(path).write_text(json.dumps(REDIRECTS, ensure_ascii=False), encoding="utf-8")
    return len(todo)
