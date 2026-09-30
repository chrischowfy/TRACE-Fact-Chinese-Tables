"""Chinese Wikipedia acquisition: fetch a page revision, parse its ``wikitable`` tables.

Every fetched page is cached as JSON (html + pageid + revid) under ``runs/cache/pages`` so a
build is reproducible from the cache alone, and every table keeps page URL, page id and
revision id for CC BY-SA attribution. Table ids are globally unique: ``zw{pageid}r{revid}t{k}``.

The HTML table parser expands rowspan/colspan, drops footnote markers (``<sup class=reference>``)
and hidden sort keys, merges multi-row headers, and turns full-width section-divider rows into a
``分节`` column when they partition the rows. It is a rewrite of the table parser in the
review-time code release (``trace_fact/zhwiki_template_transfer_v0_7.py``).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests

API_URL = "https://zh.wikipedia.org/w/api.php"
USER_AGENT = "ZHTabfact-rebuild/1.0 (academic benchmark construction; CC BY-SA attribution kept)"
REQUEST_DELAY = 1.0
MAX_RETRIES = 4
LICENSE = "CC BY-SA 4.0"

_last_request = 0.0
_T2S = None


def to_simplified(text: str) -> str:
    """Traditional -> Simplified (page titles from search/redirects can be Traditional even with variant=zh-cn)."""
    global _T2S
    if _T2S is None:
        from opencc import OpenCC  # opencc-python-reimplemented
        _T2S = OpenCC("t2s")
    return _T2S.convert(text or "")


def _api_get(params: dict[str, Any]) -> dict[str, Any]:
    global _last_request
    for attempt in range(MAX_RETRIES):
        wait = REQUEST_DELAY - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.time()
        try:
            resp = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
            if resp.status_code == 429:
                time.sleep(REQUEST_DELAY * (2 ** (attempt + 1)))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(REQUEST_DELAY * (2 ** (attempt + 1)))
    raise RuntimeError("unreachable")


def _cache_path(cache_dir: Path, title: str) -> Path:
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.json"


def fetch_page(title: str, cache_dir: str | Path, *, cache_only: bool = False) -> dict[str, Any] | None:
    """Return {title, resolved_title, pageid, revid, url, html}; None if the page does not exist."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, title)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached or None
    if cache_only:
        return None
    payload = _api_get({
        "action": "parse", "page": title, "prop": "text|revid", "format": "json",
        "formatversion": "2", "redirects": "1", "variant": "zh-cn",
    })
    parsed = payload.get("parse")
    if not parsed:
        path.write_text("{}", encoding="utf-8")
        return None
    page = {
        "title": title,
        "resolved_title": parsed.get("title") or title,
        "pageid": parsed.get("pageid"),
        "revid": parsed.get("revid"),
        "url": "https://zh.wikipedia.org/wiki/" + str(parsed.get("title") or title).replace(" ", "_"),
        "revision_url": f"https://zh.wikipedia.org/w/index.php?oldid={parsed.get('revid')}",
        "html": parsed.get("text") or "",
    }
    path.write_text(json.dumps(page, ensure_ascii=False), encoding="utf-8")
    return page


def search_titles(query: str, limit: int = 20) -> list[str]:
    payload = _api_get({"action": "query", "list": "search", "srsearch": query, "srlimit": limit,
                        "format": "json", "formatversion": "2"})
    return [r["title"] for r in (payload.get("query") or {}).get("search") or []]


# --------------------------------------------------------------------------- HTML parsing
_WS_RE = re.compile(r"[\s 　]+")
_FOOTNOTE_RE = re.compile(r"\[(?:注?\s*\d+|[a-zA-Z]|note \d+|注 \d+|來源請求|来源请求|需要校对)\]")


_NAVBOX_RE = re.compile(r"查\s*论\s*编|閱\s*論\s*編|查\s*論\s*編")


def clean_text(text: str) -> str:
    text = _NAVBOX_RE.sub("", _FOOTNOTE_RE.sub("", text or ""))
    return _WS_RE.sub(" ", text).strip()


class _TableParser(HTMLParser):
    """Collect ``wikitable`` tables as rows of {text, th, rowspan, colspan} cells."""

    SKIP_CLASSES = ("reference", "sortkey", "mw-editsection", "noprint", "flagicon", "mw-ref")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []
        self._cell: dict[str, Any] | None = None
        self._row: list[dict[str, Any]] | None = None
        self._caption: list[str] | None = None
        self._skip_depth = 0
        self._tag_stack: list[bool] = []  # per open element: does it start a skipped region

    def _capturing(self) -> bool:
        return bool(self._stack) and self._stack[-1]["capture"]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        void = tag in ("br", "img", "hr", "wbr", "meta", "link", "input")
        skip = False
        if tag in ("style", "script", "sup") and (tag != "sup" or "reference" in a.get("class", "")):
            skip = True
        if any(c in a.get("class", "").split() for c in self.SKIP_CLASSES):
            skip = True
        if "display:none" in a.get("style", "").replace(" ", ""):
            skip = True
        if not void:
            self._tag_stack.append(skip)
            if skip:
                self._skip_depth += 1
        if tag == "table":
            classes = a.get("class", "").split()
            self._stack.append({"capture": "wikitable" in classes, "rows": [], "caption": "",
                                "nested": bool(self._stack)})
            return
        if self._skip_depth or not self._capturing():
            if tag == "br" and self._cell is not None and not self._skip_depth:
                self._cell["parts"].append(" ")
            return
        if tag == "caption":
            self._caption = []
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = {"th": tag == "th", "parts": [], "rowspan": _int(a.get("rowspan")),
                          "colspan": _int(a.get("colspan")), "link": ""}
        elif tag == "a" and self._cell is not None and not self._cell["link"] and a.get("title"):
            if "redlink" not in a.get("class", "") and ":" not in a.get("title", ""):
                self._cell["link"] = a["title"]
        elif tag == "br" and self._cell is not None:
            self._cell["parts"].append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("br", "img", "hr", "wbr", "meta", "link", "input"):
            return
        if self._tag_stack:
            if self._tag_stack.pop():
                self._skip_depth -= 1
                return
        if tag == "table" and self._stack:
            t = self._stack.pop()
            if t["capture"] and not t["nested"]:
                self.tables.append({"rows": t["rows"], "caption": clean_text(t["caption"])})
            return
        if self._skip_depth or not self._capturing():
            return
        if tag == "caption" and self._caption is not None:
            self._stack[-1]["caption"] = "".join(self._caption)
            self._caption = None
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._cell["text"] = clean_text("".join(self._cell.pop("parts")))
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._stack:
                self._stack[-1]["rows"].append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not self._capturing():
            return
        if self._cell is not None:
            self._cell["parts"].append(data)
        elif self._caption is not None:
            self._caption.append(data)


def _int(value: str | None) -> int:
    try:
        return max(1, min(int(re.sub(r"\D", "", value or "") or 1), 200))
    except ValueError:
        return 1


def expand_grid(raw_rows: list[list[dict[str, Any]]]) -> list[list[tuple[str, bool, str]]]:
    """Expand rowspan/colspan into a rectangular grid of (text, is_th, wiki link title)."""
    grid: list[list[tuple[str, bool, str]]] = []
    carry: dict[int, list[Any]] = {}  # col -> [remaining, text, th, link]
    for raw in raw_rows:
        row: list[tuple[str, bool]] = []
        col = 0
        cells = list(raw)
        while cells or any(c >= col for c in carry):
            if col in carry:
                rem, text, th, link = carry[col]
                row.append((text, th, link))
                if rem <= 1:
                    del carry[col]
                else:
                    carry[col][0] = rem - 1
                col += 1
                continue
            if not cells:
                if any(c > col for c in carry):
                    row.append(("", False, ""))
                    col += 1
                    continue
                break
            cell = cells.pop(0)
            for k in range(cell["colspan"]):
                row.append((cell["text"], cell["th"], cell.get("link", "")))
                if cell["rowspan"] > 1:
                    carry[col + k] = [cell["rowspan"] - 1, cell["text"], cell["th"], cell.get("link", "")]
            col += cell["colspan"]
        grid.append(row)
    width = max((len(r) for r in grid), default=0)
    return [r + [("", False, "")] * (width - len(r)) for r in grid]


def grid_to_table(grid: list[list[tuple[str, bool, str]]]) -> tuple[list[str], list[list[str]], bool, list[str]]:
    """Split a grid into (headers, rows, has_section_column, link titles seen in data cells)."""
    if not grid:
        return [], [], False, []
    width = len(grid[0])
    header_rows: list[list[str]] = []
    i = 0
    while (i < len(grid) and all(th or not text for text, th, _ in grid[i]) and any(t for t, _, _ in grid[i])
           and not _mostly_numeric([t for t, _, _ in grid[i]])):
        header_rows.append([t for t, _, _ in grid[i]])
        i += 1
    if not header_rows:  # tables without <th>: first non-empty row is the header
        while i < len(grid) and not any(t for t, _, _ in grid[i]):
            i += 1
        if i >= len(grid):
            return [], [], False, []
        header_rows = [[t for t, _, _ in grid[i]]]
        i += 1
    # drop leading title rows that span the whole width with one repeated value
    while len(header_rows) > 1 and len({v for v in header_rows[0] if v}) == 1:
        header_rows.pop(0)
    headers = []
    for c in range(width):
        parts: list[str] = []
        for hr in header_rows:
            v = hr[c]
            if v and (not parts or parts[-1] != v):
                parts.append(v)
        name = parts[-1] if parts else f"列{c + 1}"
        if len(parts) >= 2 and _header_needs_parent(parts[-1]):
            name = parts[-2] + parts[-1]
        headers.append(name)
    seen: dict[str, int] = {}
    for c, h in enumerate(headers):
        if h in seen:
            seen[h] += 1
            headers[c] = f"{h}_{seen[h]}"
        else:
            seen[h] = 1
    rows: list[list[str]] = []
    sections: list[str | None] = []
    current: str | None = None
    links: list[str] = []
    for r in grid[i:]:
        texts = [t for t, _, _ in r]
        if not any(texts) or texts == [h.split("_")[0] for h in headers]:
            continue
        nonempty = {t for t in texts if t}
        if len(nonempty) == 1 and width >= 3 and sum(1 for t in texts if t) >= width - 1:
            current = next(iter(nonempty))
            continue
        if all(th for _, th, _ in r) and r and len(header_rows) and texts == header_rows[-1]:
            continue
        rows.append(texts)
        sections.append(current)
        links.extend(f"{t}\t{lk}" for t, _, lk in r if lk and t)
    section_counts: dict[str, int] = {}
    for s in sections:
        if s:
            section_counts[s] = section_counts.get(s, 0) + 1
    has_section = len([s for s, n in section_counts.items() if n >= 2]) >= 2 and all(s for s in sections)
    if has_section:
        headers = headers + ["分节"]
        rows = [r + [s or ""] for r, s in zip(rows, sections)]
    return headers, rows, has_section, sorted(set(links))


def _mostly_numeric(values: list[str]) -> bool:
    vals = [v for v in values if v]
    num = sum(1 for v in vals if re.fullmatch(r"[+\-–−]?[\d,，.]+%?", v))
    return len(vals) >= 2 and num / len(vals) >= 0.5


def _header_needs_parent(h: str) -> bool:
    """Leaf headers that are ambiguous without their parent (e.g. 2022 under 地区生产总值)."""
    return bool(re.fullmatch(r"\d{4}年?|[金银铜]|[男女]|数量|人数|比例|占比|排名|数值|增速|总计|合计", h))


def extract_tables(page: dict[str, Any]) -> list[dict[str, Any]]:
    parser = _TableParser()
    parser.feed(page.get("html") or "")
    tables: list[dict[str, Any]] = []
    for k, raw in enumerate(parser.tables):
        grid = expand_grid(raw["rows"])
        headers, rows, has_section, links = grid_to_table(grid)
        if not headers or not rows:
            continue
        table_id = f"zw{page['pageid']}r{page['revid']}t{k:02d}"
        title = to_simplified(page["title"])
        tables.append({
            "table_id": table_id,
            "title": to_simplified(raw["caption"]) or f"{title}（表{k + 1}）",
            "headers": [to_simplified(h) for h in headers],
            "rows": rows,
            "cell_links": links,
            "source": {
                "page_title": title, "resolved_title": page["resolved_title"],
                "page_id": page["pageid"], "revision_id": page["revid"],
                "url": page["url"], "revision_url": page["revision_url"], "table_index": k,
                "license": LICENSE, "section_column_recovered": has_section,
            },
        })
    return tables
