"""Shared evaluation code: prompts, OpenAI-compatible client, prediction parsing, scoring.

Fixes relative to the review-time harness (see audit/AUDIT_REPORT.md §2, §4):
  * the output example no longer anchors a label (it used to show {"label": "NEI", ...});
  * oracle-evidence / oracle-program keep the FULL tables and add the gold cells / program on top;
  * every unparseable output, API failure or failed program is INVALID and scored as wrong — it is
    never silently mapped to NEI;
  * families come from ``program.category`` of the released records.
"""
from __future__ import annotations

import json
import hashlib
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

LABELS = ("SUPPORTS", "REFUTES", "NEI")
SETTINGS = ("claim-only", "full-table", "oracle-evidence", "oracle-program")


def input_hash(claim):
    payload = {k: claim.get(k) for k in ("claim", "tables", "program", "evidence_cells", "context_cells")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def prepare_run(run_dir, meta, claims_path):
    """Reject id-only resume across changed datasets, prompts, code or decoding settings."""
    run_dir = Path(run_dir)
    meta = dict(meta)
    meta["claims_sha256"] = hashlib.sha256(Path(claims_path).read_bytes()).hexdigest()
    here = Path(__file__).resolve().parent
    code = [here / name for name in ("common.py", "direct.py", "sql_agents.py")]
    code.append(here.parent / "src/dart_fact/tables.py")
    meta["evaluation_code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code}
    semantic = {k:v for k,v in meta.items() if k not in ("out", "claims", "workers", "api_key_name")}
    meta["run_fingerprint"] = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    path = run_dir / "run_meta.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("run_fingerprint") != meta["run_fingerprint"]:
            raise SystemExit("run directory belongs to different/unversioned inputs or settings; use a new output directory")
        return previous
    if (run_dir / "predictions.jsonl").exists():
        raise SystemExit("predictions exist without run metadata; use a new output directory")
    meta["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta

SYSTEM = ("You are a careful Chinese table fact-checking assistant. Judge the claim using only the "
          "information given in the input. Output one JSON object and nothing else.")


def label_definitions() -> str:
    return ("Labels: SUPPORTS = the given tables entail the claim; REFUTES = the given tables contradict the claim; "
            "NEI = the given tables do not contain enough information to decide.")


def serialize_tables(tables: list[dict[str, Any]]) -> str:
    blocks = []
    for t in tables:
        blocks.append("\n".join([f"Table {t['table_id']}: {t.get('title') or ''}",
                                 "Headers: " + json.dumps(t["headers"], ensure_ascii=False),
                                 "Rows: " + json.dumps(t["rows"], ensure_ascii=False)]))
    return "\n\n".join(blocks)


def build_prompt(claim: dict[str, Any], setting: str) -> str:
    if setting not in SETTINGS:
        raise ValueError(setting)
    parts = [label_definitions(),
             'Return JSON with keys "label" (one of SUPPORTS, REFUTES, NEI), "evidence_cells" (at most 3 '
             'cells as [table_id, row, column]) and "rationale" (at most 30 words).',
             f"Claim: {claim['claim']}"]
    if setting == "claim-only":
        parts.append("No tables are provided in this setting.")
    else:
        parts.append("Tables:\n" + serialize_tables(claim["tables"]))
    if setting == "oracle-evidence":
        cells = claim.get("evidence_cells") or claim.get("context_cells") or []
        parts.append("Relevant cells (gold): " + json.dumps(
            [[c["table_id"], c["row"], c["col"], c["raw_value"]] for c in cells], ensure_ascii=False))
    if setting == "oracle-program":
        prog = claim["program"]
        parts.append("Verification program (operators to apply over the tables): " + json.dumps(
            {"skeleton": prog["skeleton_id"], "operators": prog["operators"]}, ensure_ascii=False))
    return "\n\n".join(parts)


def call_llm(messages: list[dict[str, str]], *, model: str, base_url: str, api_key: str, max_tokens: int,
             temperature: float = 0.0, json_mode: bool = True, extra: dict[str, Any] | None = None,
             timeout: int = 180, retries: int = 3) -> tuple[str | None, dict[str, Any]]:
    payload: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens,
                               "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    payload.update(extra or {})
    meta: dict[str, Any] = {}
    requested_at = datetime.now(timezone.utc).isoformat()
    for attempt in range(retries):
        try:
            resp = requests.post(base_url.rstrip("/") + "/chat/completions", json=payload, timeout=timeout,
                                 headers={"Authorization": f"Bearer {api_key}"})
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + random.random())
                continue
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            meta = {"finish_reason": choice.get("finish_reason"), "usage": data.get("usage"),
                    "requested_model": model, "returned_model": data.get("model"),
                    "response_id": data.get("id"), "response_created": data.get("created"),
                    "system_fingerprint": data.get("system_fingerprint"),
                    "requested_at_utc": requested_at,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat()}
            content = choice.get("message", {}).get("content")
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content, meta
        except (requests.RequestException, KeyError, ValueError) as exc:
            meta = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            time.sleep(2 ** attempt + random.random())
    return None, meta


def parse_label(raw: str | None) -> tuple[str, str | None]:
    """Return (label, error). label is INVALID when nothing usable was produced."""
    if not raw or not raw.strip():
        return "INVALID", "empty_output"
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    candidates += [text] + re.findall(r"\{.*\}", text, re.S)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            lab = str(obj.get("label", "")).strip().upper().replace(" ", "_")
            lab = {"SUPPORT": "SUPPORTS", "SUPPORTED": "SUPPORTS", "REFUTE": "REFUTES", "REFUTED": "REFUTES",
                   "NOT_ENOUGH_INFO": "NEI", "NOT_ENOUGH_INFORMATION": "NEI"}.get(lab, lab)
            return (lab, None) if lab in LABELS else ("INVALID", "invalid_label")
    return "INVALID", "invalid_json"


def api_key(env_name: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise SystemExit(f"environment variable {env_name} is empty")
    return key


# ------------------------------------------------------------------------------------------ scoring
def label_scores(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    f1 = {}
    for lab in LABELS:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    acc = sum(g == p for g, p in pairs) / len(pairs) if pairs else 0.0
    return {"accuracy": acc, "macro_f1": sum(f1.values()) / 3, "f1": f1}


def present_macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Macro-F1 over the labels that occur in the gold labels of this subset (a family without NEI instances
    must not be averaged with an NEI F1 of zero)."""
    f1 = label_scores(pairs)["f1"]
    present = {g for g, _ in pairs}
    return sum(f1[lab] for lab in LABELS if lab in present) / max(1, len(present))


def bootstrap_ci(pairs: list[tuple[str, str]], metric: str = "macro_f1", n: int = 1000, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        vals.append(label_scores(sample)[metric])
    vals.sort()
    return [round(100 * vals[int(0.025 * n)], 1), round(100 * vals[int(0.975 * n) - 1], 1)]


def score(claims: list[dict[str, Any]], preds: dict[str, str], *, ci: bool = True) -> dict[str, Any]:
    pairs = [(c["label"], preds.get(c["id"], "INVALID")) for c in claims]
    main = label_scores(pairs)
    out = {"n": len(pairs), "accuracy": round(100 * main["accuracy"], 1), "macro_f1": round(100 * main["macro_f1"], 1),
           "f1": {k: round(100 * v, 1) for k, v in main["f1"].items()},
           "invalid": sum(p == "INVALID" for _, p in pairs),
           "confusion": {g: {p: sum(1 for gg, pp in pairs if gg == g and pp == p) for p in (*LABELS, "INVALID")}
                         for g in LABELS}}
    if ci:
        out["macro_f1_ci95"] = bootstrap_ci(pairs)
        out["accuracy_ci95"] = bootstrap_ci(pairs, "accuracy")
    for field, getter in (("by_group", lambda c: c["program"]["category"]), ("by_topology", lambda c: c["table_topology"]),
                          ("by_domain", lambda c: c["domain"])):
        groups: dict[str, list] = {}
        for c in claims:
            groups.setdefault(getter(c), []).append((c["label"], preds.get(c["id"], "INVALID")))
        out[field] = {g: {"n": len(ps), "accuracy": round(100 * label_scores(ps)["accuracy"], 1),
                          "macro_f1": round(100 * present_macro_f1(ps), 1)} for g, ps in sorted(groups.items())}
    return out
