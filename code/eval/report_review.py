"""Publish complete, input-verified review-repair evaluation tables and audit metadata.

No calls to models, no data selection, and no changes to the frozen benchmark.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import common as C

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from export_review_paper import esc, table


def checked_run(path, claims, sha):
    meta = json.loads((path / "run_meta.json").read_text())
    if meta.get("claims_sha256") != sha or meta.get("limit") != 0:
        raise ValueError(f"dataset mismatch or pilot: {path}")
    if json.loads(meta["extra"]).get("thinking", {}).get("type") != "disabled":
        raise ValueError("this report only accepts the non-thinking protocol")
    records = {r["id"]:r for r in map(json.loads, (path / "predictions.jsonl").read_text().splitlines())}
    if set(records) != {c["id"] for c in claims}:
        raise ValueError(f"incomplete or extraneous predictions: {path}")
    for c in claims:
        r = records[c["id"]]
        if r.get("input_sha256") != C.input_hash(c) or r.get("gold") != c["label"] or r.get("run_fingerprint") != meta.get("run_fingerprint"):
            raise ValueError(f"prediction identity mismatch: {c['id']}")
    report = C.score(claims, {k:r["label"] for k,r in records.items()})
    calls = [call.get("api", {}) for r in records.values()
             for call in (r.get("api_calls") or [{"api":r.get("api", {})}])]
    report["api_audit"] = {
        "requested_model":meta["model"],
        "returned_models":dict(collections.Counter(c.get("returned_model", "not_returned") for c in calls)),
        "finish_reasons":dict(collections.Counter(c.get("finish_reason", "not_returned") for c in calls)),
        "http_or_transport_errors":sum(bool(c.get("error")) for c in calls),
        "calls":len(calls),
        "usage":{key:sum((c.get("usage") or {}).get(key, 0) or 0 for c in calls)
                 for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "started_utc":min((c["requested_at_utc"] for c in calls if c.get("requested_at_utc")), default=None),
        "finished_utc":max((c["completed_at_utc"] for c in calls if c.get("completed_at_utc")), default=None),
        "run_fingerprint":meta["run_fingerprint"],
        "prediction_sha256":hashlib.sha256((path / "predictions.jsonl").read_bytes()).hexdigest(),
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path, default=ROOT / "runs/review_repair_r5_eval/label_first/full")
    p.add_argument("--release", type=Path, default=ROOT / "data/review_repair_r5")
    p.add_argument("--paper-out", type=Path, default=ROOT.parent / "ccl2026/generated_review_repair")
    a = p.parse_args()
    path = a.release / "claims.jsonl"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    freeze = json.loads((a.release / "freeze_manifest.json").read_text())
    if sha != freeze["claims_sha256"]:
        raise SystemExit("frozen data changed")
    claims = list(map(json.loads, path.read_text().splitlines()))
    spec = [("deepseek-flash", "full-table", "Flash / full table"),
            ("deepseek-v4-pro", "full-table", "V4-Pro / full table"),
            ("deepseek-flash", "claim-only", "Flash / claim only"),
            ("deepseek-flash", "oracle-evidence", "Flash / oracle evidence"),
            ("deepseek-flash", "oracle-program", "Flash / oracle program"),
            ("deepseek-flash", "text2sql", "Flash / text-to-SQL"),
            ("deepseek-flash", "reactable", "Flash / ReAcTable-style")]
    reports, main_rows = {}, []
    # Do all validation and scoring before writing any paper table.
    for model, protocol, label in spec:
        key = model + "__" + protocol
        report = checked_run(a.run_root / key, claims, sha)
        reports[key] = report
        main_rows.append([label, report["accuracy"], report["macro_f1"]] + [report["f1"][k] for k in C.LABELS])
    a.paper_out.mkdir(parents=True, exist_ok=True)
    (a.paper_out / "results.tex").write_text(table(["Model / protocol", "Acc.", "Macro-F1", "S F1", "R F1", "NEI F1"], main_rows, "lrrrrr"))
    flash, pro = [reports[m + "__full-table"] for m in ("deepseek-flash", "deepseek-v4-pro")]
    breakdown = []
    for name, metric in (("Group", "by_group"), ("Topology", "by_topology")):
        for group, v in flash[metric].items():
            pv = pro[metric][group]
            breakdown.append([esc(group), v["n"], v["accuracy"], v["macro_f1"], pv["accuracy"], pv["macro_f1"]])
    (a.paper_out / "breakdown.tex").write_text(table(["Subset", "$n$", "Flash Acc.", "Flash F1", "Pro Acc.", "Pro F1"], breakdown, "lrrrrr"))
    summary = {"release":freeze["release"], "claims_sha256":sha, "status":"complete",
               "protocol_runs":len(reports), "reports":reports,
               "human_validation":"pending", "historical_checkpoint_identity":"not_established_by_api_alias"}
    (a.paper_out / "measured_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:{name:v[name] for name in ("n", "accuracy", "macro_f1", "invalid")} for k,v in reports.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
