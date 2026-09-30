"""Hash-checked error/axis export after a complete frozen-release run.

Writes data for qualitative analysis, not automatically invented explanations.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import common as C


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--claims", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.run / "run_meta.json").read_text())
    if meta["claims_sha256"] != hashlib.sha256(a.claims.read_bytes()).hexdigest() or meta.get("limit") != 0:
        raise SystemExit("wrong dataset or pilot run")
    claims = [json.loads(l) for l in a.claims.read_text().splitlines()]
    records = {r["id"]:r for r in map(json.loads, (a.run / "predictions.jsonl").read_text().splitlines())}
    if set(records) != {r["id"] for r in claims}:
        raise SystemExit("incomplete or extraneous prediction IDs")
    axes, errors = collections.defaultdict(list), []
    for claim in claims:
        r = records[claim["id"]]
        if r.get("input_sha256") != C.input_hash(claim) or r.get("gold") != claim["label"] or r.get("run_fingerprint") != meta["run_fingerprint"]:
            raise SystemExit("prediction identity mismatch")
        axis = claim["quality_flags"].get("perturbation_axis", "none")
        if claim["label"] in ("REFUTES", "NEI"):
            axes[claim["label"] + ":" + axis].append((claim["label"], r["label"]))
        if r["label"] != claim["label"]:
            errors.append({"id":claim["id"], "claim":claim["claim"], "gold":claim["label"],
                "prediction":r["label"], "group":claim["program"]["category"], "axis":axis,
                "error_type":"invalid_output" if r["label"] == "INVALID" else "label_confusion",
                "raw_output":r.get("raw_output"), "turns":r.get("turns"),
                "qualitative_explanation":"REQUIRES HUMAN INTERPRETATION; NOT AUTO-INFERRED"})
    report = {"claims_sha256":meta["claims_sha256"], "run_fingerprint":meta["run_fingerprint"],
        "diagnostic_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "axis_accuracy":{k:{"n":len(v), "accuracy":round(100*C.label_scores(v)["accuracy"],1)} for k,v in sorted(axes.items())},
        "error_count":len(errors), "errors":errors}
    (a.run / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"exported {len(errors)} error records, without invented explanations")


if __name__ == "__main__":
    main()
