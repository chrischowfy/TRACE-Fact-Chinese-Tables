"""Deterministically replay a frozen release from cached source pages.

The normal construction pipeline intentionally evolves as new package builders and
quality gates are added.  A published release therefore carries a compact manifest
that freezes instance selection and accepted surface forms.  Replay still rebuilds
every evidence table from the page cache and re-executes every stored program; it
does not copy the released ``claims.jsonl``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .executor import InvalidProgram, execute
from .packages import build_packages
from .resolve import load as load_redirects
from .tables import profile_table


MANIFEST = "rebuild_manifest.jsonl"
META = "rebuild_meta.json"
PACKAGES = "release_packages.jsonl"
SUMMARY = "release_build_summary.json"


class FrozenReplayError(RuntimeError):
    """The cached sources no longer satisfy the frozen release manifest."""


def find_manifest(cache_dir: str | Path) -> Path | None:
    path = Path(cache_dir).parent / MANIFEST
    return path if path.is_file() else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(data)


def _checked_bytes(path: Path, expected: str, description: str) -> bytes:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise FrozenReplayError(f"missing {description}: {path}") from exc
    actual = _sha256(data)
    if actual != expected:
        raise FrozenReplayError(
            f"{description} hash mismatch: expected {expected}, got {actual} ({path})"
        )
    return data


def _public_table(source: dict[str, Any], frozen_id: str) -> dict[str, Any]:
    """Project a builder table to the public schema while restoring its frozen id."""
    table = {
        "table_id": frozen_id,
        "title": source.get("title"),
        "headers": source["headers"],
        "rows": source["rows"],
    }
    if "period" in source:
        table["period"] = source["period"]
    table["source"] = source["source"]
    return table


def replay(cache_dir: str | Path, out_dir: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Rebuild and verify the release described by the manifest next to ``cache_dir``."""
    cache = Path(cache_dir)
    manifest = Path(manifest_path) if manifest_path else cache.parent / MANIFEST
    bundle = manifest.parent
    try:
        meta = json.loads((bundle / META).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FrozenReplayError(f"missing frozen replay metadata: {bundle / META}") from exc

    if meta.get("format_version") != 1:
        raise FrozenReplayError(f"unsupported frozen replay format: {meta.get('format_version')!r}")

    manifest_data = _checked_bytes(manifest, meta["manifest_sha256"], "rebuild manifest")
    package_data = _checked_bytes(bundle / PACKAGES, meta["packages_sha256"], "package inventory")
    summary_data = _checked_bytes(bundle / SUMMARY, meta["build_summary_sha256"], "build summary")

    load_redirects(cache.parent / "redirects.json")
    package_list = build_packages(str(cache), cache_only=True)
    package_by_id = {p["package_id"]: p for p in package_list}
    if len(package_by_id) != len(package_list):
        raise FrozenReplayError("current package builder produced duplicate package ids")

    inventory = [json.loads(line) for line in package_data.decode("utf-8").splitlines() if line.strip()]
    if len(inventory) != meta["package_records"]:
        raise FrozenReplayError(
            f"package inventory count mismatch: expected {meta['package_records']}, got {len(inventory)}"
        )
    inventory_ids = [item["package_id"] for item in inventory]
    if len(set(inventory_ids)) != len(inventory_ids):
        raise FrozenReplayError("frozen package inventory contains duplicate package ids")

    specs = [json.loads(line) for line in manifest_data.decode("utf-8").splitlines() if line.strip()]
    if len(specs) != meta["claims"]:
        raise FrozenReplayError(f"claim count mismatch: expected {meta['claims']}, got {len(specs)}")

    package_table_hashes = meta["package_table_sha256"]
    table_cache: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    lines: list[str] = []
    seen_ids: set[str] = set()
    for spec in specs:
        claim_id = spec["id"]
        if claim_id in seen_ids:
            raise FrozenReplayError(f"duplicate claim id in manifest: {claim_id}")
        seen_ids.add(claim_id)

        package_id = spec["evidence_package_id"]
        package = package_by_id.get(package_id)
        if package is None:
            raise FrozenReplayError(f"claim {claim_id}: package not produced from cache: {package_id}")
        frozen_ids = tuple(spec["table_ids"])
        if len(package["tables"]) != len(frozen_ids):
            raise FrozenReplayError(
                f"claim {claim_id}: table count changed for {package_id}: "
                f"{len(package['tables'])} != {len(frozen_ids)}"
            )

        cache_key = (package_id, frozen_ids)
        tables = table_cache.get(cache_key)
        if tables is None:
            tables = [_public_table(table, table_id) for table, table_id in zip(package["tables"], frozen_ids)]
            expected_hashes = package_table_hashes.get(package_id)
            if expected_hashes is None or len(expected_hashes) != len(tables):
                raise FrozenReplayError(f"claim {claim_id}: no frozen table hashes for package {package_id}")
            for table, expected in zip(tables, expected_hashes):
                table_id = table["table_id"]
                actual = _object_sha256(table)
                if actual != expected:
                    raise FrozenReplayError(
                        f"claim {claim_id}: table {table_id} changed: expected {expected}, got {actual}"
                    )
            table_cache[cache_key] = tables

        profiles = {table["table_id"]: profile_table(table) for table in tables}
        try:
            run = execute(spec["program"]["operators"], profiles)
        except InvalidProgram as exc:
            raise FrozenReplayError(f"claim {claim_id}: program is no longer executable: {exc}") from exc
        if run["label"] != spec["label"]:
            raise FrozenReplayError(
                f"claim {claim_id}: label changed from {spec['label']} to {run['label']}"
            )

        evidence = run["cells"] if spec["label"] != "NEI" else []
        context = run["cells"] if spec["label"] == "NEI" else []
        record = {
            "id": claim_id,
            "claim": spec["claim"],
            "label": spec["label"],
            "surface": spec["surface"],
            "evidence_package_id": package_id,
            "tables": tables,
            "evidence_cells": evidence,
            "context_cells": context,
            "program": spec["program"],
            "table_topology": spec["table_topology"],
            "package_kind": spec["package_kind"],
            "quality_flags": spec["quality_flags"],
            "domain": spec["domain"],
            "series": spec["series"],
            "topic": spec["topic"],
            "license": spec["license"],
        }
        line = json.dumps(record, ensure_ascii=False)
        if _sha256(line.encode("utf-8")) != spec["record_sha256"]:
            raise FrozenReplayError(f"claim {claim_id}: reconstructed record differs from frozen record")
        lines.append(line)

    claims_data = ("\n".join(lines) + "\n").encode("utf-8")
    actual_claims_hash = _sha256(claims_data)
    if actual_claims_hash != meta["claims_sha256"]:
        raise FrozenReplayError(
            f"rebuilt claims hash mismatch: expected {meta['claims_sha256']}, got {actual_claims_hash}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "claims.jsonl").write_bytes(claims_data)
    (out / "packages.jsonl").write_bytes(package_data)
    (out / "build_summary.json").write_bytes(summary_data)
    return json.loads(summary_data)
