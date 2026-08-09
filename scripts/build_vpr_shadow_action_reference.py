#!/usr/bin/env python3
"""Freeze sentinel action hashes from the authoritative v1.2 train-150 result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from vpr_shadow_capture_gate import (
        ACTION_HASH_CONTRACT,
        canonical_action_hash,
        load_action_sequences,
    )
    from vpr_shadow_data import sha256
except ImportError:
    from scripts.vpr_shadow_capture_gate import (
        ACTION_HASH_CONTRACT,
        canonical_action_hash,
        load_action_sequences,
    )
    from scripts.vpr_shadow_data import sha256


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--baseline-episodes", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    if split.get("role") != "engineering_sentinel":
        raise ValueError("action reference is restricted to engineering sentinel")
    if split.get("source_manifest_sha256") != sha256(args.source_manifest):
        raise ValueError("sentinel/source manifest mismatch")
    if (
        summary.get("schema") != "ascent_vo_submap_v1_2_b2_screen_v1"
        or summary.get("technical_status") != "PASS"
        or int(summary.get("strict_error_count", -1)) != 0
    ):
        raise ValueError("baseline v1.2 result is not strict-PASS")
    provenance = summary.get("provenance") or {}
    if provenance.get("source_commit") != args.expected_source_commit:
        raise ValueError("unexpected v1.2 baseline source commit")
    if provenance.get("manifest_sha256") != sha256(args.source_manifest):
        raise ValueError("v1.2 result/source manifest mismatch")

    identities: dict[str, tuple[str, str]] = {}
    for chunk in source["chunks"]:
        identity_path = Path(chunk["identity_path"])
        if sha256(identity_path) != chunk["identity_sha256"]:
            raise ValueError(f"identity hash mismatch: {identity_path}")
        for row in read_csv(identity_path):
            logical_id = row["logical_case_id"]
            identities[logical_id] = (
                str(chunk["chunk_id"]), str(row["runtime_episode_id"])
            )
    baseline_rows = {
        row["logical_case_id"]: row for row in read_csv(args.baseline_episodes)
    }
    action_cache: dict[str, dict[str, list[int]]] = {}
    entries: list[dict[str, Any]] = []
    for split_row in split["episodes"]:
        logical_id = str(split_row["logical_case_id"])
        baseline = baseline_rows.get(logical_id)
        if baseline is None or logical_id not in identities:
            raise ValueError(f"sentinel missing from v1.2 result: {logical_id}")
        chunk_id, runtime_id = identities[logical_id]
        if chunk_id != split_row["source_chunk_id"] or baseline["chunk_id"] != chunk_id:
            raise ValueError(f"sentinel chunk mismatch: {logical_id}")
        vo_path = Path(baseline["evidence_vo_diagnostics"]).resolve()
        cache_key = str(vo_path)
        if cache_key not in action_cache:
            action_cache[cache_key] = load_action_sequences(vo_path)[0]
        actions = action_cache[cache_key].get(runtime_id)
        if actions is None or len(actions) != int(baseline["b2_action_steps"]):
            raise ValueError(f"sentinel action evidence mismatch: {logical_id}")
        entries.append(
            {
                "logical_case_id": logical_id,
                "chunk_id": chunk_id,
                "action_count": len(actions),
                "action_hash": canonical_action_hash(actions),
            }
        )
    payload = {
        "schema": "ascent_v1_4_vpr_shadow_action_reference_v1",
        "role": "engineering_sentinel",
        "behavior_reference": "frozen_submap_v1.2",
        "action_hash_contract": ACTION_HASH_CONTRACT,
        "source_commit": args.expected_source_commit,
        "split_manifest_sha256": sha256(args.split_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "baseline_summary_sha256": sha256(args.baseline_summary),
        "baseline_episodes_sha256": sha256(args.baseline_episodes),
        "episode_count": len(entries),
        "episodes": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**payload, "output_sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
