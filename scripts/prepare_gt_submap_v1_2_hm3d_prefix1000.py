#!/usr/bin/env python3
"""Bind the existing HM3D val-prefix1000 inputs to the GT+submap control.

The materialized Habitat episodes are immutable transport inputs shared with
the frozen VO+submap-v1.2 run.  This preparer writes only a new hash-bound job
configuration and reuse audit under the GT experiment's ignored artifact root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT = Path(
    "/scratch/e1538633/liuyi/drift-aware-submap-exploration"
)
SOURCE_INPUT_ROOT = (
    PROJECT / "artifacts/objectnav/submap_v1_2/inputs"
)
ARTIFACT_ROOT = PROJECT / "artifacts/objectnav/gt_submap_v1_2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument(
        "--artifact-root", type=Path, default=ARTIFACT_ROOT
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    source_inputs = project / "artifacts/objectnav/submap_v1_2/inputs"
    artifact_root = args.artifact_root.resolve()

    selection = (
        source_inputs
        / "hm3d_val1000_canonical_20260802_v2/selection.csv"
    )
    baseline = (
        source_inputs
        / "hm3d_val1000_canonical_20260802_v2/ascent_vo_baseline.csv"
    )
    preparation = (
        source_inputs
        / "hm3d_val1000_canonical_20260802_v2/preparation_audit.json"
    )
    manifest = (
        source_inputs
        / "hm3d_val1000_materialized_20260802_v3/chunk_manifest.json"
    )
    gate_manifest = (
        source_inputs
        / "hm3d_val5_materialized_20260802_v3/chunk_manifest.json"
    )
    inputs = (selection, baseline, preparation, manifest, gate_manifest)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")

    rows = read_csv(selection)
    expected_ids = [f"hm3d_val_{index:04d}" for index in range(1000)]
    if [row["logical_case_id"] for row in rows] != expected_ids:
        raise ValueError("selection is not canonical HM3D val prefix1000")
    if [int(row["dataset_case_index"]) for row in rows] != list(
        range(1000)
    ):
        raise ValueError("selection dataset indices are not 0..999")
    baseline_rows = read_csv(baseline)
    if [row["logical_case_id"] for row in baseline_rows] != expected_ids:
        raise ValueError("ASCENT+VO reference is not episode matched")

    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    gate_value = json.loads(gate_manifest.read_text(encoding="utf-8"))
    if (
        manifest_value.get("episode_count") != 1000
        or len(manifest_value.get("chunks", [])) != 50
        or manifest_value.get("logical_case_ids") != expected_ids
    ):
        raise ValueError("main transport manifest contract mismatch")
    if (
        gate_value.get("episode_count") != 5
        or len(gate_value.get("chunks", [])) != 1
        or gate_value.get("logical_case_ids") != expected_ids[:5]
    ):
        raise ValueError("five-episode gate manifest contract mismatch")

    output_dir = artifact_root / "inputs/hm3d_val_prefix1000_20260807_v1"
    job_config_path = output_dir / "job_config.json"
    reuse_audit_path = output_dir / "reuse_audit.json"
    job_config = {
        "schema": "ascent_gt_submap_v1_2_official_val_job_v1",
        "job_key": "hm3d_val_prefix1000_s0",
        "job_name": "gtsm12h1k",
        "dataset": "hm3d",
        "artifact_root": str(artifact_root),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "gate_manifest": str(gate_manifest.resolve()),
        "gate_manifest_sha256": sha256(gate_manifest),
        "selection": str(selection.resolve()),
        "selection_sha256": sha256(selection),
        "baseline_csv": str(baseline.resolve()),
        "baseline_sha256": sha256(baseline),
        "preparation_audit": str(preparation.resolve()),
        "preparation_audit_sha256": sha256(preparation),
        "expected_episodes": 1000,
        "expected_chunks": 50,
        "scientific_start_index": 0,
        "scientific_stop_index_exclusive": 1000,
        "expected_official_total": 2000,
        "scientific_split": "val",
        "transport_split": "train",
        "selection_rule": "canonical_official_val_prefix_no_metric_filter",
        "metrics_used_for_selection": False,
        "pose_source": "habitat_ground_truth",
        "policy_gt_isolation": False,
        "auxiliary_vo_sensors": False,
        "zhao_checkpoint_loaded": False,
        "evaluation_gt_only": True,
        "method_version": "submap_v1.2",
        "fixed_technical_retry_per_failed_unit": 1,
        "metric_driven_retry": False,
        "gate_episodes": 5,
        "gate_chunks": 1,
        "lane_count": 3,
        "comparison_reference": "episode_matched_ascent_vo_read_only",
        "transport_reuse": "frozen_submap_v1_2_prefix1000_no_copy",
    }
    reuse_audit = {
        "schema": "ascent_gt_submap_v1_2_input_reuse_v1",
        "status": "PASS",
        "claim": "GT pose plus frozen submap-v1.2 specificity control",
        "episode_count": 1000,
        "chunk_count": 50,
        "metrics_used_for_selection": False,
        "transport_files_copied": False,
        "source_input_root": str(source_inputs.resolve()),
        "job_config": str(job_config_path),
        "bound_inputs": {
            path.name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in inputs
        },
    }
    write_json_exclusive(job_config_path, job_config)
    write_json_exclusive(reuse_audit_path, reuse_audit)
    print(job_config_path)
    print(reuse_audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
