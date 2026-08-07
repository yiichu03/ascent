#!/usr/bin/env python3
"""Bind canonical HM3D val-prefix1000 shards to the GT+submap control.

The three 334/333/333 Habitat transport shards were materialized before the
GT control was proposed. They are method-agnostic episode containers, so this
preparer verifies their identity against the frozen submap-v1.2 prefix and
writes only GT-specific, hash-bound job configurations. No episode, scene,
navmesh, or semantic asset is copied.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROJECT = Path("/scratch/e1538633/liuyi/drift-aware-submap-exploration")
ARTIFACT_ROOT = PROJECT / "artifacts/objectnav/gt_submap_v1_2"
OUTPUT_VERSION = "hm3d_val_prefix1000_3pbs_20260807_v2"
SHARD_SIZES = (334, 333, 333)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} hash mismatch: expected={expected}, observed={observed}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    project = args.project_root.resolve()
    artifact_root = args.artifact_root.resolve()
    v12_inputs = project / "artifacts/objectnav/submap_v1_2/inputs"
    transport_root = (
        project
        / "artifacts/objectnav/submap_v1_3/inputs/"
        "hm3d_val1000_20260806_v1"
    )

    v12_selection = (
        v12_inputs / "hm3d_val1000_canonical_20260802_v2/selection.csv"
    )
    v12_baseline = (
        v12_inputs
        / "hm3d_val1000_canonical_20260802_v2/ascent_vo_baseline.csv"
    )
    gate_manifest = (
        v12_inputs
        / "hm3d_val5_materialized_20260802_v3/chunk_manifest.json"
    )
    source_plan_path = transport_root / "submission_plan.json"
    required = (v12_selection, v12_baseline, gate_manifest, source_plan_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen input(s): {missing}")

    expected_ids = [f"hm3d_val_{index:04d}" for index in range(1000)]
    v12_selection_rows = read_csv(v12_selection)
    v12_baseline_rows = read_csv(v12_baseline)
    if [row["logical_case_id"] for row in v12_selection_rows] != expected_ids:
        raise ValueError("submap-v1.2 selection is not HM3D val prefix1000")
    if [row["logical_case_id"] for row in v12_baseline_rows] != expected_ids:
        raise ValueError("submap-v1.2 ASCENT+VO baseline is not episode matched")
    baseline_by_id = {
        row["logical_case_id"]: row for row in v12_baseline_rows
    }
    selection_by_id = {
        row["logical_case_id"]: row for row in v12_selection_rows
    }

    gate = read_json(gate_manifest)
    smoke_ids = expected_ids[:5]
    if (
        gate.get("dataset") != "hm3d"
        or gate.get("split") != "train"
        or gate.get("episode_count") != 5
        or len(gate.get("chunks", [])) != 1
        or gate.get("logical_case_ids") != smoke_ids
    ):
        raise ValueError("five-episode gate manifest contract mismatch")

    source_plan = read_json(source_plan_path)
    require_hash(
        Path(source_plan["aggregate_selection"]),
        source_plan["aggregate_selection_sha256"],
        "source aggregate selection",
    )
    require_hash(
        Path(source_plan["aggregate_baseline"]),
        source_plan["aggregate_baseline_sha256"],
        "source aggregate baseline",
    )
    plan_checks = {
        "schema": source_plan.get("schema")
        == "ascent_vo_submap_v1_3_official_val_plan_v1",
        "status": source_plan.get("status") == "PASS",
        "dataset": source_plan.get("dataset") == "hm3d",
        "start": source_plan.get("selected_start_index") == 0,
        "count": source_plan.get("selected_episode_count") == 1000,
        "metric_free": source_plan.get("metrics_used_for_selection") is False,
        "sizes": source_plan.get("shard_sizes") == list(SHARD_SIZES),
        "jobs": len(source_plan.get("jobs", [])) == 3,
    }
    if not all(plan_checks.values()):
        raise ValueError(f"source transport plan mismatch: {plan_checks}")

    transport_root_file = Path(source_plan["transport_root"])
    require_hash(
        transport_root_file,
        source_plan["transport_root_sha256"],
        "transport root",
    )
    with gzip.open(transport_root_file, "rt", encoding="utf-8") as handle:
        root_payload = json.load(handle)
    if "content_scenes_path" in root_payload or root_payload.get("episodes") != []:
        raise ValueError("transport root is not an empty method-agnostic root")

    output_root = artifact_root / "inputs" / OUTPUT_VERSION
    jobs: list[dict[str, Any]] = []
    offset = 0
    for shard_index, (source_job, expected_size) in enumerate(
        zip(source_plan["jobs"], SHARD_SIZES)
    ):
        source_config_path = Path(source_job["job_config"])
        require_hash(
            source_config_path,
            source_job["job_config_sha256"],
            f"source shard {shard_index} config",
        )
        source_config = read_json(source_config_path)
        start, stop = offset, offset + expected_size
        source_checks = {
            "schema": source_config.get("schema")
            == "ascent_vo_submap_v1_3_official_val_job_v1",
            "dataset": source_config.get("dataset") == "hm3d",
            "start": source_config.get("scientific_start_index") == start,
            "stop": source_config.get("scientific_stop_index_exclusive")
            == stop,
            "episodes": source_config.get("expected_episodes")
            == expected_size,
            "metric_free": source_config.get("metrics_used_for_selection")
            is False,
        }
        if not all(source_checks.values()):
            raise ValueError(
                f"source shard {shard_index} mismatch: {source_checks}"
            )

        selection_path = Path(source_config["selection"])
        baseline_path = Path(source_config["baseline_csv"])
        manifest_path = Path(source_config["manifest"])
        require_hash(
            selection_path,
            source_config["selection_sha256"],
            f"shard {shard_index} selection",
        )
        require_hash(
            baseline_path,
            source_config["baseline_sha256"],
            f"shard {shard_index} baseline",
        )
        require_hash(
            manifest_path,
            source_config["manifest_sha256"],
            f"shard {shard_index} manifest",
        )
        shard_ids = expected_ids[start:stop]
        selection_rows = read_csv(selection_path)
        baseline_rows = read_csv(baseline_path)
        if [row["logical_case_id"] for row in selection_rows] != shard_ids:
            raise ValueError(f"shard {shard_index} selection identity mismatch")
        if [row["logical_case_id"] for row in baseline_rows] != shard_ids:
            raise ValueError(f"shard {shard_index} baseline identity mismatch")
        for row in selection_rows:
            frozen = selection_by_id[row["logical_case_id"]]
            if (
                row.get("scene_id") != frozen.get("scene_id")
                or row.get("target_category") != frozen.get("target_category")
            ):
                raise ValueError(
                    f"transport episode differs from v1.2: "
                    f"{row['logical_case_id']}"
                )
        for row in baseline_rows:
            frozen = baseline_by_id[row["logical_case_id"]]
            for field in ("success", "spl", "action_steps"):
                if row.get(field) != frozen.get(field):
                    raise ValueError(
                        f"baseline differs from v1.2: "
                        f"{row['logical_case_id']}:{field}"
                    )

        manifest = read_json(manifest_path)
        if (
            manifest.get("dataset") != "hm3d"
            or manifest.get("split") != "train"
            or manifest.get("logical_case_ids") != shard_ids
            or manifest.get("episode_count") != expected_size
            or len(manifest.get("chunks", []))
            != int(source_config["expected_chunks"])
        ):
            raise ValueError(f"shard {shard_index} manifest identity mismatch")

        shard_root = output_root / f"shard_{shard_index}"
        audit_path = shard_root / "preparation_audit.json"
        audit = {
            "schema": "ascent_gt_submap_v1_2_transport_reuse_v2",
            "status": "PASS",
            "claim": "GT pose plus frozen submap-v1.2 specificity control",
            "dataset": "hm3d",
            "split": "val",
            "selection_rule": (
                "canonical_official_val_contiguous_index_no_metric_filter"
            ),
            "metrics_used_for_selection": False,
            "selected_start_index": start,
            "selected_stop_index_exclusive": stop,
            "selected_episode_count": expected_size,
            "matched_ascent_vo_count": expected_size,
            "chunk_count": len(manifest["chunks"]),
            "transport_files_copied": False,
            "transport_root_file": str(transport_root_file.resolve()),
            "transport_root_file_sha256": sha256(transport_root_file),
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256(selection_path),
            "ascent_vo_baseline": str(baseline_path.resolve()),
            "ascent_vo_baseline_sha256": sha256(baseline_path),
            "main_manifest": str(manifest_path.resolve()),
            "main_manifest_sha256": sha256(manifest_path),
            "source_transport_plan": str(source_plan_path.resolve()),
            "source_transport_plan_sha256": sha256(source_plan_path),
            "source_transport_job_config": str(source_config_path.resolve()),
            "source_transport_job_config_sha256": sha256(source_config_path),
        }
        write_json_exclusive(audit_path, audit)

        config_path = shard_root / "job_config.json"
        job_key = f"hm3d_val1000_gt_submap_s{shard_index}"
        config = {
            "schema": "ascent_gt_submap_v1_2_official_val_job_v1",
            "job_key": job_key,
            "job_name": f"gtsm12h{shard_index}",
            "dataset": "hm3d",
            "artifact_root": str(artifact_root),
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "gate_manifest": str(gate_manifest.resolve()),
            "gate_manifest_sha256": sha256(gate_manifest),
            "smoke_logical_case_ids": smoke_ids,
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256(selection_path),
            "baseline_csv": str(baseline_path.resolve()),
            "baseline_sha256": sha256(baseline_path),
            "preparation_audit": str(audit_path.resolve()),
            "preparation_audit_sha256": sha256(audit_path),
            "expected_episodes": expected_size,
            "expected_chunks": len(manifest["chunks"]),
            "scientific_start_index": start,
            "scientific_stop_index_exclusive": stop,
            "expected_official_total": 2000,
            "scientific_split": "val",
            "transport_split": "train",
            "selection_rule": audit["selection_rule"],
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
            "transport_reuse": "verified_v1_3_container_method_v1_2_behavior",
        }
        write_json_exclusive(config_path, config)
        jobs.append(
            {
                "job_key": job_key,
                "job_name": config["job_name"],
                "job_config": str(config_path.resolve()),
                "job_config_sha256": sha256(config_path),
                "shard_index": shard_index,
                "start_index": start,
                "stop_index_exclusive": stop,
                "episodes": expected_size,
                "chunks": len(manifest["chunks"]),
            }
        )
        offset = stop

    if offset != 1000:
        raise RuntimeError("three-shard accounting mismatch")
    plan_path = output_root / "submission_plan.json"
    plan = {
        "schema": "ascent_gt_submap_v1_2_official_val_plan_v2",
        "status": "PASS",
        "dataset": "hm3d",
        "scientific_split": "val",
        "selected_start_index": 0,
        "selected_episode_count": 1000,
        "expected_official_total": 2000,
        "shard_sizes": list(SHARD_SIZES),
        "selection_rule": (
            "canonical_official_val_contiguous_index_no_metric_filter"
        ),
        "metrics_used_for_selection": False,
        "pose_source": "habitat_ground_truth",
        "method_version": "submap_v1.2",
        "transport_files_copied": False,
        "smoke_manifest": str(gate_manifest.resolve()),
        "smoke_manifest_sha256": sha256(gate_manifest),
        "smoke_logical_case_ids": smoke_ids,
        "aggregate_selection": str(v12_selection.resolve()),
        "aggregate_selection_sha256": sha256(v12_selection),
        "aggregate_baseline": str(v12_baseline.resolve()),
        "aggregate_baseline_sha256": sha256(v12_baseline),
        "source_transport_plan": str(source_plan_path.resolve()),
        "source_transport_plan_sha256": sha256(source_plan_path),
        "jobs": jobs,
    }
    write_json_exclusive(plan_path, plan)
    print(plan_path)
    for job in jobs:
        print(job["job_config"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
