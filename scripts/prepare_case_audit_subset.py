#!/usr/bin/env python3
"""Prepare three balanced HM3D train case-audit shards from existing traces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


TARGETS = ("bed", "chair", "plant", "sofa", "toilet", "tv_monitor")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scene_round_robin(rows: Sequence[Dict[str, str]]) -> list[Dict[str, str]]:
    by_scene: Dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in sorted(
        rows,
        key=lambda item: (
            -int(float(item["steps"])),
            -float(item["spatial_revisit_proxy_score"]),
            item["project_row_id"],
        ),
    ):
        by_scene[row["source_scene_key"]].append(row)
    scenes = sorted(
        by_scene,
        key=lambda scene: (
            -int(float(by_scene[scene][0]["steps"])),
            -float(by_scene[scene][0]["spatial_revisit_proxy_score"]),
            scene,
        ),
    )
    output = []
    round_index = 0
    while True:
        added = False
        for scene in scenes:
            if round_index < len(by_scene[scene]):
                output.append(by_scene[scene][round_index])
                added = True
        if not added:
            return output
        round_index += 1


def select(rows: Sequence[Dict[str, str]]) -> list[Dict[str, str]]:
    eligible = [
        row
        for row in rows
        if row.get("dataset") == "hm3d"
        and row.get("official_source_split") == "train"
        and row.get("technical_ok") == "1"
        and row.get("analysis_ready") == "1"
        and float(row.get("success", 0.0)) >= 0.5
    ]
    selected = []
    for target in TARGETS:
        target_rows = [
            row for row in eligible if row.get("target_category") == target
        ]
        revisit = scene_round_robin(
            [
                row
                for row in target_rows
                if row.get("spatial_revisit_proxy_eligible") == "1"
            ]
        )
        chosen = revisit[:30]
        if len(chosen) < 30:
            chosen_ids = {row["project_row_id"] for row in chosen}
            controls = scene_round_robin(
                [
                    row
                    for row in target_rows
                    if row["project_row_id"] not in chosen_ids
                ]
            )
            chosen.extend(controls[: 30 - len(chosen)])
        if len(chosen) != 30:
            raise ValueError(f"target {target} has only {len(chosen)} cases")
        selected.extend(chosen)
    if len(selected) != 180 or len(
        {row["project_row_id"] for row in selected}
    ) != 180:
        raise ValueError("selection is not 180 unique episodes")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-root-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    features_path = args.features.resolve()
    source_root = args.source_root_file.resolve()
    output_root = args.output_root.resolve()
    if not features_path.is_file() or not source_root.is_file():
        raise FileNotFoundError("features or source root is missing")
    output_root.mkdir(parents=True, exist_ok=False)

    selected = select(read_csv(features_path))
    selected_by_target = {
        target: [row for row in selected if row["target_category"] == target]
        for target in TARGETS
    }
    shards: list[list[Dict[str, str]]] = [[], [], []]
    for target in TARGETS:
        for index, row in enumerate(selected_by_target[target]):
            shards[index % 3].append(row)

    source_cache: Dict[str, Dict[str, Any]] = {}
    hash_cache: Dict[str, str] = {}
    preparation = []
    selection_fields = (
        "manifest_version",
        "selection_rule",
        "dataset_case_index",
        "logical_case_id",
        "dataset",
        "split",
        "chunk_id",
        "source_root_file",
        "source_root_sha256",
        "source_content_file",
        "source_content_sha256",
        "source_episode_index",
        "source_episode_id",
        "scene_id",
        "scene_key",
        "target_category",
        "geodesic_distance",
        "episode_seed",
        "original_steps",
        "spatial_revisit_proxy_eligible",
        "spatial_revisit_proxy_score",
    )
    baseline_fields = (
        "logical_case_id",
        "scene_id",
        "original_success",
        "original_spl",
        "original_action_steps",
        "spatial_revisit_proxy_eligible",
        "spatial_revisit_proxy_score",
    )
    root_hash = sha256(source_root)
    for shard_index, shard_rows in enumerate(shards):
        shard_root = output_root / f"shard{shard_index}"
        shard_root.mkdir()
        selection_rows = []
        baseline_rows = []
        for dataset_index, source_row in enumerate(shard_rows):
            source_path = Path(source_row["source_episode_file"]).resolve()
            source_key = str(source_path)
            if source_key not in source_cache:
                with gzip.open(
                    source_path, "rt", encoding="utf-8"
                ) as handle:
                    source_cache[source_key] = json.load(handle)
                hash_cache[source_key] = sha256(source_path)
            episode_index = int(source_row["source_episode_index"])
            episode = source_cache[source_key]["episodes"][episode_index]
            checks = {
                "episode_id": source_row["official_episode_id"],
                "scene_id": source_row["source_scene_id"],
                "object_category": source_row["target_category"],
            }
            if any(str(episode.get(key)) != str(value) for key, value in checks.items()):
                raise ValueError(
                    f"source identity mismatch: {source_row['project_row_id']}"
                )
            geodesic = float((episode.get("info") or {})["geodesic_distance"])
            logical_id = source_row["project_row_id"]
            selection_rows.append(
                {
                    "manifest_version": "v1_4_case_audit_train180_v1",
                    "selection_rule": (
                        "original_ascent_success_analysis_ready_target30_"
                        "revisit_first_scene_round_robin_case_mining"
                    ),
                    "dataset_case_index": dataset_index,
                    "logical_case_id": logical_id,
                    "dataset": "hm3d",
                    "split": "train",
                    "chunk_id": f"hm3d_v14_case_s{shard_index}_c{dataset_index // 10:03d}",
                    "source_root_file": str(source_root),
                    "source_root_sha256": root_hash,
                    "source_content_file": source_key,
                    "source_content_sha256": hash_cache[source_key],
                    "source_episode_index": episode_index,
                    "source_episode_id": source_row["official_episode_id"],
                    "scene_id": source_row["source_scene_id"],
                    "scene_key": source_row["source_scene_key"],
                    "target_category": source_row["target_category"],
                    "geodesic_distance": f"{geodesic:.8f}",
                    "episode_seed": 100,
                    "original_steps": int(float(source_row["steps"])),
                    "spatial_revisit_proxy_eligible": source_row[
                        "spatial_revisit_proxy_eligible"
                    ],
                    "spatial_revisit_proxy_score": source_row[
                        "spatial_revisit_proxy_score"
                    ],
                }
            )
            baseline_rows.append(
                {
                    "logical_case_id": logical_id,
                    "scene_id": source_row["source_scene_id"],
                    "original_success": source_row["success"],
                    "original_spl": source_row["spl"],
                    "original_action_steps": int(float(source_row["steps"])),
                    "spatial_revisit_proxy_eligible": source_row[
                        "spatial_revisit_proxy_eligible"
                    ],
                    "spatial_revisit_proxy_score": source_row[
                        "spatial_revisit_proxy_score"
                    ],
                }
            )
        selection_path = shard_root / "selection.csv"
        baseline_path = shard_root / "original_ascent_baseline.csv"
        write_csv(selection_path, selection_rows, selection_fields)
        write_csv(baseline_path, baseline_rows, baseline_fields)
        preparation.append(
            {
                "shard": shard_index,
                "episode_count": len(selection_rows),
                "scene_count": len(
                    {row["scene_key"] for row in selection_rows}
                ),
                "target_counts": {
                    target: sum(
                        row["target_category"] == target
                        for row in selection_rows
                    )
                    for target in TARGETS
                },
                "revisit_proxy_count": sum(
                    int(row["spatial_revisit_proxy_eligible"])
                    for row in selection_rows
                ),
                "selection": str(selection_path),
                "selection_sha256": sha256(selection_path),
                "original_baseline": str(baseline_path),
                "original_baseline_sha256": sha256(baseline_path),
            }
        )
    audit = {
        "schema": "ascent_vo_submap_v1_4_case_audit_preparation_v1",
        "status": "PASS",
        "scientific_role": "mechanism_enriched_case_mining_not_population_estimate",
        "selection_uses_original_ascent_outcome": True,
        "selection_uses_future_b1_or_b2_outcome": False,
        "features": str(features_path),
        "features_sha256": sha256(features_path),
        "source_root_file": str(source_root),
        "source_root_sha256": root_hash,
        "total_episode_count": 180,
        "shards": preparation,
    }
    audit_path = output_root / "preparation_audit.json"
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
