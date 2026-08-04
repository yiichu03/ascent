#!/usr/bin/env python3
"""Prepare hash-bound official-val shards for frozen submap-v1.2.

Episode identity is selected by canonical index before navigation metrics are
joined.  Materialized Habitat inputs contain only selected episodes and goal
metadata; scene, navmesh, and semantic assets remain shared in place.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from materialize_submap_screen import materialize


SELECTION_FIELDS = (
    "manifest_version",
    "selection_rule",
    "dataset_case_index",
    "logical_case_id",
    "dataset",
    "split",
    "chunk_id",
    "source_root_file",
    "source_root_sha256",
    "scientific_source_root_file",
    "scientific_source_root_sha256",
    "source_content_file",
    "source_content_sha256",
    "source_episode_index",
    "source_episode_id",
    "scene_id",
    "scene_key",
    "target_category",
    "geodesic_distance",
    "euclidean_distance",
    "episode_seed",
)

BASELINE_FIELDS = (
    "chunk_id",
    "runtime_episode_id",
    "logical_case_id",
    "dataset",
    "arm",
    "map_pose_source",
    "control_pose_source",
    "episode_seed",
    "arm_order_index",
    "scene_id",
    "action_steps",
    "success",
    "spl",
    "soft_spl",
    "distance_to_goal",
    "selected_attempt",
    "selected_attempt_priority",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_gzip_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with path.open("xb") as stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0
        ) as gzip_stream:
            gzip_stream.write(raw)


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def shard_sizes(total: int, shards: int) -> list[int]:
    quotient, remainder = divmod(total, shards)
    return [
        quotient + (1 if index < remainder else 0)
        for index in range(shards)
    ]


def canonical_ids(dataset: str, total: int) -> list[str]:
    return [f"{dataset}_val_{index:04d}" for index in range(total)]


def build_selection(
    *,
    dataset: str,
    manifest_rows: Sequence[Mapping[str, str]],
    vo_rows: Sequence[Mapping[str, str]],
    source_root: Path,
    transport_root: Path,
    start_index: int,
    episodes: int,
    chunk_size: int,
    habitat_seed: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected_total = len(manifest_rows)
    expected_ids = canonical_ids(dataset, expected_total)
    observed_ids = [row["case_id"] for row in manifest_rows]
    if observed_ids != expected_ids:
        raise ValueError(
            f"official manifest is not canonical {dataset} val order"
        )
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("duplicate official logical identity")

    vo_by_id: dict[str, Mapping[str, str]] = {}
    for row in vo_rows:
        case_id = row.get("case_id", "")
        if case_id in vo_by_id:
            raise ValueError(f"duplicate VO baseline identity: {case_id}")
        if row.get("technical_valid") != "1":
            raise ValueError(f"VO baseline not technical-valid: {case_id}")
        vo_by_id[case_id] = row
    if set(vo_by_id) != set(observed_ids):
        raise ValueError("VO baseline does not cover the official manifest")

    selected = manifest_rows[start_index : start_index + episodes]
    if len(selected) != episodes:
        raise ValueError(
            f"selected {len(selected)} rows, expected {episodes}"
        )
    transport_sha = sha256(transport_root)
    source_root_sha = sha256(source_root)
    payload_cache: dict[str, dict[str, Any]] = {}
    content_hashes: dict[str, str] = {}
    selection_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for local_index, row in enumerate(selected):
        case_id = row["case_id"]
        global_index = start_index + local_index
        if row.get("dataset") != dataset or row.get("split") != "val":
            raise ValueError(f"unexpected dataset/split: {case_id}")
        source_path = Path(row["source_content_file"]).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_key = str(source_path)
        if source_key not in payload_cache:
            payload_cache[source_key] = read_gzip_json(source_path)
            content_hashes[source_key] = sha256(source_path)
        payload = payload_cache[source_key]
        source_episode_index = int(row["source_episode_index"])
        episode = payload["episodes"][source_episode_index]
        for field, expected in {
            "episode_id": row["episode_id"],
            "scene_id": row["scene_id"],
            "object_category": row["target_category"],
        }.items():
            if str(episode.get(field)) != str(expected):
                raise ValueError(
                    f"{case_id} {field}: {episode.get(field)!r} "
                    f"!= {expected!r}"
                )
        info = episode.get("info") or {}
        geodesic = float(info["geodesic_distance"])
        euclidean = float(info["euclidean_distance"])
        chunk_id = f"{label}_c{local_index // chunk_size:03d}"
        selection_rows.append(
            {
                "manifest_version": (
                    "ascent_vo_submap_v1_2_official_val_expansion_v1"
                ),
                "selection_rule": (
                    "canonical_official_val_contiguous_index_no_metric_filter"
                ),
                "dataset_case_index": global_index,
                "logical_case_id": case_id,
                "dataset": dataset,
                "split": "val",
                "chunk_id": chunk_id,
                "source_root_file": str(transport_root),
                "source_root_sha256": transport_sha,
                "scientific_source_root_file": str(source_root),
                "scientific_source_root_sha256": source_root_sha,
                "source_content_file": str(source_path),
                "source_content_sha256": content_hashes[source_key],
                "source_episode_index": source_episode_index,
                "source_episode_id": row["episode_id"],
                "scene_id": row["scene_id"],
                "scene_key": row["source_scene"],
                "target_category": row["target_category"],
                "geodesic_distance": f"{geodesic:.8f}",
                "euclidean_distance": f"{euclidean:.8f}",
                "episode_seed": habitat_seed,
            }
        )

        baseline = vo_by_id[case_id]
        for field, expected in {
            "dataset": dataset,
            "split": "val",
            "scene_id": row["scene_id"],
            "episode_id": row["episode_id"],
            "target_category": row["target_category"],
            "source_content_file": row["source_content_file"],
            "source_episode_index": row["source_episode_index"],
        }.items():
            if str(baseline.get(field)) != str(expected):
                raise ValueError(
                    f"{case_id} baseline {field}: "
                    f"{baseline.get(field)!r} != {expected!r}"
                )
        baseline_rows.append(
            {
                "chunk_id": chunk_id,
                "runtime_episode_id": baseline["runtime_episode_id"],
                "logical_case_id": case_id,
                "dataset": dataset,
                "arm": "A",
                "map_pose_source": "vo",
                "control_pose_source": "vo",
                "episode_seed": habitat_seed,
                "arm_order_index": 0,
                "scene_id": row["scene_id"],
                "action_steps": baseline["action_steps"],
                "success": baseline["success"],
                "spl": baseline["spl"],
                "soft_spl": baseline["soft_spl"],
                "distance_to_goal": baseline["distance_to_goal"],
                "selected_attempt": "official_ascent_vo_terminal",
                "selected_attempt_priority": baseline["selected_priority"],
            }
        )

    metadata = {
        "source_content_file_count": len(payload_cache),
        "first_logical_case_id": selection_rows[0]["logical_case_id"],
        "last_logical_case_id": selection_rows[-1]["logical_case_id"],
    }
    return selection_rows, baseline_rows, metadata


def materialize_view(
    *,
    selection: Path,
    output_root: Path,
    scene_dataset_config: Path,
    dataset: str,
    episodes: int,
    chunk_size: int,
    label: str,
) -> dict[str, Any]:
    return materialize(
        SimpleNamespace(
            selection=selection,
            output_root=output_root,
            scene_dataset_config=scene_dataset_config,
            dataset=dataset,
            episodes=episodes,
            start_index=0,
            chunk_size=chunk_size,
            label=label,
        )
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.episode_manifest,
        args.vo_episodes,
        args.source_root_file,
        args.scene_dataset_config,
    ):
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    if args.start_index + args.episodes > args.expected_total:
        raise ValueError("selected range exceeds expected official total")
    if args.episodes < args.shards * 5:
        raise ValueError("every shard requires at least five gate episodes")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    episode_manifest = args.episode_manifest.resolve()
    vo_episodes = args.vo_episodes.resolve()
    source_root = args.source_root_file.resolve()
    scene_dataset_config = args.scene_dataset_config.resolve()
    manifest_rows = read_csv(episode_manifest)
    vo_rows = read_csv(vo_episodes)
    if len(manifest_rows) != args.expected_total:
        raise ValueError("official manifest total mismatch")
    if len(vo_rows) != args.expected_total:
        raise ValueError("VO baseline total mismatch")

    scientific_root_payload = read_gzip_json(source_root)
    transport_payload = dict(scientific_root_payload)
    transport_payload.pop("content_scenes_path", None)
    transport_payload["episodes"] = []
    transport_payload["goals_by_category"] = {}
    transport_root = output_root / "transport_root.json.gz"
    write_gzip_json(transport_root, transport_payload)

    selection_rows, baseline_rows, selection_meta = build_selection(
        dataset=args.dataset,
        manifest_rows=manifest_rows,
        vo_rows=vo_rows,
        source_root=source_root,
        transport_root=transport_root,
        start_index=args.start_index,
        episodes=args.episodes,
        chunk_size=args.chunk_size,
        habitat_seed=args.habitat_seed,
        label=args.label,
    )
    aggregate_selection = output_root / "selection.csv"
    aggregate_baseline = output_root / "ascent_vo_baseline.csv"
    write_csv(aggregate_selection, selection_rows, SELECTION_FIELDS)
    write_csv(aggregate_baseline, baseline_rows, BASELINE_FIELDS)

    sizes = shard_sizes(args.episodes, args.shards)
    jobs = []
    offset = 0
    for shard_index, size in enumerate(sizes):
        shard_root = output_root / f"shard_{shard_index}"
        shard_root.mkdir()
        shard_selection_rows = selection_rows[offset : offset + size]
        shard_baseline_rows = baseline_rows[offset : offset + size]
        selection_path = shard_root / "selection.csv"
        baseline_path = shard_root / "ascent_vo_baseline.csv"
        audit_path = shard_root / "preparation_audit.json"
        write_csv(selection_path, shard_selection_rows, SELECTION_FIELDS)
        write_csv(baseline_path, shard_baseline_rows, BASELINE_FIELDS)
        shard_label = f"{args.label}_s{shard_index}"
        main = materialize_view(
            selection=selection_path,
            output_root=shard_root / "materialized",
            scene_dataset_config=scene_dataset_config,
            dataset=args.dataset,
            episodes=size,
            chunk_size=args.chunk_size,
            label=shard_label,
        )
        gate = materialize_view(
            selection=selection_path,
            output_root=shard_root / "gate_materialized",
            scene_dataset_config=scene_dataset_config,
            dataset=args.dataset,
            episodes=5,
            chunk_size=5,
            label=f"{shard_label}_gate5",
        )
        shard_start = args.start_index + offset
        audit = {
            "schema": (
                "ascent_vo_submap_v1_2_official_val_preparation_v1"
            ),
            "status": "PASS",
            "dataset": args.dataset,
            "split": "val",
            "selection_rule": (
                "canonical_official_val_contiguous_index_no_metric_filter"
            ),
            "metrics_used_for_selection": False,
            "expected_official_total": args.expected_total,
            "selected_start_index": shard_start,
            "selected_stop_index_exclusive": shard_start + size,
            "selected_episode_count": size,
            "matched_ascent_vo_count": size,
            "chunk_size": args.chunk_size,
            "chunk_count": main["chunk_count"],
            "first_logical_case_id": shard_selection_rows[0][
                "logical_case_id"
            ],
            "last_logical_case_id": shard_selection_rows[-1][
                "logical_case_id"
            ],
            "habitat_seed": args.habitat_seed,
            "episode_manifest": str(episode_manifest),
            "episode_manifest_sha256": sha256(episode_manifest),
            "vo_episodes": str(vo_episodes),
            "vo_episodes_sha256": sha256(vo_episodes),
            "source_root_file": str(source_root),
            "source_root_file_sha256": sha256(source_root),
            "transport_root_file": str(transport_root),
            "transport_root_file_sha256": sha256(transport_root),
            "transport_root_content_scenes_path_removed": True,
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "ascent_vo_baseline": str(baseline_path),
            "ascent_vo_baseline_sha256": sha256(baseline_path),
            "main_manifest": main["manifest_path"],
            "main_manifest_sha256": main["manifest_sha256"],
            "gate_manifest": gate["manifest_path"],
            "gate_manifest_sha256": gate["manifest_sha256"],
        }
        write_json(audit_path, audit)
        config_path = shard_root / "job_config.json"
        job_key = f"{args.label}_s{shard_index}"
        config = {
            "schema": "ascent_vo_submap_v1_2_official_val_job_v1",
            "job_key": job_key,
            "job_name": (
                "v12hmt" if args.dataset == "hm3d" else f"v12m{shard_index}"
            ),
            "dataset": args.dataset,
            "scientific_split": "val",
            "transport_split": "train",
            "scientific_start_index": shard_start,
            "scientific_stop_index_exclusive": shard_start + size,
            "expected_official_total": args.expected_total,
            "expected_episodes": size,
            "expected_chunks": main["chunk_count"],
            "gate_episodes": 5,
            "gate_chunks": 1,
            "lane_count": 3,
            "artifact_root": str(args.artifact_root.resolve()),
            "manifest": main["manifest_path"],
            "manifest_sha256": main["manifest_sha256"],
            "gate_manifest": gate["manifest_path"],
            "gate_manifest_sha256": gate["manifest_sha256"],
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "baseline_csv": str(baseline_path),
            "baseline_sha256": sha256(baseline_path),
            "preparation_audit": str(audit_path),
            "preparation_audit_sha256": sha256(audit_path),
            "selection_rule": audit["selection_rule"],
            "metrics_used_for_selection": False,
            "pose_source": "zhao_rgbd_2021",
            "policy_gt_isolation": True,
            "evaluation_gt_only": True,
            "method_version": "submap_v1.2",
            "fixed_technical_retry_per_failed_unit": 1,
            "metric_driven_retry": False,
        }
        write_json(config_path, config)
        jobs.append(
            {
                "job_key": job_key,
                "job_config": str(config_path),
                "job_config_sha256": sha256(config_path),
                "dataset": args.dataset,
                "shard_index": shard_index,
                "start_index": shard_start,
                "episodes": size,
                "chunks": main["chunk_count"],
            }
        )
        offset += size

    if offset != args.episodes:
        raise RuntimeError("shard accounting mismatch")
    plan = {
        "schema": "ascent_vo_submap_v1_2_official_val_plan_v1",
        "status": "PASS",
        "dataset": args.dataset,
        "expected_official_total": args.expected_total,
        "selected_start_index": args.start_index,
        "selected_episode_count": args.episodes,
        "shard_sizes": sizes,
        "selection_rule": (
            "canonical_official_val_contiguous_index_no_metric_filter"
        ),
        "metrics_used_for_selection": False,
        "aggregate_selection": str(aggregate_selection),
        "aggregate_selection_sha256": sha256(aggregate_selection),
        "aggregate_baseline": str(aggregate_baseline),
        "aggregate_baseline_sha256": sha256(aggregate_baseline),
        "transport_root": str(transport_root),
        "transport_root_sha256": sha256(transport_root),
        "source_content_file_count": selection_meta[
            "source_content_file_count"
        ],
        "jobs": jobs,
    }
    plan_path = output_root / "submission_plan.json"
    write_json(plan_path, plan)
    plan["submission_plan"] = str(plan_path)
    plan["submission_plan_sha256"] = sha256(plan_path)
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("hm3d", "mp3d"), required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--vo-episodes", type=Path, required=True)
    parser.add_argument("--source-root-file", type=Path, required=True)
    parser.add_argument("--scene-dataset-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--habitat-seed", type=int, default=100)
    parser.add_argument("--label", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.episodes,
        args.expected_total,
        args.shards,
        args.chunk_size,
    ) < 1 or args.start_index < 0:
        raise SystemExit("counts must be positive and start-index nonnegative")
    result = prepare(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
