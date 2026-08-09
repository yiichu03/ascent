#!/usr/bin/env python3
"""Run retrieval and RGB-D verification after frozen v1.2 navigation ends."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from vpr_shadow_data import (
        QueryEvent,
        ShadowFrame,
        build_query_events,
        load_keyframes,
        load_submap_graphs,
        sha256,
    )
    from vpr_shadow_geometry import (
        aggregate_multiframe_support,
        decode_normalized_depth,
        geometry_contract,
        lift_matched_keypoints,
        submap_transform_from_pair,
        verify_rigid_correspondences,
    )
    from vpr_shadow_models import (
        GlobalRetriever,
        LocalGeometryMatcher,
        load_and_validate_registry,
    )
except ImportError:  # imported through the repository root in unit tests
    from scripts.vpr_shadow_data import (
        QueryEvent,
        ShadowFrame,
        build_query_events,
        load_keyframes,
        load_submap_graphs,
        sha256,
    )
    from scripts.vpr_shadow_geometry import (
        aggregate_multiframe_support,
        decode_normalized_depth,
        geometry_contract,
        lift_matched_keypoints,
        submap_transform_from_pair,
        verify_rigid_correspondences,
    )
    from scripts.vpr_shadow_models import (
        GlobalRetriever,
        LocalGeometryMatcher,
        load_and_validate_registry,
    )


RAW_SCHEMA = "ascent_v1_4_vpr_shadow_raw_candidates_v1"
TOP_CANDIDATE_SUBMAPS = 5
QUERY_WINDOW_SIZE = 3
MIN_QUERY_FRAMES = 2
MIN_STEP_SEPARATION = 30
VO_CONSISTENT_DISTANCE_M = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--submap-diagnostics", type=Path, required=True)
    parser.add_argument("--retriever", choices=("mixvpr", "megaloc"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-keyframe-file-hashes", action="store_true")
    return parser.parse_args()


def _write_jsonl(stream, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _vo_distance(
    query_frames: Sequence[ShadowFrame], historical: Sequence[ShadowFrame]
) -> float:
    return min(
        float(np.linalg.norm(query.world_pose_vo[:2] - old.world_pose_vo[:2]))
        for query in query_frames
        for old in historical
    )


def _candidate_scores(
    event: QueryEvent,
    descriptors: Mapping[str, np.ndarray],
) -> list[tuple[str, float, list[tuple[ShadowFrame, ShadowFrame, float]]]]:
    query_values = np.stack(
        [descriptors[frame.frame_id] for frame in event.query_window]
    )
    candidates = []
    for submap_id, historical_frames in event.candidate_frames.items():
        historical_values = np.stack(
            [descriptors[frame.frame_id] for frame in historical_frames]
        )
        similarities = query_values @ historical_values.T
        pairs = []
        for query_index, query_frame in enumerate(event.query_window):
            historical_index = int(np.argmax(similarities[query_index]))
            pairs.append(
                (
                    query_frame,
                    historical_frames[historical_index],
                    float(similarities[query_index, historical_index]),
                )
            )
        candidates.append((submap_id, float(np.max(similarities)), pairs))
    candidates.sort(key=lambda value: (-value[1], value[0]))
    return candidates[:TOP_CANDIDATE_SUBMAPS]


def _verify_pair(
    *,
    query: ShadowFrame,
    historical: ShadowFrame,
    retrieval_similarity: float,
    matcher: LocalGeometryMatcher,
) -> dict[str, Any]:
    query_keypoints, historical_keypoints, matches = matcher.match(
        query, historical
    )
    query_depth = decode_normalized_depth(query.depth_path)
    historical_depth = decode_normalized_depth(historical.depth_path)
    query_points, historical_points = lift_matched_keypoints(
        keypoints_source=query_keypoints,
        keypoints_target=historical_keypoints,
        matches=matches,
        depth_source=query_depth,
        depth_target=historical_depth,
        source_calibration=query.calibration,
        target_calibration=historical.calibration,
    )
    verification = verify_rigid_correspondences(
        query_points,
        historical_points,
        match_count=len(matches),
        seed_key=f"{query.frame_id}->{historical.frame_id}",
    )
    record = {
        "query_frame_id": query.frame_id,
        "historical_frame_id": historical.frame_id,
        "retrieval_similarity": float(retrieval_similarity),
        **verification.as_dict(),
        "submap_transform_target_from_source": None,
    }
    if verification.transform_target_from_source is not None:
        submap_transform = submap_transform_from_pair(
            target_submap_from_target_camera=historical.tf_camera_to_submap,
            target_camera_from_source_camera=(
                verification.transform_target_from_source
            ),
            source_submap_from_source_camera=query.tf_camera_to_submap,
        )
        record["submap_transform_target_from_source"] = submap_transform.tolist()
    return record


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    registry = load_and_validate_registry(project_root, args.registry)
    capture_metadata, frames = load_keyframes(
        args.capture_manifest,
        verify_hashes=not args.skip_keyframe_file_hashes,
    )
    graphs = load_submap_graphs(args.submap_diagnostics)
    events = build_query_events(
        frames,
        graphs,
        query_window_size=QUERY_WINDOW_SIZE,
        min_query_frames=MIN_QUERY_FRAMES,
        min_step_separation=MIN_STEP_SEPARATION,
    )
    global_model = GlobalRetriever(
        name=args.retriever,
        project_root=project_root,
        registry=registry,
        device=args.device,
    )
    descriptors = global_model.encode(frames, batch_size=args.batch_size)
    descriptor_ids = sorted(descriptors)
    np.savez_compressed(
        output_dir / "descriptors.npz",
        frame_ids=np.asarray(descriptor_ids),
        descriptors=np.stack([descriptors[value] for value in descriptor_ids]),
    )
    matcher = LocalGeometryMatcher(
        project_root=project_root,
        registry=registry,
        device=args.device,
    )
    raw_path = output_dir / "raw_candidates.jsonl"
    candidate_count = 0
    geometry_pass_count = 0
    with raw_path.open("x", encoding="utf-8", buffering=1) as stream:
        _write_jsonl(
            stream,
            {
                "record_type": "vpr_shadow_raw_metadata",
                "schema": RAW_SCHEMA,
                "retriever": args.retriever,
                "model_registry_sha256": sha256(args.registry),
                "capture_manifest_sha256": sha256(args.capture_manifest),
                "submap_diagnostics_sha256": sha256(args.submap_diagnostics),
                "capture_contract": capture_metadata["contract"],
                "policy_gt_isolation": True,
                "evaluation_gt_read": False,
                "planner_write_access": False,
                "calibration_applied": False,
                "top_candidate_submaps": TOP_CANDIDATE_SUBMAPS,
                "query_window_size": QUERY_WINDOW_SIZE,
                "min_query_frames": MIN_QUERY_FRAMES,
                "min_step_separation": MIN_STEP_SEPARATION,
                "vo_consistent_distance_m": VO_CONSISTENT_DISTANCE_M,
                "geometry_contract": geometry_contract(),
            },
        )
        for event_index, event in enumerate(events):
            for rank, (submap_id, score, pairs) in enumerate(
                _candidate_scores(event, descriptors), start=1
            ):
                pair_records = [
                    _verify_pair(
                        query=query,
                        historical=historical,
                        retrieval_similarity=similarity,
                        matcher=matcher,
                    )
                    for query, historical, similarity in pairs
                ]
                aggregate = aggregate_multiframe_support(pair_records)
                historical_frames = event.candidate_frames[submap_id]
                vo_distance = _vo_distance(
                    event.query_window, historical_frames
                )
                geometry_pass_count += int(aggregate["geometry_pass"])
                candidate_count += 1
                _write_jsonl(
                    stream,
                    {
                        "record_type": "vpr_shadow_candidate",
                        "event_id": event.event_id,
                        "event_index": event_index,
                        "episode_sequence": (
                            event.query_frame.episode_sequence
                        ),
                        "query_action_step": event.query_frame.action_step,
                        "query_submap_id": event.query_frame.submap_id,
                        "query_floor_id": event.query_frame.floor_id,
                        "query_frame_ids": [
                            frame.frame_id for frame in event.query_window
                        ],
                        "query_frame_steps": [
                            frame.action_step for frame in event.query_window
                        ],
                        "candidate_submap_id": submap_id,
                        "candidate_floor_id": historical_frames[0].floor_id,
                        "candidate_frame_ids": [
                            frame.frame_id for frame in historical_frames
                        ],
                        "candidate_frame_steps": [
                            frame.action_step for frame in historical_frames
                        ],
                        "retrieval_rank": rank,
                        "retrieval_score": score,
                        "vo_min_distance_m": vo_distance,
                        "vo_stratum": (
                            "vo_consistent_fragmentation"
                            if vo_distance < VO_CONSISTENT_DISTANCE_M
                            else "vo_inconsistent_revisit"
                        ),
                        "pair_records": pair_records,
                        **aggregate,
                    },
                )
            if (event_index + 1) % 10 == 0:
                print(
                    f"processed_query_events={event_index + 1}/{len(events)}",
                    flush=True,
                )
    summary = {
        "schema": "ascent_v1_4_vpr_shadow_raw_summary_v1",
        "technical_status": "PASS",
        "retriever": args.retriever,
        "keyframe_count": len(frames),
        "query_event_count": len(events),
        "candidate_count": candidate_count,
        "geometry_pass_candidate_count": geometry_pass_count,
        "raw_candidates_path": str(raw_path),
        "raw_candidates_sha256": sha256(raw_path),
        "descriptors_sha256": sha256(output_dir / "descriptors.npz"),
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
