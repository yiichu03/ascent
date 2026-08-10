#!/usr/bin/env python3
"""Evaluation-only GT labels for completed VPR shadow candidate streams."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from vpr_shadow_data import (
        EpisodeGraph,
        load_submap_graphs,
        read_jsonl,
        sha256,
    )
except ImportError:  # imported as scripts.score_vpr_shadow_gt in tests
    from scripts.vpr_shadow_data import (
        EpisodeGraph,
        load_submap_graphs,
        read_jsonl,
        sha256,
    )


LABELED_SCHEMA = "ascent_v1_4_vpr_shadow_gt_labeled_candidates_v2"
SUMMARY_SCHEMA = "ascent_v1_4_vpr_shadow_gt_score_summary_v2"
EPISODE_JOIN_CONTRACT = "capture_sequence_to_logical_to_runtime_v1"
RETURN_RADIUS_M = 0.75
SAME_FLOOR_HEIGHT_M = 0.75
MIN_GAP_STEPS = 30
MIN_EXCURSION_M = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--vo-diagnostics", type=Path, required=True)
    parser.add_argument("--submap-diagnostics", type=Path, required=True)
    parser.add_argument("--episode-identity", type=Path, required=True)
    parser.add_argument("--capture-episodes", type=Path, required=True)
    parser.add_argument("--chunk-id", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _identity(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        runtime = int(row["runtime_episode_id"])
        if runtime in output:
            raise ValueError(f"duplicate runtime episode id {runtime}")
        output[runtime] = dict(row)
    return output


def _gt_tracks(path: Path) -> dict[int, dict[int, np.ndarray]]:
    tracks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for row in read_jsonl(path):
        if row.get("record_type") != "vo_step":
            continue
        if row.get("localization_error_available") is not True:
            continue
        gt = row.get("gt_start_aligned_pose")
        if not isinstance(gt, Mapping):
            continue
        episode = int(row["episode_id"])
        step = int(row["action_step"])
        value = np.asarray(
            [float(gt["x"]), float(gt["y"]), float(gt["height"])],
            dtype=np.float64,
        )
        if not np.isfinite(value).all():
            raise ValueError("non-finite evaluation GT trajectory")
        if step in tracks[episode]:
            raise ValueError(f"duplicate GT action step e{episode}/s{step}")
        tracks[episode][step] = value
    return {episode: dict(values) for episode, values in tracks.items()}


def _sequence_runtime_map(
    *,
    capture_rows: Sequence[Mapping[str, Any]],
    chunk_id: str,
    identities: Mapping[int, Mapping[str, str]],
    tracks: Mapping[int, Mapping[int, np.ndarray]],
    graphs: Mapping[int, EpisodeGraph],
) -> dict[int, int]:
    """Join asynchronous policy sequence IDs to Habitat runtime IDs."""

    rows = [row for row in capture_rows if str(row.get("chunk_id")) == chunk_id]
    logical_to_runtime = {
        str(identity["logical_case_id"]): int(runtime)
        for runtime, identity in identities.items()
    }
    if not rows or len(rows) != len(identities):
        raise ValueError("capture episode join does not cover the scored unit")
    output: dict[int, int] = {}
    seen_runtime: set[int] = set()
    for row in rows:
        logical_id = str(row["logical_case_id"])
        if logical_id not in logical_to_runtime:
            raise ValueError("capture episode logical identity is absent")
        sequence = int(row["episode_sequence"])
        runtime = logical_to_runtime[logical_id]
        if sequence in output or runtime in seen_runtime:
            raise ValueError("capture episode join is not one-to-one")
        if sequence not in graphs or runtime not in tracks:
            raise ValueError("capture episode join misses graph or GT track")
        action_count = int(row["action_count"])
        graph_steps = set(graphs[sequence].submap_by_step)
        track_steps = set(tracks[runtime])
        if (
            graph_steps != set(range(action_count))
            or track_steps != set(range(1, action_count + 1))
        ):
            raise ValueError("capture episode join action-count mismatch")
        output[sequence] = runtime
        seen_runtime.add(runtime)
    if set(output) != set(graphs) or seen_runtime != set(tracks):
        raise ValueError("capture episode join leaves unmatched identities")
    return output


def _candidate_label(
    *,
    query_steps: Sequence[int],
    candidate_submap_id: str,
    graph: EpisodeGraph,
    track: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    historical_steps = sorted(
        step
        for step, submap_id in graph.submap_by_step.items()
        if submap_id == candidate_submap_id and step in track
    )
    pairs: list[tuple[float, float, int, int]] = []
    cross_floor_pairs: list[tuple[float, float, int, int]] = []
    for query_step in query_steps:
        if query_step not in track:
            continue
        query = track[query_step]
        for old_step in historical_steps:
            old = track[old_step]
            height = abs(float(query[2] - old[2]))
            if query_step - old_step < MIN_GAP_STEPS:
                continue
            intermediate = [
                value
                for step, value in track.items()
                if old_step < step < query_step
            ]
            if not intermediate:
                continue
            excursion = max(
                float(np.linalg.norm(value - old)) for value in intermediate
            )
            if excursion < MIN_EXCURSION_M:
                continue
            planar = float(np.linalg.norm(query[:2] - old[:2]))
            if planar <= RETURN_RADIUS_M and height <= SAME_FLOOR_HEIGHT_M:
                pairs.append((planar, height, old_step, query_step))
            elif planar <= RETURN_RADIUS_M and height > SAME_FLOOR_HEIGHT_M:
                cross_floor_pairs.append((planar, height, old_step, query_step))
    representative = min(
        pairs, key=lambda value: (value[0], value[1], -value[2], value[3])
    ) if pairs else None
    return {
        "gt_same_place_label": bool(pairs),
        "gt_cross_floor_negative": bool(not pairs and cross_floor_pairs),
        "gt_representative_planar_distance_m": (
            None if representative is None else representative[0]
        ),
        "gt_representative_height_distance_m": (
            None if representative is None else representative[1]
        ),
        "gt_representative_historical_step": (
            None if representative is None else representative[2]
        ),
        "gt_representative_query_step": (
            None if representative is None else representative[3]
        ),
    }


def _eligible_candidate_ids(
    *,
    query_submap_id: str,
    query_action_step: int,
    query_floor_id: int,
    graph: EpisodeGraph,
) -> list[str]:
    current_start = graph.submap_first_step[query_submap_id]
    output = []
    for candidate, candidate_start in graph.submap_first_step.items():
        if candidate == query_submap_id or candidate_start >= current_start:
            continue
        if graph.floor_by_submap.get(candidate) != query_floor_id:
            continue
        if frozenset((candidate, query_submap_id)) in graph.direct_edges:
            continue
        if not any(
            submap_id == candidate and step <= query_action_step - MIN_GAP_STEPS
            for step, submap_id in graph.submap_by_step.items()
        ):
            continue
        output.append(candidate)
    return sorted(output)


def main() -> int:
    args = parse_args()
    raw_rows = read_jsonl(args.raw_candidates)
    if not raw_rows or raw_rows[0].get("record_type") != "vpr_shadow_raw_metadata":
        raise ValueError("missing raw VPR metadata")
    raw_metadata = raw_rows[0]
    if raw_metadata.get("evaluation_gt_read") is not False:
        raise ValueError("raw association stage was not GT isolated")
    if raw_metadata.get("calibration_applied") is not False:
        raise ValueError("raw association unexpectedly contains calibration")
    candidates = [
        row for row in raw_rows[1:]
        if row.get("record_type") == "vpr_shadow_candidate"
    ]
    identities = _identity(args.episode_identity)
    tracks = _gt_tracks(args.vo_diagnostics)
    graphs = load_submap_graphs(args.submap_diagnostics)
    capture_rows = read_jsonl(args.capture_episodes)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    allowed_ids = {
        str(row["logical_case_id"]) for row in split.get("episodes", [])
    }
    if not identities or set(identities) != set(tracks):
        raise ValueError("identity and GT episode sets differ")
    if len(identities) != len(graphs):
        raise ValueError("submap and runtime episode counts differ")
    for runtime, identity in identities.items():
        if identity["logical_case_id"] not in allowed_ids:
            raise ValueError("captured episode is outside the frozen split role")
    for row in capture_rows:
        if str(row.get("chunk_id")) != str(args.chunk_id):
            continue
        expected_paths = {
            "vo_diagnostics": args.vo_diagnostics,
            "submap_diagnostics": args.submap_diagnostics,
            "identity_path": args.episode_identity,
        }
        for key, expected in expected_paths.items():
            if Path(str(row.get(key, ""))).resolve() != expected.resolve():
                raise ValueError(f"capture episode {key} differs from scored unit")
    sequence_to_runtime = _sequence_runtime_map(
        capture_rows=capture_rows,
        chunk_id=str(args.chunk_id),
        identities=identities,
        tracks=tracks,
        graphs=graphs,
    )

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_event[str(row["event_id"])].append(row)
    labeled_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for event_id, rows in sorted(
        by_event.items(), key=lambda item: int(item[1][0]["event_index"])
    ):
        rows.sort(key=lambda row: int(row["retrieval_rank"]))
        exemplar = rows[0]
        sequence = int(exemplar["episode_sequence"])
        if sequence not in sequence_to_runtime:
            raise ValueError("raw event has no capture episode identity join")
        runtime = sequence_to_runtime[sequence]
        graph = graphs[sequence]
        track = tracks[runtime]
        identity = identities[runtime]
        query_steps = [int(value) for value in exemplar["query_frame_steps"]]
        eligible = _eligible_candidate_ids(
            query_submap_id=str(exemplar["query_submap_id"]),
            query_action_step=int(exemplar["query_action_step"]),
            query_floor_id=int(exemplar["query_floor_id"]),
            graph=graph,
        )
        all_labels = {
            candidate: _candidate_label(
                query_steps=query_steps,
                candidate_submap_id=candidate,
                graph=graph,
                track=track,
            )
            for candidate in eligible
        }
        true_candidates = sorted(
            candidate
            for candidate, label in all_labels.items()
            if label["gt_same_place_label"]
        )
        proposed_true_ranks = []
        for row in rows:
            candidate = str(row["candidate_submap_id"])
            if candidate not in all_labels:
                raise ValueError("raw candidate violates causal eligibility")
            label = all_labels[candidate]
            proposed_true_ranks.extend(
                [int(row["retrieval_rank"])]
                if label["gt_same_place_label"] else []
            )
            labeled_rows.append(
                {
                    **row,
                    "logical_case_id": identity["logical_case_id"],
                    "scene_id": identity["scene_id"],
                    **label,
                }
            )
        event_rows.append(
            {
                "record_type": "vpr_shadow_gt_event",
                "event_id": event_id,
                "event_index": int(exemplar["event_index"]),
                "episode_sequence": sequence,
                "logical_case_id": identity["logical_case_id"],
                "scene_id": identity["scene_id"],
                "query_action_step": int(exemplar["query_action_step"]),
                "query_submap_id": str(exemplar["query_submap_id"]),
                "eligible_candidate_count": len(eligible),
                "gt_true_candidate_submap_ids": true_candidates,
                "gt_revisit_exposure": bool(true_candidates),
                "best_proposed_true_rank": (
                    min(proposed_true_ranks) if proposed_true_ranks else None
                ),
            }
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    labeled_path = output_dir / "labeled_candidates.jsonl"
    with labeled_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "record_type": "vpr_shadow_gt_metadata",
                    "schema": LABELED_SCHEMA,
                    "evaluation_only_gt": True,
                    "runtime_policy_access": False,
                    "raw_candidates_sha256": sha256(args.raw_candidates),
                    "vo_diagnostics_sha256": sha256(args.vo_diagnostics),
                    "submap_diagnostics_sha256": sha256(
                        args.submap_diagnostics
                    ),
                    "episode_identity_sha256": sha256(args.episode_identity),
                    "capture_episodes_sha256": sha256(args.capture_episodes),
                    "episode_join_contract": EPISODE_JOIN_CONTRACT,
                    "chunk_id": str(args.chunk_id),
                    "split_manifest_sha256": sha256(args.split_manifest),
                    "thresholds": {
                        "return_radius_m": RETURN_RADIUS_M,
                        "same_floor_height_m": SAME_FLOOR_HEIGHT_M,
                        "min_gap_steps": MIN_GAP_STEPS,
                        "min_excursion_m": MIN_EXCURSION_M,
                    },
                    "retriever": raw_metadata["retriever"],
                },
                sort_keys=True,
            ) + "\n"
        )
        for row in labeled_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    events_path = output_dir / "events.jsonl"
    with events_path.open("x", encoding="utf-8") as stream:
        for row in event_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    exposure_events = [row for row in event_rows if row["gt_revisit_exposure"]]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "technical_status": "PASS",
        "retriever": raw_metadata["retriever"],
        "candidate_count": len(labeled_rows),
        "query_event_count": len(event_rows),
        "evaluated_episode_count": len(identities),
        "evaluated_logical_case_ids": [
            identities[index]["logical_case_id"] for index in sorted(identities)
        ],
        "gt_revisit_exposure_event_count": len(exposure_events),
        "proposal_recall_at_1": (
            sum(row["best_proposed_true_rank"] == 1 for row in exposure_events)
            / len(exposure_events) if exposure_events else None
        ),
        "proposal_recall_at_5": (
            sum(row["best_proposed_true_rank"] is not None for row in exposure_events)
            / len(exposure_events) if exposure_events else None
        ),
        "labeled_candidates_sha256": sha256(labeled_path),
        "events_sha256": sha256(events_path),
        "capture_episodes_sha256": sha256(args.capture_episodes),
        "episode_join_contract": EPISODE_JOIN_CONTRACT,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
