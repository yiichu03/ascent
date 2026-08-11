#!/usr/bin/env python3
"""Strict paired evidence gate for ASCENT-VO submap B1/B2 screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


REQUIRED_METRICS = {
    "success",
    "spl",
    "soft_spl",
    "distance_to_goal",
}
SUBMAP_RECORD_TYPES = {
    "submap_run_metadata",
    "submap_episode_reset",
    "submap_action_endpoint",
    "submap_event",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def read_jsonl(path: Path) -> tuple[list[Dict[str, Any]], list[str]]:
    output = []
    errors = []
    if not path.is_file():
        return output, [f"missing:{path}"]
    with path.open(errors="replace", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"parse:{path}:{line_number}")
                continue
            if not isinstance(value, dict):
                errors.append(f"non_object:{path}:{line_number}")
                continue
            output.append(value)
    return output, errors


def normalized_scene(value: Any) -> str:
    text = str(value or "").replace("\\", "/")
    for dataset in ("hm3d", "mp3d"):
        for marker in (f"/{dataset}/", f"{dataset}/"):
            if marker in text:
                suffix = text.split(marker, 1)[1]
                return f"{dataset}/" + suffix
    return text.lstrip("/")


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: Iterable[float]) -> Optional[float]:
    clean = [float(value) for value in values]
    return statistics.fmean(clean) if clean else None


def load_manifest(
    path: Path, expected_episodes: int, expected_dataset: str
) -> tuple[Dict[str, Dict[str, Any]], list[str], list[str]]:
    manifest = read_json(path)
    errors = []
    if manifest.get("schema") != "ascent_vo_submap_screen_materialized_v1":
        errors.append("manifest_schema")
    if (
        manifest.get("dataset") != expected_dataset
        or manifest.get("split") != "train"
    ):
        errors.append("manifest_dataset_split")
    if int(manifest.get("episode_count", -1)) != expected_episodes:
        errors.append("manifest_episode_count")
    chunks: Dict[str, Dict[str, Any]] = {}
    logical_ids = []
    for chunk in manifest.get("chunks", []):
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in chunks:
            errors.append(f"duplicate_chunk:{chunk_id}")
            continue
        identity_path = Path(chunk["identity_path"]).resolve()
        if (
            not identity_path.is_file()
            or sha256(identity_path) != chunk["identity_sha256"]
        ):
            errors.append(f"identity_hash:{chunk_id}")
            continue
        identities = read_csv(identity_path)
        if len(identities) != int(chunk["episode_count"]):
            errors.append(f"identity_count:{chunk_id}")
        runtime_ids = [row["runtime_episode_id"] for row in identities]
        if runtime_ids != [str(index) for index in range(len(identities))]:
            errors.append(f"runtime_ids:{chunk_id}")
        for row in identities:
            row["chunk_id"] = chunk_id
            logical_ids.append(row["logical_case_id"])
        chunks[chunk_id] = {
            **chunk,
            "identities": identities,
            "identity_by_runtime": {
                row["runtime_episode_id"]: row for row in identities
            },
        }
    if (
        len(logical_ids) != expected_episodes
        or len(set(logical_ids)) != expected_episodes
    ):
        errors.append("logical_identity_coverage")
    if logical_ids != list(manifest.get("logical_case_ids", [])):
        errors.append("manifest_logical_order")
    return chunks, logical_ids, errors


def validate_vo_metadata(
    metadata: Mapping[str, Any],
    *,
    source_commit: str,
    forward_sha256: str,
    turn_sha256: str,
) -> list[str]:
    expected = {
        "provider": "zhao_rgbd_2021",
        "ascent_source_commit": source_commit,
        "forward_checkpoint_sha256": forward_sha256,
        "turn_checkpoint_sha256": turn_sha256,
        "action_contract": "native_0.25m_or_30deg_single_pair",
        "pose_initialization": "episode_local_zero_se2",
        "gt_policy_isolation": True,
    }
    return [
        f"vo_metadata:{key}:{metadata.get(key)!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]


def validate_submap_metadata(
    metadata: Mapping[str, Any],
    *,
    mode: str,
    source_commit: str,
    calibration: Optional[Mapping[str, Any]],
) -> list[str]:
    errors = []
    expected = {
        "ascent_source_commit": source_commit,
        "pose_source": "zhao_rgbd_2021",
        "policy_gt_isolation": True,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"submap_metadata:{key}:{metadata.get(key)!r}")
    config = metadata.get("config")
    if not isinstance(config, Mapping) or config.get("enabled") is not True:
        errors.append("submap_metadata:enabled")
        return errors
    method_version = metadata.get("method_version")
    if method_version in {
        "submap_v1.1",
        "submap_v1.2",
        "submap_v1.4_oracle_task_memory",
    }:
        v1_1_expected = {
            "split_contract": "vo_anchor_and_rgbd_overlap_joint",
            "fallback_contract": "ascent_local_first_persistent_route",
        }
        for key, value in v1_1_expected.items():
            if metadata.get(key) != value:
                errors.append(
                    f"submap_metadata:{key}:{metadata.get(key)!r}"
                )
        expected_config = {
            "min_action_endpoints": 20,
            "min_anchor_displacement_m": 1.5,
            "overlap_threshold": 0.35,
            "low_overlap_consecutive": 3,
            "gateway_frontier_resolution_radius_m": 1.0,
            "gateway_reached_radius_m": 0.9,
            "route_min_progress_m": 0.30,
            "route_max_stagnation_actions": 30,
            "route_max_waypoint_actions": 60,
            "provisional_thresholds": False,
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                errors.append(
                    f"submap_metadata:fixed_config:{key}:"
                    f"{config.get(key)!r}:{value!r}"
                )
        if method_version in {
            "submap_v1.2",
            "submap_v1.4_oracle_task_memory",
        }:
            v1_2_expected = {
                "handoff_enabled": True,
                "exhaustion_recovery_enabled": True,
                "continuity_contract": (
                    "single_connected_handoff_and_one_shot_360_recovery"
                ),
            }
            for key, value in v1_2_expected.items():
                if metadata.get(key) != value:
                    errors.append(
                        f"submap_metadata:{key}:"
                        f"{metadata.get(key)!r}:{value!r}"
                    )
        if method_version == "submap_v1.4_oracle_task_memory":
            if metadata.get("place_memory_enabled") is not True:
                errors.append("submap_metadata:place_memory_enabled")
            if metadata.get("place_memory_contract") != (
                "opaque_same_place_identity_then_independent_excursion_query"
            ):
                errors.append("submap_metadata:place_memory_contract")
            place_config = metadata.get("place_memory_config")
            expected_place_config = {
                "enabled": True,
                "low_gain_area_m2": 0.5,
                "shadow_low_gain_area_m2": 1.0,
                "branch_association_radius_m": 0.5,
                "branch_match_radius_m": 1.0,
                "branch_match_margin_m": 0.25,
                "persistent_frontier_observations": 2,
                "minimum_arrival_observations": 2,
                "minimum_repeat_arrival_observations": 3,
                "minimum_excursion_start_distance_m": 1.4,
            }
            if not isinstance(place_config, Mapping):
                errors.append("submap_metadata:place_memory_config")
            else:
                for key, value in expected_place_config.items():
                    if place_config.get(key) != value:
                        errors.append(
                            f"submap_metadata:place_memory:{key}:"
                            f"{place_config.get(key)!r}:{value!r}"
                        )
        return errors
    if calibration is None and mode == "smoke":
        if config.get("provisional_thresholds") is not True:
            errors.append("submap_metadata:smoke_not_provisional")
        expected_smoke = {
            "min_action_endpoints": 3,
            "min_path_length_m": 0.25,
            "overlap_threshold": 0.30,
            "low_overlap_consecutive": 2,
            "max_motion_budget_m": 0.75,
        }
        for key, value in expected_smoke.items():
            if config.get(key) != value:
                errors.append(
                    f"submap_metadata:smoke_config:{key}:"
                    f"{config.get(key)!r}:{value!r}"
                )
    else:
        if config.get("provisional_thresholds") is not False:
            errors.append("submap_metadata:calibrated_still_provisional")
        if calibration is None:
            errors.append("submap_metadata:missing_calibration")
        else:
            for key, value in calibration["config"].items():
                if config.get(key) != value:
                    errors.append(
                        f"submap_metadata:calibration:{key}:"
                        f"{config.get(key)!r}:{value!r}"
                    )
    return errors


def parse_attempt(
    row: Mapping[str, str],
    *,
    chunk: Mapping[str, Any],
    mode: str,
    source_commit: str,
    forward_sha256: str,
    turn_sha256: str,
    calibration: Optional[Mapping[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    condition = row["condition"]
    errors = []
    vo_path = Path(row["vo_diagnostics"]).resolve()
    vo_records, vo_errors = read_jsonl(vo_path)
    errors.extend(vo_errors)
    counts = Counter(record.get("record_type") for record in vo_records)
    metadata = [
        record
        for record in vo_records
        if record.get("record_type") == "run_metadata"
    ]
    if len(metadata) != 1:
        errors.append(
            f"{condition}:{row['chunk_id']}:vo_metadata_count:{len(metadata)}"
        )
    else:
        errors.extend(
            validate_vo_metadata(
                metadata[0],
                source_commit=source_commit,
                forward_sha256=forward_sha256,
                turn_sha256=turn_sha256,
            )
        )
    if counts["vo_technical_error"]:
        errors.append(
            f"{condition}:{row['chunk_id']}:vo_technical_errors:"
            f"{counts['vo_technical_error']}"
        )
    unknown_vo = set(counts) - {
        "run_metadata",
        "vo_step",
        "episode_end",
        "vo_technical_error",
    }
    if unknown_vo:
        errors.append(
            f"{condition}:{row['chunk_id']}:unknown_vo:{sorted(unknown_vo)}"
        )
    steps: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    ends: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    episode_sequences: Dict[str, list[int]] = defaultdict(list)
    for record in vo_records:
        kind = record.get("record_type")
        if kind == "vo_step":
            steps[str(record.get("episode_id"))].append(record)
        elif kind == "episode_end":
            runtime_id = str(record.get("episode_id"))
            episode_sequences[runtime_id].append(
                sum(len(values) for values in ends.values())
            )
            ends[runtime_id].append(record)

    submap_by_sequence: Dict[int, Dict[str, Any]] = {}
    submap_method_version = None
    submap_path_text = row.get("submap_diagnostics", "")
    if condition == "B1":
        if submap_path_text and Path(submap_path_text).exists():
            errors.append(
                f"B1:{row['chunk_id']}:unexpected_submap_diagnostics"
            )
    elif condition == "B2":
        submap_path = Path(submap_path_text).resolve()
        submap_records, submap_errors = read_jsonl(submap_path)
        errors.extend(submap_errors)
        unknown = {
            record.get("record_type") for record in submap_records
        } - SUBMAP_RECORD_TYPES
        if unknown:
            errors.append(
                f"B2:{row['chunk_id']}:unknown_submap:{sorted(unknown)}"
            )
        submap_metadata = [
            record
            for record in submap_records
            if record.get("record_type") == "submap_run_metadata"
        ]
        if len(submap_metadata) != 1:
            errors.append(
                f"B2:{row['chunk_id']}:submap_metadata_count:"
                f"{len(submap_metadata)}"
            )
        else:
            submap_method_version = submap_metadata[0].get(
                "method_version"
            )
            errors.extend(
                validate_submap_metadata(
                    submap_metadata[0],
                    mode=mode,
                    source_commit=source_commit,
                    calibration=calibration,
                )
            )
        grouped: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"resets": 0, "endpoints": [], "events": []}
        )
        for record in submap_records:
            if "episode_sequence" not in record:
                continue
            sequence = int(record["episode_sequence"])
            kind = record.get("record_type")
            if kind == "submap_episode_reset":
                grouped[sequence]["resets"] += 1
            elif kind == "submap_action_endpoint":
                grouped[sequence]["endpoints"].append(record)
            elif kind == "submap_event":
                grouped[sequence]["events"].append(record)
        submap_by_sequence = dict(grouped)

    accepted: Dict[str, Dict[str, Any]] = {}
    for runtime_id, identity in chunk["identity_by_runtime"].items():
        local_errors = []
        episode_submap_events: list[Dict[str, Any]] = []
        episode_ends = ends.get(runtime_id, [])
        if len(episode_ends) != 1:
            continue
        end = episode_ends[0]
        if normalized_scene(end.get("scene_id")) != normalized_scene(
            identity["scene_id"]
        ):
            local_errors.append("scene")
        if (
            end.get("evaluation_pose_source") != "habitat_ground_truth"
            or end.get("policy_pose_source") != "zhao_rgbd_2021"
            or end.get("gt_policy_isolation") is not True
        ):
            local_errors.append("pose_contract")
        metrics = end.get("native_metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
            local_errors.append("metrics")
        if not REQUIRED_METRICS.issubset(metrics):
            local_errors.append("metric_keys")
        elif any(finite_float(metrics[key]) is None for key in REQUIRED_METRICS):
            local_errors.append("metric_finite")
        action_steps = int(end.get("action_steps", -1))
        episode_steps = sorted(
            steps.get(runtime_id, []),
            key=lambda item: int(item.get("action_step", -1)),
        )
        if action_steps < 1 or len(episode_steps) != action_steps:
            local_errors.append("action_step_count")
        if [
            int(item.get("action_step", -1)) for item in episode_steps
        ] != list(range(1, len(episode_steps) + 1)):
            local_errors.append("action_step_sequence")
        if any(item.get("finite") is not True for item in episode_steps):
            local_errors.append("vo_finite")
        if condition == "B2":
            sequence_values = episode_sequences.get(runtime_id, [])
            if len(sequence_values) != 1:
                local_errors.append("submap_episode_sequence")
                submap = None
            else:
                submap = submap_by_sequence.get(sequence_values[0])
            if submap is None or submap["resets"] != 1:
                local_errors.append("submap_reset")
                submap = {"endpoints": [], "events": []}
            episode_submap_events = list(submap["events"])
            endpoint_steps = sorted(
                int(item.get("action_step", -1))
                for item in submap["endpoints"]
            )
            if endpoint_steps != list(range(action_steps)):
                local_errors.append("submap_endpoint_sequence")
            events = Counter(
                str(item.get("event")) for item in submap["events"]
            )
            if submap_method_version in {
                "submap_v1.1",
                "submap_v1.2",
                "submap_v1.4_oracle_task_memory",
            }:
                selected_candidates = Counter(
                    str(item.get("candidate_key"))
                    for item in submap["events"]
                    if item.get("event") == "remote_route_selected"
                )
                if any(
                    count > 1
                    for candidate, count in selected_candidates.items()
                    if candidate not in {"", "None"}
                ):
                    local_errors.append("remote_candidate_reselected")
                if any(
                    events[name] > 0
                    for name in (
                        "submap_revisit",
                        "gateway_revisit_requested",
                        "gateway_route_action",
                    )
                ):
                    local_errors.append("legacy_revisit_event")
                for endpoint in submap["endpoints"]:
                    decision = endpoint.get("decision", {})
                    if not isinstance(decision, Mapping):
                        local_errors.append("split_decision_schema")
                        break
                    if decision.get("reason") == "motion_budget":
                        local_errors.append("motion_budget_split")
                        break
                    if (
                        decision.get("should_split") is True
                        and decision.get("reason") == "low_overlap"
                        and (
                            decision.get("mature") is not True
                            or finite_float(
                                decision.get("anchor_displacement_m")
                            )
                            is None
                            or float(
                                decision.get("anchor_displacement_m")
                            )
                            < 1.5
                            or int(
                                decision.get("low_overlap_streak", -1)
                            )
                            < 3
                            or decision.get("route_action") is True
                        )
                    ):
                        local_errors.append("joint_split_contract")
                        break
            if submap_method_version in {
                "submap_v1.2",
                "submap_v1.4_oracle_task_memory",
            }:
                if events["submap_exhaustion_recovery"] > 1:
                    local_errors.append("recovery_repeated")
                if events["exhaustion_recovery_started"] > 1:
                    local_errors.append("recovery_start_repeated")
                if (
                    events["submap_exhaustion_recovery"]
                    != events["exhaustion_recovery_started"]
                ):
                    local_errors.append("recovery_boundary_contract")
                floor_priority_skips = sum(
                    item.get("event") == "exhaustion_recovery_skipped"
                    and item.get("reason") == "floor_transition_priority"
                    for item in episode_submap_events
                )
                if events["exhaustion_recovery_requested"] != (
                    events["exhaustion_recovery_started"]
                    + floor_priority_skips
                ):
                    local_errors.append("recovery_request_contract")
                if events["exhaustion_recovery_scan_turn"] > 11:
                    local_errors.append("recovery_turn_budget")
                if events["exhaustion_recovery_scan_completed"] > 0 and (
                    events["exhaustion_recovery_scan_turn"] != 11
                ):
                    local_errors.append("recovery_full_scan_contract")
                for item in episode_submap_events:
                    if item.get("event") != "handoff_created":
                        continue
                    waypoint = item.get("waypoint_local")
                    replayed = item.get("replayed_depth_frames")
                    if (
                        not isinstance(waypoint, list)
                        or len(waypoint) != 2
                        or any(finite_float(value) is None for value in waypoint)
                        or not isinstance(replayed, int)
                        or not 1 <= replayed <= 4
                    ):
                        local_errors.append("handoff_creation_schema")
                        break
                for item in episode_submap_events:
                    if item.get("event") == "search_attempt_finished":
                        if item.get("status") == "consumed":
                            local_errors.append(
                                "place_memory_direct_consumed_without_revisit"
                            )
                            break
                        if item.get("provisional_low_gain") is True and (
                            item.get("status") != "unresolved"
                            or item.get("excursion_qualified") is not True
                        ):
                            local_errors.append(
                                "place_memory_provisional_excursion_contract"
                            )
                            break
                    elif item.get("event") == "search_branch_consumed":
                        if (
                            item.get("reason")
                            != "matched_repeat_low_gain_excursion"
                            or not isinstance(
                                item.get("historical_branch_ids"), list
                            )
                            or not item.get("historical_branch_ids")
                            or not str(
                                item.get("confirmation_source_submap_id")
                                or ""
                            )
                            or int(item.get("last_arrival_observations", 0)) < 3
                            or float(
                                item.get("last_start_robot_distance_m", 0.0)
                            )
                            < 1.4
                        ):
                            local_errors.append(
                                "place_memory_consumed_promotion_contract"
                            )
                            break
                    elif item.get("event") == "search_branch_repeat_cutoff":
                        if (
                            item.get("reason") != "qualified_repeat_low_gain"
                            or int(item.get("arrival_observations", 0)) < 3
                            or int(
                                item.get(
                                    "minimum_repeat_arrival_observations", 0
                                )
                            )
                            != 3
                            or float(
                                item.get("start_robot_distance_m", 0.0)
                            )
                            < 1.4
                            or float(item.get("coverage_delta_m2", 1.0))
                            >= 0.5
                            or int(
                                item.get("residual_alternative_count", 0)
                            )
                            < 1
                        ):
                            local_errors.append(
                                "place_memory_repeat_cutoff_contract"
                            )
                            break
                        if not any(
                            event.get("event") == "place_rerank_evaluated"
                            and event.get("step") == item.get("step")
                            and event.get("decision_changed") is True
                            for event in episode_submap_events
                        ):
                            local_errors.append(
                                "place_memory_repeat_cutoff_without_rerank"
                            )
                            break
        else:
            events = Counter()
        if local_errors:
            errors.append(
                f"{condition}:{identity['logical_case_id']}:"
                + "|".join(local_errors)
            )
            continue
        association_events = [
            item
            for item in episode_submap_events
            if item.get("event") == "place_association_accepted"
        ]
        finished_search_events = [
            item
            for item in episode_submap_events
            if item.get("event") == "search_attempt_finished"
        ]
        consumed_branch_events = [
            item
            for item in episode_submap_events
            if item.get("event") == "search_branch_consumed"
        ]
        association_to_productive_costs = []
        first_association_by_place: Dict[str, int] = {}
        for item in association_events:
            place_id = str(item.get("place_id") or "")
            step = int(item.get("step", -1))
            if place_id and step >= 0:
                first_association_by_place.setdefault(place_id, step)
        for place_id, association_step in first_association_by_place.items():
            productive_steps = [
                int(item.get("step", -1))
                for item in finished_search_events
                if str(item.get("place_id") or "") == place_id
                and item.get("status") == "productive"
                and int(item.get("step", -1)) >= association_step
            ]
            if productive_steps:
                association_to_productive_costs.append(
                    min(productive_steps) - association_step
                )
        search_statuses = Counter(
            str(item.get("status")) for item in finished_search_events
        )
        rerank_events = [
            item
            for item in episode_submap_events
            if item.get("event") == "place_rerank_evaluated"
        ]
        accepted[identity["logical_case_id"]] = {
            "logical_case_id": identity["logical_case_id"],
            "chunk_id": row["chunk_id"],
            "runtime_episode_id": runtime_id,
            "scene_id": normalized_scene(identity["scene_id"]),
            "target_category": identity["target_category"],
            "condition": condition,
            "priority": int(row["priority"]),
            "action_steps": action_steps,
            "success": float(metrics["success"]),
            "spl": float(metrics["spl"]),
            "soft_spl": float(metrics["soft_spl"]),
            "distance_to_goal": float(metrics["distance_to_goal"]),
            "mean_translation_error": mean(
                float(item["translation_error"])
                for item in episode_steps
                if item.get("localization_error_available") is True
                and finite_float(item.get("translation_error")) is not None
            ),
            "final_translation_error": next(
                (
                    float(item["translation_error"])
                    for item in reversed(episode_steps)
                    if item.get("localization_error_available") is True
                    and finite_float(item.get("translation_error"))
                    is not None
                ),
                math.nan,
            ),
            "submap_split_count": events["submap_split"],
            "handoff_created_count": events["handoff_created"],
            "handoff_action_count": events["handoff_action"],
            "handoff_completed_count": events["handoff_completed"],
            "handoff_cancelled_count": events["handoff_cancelled"],
            "handoff_skipped_count": events["handoff_skipped"],
            "exhaustion_recovery_count": events[
                "submap_exhaustion_recovery"
            ],
            "exhaustion_recovery_scan_turn_count": events[
                "exhaustion_recovery_scan_turn"
            ],
            "exhaustion_recovery_scan_completed_count": events[
                "exhaustion_recovery_scan_completed"
            ],
            "exhaustion_recovery_target_reacquired_count": events[
                "exhaustion_recovery_target_reacquired"
            ],
            "submap_revisit_count": events["submap_revisit"],
            "gateway_route_action_count": events["gateway_route_action"],
            "gateway_revisit_requested_count": events[
                "gateway_revisit_requested"
            ],
            "remote_direct_action_count": events[
                "remote_destination_direct_action"
            ],
            "remote_boundary_crossing_count": events[
                "remote_frontier_boundary_crossing"
            ],
            "remote_route_selected_count": events[
                "remote_route_selected"
            ],
            "remote_route_waypoint_action_count": events[
                "remote_route_waypoint_action"
            ],
            "remote_route_waypoint_reached_count": events[
                "remote_route_waypoint_reached"
            ],
            "remote_route_finished_count": events[
                "remote_route_finished"
            ],
            "remote_route_target_kinds": dict(
                Counter(
                    str(item.get("target_kind"))
                    for item in episode_submap_events
                    if item.get("event") == "remote_route_selected"
                )
            ),
            "remote_route_outcomes": dict(
                Counter(
                    str(item.get("outcome"))
                    for item in episode_submap_events
                    if item.get("event") == "remote_route_finished"
                )
            ),
            "submap_split_reasons": dict(
                Counter(
                    str(item.get("reason"))
                    for item in episode_submap_events
                    if item.get("event") == "submap_split"
                )
            ),
            "handoff_outcomes": dict(
                Counter(
                    str(item.get("reason"))
                    for item in episode_submap_events
                    if item.get("event")
                    in {"handoff_completed", "handoff_cancelled"}
                )
            ),
            "handoff_skip_reasons": dict(
                Counter(
                    str(item.get("reason"))
                    for item in episode_submap_events
                    if item.get("event") == "handoff_skipped"
                )
            ),
            "exhaustion_recovery_skip_reasons": dict(
                Counter(
                    str(item.get("reason"))
                    for item in episode_submap_events
                    if item.get("event")
                    == "exhaustion_recovery_skipped"
                )
            ),
            "place_association_accepted_count": events[
                "place_association_accepted"
            ],
            "place_association_rejected_count": events[
                "place_association_rejected"
            ],
            "place_association_rejection_reasons": dict(
                Counter(
                    str(item.get("reason"))
                    for item in episode_submap_events
                    if item.get("event") == "place_association_rejected"
                )
            ),
            "search_attempt_started_count": events[
                "search_attempt_started"
            ],
            "search_attempt_finished_count": events[
                "search_attempt_finished"
            ],
            "search_status_counts": dict(search_statuses),
            "search_branch_consumed_count": len(consumed_branch_events),
            "search_branch_revisit_available_count": events[
                "search_branch_revisit_available"
            ],
            "search_branch_revisit_productive_count": events[
                "search_branch_revisit_productive"
            ],
            "search_branch_revisit_inconclusive_count": events[
                "search_branch_revisit_inconclusive"
            ],
            "search_branch_repeat_cutoff_count": events[
                "search_branch_repeat_cutoff"
            ],
            "search_provisional_low_gain_count": sum(
                item.get("provisional_low_gain") is True
                for item in finished_search_events
            ),
            "search_excursion_qualified_count": sum(
                item.get("excursion_qualified") is True
                for item in finished_search_events
            ),
            "search_rejected_short_arrival_count": sum(
                item.get("reached") is True
                and item.get("excursion_qualified") is not True
                and item.get("status") == "unresolved"
                for item in finished_search_events
            ),
            "search_coverage_delta_m2": sum(
                float(item.get("coverage_delta_m2", 0.0))
                for item in finished_search_events
            ),
            "search_action_cost": sum(
                int(item.get("action_cost", 0))
                for item in finished_search_events
            ),
            "search_target_gain_count": sum(
                item.get("target_gain") is True
                for item in finished_search_events
            ),
            "search_exit_gain_count": sum(
                item.get("exit_gain") is True
                for item in finished_search_events
            ),
            "place_rerank_evaluated_count": len(rerank_events),
            "place_rerank_changed_count": sum(
                item.get("decision_changed") is True
                for item in rerank_events
            ),
            "place_intervention_types": dict(
                Counter(
                    str(item.get("intervention"))
                    for item in rerank_events
                    if item.get("decision_changed") is True
                )
            ),
            "association_to_productive_count": len(
                association_to_productive_costs
            ),
            "association_to_productive_action_cost": sum(
                association_to_productive_costs
            ),
            "evidence_vo_diagnostics": str(vo_path),
            "evidence_submap_diagnostics": submap_path_text,
        }
    return accepted, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--expected-dataset",
        choices=("hm3d", "mp3d"),
        default="hm3d",
    )
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--forward-checkpoint-sha256", required=True)
    parser.add_argument("--turn-checkpoint-sha256", required=True)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    chunks, logical_order, strict_errors = load_manifest(
        args.manifest, args.expected_episodes, args.expected_dataset
    )
    calibration = None
    if args.mode == "full" or args.calibration_json is not None:
        if args.calibration_json is None:
            strict_errors.append("missing_calibration_json")
        else:
            calibration = read_json(args.calibration_json)
            if calibration.get("status") != "PASS":
                strict_errors.append("calibration_not_pass")
    registry = read_csv(args.registry)
    attempts: Dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(
        list
    )
    for registry_order, entry in enumerate(registry):
        inventory_path = Path(entry["inventory"]).resolve()
        if not inventory_path.is_file():
            strict_errors.append(f"missing_inventory:{inventory_path}")
            continue
        for row in read_csv(inventory_path):
            if row.get("stage") != "screen":
                continue
            condition = row.get("condition", "")
            chunk_id = row.get("chunk_id", "")
            if condition not in {"B1", "B2"} or chunk_id not in chunks:
                strict_errors.append(
                    f"bad_inventory_row:{condition}:{chunk_id}"
                )
                continue
            accepted, errors = parse_attempt(
                row,
                chunk=chunks[chunk_id],
                mode=args.mode,
                source_commit=args.source_commit,
                forward_sha256=args.forward_checkpoint_sha256,
                turn_sha256=args.turn_checkpoint_sha256,
                calibration=calibration,
            )
            strict_errors.extend(errors)
            for logical_id, episode in accepted.items():
                episode["registry_order"] = registry_order
                attempts[(condition, logical_id)].append(episode)

    selected: Dict[tuple[str, str], Dict[str, Any]] = {}
    duplicate_priority = []
    for key, values in attempts.items():
        values.sort(
            key=lambda item: (
                int(item["priority"]),
                int(item["registry_order"]),
            )
        )
        priorities = Counter(int(item["priority"]) for item in values)
        if any(count > 1 for count in priorities.values()):
            duplicate_priority.append(f"{key}:{dict(priorities)}")
        selected[key] = values[0]
    strict_errors.extend(
        f"duplicate_episode_priority:{value}" for value in duplicate_priority
    )

    episode_rows = []
    missing = []
    flips = Counter()
    for logical_id in logical_order:
        b1 = selected.get(("B1", logical_id))
        b2 = selected.get(("B2", logical_id))
        if b1 is None or b2 is None:
            missing.append(
                {
                    "logical_case_id": logical_id,
                    "b1": b1 is not None,
                    "b2": b2 is not None,
                }
            )
            continue
        if b1["scene_id"] != b2["scene_id"]:
            strict_errors.append(f"paired_scene:{logical_id}")
            continue
        b1_success = int(b1["success"] >= 0.5)
        b2_success = int(b2["success"] >= 0.5)
        flips[f"{b1_success}->{b2_success}"] += 1
        episode_rows.append(
            {
                "logical_case_id": logical_id,
                "chunk_id": b1["chunk_id"],
                "scene_id": b1["scene_id"],
                "target_category": b1["target_category"],
                "b1_success": b1["success"],
                "b2_success": b2["success"],
                "success_delta": b2["success"] - b1["success"],
                "b1_spl": b1["spl"],
                "b2_spl": b2["spl"],
                "spl_delta": b2["spl"] - b1["spl"],
                "b1_action_steps": b1["action_steps"],
                "b2_action_steps": b2["action_steps"],
                "action_steps_delta": (
                    b2["action_steps"] - b1["action_steps"]
                ),
                "b1_final_translation_error": b1[
                    "final_translation_error"
                ],
                "b2_final_translation_error": b2[
                    "final_translation_error"
                ],
                "submap_split_count": b2["submap_split_count"],
                "submap_revisit_count": b2["submap_revisit_count"],
                "gateway_route_action_count": b2[
                    "gateway_route_action_count"
                ],
                "gateway_revisit_requested_count": b2[
                    "gateway_revisit_requested_count"
                ],
                "remote_direct_action_count": b2[
                    "remote_direct_action_count"
                ],
                "remote_boundary_crossing_count": b2[
                    "remote_boundary_crossing_count"
                ],
                "b1_priority": b1["priority"],
                "b2_priority": b2["priority"],
            }
        )
    paired_count = len(episode_rows)
    split_episodes = sum(
        row["submap_split_count"] > 0 for row in episode_rows
    )
    split_count = sum(
        row["submap_split_count"] for row in episode_rows
    )
    gateway_route_episodes = sum(
        row["gateway_route_action_count"] > 0 for row in episode_rows
    )
    if args.mode == "smoke":
        exposure_status = (
            "PASS" if paired_count == 5 and split_count >= 1 else "FAIL"
        )
    else:
        exposure_status = (
            "SUFFICIENT"
            if split_episodes >= 15 and split_count >= 15
            else "INSUFFICIENT"
        )
    technical_status = (
        "PASS"
        if (
            not strict_errors
            and not missing
            and paired_count == args.expected_episodes
        )
        else "FAIL"
    )
    summary = {
        "schema": "ascent_vo_submap_paired_screen_v1",
        "dataset": args.expected_dataset,
        "mode": args.mode,
        "technical_status": technical_status,
        "exposure_status": exposure_status,
        "expected_episodes": args.expected_episodes,
        "paired_valid_episodes": paired_count,
        "b1_valid_episodes": sum(
            ("B1", logical_id) in selected for logical_id in logical_order
        ),
        "b2_valid_episodes": sum(
            ("B2", logical_id) in selected for logical_id in logical_order
        ),
        "missing": missing,
        "strict_error_count": len(strict_errors),
        "strict_errors": strict_errors,
        "metrics": {
            "b1_sr": mean(row["b1_success"] for row in episode_rows),
            "b2_sr": mean(row["b2_success"] for row in episode_rows),
            "sr_delta": mean(
                row["success_delta"] for row in episode_rows
            ),
            "b1_spl": mean(row["b1_spl"] for row in episode_rows),
            "b2_spl": mean(row["b2_spl"] for row in episode_rows),
            "spl_delta": mean(row["spl_delta"] for row in episode_rows),
            "b1_mean_action_steps": mean(
                row["b1_action_steps"] for row in episode_rows
            ),
            "b2_mean_action_steps": mean(
                row["b2_action_steps"] for row in episode_rows
            ),
            "success_flips": dict(sorted(flips.items())),
        },
        "mechanism": {
            "split_episodes": split_episodes,
            "split_count": split_count,
            "revisit_count": sum(
                row["submap_revisit_count"] for row in episode_rows
            ),
            "gateway_route_episodes": gateway_route_episodes,
            "gateway_route_actions": sum(
                row["gateway_route_action_count"]
                for row in episode_rows
            ),
            "gateway_revisit_requests": sum(
                row["gateway_revisit_requested_count"]
                for row in episode_rows
            ),
            "remote_direct_actions": sum(
                row["remote_direct_action_count"]
                for row in episode_rows
            ),
            "remote_boundary_crossings": sum(
                row["remote_boundary_crossing_count"]
                for row in episode_rows
            ),
        },
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "registry": str(args.registry.resolve()),
            "registry_sha256": sha256(args.registry),
            "source_commit": args.source_commit,
            "calibration_json": (
                None
                if args.calibration_json is None
                else str(args.calibration_json.resolve())
            ),
            "calibration_sha256": (
                None
                if args.calibration_json is None
                else sha256(args.calibration_json)
            ),
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(
            summary, handle, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
    fields = list(episode_rows[0]) if episode_rows else [
        "logical_case_id"
    ]
    with (output_dir / "episodes.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(episode_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if technical_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
