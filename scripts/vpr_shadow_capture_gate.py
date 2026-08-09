"""Shared strict checks for passive v1.4 VPR-shadow capture evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from vpr_shadow_data import (
        load_keyframes,
        read_jsonl,
        resolve_submap_identity_floors,
        sha256,
    )
except ImportError:
    from scripts.vpr_shadow_data import (
        load_keyframes,
        read_jsonl,
        resolve_submap_identity_floors,
        sha256,
    )


ACTION_HASH_CONTRACT = "sha256(json_compact_integer_action_list_plus_lf)"
CAPTURE_RECORD_TYPES = {
    "vpr_shadow_run_metadata",
    "vpr_shadow_episode_reset",
    "vpr_shadow_keyframe",
}
FORBIDDEN_RUNTIME_KEYS = {
    "gps",
    "compass",
    "ground_truth",
    "gt_pose",
    "gt_start_aligned_pose",
    "success",
    "spl",
    "soft_spl",
    "distance_to_goal",
}
EXPECTED_KEYFRAME_CONFIG = {
    "milestone_action_offsets": [0, 4, 10, 20, 40, 80, 160, 320],
    "max_keyframes_per_submap": 8,
    "min_translation_m": 0.5,
    "min_yaw_rad": math.radians(30.0),
    "min_depth_valid_fraction": 0.25,
    "jpeg_quality": 95,
    "depth_quantization_levels": 65535,
}


def canonical_action_hash(actions: Sequence[int]) -> str:
    payload = json.dumps(
        [int(action) for action in actions], separators=(",", ":")
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_action_sequences(path: Path) -> tuple[dict[str, list[int]], list[str]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    order: list[str] = []
    ends: Counter[str] = Counter()
    for row in read_jsonl(path):
        record_type = row.get("record_type")
        if record_type == "vo_step":
            runtime_id = str(row["episode_id"])
            grouped[runtime_id].append(
                (int(row["action_step"]), int(row["action"]))
            )
        elif record_type == "episode_end":
            runtime_id = str(row["episode_id"])
            order.append(runtime_id)
            ends[runtime_id] += 1
    if any(count != 1 for count in ends.values()) or set(grouped) != set(ends):
        raise ValueError("VO action stream has incomplete or duplicate episodes")
    output: dict[str, list[int]] = {}
    for runtime_id, rows in grouped.items():
        rows.sort()
        if [step for step, _ in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"non-contiguous VO actions for runtime {runtime_id}")
        output[runtime_id] = [action for _, action in rows]
    return output, order


def _walk_keys(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            output.add(str(key).lower())
            output.update(_walk_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            output.update(_walk_keys(item))
    return output


def _config_matches(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_KEYFRAME_CONFIG):
        return False
    for key, expected in EXPECTED_KEYFRAME_CONFIG.items():
        observed = value[key]
        if isinstance(expected, float):
            if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
                return False
        elif observed != expected:
            return False
    return True


def validate_capture_attempt(
    *,
    capture_manifest: Path,
    submap_diagnostics: Path,
    vo_diagnostics: Path,
    source_commit: str,
    dataset: str,
) -> tuple[dict[str, Any], list[str]]:
    """Validate one completed navigation process without emitting outcomes."""

    errors: list[str] = []
    capture_manifest = Path(capture_manifest).resolve()
    try:
        rows = read_jsonl(capture_manifest)
        metadata, frames = load_keyframes(capture_manifest, verify_hashes=True)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {}, [f"capture_read:{type(exc).__name__}:{exc}"]
    unknown = sorted(
        {str(row.get("record_type")) for row in rows} - CAPTURE_RECORD_TYPES
    )
    if unknown:
        errors.append(f"capture_unknown_records:{unknown}")
    forbidden = set().union(*(_walk_keys(row) for row in rows)) & FORBIDDEN_RUNTIME_KEYS
    if forbidden:
        errors.append(f"capture_forbidden_keys:{sorted(forbidden)}")
    expected_metadata = {
        "ascent_source_commit": source_commit,
        "dataset": dataset,
        "pose_source": "zhao_rgbd_2021",
        "navigation_behavior": "frozen_submap_v1.2",
        "capture_point": "pre_planner_policy_visible_observation",
        "policy_gt_isolation": True,
        "planner_write_access": False,
        "association_runtime_enabled": False,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            errors.append(f"capture_metadata:{key}:{metadata.get(key)!r}")
    if not _config_matches(metadata.get("keyframe_config")):
        errors.append("capture_keyframe_config")

    resets = [
        row for row in rows
        if row.get("record_type") == "vpr_shadow_episode_reset"
    ]
    reset_sequences = [int(row["episode_sequence"]) for row in resets]
    if len(reset_sequences) != len(set(reset_sequences)):
        errors.append("capture_duplicate_episode_reset")
    try:
        _, runtime_order = load_action_sequences(vo_diagnostics)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        runtime_order = []
        errors.append(f"capture_vo_actions:{type(exc).__name__}:{exc}")
    expected_sequences = list(range(len(runtime_order)))
    if reset_sequences != expected_sequences:
        errors.append(
            f"capture_episode_sequences:{reset_sequences}:{expected_sequences}"
        )

    endpoints: dict[tuple[int, int], Mapping[str, Any]] = {}
    try:
        submap_rows = read_jsonl(submap_diagnostics)
    except (OSError, ValueError) as exc:
        submap_rows = []
        errors.append(f"capture_submap_read:{type(exc).__name__}:{exc}")
    try:
        floor_by_submap, floor_transition_keys = resolve_submap_identity_floors(
            submap_rows
        )
    except (KeyError, TypeError, ValueError) as exc:
        floor_by_submap = {}
        floor_transition_keys = set()
        errors.append(f"capture_floor_contract:{type(exc).__name__}:{exc}")
    for row in submap_rows:
        if row.get("record_type") != "submap_action_endpoint":
            continue
        key = (int(row["episode_sequence"]), int(row["action_step"]))
        if key in endpoints:
            errors.append(f"capture_duplicate_endpoint:{key}")
        endpoints[key] = row

    frame_counts: Counter[tuple[int, str]] = Counter()
    episode_frame_counts: Counter[int] = Counter()
    admissions: dict[tuple[int, str], list[int]] = defaultdict(list)
    referenced_files: set[Path] = set()
    for frame in frames:
        key = (frame.episode_sequence, frame.action_step)
        endpoint = endpoints.get(key)
        if endpoint is None:
            errors.append(f"capture_missing_endpoint:{key}")
            continue
        if str(endpoint.get("submap_id")) != frame.submap_id:
            errors.append(f"capture_submap_binding:{frame.frame_id}")
        writer_floor = floor_by_submap.get(frame.episode_sequence, {}).get(
            frame.submap_id
        )
        if writer_floor is None or writer_floor != frame.floor_id:
            errors.append(f"capture_floor_binding:{frame.frame_id}")
        for name, captured, recorded in (
            ("world", frame.world_pose_vo, endpoint.get("world_pose_vo")),
            ("local", frame.local_pose, endpoint.get("local_pose")),
        ):
            try:
                equal = np.allclose(
                    captured,
                    np.asarray(recorded, dtype=np.float64),
                    rtol=0.0,
                    atol=1e-9,
                )
            except (TypeError, ValueError):
                equal = False
            if not equal:
                errors.append(f"capture_{name}_pose_binding:{frame.frame_id}")
        if not np.allclose(
            frame.tf_camera_to_submap[:2, 3],
            frame.local_pose[:2],
            rtol=0.0,
            atol=1e-9,
        ):
            errors.append(f"capture_camera_translation:{frame.frame_id}")
        yaw = math.atan2(
            float(frame.tf_camera_to_submap[1, 0]),
            float(frame.tf_camera_to_submap[0, 0]),
        )
        yaw_error = (yaw - float(frame.local_pose[2]) + math.pi) % (2 * math.pi) - math.pi
        if abs(yaw_error) > 1e-9:
            errors.append(f"capture_camera_yaw:{frame.frame_id}")
        submap_key = (frame.episode_sequence, frame.submap_id)
        frame_counts[submap_key] += 1
        episode_frame_counts[frame.episode_sequence] += 1
        admissions[submap_key].append(frame.admission_index)
        referenced_files.update((frame.rgb_path.resolve(), frame.depth_path.resolve()))

    if any(count > 8 for count in frame_counts.values()):
        errors.append("capture_submap_bound_exceeded")
    for key, values in admissions.items():
        if sorted(values) != list(range(len(values))):
            errors.append(f"capture_admission_sequence:{key}:{sorted(values)}")
    missing_frame_episodes = sorted(
        set(expected_sequences) - set(episode_frame_counts)
    )
    if missing_frame_episodes:
        errors.append(f"capture_episode_without_keyframe:{missing_frame_episodes}")
    actual_frame_files = {
        path.resolve()
        for path in (capture_manifest.parent / "frames").rglob("*")
        if path.is_file()
    }
    if actual_frame_files != referenced_files:
        errors.append(
            "capture_unbound_frame_files:"
            f"{len(actual_frame_files - referenced_files)}:"
            f"{len(referenced_files - actual_frame_files)}"
        )

    report = {
        "capture_manifest": str(capture_manifest),
        "capture_manifest_sha256": sha256(capture_manifest),
        "keyframe_count": len(frames),
        "episode_count": len(expected_sequences),
        "submap_with_keyframes_count": len(frame_counts),
        "max_keyframes_per_submap_observed": max(frame_counts.values(), default=0),
        "episode_frame_counts": {
            str(key): value for key, value in sorted(episode_frame_counts.items())
        },
        "runtime_episode_order": runtime_order,
        "validated_floor_transition_count": len(floor_transition_keys),
    }
    return report, errors
