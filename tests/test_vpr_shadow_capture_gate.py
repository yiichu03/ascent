from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ascent.submaps.vpr_shadow import VPRShadowKeyframeWriter
from scripts.vpr_shadow_capture_gate import (
    ACTION_HASH_CONTRACT,
    canonical_action_hash,
    validate_capture_attempt,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_action_hash_is_unambiguous_and_stable() -> None:
    assert ACTION_HASH_CONTRACT.endswith("plus_lf)")
    assert canonical_action_hash([1, 23]) != canonical_action_hash([12, 3])
    assert canonical_action_hash([1, 2, 3]) == canonical_action_hash((1, 2, 3))


def test_capture_gate_binds_frames_to_submap_endpoints(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    writer = VPRShadowKeyframeWriter(
        capture,
        metadata={
            "run_id": "unit",
            "ascent_source_commit": "a" * 40,
            "dataset": "hm3d",
            "pose_source": "zhao_rgbd_2021",
            "navigation_behavior": "frozen_submap_v1.2",
            "capture_point": "pre_planner_policy_visible_observation",
        },
    )
    writer.reset_env(env=0, episode_sequence=0)
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    depth = np.full((24, 32), 0.5, dtype=np.float32)
    endpoints = []
    for step, x in ((0, 0.0), (4, 0.6)):
        transform = np.eye(4)
        transform[0, 3] = x
        assert writer.observe(
            env=0,
            episode_sequence=0,
            action_step=step,
            submap_id="sm0",
            floor_id=0,
            world_pose_vo=[x, 0.0, 0.0],
            local_pose=[x, 0.0, 0.0],
            tf_camera_to_submap=transform,
            rgb=rgb,
            normalized_depth=depth,
            min_depth=0.5,
            max_depth=5.0,
            fx=16.0,
            fy=16.0,
        )
    writer.close()
    for step in range(5):
        x = 0.6 if step == 4 else 0.0
        endpoints.append(
            {
                "record_type": "submap_action_endpoint",
                "episode_sequence": 0,
                "action_step": step,
                "submap_id": "sm0",
                "floor_id": 0,
                "world_pose_vo": [x, 0.0, 0.0],
                "local_pose": [x, 0.0, 0.0],
            }
        )
    submap = tmp_path / "submap.jsonl"
    _write_jsonl(submap, endpoints)
    vo = tmp_path / "vo.jsonl"
    _write_jsonl(
        vo,
        [
            {
                "record_type": "vo_step",
                "episode_id": "0",
                "action_step": step,
                "action": 1,
            }
            for step in range(1, 6)
        ]
        + [{"record_type": "episode_end", "episode_id": "0"}],
    )
    report, errors = validate_capture_attempt(
        capture_manifest=capture / "keyframes.jsonl",
        submap_diagnostics=submap,
        vo_diagnostics=vo,
        source_commit="a" * 40,
        dataset="hm3d",
    )
    assert errors == []
    assert report["keyframe_count"] == 2
    assert report["episode_frame_counts"] == {"0": 2}
