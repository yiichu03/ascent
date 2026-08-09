from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ascent.submaps.vpr_shadow import (
    VPR_SHADOW_CONTRACT,
    VPRShadowKeyframeConfig,
    VPRShadowKeyframeWriter,
)


def _observe(
    writer: VPRShadowKeyframeWriter,
    *,
    step: int,
    x: float,
    yaw: float = 0.0,
    episode_sequence: int = 0,
    submap_id: str = "env0_s0",
    valid_depth: bool = True,
) -> bool:
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[..., 0] = step % 255
    depth = np.full(
        (24, 32), 0.5 if valid_depth else 0.0, dtype=np.float32
    )
    camera = np.eye(4, dtype=np.float64)
    camera[0, 3] = x
    return writer.observe(
        env=0,
        episode_sequence=episode_sequence,
        action_step=step,
        submap_id=submap_id,
        floor_id=0,
        world_pose_vo=[x, 0.0, yaw],
        local_pose=[x, 0.0, yaw],
        tf_camera_to_submap=camera,
        rgb=rgb,
        normalized_depth=depth,
        min_depth=0.5,
        max_depth=5.0,
        fx=16.0,
        fy=16.0,
    )


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_frozen_keyframe_contract_is_bounded() -> None:
    config = VPRShadowKeyframeConfig()
    assert config.milestone_action_offsets == (
        0,
        4,
        10,
        20,
        40,
        80,
        160,
        320,
    )
    assert config.max_keyframes_per_submap == 8
    assert config.min_translation_m == pytest.approx(0.5)
    assert np.rad2deg(config.min_yaw_rad) == pytest.approx(30.0)


def test_writer_admits_only_causal_milestones_and_round_trips_depth(
    tmp_path: Path,
) -> None:
    output = tmp_path / "capture"
    writer = VPRShadowKeyframeWriter(
        output, metadata={"run_id": "unit", "pose_source": "zhao_rgbd_2021"}
    )
    admitted = []
    for step in range(400):
        if _observe(writer, step=step, x=0.2 * step):
            admitted.append(step)
    writer.close()

    assert admitted == [0, 4, 10, 20, 40, 80, 160, 320]
    rows = _records(output / "keyframes.jsonl")
    assert rows[0]["contract"] == VPR_SHADOW_CONTRACT
    assert rows[0]["planner_write_access"] is False
    frames = [row for row in rows if row["record_type"] == "vpr_shadow_keyframe"]
    assert len(frames) == 8
    assert [row["admission_index"] for row in frames] == list(range(8))

    first = frames[0]
    depth_path = output / first["depth_path"]
    encoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    assert encoded.dtype == np.uint16
    decoded = encoded.astype(np.float32) / 65535.0
    np.testing.assert_allclose(decoded, 0.5, atol=1.0 / 65535.0)
    for key in ("rgb_path", "depth_path"):
        path = output / first[key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == first[
            key.replace("_path", "_file_sha256")
        ]


def test_motion_gate_waits_without_spending_a_milestone(tmp_path: Path) -> None:
    writer = VPRShadowKeyframeWriter(tmp_path / "capture", metadata={})
    assert _observe(writer, step=0, x=0.0)
    for step in range(1, 10):
        assert not _observe(writer, step=step, x=0.0)
    assert _observe(writer, step=10, x=0.6)
    writer.close()

    frames = [
        row
        for row in _records(tmp_path / "capture/keyframes.jsonl")
        if row["record_type"] == "vpr_shadow_keyframe"
    ]
    assert [row["action_step"] for row in frames] == [0, 10]
    assert frames[1]["milestone_action_offset"] == 4


def test_invalid_depth_is_unknown_and_not_saved(tmp_path: Path) -> None:
    writer = VPRShadowKeyframeWriter(tmp_path / "capture", metadata={})
    assert not _observe(writer, step=0, x=0.0, valid_depth=False)
    assert _observe(writer, step=1, x=0.0, valid_depth=True)
    writer.close()
    frames = [
        row
        for row in _records(tmp_path / "capture/keyframes.jsonl")
        if row["record_type"] == "vpr_shadow_keyframe"
    ]
    assert len(frames) == 1
    assert frames[0]["action_step"] == 1
    assert frames[0]["action_offset"] == 1


def test_episode_reset_discards_only_bounded_in_memory_state(
    tmp_path: Path,
) -> None:
    writer = VPRShadowKeyframeWriter(tmp_path / "capture", metadata={})
    assert _observe(writer, step=0, x=0.0, episode_sequence=0)
    writer.reset_env(env=0, episode_sequence=1)
    assert _observe(writer, step=0, x=0.0, episode_sequence=1)
    assert len(writer._active_by_env) == 1
    assert writer._active_by_env[0].episode_sequence == 1
    writer.close()


def test_writer_rejects_evaluation_or_outcome_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evaluation-only"):
        VPRShadowKeyframeWriter(
            tmp_path / "capture", metadata={"gt_pose": [0.0, 0.0, 0.0]}
        )
    with pytest.raises(ValueError, match="evaluation-only"):
        VPRShadowKeyframeWriter(
            tmp_path / "capture2", metadata={"success": 1}
        )
