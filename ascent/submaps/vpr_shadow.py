"""Bounded, policy-passive RGB-D keyframes for VPR diagnostics.

This module is deliberately a writer only.  It cannot query place matches or
return a planner decision, and it rejects evaluation-only pose vocabulary.
The resulting keyframes are consumed after navigation by the v1.4 shadow
association tools.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


VPR_SHADOW_CONTRACT = "ascent_v1.4_vpr_shadow_keyframes_v1"


@dataclass(frozen=True)
class VPRShadowKeyframeConfig:
    """Frozen causal keyframe policy used before any VPR score is observed."""

    milestone_action_offsets: tuple[int, ...] = (
        0,
        4,
        10,
        20,
        40,
        80,
        160,
        320,
    )
    min_translation_m: float = 0.50
    min_yaw_rad: float = math.radians(30.0)
    min_depth_valid_fraction: float = 0.25
    jpeg_quality: int = 95
    depth_quantization_levels: int = 65535

    def __post_init__(self) -> None:
        milestones = tuple(int(value) for value in self.milestone_action_offsets)
        if not milestones or milestones[0] != 0:
            raise ValueError("keyframe milestones must start at action offset zero")
        if any(value < 0 for value in milestones):
            raise ValueError("keyframe milestones must be nonnegative")
        if any(a >= b for a, b in zip(milestones, milestones[1:])):
            raise ValueError("keyframe milestones must be strictly increasing")
        if self.min_translation_m <= 0.0:
            raise ValueError("min_translation_m must be positive")
        if not 0.0 < self.min_yaw_rad <= math.pi:
            raise ValueError("min_yaw_rad must be in (0, pi]")
        if not 0.0 <= self.min_depth_valid_fraction <= 1.0:
            raise ValueError("min_depth_valid_fraction must be in [0, 1]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.depth_quantization_levels != 65535:
            raise ValueError("v1 keyframe depth encoding is frozen at uint16/65535")
        object.__setattr__(self, "milestone_action_offsets", milestones)

    @property
    def max_keyframes_per_submap(self) -> int:
        return len(self.milestone_action_offsets)

    def as_dict(self) -> dict[str, Any]:
        return {
            "milestone_action_offsets": list(self.milestone_action_offsets),
            "max_keyframes_per_submap": self.max_keyframes_per_submap,
            "min_translation_m": float(self.min_translation_m),
            "min_yaw_rad": float(self.min_yaw_rad),
            "min_depth_valid_fraction": float(self.min_depth_valid_fraction),
            "jpeg_quality": int(self.jpeg_quality),
            "depth_quantization_levels": int(self.depth_quantization_levels),
        }


@dataclass
class _ActiveBank:
    episode_sequence: int
    submap_id: str
    start_action_step: int
    selected_local_poses: list[np.ndarray] = field(default_factory=list)


class VPRShadowKeyframeWriter:
    """Append a bounded set of causal RGB-D frames without planner feedback."""

    _FORBIDDEN_KEYS = {
        "gps",
        "compass",
        "ground_truth",
        "gt_pose",
        "gt_start_aligned_pose",
        "success",
        "spl",
    }

    def __init__(
        self,
        output_dir: Path,
        *,
        metadata: Mapping[str, Any],
        config: VPRShadowKeyframeConfig | None = None,
    ) -> None:
        self.config = config or VPRShadowKeyframeConfig()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "keyframes.jsonl"
        self._stream = self.manifest_path.open(
            "x", encoding="utf-8", buffering=1
        )
        self._active_by_env: dict[int, _ActiveBank] = {}
        self._write(
            {
                "record_type": "vpr_shadow_run_metadata",
                "contract": VPR_SHADOW_CONTRACT,
                "policy_gt_isolation": True,
                "planner_write_access": False,
                "association_runtime_enabled": False,
                "keyframe_config": self.config.as_dict(),
                **dict(metadata),
            }
        )

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def reset_env(self, *, env: int, episode_sequence: int) -> None:
        """Drop the only in-memory bank when an episode boundary is observed."""

        self._active_by_env.pop(int(env), None)
        self._write(
            {
                "record_type": "vpr_shadow_episode_reset",
                "env": int(env),
                "episode_sequence": int(episode_sequence),
            }
        )

    def observe(
        self,
        *,
        env: int,
        episode_sequence: int,
        action_step: int,
        submap_id: str,
        floor_id: int,
        world_pose_vo: Sequence[float],
        local_pose: Sequence[float],
        tf_camera_to_submap: np.ndarray,
        rgb: np.ndarray,
        normalized_depth: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> bool:
        """Persist the observation iff the frozen causal selector admits it."""

        env = int(env)
        episode_sequence = int(episode_sequence)
        action_step = int(action_step)
        submap_id = str(submap_id)
        bank = self._active_by_env.get(env)
        if (
            bank is None
            or bank.episode_sequence != episode_sequence
            or bank.submap_id != submap_id
        ):
            bank = _ActiveBank(
                episode_sequence=episode_sequence,
                submap_id=submap_id,
                start_action_step=action_step,
            )
            self._active_by_env[env] = bank

        admission_index = len(bank.selected_local_poses)
        if admission_index >= self.config.max_keyframes_per_submap:
            return False
        action_offset = action_step - bank.start_action_step
        milestone = self.config.milestone_action_offsets[admission_index]
        if action_offset < milestone:
            return False

        local = _finite_pose(local_pose, "local_pose")
        world = _finite_pose(world_pose_vo, "world_pose_vo")
        camera_tf = np.asarray(tf_camera_to_submap, dtype=np.float64)
        if camera_tf.shape != (4, 4) or not np.isfinite(camera_tf).all():
            raise ValueError("tf_camera_to_submap must be a finite 4x4 matrix")
        if bank.selected_local_poses:
            previous = bank.selected_local_poses[-1]
            translation = float(np.linalg.norm(local[:2] - previous[:2]))
            yaw = abs(_wrap_angle(float(local[2] - previous[2])))
            if (
                translation < self.config.min_translation_m
                and yaw < self.config.min_yaw_rad
            ):
                return False

        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"rgb must have shape HxWx3, got {image.shape}")
        if image.dtype != np.uint8:
            if not np.isfinite(image).all():
                raise ValueError("rgb must be finite")
            image = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image = np.ascontiguousarray(image)
        depth = np.asarray(normalized_depth, dtype=np.float32)
        if depth.ndim != 2 or depth.shape != image.shape[:2]:
            raise ValueError(
                "normalized_depth must match the RGB height and width"
            )
        valid = np.isfinite(depth) & (depth > 0.0) & (depth < 1.0)
        valid_fraction = float(np.mean(valid))
        if valid_fraction < self.config.min_depth_valid_fraction:
            return False
        if not (
            np.isfinite([min_depth, max_depth, fx, fy]).all()
            and 0.0 <= float(min_depth) < float(max_depth)
            and float(fx) > 0.0
            and float(fy) > 0.0
        ):
            raise ValueError("invalid finite RGB-D calibration")

        relative_dir = (
            Path("frames")
            / f"episode_{episode_sequence:04d}_env_{env}"
            / _safe_component(submap_id)
        )
        stem = f"kf_{admission_index:02d}_step_{action_step:04d}"
        rgb_relative = relative_dir / f"{stem}.rgb.jpg"
        depth_relative = relative_dir / f"{stem}.depth.png"
        rgb_path = self.output_dir / rgb_relative
        depth_path = self.output_dir / depth_relative
        rgb_path.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _imwrite_atomic(
            rgb_path,
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)],
        )
        safe_depth = np.where(valid, np.clip(depth, 0.0, 1.0), 0.0)
        encoded_depth = np.rint(
            safe_depth * self.config.depth_quantization_levels
        ).astype(np.uint16)
        _imwrite_atomic(
            depth_path,
            encoded_depth,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )

        bank.selected_local_poses.append(local.copy())
        self._write(
            {
                "record_type": "vpr_shadow_keyframe",
                "env": env,
                "episode_sequence": episode_sequence,
                "action_step": action_step,
                "action_offset": action_offset,
                "milestone_action_offset": int(milestone),
                "admission_index": admission_index,
                "submap_id": submap_id,
                "floor_id": int(floor_id),
                "world_pose_vo": world.tolist(),
                "local_pose": local.tolist(),
                "tf_camera_to_submap": camera_tf.tolist(),
                "rgb_path": rgb_relative.as_posix(),
                "depth_path": depth_relative.as_posix(),
                "rgb_file_sha256": _sha256(rgb_path),
                "depth_file_sha256": _sha256(depth_path),
                "rgb_shape": list(image.shape),
                "depth_shape": list(depth.shape),
                "depth_encoding": "normalized_uint16_over_65535",
                "depth_valid_fraction": valid_fraction,
                "min_depth": float(min_depth),
                "max_depth": float(max_depth),
                "fx": float(fx),
                "fy": float(fy),
            }
        )
        return True

    def _write(self, record: Mapping[str, Any]) -> None:
        lowered = {str(key).lower() for key in record}
        forbidden = lowered.intersection(self._FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(
                "evaluation-only field reached VPR shadow writer: "
                f"{sorted(forbidden)}"
            )
        self._stream.write(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        )


def _finite_pose(values: Sequence[float], name: str) -> np.ndarray:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite SE(2) pose")
    return pose


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _safe_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not safe:
        safe = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return safe[:96]


def _imwrite_atomic(
    path: Path, image: np.ndarray, parameters: list[int]
) -> None:
    temporary = path.with_name(
        f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    )
    if not cv2.imwrite(str(temporary), image, parameters):
        raise OSError(f"OpenCV failed to write {temporary}")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
