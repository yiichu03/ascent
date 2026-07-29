"""Episode-local SE(2) pose provider backed by Zhao RGB-D VO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from ascent.vo.zhao_model import (
    DEPTH_INVALID_POLICY,
    DepthValidityStats,
    PreparedZhaoFrame,
    ZhaoActionModels,
    ZhaoFramePreprocessor,
)


STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
LOOK_UP = 4
LOOK_DOWN = 5

VO_RGB_KEY = "vo_rgb"
VO_DEPTH_KEY = "vo_depth"
FORBIDDEN_POLICY_OBSERVATION_KEYS = frozenset(
    {
        "gps",
        "compass",
        "heading",
        "base_explorer",
        "frontier_sensor",
        "gt_start_aligned_pose",
        VO_RGB_KEY,
        VO_DEPTH_KEY,
    }
)


class VOInferenceError(RuntimeError):
    """A fail-closed VO input, model, or composition failure."""


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def compose_zhao_delta(
    pose: np.ndarray, local_delta: Sequence[float]
) -> np.ndarray:
    """Compose Zhao [right, habitat-z, left-yaw] into ASCENT [forward,left]."""

    pose = np.asarray(pose, dtype=np.float64)
    delta = np.asarray(local_delta, dtype=np.float64)
    if pose.shape != (3,) or delta.shape != (3,):
        raise VOInferenceError(
            f"bad SE(2) shapes pose={pose.shape}, delta={delta.shape}"
        )
    if not np.isfinite(pose).all() or not np.isfinite(delta).all():
        raise VOInferenceError("non-finite SE(2) input")
    right, habitat_z, delta_yaw = delta
    local_forward_left = np.array([-habitat_z, -right])
    yaw = float(pose[2])
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    output = pose.copy()
    output[:2] += rotation @ local_forward_left
    output[2] = wrap_angle(yaw + float(delta_yaw))
    if not np.isfinite(output).all():
        raise VOInferenceError("non-finite composed pose")
    return output


@dataclass(frozen=True)
class PoseUpdate:
    env: int
    action: int
    status: str
    pose_before: tuple[float, float, float]
    pose_after: tuple[float, float, float]
    local_deltas: tuple[tuple[float, float, float], ...]
    vo_inferences: int
    next_episode_reset: bool = False
    previous_depth_validity: DepthValidityStats | None = None
    current_depth_validity: DepthValidityStats | None = None
    current_frame_role: str = "unspecified"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "env": self.env,
            "action": self.action,
            "status": self.status,
            "pose_before": list(self.pose_before),
            "pose_after": list(self.pose_after),
            "local_deltas": [list(delta) for delta in self.local_deltas],
            "vo_inferences": self.vo_inferences,
            "next_episode_reset": self.next_episode_reset,
            "depth_invalid_policy": DEPTH_INVALID_POLICY,
            "previous_depth_validity": (
                None
                if self.previous_depth_validity is None
                else self.previous_depth_validity.as_dict()
            ),
            "current_depth_validity": (
                None
                if self.current_depth_validity is None
                else self.current_depth_validity.as_dict()
            ),
            "current_frame_role": self.current_frame_role,
            "finite": True,
        }


class ZhaoRGBDPoseProvider:
    """Maintains one GT-free episode-local planar estimate per environment."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device,
        checkpoint_dir: Path,
        source_min_depth: float,
        source_max_depth: float,
        source_hfov_degrees: float,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.models = ZhaoActionModels(checkpoint_dir, device)
        self.preprocessor = ZhaoFramePreprocessor(
            device=device,
            source_min_depth=source_min_depth,
            source_max_depth=source_max_depth,
            source_hfov_degrees=source_hfov_degrees,
        )
        self._poses = [
            np.zeros(3, dtype=np.float64) for _ in range(num_envs)
        ]
        self._previous_frames: List[PreparedZhaoFrame | None] = [
            None for _ in range(num_envs)
        ]

    @property
    def poses(self) -> List[np.ndarray]:
        return [pose.copy() for pose in self._poses]

    def reset_batch(
        self, observations: List[Dict[str, Any]]
    ) -> List[PoseUpdate]:
        if len(observations) != self.num_envs:
            raise VOInferenceError(
                f"expected {self.num_envs} reset observations, got {len(observations)}"
            )
        return [
            self._reset_env(observation, env)
            for env, observation in enumerate(observations)
        ]

    def update_batch(
        self,
        observations: List[Dict[str, Any]],
        actions: Sequence[int],
        dones: Sequence[bool],
    ) -> List[PoseUpdate]:
        if not (
            len(observations)
            == len(actions)
            == len(dones)
            == self.num_envs
        ):
            raise VOInferenceError("VO batch lengths do not match num_envs")
        return [
            self._update_env(
                observations[env], int(actions[env]), bool(dones[env]), env
            )
            for env in range(self.num_envs)
        ]

    def inject_estimated_pose(
        self, observations: List[Dict[str, Any]]
    ) -> None:
        if len(observations) != self.num_envs:
            raise VOInferenceError("pose injection batch length mismatch")
        for env, observation in enumerate(observations):
            forbidden = FORBIDDEN_POLICY_OBSERVATION_KEYS.intersection(
                observation
            )
            if forbidden:
                raise VOInferenceError(
                    f"policy observation still contains forbidden keys: {sorted(forbidden)}"
                )
            observation["estimated_pose"] = self._poses[env].astype(
                np.float32
            )

    def _reset_env(
        self, observation: Dict[str, Any], env: int
    ) -> PoseUpdate:
        rgb, depth = self._pop_single_frame(observation)
        prepared = self.preprocessor.prepare(rgb, depth)
        self._poses[env] = np.zeros(3, dtype=np.float64)
        self._previous_frames[env] = prepared
        zero = (0.0, 0.0, 0.0)
        return PoseUpdate(
            env=env,
            action=STOP,
            status="episode_reset",
            pose_before=zero,
            pose_after=zero,
            local_deltas=(),
            vo_inferences=0,
            next_episode_reset=True,
            current_depth_validity=_frame_depth_validity(prepared),
            current_frame_role="episode_reset",
        )

    def _update_env(
        self,
        observation: Dict[str, Any],
        action: int,
        done: bool,
        env: int,
    ) -> PoseUpdate:
        pose_before = self._poses[env].copy()
        if done:
            # VectorEnv already replaced terminal observations with the next
            # episode's reset frame. No future policy decision consumes the
            # terminal transition. STOP is identity by contract.
            terminal_status = (
                "terminal_identity_then_autoreset"
                if action == STOP
                else "terminal_no_policy_successor_then_autoreset"
            )
            previous_depth_validity = _frame_depth_validity(
                self._previous_frames[env]
            )
            reset_update = self._reset_env(observation, env)
            return PoseUpdate(
                env=env,
                action=action,
                status=terminal_status,
                pose_before=tuple(map(float, pose_before)),
                pose_after=tuple(map(float, pose_before)),
                local_deltas=(),
                vo_inferences=0,
                next_episode_reset=True,
                previous_depth_validity=previous_depth_validity,
                current_depth_validity=reset_update.current_depth_validity,
                current_frame_role="next_episode_reset",
            )

        previous = self._previous_frames[env]
        if previous is None:
            raise VOInferenceError(f"environment {env} was not reset")

        if action in (LOOK_UP, LOOK_DOWN, STOP):
            rgb, depth = self._pop_single_frame(observation)
            current = self.preprocessor.prepare(rgb, depth)
            self._previous_frames[env] = current
            return PoseUpdate(
                env=env,
                action=action,
                status="identity",
                pose_before=tuple(map(float, pose_before)),
                pose_after=tuple(map(float, pose_before)),
                local_deltas=(),
                vo_inferences=0,
                previous_depth_validity=_frame_depth_validity(previous),
                current_depth_validity=_frame_depth_validity(current),
                current_frame_role="identity_refresh",
            )

        if action not in (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT):
            raise VOInferenceError(f"unsupported action id {action}")

        rgb, depth = self._pop_single_frame(observation)
        try:
            current = self.preprocessor.prepare(rgb, depth)
            # The released checkpoint was trained and deployed with one
            # adjacent pair per native action: 0.25 m forward or 30-degree
            # turn. Do not subdivide, rescale, or blend this estimate.
            delta = self.models.estimate(action, previous, current)
            pose = compose_zhao_delta(pose_before, delta)
        except Exception as exc:
            raise VOInferenceError(
                f"VO action {action} failed closed in env {env}: {exc}"
            ) from exc

        self._poses[env] = pose
        self._previous_frames[env] = current
        return PoseUpdate(
            env=env,
            action=action,
            status="ok",
            pose_before=tuple(map(float, pose_before)),
            pose_after=tuple(map(float, pose)),
            local_deltas=(tuple(map(float, delta)),),
            vo_inferences=1,
            previous_depth_validity=_frame_depth_validity(previous),
            current_depth_validity=_frame_depth_validity(current),
            current_frame_role="action_successor",
        )

    @staticmethod
    def _pop_single_frame(
        observation: Dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            rgb = np.asarray(observation.pop(VO_RGB_KEY))
            depth = np.asarray(observation.pop(VO_DEPTH_KEY))
        except KeyError as exc:
            raise VOInferenceError(
                f"missing auxiliary VO sensor {exc.args[0]}"
            ) from exc
        return rgb, depth


def _frame_depth_validity(
    frame: PreparedZhaoFrame | Any | None,
) -> DepthValidityStats | None:
    if frame is None:
        return None
    validity = getattr(frame, "depth_validity", None)
    if validity is not None and not isinstance(validity, DepthValidityStats):
        raise VOInferenceError(
            f"bad prepared-frame depth validity type {type(validity).__name__}"
        )
    return validity
