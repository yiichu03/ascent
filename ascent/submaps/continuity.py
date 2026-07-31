"""Boundary-continuity primitives for conservative ASCENT submaps.

These helpers consume policy-visible Zhao-VO poses and RGB-D geometry only.
They deliberately do not accept Habitat evaluation poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ascent.submaps.geometry import pose_to_matrix


@dataclass(frozen=True)
class DepthGeometryFrame:
    """One normalized depth observation expressed in a source submap."""

    depth: np.ndarray
    tf_camera_to_source: np.ndarray
    min_depth: float
    max_depth: float
    fx: float
    fy: float
    camera_fov: float
    action_step: int

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float32)
        transform = np.asarray(self.tf_camera_to_source, dtype=np.float64)
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ValueError("depth replay frame must be a finite 2-D array")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera replay transform must be finite 4x4")
        if self.max_depth <= self.min_depth:
            raise ValueError("depth replay range must be positive")
        depth = depth.copy()
        transform = transform.copy()
        depth.setflags(write=False)
        transform.setflags(write=False)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "tf_camera_to_source", transform)


@dataclass
class BoundaryHandoff:
    """A single connected waypoint inherited across one low-overlap split."""

    waypoint_local: np.ndarray
    source_submap_id: str
    destination_submap_id: str
    created_step: int
    reference_distance_m: float
    stagnation_decisions: int = 0

    def __post_init__(self) -> None:
        waypoint = np.asarray(self.waypoint_local, dtype=np.float64)
        if waypoint.shape != (2,) or not np.isfinite(waypoint).all():
            raise ValueError("handoff waypoint must be finite XY")
        self.waypoint_local = waypoint.copy()

    def observe_distance(
        self,
        distance_m: float,
        *,
        progress_threshold_m: float,
        max_stagnation_decisions: int,
    ) -> bool:
        """Return true when the inherited waypoint has become stuck."""

        distance = float(distance_m)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError("handoff distance must be finite and nonnegative")
        if abs(distance - self.reference_distance_m) >= progress_threshold_m:
            self.reference_distance_m = distance
            self.stagnation_decisions = 0
            return False
        self.stagnation_decisions += 1
        return self.stagnation_decisions >= max_stagnation_decisions


@dataclass
class ExhaustionRecovery:
    """Per-episode one-shot no-frontier recovery state."""

    used: bool = False
    active: bool = False
    turns_remaining: int = 0
    started_step: Optional[int] = None
    submap_id: Optional[str] = None

    def start(
        self,
        *,
        started_step: int,
        submap_id: str,
        total_turns: int,
        turns_already_issued: int,
    ) -> None:
        if self.used:
            raise RuntimeError("exhaustion recovery is one-shot per episode")
        remaining = int(total_turns) - int(turns_already_issued)
        if remaining < 0:
            raise ValueError("issued recovery turns exceed the scan budget")
        self.used = True
        self.active = True
        self.turns_remaining = remaining
        self.started_step = int(started_step)
        self.submap_id = str(submap_id)

    def issue_turn(self) -> int:
        if not self.active or self.turns_remaining <= 0:
            raise RuntimeError("recovery scan has no turn available")
        self.turns_remaining -= 1
        return self.turns_remaining

    def finish(self) -> None:
        self.active = False
        self.turns_remaining = 0


def transform_camera_to_destination(
    tf_camera_to_source: np.ndarray,
    source_anchor_world: Sequence[float],
    destination_anchor_world: Sequence[float],
) -> np.ndarray:
    """Re-express a camera transform between two Zhao-VO anchored frames."""

    source_from_camera = np.asarray(tf_camera_to_source, dtype=np.float64)
    if source_from_camera.shape != (4, 4):
        raise ValueError("camera transform must have shape (4, 4)")
    world_from_source_2d = pose_to_matrix(source_anchor_world)
    world_from_destination_2d = pose_to_matrix(destination_anchor_world)
    destination_from_source_2d = (
        np.linalg.inv(world_from_destination_2d) @ world_from_source_2d
    )
    destination_from_source = np.eye(4, dtype=np.float64)
    destination_from_source[:2, :2] = destination_from_source_2d[:2, :2]
    destination_from_source[:2, 3] = destination_from_source_2d[:2, 2]
    return destination_from_source @ source_from_camera


def replay_depth_geometry(
    frames: Sequence[DepthGeometryFrame],
    *,
    source_anchor_world: Sequence[float],
    destination_anchor_world: Sequence[float],
    obstacle_map: object,
) -> int:
    """Warm only obstacle/free-space geometry in a successor map."""

    replayed = 0
    for frame in frames[-4:]:
        transform = transform_camera_to_destination(
            frame.tf_camera_to_source,
            source_anchor_world,
            destination_anchor_world,
        )
        obstacle_map.update_geometry_only(
            frame.depth,
            transform,
            frame.min_depth,
            frame.max_depth,
            frame.fx,
            frame.fy,
            frame.camera_fov,
        )
        replayed += 1
    return replayed


def select_connected_handoff_waypoint(
    obstacle_map: object,
    *,
    robot_xy: Sequence[float],
    old_frontier_direction_xy: Sequence[float],
) -> Optional[np.ndarray]:
    """Pick the furthest observed safe point along an inherited direction."""

    robot = np.asarray(robot_xy, dtype=np.float64)
    target = np.asarray(old_frontier_direction_xy, dtype=np.float64)
    if (
        robot.shape != (2,)
        or target.shape != (2,)
        or not np.isfinite(robot).all()
        or not np.isfinite(target).all()
        or np.array_equal(robot, target)
    ):
        return None

    labels, robot_label = _robot_component(obstacle_map, robot)
    if robot_label is None:
        return None
    robot_px = np.asarray(
        obstacle_map._xy_to_px(robot.reshape(1, 2))[0], dtype=np.int64
    )
    target_px = np.asarray(
        obstacle_map._xy_to_px(target.reshape(1, 2))[0], dtype=np.int64
    )
    delta = target_px - robot_px
    samples = int(max(abs(int(delta[0])), abs(int(delta[1])))) + 1
    if samples <= 1:
        return None
    line = np.rint(
        np.linspace(robot_px, target_px, num=samples, endpoint=True)
    ).astype(np.int64)
    height, width = labels.shape
    valid = (
        (line[:, 0] >= 0)
        & (line[:, 0] < width)
        & (line[:, 1] >= 0)
        & (line[:, 1] < height)
    )
    line = line[valid]
    if len(line) == 0:
        return None
    candidates = line[labels[line[:, 1], line[:, 0]] == robot_label]
    if len(candidates) == 0:
        return None
    candidates = candidates[
        np.linalg.norm(candidates - robot_px, axis=1) > 0.0
    ]
    if len(candidates) == 0:
        return None
    waypoint = np.asarray(
        obstacle_map._px_to_xy(candidates[-1:].copy())[0],
        dtype=np.float64,
    )
    if waypoint.shape != (2,) or not np.isfinite(waypoint).all():
        return None
    return waypoint


def waypoint_in_robot_component(
    obstacle_map: object,
    *,
    robot_xy: Sequence[float],
    waypoint_xy: Sequence[float],
) -> bool:
    """Check whether a waypoint remains in the robot's observed safe component."""

    robot = np.asarray(robot_xy, dtype=np.float64)
    waypoint = np.asarray(waypoint_xy, dtype=np.float64)
    labels, robot_label = _robot_component(obstacle_map, robot)
    if robot_label is None or waypoint.shape != (2,) or not np.isfinite(waypoint).all():
        return False
    pixel = np.asarray(
        obstacle_map._xy_to_px(waypoint.reshape(1, 2))[0], dtype=np.int64
    )
    height, width = labels.shape
    if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        return False
    return int(labels[pixel[1], pixel[0]]) == robot_label


def _robot_component(
    obstacle_map: object, robot_xy: np.ndarray
) -> tuple[np.ndarray, Optional[int]]:
    safe = np.asarray(obstacle_map.explored_area, dtype=bool) & np.asarray(
        obstacle_map._strict_navigable_map, dtype=bool
    )
    if safe.ndim != 2 or not safe.any():
        return np.zeros_like(safe, dtype=np.int32), None
    _, labels = cv2.connectedComponents(safe.astype(np.uint8), connectivity=8)
    robot_px = np.asarray(
        obstacle_map._xy_to_px(robot_xy.reshape(1, 2))[0], dtype=np.int64
    )
    height, width = safe.shape
    label = None
    if 0 <= robot_px[0] < width and 0 <= robot_px[1] < height:
        candidate = int(labels[robot_px[1], robot_px[0]])
        if candidate > 0:
            label = candidate
    return labels, label
