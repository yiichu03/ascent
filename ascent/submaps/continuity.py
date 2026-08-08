"""Boundary-continuity primitives for conservative ASCENT submaps.

These helpers consume policy-visible Zhao-VO poses and RGB-D geometry only.
They deliberately do not accept Habitat evaluation poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

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
    live_frontier_confirmation_pending: bool = False
    ray_origin_local: Optional[np.ndarray] = None
    inherited_target_local: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        waypoint = np.asarray(self.waypoint_local, dtype=np.float64)
        if waypoint.shape != (2,) or not np.isfinite(waypoint).all():
            raise ValueError("handoff waypoint must be finite XY")
        self.waypoint_local = waypoint.copy()
        for field_name in ("ray_origin_local", "inherited_target_local"):
            value = getattr(self, field_name)
            if value is None:
                continue
            point = np.asarray(value, dtype=np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(f"handoff {field_name} must be finite XY")
            setattr(self, field_name, point.copy())
        if self.live_frontier_confirmation_pending and (
            self.ray_origin_local is None
            or self.inherited_target_local is None
        ):
            raise ValueError(
                "pending live-frontier confirmation requires a ray origin "
                "and inherited target"
            )

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


@dataclass(frozen=True)
class LiveFrontierConfirmation:
    """Result and audit counts for one destination-map confirmation."""

    waypoint_local: Optional[np.ndarray]
    reason: str
    raw_candidate_count: int
    live_candidate_count: int
    enabled_candidate_count: int
    connected_execution_waypoint_count: int
    forward_candidate_count: int
    corridor_candidate_count: int
    selected_live_frontier_local: Optional[np.ndarray] = None
    selected_projection_m: Optional[float] = None
    selected_lateral_distance_m: Optional[float] = None
    selected_target_distance_m: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in (
            "waypoint_local",
            "selected_live_frontier_local",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            point = np.asarray(value, dtype=np.float64)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(
                    f"live-frontier confirmation {field_name} must be finite XY"
                )
            point = point.copy()
            point.setflags(write=False)
            object.__setattr__(self, field_name, point)

    def event_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "reason": self.reason,
            "raw_candidate_count": int(self.raw_candidate_count),
            "live_candidate_count": int(self.live_candidate_count),
            "enabled_candidate_count": int(self.enabled_candidate_count),
            "connected_execution_waypoint_count": int(
                self.connected_execution_waypoint_count
            ),
            "forward_candidate_count": int(self.forward_candidate_count),
            "corridor_candidate_count": int(
                self.corridor_candidate_count
            ),
        }
        if self.waypoint_local is not None:
            payload["waypoint_local"] = self.waypoint_local.tolist()
        if self.selected_live_frontier_local is not None:
            payload["selected_live_frontier_local"] = (
                self.selected_live_frontier_local.tolist()
            )
            payload["selected_projection_m"] = float(
                self.selected_projection_m
            )
            payload["selected_lateral_distance_m"] = float(
                self.selected_lateral_distance_m
            )
            payload["selected_target_distance_m"] = float(
                self.selected_target_distance_m
            )
        return payload


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


def confirm_handoff_with_live_frontiers(
    obstacle_map: object,
    *,
    robot_xy: Sequence[float],
    ray_origin_xy: Sequence[float],
    inherited_target_xy: Sequence[float],
    corridor_radius_m: float,
) -> LiveFrontierConfirmation:
    """Confirm live directional evidence, then project a safe waypoint.

    The destination map is expected to have consumed its normal current RGB-D
    update before this function is called.  A frontier lies on the dilated
    explored contour, so it is evidence rather than an execution waypoint.
    Candidate ordering never affects the answer: target distance, X, then Y
    form the deterministic tie-break before safe connected projection.
    """

    robot = np.asarray(robot_xy, dtype=np.float64)
    origin = np.asarray(ray_origin_xy, dtype=np.float64)
    target = np.asarray(inherited_target_xy, dtype=np.float64)
    if any(
        point.shape != (2,) or not np.isfinite(point).all()
        for point in (robot, origin, target)
    ):
        raise ValueError("live-frontier confirmation inputs must be finite XY")
    radius = float(corridor_radius_m)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("live-frontier corridor radius must be nonnegative")

    raw = np.asarray(obstacle_map.frontiers, dtype=np.float64)
    if raw.size == 0:
        raw = np.empty((0, 2), dtype=np.float64)
    elif raw.size % 2 != 0:
        raise ValueError("live frontier array must contain XY pairs")
    else:
        raw = raw.reshape(-1, 2)
    raw_count = len(raw)

    live = raw[
        np.isfinite(raw).all(axis=1)
        & ~np.all(raw == np.zeros(2, dtype=np.float64), axis=1)
    ]
    live_count = len(live)
    disabled = obstacle_map._disabled_frontiers
    enabled = np.asarray(
        [point for point in live if tuple(point) not in disabled],
        dtype=np.float64,
    )
    if enabled.size == 0:
        enabled = np.empty((0, 2), dtype=np.float64)
    else:
        enabled = enabled.reshape(-1, 2)
    enabled_count = len(enabled)
    direction = target - origin
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= np.finfo(np.float64).eps:
        return LiveFrontierConfirmation(
            waypoint_local=None,
            reason="invalid_inherited_ray",
            raw_candidate_count=raw_count,
            live_candidate_count=live_count,
            enabled_candidate_count=enabled_count,
            connected_execution_waypoint_count=0,
            forward_candidate_count=0,
            corridor_candidate_count=0,
        )
    unit_direction = direction / direction_norm
    offsets = enabled - origin
    projections = offsets @ unit_direction
    forward_mask = projections > 0.0
    forward = enabled[forward_mask]
    forward_projections = projections[forward_mask]
    forward_count = len(forward)
    lateral = np.linalg.norm(
        (forward - origin)
        - forward_projections.reshape(-1, 1) * unit_direction,
        axis=1,
    )
    corridor_mask = lateral <= radius
    corridor = forward[corridor_mask]
    corridor_projections = forward_projections[corridor_mask]
    corridor_lateral = lateral[corridor_mask]
    corridor_count = len(corridor)

    if corridor_count == 0:
        if raw_count == 0 or live_count == 0:
            reason = "no_live_frontiers"
        elif enabled_count == 0:
            reason = "no_enabled_frontiers"
        elif forward_count == 0:
            reason = "no_forward_frontiers"
        else:
            reason = "no_frontiers_in_ray_corridor"
        return LiveFrontierConfirmation(
            waypoint_local=None,
            reason=reason,
            raw_candidate_count=raw_count,
            live_candidate_count=live_count,
            enabled_candidate_count=enabled_count,
            connected_execution_waypoint_count=0,
            forward_candidate_count=forward_count,
            corridor_candidate_count=0,
        )

    target_distances = np.linalg.norm(corridor - target, axis=1)
    order = np.lexsort(
        (corridor[:, 1], corridor[:, 0], target_distances)
    )
    selected = int(order[0])
    selected_live_frontier = corridor[selected]
    execution_waypoint = select_connected_handoff_waypoint(
        obstacle_map,
        robot_xy=robot,
        old_frontier_direction_xy=selected_live_frontier,
    )
    if execution_waypoint is None:
        return LiveFrontierConfirmation(
            waypoint_local=None,
            reason="no_connected_execution_waypoint",
            raw_candidate_count=raw_count,
            live_candidate_count=live_count,
            enabled_candidate_count=enabled_count,
            connected_execution_waypoint_count=0,
            forward_candidate_count=forward_count,
            corridor_candidate_count=corridor_count,
            selected_live_frontier_local=selected_live_frontier,
            selected_projection_m=float(corridor_projections[selected]),
            selected_lateral_distance_m=float(corridor_lateral[selected]),
            selected_target_distance_m=float(target_distances[selected]),
        )
    return LiveFrontierConfirmation(
        waypoint_local=execution_waypoint,
        reason="confirmed",
        raw_candidate_count=raw_count,
        live_candidate_count=live_count,
        enabled_candidate_count=enabled_count,
        connected_execution_waypoint_count=1,
        forward_candidate_count=forward_count,
        corridor_candidate_count=corridor_count,
        selected_live_frontier_local=selected_live_frontier,
        selected_projection_m=float(corridor_projections[selected]),
        selected_lateral_distance_m=float(corridor_lateral[selected]),
        selected_target_distance_m=float(target_distances[selected]),
    )


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
