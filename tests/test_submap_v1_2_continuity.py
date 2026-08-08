from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from ascent.ascent_policy import Ascent_Policy
from ascent.mapping.obstacle_map import ObstacleMap
from ascent.submaps import (
    BoundaryHandoff,
    DepthGeometryFrame,
    ExhaustionRecovery,
    MapPayload,
    SubmapLifecycleConfig,
    SubmapManager,
    confirm_handoff_with_live_frontiers,
    select_connected_handoff_waypoint,
    transform_camera_to_destination,
    waypoint_in_robot_component,
)
from constants import MOVE_FORWARD


def _map() -> ObstacleMap:
    return ObstacleMap(
        min_height=0.2,
        max_height=1.2,
        agent_radius=0.2,
        size=100,
        pixels_per_meter=10,
    )


def _payload() -> MapPayload:
    return MapPayload(
        obstacle_map=_map(),
        value_map=SimpleNamespace(),
        object_map=SimpleNamespace(),
    )


def _mark_connected_region(
    obstacle: ObstacleMap, points: list[np.ndarray]
) -> None:
    pixels = obstacle._xy_to_px(np.asarray(points, dtype=np.float64))
    x0 = max(0, int(np.min(pixels[:, 0])) - 1)
    x1 = min(obstacle.explored_area.shape[1], int(np.max(pixels[:, 0])) + 2)
    y0 = max(0, int(np.min(pixels[:, 1])) - 1)
    y1 = min(obstacle.explored_area.shape[0], int(np.max(pixels[:, 1])) + 2)
    obstacle.explored_area[y0:y1, x0:x1] = True
    obstacle._strict_navigable_map[y0:y1, x0:x1] = True


def _obstacle_with_detected_frontier(
    shape: str,
) -> tuple[ObstacleMap, np.ndarray, np.ndarray, np.ndarray]:
    obstacle = _map()
    obstacle._tight_search_thresh = True
    obstacle._navigable_map.fill(True)
    obstacle._strict_navigable_map.fill(True)
    explored = np.zeros_like(obstacle.explored_area, dtype=np.uint8)
    if shape == "circle":
        cv2.circle(explored, (50, 50), 18, 1, -1)
    elif shape == "rectangle":
        cv2.rectangle(explored, (30, 35), (70, 65), 1, -1)
    elif shape == "corridor":
        cv2.rectangle(explored, (45, 45), (75, 55), 1, -1)
    else:
        raise ValueError(shape)
    obstacle.explored_area[:] = explored.astype(bool)
    frontier_pixels = obstacle._get_frontiers()
    assert len(frontier_pixels) == 1
    obstacle._frontiers_px = frontier_pixels
    obstacle.frontiers = obstacle._px_to_xy(frontier_pixels)
    robot = obstacle._px_to_xy(np.array([[50, 50]]))[0]
    frontier = obstacle.frontiers[0]
    rounded_frontier_px = np.rint(frontier_pixels[0]).astype(np.int64)
    assert not obstacle.explored_area[
        rounded_frontier_px[1], rounded_frontier_px[0]
    ]
    return obstacle, robot, frontier, frontier_pixels[0]


def _policy_for_handoff_execution(
    obstacle: ObstacleMap,
    handoff: BoundaryHandoff,
    *,
    live_confirmation_enabled: bool,
) -> tuple[Ascent_Policy, list[tuple[str, dict]], list[np.ndarray]]:
    policy = object.__new__(Ascent_Policy)
    policy._submap_handoff = [handoff]
    policy._submap_handoff_live_frontier_confirmation_enabled = (
        live_confirmation_enabled
    )
    policy._submap_config = SubmapLifecycleConfig(enabled=True)
    policy._observations_cache = [{"robot_xy": np.zeros(2)}]
    policy._map_controller = SimpleNamespace(_obstacle_map=[obstacle])
    policy._pointnav_stop_radius = 0.2
    policy._last_goal = [handoff.waypoint_local.copy()]
    policy.cur_frontier = [handoff.waypoint_local.copy()]
    events: list[tuple[str, dict]] = []
    targets: list[np.ndarray] = []

    def record(_env: int, event: str, **payload: object) -> None:
        events.append((event, dict(payload)))

    def pointnav(
        _observations: object,
        target: np.ndarray,
        *,
        stop: bool,
        env: int,
        stop_radius: float,
    ) -> torch.Tensor:
        assert not stop
        assert env == 0
        assert stop_radius == policy._pointnav_stop_radius
        targets.append(np.asarray(target, dtype=np.float64).copy())
        return torch.tensor([MOVE_FORWARD], dtype=torch.int64)

    policy._record_submap_policy_event = record
    policy._pointnav = pointnav
    return policy, events, targets


def test_camera_replay_uses_only_relative_vo_anchors() -> None:
    source_camera = np.eye(4)
    source_camera[:2, 3] = [2.0, 1.0]
    transformed = transform_camera_to_destination(
        source_camera,
        source_anchor_world=[10.0, 4.0, 0.0],
        destination_anchor_world=[12.0, 4.0, 0.0],
    )
    np.testing.assert_allclose(transformed[:2, 3], [0.0, 1.0])


def test_depth_replay_frame_is_immutable_and_rejects_bad_pose() -> None:
    depth = np.ones((4, 4), dtype=np.float32)
    frame = DepthGeometryFrame(
        depth=depth,
        tf_camera_to_source=np.eye(4),
        min_depth=0.0,
        max_depth=4.0,
        fx=2.0,
        fy=2.0,
        camera_fov=np.pi / 2,
        action_step=3,
    )
    depth.fill(0.0)
    assert np.all(frame.depth == 1.0)
    with pytest.raises(ValueError, match="4x4"):
        DepthGeometryFrame(
            depth=np.ones((4, 4)),
            tf_camera_to_source=np.eye(3),
            min_depth=0.0,
            max_depth=4.0,
            fx=2.0,
            fy=2.0,
            camera_fov=np.pi / 2,
            action_step=0,
        )


def test_handoff_waypoint_stays_in_robot_observed_component() -> None:
    obstacle = _map()
    robot = np.array([0.0, 0.0])
    robot_px = obstacle._xy_to_px(robot.reshape(1, 2))[0]
    target = np.array([2.0, 0.0])
    target_px = obstacle._xy_to_px(target.reshape(1, 2))[0]
    x0, x1 = sorted([int(robot_px[0]), int(target_px[0])])
    y = int(robot_px[1])
    obstacle.explored_area[y - 1 : y + 2, x0 : x1 + 1] = True
    obstacle._strict_navigable_map[y - 1 : y + 2, x0 : x1 + 1] = True

    waypoint = select_connected_handoff_waypoint(
        obstacle,
        robot_xy=robot,
        old_frontier_direction_xy=target,
    )
    assert waypoint is not None
    assert np.linalg.norm(waypoint - robot) > 0.0
    assert waypoint_in_robot_component(
        obstacle, robot_xy=robot, waypoint_xy=waypoint
    )

    obstacle._strict_navigable_map.fill(False)
    assert not waypoint_in_robot_component(
        obstacle, robot_xy=robot, waypoint_xy=waypoint
    )


def test_handoff_stuck_guard_reuses_frozen_ascent_constants() -> None:
    state = BoundaryHandoff(
        waypoint_local=np.array([2.0, 0.0]),
        source_submap_id="old",
        destination_submap_id="new",
        created_step=10,
        reference_distance_m=2.0,
    )
    for _ in range(19):
        assert not state.observe_distance(
            1.9,
            progress_threshold_m=0.3,
            max_stagnation_decisions=20,
        )
    assert state.observe_distance(
        1.9,
        progress_threshold_m=0.3,
        max_stagnation_decisions=20,
    )
    assert not state.observe_distance(
        1.5,
        progress_threshold_m=0.3,
        max_stagnation_decisions=20,
    )


def test_live_frontier_confirmation_retargets_deterministically() -> None:
    obstacle = _map()
    robot = np.array([0.0, 0.0])
    target = np.array([3.0, 0.0])
    lower_tie = np.array([2.8, -0.1])
    upper_tie = np.array([2.8, 0.1])
    obstacle.frontiers = np.vstack(
        [np.zeros(2), upper_tie, np.array([2.4, 0.2]), lower_tie]
    )
    _mark_connected_region(
        obstacle, [robot, target, lower_tie, upper_tie]
    )

    result = confirm_handoff_with_live_frontiers(
        obstacle,
        robot_xy=robot,
        ray_origin_xy=np.zeros(2),
        inherited_target_xy=target,
        corridor_radius_m=1.0,
    )

    assert result.reason == "confirmed"
    np.testing.assert_allclose(
        result.selected_live_frontier_local, lower_tie
    )
    assert waypoint_in_robot_component(
        obstacle, robot_xy=robot, waypoint_xy=result.waypoint_local
    )
    assert result.raw_candidate_count == 4
    assert result.live_candidate_count == 3
    assert result.corridor_candidate_count == 3
    assert result.connected_execution_waypoint_count == 1


def test_live_frontier_confirmation_filters_disabled_candidate() -> None:
    obstacle = _map()
    robot = np.array([0.0, 0.0])
    disabled = np.array([2.9, 0.0])
    enabled = np.array([2.5, 0.2])
    obstacle.frontiers = np.vstack([disabled, enabled])
    obstacle._disabled_frontiers.add(tuple(disabled))
    _mark_connected_region(obstacle, [robot, disabled, enabled])

    result = confirm_handoff_with_live_frontiers(
        obstacle,
        robot_xy=robot,
        ray_origin_xy=np.zeros(2),
        inherited_target_xy=np.array([3.0, 0.0]),
        corridor_radius_m=1.0,
    )

    assert result.reason == "confirmed"
    assert result.enabled_candidate_count == 1
    np.testing.assert_allclose(
        result.selected_live_frontier_local, enabled
    )
    assert waypoint_in_robot_component(
        obstacle, robot_xy=robot, waypoint_xy=result.waypoint_local
    )


def test_live_frontier_confirmation_rejects_without_safe_execution_point() -> None:
    obstacle = _map()
    robot = np.array([0.0, 0.0])
    candidate = np.array([2.0, 0.0])
    obstacle.frontiers = candidate.reshape(1, 2)
    robot_px = obstacle._xy_to_px(robot.reshape(1, 2))[0]
    x, y = int(robot_px[0]), int(robot_px[1])
    obstacle.explored_area[y, x] = True
    obstacle._strict_navigable_map[y, x] = True

    result = confirm_handoff_with_live_frontiers(
        obstacle,
        robot_xy=robot,
        ray_origin_xy=np.zeros(2),
        inherited_target_xy=np.array([3.0, 0.0]),
        corridor_radius_m=1.0,
    )

    assert result.waypoint_local is None
    assert result.reason == "no_connected_execution_waypoint"
    assert result.connected_execution_waypoint_count == 0
    assert result.corridor_candidate_count == 1
    np.testing.assert_allclose(
        result.selected_live_frontier_local, candidate
    )


@pytest.mark.parametrize("shape", ["circle", "rectangle", "corridor"])
def test_real_detector_frontier_confirms_via_safe_projection(
    shape: str,
) -> None:
    obstacle, robot, live_frontier, _ = _obstacle_with_detected_frontier(
        shape
    )
    inherited_target = robot + 1.2 * (live_frontier - robot)

    result = confirm_handoff_with_live_frontiers(
        obstacle,
        robot_xy=robot,
        ray_origin_xy=robot,
        inherited_target_xy=inherited_target,
        corridor_radius_m=1.0,
    )

    assert result.reason == "confirmed"
    np.testing.assert_allclose(
        result.selected_live_frontier_local, live_frontier
    )
    assert not np.allclose(result.waypoint_local, live_frontier)
    assert waypoint_in_robot_component(
        obstacle, robot_xy=robot, waypoint_xy=result.waypoint_local
    )


def test_handoff_flag_off_keeps_v1_2_waypoint_without_live_frontier() -> None:
    obstacle = _map()
    old_waypoint = np.array([1.5, 0.0])
    obstacle.frontiers = np.zeros((1, 2), dtype=np.float64)
    _mark_connected_region(obstacle, [np.zeros(2), old_waypoint])
    handoff = BoundaryHandoff(
        waypoint_local=old_waypoint,
        source_submap_id="old",
        destination_submap_id="new",
        created_step=10,
        reference_distance_m=1.5,
    )
    policy, events, targets = _policy_for_handoff_execution(
        obstacle, handoff, live_confirmation_enabled=False
    )

    action = policy._execute_handoff(
        {}, 0, torch.ones((1, 1), dtype=torch.bool)
    )

    assert action is not None and action.item() == MOVE_FORWARD
    np.testing.assert_allclose(targets, [old_waypoint])
    assert [event for event, _ in events] == ["handoff_action"]


def test_handoff_confirms_and_executes_retarget_in_same_decision() -> None:
    obstacle, robot, live_frontier, _ = _obstacle_with_detected_frontier(
        "corridor"
    )
    inherited_target = robot + 1.2 * (live_frontier - robot)
    expected_waypoint = select_connected_handoff_waypoint(
        obstacle,
        robot_xy=robot,
        old_frontier_direction_xy=live_frontier,
    )
    assert expected_waypoint is not None
    handoff = BoundaryHandoff(
        waypoint_local=np.array([1.2, 0.0]),
        source_submap_id="old",
        destination_submap_id="new",
        created_step=10,
        reference_distance_m=1.2,
        live_frontier_confirmation_pending=True,
        ray_origin_local=np.zeros(2),
        inherited_target_local=inherited_target,
    )
    policy, events, targets = _policy_for_handoff_execution(
        obstacle, handoff, live_confirmation_enabled=True
    )

    action = policy._execute_handoff(
        {}, 0, torch.ones((1, 1), dtype=torch.bool)
    )

    assert action is not None and action.item() == MOVE_FORWARD
    np.testing.assert_allclose(targets, [expected_waypoint])
    assert not handoff.live_frontier_confirmation_pending
    assert [event for event, _ in events] == [
        "handoff_live_frontier_confirmed",
        "handoff_action",
    ]


def test_handoff_rejection_returns_control_without_pointnav_action() -> None:
    obstacle = _map()
    old_waypoint = np.array([1.2, 0.0])
    obstacle.frontiers = np.zeros((1, 2), dtype=np.float64)
    _mark_connected_region(obstacle, [np.zeros(2), old_waypoint])
    handoff = BoundaryHandoff(
        waypoint_local=old_waypoint,
        source_submap_id="old",
        destination_submap_id="new",
        created_step=10,
        reference_distance_m=1.2,
        live_frontier_confirmation_pending=True,
        ray_origin_local=np.zeros(2),
        inherited_target_local=np.array([3.0, 0.0]),
    )
    policy, events, targets = _policy_for_handoff_execution(
        obstacle, handoff, live_confirmation_enabled=True
    )

    action = policy._execute_handoff(
        {}, 0, torch.ones((1, 1), dtype=torch.bool)
    )

    assert action is None
    assert policy._submap_handoff[0] is None
    assert targets == []
    assert [event for event, _ in events] == [
        "handoff_live_frontier_rejected",
        "handoff_cancelled",
    ]
    assert events[0][1]["reason"] == "no_live_frontiers"


def test_exhaustion_recovery_is_one_shot_and_counts_first_turn() -> None:
    state = ExhaustionRecovery()
    state.start(
        started_step=30,
        submap_id="recovery",
        total_turns=12,
        turns_already_issued=1,
    )
    assert state.used and state.active
    assert state.turns_remaining == 11
    for _ in range(11):
        state.issue_turn()
    assert state.turns_remaining == 0
    state.finish()
    with pytest.raises(RuntimeError, match="one-shot"):
        state.start(
            started_step=50,
            submap_id="second",
            total_turns=12,
            turns_already_issued=1,
        )


def test_manager_exhaustion_boundary_is_explicit_and_same_floor() -> None:
    manager = SubmapManager(1, SubmapLifecycleConfig(enabled=True))
    old = manager.start(0, [0.0, 0.0, 0.0], 0, _payload(), 0)
    new, edge, _ = manager.commit_exhaustion_recovery(
        env=0,
        world_pose=[1.0, 0.0, 0.0],
        floor_id=0,
        new_payload=_payload(),
        step=20,
        frontiers_local=np.empty((0, 2)),
    )
    assert old.state.value == "frozen"
    assert new.state.value == "active"
    assert edge.kind == "exhaustion_recovery"
    assert manager.events(0)[-1]["event"] == "submap_exhaustion_recovery"


def test_recovery_request_requires_13_actions_and_never_repeats() -> None:
    policy = object.__new__(Ascent_Policy)
    policy._submap_exhaustion_recovery_enabled = True
    policy._submap_exhaustion_recovery = [ExhaustionRecovery()]
    policy.max_episode_steps = 500
    policy._num_steps = [487]
    policy._observations_cache = [{"robot_xy": np.zeros(2)}]
    policy._get_target_object_location = lambda position, env: None
    policy._record_submap_policy_event = lambda *args, **kwargs: None
    action = policy._request_exhaustion_recovery(
        0, torch.ones((1, 1), dtype=torch.bool)
    )
    assert action is not None and action.item() == 2
    assert policy._observations_cache[0][
        "submap_exhaustion_recovery_request"
    ]

    policy._submap_exhaustion_recovery[0].used = True
    assert (
        policy._request_exhaustion_recovery(
            0, torch.ones((1, 1), dtype=torch.bool)
        )
        is None
    )

    policy._submap_exhaustion_recovery[0] = ExhaustionRecovery()
    policy._num_steps[0] = 488
    assert (
        policy._request_exhaustion_recovery(
            0, torch.ones((1, 1), dtype=torch.bool)
        )
        is None
    )
