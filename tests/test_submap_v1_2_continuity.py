from __future__ import annotations

from types import SimpleNamespace

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
    select_connected_handoff_waypoint,
    transform_camera_to_destination,
    waypoint_in_robot_component,
)


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
