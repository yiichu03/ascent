from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ascent.submaps import (
    DepthGeometryFrame,
    ExhaustionRecovery,
    FrontierStatus,
    MapPayload,
    RemoteRoute,
    SubmapDiagnosticsWriter,
    SubmapLifecycleConfig,
    SubmapManager,
    SubmapQueryView,
    ViewOverlapConfig,
    estimate_view_overlap,
    relative_pose,
    transfer_stair_topology,
    transform_points_xy,
)
from ascent.mapping.obstacle_map import ObstacleMap
from ascent.ascent_policy import Ascent_Policy
from ascent.map_controller import Map_Controller


def payload(tag: str) -> MapPayload:
    return MapPayload(
        obstacle_map=SimpleNamespace(tag=f"obstacle-{tag}"),
        value_map=SimpleNamespace(tag=f"value-{tag}"),
        object_map=SimpleNamespace(tag=f"object-{tag}"),
    )


def _obstacle_map(
    *, allow_step_zero_frontier_projection: bool = False
) -> ObstacleMap:
    return ObstacleMap(
        min_height=0.1,
        max_height=1.5,
        area_thresh=1.0,
        agent_radius=0.18,
        hole_area_thresh=0.1,
        size=64,
        allow_step_zero_frontier_projection=(
            allow_step_zero_frontier_projection
        ),
    )


def test_episode_start_preserves_ascent_step_zero_frontier_guard() -> None:
    obstacle_map = _obstacle_map()
    frontier = np.array([0.8, -0.75], dtype=np.float64)
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    obstacle_map.frontiers = frontier.reshape(1, 2)

    assert obstacle_map._floor_num_steps == 0
    obstacle_map.project_frontiers_to_rgb_hush(rgb)

    assert obstacle_map.previous_frontiers == []
    assert obstacle_map.frontier_visualization_info == {}
    assert obstacle_map._each_step_rgb == {}


def test_fresh_submap_frontier_has_step_zero_visualization_record() -> None:
    obstacle_map = _obstacle_map(
        allow_step_zero_frontier_projection=True
    )
    frontier = np.array([0.8, -0.75], dtype=np.float64)
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    obstacle_map.frontiers = frontier.reshape(1, 2)

    assert obstacle_map._floor_num_steps == 0
    obstacle_map.project_frontiers_to_rgb_hush(rgb)

    assert obstacle_map.frontier_visualization_info[tuple(frontier)] == {
        "floor_num_steps": 0
    }
    floor_step, cached_rgb = obstacle_map.extract_frontiers_with_image(
        frontier
    )
    assert floor_step == 0
    np.testing.assert_array_equal(cached_rgb, rgb)


def test_episode_reset_revokes_successor_step_zero_permission() -> None:
    obstacle_map = _obstacle_map(
        allow_step_zero_frontier_projection=True
    )
    frontier = np.array([0.8, -0.75], dtype=np.float64)
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)

    obstacle_map.reset()
    obstacle_map.frontiers = frontier.reshape(1, 2)
    obstacle_map.project_frontiers_to_rgb_hush(rgb)

    assert obstacle_map._allow_step_zero_frontier_projection is False
    assert obstacle_map.frontier_visualization_info == {}


def test_relative_pose_and_point_transform_round_trip() -> None:
    anchor_a = np.array([4.0, -2.0, np.deg2rad(35.0)])
    anchor_b = np.array([-1.0, 3.0, np.deg2rad(-20.0)])
    world_pose = np.array([5.2, 0.4, np.deg2rad(80.0)])
    pose_a = relative_pose(anchor_a, world_pose)
    world_xy = transform_points_xy(pose_a[:2], anchor_a, np.zeros(3))
    np.testing.assert_allclose(world_xy, world_pose[:2], atol=1e-8)

    points_a = np.array([[0.0, 0.0], [1.0, -0.5], [4.0, 2.0]])
    points_b = transform_points_xy(points_a, anchor_a, anchor_b)
    reconstructed = transform_points_xy(points_b, anchor_b, anchor_a)
    np.testing.assert_allclose(reconstructed, points_a, atol=1e-8)


def test_default_off_never_requests_split() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=False,
            min_action_endpoints=1,
            min_path_length_m=0.0,
            max_motion_budget_m=0.1,
        ),
    )
    manager.start(0, [0.0, 0.0, 0.0], 0, payload("initial"), 0)
    decision = manager.observe_action_endpoint(0, [2.0, 0.0, 0.0], 0, 0.0)
    assert not decision.should_split
    assert decision.reason is None


def test_joint_anchor_overlap_split_freezes_old_map_and_creates_gateway() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=2,
            min_anchor_displacement_m=1.0,
            overlap_threshold=0.5,
            low_overlap_consecutive=2,
            gateway_frontier_resolution_radius_m=0.3,
        ),
    )
    first = manager.start(0, [2.0, 1.0, 0.2], 0, payload("first"), 0)
    assert not manager.observe_action_endpoint(
        0, [2.6, 1.0, 0.2], 0, 0.2
    ).should_split
    decision = manager.observe_action_endpoint(0, [3.2, 1.0, 0.2], 0, 0.2)
    assert decision.should_split
    assert decision.reason == "low_overlap"
    assert decision.anchor_displacement_m == pytest.approx(1.2)

    frontiers = np.array([[1.2, 0.0], [4.0, 2.0]])
    second, edge, resolved = manager.commit_split(
        0,
        [3.2, 1.0, 0.2],
        0,
        payload("second"),
        2,
        frontiers,
        frontier_scores=[0.7, 0.4],
    )
    assert first.state.value == "frozen"
    assert second.state.value == "active"
    assert edge.source_submap_id == first.submap_id
    assert edge.destination_submap_id == second.submap_id
    assert len(resolved) == 1
    assert manager.registry(0).get(resolved[0]).status is FrontierStatus.RESOLVED
    with pytest.raises(RuntimeError, match="frozen"):
        first.require_active()
    with pytest.raises(ValueError):
        first.frozen_frontiers_local[0, 0] = 100.0


def test_low_overlap_requires_maturity_and_consecutive_endpoints() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=3,
            min_anchor_displacement_m=0.5,
            overlap_threshold=0.4,
            low_overlap_consecutive=2,
            max_motion_budget_m=100.0,
        ),
    )
    manager.start(0, [0.0, 0.0, 0.0], 0, payload("initial"), 0)
    assert not manager.observe_action_endpoint(
        0, [0.1, 0.0, 0.0], 0, 0.2
    ).should_split
    assert not manager.observe_action_endpoint(
        0, [0.3, 0.0, 0.0], 0, 0.2
    ).should_split
    decision = manager.observe_action_endpoint(0, [0.6, 0.0, 0.0], 0, 0.2)
    assert decision.should_split
    assert decision.reason == "low_overlap"


def test_long_cumulative_motion_without_anchor_displacement_does_not_split() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=4,
            min_anchor_displacement_m=1.5,
            overlap_threshold=0.5,
            low_overlap_consecutive=3,
            max_motion_budget_m=0.1,
            rotation_weight_m_per_rad=10.0,
        ),
    )
    manager.start(0, [0.0, 0.0, 0.0], 0, payload("initial"), 0)
    poses = [
        [1.0, 0.0, np.pi],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, np.pi],
        [0.0, 0.0, 0.0],
    ]
    decision = None
    for pose in poses:
        decision = manager.observe_action_endpoint(0, pose, 0, 0.0)
        assert not decision.should_split
    assert decision is not None
    assert decision.motion_budget_m > 10.0
    assert decision.anchor_displacement_m == pytest.approx(0.0)
    assert not decision.mature


def test_anchor_displacement_without_low_rgbd_overlap_does_not_split() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=2,
            min_anchor_displacement_m=1.0,
            overlap_threshold=0.35,
            low_overlap_consecutive=2,
        ),
    )
    manager.start(0, [0.0, 0.0, 0.0], 0, payload("initial"), 0)
    manager.observe_action_endpoint(0, [0.6, 0.0, 0.0], 0, 0.8)
    decision = manager.observe_action_endpoint(
        0, [1.2, 0.0, 0.0], 0, 0.8
    )

    assert decision.mature
    assert decision.low_overlap_streak == 0
    assert not decision.should_split


def test_missing_overlap_breaks_consecutive_low_overlap_evidence() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=1,
            min_anchor_displacement_m=0.0,
            overlap_threshold=0.35,
            low_overlap_consecutive=2,
        ),
    )
    manager.start(0, [0.0, 0.0, 0.0], 0, payload("initial"), 0)
    first = manager.observe_action_endpoint(
        0, [0.1, 0.0, 0.0], 0, 0.1
    )
    missing = manager.observe_action_endpoint(
        0, [0.2, 0.0, 0.0], 0, None
    )
    final = manager.observe_action_endpoint(
        0, [0.3, 0.0, 0.0], 0, 0.1
    )

    assert first.low_overlap_streak == 1
    assert missing.low_overlap_streak == 0
    assert final.low_overlap_streak == 1
    assert not final.should_split


def test_floor_change_is_a_hard_boundary_before_maturity() -> None:
    manager = SubmapManager(1, SubmapLifecycleConfig(enabled=True))
    manager.start(0, [0.0, 0.0, 0.0], 0, payload("initial"), 0)
    decision = manager.observe_action_endpoint(0, [0.0, 0.0, 0.0], 1, None)
    assert decision.should_split
    assert decision.reason == "floor_change"
    assert not decision.mature


def test_query_view_routes_to_next_gateway_without_mutating_frontiers() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=1,
            min_anchor_displacement_m=0.5,
            overlap_threshold=0.5,
            low_overlap_consecutive=1,
            gateway_frontier_resolution_radius_m=0.1,
        ),
    )
    first = manager.start(0, [0.0, 0.0, 0.0], 0, payload("first"), 0)
    manager.observe_action_endpoint(0, [1.0, 0.0, 0.0], 0, 0.0)
    second, _, _ = manager.commit_split(
        0,
        [1.0, 0.0, 0.0],
        0,
        payload("second"),
        1,
        np.array([[0.0, 2.0]]),
        frontier_scores=[0.8],
    )
    view = SubmapQueryView(manager.graph(0), manager.registry(0), second.submap_id)
    plans = view.remote_frontier_plans()
    assert len(plans) == 1
    assert plans[0].source_submap_id == first.submap_id
    np.testing.assert_allclose(plans[0].next_gateway_local_xy, np.zeros(2))

    plans[0].next_gateway_local_xy[0] = 99.0
    edge = manager.graph(0).next_gateway(second.submap_id, first.submap_id)
    assert edge is not None
    np.testing.assert_allclose(edge.endpoint_for(second.submap_id)[:2], np.zeros(2))


def test_nonfloor_split_is_deferred_during_transition_execution() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=1,
            min_anchor_displacement_m=0.5,
            overlap_threshold=0.5,
            low_overlap_consecutive=1,
        ),
    )
    manager.start(0, [0.0, 0.0, 0.0], 0, payload("initial"), 0)
    decision = manager.observe_action_endpoint(
        0,
        [1.0, 0.0, 0.0],
        0,
        0.0,
        allow_nonfloor_split=False,
    )
    assert decision.motion_budget_m >= 1.0
    assert not decision.should_split


def test_persistent_route_advances_without_creating_revisit_submaps() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=1,
            min_anchor_displacement_m=0.5,
            overlap_threshold=0.5,
            low_overlap_consecutive=1,
            gateway_frontier_resolution_radius_m=0.1,
        ),
    )
    first = manager.start(0, [0.0, 0.0, 0.0], 0, payload("first"), 0)
    manager.observe_action_endpoint(0, [1.0, 0.0, 0.0], 0, 0.0)
    second, gateway, _ = manager.commit_split(
        0,
        [1.0, 0.0, 0.0],
        0,
        payload("second"),
        1,
        np.array([[0.0, 2.0]]),
    )
    route = RemoteRoute.build(
        graph=manager.graph(0),
        route_id="env0:route0000",
        active_submap_id=second.submap_id,
        destination_submap_id=first.submap_id,
        destination_local_xy=np.array([0.0, 2.0]),
        target_kind="frontier",
        candidate_key=f"frontier:{first.submap_id}:f0000",
        frontier_id=f"{first.submap_id}:f0000",
        selected_step=2,
    )

    assert len(route.waypoints) == 2
    assert route.current_waypoint.kind == "gateway"
    assert route.current_waypoint.edge_id == gateway.edge_id
    assert route.current_waypoint.owner_submap_id == second.submap_id
    np.testing.assert_allclose(
        route.project_current_waypoint(), np.zeros(2)
    )
    route.reach_current_waypoint(
        manager.graph(0), np.array([0.2, 0.1])
    )
    assert route.current_waypoint.kind == "destination_frontier"
    assert route.current_waypoint.owner_submap_id == first.submap_id
    np.testing.assert_allclose(
        route.project_current_waypoint(), [-0.8, 2.1]
    )
    assert manager.active_bundle(0).submap_id == second.submap_id
    assert len(manager.graph(0).nodes) == 2
    assert len(manager.graph(0).edges) == 1


def test_attempted_frontier_is_not_eligible_for_another_route() -> None:
    registry = SubmapManager(
        1, SubmapLifecycleConfig(enabled=True)
    ).registry(0)
    record = registry.register_submap_frontiers(
        "env0:sm0000", np.array([[1.0, 2.0]]), 1
    )[0]
    registry.mark_selected(record.frontier_id, 2)
    registry.mark_attempted(record.frontier_id, 3)

    assert record.status is FrontierStatus.ATTEMPTED
    assert registry.eligible() == []


def test_route_no_progress_guard_is_bounded_per_waypoint() -> None:
    manager = SubmapManager(
        1, SubmapLifecycleConfig(enabled=True)
    )
    bundle = manager.start(
        0, [0.0, 0.0, 0.0], 0, payload("initial"), 0
    )
    route = RemoteRoute.build(
        graph=manager.graph(0),
        route_id="env0:route0000",
        active_submap_id=bundle.submap_id,
        destination_submap_id=bundle.submap_id,
        destination_local_xy=np.array([2.0, 0.0]),
        target_kind="semantic",
        candidate_key="semantic:env0:sm0000:chair",
        selected_step=1,
    )

    assert (
        route.observe_distance(
            2.0,
            min_progress_m=0.3,
            max_stagnation_actions=2,
            max_waypoint_actions=10,
        )
        is None
    )
    assert (
        route.observe_distance(
            1.9,
            min_progress_m=0.3,
            max_stagnation_actions=2,
            max_waypoint_actions=10,
        )
        is None
    )
    assert (
        route.observe_distance(
            1.8,
            min_progress_m=0.3,
            max_stagnation_actions=2,
            max_waypoint_actions=10,
        )
        == "no_progress"
    )


def _no_frontier_obstacle(*, reinitialize_flag: bool) -> SimpleNamespace:
    return SimpleNamespace(
        _disabled_frontiers=set(),
        _reinitialize_flag=reinitialize_flag,
        _floor_num_steps=100 if reinitialize_flag else 10,
        _explored_up_stair=True,
        _explored_down_stair=True,
        _up_stair_frontiers=np.empty((0, 2)),
        _down_stair_frontiers=np.empty((0, 2)),
        _this_floor_explored=False,
    )


def test_remote_fallback_runs_only_after_original_no_local_path() -> None:
    policy = object.__new__(Ascent_Policy)
    obstacle = _no_frontier_obstacle(reinitialize_flag=True)
    policy._observations_cache = [
        {"frontier_sensor": np.empty((0, 2))}
    ]
    policy._map_controller = SimpleNamespace(
        _obstacle_map=[obstacle]
    )
    policy._submap_enabled = True
    policy._stop_action = torch.tensor([[0]], dtype=torch.int64)
    calls = []
    policy._submap_remote_fallback_action = (
        lambda observations, env, masks: (
            calls.append("remote")
            or torch.tensor([[1]], dtype=torch.int64)
        )
    )

    action = policy._explore(
        observations=None,
        env=0,
        masks=torch.ones((1, 1), dtype=torch.bool),
    )

    assert action.item() == 1
    assert calls == ["remote"]
    assert obstacle._this_floor_explored


def test_original_stairwell_reinitialization_precedes_remote_fallback() -> None:
    policy = object.__new__(Ascent_Policy)
    obstacle = _no_frontier_obstacle(reinitialize_flag=False)
    obstacle._explored_up_stair = False
    policy._observations_cache = [
        {"frontier_sensor": np.empty((0, 2))}
    ]
    policy._map_controller = SimpleNamespace(
        _obstacle_map=[obstacle]
    )
    policy._submap_enabled = True
    policy._stop_action = torch.tensor([[0]], dtype=torch.int64)
    policy._handle_stairwell_reinitialization = (
        lambda env, masks: torch.tensor([[2]], dtype=torch.int64)
    )
    policy._submap_remote_fallback_action = (
        lambda observations, env, masks: pytest.fail(
            "remote fallback preempted ASCENT reinitialization"
        )
    )

    action = policy._explore(
        observations=None,
        env=0,
        masks=torch.ones((1, 1), dtype=torch.bool),
    )

    assert action.item() == 2


class _OverlapMap:
    def __init__(self, explored: np.ndarray) -> None:
        self.explored_area = explored
        self.pixels_per_meter = 10
        self._origin = np.array([50, 50])

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        px = np.rint(points[:, ::-1] * self.pixels_per_meter) + self._origin
        px[:, 0] = self.explored_area.shape[0] - px[:, 0]
        return px.astype(int)


def test_rgbd_overlap_is_read_only_and_reports_known_view() -> None:
    explored = np.ones((100, 100), dtype=bool)
    before = explored.copy()
    overlap = estimate_view_overlap(
        normalized_depth=np.full((64, 64), 0.4, dtype=np.float32),
        tf_camera_to_local=np.eye(4),
        min_depth=0.0,
        max_depth=4.0,
        fx=32.0,
        fy=32.0,
        obstacle_map=_OverlapMap(explored),
        config=ViewOverlapConfig(
            sample_stride=8,
            ray_samples=3,
            explored_dilation_cells=0,
            min_valid_samples=16,
            min_explored_cells=1,
        ),
    )
    assert overlap == pytest.approx(1.0)
    np.testing.assert_array_equal(explored, before)


def test_submap_diagnostics_rejects_evaluation_pose_keys(tmp_path) -> None:
    path = tmp_path / "submaps.jsonl"
    writer = SubmapDiagnosticsWriter(path, {"pose_source": "vo"})
    writer.record_episode_reset(env=0, episode_sequence=0)
    with pytest.raises(ValueError, match="evaluation-only"):
        writer.record_event(
            env=0,
            episode_sequence=0,
            event={"gt_pose": [0.0, 0.0, 0.0]},
        )
    writer.close()
    with pytest.raises(FileExistsError):
        SubmapDiagnosticsWriter(path, {"pose_source": "vo"})


def test_stair_topology_transfer_does_not_copy_metric_evidence() -> None:
    source = ObstacleMap(
        min_height=0.2,
        max_height=1.2,
        agent_radius=0.2,
        size=100,
        pixels_per_meter=10,
    )
    destination = ObstacleMap(
        min_height=0.2,
        max_height=1.2,
        agent_radius=0.2,
        size=100,
        pixels_per_meter=10,
    )
    source_point = np.array([[1.0, 0.0]])
    source_pixel = source._xy_to_px(source_point)[0]
    source._up_stair_map[source_pixel[1], source_pixel[0]] = True
    source._up_stair_frontiers = source_point.copy()
    source._has_up_stair = True
    source._map[source_pixel[1], source_pixel[0]] = True
    source.explored_area[source_pixel[1], source_pixel[0]] = True

    transfer_stair_topology(
        source_obstacle_map=source,
        destination_obstacle_map=destination,
        source_anchor_world=np.array([0.0, 0.0, 0.0]),
        destination_anchor_world=np.array([1.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(
        destination._up_stair_frontiers, [[0.0, 0.0]]
    )
    destination_pixel = destination._xy_to_px(
        np.array([[0.0, 0.0]])
    )[0]
    assert destination._up_stair_map[
        destination_pixel[1], destination_pixel[0]
    ]
    assert destination._has_up_stair
    assert not destination._map.any()
    assert not destination.explored_area.any()


class _ValueMapStub:
    def sort_waypoints(self, waypoints, radius, reduce_fn=None):
        return np.asarray(waypoints), [0.7] * len(waypoints)


class _ObjectMapStub:
    def has_object(self, target):
        return False


class _PointNavStub:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _ControllerStub:
    def __init__(self, initial, replacement):
        self.payload = initial
        self.replacement = replacement
        self._obstacle_map = [initial.obstacle_map]
        self._policy_floor_id = [0]
        self._cur_floor_index = [0]
        self._climb_stair_over = [True]
        self._climb_stair_flag = [0]
        self.reset_calls = []
        self.step_zero_projection_permissions = []

    def current_map_payload(self, env):
        return self.payload

    def create_empty_map_payload(
        self, *, allow_step_zero_frontier_projection=False
    ):
        self.step_zero_projection_permissions.append(
            allow_step_zero_frontier_projection
        )
        return self.replacement

    def install_map_payload(self, env, payload):
        self.payload = payload
        self._obstacle_map[env] = payload.obstacle_map

    def reset_submap_local_navigation_state(
        self, env, *, initialize_new_floor
    ):
        self.reset_calls.append(initialize_new_floor)


def _runtime_payload() -> MapPayload:
    return MapPayload(
        obstacle_map=ObstacleMap(
            min_height=0.2,
            max_height=1.2,
            agent_radius=0.2,
            size=100,
            pixels_per_meter=10,
        ),
        value_map=_ValueMapStub(),
        object_map=_ObjectMapStub(),
    )


def test_policy_action_end_swaps_to_fresh_active_payload() -> None:
    config = SubmapLifecycleConfig(
        enabled=True,
        min_action_endpoints=1,
        min_anchor_displacement_m=0.5,
        overlap_threshold=0.5,
        low_overlap_consecutive=1,
    )
    initial = _runtime_payload()
    replacement = _runtime_payload()
    initial.obstacle_map.frontiers = np.array([[0.0, 2.0]])
    manager = SubmapManager(1, config)
    old = manager.start(0, [0.0, 0.0, 0.0], 0, initial, 0)

    policy = object.__new__(Ascent_Policy)
    policy._submap_manager = manager
    policy._submap_config = config
    policy._map_controller = _ControllerStub(initial, replacement)
    policy._observations_cache = [
        {
            "world_pose_vo": np.array([1.0, 0.0, 0.0]),
            "robot_xy": np.array([1.0, 0.0]),
            "robot_heading": 0.0,
            "submap_overlap_before_update": 0.0,
        }
    ]
    policy._submap_boundary_frames = [[]]
    policy._submap_depth_frames = [[]]
    policy._submap_handoff_enabled = False
    policy._submap_exhaustion_recovery_enabled = False
    policy._submap_handoff = [None]
    policy._submap_exhaustion_recovery = [ExhaustionRecovery()]
    policy._submap_diagnostics = None
    policy._submap_episode_sequence = [0]
    policy._submap_event_cursor = [0]
    policy._policy_info = [{}]
    policy._pointnav_policy = [_PointNavStub()]
    policy._last_goal = [np.ones(2)]
    policy._try_to_navigate_step = [9]
    policy._try_to_navigate = [True]
    policy._last_frontier_distance = [3.0]
    policy.min_distance_xy = [2.0]
    policy.cur_frontier = [np.ones(2)]
    policy.llm_planner = SimpleNamespace(
        _last_frontier=[np.zeros(2)],
        reset_submap_local_state=lambda env: None
    )

    policy._finish_submap_action(env=0, action_step=1)

    active = manager.active_bundle(0)
    assert old.state.value == "frozen"
    assert active.payload.identity == replacement.identity
    assert policy._map_controller.payload.identity == replacement.identity
    assert policy._map_controller.step_zero_projection_permissions == [True]
    assert policy._pointnav_policy[0].reset_count == 1
    assert policy._policy_info[0]["submap_split_reason"] == "low_overlap"
    assert policy._policy_info[0]["submap_count"] == 2


def _depth_replay_frames_at(x: float) -> list[DepthGeometryFrame]:
    frames = []
    for step, yaw in enumerate(
        [0.0, np.pi / 2, np.pi, -np.pi / 2]
    ):
        transform = np.eye(4, dtype=np.float64)
        transform[:2, :2] = [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)],
        ]
        transform[0, 3] = x
        transform[2, 3] = 0.88
        frames.append(
            DepthGeometryFrame(
                depth=np.full((16, 16), 0.5, dtype=np.float32),
                tf_camera_to_source=transform,
                min_depth=0.0,
                max_depth=4.0,
                fx=8.0,
                fy=8.0,
                camera_fov=np.pi / 2,
                action_step=step,
            )
        )
    return frames


def _boundary_policy(
    initial: MapPayload,
    replacement: MapPayload,
    manager: SubmapManager,
) -> tuple[Ascent_Policy, _ControllerStub, _PointNavStub]:
    controller = _ControllerStub(initial, replacement)
    pointnav = _PointNavStub()
    policy = object.__new__(Ascent_Policy)
    policy._submap_manager = manager
    policy._submap_config = manager.config
    policy._map_controller = controller
    policy._observations_cache = [
        {
            "world_pose_vo": np.array([1.0, 0.0, 0.0]),
            "robot_xy": np.array([1.0, 0.0]),
            "robot_heading": 0.0,
            "submap_overlap_before_update": 0.0,
            "policy_mode": "explore",
        }
    ]
    policy._submap_boundary_frames = [[]]
    policy._submap_depth_frames = [_depth_replay_frames_at(1.0)]
    policy._submap_diagnostics = None
    policy._submap_episode_sequence = [0]
    policy._submap_event_cursor = [0]
    policy._policy_info = [{}]
    policy._pointnav_policy = [pointnav]
    policy._pointnav_stop_radius = 0.9
    policy._last_goal = [np.array([3.0, 0.0])]
    policy._try_to_navigate_step = [0]
    policy._try_to_navigate = [False]
    policy._last_frontier_distance = [2.0]
    policy.min_distance_xy = [np.inf]
    policy.cur_frontier = [np.array([3.0, 0.0])]
    policy._submap_remote_route = [None]
    policy._submap_handoff = [None]
    policy._submap_exhaustion_recovery = [ExhaustionRecovery()]
    policy._pitch_angle = [0]
    policy.llm_planner = SimpleNamespace(
        _last_frontier=[np.array([3.0, 0.0])],
        reset_submap_local_state=lambda env: None,
    )
    return policy, controller, pointnav


def test_low_overlap_handoff_replays_geometry_and_preserves_pointnav() -> None:
    config = SubmapLifecycleConfig(
        enabled=True,
        min_action_endpoints=1,
        min_anchor_displacement_m=0.5,
        overlap_threshold=0.5,
        low_overlap_consecutive=1,
    )
    initial = _runtime_payload()
    replacement = _runtime_payload()
    initial.obstacle_map.frontiers = np.array([[3.0, 0.0]])
    manager = SubmapManager(1, config)
    manager.start(0, [0.0, 0.0, 0.0], 0, initial, 0)
    policy, _, pointnav = _boundary_policy(
        initial, replacement, manager
    )
    policy._submap_handoff_enabled = True
    policy._submap_exhaustion_recovery_enabled = False

    policy._finish_submap_action(env=0, action_step=20)

    assert policy._submap_handoff[0] is not None
    assert pointnav.reset_count == 0
    np.testing.assert_allclose(
        policy._last_goal[0],
        policy._submap_handoff[0].waypoint_local,
    )
    assert policy._policy_info[0][
        "submap_boundary_replayed_depth_frames"
    ] == 4


def test_low_overlap_handoff_rejects_a_stale_current_frontier() -> None:
    config = SubmapLifecycleConfig(
        enabled=True,
        min_action_endpoints=1,
        min_anchor_displacement_m=0.5,
        overlap_threshold=0.5,
        low_overlap_consecutive=1,
    )
    initial = _runtime_payload()
    replacement = _runtime_payload()
    initial.obstacle_map.frontiers = np.array([[4.0, 0.0]])
    manager = SubmapManager(1, config)
    manager.start(0, [0.0, 0.0, 0.0], 0, initial, 0)
    policy, _, pointnav = _boundary_policy(
        initial, replacement, manager
    )
    policy._submap_handoff_enabled = True
    policy._submap_exhaustion_recovery_enabled = False

    policy._finish_submap_action(env=0, action_step=20)

    assert policy._submap_handoff[0] is None
    assert pointnav.reset_count == 1
    assert policy._policy_info[0][
        "submap_boundary_replayed_depth_frames"
    ] == 0


def test_no_frontier_request_commits_one_clean_recovery_submap() -> None:
    config = SubmapLifecycleConfig(
        enabled=True,
        min_action_endpoints=20,
        min_anchor_displacement_m=1.5,
        overlap_threshold=0.35,
        low_overlap_consecutive=3,
    )
    initial = _runtime_payload()
    replacement = _runtime_payload()
    initial.obstacle_map.frontiers = np.array([[2.0, 0.0]])
    initial.obstacle_map._disabled_frontiers.add((2.0, 0.0))
    manager = SubmapManager(1, config)
    old = manager.start(0, [0.0, 0.0, 0.0], 0, initial, 0)
    policy, _, pointnav = _boundary_policy(
        initial, replacement, manager
    )
    policy._submap_handoff_enabled = True
    policy._submap_exhaustion_recovery_enabled = True
    policy._observations_cache[0][
        "submap_exhaustion_recovery_request"
    ] = True

    policy._finish_submap_action(env=0, action_step=40)

    active = manager.active_bundle(0)
    state = policy._submap_exhaustion_recovery[0]
    assert old.state.value == "frozen"
    assert old.frozen_frontiers_local.shape == (0, 2)
    assert active.parent_submap_id == old.submap_id
    assert manager.events(0)[-1]["event"] == "submap_exhaustion_recovery"
    assert state.used and state.active and state.turns_remaining == 11
    assert pointnav.reset_count == 1
    assert policy._submap_handoff[0] is None
    assert policy._policy_info[0]["submap_split_reason"] == (
        "no_frontier_exhaustion"
    )


def test_policy_binding_guard_rejects_a_frozen_or_foreign_payload() -> None:
    manager = SubmapManager(1, SubmapLifecycleConfig(enabled=True))
    active_payload = payload("active")
    manager.start(0, [0.0, 0.0, 0.0], 0, active_payload, 0)
    policy = object.__new__(Ascent_Policy)
    policy._submap_manager = manager
    policy._map_controller = SimpleNamespace(
        current_map_payload=lambda env: payload("foreign")
    )
    with pytest.raises(RuntimeError, match="not the active submap"):
        policy._assert_active_submap_binding(0)


def test_disabled_policy_never_queries_remote_submap_semantics() -> None:
    class _ForbiddenManager:
        def __getattr__(self, name):
            raise AssertionError(
                f"disabled submap policy accessed manager method {name}"
            )

    policy = object.__new__(Ascent_Policy)
    policy._submap_enabled = False
    policy._submap_manager = _ForbiddenManager()

    assert (
        policy._submap_remote_semantic_action(
            observations=None,
            env=0,
            masks=torch.zeros((1, 1), dtype=torch.bool),
        )
        is None
    )


def test_controller_frame_switch_clears_ephemeral_stair_coordinates() -> None:
    controller = object.__new__(Map_Controller)
    controller._carrot_goal_xy = [np.array([[1.0, 2.0]])]
    controller._last_carrot_xy = [np.array([[1.0, 2.0]])]
    controller._last_carrot_px = [np.array([[10, 20]])]
    controller._stair_frontier = [np.array([[3.0, 4.0]])]
    controller._temp_stair_map = [np.ones((4, 4), dtype=bool)]
    controller._frontier_stick_step = [8]
    controller._get_close_to_stair_step = [9]
    controller._double_check_goal = [True]
    controller.cur_dis_to_goal = [1.0]
    controller._initialize_step = [7]
    controller._done_initializing = [True]
    controller._obstacle_map = [
        SimpleNamespace(_done_initializing=True)
    ]

    controller.reset_submap_local_navigation_state(
        0, initialize_new_floor=False
    )

    assert controller._carrot_goal_xy[0] == []
    assert controller._last_carrot_xy[0] == []
    assert controller._last_carrot_px[0] == []
    assert controller._stair_frontier[0] is None
    assert controller._temp_stair_map[0] == []
    assert controller._frontier_stick_step[0] == 0
    assert controller._get_close_to_stair_step[0] == 0
    assert controller._double_check_goal[0] is False
    assert np.isinf(controller.cur_dis_to_goal[0])
