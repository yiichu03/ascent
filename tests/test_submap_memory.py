from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ascent.submaps import (
    FrontierStatus,
    MapPayload,
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


def payload(tag: str) -> MapPayload:
    return MapPayload(
        obstacle_map=SimpleNamespace(tag=f"obstacle-{tag}"),
        value_map=SimpleNamespace(tag=f"value-{tag}"),
        object_map=SimpleNamespace(tag=f"object-{tag}"),
    )


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


def test_motion_split_freezes_old_map_and_creates_gateway() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=2,
            min_path_length_m=0.5,
            overlap_threshold=0.1,
            low_overlap_consecutive=3,
            max_motion_budget_m=1.0,
            gateway_frontier_resolution_radius_m=0.3,
        ),
    )
    first = manager.start(0, [2.0, 1.0, 0.2], 0, payload("first"), 0)
    assert not manager.observe_action_endpoint(
        0, [2.6, 1.0, 0.2], 0, 0.9
    ).should_split
    decision = manager.observe_action_endpoint(0, [3.2, 1.0, 0.2], 0, 0.9)
    assert decision.should_split
    assert decision.reason == "motion_budget"

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
            min_path_length_m=0.5,
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
            min_path_length_m=0.0,
            max_motion_budget_m=0.1,
            gateway_frontier_resolution_radius_m=0.1,
        ),
    )
    first = manager.start(0, [0.0, 0.0, 0.0], 0, payload("first"), 0)
    manager.observe_action_endpoint(0, [1.0, 0.0, 0.0], 0, 1.0)
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
            min_path_length_m=0.0,
            max_motion_budget_m=0.1,
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


def test_revisit_creates_a_fresh_writer_and_direct_frontier_plan() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True,
            min_action_endpoints=1,
            min_path_length_m=0.0,
            max_motion_budget_m=0.1,
            gateway_frontier_resolution_radius_m=0.1,
        ),
    )
    first = manager.start(0, [0.0, 0.0, 0.0], 0, payload("first"), 0)
    manager.observe_action_endpoint(0, [1.0, 0.0, 0.0], 0, 1.0)
    second, gateway, _ = manager.commit_split(
        0,
        [1.0, 0.0, 0.0],
        0,
        payload("second"),
        1,
        np.array([[0.0, 2.0]]),
    )
    revisit, revisit_edge, _ = manager.commit_revisit(
        0,
        [1.0, 0.0, 0.0],
        0,
        payload("revisit"),
        2,
        gateway.edge_id,
        np.empty((0, 2)),
    )

    assert second.state.value == "frozen"
    assert revisit.state.value == "active"
    assert revisit.reference_submap_id == first.submap_id
    assert revisit_edge.kind == "revisit"
    plan = manager.query_view(0).best_remote_frontier_plan()
    assert plan is not None
    assert plan.execution_kind == "direct_frontier"
    assert plan.next_gateway_edge_id is None
    np.testing.assert_allclose(plan.execution_local_xy, [-1.0, 2.0])


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
        self._policy_floor_id = [0]
        self._cur_floor_index = [0]
        self._climb_stair_over = [True]
        self.reset_calls = []

    def current_map_payload(self, env):
        return self.payload

    def create_empty_map_payload(self):
        return self.replacement

    def install_map_payload(self, env, payload):
        self.payload = payload

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
        min_path_length_m=0.0,
        max_motion_budget_m=0.5,
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
            "submap_overlap_before_update": 1.0,
        }
    ]
    policy._submap_pending_revisit = [None]
    policy._submap_boundary_frames = [[]]
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
    policy._submap_remote_frontier_id = ["old"]
    policy.llm_planner = SimpleNamespace(
        _last_frontier=[np.zeros(2)],
        reset_submap_local_state=lambda env: None
    )

    policy._finish_submap_action(env=0, action_step=1)

    active = manager.active_bundle(0)
    assert old.state.value == "frozen"
    assert active.payload.identity == replacement.identity
    assert policy._map_controller.payload.identity == replacement.identity
    assert policy._pointnav_policy[0].reset_count == 1
    assert policy._policy_info[0]["submap_split_reason"] == "motion_budget"
    assert policy._policy_info[0]["submap_count"] == 2


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
