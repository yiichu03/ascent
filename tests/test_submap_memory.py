from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ascent.submaps import (
    FrontierStatus,
    MapPayload,
    SubmapLifecycleConfig,
    SubmapManager,
    SubmapQueryView,
    relative_pose,
    transform_points_xy,
)


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
