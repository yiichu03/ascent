from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ascent.ascent_policy import Ascent_Policy
from ascent.submaps import (
    FrontierStatus,
    MapPayload,
    SubmapLifecycleConfig,
    SubmapManager,
)


class _ConnectedMap:
    def __init__(self, safe_points: list[np.ndarray] | None = None) -> None:
        self.pixels_per_meter = 10
        self._origin = np.array([50, 50])
        self.explored_area = np.zeros((101, 101), dtype=bool)
        self._strict_navigable_map = np.zeros((101, 101), dtype=bool)
        self.frontiers = np.empty((0, 2), dtype=np.float64)
        self._disabled_frontiers: set[tuple[float, float]] = set()
        if safe_points is None:
            self.explored_area[:] = True
            self._strict_navigable_map[:] = True
        else:
            points = np.asarray(safe_points, dtype=np.float64).reshape(-1, 2)
            pixels = self._xy_to_px(points)
            self.explored_area[pixels[:, 1], pixels[:, 0]] = True
            self._strict_navigable_map[pixels[:, 1], pixels[:, 0]] = True

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        pixels = (
            np.rint(np.asarray(points)[:, ::-1] * self.pixels_per_meter)
            + self._origin
        )
        pixels[:, 0] = self.explored_area.shape[0] - pixels[:, 0]
        return pixels.astype(int)


def _payload(obstacle_map: _ConnectedMap | None = None) -> MapPayload:
    return MapPayload(
        obstacle_map=obstacle_map or _ConnectedMap(),
        value_map=SimpleNamespace(),
        object_map=SimpleNamespace(),
    )


def _tentative_three_submap_manager(
    *, active_map: _ConnectedMap | None = None
) -> tuple[SubmapManager, str]:
    config = SubmapLifecycleConfig(
        enabled=True,
        frontier_evidence_maturity_enabled=True,
        min_action_endpoints=1,
        min_anchor_displacement_m=1.5,
        overlap_threshold=0.5,
        low_overlap_consecutive=1,
        gateway_frontier_resolution_radius_m=0.1,
    )
    manager = SubmapManager(1, config)
    first = manager.start(0, [0.0, 0.0, 0.0], 0, _payload(), 0)
    manager.observe_action_endpoint(0, [2.0, 0.0, 0.0], 0, 0.0)
    manager.commit_split(
        0,
        [2.0, 0.0, 0.0],
        0,
        _payload(),
        1,
        np.array([[5.0, 0.0]]),
    )
    frontier_id = f"{first.submap_id}:f0000"
    registry = manager.registry(0)
    registry.mark_selected(frontier_id, 2)
    registry.mark_remote_failure(
        frontier_id,
        submap_id=manager.active_bundle(0).submap_id,
        view_local_xy=np.zeros(2),
        step=3,
        reason="no_progress",
    )
    manager.observe_action_endpoint(0, [4.0, 0.0, 0.0], 0, 0.0)
    manager.commit_split(
        0,
        [4.0, 0.0, 0.0],
        0,
        _payload(active_map),
        4,
        np.empty((0, 2)),
    )
    return manager, frontier_id


def _review_policy(
    manager: SubmapManager, *, robot_xy: np.ndarray
) -> tuple[Ascent_Policy, list[dict[str, object]]]:
    policy = object.__new__(Ascent_Policy)
    policy._submap_manager = manager
    policy._submap_config = manager.config
    policy._submap_frontier_evidence_maturity_enabled = True
    policy._observations_cache = [{"robot_xy": robot_xy.copy()}]
    policy._num_steps = [10]
    policy._submap_remote_route = [None]
    policy._submap_route_sequence = [0]
    policy._submap_attempted_semantics = [set()]
    events: list[dict[str, object]] = []
    policy._record_submap_policy_event = (
        lambda env, event, **payload: events.append(
            {"event": event, **payload}
        )
    )
    return policy, events


def test_feature_flag_off_preserves_v1_2_attempted_transition() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True, frontier_evidence_maturity_enabled=False
        ),
    )
    record = manager.registry(0).register_submap_frontiers(
        "env0:sm0000", np.array([[1.0, 0.0]]), 1
    )[0]
    manager.registry(0).mark_selected(record.frontier_id, 2)

    status = manager.registry(0).mark_remote_failure(
        record.frontier_id,
        submap_id="env0:sm0001",
        view_local_xy=np.zeros(2),
        step=3,
        reason="route_rejected",
    )

    assert status is FrontierStatus.ATTEMPTED
    assert manager.registry(0).eligible() == []
    assert manager.registry(0).tentative() == []


def test_first_remote_failure_is_only_tentatively_suppressed() -> None:
    manager, frontier_id = _tentative_three_submap_manager()
    record = manager.registry(0).get(frontier_id)

    assert record.status is FrontierStatus.TENTATIVE_SUPPRESSION
    assert record.negative_support_submap_id == "env0:sm0001"
    np.testing.assert_allclose(
        record.negative_support_view_local_xy, np.zeros(2)
    )
    assert record.negative_support_step == 3
    assert record.negative_support_reason == "no_progress"
    assert manager.registry(0).eligible() == []


def test_route_build_rejection_is_tentative_when_enabled() -> None:
    manager = SubmapManager(
        1,
        SubmapLifecycleConfig(
            enabled=True, frontier_evidence_maturity_enabled=True
        ),
    )
    manager.start(0, [0.0, 0.0, 0.0], 0, _payload(), 0)
    record = manager.registry(0).register_submap_frontiers(
        "env0:missing", np.array([[1.0, 0.0]]), 1
    )[0]
    policy, events = _review_policy(manager, robot_xy=np.zeros(2))

    route = policy._start_remote_route(
        0,
        destination_submap_id="env0:missing",
        destination_local_xy=record.local_xy,
        target_kind="frontier",
        candidate_key=f"frontier:{record.frontier_id}",
        frontier_id=record.frontier_id,
    )

    assert route is None
    assert record.status is FrontierStatus.TENTATIVE_SUPPRESSION
    assert record.negative_support_reason == "route_rejected"
    assert [event["event"] for event in events] == [
        "frontier_evidence_tentative",
        "remote_route_rejected",
    ]


def test_independent_connected_live_denial_confirms_retirement() -> None:
    manager, frontier_id = _tentative_three_submap_manager()
    policy, events = _review_policy(manager, robot_xy=np.zeros(2))

    plan = policy._review_tentative_frontiers(0)

    record = manager.registry(0).get(frontier_id)
    assert plan is None
    assert record.status is FrontierStatus.CONFIRMED_RETIRED
    assert record.confirmed_retirement_reason == "independent_live_denial"
    assert [event["event"] for event in events] == [
        "frontier_evidence_promoted",
        "frontier_evidence_confirmed",
    ]


def test_no_frontier_fallback_allows_one_connected_reconsideration() -> None:
    manager, frontier_id = _tentative_three_submap_manager()
    policy, events = _review_policy(manager, robot_xy=np.array([-1.0, 0.0]))

    plan = policy._review_tentative_frontiers(0)
    assert plan is not None
    route = policy._start_remote_route(
        0,
        destination_submap_id=plan.source_submap_id,
        destination_local_xy=manager.registry(0).get(frontier_id).local_xy,
        target_kind="frontier",
        candidate_key=f"frontier:{frontier_id}",
        frontier_id=frontier_id,
        frontier_reconsideration=True,
    )

    record = manager.registry(0).get(frontier_id)
    assert route is not None
    assert record.status is FrontierStatus.RECONSIDERING
    assert record.reconsideration_count == 1
    assert record.reconsideration_submap_id == "env0:sm0002"
    assert any(
        event["event"] == "frontier_evidence_reconsidered"
        for event in events
    )


def test_reconsideration_failure_confirms_and_cannot_loop() -> None:
    manager, frontier_id = _tentative_three_submap_manager()
    policy, _ = _review_policy(manager, robot_xy=np.array([-1.0, 0.0]))
    plan = policy._review_tentative_frontiers(0)
    assert plan is not None
    policy._start_remote_route(
        0,
        destination_submap_id=plan.source_submap_id,
        destination_local_xy=manager.registry(0).get(frontier_id).local_xy,
        target_kind="frontier",
        candidate_key=f"frontier:{frontier_id}",
        frontier_id=frontier_id,
        frontier_reconsideration=True,
    )

    policy._finish_remote_route(0, outcome="no_progress")

    record = manager.registry(0).get(frontier_id)
    assert record.status is FrontierStatus.CONFIRMED_RETIRED
    assert record.reconsideration_count == 1
    assert record.confirmed_retirement_reason == (
        "reconsideration_failed:no_progress"
    )
    assert policy._review_tentative_frontiers(0) is None
    with pytest.raises(RuntimeError, match="Cannot reconsider"):
        manager.registry(0).mark_reconsidering(
            frontier_id, submap_id="env0:sm0003", step=20
        )


def test_disconnected_execution_waypoint_blocks_reconsideration() -> None:
    active_map = _ConnectedMap(safe_points=[np.array([2.0, 0.0])])
    manager, frontier_id = _tentative_three_submap_manager(
        active_map=active_map
    )
    policy, events = _review_policy(manager, robot_xy=np.array([2.0, 0.0]))

    plan = policy._review_tentative_frontiers(0)

    record = manager.registry(0).get(frontier_id)
    assert plan is None
    assert record.status is FrontierStatus.TENTATIVE_SUPPRESSION
    assert record.reconsideration_count == 0
    assert events[-1]["event"] == (
        "frontier_evidence_reconsideration_skipped"
    )
    assert events[-1]["reason"] == "execution_waypoint_disconnected"
