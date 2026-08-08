from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ascent.ascent_policy import Ascent_Policy
from ascent.submaps import (
    FrontierStatus,
    MapPayload,
    RemoteRoute,
    SubmapLifecycleConfig,
    SubmapManager,
)


def _payload(tag: str) -> MapPayload:
    return MapPayload(
        obstacle_map=SimpleNamespace(tag=f"obstacle-{tag}"),
        value_map=SimpleNamespace(tag=f"value-{tag}"),
        object_map=SimpleNamespace(tag=f"object-{tag}"),
    )


def _frontier_route() -> tuple[SubmapManager, RemoteRoute, str]:
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
    first = manager.start(
        0, [0.0, 0.0, 0.0], 0, _payload("first"), 0
    )
    manager.observe_action_endpoint(0, [1.0, 0.0, 0.0], 0, 0.0)
    second, _, _ = manager.commit_split(
        0,
        [1.0, 0.0, 0.0],
        0,
        _payload("second"),
        1,
        np.array([[0.0, 2.0]], dtype=np.float64),
    )
    frontier = manager.registry(0).eligible(first.submap_id)[0]
    manager.registry(0).mark_selected(frontier.frontier_id, 2)
    route = RemoteRoute.build(
        graph=manager.graph(0),
        route_id="env0:route0000",
        active_submap_id=second.submap_id,
        destination_submap_id=first.submap_id,
        destination_local_xy=frontier.local_xy,
        target_kind="frontier",
        candidate_key=f"frontier:{frontier.frontier_id}",
        frontier_id=frontier.frontier_id,
        selected_step=2,
    )
    return manager, route, frontier.frontier_id


def _route_policy(
    *, enabled: bool
) -> tuple[Ascent_Policy, RemoteRoute, str, list[dict[str, object]]]:
    manager, route, frontier_id = _frontier_route()
    events: list[dict[str, object]] = []
    policy = object.__new__(Ascent_Policy)
    policy._submap_manager = manager
    policy._submap_config = manager.config
    policy._submap_enabled = True
    policy._submap_route_gateway_replan_enabled = enabled
    policy._submap_remote_route = [route]
    policy._observations_cache = [
        {
            "robot_xy": np.zeros(2, dtype=np.float64),
            "submap_route_action": False,
        }
    ]
    policy._num_steps = [10]
    policy._pointnav_stop_radius = 0.9
    policy._pointnav = lambda *args, **kwargs: torch.tensor(
        [[1]], dtype=torch.int64
    )
    policy._record_submap_policy_event = (
        lambda env, event, **payload: events.append(
            {"event": event, **payload}
        )
    )
    return policy, route, frontier_id, events


def _no_frontier_obstacle() -> SimpleNamespace:
    return SimpleNamespace(
        _disabled_frontiers=set(),
        _reinitialize_flag=True,
        _floor_num_steps=100,
        _explored_up_stair=True,
        _explored_down_stair=True,
        _up_stair_frontiers=np.empty((0, 2)),
        _down_stair_frontiers=np.empty((0, 2)),
        _this_floor_explored=False,
    )


def _append_local_split(manager: SubmapManager, index: int) -> None:
    active = manager.active_bundle(0)
    next_pose = np.asarray(active.anchor_pose_world).copy()
    next_pose[0] += 1.0
    decision = manager.observe_action_endpoint(
        0, next_pose, 0, 0.0
    )
    assert decision.should_split
    manager.commit_split(
        0,
        next_pose,
        0,
        _payload(f"local-{index}"),
        20 + index,
        np.empty((0, 2), dtype=np.float64),
    )


def test_route_gateway_replan_flag_off_preserves_persistent_execution() -> None:
    policy, route, frontier_id, events = _route_policy(enabled=False)

    action = policy._execute_remote_route(
        observations=None,
        env=0,
        masks=torch.ones((1, 1), dtype=torch.bool),
    )

    assert action is not None and action.item() == 1
    assert route.cursor == 1
    assert not route.awaiting_local_replan
    assert policy._submap_manager.registry(0).get(
        frontier_id
    ).status is FrontierStatus.SELECTED
    assert not any("gateway_replan" in event["event"] for event in events)


def test_gateway_pause_keeps_route_and_does_not_attempt_frontier() -> None:
    policy, route, frontier_id, events = _route_policy(enabled=True)

    action = policy._execute_remote_route(
        observations=None,
        env=0,
        masks=torch.ones((1, 1), dtype=torch.bool),
    )

    assert action is None
    assert policy._submap_remote_route[0] is route
    assert route.cursor == 1
    assert route.awaiting_local_replan
    frontier = policy._submap_manager.registry(0).get(frontier_id)
    assert frontier.status is FrontierStatus.SELECTED
    assert frontier.attempt_count == 0
    assert [event["event"] for event in events][-1] == (
        "remote_route_gateway_replan_paused"
    )


def test_live_frontier_keeps_route_paused_and_uses_native_explore() -> None:
    policy, route, _, events = _route_policy(enabled=True)
    assert (
        policy._execute_remote_route(
            observations=None,
            env=0,
            masks=torch.ones((1, 1), dtype=torch.bool),
        )
        is None
    )
    local_frontier = np.array([2.0, 0.0], dtype=np.float64)
    obstacle = SimpleNamespace(_disabled_frontiers=set())
    policy._observations_cache[0]["frontier_sensor"] = np.array(
        [local_frontier]
    )
    policy._map_controller = SimpleNamespace(
        _obstacle_map=[obstacle],
        _value_map=[SimpleNamespace()],
        _object_map=[SimpleNamespace()],
        _obstacle_map_list=[[obstacle]],
        _value_map_list=[[SimpleNamespace()]],
        _object_map_list=[[SimpleNamespace()]],
        _cur_floor_index=[0],
        _frontier_stick_step=[0],
    )
    policy._last_frontier_distance = [0.0]
    policy.cur_frontier = [np.empty(0)]
    policy.topk = 3
    selected_goals: list[np.ndarray] = []
    policy.llm_planner = SimpleNamespace(
        _get_best_frontier_with_llm=(
            lambda *args, **kwargs: (local_frontier.copy(), 1.0)
        )
    )
    policy._pointnav = lambda observations, goal, **kwargs: (
        selected_goals.append(np.asarray(goal).copy())
        or torch.tensor([[1]], dtype=torch.int64)
    )

    policy._expose_remote_route_local_replan(0)
    action = policy._explore(
        observations=None,
        env=0,
        masks=torch.ones((1, 1), dtype=torch.bool),
    )

    assert action.item() == 1
    assert route.awaiting_local_replan
    assert policy._submap_remote_route[0] is route
    np.testing.assert_allclose(selected_goals, [local_frontier])
    assert sum(
        event["event"] == "remote_route_gateway_local_replan"
        for event in events
    ) == 1
    assert not any(
        event["event"] == "remote_route_gateway_replan_resumed"
        for event in events
    )


def test_native_no_frontier_fallback_resumes_next_route_segment() -> None:
    policy, route, _, events = _route_policy(enabled=True)
    assert (
        policy._execute_remote_route(
            observations=None,
            env=0,
            masks=torch.ones((1, 1), dtype=torch.bool),
        )
        is None
    )
    obstacle = _no_frontier_obstacle()
    policy._observations_cache[0]["frontier_sensor"] = np.empty((0, 2))
    policy._map_controller = SimpleNamespace(_obstacle_map=[obstacle])
    policy._stop_action = torch.tensor([[0]], dtype=torch.int64)
    policy._request_exhaustion_recovery = lambda env, masks: None

    action = policy._explore(
        observations=None,
        env=0,
        masks=torch.ones((1, 1), dtype=torch.bool),
    )

    assert action.item() == 1
    assert not route.awaiting_local_replan
    assert policy._submap_remote_route[0] is route
    assert any(
        event["event"] == "remote_route_gateway_replan_resumed"
        for event in events
    )


@pytest.mark.parametrize("local_split_count", [0, 1, 2])
def test_paused_route_resumes_after_local_splits(
    local_split_count: int,
) -> None:
    policy, route, frontier_id, events = _route_policy(enabled=True)
    masks = torch.ones((1, 1), dtype=torch.bool)
    assert policy._execute_remote_route(None, 0, masks) is None
    for index in range(local_split_count):
        _append_local_split(policy._submap_manager, index)
    policy._observations_cache[0]["robot_xy"] = np.zeros(2)

    action = policy._submap_remote_fallback_action(None, 0, masks)

    assert action is not None and action.item() == 1
    assert policy._submap_remote_route[0] is route
    assert route.execution_submap_id == (
        policy._submap_manager.active_bundle(0).submap_id
    )
    assert policy._submap_manager.registry(0).get(
        frontier_id
    ).status is FrontierStatus.SELECTED
    assert not any(
        event.get("outcome") == "execution_frame_disconnected"
        for event in events
    )


def test_local_target_preempts_paused_route_with_explicit_event() -> None:
    policy, route, frontier_id, events = _route_policy(enabled=True)
    assert (
        policy._execute_remote_route(
            observations=None,
            env=0,
            masks=torch.ones((1, 1), dtype=torch.bool),
        )
        is None
    )

    policy._finish_remote_route(
        0, outcome="preempted_by_local_detection"
    )

    assert policy._submap_remote_route[0] is None
    frontier = policy._submap_manager.registry(0).get(frontier_id)
    assert frontier.status is FrontierStatus.ACTIVE
    assert frontier.attempt_count == 0
    preempt = next(
        event
        for event in events
        if event["event"] == "remote_route_gateway_replan_preempted"
    )
    assert preempt["frontier_disposition"] == (
        "released_selected_unattempted"
    )


def test_eval_configs_expose_default_off_route_gateway_replan_flag() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = (
        "route_gateway_replan_enabled: "
        "${oc.env:ASCENT_SUBMAP_ROUTE_GATEWAY_REPLAN_ENABLED,false}"
    )
    for relative in (
        "experiments/eval_ascent_hm3d.yaml",
        "experiments/eval_ascent_mp3d.yaml",
    ):
        assert expected in (root / relative).read_text(encoding="utf-8")
