from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ascent.llm_planner import Ascent_LLM_Planner
from ascent.submaps.place_memory import (
    OracleSamePlaceEvent,
    PlaceConditionedResidualMemory,
    PlaceMemoryConfig,
    SearchBranchStatus,
)
from ascent.submaps.types import (
    GatewayEdge,
    MapPayload,
    SubmapBundle,
    SubmapGraph,
)
from ascent.vo.oracle_place import OraclePlaceConfig, SamePlaceOracle


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeObstacleMap:
    def __init__(self) -> None:
        self.explored_area = np.zeros((20, 20), dtype=bool)
        self.pixels_per_meter = 10.0


def _bundle(
    submap_id: str, anchor: tuple[float, float, float], floor: int = 0
) -> SubmapBundle:
    return SubmapBundle(
        submap_id=submap_id,
        floor_id=floor,
        anchor_pose_world=np.asarray(anchor, dtype=np.float64),
        creation_step=0,
        payload=MapPayload(
            obstacle_map=SimpleNamespace(),
            value_map=SimpleNamespace(),
            object_map=SimpleNamespace(),
        ),
    )


def _edge(edge_id: str, source: str, destination: str) -> GatewayEdge:
    return GatewayEdge(
        edge_id=edge_id,
        source_submap_id=source,
        destination_submap_id=destination,
        source_local_pose=np.zeros(3),
        destination_local_pose=np.zeros(3),
        creation_step=1,
        source_floor_id=0,
        destination_floor_id=0,
        relative_transform=np.eye(3),
    )


def _three_node_graph() -> tuple[SubmapGraph, SubmapBundle]:
    graph = SubmapGraph()
    first = _bundle("submap-a", (0.0, 0.0, 0.0))
    middle = _bundle("submap-b", (1.0, 0.0, 0.0))
    current = _bundle("submap-c", (0.0, 0.0, 0.0))
    for node in (first, middle, current):
        graph.add_node(node)
    graph.add_edge(_edge("edge-ab", "submap-a", "submap-b"))
    graph.add_edge(_edge("edge-bc", "submap-b", "submap-c"))
    return graph, current


def _enabled_memory() -> PlaceConditionedResidualMemory:
    return PlaceConditionedResidualMemory(
        1, PlaceMemoryConfig(enabled=True)
    )


def _finish_branch(
    memory: PlaceConditionedResidualMemory,
    obstacle: FakeObstacleMap,
    *,
    coverage_pixels: int = 0,
    target_gain: bool = False,
) -> SearchBranchStatus:
    frontiers = np.asarray([[2.0, 0.0], [4.0, 0.0]])
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=frontiers[0],
        step=5,
        obstacle_map=obstacle,
        robot_xy=np.zeros(2),
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=frontiers,
    )
    obstacle.explored_area.flat[:coverage_pixels] = True
    status = memory.observe(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        step=15,
        robot_xy=np.asarray([2.0, 0.0]),
        obstacle_map=obstacle,
        target_present=target_gain,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=frontiers,
        arrival_radius_m=0.5,
    )
    if target_gain:
        assert status is SearchBranchStatus.PRODUCTIVE
        return status
    assert status is None
    assert (
        memory.observe(
            env=0,
            target="chair",
            source_submap_id="submap-a",
            floor_id=0,
            step=16,
            robot_xy=np.asarray([2.0, 0.0]),
            obstacle_map=obstacle,
            target_present=False,
            up_stair_present=False,
            down_stair_present=False,
            frontiers=frontiers,
            arrival_radius_m=0.5,
        )
        is None
    )
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=frontiers[1],
        step=17,
        obstacle_map=obstacle,
        robot_xy=np.asarray([2.0, 0.0]),
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=frontiers,
    )
    return memory.branches(0)[0].status


def test_oracle_policy_event_is_a_sealed_identity_only_message() -> None:
    assert {field.name for field in fields(OracleSamePlaceEvent)} == {
        "event_sequence",
        "reference_submap_id",
    }
    event = OracleSamePlaceEvent(1, "submap-a")
    assert event.reference_submap_id == "submap-a"
    with pytest.raises(TypeError):
        OracleSamePlaceEvent(  # type: ignore[call-arg]
            1, "submap-a", physical_distance_m=0.2
        )

    policy_source = (REPO_ROOT / "ascent" / "ascent_policy.py").read_text()
    assert "gt_start_aligned_pose" not in policy_source
    memory_source = (
        REPO_ROOT / "ascent" / "submaps" / "place_memory.py"
    ).read_text()
    assert "from habitat" not in memory_source
    assert "import habitat" not in memory_source


def test_same_place_oracle_requires_return_excursion_and_vo_consistency() -> None:
    oracle = SamePlaceOracle(1, OraclePlaceConfig(enabled=True))
    first = oracle.observe(
        env=0,
        dataset="hm3d",
        scene_id="scene",
        episode_id="episode",
        action_step=0,
        gt_pose={"x": 0.0, "y": 0.0, "height": 0.0},
        vo_pose=[0.0, 0.0, 0.0],
        current_submap_id="submap-a",
    )
    assert first.policy_events == ()
    oracle.observe(
        env=0,
        dataset="hm3d",
        scene_id="scene",
        episode_id="episode",
        action_step=10,
        gt_pose={"x": 2.5, "y": 0.0, "height": 0.0},
        vo_pose=[1.0, 0.0, 0.0],
        current_submap_id="submap-b",
    )
    returned = oracle.observe(
        env=0,
        dataset="hm3d",
        scene_id="scene",
        episode_id="episode",
        action_step=31,
        gt_pose={"x": 0.1, "y": 0.0, "height": 0.0},
        vo_pose=[1.49, 0.0, 0.0],
        current_submap_id="submap-c",
    )
    assert returned.policy_events == (OracleSamePlaceEvent(1, "submap-a"),)
    assert returned.diagnostic["matches"][0]["policy_event_emitted"] is True

    inconsistent = SamePlaceOracle(1, OraclePlaceConfig(enabled=True))
    inconsistent.observe(
        env=0,
        dataset="hm3d",
        scene_id="scene",
        episode_id="episode",
        action_step=0,
        gt_pose={"x": 0.0, "y": 0.0, "height": 0.0},
        vo_pose=[0.0, 0.0, 0.0],
        current_submap_id="submap-a",
    )
    inconsistent.observe(
        env=0,
        dataset="hm3d",
        scene_id="scene",
        episode_id="episode",
        action_step=10,
        gt_pose={"x": 2.5, "y": 0.0, "height": 0.0},
        vo_pose=[1.0, 0.0, 0.0],
        current_submap_id="submap-b",
    )
    rejected = inconsistent.observe(
        env=0,
        dataset="hm3d",
        scene_id="scene",
        episode_id="episode",
        action_step=31,
        gt_pose={"x": 0.1, "y": 0.0, "height": 0.0},
        vo_pose=[1.5, 0.0, 0.0],
        current_submap_id="submap-c",
    )
    assert rejected.policy_events == ()
    assert rejected.diagnostic["physical_return_candidate_count"] == 1
    assert rejected.diagnostic["vo_consistent_candidate_count"] == 0


def test_graph_direct_association_is_rejected_but_nondirect_is_accepted() -> None:
    graph, _ = _three_node_graph()
    memory = _enabled_memory()
    assert not memory.accept_oracle_event(
        env=0,
        event=OracleSamePlaceEvent(1, "submap-a"),
        current_submap_id="submap-b",
        graph=graph,
        step=40,
    )
    assert memory.accept_oracle_event(
        env=0,
        event=OracleSamePlaceEvent(2, "submap-a"),
        current_submap_id="submap-c",
        graph=graph,
        step=50,
    )
    assert memory.place_for_submap(0, "submap-a") == memory.place_for_submap(
        0, "submap-c"
    )


def test_low_gain_excursion_is_provisional_but_real_information_is_productive() -> None:
    memory = _enabled_memory()
    assert _finish_branch(memory, FakeObstacleMap()) is SearchBranchStatus.UNRESOLVED
    record = memory.branches(0)[0]
    assert record.reached
    assert record.coverage_delta_m2 == 0.0
    assert record.provisional_low_gain
    assert record.qualified_excursions == 1

    coverage_memory = _enabled_memory()
    assert (
        _finish_branch(
            coverage_memory, FakeObstacleMap(), coverage_pixels=50
        )
        is SearchBranchStatus.PRODUCTIVE
    )

    target_memory = _enabled_memory()
    assert (
        _finish_branch(
            target_memory, FakeObstacleMap(), target_gain=True
        )
        is SearchBranchStatus.PRODUCTIVE
    )


def test_adjacent_one_frame_arrival_never_gains_suppression_authority() -> None:
    graph, _ = _three_node_graph()
    memory = _enabled_memory()
    obstacle = FakeObstacleMap()
    frontiers = np.asarray([[2.0, 0.0], [4.0, 0.0]])
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=frontiers[0],
        step=1,
        obstacle_map=obstacle,
        robot_xy=np.zeros(2),
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=frontiers,
    )
    assert (
        memory.observe(
            env=0,
            target="chair",
            source_submap_id="submap-a",
            floor_id=0,
            step=8,
            robot_xy=frontiers[0],
            obstacle_map=obstacle,
            target_present=False,
            up_stair_present=False,
            down_stair_present=False,
            frontiers=frontiers,
            arrival_radius_m=0.5,
        )
        is None
    )
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=frontiers[1],
        step=9,
        obstacle_map=obstacle,
        robot_xy=frontiers[0],
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=frontiers,
    )
    record = memory.branches(0)[0]
    assert record.status is SearchBranchStatus.UNRESOLVED
    assert not record.provisional_low_gain
    assert memory.accept_oracle_event(
        env=0,
        event=OracleSamePlaceEvent(1, "submap-a"),
        current_submap_id="submap-c",
        graph=graph,
        step=40,
    )
    assert record.status is SearchBranchStatus.UNRESOLVED
    assert not any(
        item["event"] == "search_branch_consumed"
        for item in memory.drain_events(0)
    )


def test_nearby_start_is_refresh_not_independent_excursion() -> None:
    memory = _enabled_memory()
    obstacle = FakeObstacleMap()
    frontiers = np.asarray([[0.2, 0.0], [2.0, 0.0]])
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=frontiers[0],
        step=1,
        obstacle_map=obstacle,
        robot_xy=np.zeros(2),
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=frontiers,
    )
    for step in (2, 3):
        memory.observe(
            env=0,
            target="chair",
            source_submap_id="submap-a",
            floor_id=0,
            step=step,
            robot_xy=frontiers[0],
            obstacle_map=obstacle,
            target_present=False,
            up_stair_present=False,
            down_stair_present=False,
            frontiers=frontiers,
            arrival_radius_m=0.5,
        )
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=frontiers[1],
        step=4,
        obstacle_map=obstacle,
        robot_xy=frontiers[0],
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=frontiers,
    )
    assert not memory.branches(0)[0].provisional_low_gain


def test_selected_new_frontier_preserves_productive_search() -> None:
    memory = _enabled_memory()
    obstacle = FakeObstacleMap()
    initial = np.asarray([[2.0, 0.0]])
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=initial[0],
        step=1,
        obstacle_map=obstacle,
        robot_xy=np.zeros(2),
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=initial,
    )
    for step in (8, 9):
        memory.observe(
            env=0,
            target="chair",
            source_submap_id="submap-a",
            floor_id=0,
            step=step,
            robot_xy=initial[0],
            obstacle_map=obstacle,
            target_present=False,
            up_stair_present=False,
            down_stair_present=False,
            frontiers=initial,
            arrival_radius_m=0.5,
        )
    memory.register_selection(
        env=0,
        target="chair",
        source_submap_id="submap-a",
        floor_id=0,
        selected_frontier=[4.0, 0.0],
        step=10,
        obstacle_map=obstacle,
        robot_xy=initial[0],
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=[[2.0, 0.0], [4.0, 0.0]],
    )
    record = memory.branches(0)[0]
    assert record.status is SearchBranchStatus.PRODUCTIVE
    assert record.exit_gain
    assert not record.provisional_low_gain


def test_consumed_top1_reranks_to_highest_live_residual_alternative() -> None:
    graph, current = _three_node_graph()
    memory = _enabled_memory()
    assert _finish_branch(memory, FakeObstacleMap()) is SearchBranchStatus.UNRESOLVED
    assert memory.accept_oracle_event(
        env=0,
        event=OracleSamePlaceEvent(1, "submap-a"),
        current_submap_id="submap-c",
        graph=graph,
        step=40,
    )
    assert memory.branches(0)[0].status is SearchBranchStatus.CONSUMED
    consumed_event = next(
        item
        for item in memory.drain_events(0)
        if item["event"] == "search_branch_consumed"
    )
    assert consumed_event["last_arrival_observations"] >= 2
    assert consumed_event["last_start_robot_distance_m"] >= 1.4
    assert consumed_event["reason"] == (
        "oracle_confirmed_independent_place_revisit"
    )
    decision = memory.arbitrate(
        env=0,
        target="chair",
        active=current,
        graph=graph,
        base_frontier=[2.0, 0.0],
        base_value=0.9,
        sorted_frontiers=[[2.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        sorted_values=[0.9, 0.8, 0.7],
        topk=3,
        step=45,
    )
    assert decision.changed
    assert decision.reason == "consumed_to_residual"
    np.testing.assert_allclose(decision.final_frontier, [4.0, 0.0])

    no_alternative = memory.arbitrate(
        env=0,
        target="chair",
        active=current,
        graph=graph,
        base_frontier=[2.0, 0.0],
        base_value=0.9,
        sorted_frontiers=[[2.0, 0.0]],
        sorted_values=[0.9],
        topk=3,
        step=46,
    )
    assert not no_alternative.changed
    assert no_alternative.reason == "no_live_residual_alternative"


def _planner() -> Ascent_LLM_Planner:
    planner = object.__new__(Ascent_LLM_Planner)
    planner._last_value = [float("-inf")]
    planner._last_frontier = [np.zeros(2)]
    planner._force_frontier = [np.zeros(2)]
    planner._sort_frontiers_by_value = lambda *args: (
        np.asarray([[1.0, 0.0], [2.0, 0.0]]),
        [0.9, 0.8],
    )
    planner._try_force_frontier = lambda *args: (None, None)
    planner._try_nearby_frontier = lambda *args: (None, None, False)
    planner._decide_frontier_with_llm = lambda *args: (
        np.asarray([1.0, 0.0]),
        0.9,
    )
    planner._handle_frontier_stick_and_disable = lambda *args: None
    return planner


def test_planner_arbitration_is_post_rank_and_rejects_nonlive_goals() -> None:
    planner = _planner()
    obstacle = SimpleNamespace(_finish_first_explore=False)
    result, value = planner._get_best_frontier_with_llm(
        [{"robot_xy": np.zeros(2)}],
        [obstacle],
        [SimpleNamespace()],
        [SimpleNamespace()],
        [[]],
        [[]],
        [[]],
        np.asarray([[1.0, 0.0], [2.0, 0.0]]),
        candidate_arbitrator=lambda base, base_value, points, values: (
            points[1],
            values[1],
        ),
    )
    np.testing.assert_allclose(result, [2.0, 0.0])
    assert value == pytest.approx(0.8)

    planner = _planner()
    obstacle = SimpleNamespace(_finish_first_explore=False)
    with pytest.raises(RuntimeError, match="non-live"):
        planner._get_best_frontier_with_llm(
            [{"robot_xy": np.zeros(2)}],
            [obstacle],
            [SimpleNamespace()],
            [SimpleNamespace()],
            [[]],
            [[]],
            [[]],
            np.asarray([[1.0, 0.0], [2.0, 0.0]]),
            candidate_arbitrator=lambda *args: (np.asarray([9.0, 9.0]), 1.0),
        )
