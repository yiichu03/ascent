from __future__ import annotations

from pathlib import Path

import numpy as np

import json

import pytest

from scripts.vpr_shadow_data import (
    EpisodeGraph,
    ShadowFrame,
    build_query_events,
    load_submap_graphs,
)


def _frame(step: int, submap: str, floor: int = 0) -> ShadowFrame:
    return ShadowFrame(
        capture_root=Path("/capture"),
        env=0,
        episode_sequence=0,
        action_step=step,
        admission_index=step,
        submap_id=submap,
        floor_id=floor,
        world_pose_vo=np.zeros(3),
        local_pose=np.zeros(3),
        tf_camera_to_submap=np.eye(4),
        rgb_path=Path(f"/{step}.jpg"),
        depth_path=Path(f"/{step}.png"),
        min_depth=0.5,
        max_depth=5.0,
        fx=10.0,
        fy=10.0,
    )


def test_queries_are_causal_floor_consistent_and_graph_non_direct() -> None:
    frames = [
        _frame(0, "s0"),
        _frame(10, "s0"),
        _frame(30, "s1"),
        _frame(40, "s1"),
        _frame(70, "s2"),
        _frame(80, "s2"),
        _frame(90, "s2"),
    ]
    graph = EpisodeGraph(
        direct_edges={frozenset(("s0", "s1")), frozenset(("s1", "s2"))},
        submap_first_step={"s0": 0, "s1": 30, "s2": 70},
        submap_last_step={"s0": 10, "s1": 40, "s2": 90},
        submap_by_step={},
        floor_by_submap={"s0": 0, "s1": 0, "s2": 0},
    )
    events = build_query_events(frames, {0: graph})
    assert [event.query_frame.action_step for event in events] == [80, 90]
    assert set(events[0].candidate_frames) == {"s0"}
    assert [frame.action_step for frame in events[0].query_window] == [70, 80]
    assert [frame.action_step for frame in events[1].query_window] == [70, 80, 90]


def test_wrong_policy_floor_candidate_is_not_proposed() -> None:
    frames = [
        _frame(0, "s0", floor=1),
        _frame(10, "s0", floor=1),
        _frame(50, "s1", floor=0),
        _frame(60, "s1", floor=0),
    ]
    graph = EpisodeGraph(
        direct_edges=set(),
        submap_first_step={"s0": 0, "s1": 50},
        submap_last_step={"s0": 10, "s1": 60},
        submap_by_step={},
        floor_by_submap={"s0": 1, "s1": 0},
    )
    assert build_query_events(frames, {0: graph}) == []


def test_floor_change_endpoint_keeps_source_submap_writer_floor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "submaps.jsonl"
    rows = [
        {
            "record_type": "submap_action_endpoint",
            "episode_sequence": 0,
            "action_step": 0,
            "submap_id": "s0",
            "floor_id": 2,
            "decision": {
                "floor_changed": False,
                "should_split": False,
                "reason": None,
            },
        },
        {
            "record_type": "submap_action_endpoint",
            "episode_sequence": 0,
            "action_step": 1,
            "submap_id": "s0",
            "floor_id": 3,
            "decision": {
                "floor_changed": True,
                "should_split": True,
                "reason": "floor_change",
            },
        },
        {
            "record_type": "submap_event",
            "episode_sequence": 0,
            "step": 1,
            "event": "submap_split",
            "reason": "floor_change",
            "source_submap_id": "s0",
            "destination_submap_id": "s1",
        },
        {
            "record_type": "submap_action_endpoint",
            "episode_sequence": 0,
            "action_step": 2,
            "submap_id": "s1",
            "floor_id": 3,
            "decision": {
                "floor_changed": False,
                "should_split": False,
                "reason": None,
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    graph = load_submap_graphs(path)[0]
    assert graph.floor_by_submap == {"s0": 2, "s1": 3}
    assert frozenset(("s0", "s1")) in graph.direct_edges


def test_unconfirmed_floor_change_remains_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "submaps.jsonl"
    rows = [
        {
            "record_type": "submap_action_endpoint",
            "episode_sequence": 0,
            "action_step": 0,
            "submap_id": "s0",
            "floor_id": 0,
            "decision": {
                "floor_changed": False,
                "should_split": False,
                "reason": None,
            },
        },
        {
            "record_type": "submap_action_endpoint",
            "episode_sequence": 0,
            "action_step": 1,
            "submap_id": "s0",
            "floor_id": 1,
            "decision": {
                "floor_changed": False,
                "should_split": False,
                "reason": None,
            },
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stable writer floor"):
        load_submap_graphs(path)
