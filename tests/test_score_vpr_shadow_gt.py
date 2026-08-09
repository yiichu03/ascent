from __future__ import annotations

import numpy as np

from scripts.score_vpr_shadow_gt import _candidate_label
from scripts.vpr_shadow_data import EpisodeGraph


def _graph() -> EpisodeGraph:
    return EpisodeGraph(
        direct_edges=set(),
        submap_first_step={"old": 1, "current": 70},
        submap_last_step={"old": 20, "current": 90},
        submap_by_step={
            **{step: "old" for step in range(1, 21)},
            **{step: "current" for step in range(70, 91)},
        },
        floor_by_submap={"old": 0, "current": 0},
    )


def test_same_place_label_requires_departure_before_return() -> None:
    track = {}
    for step in range(1, 101):
        if step < 30:
            x = 0.0
        elif step < 60:
            x = 3.0
        else:
            x = max(0.4, 3.0 - 0.13 * (step - 60))
        track[step] = np.array([x, 0.0, 0.0])
    result = _candidate_label(
        query_steps=[78, 80],
        candidate_submap_id="old",
        graph=_graph(),
        track=track,
    )
    assert result["gt_same_place_label"] is True
    assert result["gt_cross_floor_negative"] is False
    assert result["gt_representative_historical_step"] is not None

    stationary = {step: np.zeros(3) for step in range(1, 101)}
    assert _candidate_label(
        query_steps=[80],
        candidate_submap_id="old",
        graph=_graph(),
        track=stationary,
    )["gt_same_place_label"] is False


def test_height_separated_candidate_is_cross_floor_negative() -> None:
    track = {
        **{step: np.array([0.0, 0.0, 0.0]) for step in range(1, 21)},
        **{step: np.array([3.0, 0.0, 1.5]) for step in range(21, 70)},
        **{step: np.array([0.0, 0.0, 1.5]) for step in range(70, 91)},
    }
    result = _candidate_label(
        query_steps=[80],
        candidate_submap_id="old",
        graph=_graph(),
        track=track,
    )
    assert result["gt_same_place_label"] is False
    assert result["gt_cross_floor_negative"] is True
