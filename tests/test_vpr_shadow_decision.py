from __future__ import annotations

from scripts.vpr_shadow_decision import (
    PRIMARY_STRATUM,
    choose_zero_false_positive_threshold,
    simulate_acceptance,
)


def _row(
    *,
    event: int,
    score: float,
    label: bool,
    episode: str,
    scene: str,
    submap: str,
) -> dict:
    return {
        "event_index": event,
        "event_id": f"event-{event}",
        "geometry_pass": True,
        "vo_stratum": PRIMARY_STRATUM,
        "retrieval_score": score,
        "retrieval_rank": 1,
        "candidate_submap_id": f"old-{event}",
        "query_submap_id": submap,
        "logical_case_id": episode,
        "scene_id": scene,
        "gt_same_place_label": label,
        "gt_cross_floor_negative": False,
    }


def test_zero_fp_threshold_prefers_coverage_without_crossing_false_score() -> None:
    rows = [
        _row(event=0, score=0.90, label=True, episode="e0", scene="a", submap="s2"),
        _row(event=1, score=0.85, label=False, episode="e1", scene="b", submap="s2"),
        _row(event=2, score=0.80, label=True, episode="e2", scene="c", submap="s3"),
    ]
    result = choose_zero_false_positive_threshold(rows)
    assert result["retrieval_threshold"] == 0.90
    assert result["true_accepted_event_count"] == 1
    assert result["false_accepted_event_count"] == 0


def test_first_acceptance_resolves_current_submap_causally() -> None:
    rows = [
        _row(event=0, score=0.9, label=True, episode="e0", scene="a", submap="s2"),
        _row(event=1, score=0.95, label=False, episode="e0", scene="a", submap="s2"),
    ]
    result = simulate_acceptance(
        rows, retrieval_threshold=0.5, deployment_stratum=PRIMARY_STRATUM
    )
    assert result["accepted_event_count"] == 1
    assert result["false_accepted_event_count"] == 0


def test_reused_local_event_indices_do_not_merge_across_episodes() -> None:
    rows = [
        _row(event=0, score=0.9, label=True, episode="e0", scene="a", submap="s2"),
        _row(event=0, score=0.9, label=True, episode="e1", scene="b", submap="s2"),
    ]
    result = simulate_acceptance(
        rows, retrieval_threshold=0.5, deployment_stratum=PRIMARY_STRATUM
    )
    assert result["accepted_event_count"] == 2
    assert result["true_accepted_episode_count"] == 2
