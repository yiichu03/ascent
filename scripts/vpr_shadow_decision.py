"""Frozen causal acceptance simulation for labeled VPR shadow candidates."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PRIMARY_STRATUM = "vo_consistent_fragmentation"
DIAGNOSTIC_STRATUM = "vo_inconsistent_revisit"


def load_labeled(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = [json.loads(line) for line in path.read_text().splitlines()]
    if not values or values[0].get("record_type") != "vpr_shadow_gt_metadata":
        raise ValueError("missing labeled-candidate metadata")
    if values[0].get("evaluation_only_gt") is not True:
        raise ValueError("labeled candidates are not isolated evaluation data")
    rows = [
        value for value in values[1:]
        if value.get("record_type") == "vpr_shadow_candidate"
    ]
    return values[0], rows


def simulate_acceptance(
    rows: Sequence[Mapping[str, Any]],
    *,
    retrieval_threshold: float,
    deployment_stratum: str,
) -> dict[str, Any]:
    by_event: dict[
        tuple[str, int, str], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        by_event[
            (
                str(row["logical_case_id"]),
                int(row["event_index"]),
                str(row["event_id"]),
            )
        ].append(row)
    resolved_submaps: set[tuple[str, str]] = set()
    accepted: list[dict[str, Any]] = []
    for _, event_rows in sorted(by_event.items()):
        exemplar = event_rows[0]
        query_key = (
            str(exemplar["logical_case_id"]),
            str(exemplar["query_submap_id"]),
        )
        if query_key in resolved_submaps:
            continue
        eligible = [
            row
            for row in event_rows
            if row.get("geometry_pass") is True
            and row.get("vo_stratum") == deployment_stratum
            and float(row["retrieval_score"]) >= float(retrieval_threshold)
        ]
        if not eligible:
            continue
        selected = min(
            eligible,
            key=lambda row: (
                -float(row["retrieval_score"]),
                int(row["retrieval_rank"]),
                str(row["candidate_submap_id"]),
            ),
        )
        accepted.append(dict(selected))
        resolved_submaps.add(query_key)
    true_rows = [row for row in accepted if row["gt_same_place_label"]]
    false_rows = [row for row in accepted if not row["gt_same_place_label"]]
    precision = len(true_rows) / len(accepted) if accepted else None
    return {
        "retrieval_threshold": float(retrieval_threshold),
        "deployment_stratum": deployment_stratum,
        "accepted_event_count": len(accepted),
        "true_accepted_event_count": len(true_rows),
        "false_accepted_event_count": len(false_rows),
        "precision": precision,
        "true_accepted_episode_count": len(
            {str(row["logical_case_id"]) for row in true_rows}
        ),
        "true_accepted_scene_count": len(
            {str(row["scene_id"]) for row in true_rows}
        ),
        "cross_floor_accepted_count": sum(
            bool(row["gt_cross_floor_negative"]) for row in accepted
        ),
        "accepted": accepted,
    }


def choose_zero_false_positive_threshold(
    rows: Sequence[Mapping[str, Any]],
    *,
    deployment_stratum: str = PRIMARY_STRATUM,
) -> dict[str, Any]:
    scores = sorted(
        {
            float(row["retrieval_score"])
            for row in rows
            if row.get("geometry_pass") is True
            and row.get("vo_stratum") == deployment_stratum
        }
    )
    reject_all = math.nextafter(max(scores), math.inf) if scores else 1.000001
    thresholds = [reject_all, *scores]
    valid = []
    for threshold in thresholds:
        result = simulate_acceptance(
            rows,
            retrieval_threshold=threshold,
            deployment_stratum=deployment_stratum,
        )
        if result["false_accepted_event_count"] == 0:
            valid.append(result)
    if not valid:
        raise AssertionError("reject-all threshold must have zero false positives")
    return max(
        valid,
        key=lambda result: (
            int(result["true_accepted_episode_count"]),
            int(result["true_accepted_scene_count"]),
            int(result["true_accepted_event_count"]),
            -float(result["retrieval_threshold"]),
        ),
    )


def compact_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "accepted"}
