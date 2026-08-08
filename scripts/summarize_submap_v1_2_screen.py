#!/usr/bin/env python3
"""Strict B2-only submap-v1.2 gate against frozen pose-factorial arm A."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

from summarize_submap_screen import (
    load_manifest,
    mean,
    normalized_scene,
    parse_attempt,
    read_csv,
    sha256,
)


VARIANT_FEATURES = {
    "frontier-confirm": (
        "ASCENT_SUBMAP_HANDOFF_LIVE_CONFIRMATION_ENABLED",
        "handoff_live_frontier_confirmation_enabled",
    ),
    "route-leash": (
        "ASCENT_SUBMAP_ROUTE_GATEWAY_REPLAN_ENABLED",
        "route_gateway_replan_enabled",
    ),
    "evidence-maturity": (
        "ASCENT_SUBMAP_FRONTIER_EVIDENCE_MATURITY_ENABLED",
        "frontier_evidence_maturity_enabled",
    ),
}


def load_baseline(
    paths: Iterable[Path],
    *,
    dataset: str,
    logical_order: list[str],
) -> tuple[Dict[str, Dict[str, Any]], list[str], list[Dict[str, str]]]:
    rows: list[Dict[str, str]] = []
    errors = []
    for path in paths:
        if not path.is_file():
            errors.append(f"missing_baseline:{path}")
            continue
        rows.extend(read_csv(path))
    baseline: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("dataset") != dataset or row.get("arm") != "A":
            continue
        logical_id = row.get("logical_case_id", "")
        if logical_id in baseline:
            errors.append(f"duplicate_baseline:{logical_id}")
            continue
        try:
            baseline[logical_id] = {
                "logical_case_id": logical_id,
                "scene_id": normalized_scene(row["scene_id"]),
                "success": float(row["success"]),
                "spl": float(row["spl"]),
                "action_steps": int(row["action_steps"]),
            }
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_baseline:{logical_id}")
    expected = set(logical_order)
    missing = sorted(expected - set(baseline))
    extra = sorted(set(baseline) - expected)
    if missing:
        errors.append(f"baseline_missing:{len(missing)}")
    if extra:
        errors.append(f"baseline_extra:{len(extra)}")
    return baseline, errors, rows


def sum_counters(
    episodes: Iterable[Dict[str, Any]], key: str
) -> Dict[str, int]:
    result: Counter[str] = Counter()
    for episode in episodes:
        result.update(episode.get(key, {}))
    return dict(sorted(result.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--expected-dataset",
        choices=("hm3d", "mp3d"),
        required=True,
    )
    parser.add_argument(
        "--expected-split", choices=("train", "val"), default="train"
    )
    parser.add_argument(
        "--variant-name", choices=tuple(VARIANT_FEATURES), required=True
    )
    parser.add_argument("--feature-env-key", required=True)
    parser.add_argument("--feature-config-key", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--forward-checkpoint-sha256", required=True)
    parser.add_argument("--turn-checkpoint-sha256", required=True)
    parser.add_argument(
        "--baseline-episodes-csv",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_feature = VARIANT_FEATURES[args.variant_name]
    if (args.feature_env_key, args.feature_config_key) != expected_feature:
        parser.error(
            "variant feature mismatch: "
            f"{args.variant_name} requires {expected_feature!r}"
        )

    chunks, logical_order, strict_errors = load_manifest(
        args.manifest,
        args.expected_episodes,
        args.expected_dataset,
        args.expected_split,
    )
    registry_rows = read_csv(args.registry)
    attempts: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for registry_order, entry in enumerate(registry_rows):
        inventory = Path(entry["inventory"]).resolve()
        if not inventory.is_file():
            strict_errors.append(f"missing_inventory:{inventory}")
            continue
        for row in read_csv(inventory):
            if row.get("stage") != "screen":
                continue
            condition = row.get("condition")
            chunk_id = row.get("chunk_id", "")
            if condition != "B2" or chunk_id not in chunks:
                strict_errors.append(
                    f"unexpected_inventory:{condition}:{chunk_id}"
                )
                continue
            accepted, errors = parse_attempt(
                row,
                chunk=chunks[chunk_id],
                mode=args.mode,
                source_commit=args.source_commit,
                forward_sha256=args.forward_checkpoint_sha256,
                turn_sha256=args.turn_checkpoint_sha256,
                calibration=None,
                expected_feature_config_key=args.feature_config_key,
            )
            strict_errors.extend(errors)
            for logical_id, episode in accepted.items():
                episode["registry_order"] = registry_order
                attempts[logical_id].append(episode)

    selected: Dict[str, Dict[str, Any]] = {}
    for logical_id, values in attempts.items():
        values.sort(
            key=lambda item: (
                int(item["priority"]),
                int(item["registry_order"]),
            )
        )
        priorities = Counter(int(item["priority"]) for item in values)
        if any(count > 1 for count in priorities.values()):
            strict_errors.append(
                f"duplicate_episode_priority:{logical_id}:{dict(priorities)}"
            )
        selected[logical_id] = values[0]

    missing_b2 = [
        logical_id
        for logical_id in logical_order
        if logical_id not in selected
    ]
    baseline: Dict[str, Dict[str, Any]] = {}
    baseline_errors: list[str] = []
    if args.baseline_episodes_csv:
        baseline, baseline_errors, _ = load_baseline(
            args.baseline_episodes_csv,
            dataset=args.expected_dataset,
            logical_order=logical_order,
        )
        strict_errors.extend(baseline_errors)
    elif args.mode == "full":
        strict_errors.append("missing_historical_baseline")

    episode_rows = []
    flips: Counter[str] = Counter()
    for logical_id in logical_order:
        b2 = selected.get(logical_id)
        if b2 is None:
            continue
        base = baseline.get(logical_id)
        if base is not None and base["scene_id"] != b2["scene_id"]:
            strict_errors.append(f"baseline_scene:{logical_id}")
            continue
        base_success = None if base is None else base["success"]
        if base_success is not None:
            flips[
                f"{int(base_success >= 0.5)}->{int(b2['success'] >= 0.5)}"
            ] += 1
        episode_rows.append(
            {
                "logical_case_id": logical_id,
                "chunk_id": b2["chunk_id"],
                "scene_id": b2["scene_id"],
                "target_category": b2["target_category"],
                "b1_success": base_success,
                "b2_success": b2["success"],
                "success_delta": (
                    None
                    if base is None
                    else b2["success"] - base["success"]
                ),
                "b1_spl": None if base is None else base["spl"],
                "b2_spl": b2["spl"],
                "spl_delta": (
                    None if base is None else b2["spl"] - base["spl"]
                ),
                "b1_action_steps": (
                    None if base is None else base["action_steps"]
                ),
                "b2_action_steps": b2["action_steps"],
                "submap_split_count": b2["submap_split_count"],
                "handoff_created_count": b2["handoff_created_count"],
                "handoff_action_count": b2["handoff_action_count"],
                "handoff_completed_count": b2[
                    "handoff_completed_count"
                ],
                "handoff_cancelled_count": b2[
                    "handoff_cancelled_count"
                ],
                "handoff_skipped_count": b2["handoff_skipped_count"],
                "exhaustion_recovery_count": b2[
                    "exhaustion_recovery_count"
                ],
                "exhaustion_recovery_scan_turn_count": b2[
                    "exhaustion_recovery_scan_turn_count"
                ],
                "exhaustion_recovery_scan_completed_count": b2[
                    "exhaustion_recovery_scan_completed_count"
                ],
                "exhaustion_recovery_target_reacquired_count": b2[
                    "exhaustion_recovery_target_reacquired_count"
                ],
                "remote_route_selected_count": b2[
                    "remote_route_selected_count"
                ],
                "remote_route_waypoint_action_count": b2[
                    "remote_route_waypoint_action_count"
                ],
                "remote_route_waypoint_reached_count": b2[
                    "remote_route_waypoint_reached_count"
                ],
                "remote_route_finished_count": b2[
                    "remote_route_finished_count"
                ],
                "submap_event_counts_json": json.dumps(
                    b2["submap_event_counts"],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "selected_attempt_priority": b2["priority"],
                "evidence_vo_diagnostics": b2[
                    "evidence_vo_diagnostics"
                ],
                "evidence_submap_diagnostics": b2[
                    "evidence_submap_diagnostics"
                ],
            }
        )

    b2_count = len(episode_rows)
    baseline_count = sum(
        row["b1_success"] is not None for row in episode_rows
    )
    technical_status = (
        "PASS"
        if (
            not strict_errors
            and not missing_b2
            and b2_count == args.expected_episodes
            and (
                args.mode == "smoke"
                or baseline_count == args.expected_episodes
            )
        )
        else "FAIL"
    )
    route_selected = sum(
        row["remote_route_selected_count"] for row in episode_rows
    )
    split_count = sum(
        row["submap_split_count"] for row in episode_rows
    )
    handoff_created = sum(
        row["handoff_created_count"] for row in episode_rows
    )
    recovery_count = sum(
        row["exhaustion_recovery_count"] for row in episode_rows
    )
    exposed = []
    if handoff_created:
        exposed.append("HANDOFF")
    if recovery_count:
        exposed.append("RECOVERY")
    if route_selected:
        exposed.append("ROUTE")
    if not exposed and split_count:
        exposed.append("SPLIT_ONLY")
    mechanism_exposure = "+".join(exposed) if exposed else "NO_EXPOSURE"

    b1_sr = mean(
        row["b1_success"]
        for row in episode_rows
        if row["b1_success"] is not None
    )
    b2_sr = mean(row["b2_success"] for row in episode_rows)
    sr_delta = (
        None if b1_sr is None or b2_sr is None else b2_sr - b1_sr
    )
    performance_verdict = (
        "ENGINEERING_ONLY" if args.mode == "smoke" else "DESCRIPTIVE_ONLY"
    )

    selected_episodes = list(selected.values())
    summary = {
        "schema": "ascent_vo_submap_v1_2_b2_screen_v1",
        "dataset": args.expected_dataset,
        "split": args.expected_split,
        "variant_name": args.variant_name,
        "feature_env_key": args.feature_env_key,
        "feature_config_key": args.feature_config_key,
        "mode": args.mode,
        "technical_status": technical_status,
        "mechanism_exposure": mechanism_exposure,
        "performance_verdict": performance_verdict,
        "decision_threshold_note": (
            "No embedded performance threshold; SR/SPL and flips are "
            "descriptive and never drive retry."
        ),
        "expected_episodes": args.expected_episodes,
        "b2_valid_episodes": b2_count,
        "historical_b1_matched_episodes": baseline_count,
        "missing_b2": missing_b2,
        "strict_error_count": len(strict_errors),
        "strict_errors": strict_errors,
        "metrics": {
            "b1_sr": b1_sr,
            "b2_sr": b2_sr,
            "sr_delta": sr_delta,
            "b1_spl": mean(
                row["b1_spl"]
                for row in episode_rows
                if row["b1_spl"] is not None
            ),
            "b2_spl": mean(row["b2_spl"] for row in episode_rows),
            "spl_delta": mean(
                row["spl_delta"]
                for row in episode_rows
                if row["spl_delta"] is not None
            ),
            "b1_mean_action_steps": mean(
                row["b1_action_steps"]
                for row in episode_rows
                if row["b1_action_steps"] is not None
            ),
            "b2_mean_action_steps": mean(
                row["b2_action_steps"] for row in episode_rows
            ),
            "success_flips": dict(sorted(flips.items())),
        },
        "mechanism": {
            "submap_event_counts": sum_counters(
                selected_episodes, "submap_event_counts"
            ),
            "split_episodes": sum(
                row["submap_split_count"] > 0 for row in episode_rows
            ),
            "split_count": split_count,
            "handoff_created_episodes": sum(
                row["handoff_created_count"] > 0 for row in episode_rows
            ),
            "handoff_created_count": handoff_created,
            "handoff_action_count": sum(
                row["handoff_action_count"] for row in episode_rows
            ),
            "handoff_completed_count": sum(
                row["handoff_completed_count"] for row in episode_rows
            ),
            "handoff_cancelled_count": sum(
                row["handoff_cancelled_count"] for row in episode_rows
            ),
            "handoff_skipped_count": sum(
                row["handoff_skipped_count"] for row in episode_rows
            ),
            "handoff_outcomes": sum_counters(
                selected_episodes, "handoff_outcomes"
            ),
            "handoff_skip_reasons": sum_counters(
                selected_episodes, "handoff_skip_reasons"
            ),
            "exhaustion_recovery_episodes": sum(
                row["exhaustion_recovery_count"] > 0
                for row in episode_rows
            ),
            "exhaustion_recovery_count": recovery_count,
            "exhaustion_recovery_scan_turn_count": sum(
                row["exhaustion_recovery_scan_turn_count"]
                for row in episode_rows
            ),
            "exhaustion_recovery_scan_completed_count": sum(
                row["exhaustion_recovery_scan_completed_count"]
                for row in episode_rows
            ),
            "exhaustion_recovery_target_reacquired_count": sum(
                row["exhaustion_recovery_target_reacquired_count"]
                for row in episode_rows
            ),
            "exhaustion_recovery_skip_reasons": sum_counters(
                selected_episodes, "exhaustion_recovery_skip_reasons"
            ),
            "route_selected_episodes": sum(
                row["remote_route_selected_count"] > 0
                for row in episode_rows
            ),
            "route_selected_count": route_selected,
            "route_waypoint_actions": sum(
                row["remote_route_waypoint_action_count"]
                for row in episode_rows
            ),
            "route_waypoints_reached": sum(
                row["remote_route_waypoint_reached_count"]
                for row in episode_rows
            ),
            "route_finished_count": sum(
                row["remote_route_finished_count"]
                for row in episode_rows
            ),
            "route_target_kinds": sum_counters(
                selected_episodes, "remote_route_target_kinds"
            ),
            "route_outcomes": sum_counters(
                selected_episodes, "remote_route_outcomes"
            ),
            "split_reasons": sum_counters(
                selected_episodes, "submap_split_reasons"
            ),
            "legacy_revisit_count": sum(
                episode["submap_revisit_count"]
                for episode in selected_episodes
            ),
        },
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "registry": str(args.registry.resolve()),
            "registry_sha256": sha256(args.registry),
            "source_commit": args.source_commit,
            "variant_name": args.variant_name,
            "feature_env_key": args.feature_env_key,
            "feature_config_key": args.feature_config_key,
            "historical_baseline_csvs": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                }
                for path in args.baseline_episodes_csv
                if path.is_file()
            ],
            "metric_driven_retry": False,
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(
            summary, handle, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
    fields = list(episode_rows[0]) if episode_rows else [
        "logical_case_id"
    ]
    with (output_dir / "episodes.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(episode_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if technical_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
