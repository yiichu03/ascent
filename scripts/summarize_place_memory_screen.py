#!/usr/bin/env python3
"""Strict v1.4-only place-memory-v1.4 gate against frozen v1.2 per-episode results."""

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
    read_jsonl,
    sha256,
)


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
        logical_id = row.get("logical_case_id", "")
        if not logical_id:
            continue
        if logical_id in baseline:
            errors.append(f"duplicate_baseline:{logical_id}")
            continue
        try:
            baseline[logical_id] = {
                "logical_case_id": logical_id,
                "scene_id": normalized_scene(row["scene_id"]),
                "success": float(row["b2_success"]),
                "spl": float(row["b2_spl"]),
                "action_steps": int(row["b2_action_steps"]),
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


def validate_oracle_diagnostics(
    row: Dict[str, str], *, expected_episodes: int, dataset: str
) -> tuple[str, list[str]]:
    path_text = row.get("oracle_diagnostics", "")
    path = Path(path_text).resolve()
    records, errors = read_jsonl(path)
    prefix = f"oracle:{row.get('chunk_id', '')}"
    if errors:
        return str(path), [f"{prefix}:{item}" for item in errors]
    known = {
        "oracle_place_run_metadata",
        "oracle_place_step",
        "oracle_place_episode_end",
    }
    unknown = sorted({item.get("record_type") for item in records} - known)
    if unknown:
        errors.append(f"{prefix}:unknown:{unknown}")
    metadata = [
        item
        for item in records
        if item.get("record_type") == "oracle_place_run_metadata"
    ]
    if len(metadata) != 1:
        errors.append(f"{prefix}:metadata_count:{len(metadata)}")
    else:
        value = metadata[0]
        expected = {
            "method": "v1.4_oracle_task_memory",
            "dataset": dataset,
            "gt_policy_isolation": True,
            "policy_event_fields": [
                "event_sequence",
                "reference_submap_id",
            ],
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                errors.append(f"{prefix}:metadata:{key}:{value.get(key)!r}")
        oracle_config = value.get("oracle_config", {})
        exact_config = {
            "enabled": True,
            "physical_planar_radius_m": 0.75,
            "physical_height_radius_m": 0.75,
            "min_step_separation": 30,
            "min_excursion_m": 2.0,
            "vo_consistent_radius_m": 1.5,
        }
        for key, expected_value in exact_config.items():
            if oracle_config.get(key) != expected_value:
                errors.append(
                    f"{prefix}:oracle_config:{key}:"
                    f"{oracle_config.get(key)!r}"
                )
    ends = sum(
        item.get("record_type") == "oracle_place_episode_end"
        for item in records
    )
    if ends != expected_episodes:
        errors.append(f"{prefix}:episode_end_count:{ends}:{expected_episodes}")
    return str(path), errors


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

    chunks, logical_order, strict_errors = load_manifest(
        args.manifest, args.expected_episodes, args.expected_dataset
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
            )
            strict_errors.extend(errors)
            oracle_path = str(
                Path(row.get("oracle_diagnostics", "")).resolve()
            )
            if row.get("terminal_class") == "complete":
                oracle_path, oracle_errors = validate_oracle_diagnostics(
                    row,
                    expected_episodes=int(
                        chunks[chunk_id]["episode_count"]
                    ),
                    dataset=args.expected_dataset,
                )
                strict_errors.extend(oracle_errors)
            for logical_id, episode in accepted.items():
                episode["registry_order"] = registry_order
                episode["evidence_oracle_diagnostics"] = oracle_path
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

    missing_v14 = [
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
        v14 = selected.get(logical_id)
        if v14 is None:
            continue
        base = baseline.get(logical_id)
        if base is not None and base["scene_id"] != v14["scene_id"]:
            strict_errors.append(f"baseline_scene:{logical_id}")
            continue
        base_success = None if base is None else base["success"]
        if base_success is not None:
            flips[
                f"{int(base_success >= 0.5)}->{int(v14['success'] >= 0.5)}"
            ] += 1
        episode_rows.append(
            {
                "logical_case_id": logical_id,
                "chunk_id": v14["chunk_id"],
                "scene_id": v14["scene_id"],
                "target_category": v14["target_category"],
                "v12_success": base_success,
                "v14_success": v14["success"],
                "success_delta": (
                    None
                    if base is None
                    else v14["success"] - base["success"]
                ),
                "v12_spl": None if base is None else base["spl"],
                "v14_spl": v14["spl"],
                "spl_delta": (
                    None if base is None else v14["spl"] - base["spl"]
                ),
                "v12_action_steps": (
                    None if base is None else base["action_steps"]
                ),
                "v14_action_steps": v14["action_steps"],
                "submap_split_count": v14["submap_split_count"],
                "handoff_created_count": v14["handoff_created_count"],
                "handoff_action_count": v14["handoff_action_count"],
                "handoff_completed_count": v14[
                    "handoff_completed_count"
                ],
                "handoff_cancelled_count": v14[
                    "handoff_cancelled_count"
                ],
                "handoff_skipped_count": v14["handoff_skipped_count"],
                "exhaustion_recovery_count": v14[
                    "exhaustion_recovery_count"
                ],
                "exhaustion_recovery_scan_turn_count": v14[
                    "exhaustion_recovery_scan_turn_count"
                ],
                "exhaustion_recovery_scan_completed_count": v14[
                    "exhaustion_recovery_scan_completed_count"
                ],
                "exhaustion_recovery_target_reacquired_count": v14[
                    "exhaustion_recovery_target_reacquired_count"
                ],
                "remote_route_selected_count": v14[
                    "remote_route_selected_count"
                ],
                "remote_route_waypoint_action_count": v14[
                    "remote_route_waypoint_action_count"
                ],
                "remote_route_waypoint_reached_count": v14[
                    "remote_route_waypoint_reached_count"
                ],
                "remote_route_finished_count": v14[
                    "remote_route_finished_count"
                ],
                "place_association_accepted_count": v14[
                    "place_association_accepted_count"
                ],
                "place_association_rejected_count": v14[
                    "place_association_rejected_count"
                ],
                "place_rerank_evaluated_count": v14[
                    "place_rerank_evaluated_count"
                ],
                "place_rerank_changed_count": v14[
                    "place_rerank_changed_count"
                ],
                "search_attempt_started_count": v14[
                    "search_attempt_started_count"
                ],
                "search_attempt_finished_count": v14[
                    "search_attempt_finished_count"
                ],
                "search_consumed_count": v14[
                    "search_branch_consumed_count"
                ],
                "search_revisit_available_count": v14[
                    "search_branch_revisit_available_count"
                ],
                "search_revisit_productive_count": v14[
                    "search_branch_revisit_productive_count"
                ],
                "search_revisit_inconclusive_count": v14[
                    "search_branch_revisit_inconclusive_count"
                ],
                "search_finished_consumed_count": v14[
                    "search_status_counts"
                ].get("consumed", 0),
                "search_provisional_low_gain_count": v14[
                    "search_provisional_low_gain_count"
                ],
                "search_excursion_qualified_count": v14[
                    "search_excursion_qualified_count"
                ],
                "search_rejected_short_arrival_count": v14[
                    "search_rejected_short_arrival_count"
                ],
                "search_productive_count": v14[
                    "search_status_counts"
                ].get("productive", 0),
                "search_unresolved_count": v14[
                    "search_status_counts"
                ].get("unresolved", 0),
                "search_temp_blocked_count": v14[
                    "search_status_counts"
                ].get("temp_blocked", 0),
                "search_coverage_delta_m2": v14[
                    "search_coverage_delta_m2"
                ],
                "search_action_cost": v14["search_action_cost"],
                "search_target_gain_count": v14[
                    "search_target_gain_count"
                ],
                "search_exit_gain_count": v14[
                    "search_exit_gain_count"
                ],
                "association_to_productive_count": v14[
                    "association_to_productive_count"
                ],
                "association_to_productive_action_cost": v14[
                    "association_to_productive_action_cost"
                ],
                "place_intervention_started_count": v14[
                    "place_intervention_types"
                ].get("started", 0),
                "place_intervention_continued_count": v14[
                    "place_intervention_types"
                ].get("continued", 0),
                "selected_attempt_priority": v14["priority"],
                "evidence_vo_diagnostics": v14[
                    "evidence_vo_diagnostics"
                ],
                "evidence_submap_diagnostics": v14[
                    "evidence_submap_diagnostics"
                ],
                "evidence_oracle_diagnostics": v14[
                    "evidence_oracle_diagnostics"
                ],
            }
        )

    v14_count = len(episode_rows)
    baseline_count = sum(
        row["v12_success"] is not None for row in episode_rows
    )
    technical_status = (
        "PASS"
        if (
            not strict_errors
            and not missing_v14
            and v14_count == args.expected_episodes
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
    association_count = sum(
        row["place_association_accepted_count"] for row in episode_rows
    )
    rerank_changed = sum(
        row["place_rerank_changed_count"] for row in episode_rows
    )
    if association_count:
        exposed.append("ASSOCIATION")
    if rerank_changed:
        exposed.append("RERANK")
    if handoff_created:
        exposed.append("HANDOFF")
    if recovery_count:
        exposed.append("RECOVERY")
    if route_selected:
        exposed.append("ROUTE")
    if not exposed and split_count:
        exposed.append("SPLIT_ONLY")
    mechanism_exposure = "+".join(exposed) if exposed else "NO_EXPOSURE"

    v12_sr = mean(
        row["v12_success"]
        for row in episode_rows
        if row["v12_success"] is not None
    )
    v14_sr = mean(row["v14_success"] for row in episode_rows)
    sr_delta = (
        None if v12_sr is None or v14_sr is None else v14_sr - v12_sr
    )
    if args.mode != "full" or sr_delta is None:
        performance_verdict = "ENGINEERING_ONLY"
    elif sr_delta >= 0.02:
        performance_verdict = "SCREEN_POSITIVE"
    elif sr_delta <= -0.02:
        performance_verdict = "SCREEN_NEGATIVE"
    else:
        performance_verdict = "SCREEN_NEAR_TIE"

    selected_episodes = list(selected.values())
    summary = {
        "schema": "ascent_vo_submap_v1_4_oracle_task_memory_screen_v2",
        "dataset": args.expected_dataset,
        "mode": args.mode,
        "technical_status": technical_status,
        "mechanism_exposure": mechanism_exposure,
        "performance_verdict": performance_verdict,
        "decision_threshold_note": (
            "descriptive screen: SR delta >= +0.02 positive, "
            "<= -0.02 negative, otherwise near-tie; never drives retry"
        ),
        "expected_episodes": args.expected_episodes,
        "v14_valid_episodes": v14_count,
        "historical_v12_matched_episodes": baseline_count,
        "missing_v14": missing_v14,
        "strict_error_count": len(strict_errors),
        "strict_errors": strict_errors,
        "metrics": {
            "v12_sr": v12_sr,
            "v14_sr": v14_sr,
            "sr_delta": sr_delta,
            "v12_spl": mean(
                row["v12_spl"]
                for row in episode_rows
                if row["v12_spl"] is not None
            ),
            "v14_spl": mean(row["v14_spl"] for row in episode_rows),
            "spl_delta": mean(
                row["spl_delta"]
                for row in episode_rows
                if row["spl_delta"] is not None
            ),
            "v12_mean_action_steps": mean(
                row["v12_action_steps"]
                for row in episode_rows
                if row["v12_action_steps"] is not None
            ),
            "v14_mean_action_steps": mean(
                row["v14_action_steps"] for row in episode_rows
            ),
            "success_flips": dict(sorted(flips.items())),
        },
        "mechanism": {
            "association_exposed_episodes": sum(
                row["place_association_accepted_count"] > 0
                for row in episode_rows
            ),
            "association_accepted_count": association_count,
            "association_rejected_count": sum(
                row["place_association_rejected_count"]
                for row in episode_rows
            ),
            "rerank_evaluated_episodes": sum(
                row["place_rerank_evaluated_count"] > 0
                for row in episode_rows
            ),
            "rerank_evaluated_count": sum(
                row["place_rerank_evaluated_count"]
                for row in episode_rows
            ),
            "decision_changed_episodes": sum(
                row["place_rerank_changed_count"] > 0
                for row in episode_rows
            ),
            "decision_changed_count": rerank_changed,
            "search_attempt_started_count": sum(
                row["search_attempt_started_count"] for row in episode_rows
            ),
            "search_attempt_finished_count": sum(
                row["search_attempt_finished_count"] for row in episode_rows
            ),
            "search_consumed_count": sum(
                row["search_consumed_count"] for row in episode_rows
            ),
            "search_revisit_available_count": sum(
                row["search_revisit_available_count"]
                for row in episode_rows
            ),
            "search_revisit_productive_count": sum(
                row["search_revisit_productive_count"]
                for row in episode_rows
            ),
            "search_revisit_inconclusive_count": sum(
                row["search_revisit_inconclusive_count"]
                for row in episode_rows
            ),
            "search_finished_consumed_count": sum(
                row["search_finished_consumed_count"]
                for row in episode_rows
            ),
            "search_provisional_low_gain_count": sum(
                row["search_provisional_low_gain_count"]
                for row in episode_rows
            ),
            "search_excursion_qualified_count": sum(
                row["search_excursion_qualified_count"]
                for row in episode_rows
            ),
            "search_rejected_short_arrival_count": sum(
                row["search_rejected_short_arrival_count"]
                for row in episode_rows
            ),
            "decision_started_count": sum(
                row["place_intervention_started_count"]
                for row in episode_rows
            ),
            "decision_continued_count": sum(
                row["place_intervention_continued_count"]
                for row in episode_rows
            ),
            "search_productive_count": sum(
                row["search_productive_count"] for row in episode_rows
            ),
            "search_unresolved_count": sum(
                row["search_unresolved_count"] for row in episode_rows
            ),
            "search_temp_blocked_count": sum(
                row["search_temp_blocked_count"] for row in episode_rows
            ),
            "search_coverage_delta_m2": sum(
                row["search_coverage_delta_m2"] for row in episode_rows
            ),
            "search_action_cost": sum(
                row["search_action_cost"] for row in episode_rows
            ),
            "new_coverage_per_search_action": (
                sum(row["search_coverage_delta_m2"] for row in episode_rows)
                / sum(row["search_action_cost"] for row in episode_rows)
                if sum(row["search_action_cost"] for row in episode_rows) > 0
                else None
            ),
            "target_gain_count": sum(
                row["search_target_gain_count"] for row in episode_rows
            ),
            "exit_gain_count": sum(
                row["search_exit_gain_count"] for row in episode_rows
            ),
            "association_to_productive_count": sum(
                row["association_to_productive_count"]
                for row in episode_rows
            ),
            "association_to_productive_mean_action_cost": (
                sum(
                    row["association_to_productive_action_cost"]
                    for row in episode_rows
                )
                / sum(
                    row["association_to_productive_count"]
                    for row in episode_rows
                )
                if sum(
                    row["association_to_productive_count"]
                    for row in episode_rows
                )
                > 0
                else None
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
