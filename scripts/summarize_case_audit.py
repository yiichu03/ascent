#!/usr/bin/env python3
"""Strict paired ASCENT+VO/v1.2-shadow gate for v1.4 case mining."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

from summarize_place_memory_screen import validate_oracle_diagnostics
from summarize_submap_screen import (
    load_manifest,
    mean,
    normalized_scene,
    parse_attempt,
    read_csv,
    sha256,
)


def load_original_baseline(
    path: Path, logical_order: list[str]
) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    errors = []
    rows = read_csv(path)
    output: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        logical_id = row.get("logical_case_id", "")
        if not logical_id or logical_id in output:
            errors.append(f"invalid_or_duplicate_original:{logical_id}")
            continue
        try:
            output[logical_id] = {
                "scene_id": normalized_scene(row["scene_id"]),
                "success": float(row["original_success"]),
                "spl": float(row["original_spl"]),
                "action_steps": int(row["original_action_steps"]),
                "spatial_revisit_proxy_eligible": int(
                    row["spatial_revisit_proxy_eligible"]
                ),
                "spatial_revisit_proxy_score": float(
                    row["spatial_revisit_proxy_score"]
                ),
            }
        except (KeyError, TypeError, ValueError):
            errors.append(f"invalid_original:{logical_id}")
    expected = set(logical_order)
    if expected - set(output):
        errors.append(f"original_missing:{len(expected - set(output))}")
    if set(output) - expected:
        errors.append(f"original_extra:{len(set(output) - expected)}")
    if any(value["success"] < 0.5 for value in output.values()):
        errors.append("selection_contains_original_failure")
    return output, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--expected-dataset", choices=("hm3d",), required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--forward-checkpoint-sha256", required=True)
    parser.add_argument("--turn-checkpoint-sha256", required=True)
    parser.add_argument("--original-baseline-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    chunks, logical_order, strict_errors = load_manifest(
        args.manifest, args.expected_episodes, args.expected_dataset
    )
    attempts: Dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    registry_rows = read_csv(args.registry)
    for registry_order, entry in enumerate(registry_rows):
        inventory = Path(entry["inventory"]).resolve()
        if not inventory.is_file():
            strict_errors.append(f"missing_inventory:{inventory}")
            continue
        for row in read_csv(inventory):
            if row.get("stage") != "screen":
                continue
            condition = row.get("condition", "")
            chunk_id = row.get("chunk_id", "")
            if condition not in {"B1", "B2"} or chunk_id not in chunks:
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
            oracle_path = ""
            if condition == "B2" and row.get("terminal_class") == "complete":
                oracle_path, oracle_errors = validate_oracle_diagnostics(
                    row,
                    expected_episodes=int(chunks[chunk_id]["episode_count"]),
                    dataset=args.expected_dataset,
                    expected_method="v1.4_case_audit_shadow",
                )
                strict_errors.extend(oracle_errors)
            for logical_id, episode in accepted.items():
                episode["registry_order"] = registry_order
                episode["evidence_oracle_diagnostics"] = oracle_path
                attempts[(condition, logical_id)].append(episode)

    selected: Dict[tuple[str, str], Dict[str, Any]] = {}
    for key, values in attempts.items():
        values.sort(
            key=lambda value: (
                int(value["priority"]), int(value["registry_order"])
            )
        )
        priorities = Counter(int(value["priority"]) for value in values)
        if any(count > 1 for count in priorities.values()):
            strict_errors.append(
                f"duplicate_episode_priority:{key}:{dict(priorities)}"
            )
        selected[key] = values[0]

    original: Dict[str, Dict[str, Any]] = {}
    if args.original_baseline_csv is None:
        if args.mode == "full":
            strict_errors.append("missing_original_baseline")
    else:
        original, original_errors = load_original_baseline(
            args.original_baseline_csv.resolve(), logical_order
        )
        strict_errors.extend(original_errors)

    missing = []
    episode_rows = []
    b1_b2_flips: Counter[str] = Counter()
    original_b1_flips: Counter[str] = Counter()
    original_b2_flips: Counter[str] = Counter()
    for logical_id in logical_order:
        b1 = selected.get(("B1", logical_id))
        b2 = selected.get(("B2", logical_id))
        base = original.get(logical_id)
        if b1 is None or b2 is None or (args.mode == "full" and base is None):
            missing.append(
                {
                    "logical_case_id": logical_id,
                    "b1": b1 is not None,
                    "b2": b2 is not None,
                    "original": base is not None,
                }
            )
            continue
        if b1["scene_id"] != b2["scene_id"] or (
            base is not None and base["scene_id"] != b1["scene_id"]
        ):
            strict_errors.append(f"paired_scene:{logical_id}")
            continue
        b1_success = int(b1["success"] >= 0.5)
        b2_success = int(b2["success"] >= 0.5)
        b1_b2_flips[f"{b1_success}->{b2_success}"] += 1
        if base is not None:
            base_success = int(base["success"] >= 0.5)
            original_b1_flips[f"{base_success}->{b1_success}"] += 1
            original_b2_flips[f"{base_success}->{b2_success}"] += 1
        if b2["place_rerank_changed_count"] != 0:
            strict_errors.append(f"shadow_changed_action:{logical_id}")
        if (
            b2["place_rerank_candidate_snapshot_count"]
            != b2["place_rerank_evaluated_count"]
        ):
            strict_errors.append(f"incomplete_candidate_snapshot:{logical_id}")
        episode_rows.append(
            {
                "logical_case_id": logical_id,
                "chunk_id": b1["chunk_id"],
                "scene_id": b1["scene_id"],
                "target_category": b1["target_category"],
                "original_success": None if base is None else base["success"],
                "original_spl": None if base is None else base["spl"],
                "original_action_steps": (
                    None if base is None else base["action_steps"]
                ),
                "selection_revisit_proxy": (
                    None
                    if base is None
                    else base["spatial_revisit_proxy_eligible"]
                ),
                "b1_ascent_vo_success": b1["success"],
                "b1_ascent_vo_spl": b1["spl"],
                "b1_ascent_vo_action_steps": b1["action_steps"],
                "b2_v12_shadow_success": b2["success"],
                "b2_v12_shadow_spl": b2["spl"],
                "b2_v12_shadow_action_steps": b2["action_steps"],
                "submap_split_count": b2["submap_split_count"],
                "place_association_accepted_count": b2[
                    "place_association_accepted_count"
                ],
                "place_rerank_evaluated_count": b2[
                    "place_rerank_evaluated_count"
                ],
                "counterfactual_change_count": b2[
                    "place_rerank_counterfactual_changed_count"
                ],
                "actual_change_count": b2["place_rerank_changed_count"],
                "candidate_snapshot_count": b2[
                    "place_rerank_candidate_snapshot_count"
                ],
                "search_action_cost": b2["search_action_cost"],
                "search_coverage_delta_m2": b2[
                    "search_coverage_delta_m2"
                ],
                "search_target_gain_count": b2[
                    "search_target_gain_count"
                ],
                "search_exit_gain_count": b2["search_exit_gain_count"],
                "evidence_b1_vo": b1["evidence_vo_diagnostics"],
                "evidence_b2_vo": b2["evidence_vo_diagnostics"],
                "evidence_b2_submap": b2[
                    "evidence_submap_diagnostics"
                ],
                "evidence_b2_oracle": b2[
                    "evidence_oracle_diagnostics"
                ],
            }
        )

    valid = len(episode_rows)
    technical_status = (
        "PASS"
        if not strict_errors and not missing and valid == args.expected_episodes
        else "FAIL"
    )
    summary = {
        "schema": "ascent_vo_submap_v1_4_case_audit_shadow_v1",
        "scientific_role": "case_mining_only_not_method_improvement",
        "dataset": args.expected_dataset,
        "mode": args.mode,
        "technical_status": technical_status,
        "expected_episodes": args.expected_episodes,
        "paired_valid_episodes": valid,
        "missing": missing,
        "strict_error_count": len(strict_errors),
        "strict_errors": strict_errors,
        "policy_contract": {
            "b1": "ASCENT+Zhao-VO_without_submaps",
            "b2": "frozen_v1.2_plus_identity_only_oracle_shadow_logging",
            "actual_task_memory_action_changes": 0,
            "gt_policy_output_fields": [
                "event_sequence", "reference_submap_id"
            ],
        },
        "metrics": {
            "original_ascent_sr": mean(
                row["original_success"]
                for row in episode_rows
                if row["original_success"] is not None
            ),
            "b1_ascent_vo_sr": mean(
                row["b1_ascent_vo_success"] for row in episode_rows
            ),
            "b2_v12_shadow_sr": mean(
                row["b2_v12_shadow_success"] for row in episode_rows
            ),
            "original_to_b1_flips": dict(sorted(original_b1_flips.items())),
            "original_to_b2_flips": dict(sorted(original_b2_flips.items())),
            "b1_to_b2_flips": dict(sorted(b1_b2_flips.items())),
        },
        "mechanism": {
            "association_exposed_episodes": sum(
                row["place_association_accepted_count"] > 0
                for row in episode_rows
            ),
            "association_accepted_count": sum(
                row["place_association_accepted_count"]
                for row in episode_rows
            ),
            "rerank_evaluated_count": sum(
                row["place_rerank_evaluated_count"]
                for row in episode_rows
            ),
            "candidate_snapshot_count": sum(
                row["candidate_snapshot_count"] for row in episode_rows
            ),
            "counterfactual_change_count": sum(
                row["counterfactual_change_count"] for row in episode_rows
            ),
            "actual_change_count": sum(
                row["actual_change_count"] for row in episode_rows
            ),
            "original_success_b1_fail_b2_fail": sum(
                row["original_success"] is not None
                and row["original_success"] >= 0.5
                and row["b1_ascent_vo_success"] < 0.5
                and row["b2_v12_shadow_success"] < 0.5
                for row in episode_rows
            ),
            "original_success_b1_success_b2_fail": sum(
                row["original_success"] is not None
                and row["original_success"] >= 0.5
                and row["b1_ascent_vo_success"] >= 0.5
                and row["b2_v12_shadow_success"] < 0.5
                for row in episode_rows
            ),
        },
        "provenance": {
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "registry": str(args.registry.resolve()),
            "registry_sha256": sha256(args.registry),
            "source_commit": args.source_commit,
            "original_baseline_csv": (
                None
                if args.original_baseline_csv is None
                else str(args.original_baseline_csv.resolve())
            ),
            "original_baseline_sha256": (
                None
                if args.original_baseline_csv is None
                else sha256(args.original_baseline_csv)
            ),
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "summary.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    fields = list(episode_rows[0]) if episode_rows else ["logical_case_id"]
    with (output_dir / "episodes.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(episode_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if technical_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
