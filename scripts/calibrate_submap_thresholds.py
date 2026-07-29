#!/usr/bin/env python3
"""Freeze submap v1 thresholds from disjoint train diagnostics only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records(path: Path) -> list[Dict[str, Any]]:
    output = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: non-object")
            output.append(value)
    return output


def quantile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty quantile")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def motion_threshold(
    samples: list[Dict[str, float]], tolerance: float
) -> tuple[float, Dict[str, Any]]:
    episode_bins: Dict[tuple[int, float], list[float]] = defaultdict(list)
    for sample in samples:
        lower = math.floor(sample["motion_budget_m"] * 2.0) / 2.0
        episode_bins[
            (int(sample["episode_sequence"]), lower)
        ].append(sample["registration_error_m"])
    bins: Dict[float, list[float]] = defaultdict(list)
    for (_, lower), values in episode_bins.items():
        bins[lower].append(statistics.median(values))
    eligible = []
    for lower in sorted(bins):
        values = bins[lower]
        if len(values) < 10:
            continue
        p90 = quantile(values, 0.90)
        eligible.append(
            {
                "lower_m": lower,
                "episode_support": len(values),
                "episode_median_error_p90_m": p90,
            }
        )
    crossing = next(
        (
            row
            for row in eligible
            if row["episode_median_error_p90_m"] >= tolerance
        ),
        None,
    )
    raw = 8.0 if crossing is None else float(crossing["lower_m"])
    selected = min(8.0, max(2.0, round(raw * 2.0) / 2.0))
    return selected, {
        "rule": (
            "first_0p5m_motion_bin_with_at_least_10_episodes_"
            "whose_episode_median_registration_error_p90_reaches_0p20m"
        ),
        "eligible_bins": eligible,
        "crossing_bin": crossing,
        "fallback_if_no_crossing_m": 8.0,
        "clamp_m": [2.0, 8.0],
    }


def overlap_threshold(
    samples: list[Dict[str, float]], tolerance: float
) -> tuple[float, Dict[str, Any]]:
    candidates = (0.15, 0.20, 0.25, 0.30, 0.35)
    by_episode: Dict[int, list[Dict[str, float]]] = defaultdict(list)
    for sample in samples:
        by_episode[int(sample["episode_sequence"])].append(sample)
    positives = sum(
        sample["registration_error_m"] >= tolerance for sample in samples
    )
    negatives = len(samples) - positives
    rows = []
    for threshold in candidates:
        recalls = []
        false_rates = []
        for episode_samples in by_episode.values():
            bad = [
                sample
                for sample in episode_samples
                if sample["registration_error_m"] >= tolerance
            ]
            good = [
                sample
                for sample in episode_samples
                if sample["registration_error_m"] < tolerance
            ]
            if bad:
                recalls.append(
                    sum(
                        sample["overlap"] < threshold for sample in bad
                    )
                    / len(bad)
                )
            if good:
                false_rates.append(
                    sum(
                        sample["overlap"] < threshold for sample in good
                    )
                    / len(good)
                )
        recall = mean_or_zero(recalls)
        false_rate = mean_or_zero(false_rates)
        rows.append(
            {
                "threshold": threshold,
                "macro_bad_error_recall": recall,
                "macro_good_error_false_positive_rate": false_rate,
                "positive_episode_support": len(recalls),
                "negative_episode_support": len(false_rates),
                "youden_j": recall - false_rate,
            }
        )
    supported_rows = [
        row
        for row in rows
        if row["positive_episode_support"] >= 5
        and row["negative_episode_support"] >= 5
    ]
    if supported_rows:
        selected_row = max(
            supported_rows,
            key=lambda row: (row["youden_j"], -row["threshold"]),
        )
        selected = float(selected_row["threshold"])
        fallback = False
    else:
        selected = 0.25
        selected_row = next(
            row for row in rows if row["threshold"] == selected
        )
        fallback = True
    return selected, {
        "rule": (
            "maximize_youden_j_for_low_overlap_predicting_0p20m_error_"
            "without_navigation_metrics"
        ),
        "candidate_statistics": rows,
        "selected_statistics": selected_row,
        "fallback_due_to_class_support": fallback,
        "positive_error_samples": positives,
        "negative_error_samples": negatives,
        "episode_count": len(by_episode),
    }


def mean_or_zero(values: Iterable[float]) -> float:
    clean = list(values)
    return statistics.fmean(clean) if clean else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--forward-checkpoint-sha256", required=True)
    parser.add_argument("--turn-checkpoint-sha256", required=True)
    parser.add_argument("--error-tolerance-m", type=float, default=0.20)
    parser.add_argument("--yaw-lever-arm-m", type=float, default=2.0)
    parser.add_argument("--expected-episodes", type=int, default=30)
    args = parser.parse_args()
    with args.inventory.open(newline="", encoding="utf-8") as handle:
        inventory = [
            row
            for row in csv.DictReader(handle)
            if row.get("condition") == "CAL"
        ]
    if not inventory:
        raise ValueError("inventory has no CAL attempt")
    successful = [
        row
        for row in inventory
        if row.get("terminal_class") == "complete"
    ]
    if not successful:
        raise ValueError("calibration has no complete attempt")
    successful.sort(key=lambda row: int(row["priority"]))
    selected_attempt = successful[0]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("schema")
        != "ascent_vo_submap_screen_materialized_v1"
        or manifest.get("dataset") != "hm3d"
        or manifest.get("split") != "train"
        or int(manifest.get("episode_count", -1))
        != args.expected_episodes
        or len(manifest.get("chunks", [])) != 1
    ):
        raise ValueError("calibration manifest contract mismatch")
    if (
        selected_attempt.get("chunk_id")
        != manifest["chunks"][0]["chunk_id"]
    ):
        raise ValueError("calibration attempt/manifest chunk mismatch")
    vo_path = Path(selected_attempt["vo_diagnostics"]).resolve()
    submap_path = Path(selected_attempt["submap_diagnostics"]).resolve()
    vo = records(vo_path)
    submap = records(submap_path)
    if any(row.get("record_type") == "vo_technical_error" for row in vo):
        raise ValueError("calibration contains VO technical errors")
    known_vo = {
        "run_metadata",
        "vo_step",
        "episode_end",
        "vo_technical_error",
    }
    if any(row.get("record_type") not in known_vo for row in vo):
        raise ValueError("calibration contains unknown VO records")
    vo_metadata = [
        row for row in vo if row.get("record_type") == "run_metadata"
    ]
    if len(vo_metadata) != 1:
        raise ValueError("calibration VO metadata count is not one")
    expected_vo_metadata = {
        "provider": "zhao_rgbd_2021",
        "ascent_source_commit": args.source_commit,
        "forward_checkpoint_sha256": args.forward_checkpoint_sha256,
        "turn_checkpoint_sha256": args.turn_checkpoint_sha256,
        "action_contract": "native_0.25m_or_30deg_single_pair",
        "pose_initialization": "episode_local_zero_se2",
        "gt_policy_isolation": True,
    }
    if any(
        vo_metadata[0].get(key) != value
        for key, value in expected_vo_metadata.items()
    ):
        raise ValueError("calibration VO metadata mismatch")
    known_submap = {
        "submap_run_metadata",
        "submap_episode_reset",
        "submap_action_endpoint",
        "submap_event",
    }
    if any(row.get("record_type") not in known_submap for row in submap):
        raise ValueError("calibration contains unknown submap records")
    submap_metadata = [
        row
        for row in submap
        if row.get("record_type") == "submap_run_metadata"
    ]
    if len(submap_metadata) != 1:
        raise ValueError("calibration submap metadata count is not one")
    calibration_config = submap_metadata[0].get("config", {})
    expected_submap_metadata = {
        "ascent_source_commit": args.source_commit,
        "pose_source": "zhao_rgbd_2021",
        "policy_gt_isolation": True,
    }
    if any(
        submap_metadata[0].get(key) != value
        for key, value in expected_submap_metadata.items()
    ) or any(
        calibration_config.get(key) != value
        for key, value in {
            "enabled": True,
            "provisional_thresholds": True,
            "min_action_endpoints": 10000,
            "min_path_length_m": 10000.0,
            "overlap_threshold": 0.0,
            "low_overlap_consecutive": 10000,
            "max_motion_budget_m": 10000.0,
        }.items()
    ):
        raise ValueError("calibration submap metadata mismatch")
    vo_steps = {
        (int(row["episode_id"]), int(row["action_step"])): row
        for row in vo
        if row.get("record_type") == "vo_step"
        and row.get("localization_error_available") is True
    }
    endpoints = {
        (int(row["episode_sequence"]), int(row["action_step"])): row
        for row in submap
        if row.get("record_type") == "submap_action_endpoint"
        and str(row.get("submap_id", "")).endswith(":sm0000")
    }
    episode_ends = [
        row for row in vo if row.get("record_type") == "episode_end"
    ]
    reset_sequences = [
        int(row["episode_sequence"])
        for row in submap
        if row.get("record_type") == "submap_episode_reset"
    ]
    samples: list[Dict[str, float]] = []
    for key in sorted(set(vo_steps).intersection(endpoints)):
        vo_row = vo_steps[key]
        endpoint = endpoints[key]
        translation_error = float(vo_row["translation_error"])
        yaw_error = float(vo_row["absolute_yaw_error"])
        registration_error = math.hypot(
            translation_error,
            2.0
            * args.yaw_lever_arm_m
            * math.sin(0.5 * abs(yaw_error)),
        )
        motion = float(endpoint["decision"]["motion_budget_m"])
        overlap = endpoint.get("overlap")
        if (
            not math.isfinite(translation_error)
            or not math.isfinite(yaw_error)
            or not math.isfinite(registration_error)
            or not math.isfinite(motion)
        ):
            raise ValueError("non-finite calibration value")
        sample = {
            "episode_sequence": float(key[0]),
            "action_step": float(key[1]),
            "translation_error_m": translation_error,
            "absolute_yaw_error_rad": abs(yaw_error),
            "registration_error_m": registration_error,
            "motion_budget_m": motion,
        }
        if overlap is not None:
            overlap_value = float(overlap)
            if not math.isfinite(overlap_value):
                raise ValueError("non-finite overlap")
            sample["overlap"] = overlap_value
        samples.append(sample)
    overlap_samples = [
        sample
        for sample in samples
        if "overlap" in sample
        and sample["action_step"] >= 20
        and sample["motion_budget_m"] >= 1.5
    ]
    errors = []
    if len(episode_ends) != args.expected_episodes:
        errors.append(
            f"episode_end_count:{len(episode_ends)}:{args.expected_episodes}"
        )
    if len(samples) < 100:
        errors.append(f"matched_endpoint_count:{len(samples)}")
    if len(overlap_samples) < 20:
        errors.append(f"overlap_sample_count:{len(overlap_samples)}")
    if reset_sequences != list(range(args.expected_episodes)):
        errors.append("submap_episode_reset_sequence")
    motion, motion_audit = motion_threshold(
        samples, args.error_tolerance_m
    )
    overlap, overlap_audit = overlap_threshold(
        overlap_samples, args.error_tolerance_m
    )
    output = {
        "schema": "ascent_vo_submap_calibration_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scientific_role": "train_only_representation_threshold_calibration",
        "navigation_metrics_used": False,
        "error_tolerance_m": args.error_tolerance_m,
        "yaw_lever_arm_m": args.yaw_lever_arm_m,
        "registration_error_definition": (
            "hypot(translation_error, "
            "2*yaw_lever_arm*sin(abs(yaw_error)/2))"
        ),
        "expected_episodes": args.expected_episodes,
        "completed_episodes": len(episode_ends),
        "matched_endpoint_count": len(samples),
        "overlap_sample_count": len(overlap_samples),
        "inputs": {
            "inventory": str(args.inventory.resolve()),
            "inventory_sha256": sha256(args.inventory),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "selection_path": manifest.get("selection_path"),
            "selection_sha256": manifest.get("selection_sha256"),
            "vo_diagnostics": str(vo_path),
            "vo_diagnostics_sha256": sha256(vo_path),
            "submap_diagnostics": str(submap_path),
            "submap_diagnostics_sha256": sha256(submap_path),
        },
        "config": {
            "min_action_endpoints": 20,
            "min_path_length_m": 1.5,
            "overlap_threshold": overlap,
            "low_overlap_consecutive": 3,
            "max_motion_budget_m": motion,
            "rotation_weight_m_per_rad": 0.10,
            "gateway_frontier_resolution_radius_m": 1.0,
            "gateway_reached_radius_m": 0.9,
            "provisional_thresholds": False,
        },
        "motion_calibration": motion_audit,
        "overlap_calibration": overlap_audit,
        "translation_error_m": {
            "median": statistics.median(
                sample["translation_error_m"] for sample in samples
            ),
            "p90": quantile(
                (
                    sample["translation_error_m"]
                    for sample in samples
                ),
                0.90,
            ),
            "maximum": max(
                sample["translation_error_m"] for sample in samples
            ),
        },
        "registration_error_m": {
            "median": statistics.median(
                sample["registration_error_m"] for sample in samples
            ),
            "p90": quantile(
                (
                    sample["registration_error_m"]
                    for sample in samples
                ),
                0.90,
            ),
            "maximum": max(
                sample["registration_error_m"] for sample in samples
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
