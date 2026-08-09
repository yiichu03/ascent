#!/usr/bin/env python3
"""Close the pre-registered association-only Go/No-Go across both retrievers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from vpr_shadow_data import sha256
except ImportError:
    from scripts.vpr_shadow_data import sha256


RETRIEVERS = ("mixvpr", "megaloc")
LOCKED_ROLES = {
    "locked_in_distribution_test",
    "locked_cross_dataset_transfer_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        action="append",
        required=True,
        help="retriever=/absolute/path/to/calibration.json",
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        default=[],
        help="retriever=/absolute/path/to/evaluation.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _assignments(values: list[str]) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        name, separator, raw_path = value.partition("=")
        if separator != "=" or name not in RETRIEVERS or not raw_path:
            raise ValueError(f"invalid retriever=path assignment: {value}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"missing gate input: {path}")
        output[name].append(path)
    return dict(output)


def main() -> int:
    args = parse_args()
    calibrations = _assignments(args.calibration)
    evaluations = _assignments(args.evaluation)
    if set(calibrations) != set(RETRIEVERS) or any(
        len(paths) != 1 for paths in calibrations.values()
    ):
        raise ValueError("exactly one calibration is required for each retriever")

    records: dict[str, dict[str, Any]] = {}
    passing = []
    for retriever in RETRIEVERS:
        calibration_path = calibrations[retriever][0]
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if (
            calibration.get("schema")
            != "ascent_v1_4_vpr_shadow_calibration_v1"
            or calibration.get("technical_status") != "PASS"
            or calibration.get("retriever") != retriever
        ):
            raise ValueError(f"invalid calibration for {retriever}")
        calibration_gate = str(calibration.get("calibration_gate"))
        eval_paths = evaluations.get(retriever, [])
        if calibration_gate == "PASS":
            if len(eval_paths) != 2:
                raise ValueError(
                    f"Cal-PASS retriever {retriever} requires both locked evaluations"
                )
        elif calibration_gate == "NO_GO":
            if eval_paths:
                raise ValueError(
                    f"Cal-NO_GO retriever {retriever} cannot have locked evaluations"
                )
        else:
            raise ValueError(f"unknown calibration verdict for {retriever}")

        role_results: dict[str, dict[str, Any]] = {}
        for path in eval_paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            role = str(value.get("role"))
            if (
                value.get("schema")
                != "ascent_v1_4_vpr_shadow_locked_evaluation_v1"
                or value.get("technical_status") != "PASS"
                or value.get("retriever") != retriever
                or role not in LOCKED_ROLES
                or value.get("threshold_file_sha256") != sha256(calibration_path)
                or role in role_results
            ):
                raise ValueError(f"invalid locked evaluation for {retriever}: {path}")
            role_results[role] = {
                "association_gate": value["association_gate"],
                "dataset": value["dataset"],
                "primary": value["primary"],
                "evaluation_sha256": sha256(path),
            }
        if calibration_gate == "PASS" and set(role_results) != LOCKED_ROLES:
            raise ValueError(f"locked role coverage is incomplete for {retriever}")
        method_pass = calibration_gate == "PASS" and all(
            value["association_gate"] == "PASS"
            for value in role_results.values()
        )
        record = {
            "calibration_gate": calibration_gate,
            "calibration_sha256": sha256(calibration_path),
            "calibration_metrics": calibration["metrics"],
            "locked_evaluations": role_results,
            "association_method_gate": "PASS" if method_pass else "NO_GO",
        }
        records[retriever] = record
        if method_pass:
            precisions = [
                float(value["primary"]["precision"])
                for value in role_results.values()
            ]
            false_count = sum(
                int(value["primary"]["false_accepted_event_count"])
                for value in role_results.values()
            )
            true_episodes = sum(
                int(value["primary"]["true_accepted_episode_count"])
                for value in role_results.values()
            )
            passing.append(
                {
                    "retriever": retriever,
                    "false_accepted_event_count": false_count,
                    "minimum_locked_precision": min(precisions),
                    "total_true_accepted_episode_count": true_episodes,
                }
            )

    passing.sort(
        key=lambda value: (
            int(value["false_accepted_event_count"]),
            -float(value["minimum_locked_precision"]),
            -int(value["total_true_accepted_episode_count"]),
            RETRIEVERS.index(str(value["retriever"])),
        )
    )
    payload = {
        "schema": "ascent_v1_4_vpr_shadow_association_method_gate_v1",
        "technical_status": "PASS",
        "association_go_no_go": "GO" if passing else "NO_GO",
        "passing_retrievers": [value["retriever"] for value in passing],
        "recommended_retriever": (
            passing[0]["retriever"] if passing else None
        ),
        "recommendation_rule": (
            "fewest locked false accepts, highest minimum locked precision, "
            "largest total true-episode coverage, then fixed MixVPR/MegaLoc order"
        ),
        "retrievers": records,
        "claim_boundary": (
            "association-only gate; no PlaceMemory, planner action, SR, or SPL claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**payload, "output_sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
