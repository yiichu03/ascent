#!/usr/bin/env python3
"""Apply a frozen Cal50 threshold once to a locked VPR shadow split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from vpr_shadow_data import read_jsonl, sha256
    from vpr_shadow_decision import (
        DIAGNOSTIC_STRATUM,
        compact_metrics,
        load_labeled,
        simulate_acceptance,
    )
except ImportError:
    from scripts.vpr_shadow_data import read_jsonl, sha256
    from scripts.vpr_shadow_decision import (
        DIAGNOSTIC_STRATUM,
        compact_metrics,
        load_labeled,
        simulate_acceptance,
    )


MIN_TEST_PRECISION = 0.95
MIN_TEST_TRUE_EPISODES = 10
MIN_TEST_TRUE_SCENES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-file", type=Path, required=True)
    parser.add_argument("--labeled-candidates", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold = json.loads(args.threshold_file.read_text(encoding="utf-8"))
    if threshold.get("schema") != "ascent_v1_4_vpr_shadow_calibration_v1":
        raise ValueError("unexpected calibration schema")
    if threshold.get("calibration_gate") != "PASS":
        raise ValueError("calibration gate did not unlock locked tests")
    metadata, rows = load_labeled(args.labeled_candidates)
    if metadata["retriever"] != threshold["retriever"]:
        raise ValueError("retriever differs from frozen calibration")
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split.get("role") not in {
        "locked_in_distribution_test",
        "locked_cross_dataset_transfer_test",
    }:
        raise ValueError("evaluation requires a locked test role")
    if metadata.get("split_manifest_sha256") != sha256(args.split_manifest):
        raise ValueError("labeled candidates do not match locked split")
    primary = simulate_acceptance(
        rows,
        retrieval_threshold=float(threshold["retrieval_threshold"]),
        deployment_stratum=str(threshold["deployment_stratum"]),
    )
    precision = primary["precision"]
    gate_pass = (
        precision is not None
        and precision >= MIN_TEST_PRECISION
        and primary["true_accepted_episode_count"] >= MIN_TEST_TRUE_EPISODES
        and primary["true_accepted_scene_count"] >= MIN_TEST_TRUE_SCENES
        and primary["cross_floor_accepted_count"] == 0
    )
    diagnostic = simulate_acceptance(
        rows,
        retrieval_threshold=float(threshold["retrieval_threshold"]),
        deployment_stratum=DIAGNOSTIC_STRATUM,
    )
    events = [
        row for row in read_jsonl(args.events)
        if row.get("record_type") == "vpr_shadow_gt_event"
    ]
    exposed = [row for row in events if row["gt_revisit_exposure"]]
    payload = {
        "schema": "ascent_v1_4_vpr_shadow_locked_evaluation_v1",
        "technical_status": "PASS",
        "association_gate": "PASS" if gate_pass else "NO_GO",
        "dataset": split["dataset"],
        "role": split["role"],
        "retriever": threshold["retriever"],
        "threshold_file_sha256": sha256(args.threshold_file),
        "frozen_retrieval_threshold": threshold["retrieval_threshold"],
        "primary": compact_metrics(primary),
        "vo_inconsistent_diagnostic_only": compact_metrics(diagnostic),
        "proposal": {
            "gt_exposure_event_count": len(exposed),
            "recall_at_1": (
                sum(row["best_proposed_true_rank"] == 1 for row in exposed)
                / len(exposed) if exposed else None
            ),
            "recall_at_5": (
                sum(row["best_proposed_true_rank"] is not None for row in exposed)
                / len(exposed) if exposed else None
            ),
        },
        "gate_contract": {
            "min_precision": MIN_TEST_PRECISION,
            "min_true_accepted_episodes": MIN_TEST_TRUE_EPISODES,
            "min_true_accepted_scenes": MIN_TEST_TRUE_SCENES,
            "max_cross_floor_accepted": 0,
        },
        "labeled_candidates_sha256": sha256(args.labeled_candidates),
        "events_sha256": sha256(args.events),
        "split_manifest_sha256": sha256(args.split_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**payload, "evaluation_file_sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
