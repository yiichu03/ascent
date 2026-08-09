#!/usr/bin/env python3
"""Freeze one zero-FP HM3D-Cal50 retrieval threshold for v1.4-A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from vpr_shadow_data import sha256
    from vpr_shadow_decision import (
        PRIMARY_STRATUM,
        choose_zero_false_positive_threshold,
        compact_metrics,
        load_labeled,
    )
except ImportError:
    from scripts.vpr_shadow_data import sha256
    from scripts.vpr_shadow_decision import (
        PRIMARY_STRATUM,
        choose_zero_false_positive_threshold,
        compact_metrics,
        load_labeled,
    )


MIN_CAL_TRUE_EPISODES = 5
MIN_CAL_TRUE_SCENES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-candidates", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata, rows = load_labeled(args.labeled_candidates)
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if split.get("role") != "threshold_calibration_only":
        raise ValueError("threshold calibration requires HM3D-Cal50 role")
    if metadata.get("split_manifest_sha256") != sha256(args.split_manifest):
        raise ValueError("labeled candidates do not match calibration split")
    result = choose_zero_false_positive_threshold(
        rows, deployment_stratum=PRIMARY_STRATUM
    )
    gate_pass = (
        result["false_accepted_event_count"] == 0
        and result["true_accepted_episode_count"] >= MIN_CAL_TRUE_EPISODES
        and result["true_accepted_scene_count"] >= MIN_CAL_TRUE_SCENES
    )
    payload = {
        "schema": "ascent_v1_4_vpr_shadow_calibration_v1",
        "technical_status": "PASS",
        "calibration_gate": "PASS" if gate_pass else "NO_GO",
        "retriever": metadata["retriever"],
        "deployment_stratum": PRIMARY_STRATUM,
        "retrieval_threshold": result["retrieval_threshold"],
        "selection_rule": "zero false accepted events, then maximal true episode coverage",
        "min_cal_true_episodes": MIN_CAL_TRUE_EPISODES,
        "min_cal_true_scenes": MIN_CAL_TRUE_SCENES,
        "metrics": compact_metrics(result),
        "labeled_candidates_sha256": sha256(args.labeled_candidates),
        "split_manifest_sha256": sha256(args.split_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({**payload, "calibration_file_sha256": sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
