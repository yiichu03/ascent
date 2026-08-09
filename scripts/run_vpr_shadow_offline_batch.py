#!/usr/bin/env python3
"""Run one frozen VPR retriever over a completed passive-capture role.

All raw, GT-free association units finish before the evaluation-only scorer is
started.  Locked test roles additionally require a previously frozen Cal50
PASS file; this prevents accidental inspection of test association scores
during calibration.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from vpr_shadow_data import read_jsonl, sha256
except ImportError:
    from scripts.vpr_shadow_data import read_jsonl, sha256


CAL_ROLE = "threshold_calibration_only"
LOCKED_ROLES = {
    "locked_in_distribution_test",
    "locked_cross_dataset_transfer_test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, required=True)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--capture-episodes", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--retriever", choices=("mixvpr", "megaloc"), required=True)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-keyframe-file-hashes", action="store_true")
    return parser.parse_args()


def _safe(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return (result or "unit")[:96]


def _read_capture_units(
    *,
    summary_path: Path,
    episodes_path: Path,
    split_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != "ascent_v1_4_vpr_shadow_capture_gate_v1"
        or summary.get("technical_status") != "PASS"
        or int(summary.get("strict_error_count", -1)) != 0
    ):
        raise ValueError("capture role did not pass its strict gate")
    if summary.get("provenance", {}).get("episodes_sha256") != sha256(episodes_path):
        raise ValueError("capture episode registry hash mismatch")
    if summary.get("provenance", {}).get("split_manifest_sha256") != sha256(split_path):
        raise ValueError("capture role differs from requested split")
    if (
        summary.get("dataset") != split.get("dataset")
        or summary.get("role") != split.get("role")
        or int(summary.get("valid_episodes", -1)) != int(split.get("episode_count", -2))
    ):
        raise ValueError("capture summary/split dimensions differ")

    rows = read_jsonl(episodes_path)
    expected_ids = [str(row["logical_case_id"]) for row in split["episodes"]]
    if [str(row.get("logical_case_id")) for row in rows] != expected_ids:
        raise ValueError("capture episode order differs from frozen split")
    required_paths = (
        "capture_manifest",
        "vo_diagnostics",
        "submap_diagnostics",
        "identity_path",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        paths = []
        for key in required_paths:
            path = Path(str(row.get(key, ""))).resolve()
            if not path.is_file():
                raise ValueError(f"capture evidence path is missing: {key}={path}")
            paths.append(path)
        if sha256(paths[0]) != row.get("capture_manifest_sha256"):
            raise ValueError("captured keyframe manifest hash mismatch")
        key = (
            str(row["chunk_id"]),
            int(row["priority"]),
            int(row["registry_order"]),
            *(str(path) for path in paths),
        )
        grouped[key].append(dict(row))

    order = {logical_id: index for index, logical_id in enumerate(expected_ids)}
    units = []
    for key, values in grouped.items():
        values.sort(key=lambda row: order[str(row["logical_case_id"])])
        units.append(
            {
                "chunk_id": key[0],
                "priority": key[1],
                "registry_order": key[2],
                "capture_manifest": key[3],
                "vo_diagnostics": key[4],
                "submap_diagnostics": key[5],
                "identity_path": key[6],
                "logical_case_ids": [
                    str(row["logical_case_id"]) for row in values
                ],
            }
        )
    units.sort(key=lambda unit: order[unit["logical_case_ids"][0]])
    if [case for unit in units for case in unit["logical_case_ids"]] != expected_ids:
        raise ValueError("selected capture attempts do not partition the split")
    return summary, split, units


def _check_role_unlock(
    *, split: Mapping[str, Any], retriever: str, calibration_file: Path | None
) -> str | None:
    role = str(split.get("role"))
    if role == CAL_ROLE:
        if calibration_file is not None:
            raise ValueError("Cal50 must be processed before any calibration file exists")
        return None
    if role not in LOCKED_ROLES:
        raise ValueError(f"offline association is not enabled for role {role}")
    if calibration_file is None or not calibration_file.is_file():
        raise ValueError("locked test association requires a frozen Cal50 PASS file")
    value = json.loads(calibration_file.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "ascent_v1_4_vpr_shadow_calibration_v1"
        or value.get("calibration_gate") != "PASS"
        or value.get("retriever") != retriever
    ):
        raise ValueError("calibration file does not unlock this retriever")
    return sha256(calibration_file)


def _run(command: Sequence[str], log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(
            [str(value) for value in command],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )


def merge_scored_units(
    *,
    scored_units: Sequence[Mapping[str, Any]],
    split_path: Path,
    retriever: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Strictly merge unit-local scorer outputs into one frozen split stream."""

    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_ids = [str(row["logical_case_id"]) for row in split["episodes"]]
    order = {logical_id: index for index, logical_id in enumerate(expected_ids)}
    split_hash = sha256(split_path)
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    source_records = []
    thresholds: Mapping[str, Any] | None = None
    covered: list[str] = []
    for unit in scored_units:
        score_dir = Path(str(unit["score_dir"])).resolve()
        labeled_path = score_dir / "labeled_candidates.jsonl"
        events_path = score_dir / "events.jsonl"
        summary_path = score_dir / "summary.json"
        labeled = read_jsonl(labeled_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not labeled or labeled[0].get("record_type") != "vpr_shadow_gt_metadata":
            raise ValueError(f"unit scorer metadata missing: {score_dir}")
        metadata = labeled[0]
        if (
            metadata.get("schema")
            != "ascent_v1_4_vpr_shadow_gt_labeled_candidates_v1"
            or metadata.get("evaluation_only_gt") is not True
            or metadata.get("runtime_policy_access") is not False
            or metadata.get("split_manifest_sha256") != split_hash
            or metadata.get("retriever") != retriever
            or summary.get("schema")
            != "ascent_v1_4_vpr_shadow_gt_score_summary_v1"
            or summary.get("technical_status") != "PASS"
        ):
            raise ValueError(f"unit scorer contract mismatch: {score_dir}")
        unit_ids = [str(value) for value in unit["logical_case_ids"]]
        observed_ids = [str(value) for value in summary["evaluated_logical_case_ids"]]
        if observed_ids != unit_ids or int(summary["evaluated_episode_count"]) != len(unit_ids):
            raise ValueError(f"unit scorer episode coverage mismatch: {score_dir}")
        if thresholds is None:
            thresholds = dict(metadata["thresholds"])
        elif dict(metadata["thresholds"]) != dict(thresholds):
            raise ValueError("GT scorer threshold drift across units")
        unit_candidates = [
            row for row in labeled[1:]
            if row.get("record_type") == "vpr_shadow_candidate"
        ]
        unit_events = [
            row for row in read_jsonl(events_path)
            if row.get("record_type") == "vpr_shadow_gt_event"
        ]
        if any(str(row.get("logical_case_id")) not in unit_ids for row in unit_candidates):
            raise ValueError("candidate escaped its scored capture unit")
        if any(str(row.get("logical_case_id")) not in unit_ids for row in unit_events):
            raise ValueError("event escaped its scored capture unit")
        if len(unit_candidates) != int(summary["candidate_count"]):
            raise ValueError("unit candidate count mismatch")
        if len(unit_events) != int(summary["query_event_count"]):
            raise ValueError("unit event count mismatch")
        candidates.extend(unit_candidates)
        events.extend(unit_events)
        covered.extend(unit_ids)
        source_records.append(
            {
                "chunk_id": str(unit["chunk_id"]),
                "logical_case_ids": unit_ids,
                "labeled_candidates_sha256": sha256(labeled_path),
                "events_sha256": sha256(events_path),
                "summary_sha256": sha256(summary_path),
            }
        )
    if covered != expected_ids:
        raise ValueError("merged scorer units do not cover the frozen split in order")

    candidate_keys = [
        (
            str(row["logical_case_id"]),
            str(row["event_id"]),
            str(row["candidate_submap_id"]),
        )
        for row in candidates
    ]
    event_keys = [
        (str(row["logical_case_id"]), str(row["event_id"])) for row in events
    ]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("duplicate merged VPR candidate")
    if len(event_keys) != len(set(event_keys)):
        raise ValueError("duplicate merged VPR event")
    candidates.sort(
        key=lambda row: (
            order[str(row["logical_case_id"])],
            int(row["event_index"]),
            int(row["retrieval_rank"]),
            str(row["candidate_submap_id"]),
        )
    )
    events.sort(
        key=lambda row: (
            order[str(row["logical_case_id"])],
            int(row["event_index"]),
            str(row["event_id"]),
        )
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    labeled_path = output_dir / "labeled_candidates.jsonl"
    with labeled_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "record_type": "vpr_shadow_gt_metadata",
                    "schema": "ascent_v1_4_vpr_shadow_gt_labeled_candidates_v1",
                    "evaluation_only_gt": True,
                    "runtime_policy_access": False,
                    "retriever": retriever,
                    "split_manifest_sha256": split_hash,
                    "thresholds": dict(thresholds or {}),
                    "merged_unit_count": len(scored_units),
                    "source_units": source_records,
                },
                sort_keys=True,
            )
            + "\n"
        )
        for row in candidates:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    events_path = output_dir / "events.jsonl"
    with events_path.open("x", encoding="utf-8") as stream:
        for row in events:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    exposed = [row for row in events if row["gt_revisit_exposure"]]
    summary = {
        "schema": "ascent_v1_4_vpr_shadow_merged_score_summary_v1",
        "technical_status": "PASS",
        "dataset": split["dataset"],
        "role": split["role"],
        "retriever": retriever,
        "evaluated_episode_count": len(expected_ids),
        "evaluated_scene_count": len(split["scene_ids"]),
        "unit_count": len(scored_units),
        "candidate_count": len(candidates),
        "query_event_count": len(events),
        "gt_revisit_exposure_event_count": len(exposed),
        "proposal_recall_at_1": (
            sum(row["best_proposed_true_rank"] == 1 for row in exposed)
            / len(exposed)
            if exposed
            else None
        ),
        "proposal_recall_at_5": (
            sum(row["best_proposed_true_rank"] is not None for row in exposed)
            / len(exposed)
            if exposed
            else None
        ),
        "labeled_candidates_sha256": sha256(labeled_path),
        "events_sha256": sha256(events_path),
        "split_manifest_sha256": split_hash,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    project_root = args.project_root.resolve()
    model_registry = args.model_registry.resolve()
    summary_path = args.capture_summary.resolve()
    episodes_path = args.capture_episodes.resolve()
    split_path = args.split_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    capture_summary, split, units = _read_capture_units(
        summary_path=summary_path,
        episodes_path=episodes_path,
        split_path=split_path,
    )
    calibration_hash = _check_role_unlock(
        split=split,
        retriever=args.retriever,
        calibration_file=(
            None if args.calibration_file is None else args.calibration_file.resolve()
        ),
    )
    scripts_root = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True)
    (output_dir / "raw").mkdir()
    (output_dir / "scored").mkdir()
    (output_dir / "logs").mkdir()

    unit_records = []
    # Pass 1: no VO diagnostics, GT trajectory, threshold, or calibration is read.
    for index, unit in enumerate(units):
        name = f"unit_{index:03d}_{_safe(unit['chunk_id'])}"
        raw_dir = output_dir / "raw" / name
        command = [
            sys.executable,
            str(scripts_root / "run_vpr_shadow_association.py"),
            "--project-root",
            str(project_root),
            "--registry",
            str(model_registry),
            "--capture-manifest",
            str(unit["capture_manifest"]),
            "--submap-diagnostics",
            str(unit["submap_diagnostics"]),
            "--retriever",
            args.retriever,
            "--output-dir",
            str(raw_dir),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ]
        if args.skip_keyframe_file_hashes:
            command.append("--skip-keyframe-file-hashes")
        _run(command, output_dir / "logs" / f"{name}.raw.log")
        raw_summary = json.loads((raw_dir / "summary.json").read_text())
        if raw_summary.get("technical_status") != "PASS":
            raise ValueError(f"raw association failed: {name}")
        unit_records.append({**unit, "name": name, "raw_dir": str(raw_dir)})

    # Pass 2: evaluation-only GT labels are produced after every raw unit exists.
    scored_records = []
    for unit in unit_records:
        score_dir = output_dir / "scored" / str(unit["name"])
        command = [
            sys.executable,
            str(scripts_root / "score_vpr_shadow_gt.py"),
            "--raw-candidates",
            str(Path(unit["raw_dir"]) / "raw_candidates.jsonl"),
            "--vo-diagnostics",
            str(unit["vo_diagnostics"]),
            "--submap-diagnostics",
            str(unit["submap_diagnostics"]),
            "--episode-identity",
            str(unit["identity_path"]),
            "--split-manifest",
            str(split_path),
            "--output-dir",
            str(score_dir),
        ]
        _run(command, output_dir / "logs" / f"{unit['name']}.score.log")
        scored_records.append({**unit, "score_dir": str(score_dir)})

    merged = merge_scored_units(
        scored_units=scored_records,
        split_path=split_path,
        retriever=args.retriever,
        output_dir=output_dir / "merged",
    )
    units_path = output_dir / "units.json"
    with units_path.open("x", encoding="utf-8") as handle:
        json.dump(scored_records, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary = {
        "schema": "ascent_v1_4_vpr_shadow_offline_batch_v1",
        "technical_status": "PASS",
        "dataset": split["dataset"],
        "role": split["role"],
        "retriever": args.retriever,
        "raw_all_units_completed_before_gt_scoring": True,
        "runtime_policy_access_to_gt": False,
        "threshold_applied_during_raw_association": False,
        "calibration_unlock_sha256": calibration_hash,
        "source_commit": capture_summary["provenance"]["source_commit"],
        "capture_summary_sha256": sha256(summary_path),
        "capture_episodes_sha256": sha256(episodes_path),
        "model_registry_sha256": sha256(model_registry),
        "split_manifest_sha256": sha256(split_path),
        "unit_registry_sha256": sha256(units_path),
        "merged_summary": merged,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
