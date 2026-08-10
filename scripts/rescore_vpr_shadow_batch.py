#!/usr/bin/env python3
"""Rebuild only GT labels from immutable GT-free VPR raw candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from run_vpr_shadow_offline_batch import (
        _read_capture_units,
        _run,
        merge_scored_units,
    )
    from vpr_shadow_data import read_jsonl, sha256
except ImportError:
    from scripts.run_vpr_shadow_offline_batch import (
        _read_capture_units,
        _run,
        merge_scored_units,
    )
    from scripts.vpr_shadow_data import read_jsonl, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-summary", type=Path, required=True)
    parser.add_argument("--capture-episodes", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--raw-batch-result", type=Path, required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--offline-source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    args = parse_args()
    summary_path = args.capture_summary.resolve()
    episodes_path = args.capture_episodes.resolve()
    split_path = args.split_manifest.resolve()
    raw_batch = args.raw_batch_result.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    capture, split, units = _read_capture_units(
        summary_path=summary_path,
        episodes_path=episodes_path,
        split_path=split_path,
    )
    if (
        capture.get("schema") != "ascent_v1_4_vpr_shadow_capture_gate_v2"
        or capture.get("immutable_raw_revalidation") is not True
        or capture.get("provenance", {}).get("validator_source_commit")
        != args.offline_source_commit
    ):
        raise ValueError("rescore capture is not bound to repaired source commit")

    source_summary_path = raw_batch / "summary.json"
    source_units_path = raw_batch / "units.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_units = json.loads(source_units_path.read_text(encoding="utf-8"))
    if (
        source_summary.get("schema")
        != "ascent_v1_4_vpr_shadow_offline_batch_v1"
        or source_summary.get("technical_status") != "PASS"
        or source_summary.get("retriever") != args.retriever
        or source_summary.get("raw_all_units_completed_before_gt_scoring") is not True
        or source_summary.get("runtime_policy_access_to_gt") is not False
        or source_summary.get("threshold_applied_during_raw_association") is not False
        or source_summary.get("split_manifest_sha256") != sha256(split_path)
        or source_summary.get("capture_episodes_sha256") != sha256(episodes_path)
        or len(source_units) != len(units)
    ):
        raise ValueError("source raw batch does not satisfy rescore contract")

    output_dir.mkdir(parents=True)
    (output_dir / "logs").mkdir()
    (output_dir / "scored").mkdir()
    scorer = Path(__file__).resolve().parent / "score_vpr_shadow_gt.py"
    scored_records = []
    source_records = []
    for unit, source in zip(units, source_units):
        if (
            str(source.get("chunk_id")) != str(unit["chunk_id"])
            or list(source.get("logical_case_ids", []))
            != list(unit["logical_case_ids"])
        ):
            raise ValueError("source raw unit order differs from capture gate")
        for key in (
            "capture_manifest",
            "vo_diagnostics",
            "submap_diagnostics",
            "identity_path",
        ):
            if Path(str(source.get(key, ""))).resolve() != Path(unit[key]).resolve():
                raise ValueError(f"source raw unit {key} differs")
        raw_dir = Path(str(source["raw_dir"])).resolve()
        if not _is_within(raw_dir, raw_batch / "raw"):
            raise ValueError("source raw unit escaped its batch root")
        raw_candidates = raw_dir / "raw_candidates.jsonl"
        raw_summary_path = raw_dir / "summary.json"
        raw_summary = json.loads(raw_summary_path.read_text(encoding="utf-8"))
        raw_rows = read_jsonl(raw_candidates)
        if (
            raw_summary.get("schema") != "ascent_v1_4_vpr_shadow_raw_summary_v1"
            or raw_summary.get("technical_status") != "PASS"
            or raw_summary.get("retriever") != args.retriever
            or raw_summary.get("raw_candidates_sha256") != sha256(raw_candidates)
            or not raw_rows
            or raw_rows[0].get("record_type") != "vpr_shadow_raw_metadata"
            or raw_rows[0].get("retriever") != args.retriever
            or raw_rows[0].get("evaluation_gt_read") is not False
            or raw_rows[0].get("planner_write_access") is not False
            or raw_rows[0].get("calibration_applied") is not False
        ):
            raise ValueError("source raw unit failed immutable GT-free gate")
        name = str(source["name"])
        score_dir = output_dir / "scored" / name
        command = [
            sys.executable,
            str(scorer),
            "--raw-candidates",
            str(raw_candidates),
            "--vo-diagnostics",
            str(unit["vo_diagnostics"]),
            "--submap-diagnostics",
            str(unit["submap_diagnostics"]),
            "--episode-identity",
            str(unit["identity_path"]),
            "--capture-episodes",
            str(episodes_path),
            "--chunk-id",
            str(unit["chunk_id"]),
            "--split-manifest",
            str(split_path),
            "--output-dir",
            str(score_dir),
        ]
        _run(command, output_dir / "logs" / f"{name}.score.log")
        scored_records.append({**unit, "name": name, "score_dir": str(score_dir)})
        source_records.append(
            {
                "chunk_id": str(unit["chunk_id"]),
                "raw_candidates_sha256": sha256(raw_candidates),
                "raw_summary_sha256": sha256(raw_summary_path),
            }
        )

    merged = merge_scored_units(
        scored_units=scored_records,
        split_path=split_path,
        retriever=args.retriever,
        output_dir=output_dir / "merged",
    )
    payload = {
        "schema": "ascent_v1_4_vpr_shadow_gt_rescore_v1",
        "technical_status": "PASS",
        "dataset": split["dataset"],
        "role": split["role"],
        "retriever": args.retriever,
        "raw_association_reused_without_inference": True,
        "runtime_policy_access_to_gt": False,
        "episode_join_contract": "capture_sequence_to_logical_to_runtime_v1",
        "offline_source_commit": args.offline_source_commit,
        "capture_summary_sha256": sha256(summary_path),
        "capture_episodes_sha256": sha256(episodes_path),
        "split_manifest_sha256": sha256(split_path),
        "source_raw_batch_summary_sha256": sha256(source_summary_path),
        "source_raw_batch_units_sha256": sha256(source_units_path),
        "source_raw_units": source_records,
        "merged_summary": merged,
    }
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
