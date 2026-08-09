from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_vpr_shadow_offline_batch import (
    _check_role_unlock,
    merge_scored_units,
)
from scripts.check_vpr_shadow_offline_compatibility import classify_changed_paths
from scripts.vpr_shadow_data import sha256


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _score_unit(
    root: Path,
    *,
    logical_id: str,
    scene_id: str,
    split_hash: str,
) -> Path:
    root.mkdir()
    candidate = {
        "record_type": "vpr_shadow_candidate",
        "logical_case_id": logical_id,
        "scene_id": scene_id,
        "event_index": 0,
        "event_id": "query:e0:s2:k1",
        "query_submap_id": "s2",
        "candidate_submap_id": "s0",
        "retrieval_rank": 1,
        "retrieval_score": 0.9,
        "geometry_pass": True,
        "vo_stratum": "vo_consistent_fragmentation",
        "gt_same_place_label": True,
        "gt_cross_floor_negative": False,
    }
    metadata = {
        "record_type": "vpr_shadow_gt_metadata",
        "schema": "ascent_v1_4_vpr_shadow_gt_labeled_candidates_v1",
        "evaluation_only_gt": True,
        "runtime_policy_access": False,
        "retriever": "mixvpr",
        "split_manifest_sha256": split_hash,
        "thresholds": {"return_radius_m": 0.75},
    }
    event = {
        "record_type": "vpr_shadow_gt_event",
        "logical_case_id": logical_id,
        "scene_id": scene_id,
        "event_index": 0,
        "event_id": "query:e0:s2:k1",
        "gt_revisit_exposure": True,
        "best_proposed_true_rank": 1,
    }
    _write_jsonl(root / "labeled_candidates.jsonl", [metadata, candidate])
    _write_jsonl(root / "events.jsonl", [event])
    (root / "summary.json").write_text(
        json.dumps(
            {
                "schema": "ascent_v1_4_vpr_shadow_gt_score_summary_v1",
                "technical_status": "PASS",
                "candidate_count": 1,
                "query_event_count": 1,
                "evaluated_episode_count": 1,
                "evaluated_logical_case_ids": [logical_id],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_merger_preserves_episode_identity_when_local_event_ids_repeat(
    tmp_path: Path,
) -> None:
    split = {
        "dataset": "hm3d",
        "role": "threshold_calibration_only",
        "episode_count": 2,
        "scene_ids": ["scene-a", "scene-b"],
        "episodes": [
            {"logical_case_id": "episode-a"},
            {"logical_case_id": "episode-b"},
        ],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    digest = sha256(split_path)
    first = _score_unit(
        tmp_path / "score-a",
        logical_id="episode-a",
        scene_id="scene-a",
        split_hash=digest,
    )
    second = _score_unit(
        tmp_path / "score-b",
        logical_id="episode-b",
        scene_id="scene-b",
        split_hash=digest,
    )
    summary = merge_scored_units(
        scored_units=[
            {
                "chunk_id": "a",
                "logical_case_ids": ["episode-a"],
                "score_dir": str(first),
            },
            {
                "chunk_id": "b",
                "logical_case_ids": ["episode-b"],
                "score_dir": str(second),
            },
        ],
        split_path=split_path,
        retriever="mixvpr",
        output_dir=tmp_path / "merged",
    )
    assert summary["evaluated_episode_count"] == 2
    assert summary["candidate_count"] == 2
    assert summary["query_event_count"] == 2


def test_locked_role_requires_matching_calibration_pass(tmp_path: Path) -> None:
    split = {"role": "locked_in_distribution_test"}
    with pytest.raises(ValueError, match="requires a frozen Cal50"):
        _check_role_unlock(
            split=split, retriever="mixvpr", calibration_file=None
        )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "ascent_v1_4_vpr_shadow_calibration_v1",
                "calibration_gate": "PASS",
                "retriever": "mixvpr",
            }
        ),
        encoding="utf-8",
    )
    assert _check_role_unlock(
        split=split,
        retriever="mixvpr",
        calibration_file=calibration,
    ) == sha256(calibration)


def test_offline_pbs_is_hash_bound_and_calibration_staged() -> None:
    runner = (ROOT / "pbs/run_vpr_shadow_offline.pbs").read_text()
    submitter = (ROOT / "pbs/submit_vpr_shadow_offline.sh").read_text()
    batch = (ROOT / "scripts/run_vpr_shadow_offline_batch.py").read_text()
    assert "MODEL_REGISTRY_SHA256" in runner
    assert "CAPTURE_SUMMARY_SHA256" in runner
    assert "runtime_policy_access_to_gt=0" in runner
    assert "gpu_justification" in runner
    assert "ASCENT_VPR_OFFLINE_NO_QSUB" in submitter
    assert "calibration_required" in submitter
    assert "source_commit_not_pushed" in submitter
    assert "OFFLINE_COMPATIBILITY_SHA256" in runner
    assert "CAPTURE_SOURCE_COMMIT" in runner
    assert "OFFLINE_SOURCE_COMMIT" in runner
    assert batch.index("# Pass 1:") < batch.index("# Pass 2:")
    assert "locked test association requires a frozen Cal50 PASS file" in batch


def test_post_capture_compatibility_scope_excludes_runtime_policy() -> None:
    assert classify_changed_paths(
        [
            "scripts/vpr_shadow_data.py",
            "scripts/vpr_shadow_capture_gate.py",
            "tests/test_vpr_shadow_data.py",
        ]
    ) == []
    assert classify_changed_paths(["ascent/submaps/vpr_shadow.py"]) == [
        "ascent/submaps/vpr_shadow.py"
    ]
