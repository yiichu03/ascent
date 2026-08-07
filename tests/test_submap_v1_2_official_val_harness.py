from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "scripts" / "prepare_gt_submap_v1_2_hm3d_prefix1000.py"
CONTROLLER = ROOT / "pbs" / "run_submap_v1_2_official_val_3shared.pbs"
SUBMITTER = ROOT / "pbs" / "submit_submap_v1_2_official_val.sh"


def load_preparer():
    sys.path.insert(0, str(PREPARER.parent))
    spec = importlib.util.spec_from_file_location("official_val_preparer", PREPARER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparer_reuses_frozen_prefix_without_copying_transport() -> None:
    text = PREPARER.read_text(encoding="utf-8")
    for contract in (
        "SHARD_SIZES = (334, 333, 333)",
        '"expected_episodes": expected_size',
        '"pose_source": "habitat_ground_truth"',
        '"transport_files_copied": False',
        '"metrics_used_for_selection": False',
    ):
        assert contract in text


def test_preparer_is_importable() -> None:
    module = load_preparer()
    assert callable(module.sha256)
    assert callable(module.main)


def test_controller_enforces_scientific_and_technical_boundaries() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    for contract in (
        'config["scientific_split"] == "val"',
        'config["transport_split"] == "train"',
        'config["pose_source"] == "habitat_ground_truth"',
        'config["policy_gt_isolation"] is False',
        'config["auxiliary_vo_sensors"] is False',
        'config["evaluation_gt_only"] is True',
        'config["metric_driven_retry"] is False',
        "placement_gate_failed",
        "ASCENT_GT_SUBMAP_V12_SMOKE_ONLY",
        'config["smoke_logical_case_ids"]',
        "historical_b1_matched_episodes",
        "source_worktree_not_clean",
    ):
        assert contract in text


def test_submitter_is_fail_closed_and_supports_no_qsub() -> None:
    text = SUBMITTER.read_text(encoding="utf-8")
    for contract in (
        "ASCENT_SUBMAP_V12_OFFVAL_NO_QSUB",
        "source_commit_not_pushed",
        "queue_must_be_auto_or_autox",
        "canonical_official_val_prefix_no_metric_filter",
        "metrics_used_for_selection",
        '"pose_source": "habitat_ground_truth"',
        'value["smoke_logical_case_ids"]',
        '"smoke_only"',
        "qsub_executed",
    ):
        assert contract in text
