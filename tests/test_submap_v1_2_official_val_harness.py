from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARER = ROOT / "scripts" / "prepare_submap_v1_2_official_val_expansion.py"
CONTROLLER = ROOT / "pbs" / "run_submap_v1_2_official_val_3shared.pbs"
SUBMITTER = ROOT / "pbs" / "submit_submap_v1_2_official_val.sh"


def load_preparer():
    sys.path.insert(0, str(PREPARER.parent))
    spec = importlib.util.spec_from_file_location("official_val_preparer", PREPARER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_official_val_shard_sizes() -> None:
    module = load_preparer()
    assert module.shard_sizes(2195, 3) == [732, 732, 731]
    assert module.shard_sizes(1000, 1) == [1000]


def test_canonical_official_val_identity() -> None:
    module = load_preparer()
    ids = module.canonical_ids("hm3d", 2000)
    assert ids[999:1002] == [
        "hm3d_val_0999",
        "hm3d_val_1000",
        "hm3d_val_1001",
    ]


def test_controller_enforces_scientific_and_technical_boundaries() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    for contract in (
        'config["scientific_split"] == "val"',
        'config["transport_split"] == "train"',
        'config["policy_gt_isolation"] is True',
        'config["evaluation_gt_only"] is True',
        'config["metric_driven_retry"] is False',
        "placement_gate_failed",
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
        "canonical_official_val_contiguous_index_no_metric_filter",
        "metrics_used_for_selection",
        "qsub_executed",
    ):
        assert contract in text
