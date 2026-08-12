from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "pbs" / "run_place_memory_lane.sh"
CONTROLLER = ROOT / "pbs" / "run_place_memory_3shared.pbs"
SUBMITTER = ROOT / "pbs" / "submit_case_audit.sh"
SUMMARIZER = ROOT / "scripts" / "summarize_place_memory_screen.py"
CASE_SUMMARIZER = ROOT / "scripts" / "summarize_case_audit.py"
POLICY = ROOT / "ascent" / "ascent_policy.py"


def _load_summarizer():
    sys.path.insert(0, str(SUMMARIZER.parent))
    spec = importlib.util.spec_from_file_location(
        "summarize_place_memory_screen", SUMMARIZER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_place_memory_shell_harness_is_syntactically_valid() -> None:
    subprocess.run(
        ["bash", "-n", str(WORKER), str(CONTROLLER), str(SUBMITTER)],
        check=True,
    )


def test_harness_freezes_paired_shadow_case_audit_contract() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    submitter = SUBMITTER.read_text(encoding="utf-8")
    summarizer = CASE_SUMMARIZER.read_text(encoding="utf-8")
    policy = POLICY.read_text(encoding="utf-8")

    for token in (
        '"ascent_place_memory.enabled=true"',
        '"ascent_place_memory.oracle_enabled=true"',
        '"ascent_place_memory.shadow_only=true"',
        "oracle_physical_planar_radius_m=0.75",
        "oracle_physical_height_radius_m=0.75",
        "oracle_min_step_separation=30",
        "oracle_min_excursion_m=2.0",
        "oracle_vo_consistent_radius_m=1.5",
        "place_low_gain_area_m2=0.5",
        "place_minimum_arrival_observations=2",
        "place_minimum_repeat_arrival_observations=3",
        "place_minimum_excursion_start_distance_m=1.4",
        "place_repeat_settlement=online_when_evidence_complete_and_residual_live",
        "oracle_place_diagnostics.jsonl",
        "paired_conditions=B1_ascent_vo,B2_v1_2_oracle_shadow",
        "actual_task_memory_action_changes=0",
    ):
        assert token in worker
    assert "oracle_metadata_count" in worker
    assert "oracle_episode_end_count" in worker
    assert "conditions=(B2 B1)" in worker
    assert "conditions=(B1 B2)" in worker
    assert "scripts/summarize_case_audit.py" in controller
    assert 'value["mechanism"]["actual_change_count"]' in controller
    assert "EXPECTED_CHUNKS=6" in submitter
    assert "EXPECTED_EPISODES=60" in submitter
    assert "ASCENT_PLACE_MEMORY_V14_ORIGINAL_BASELINE_SHA256" in submitter
    assert "original_baseline_hash_mismatch" in controller
    assert "ASCENT_V14_CASE_AUDIT_NO_QSUB" in submitter
    assert "merge-base HEAD f238fc5" in submitter
    assert "qsub -q \"$QUEUE\"" in submitter
    assert "case_mining_only_not_method_improvement" in summarizer
    assert "shadow_changed_action" in summarizer
    assert "incomplete_candidate_snapshot" in summarizer
    assert "if self._place_memory_config.shadow_only:" in policy
    assert "return base_frontier, base_value" in policy


def test_oracle_diagnostics_gate_accepts_only_frozen_metadata(
    tmp_path: Path,
) -> None:
    diagnostics = tmp_path / "oracle.jsonl"
    records = [
        {
            "record_type": "oracle_place_run_metadata",
            "method": "v1.4_oracle_task_memory",
            "dataset": "hm3d",
            "gt_policy_isolation": True,
            "policy_event_fields": [
                "event_sequence",
                "reference_submap_id",
            ],
            "oracle_config": {
                "enabled": True,
                "physical_planar_radius_m": 0.75,
                "physical_height_radius_m": 0.75,
                "min_step_separation": 30,
                "min_excursion_m": 2.0,
                "vo_consistent_radius_m": 1.5,
            },
        },
        {
            "record_type": "oracle_place_step",
            "episode_id": "0",
            "action_step": 1,
        },
        {
            "record_type": "oracle_place_episode_end",
            "episode_id": "0",
            "action_steps": 1,
        },
    ]
    diagnostics.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    module = _load_summarizer()
    _, errors = module.validate_oracle_diagnostics(
        {"chunk_id": "c000", "oracle_diagnostics": str(diagnostics)},
        expected_episodes=1,
        dataset="hm3d",
    )
    assert errors == []

    records[0]["method"] = "v1.4_case_audit_shadow"
    diagnostics.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    _, wrong_gate_errors = module.validate_oracle_diagnostics(
        {"chunk_id": "c000", "oracle_diagnostics": str(diagnostics)},
        expected_episodes=1,
        dataset="hm3d",
    )
    assert any("metadata:method" in item for item in wrong_gate_errors)
    _, shadow_errors = module.validate_oracle_diagnostics(
        {"chunk_id": "c000", "oracle_diagnostics": str(diagnostics)},
        expected_episodes=1,
        dataset="hm3d",
        expected_method="v1.4_case_audit_shadow",
    )
    assert shadow_errors == []
