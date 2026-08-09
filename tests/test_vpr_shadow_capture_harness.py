from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_capture_harness_is_passive_staged_and_project_local() -> None:
    worker = (ROOT / "pbs/run_submap_v1_2_lane.sh").read_text()
    controller = (ROOT / "pbs/run_submap_v1_2_3shared.pbs").read_text()
    submitter = (ROOT / "pbs/submit_vpr_shadow_capture.sh").read_text()
    validator = (ROOT / "scripts/validate_vpr_shadow_capture.py").read_text()
    control_validator = (
        ROOT / "scripts/validate_vpr_shadow_control.py"
    ).read_text()
    assert "ASCENT_VPR_SHADOW_CAPTURE_ENABLED" in worker
    assert "ASCENT_VPR_SHADOW_OUTPUT_DIR" in worker
    assert "vpr_shadow_manifest" in worker
    assert "WORKER_MODE=full" in controller
    assert "validate_vpr_shadow_capture.py" in controller
    assert "validate_vpr_shadow_control.py" in controller
    assert "passive_control" in controller
    assert '"$RUN_ROOT/passive_control" "$BASE_PORT_ROOT" 0' in controller
    assert '"$RUN_ROOT/lanes" "$((BASE_PORT_ROOT + lane_id * 20))"' in controller
    assert "ASCENT_SUBMAP_V12_MODE=shadow" in submitter
    assert "ASCENT_VPR_SHADOW_SENTINEL_SUMMARY" in submitter
    assert "action_equivalent_episodes" in submitter
    assert "capture_off_control_required" in submitter
    assert "historical_action_reference_sha256" in submitter
    assert "artifacts/objectnav/vpr_shadow" in submitter
    assert "/scratch/e1538633/liuyi/vpr_shadow" not in submitter
    assert '"navigation_metrics_emitted": False' in validator
    assert '"association_scores_emitted": False' in validator
    assert '"capture_enabled": False' in control_validator
    assert '"same_commit_capture_off_control"' in control_validator


def test_capture_harness_does_not_run_vpr_or_change_planner() -> None:
    worker = (ROOT / "pbs/run_submap_v1_2_lane.sh").read_text()
    submitter = (ROOT / "pbs/submit_vpr_shadow_capture.sh").read_text()
    forbidden = (
        "run_vpr_shadow_association.py",
        "MixVPR",
        "MegaLoc",
        "LightGlue",
        "score_vpr_shadow_gt.py",
    )
    for token in forbidden:
        assert token not in worker
        assert token not in submitter
    assert '"planner_action_change": False' in submitter
    assert '"vpr_runtime_association": False' in submitter
