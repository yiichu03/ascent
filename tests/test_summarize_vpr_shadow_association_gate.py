from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_vpr_shadow_association_gate.py"


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _calibration(path: Path, retriever: str, verdict: str = "PASS") -> Path:
    return _write(
        path,
        {
            "schema": "ascent_v1_4_vpr_shadow_calibration_v1",
            "technical_status": "PASS",
            "calibration_gate": verdict,
            "retriever": retriever,
            "metrics": {"true_accepted_episode_count": 6},
        },
    )


def _evaluation(
    path: Path, retriever: str, role: str, calibration: Path
) -> Path:
    import hashlib

    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    return _write(
        path,
        {
            "schema": "ascent_v1_4_vpr_shadow_locked_evaluation_v1",
            "technical_status": "PASS",
            "association_gate": "PASS",
            "dataset": "hm3d" if role.startswith("locked_in") else "mp3d",
            "role": role,
            "retriever": retriever,
            "threshold_file_sha256": digest,
            "primary": {
                "precision": 1.0,
                "false_accepted_event_count": 0,
                "true_accepted_episode_count": 12,
            },
        },
    )


def test_one_retriever_pass_is_an_association_go(tmp_path: Path) -> None:
    mix_cal = _calibration(tmp_path / "mix-cal.json", "mixvpr")
    mega_cal = _calibration(tmp_path / "mega-cal.json", "megaloc", "NO_GO")
    hm = _evaluation(
        tmp_path / "hm.json",
        "mixvpr",
        "locked_in_distribution_test",
        mix_cal,
    )
    mp = _evaluation(
        tmp_path / "mp.json",
        "mixvpr",
        "locked_cross_dataset_transfer_test",
        mix_cal,
    )
    output = tmp_path / "gate.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--calibration",
            f"mixvpr={mix_cal}",
            "--calibration",
            f"megaloc={mega_cal}",
            "--evaluation",
            f"mixvpr={hm}",
            "--evaluation",
            f"mixvpr={mp}",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(output.read_text())
    assert value["association_go_no_go"] == "GO"
    assert value["passing_retrievers"] == ["mixvpr"]
    assert value["recommended_retriever"] == "mixvpr"
    assert "no PlaceMemory" in value["claim_boundary"]
