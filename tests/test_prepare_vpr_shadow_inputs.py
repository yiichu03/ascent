from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_vpr_shadow_inputs.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gzip(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with path.open("wb") as stream:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as out:
            out.write(raw)


def test_preparer_filters_by_frozen_identity_without_outcome_inputs(tmp_path: Path) -> None:
    root_file = tmp_path / "source/train.json.gz"
    content_file = tmp_path / "source/content/scene.json.gz"
    root_payload = {"episodes": [], "goals_by_category": {}}
    episodes = []
    rows = []
    for index in range(4):
        logical_id = f"hm3d_train_{index:04d}"
        scene = f"hm3d/train/scene{index}/scene{index}.basis.glb"
        episode = {
            "episode_id": str(index),
            "scene_id": scene,
            "object_category": "chair",
            "info": {"geodesic_distance": 3.0 + index},
        }
        episodes.append(episode)
        rows.append(
            {
                "logical_case_id": logical_id,
                "dataset": "hm3d",
                "dataset_case_index": str(index),
                "episode_seed": "100",
                "source_root_file": str(root_file),
                "source_content_file": str(content_file),
                "source_content_sha256": "pending",
                "source_episode_index": str(index),
                "source_episode_id": str(index),
                "scene_id": scene,
                "target_category": "chair",
                "geodesic_distance": str(3.0 + index),
            }
        )
    _gzip(root_file, root_payload)
    _gzip(content_file, {**root_payload, "episodes": episodes})
    for row in rows:
        row["source_content_sha256"] = _sha(content_file)
    selection = tmp_path / "selection.csv"
    with selection.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scene_config = tmp_path / "hm3d.scene_dataset_config.json"
    scene_config.write_text('{"stages": {}}\n', encoding="utf-8")
    source = {
        "dataset": "hm3d",
        "episode_count": 4,
        "logical_case_ids": [row["logical_case_id"] for row in rows],
        "selection_path": str(selection),
        "selection_sha256": _sha(selection),
        "scene_dataset_config": str(scene_config),
        "scene_dataset_config_sha256": _sha(scene_config),
        "chunks": [
            {"chunk_id": "source_c000", "episode_count": 2},
            {"chunk_id": "source_c001", "episode_count": 2},
        ],
    }
    source_path = tmp_path / "source_manifest.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    selected_indices = (1, 3)
    split = {
        "schema": "ascent_v1_4_vpr_shadow_split_v1",
        "dataset": "hm3d",
        "role": "threshold_calibration_only",
        "episode_count": 2,
        "source_manifest_sha256": _sha(source_path),
        "episodes": [
            {
                "logical_case_id": rows[index]["logical_case_id"],
                "scene_id": rows[index]["scene_id"],
                "source_chunk_id": "source_c000" if index < 2 else "source_c001",
                "source_episode_index": index % 2,
            }
            for index in selected_indices
        ],
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    output = tmp_path / "prepared"
    command = [
        sys.executable,
        str(SCRIPT),
        "--source-manifest",
        str(source_path),
        "--split-manifest",
        str(split_path),
        "--output-root",
        str(output),
        "--chunk-size",
        "1",
        "--label",
        "fixture",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run(
        [*command, "--verify-only"], check=True, capture_output=True, text=True
    )
    audit = json.loads((output / "preparation_audit.json").read_text())
    assert audit["outcome_fields_read"] is False
    assert audit["evaluation_gt_trajectory_read"] is False
    assert audit["logical_case_ids"] == [rows[1]["logical_case_id"], rows[3]["logical_case_id"]]
    manifest = json.loads(
        (output / "materialized/chunk_manifest.json").read_text()
    )
    assert (manifest["episode_count"], manifest["chunk_count"]) == (2, 2)
