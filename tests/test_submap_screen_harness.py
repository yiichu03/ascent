from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
CALIBRATOR = ROOT / "scripts" / "calibrate_submap_thresholds.py"
SUMMARIZER = ROOT / "scripts" / "summarize_submap_screen.py"
V1_1_SUMMARIZER = (
    ROOT / "scripts" / "summarize_submap_v1_1_screen.py"
)
V1_2_SUMMARIZER = (
    ROOT / "scripts" / "summarize_submap_v1_2_screen.py"
)
MATERIALIZER = ROOT / "scripts" / "materialize_submap_screen.py"
MANIFEST_DERIVER = ROOT / "scripts" / "derive_submap_manifest.py"
PREFLIGHT = ROOT / "scripts" / "preflight_submap_screen.py"
FORWARD = "6b571bb717366f7d80f61e919b33a45ac2f45925201c4e3011b3239a2c42e586"
TURN = "c469643f9ab35c9e1058f31fbb672a5fa3adf582987a4388bdd020dd89faf1d9"
SOURCE = "f" * 40


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(
    path: Path, rows: list[Mapping[str, Any]], fields: list[str]
) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_gzip_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with path.open("xb") as stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0
        ) as handle:
            handle.write(raw)


def make_materializer_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    scene_root = tmp_path / "scenes"
    scene_dir = scene_root / "hm3d" / "train" / "00000-scene"
    scene_dir.mkdir(parents=True)
    for name in (
        "scene.basis.glb",
        "scene.basis.navmesh",
        "scene.semantic.glb",
        "scene.semantic.txt",
    ):
        (scene_dir / name).write_text(name, encoding="utf-8")
    scene_config = (
        scene_root
        / "hm3d"
        / "hm3d_annotated_basis.scene_dataset_config.json"
    )
    scene_config.write_text('{"stages": {}}\n', encoding="utf-8")

    source_root = tmp_path / "source" / "train.json.gz"
    source_content = tmp_path / "source" / "content" / "scene.json.gz"
    root_payload = {
        "episodes": [],
        "goals_by_category": {},
        "category_to_task_category_id": {"chair": 0},
    }
    episode = {
        "episode_id": "7",
        "scene_id": "hm3d/train/00000-scene/scene.basis.glb",
        "scene_dataset_config": "./data/wrong_relative_config.json",
        "object_category": "chair",
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [],
        "info": {"geodesic_distance": 4.0},
    }
    write_gzip_json(source_root, root_payload)
    write_gzip_json(
        source_content,
        {
            **root_payload,
            "episodes": [episode],
            "goals_by_category": {},
        },
    )
    selection = tmp_path / "selection.csv"
    row = {
        "logical_case_id": "case_0",
        "dataset": "hm3d",
        "dataset_case_index": "0",
        "episode_seed": "100",
        "source_root_file": str(source_root),
        "source_content_file": str(source_content),
        "source_content_sha256": sha256(source_content),
        "source_episode_index": "0",
        "source_episode_id": "7",
        "scene_id": episode["scene_id"],
        "target_category": "chair",
        "geodesic_distance": "4.0",
    }
    write_csv(selection, [row], list(row))
    return selection, scene_root, scene_config


def test_materialized_episode_binds_hashed_absolute_scene_config(
    tmp_path: Path,
) -> None:
    selection, scene_root, scene_config = make_materializer_fixture(
        tmp_path
    )
    output_root = tmp_path / "materialized"
    subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--selection",
            str(selection),
            "--output-root",
            str(output_root),
            "--scene-dataset-config",
            str(scene_config),
            "--dataset",
            "hm3d",
            "--episodes",
            "1",
            "--chunk-size",
            "1",
            "--label",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_path = output_root / "chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["scene_dataset_config"] == str(
        scene_config.resolve()
    )
    assert manifest["scene_dataset_config_sha256"] == sha256(
        scene_config
    )
    with gzip.open(
        manifest["chunks"][0]["content_file"], "rt", encoding="utf-8"
    ) as handle:
        episode = json.load(handle)["episodes"][0]
    assert episode["scene_dataset_config"] == str(scene_config.resolve())

    preflight_output = tmp_path / "preflight.json"
    subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--manifest",
            str(manifest_path),
            "--scene-root",
            str(scene_root),
            "--expected-episodes",
            "1",
            "--expected-chunks",
            "1",
            "--output-json",
            str(preflight_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(preflight_output.read_text())["status"] == "PASS"

    val_manifest = tmp_path / "val_manifest.json"
    val_payload = json.loads(manifest_path.read_text())
    val_payload["split"] = "val"
    val_manifest.write_text(json.dumps(val_payload), encoding="utf-8")
    val_preflight = tmp_path / "val_preflight.json"
    subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--manifest",
            str(val_manifest),
            "--scene-root",
            str(scene_root),
            "--expected-split",
            "val",
            "--expected-episodes",
            "1",
            "--expected-chunks",
            "1",
            "--output-json",
            str(val_preflight),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(val_preflight.read_text())["split"] == "val"


def test_materializer_preserves_selection_order_across_source_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source" / "train.json.gz"
    root_payload = {
        "episodes": [],
        "goals_by_category": {},
        "category_to_task_category_id": {"chair": 0},
    }
    write_gzip_json(source_root, root_payload)
    rows = []
    for dataset_index, source_name in enumerate(("z_scene", "a_scene")):
        source_content = (
            tmp_path / "source" / "content" / f"{source_name}.json.gz"
        )
        episode = {
            "episode_id": str(70 + dataset_index),
            "scene_id": (
                f"hm3d/train/{source_name}/{source_name}.basis.glb"
            ),
            "object_category": "chair",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "goals": [],
            "info": {"geodesic_distance": 4.0 + dataset_index},
        }
        write_gzip_json(
            source_content,
            {
                **root_payload,
                "episodes": [episode],
                "goals_by_category": {},
            },
        )
        rows.append(
            {
                "logical_case_id": f"case_{dataset_index}",
                "dataset": "hm3d",
                "dataset_case_index": str(dataset_index),
                "episode_seed": str(100 + dataset_index),
                "source_root_file": str(source_root),
                "source_content_file": str(source_content),
                "source_content_sha256": sha256(source_content),
                "source_episode_index": "0",
                "source_episode_id": episode["episode_id"],
                "scene_id": episode["scene_id"],
                "target_category": "chair",
                "geodesic_distance": str(4.0 + dataset_index),
            }
        )
    selection = tmp_path / "selection.csv"
    write_csv(selection, rows, list(rows[0]))
    scene_config = tmp_path / "hm3d.scene_dataset_config.json"
    scene_config.write_text('{"stages": {}}\n', encoding="utf-8")
    output_root = tmp_path / "materialized"
    subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--selection",
            str(selection),
            "--output-root",
            str(output_root),
            "--scene-dataset-config",
            str(scene_config),
            "--dataset",
            "hm3d",
            "--episodes",
            "2",
            "--chunk-size",
            "2",
            "--label",
            "ordered",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(
        (output_root / "chunk_manifest.json").read_text()
    )
    chunk = manifest["chunks"][0]
    identities = read_csv(Path(chunk["identity_path"]))
    with gzip.open(
        chunk["content_file"], "rt", encoding="utf-8"
    ) as handle:
        episodes = json.load(handle)["episodes"]
    expected = ["case_0", "case_1"]
    assert manifest["logical_case_ids"] == expected
    assert [row["logical_case_id"] for row in identities] == expected
    assert [episode["episode_id"] for episode in episodes] == expected


def test_mp3d_materialized_assets_pass_dataset_specific_preflight(
    tmp_path: Path,
) -> None:
    scene_root = tmp_path / "scenes"
    scene_dir = scene_root / "mp3d" / "scene"
    scene_dir.mkdir(parents=True)
    for name in (
        "scene.glb",
        "scene.navmesh",
        "scene_semantic.ply",
        "scene.house",
    ):
        (scene_dir / name).write_text(name, encoding="utf-8")
    scene_config = (
        scene_root / "mp3d" / "mp3d.scene_dataset_config.json"
    )
    scene_config.write_text('{"stages": {}}\n', encoding="utf-8")
    source_root = tmp_path / "source" / "train.json.gz"
    source_content = tmp_path / "source" / "content" / "scene.json.gz"
    root_payload = {
        "episodes": [],
        "goals_by_category": {},
        "category_to_task_category_id": {"chair": 0},
    }
    episode = {
        "episode_id": "17",
        "scene_id": "mp3d/scene/scene.glb",
        "object_category": "chair",
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [],
        "info": {"geodesic_distance": 4.0},
    }
    write_gzip_json(source_root, root_payload)
    write_gzip_json(
        source_content,
        {**root_payload, "episodes": [episode]},
    )
    selection = tmp_path / "selection.csv"
    row = {
        "logical_case_id": "mp3d_case_0",
        "dataset": "mp3d",
        "dataset_case_index": "0",
        "episode_seed": "100",
        "source_root_file": str(source_root),
        "source_content_file": str(source_content),
        "source_content_sha256": sha256(source_content),
        "source_episode_index": "0",
        "source_episode_id": "17",
        "scene_id": episode["scene_id"],
        "target_category": "chair",
        "geodesic_distance": "4.0",
    }
    write_csv(selection, [row], list(row))
    output_root = tmp_path / "materialized"
    subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--selection",
            str(selection),
            "--output-root",
            str(output_root),
            "--scene-dataset-config",
            str(scene_config),
            "--dataset",
            "mp3d",
            "--episodes",
            "1",
            "--chunk-size",
            "1",
            "--label",
            "mp3d_fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output_json = tmp_path / "preflight.json"
    subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--manifest",
            str(output_root / "chunk_manifest.json"),
            "--scene-root",
            str(scene_root),
            "--expected-dataset",
            "mp3d",
            "--expected-episodes",
            "1",
            "--expected-chunks",
            "1",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output_json.read_text())
    assert result["dataset"] == "mp3d"
    assert result["status"] == "PASS"


def test_manifest_derivation_repairs_relocated_chunk_paths(
    tmp_path: Path,
) -> None:
    selection, scene_root, scene_config = make_materializer_fixture(
        tmp_path
    )
    original_root = tmp_path / "original_materialized"
    subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--selection",
            str(selection),
            "--output-root",
            str(original_root),
            "--scene-dataset-config",
            str(scene_config),
            "--dataset",
            "hm3d",
            "--episodes",
            "1",
            "--chunk-size",
            "1",
            "--label",
            "fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    relocated_root = tmp_path / "relocated_materialized"
    original_root.rename(relocated_root)
    source_manifest = relocated_root / "chunk_manifest.json"
    stale = json.loads(source_manifest.read_text())
    assert not Path(stale["chunks"][0]["data_path"]).exists()

    derived_manifest = tmp_path / "derived" / "shard0.json"
    subprocess.run(
        [
            sys.executable,
            str(MANIFEST_DERIVER),
            "--source-manifest",
            str(source_manifest),
            "--materialized-root",
            str(relocated_root),
            "--output-manifest",
            str(derived_manifest),
            "--chunk-indices",
            "0",
            "--variant",
            "relocated_shard0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    derived = json.loads(derived_manifest.read_text())
    assert derived["episode_count"] == 1
    assert derived["chunk_count"] == 1
    assert derived["logical_case_ids"] == ["case_0"]
    assert derived["derivation"]["source_chunk_indices"] == [0]
    assert (
        derived["derivation"]["source_manifest_sha256"]
        == sha256(source_manifest)
    )
    assert Path(derived["chunks"][0]["data_path"]).is_file()

    preflight_output = tmp_path / "derived_preflight.json"
    subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--manifest",
            str(derived_manifest),
            "--scene-root",
            str(scene_root),
            "--expected-dataset",
            "hm3d",
            "--expected-episodes",
            "1",
            "--expected-chunks",
            "1",
            "--output-json",
            str(preflight_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(preflight_output.read_text())["status"] == "PASS"


def test_unified_controller_binds_dataset_and_real_reset_contracts() -> None:
    checker = (
        ROOT / "scripts" / "check_submap_task_reset.py"
    ).read_text(encoding="utf-8")
    controller = (
        ROOT / "pbs" / "run_submap_screen_3shared.pbs"
    ).read_text(encoding="utf-8")
    submitter = (
        ROOT / "pbs" / "submit_submap_screen.sh"
    ).read_text(encoding="utf-8")
    assert "with habitat.Env(" in checker
    assert 'config_name=f"eval_ascent_{dataset}.yaml"' in checker
    assert "config.habitat.environment.iterator_options.cycle = False" in checker
    assert 'screen_task_reset.json' in controller
    assert 'calibration_task_reset.json' in controller
    assert '--expected-dataset "$DATASET"' in controller
    assert "CALIBRATION_CONTRACT=hm3d_train30_native_calibration" in controller
    assert (
        "CALIBRATION_CONTRACT=hm3d_train30_frozen_no_mp3d_retuning"
    ) in controller
    assert "echo calibration_contract=$CALIBRATION_CONTRACT" in controller
    assert 'SUMMARY_ARGS+=(--calibration-json "$CALIBRATION_OUTPUT")' in controller
    assert '"$TASK_RESET_CHECK"' in controller
    assert (
        "ARTIFACT_ROOT=${ASCENT_SUBMAP_ARTIFACT_ROOT:-"
        "$PROJECT/artifacts/objectnav/submap_v1}"
    ) in controller
    assert (
        "RUN_ROOT=$ARTIFACT_ROOT/runs/submap_${DATASET}_${MODE}_"
    ) in controller
    assert "artifact_root=$ARTIFACT_ROOT" in controller
    assert "MANIFEST_ROOT=$ARTIFACT_ROOT/manifests" in submitter
    assert "LOG_ROOT=$ARTIFACT_ROOT/pbs_logs" in submitter
    assert "SUBMISSION_ROOT=$ARTIFACT_ROOT/submissions" in submitter
    assert "ASCENT_SUBMAP_ARTIFACT_ROOT=$ARTIFACT_ROOT" in submitter
    assert "ASCENT_SUBMAP_SHARD_INDEX=$SHARD_INDEX" in submitter
    assert "ASCENT_SUBMAP_SHARD_COUNT=$SHARD_COUNT" in submitter
    assert "LANE_COUNT=$EXPECTED_CHUNKS" in controller
    assert "full_shard_episode_contract" in controller
    assert "/scratch/e1538633/liuyi/submap_v1_" not in submitter
    assert "external/ascent_vo_submap_v1_mp3d" not in controller


def test_unified_lane_runs_changed_condition_first_in_smoke() -> None:
    worker = (
        ROOT / "pbs" / "run_submap_screen_lane.sh"
    ).read_text(encoding="utf-8")
    run_unit_start = worker.index("run_unit() {")
    run_unit = worker[
        run_unit_start : worker.index(
            '\n}\n\nif [ "$RUN_CALIBRATION" = true ]', run_unit_start
        )
    ]
    assert "set +e" not in run_unit
    assert 'if (\n    export ASCENT_SUBMAP_ENABLED=' in run_unit
    assert "  ); then\n    status=0\n  else\n    status=$?\n  fi" in run_unit
    assert "42|44)" in worker
    assert "run_unit 1 calibration CAL" in worker
    assert "--config-name=eval_ascent_${DATASET}.yaml" in worker
    assert 'if [ "$MODE" = smoke ]; then\n    # Exercise' in worker
    assert "conditions=(B2 B1)" in worker
    assert "ascent_submaps.provisional_thresholds=false" in worker
    assert (
        "ascent_submaps.max_motion_budget_m="
        "${CALIBRATED[max_motion_budget_m]}"
    ) in worker
    assert "external/ascent_vo_submap_v1_mp3d" not in worker


def vo_metadata() -> dict[str, Any]:
    return {
        "record_type": "run_metadata",
        "provider": "zhao_rgbd_2021",
        "ascent_source_commit": SOURCE,
        "forward_checkpoint_sha256": FORWARD,
        "turn_checkpoint_sha256": TURN,
        "action_contract": "native_0.25m_or_30deg_single_pair",
        "pose_initialization": "episode_local_zero_se2",
        "gt_policy_isolation": True,
    }


def test_calibration_is_train_diagnostic_only(tmp_path: Path) -> None:
    vo_path = tmp_path / "vo.jsonl"
    submap_path = tmp_path / "submap.jsonl"
    vo_rows = [vo_metadata()]
    submap_rows = [
        {
            "record_type": "submap_run_metadata",
            "ascent_source_commit": SOURCE,
            "pose_source": "zhao_rgbd_2021",
            "policy_gt_isolation": True,
            "config": {
                "enabled": True,
                "provisional_thresholds": True,
                "min_action_endpoints": 10000,
                "min_path_length_m": 10000.0,
                "overlap_threshold": 0.0,
                "low_overlap_consecutive": 10000,
                "max_motion_budget_m": 10000.0,
            },
        }
    ]
    for episode in range(5):
        submap_rows.append(
            {
                "record_type": "submap_episode_reset",
                "episode_sequence": episode,
            }
        )
        for step in range(1, 26):
            motion = step * 0.2
            error = motion * 0.06
            vo_rows.append(
                {
                    "record_type": "vo_step",
                    "episode_id": str(episode),
                    "action_step": step,
                    "localization_error_available": True,
                    "translation_error": error,
                    "absolute_yaw_error": 0.01 * step,
                }
            )
            submap_rows.append(
                {
                    "record_type": "submap_action_endpoint",
                    "episode_sequence": episode,
                    "action_step": step,
                    "submap_id": "env0:sm0000",
                    "overlap": max(0.05, 0.6 - error),
                    "decision": {"motion_budget_m": motion},
                }
            )
        vo_rows.append(
            {
                "record_type": "episode_end",
                "episode_id": str(episode),
            }
        )
    write_jsonl(vo_path, vo_rows)
    write_jsonl(submap_path, submap_rows)
    inventory = tmp_path / "inventory.csv"
    write_csv(
        inventory,
        [
            {
                "priority": 0,
                "condition": "CAL",
                "chunk_id": "calibration_c000",
                "terminal_class": "complete",
                "vo_diagnostics": vo_path,
                "submap_diagnostics": submap_path,
            }
        ],
        [
            "priority",
            "condition",
            "chunk_id",
            "terminal_class",
            "vo_diagnostics",
            "submap_diagnostics",
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "ascent_vo_submap_screen_materialized_v1",
                "dataset": "hm3d",
                "split": "train",
                "episode_count": 5,
                "selection_path": "/fixed/selection.csv",
                "selection_sha256": "a" * 64,
                "chunks": [{"chunk_id": "calibration_c000"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "calibration.json"
    subprocess.run(
        [
            sys.executable,
            str(CALIBRATOR),
            "--inventory",
            str(inventory),
            "--manifest",
            str(manifest),
            "--output-json",
            str(output),
            "--source-commit",
            SOURCE,
            "--forward-checkpoint-sha256",
            FORWARD,
            "--turn-checkpoint-sha256",
            TURN,
            "--expected-episodes",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert result["status"] == "PASS"
    assert result["navigation_metrics_used"] is False
    assert result["matched_endpoint_count"] == 125
    assert result["config"]["provisional_thresholds"] is False
    assert 2.0 <= result["config"]["max_motion_budget_m"] <= 8.0


def make_vo_diagnostics(
    path: Path,
    condition: str,
    episode_count: int,
    *,
    execution_order: Optional[Iterable[int]] = None,
    action_steps_by_episode: Optional[Mapping[int, int]] = None,
) -> None:
    rows = [vo_metadata()]
    order = (
        list(range(episode_count))
        if execution_order is None
        else list(execution_order)
    )
    step_counts = dict(action_steps_by_episode or {})
    for episode in order:
        action_steps = step_counts.get(episode, 2)
        for step in range(1, action_steps + 1):
            rows.append(
                {
                    "record_type": "vo_step",
                    "scene_id": f"/asset/hm3d/train/s{episode}/s{episode}.basis.glb",
                    "episode_id": str(episode),
                    "action_step": step,
                    "finite": True,
                    "vo_inferences": 1,
                    "localization_error_available": True,
                    "translation_error": 0.01 * step,
                    "absolute_yaw_error": 0.005 * step,
                }
            )
        rows.append(
            {
                "record_type": "episode_end",
                "scene_id": f"/asset/hm3d/train/s{episode}/s{episode}.basis.glb",
                "episode_id": str(episode),
                "action_steps": action_steps,
                "native_metrics": {
                    "success": float(
                        episode == 0 or condition == "B2"
                    ),
                    "spl": 0.5,
                    "soft_spl": 0.6,
                    "distance_to_goal": 0.2,
                },
                "evaluation_pose_source": "habitat_ground_truth",
                "policy_pose_source": "zhao_rgbd_2021",
                "gt_policy_isolation": True,
            }
        )
    write_jsonl(path, rows)


def make_submap_diagnostics(
    path: Path,
    config: Mapping[str, Any],
    episode_count: int,
    *,
    execution_order: Optional[Iterable[int]] = None,
    action_steps_by_episode: Optional[Mapping[int, int]] = None,
    v1_1: bool = False,
    v1_2: bool = False,
) -> None:
    metadata = {
        "record_type": "submap_run_metadata",
        "ascent_source_commit": SOURCE,
        "pose_source": "zhao_rgbd_2021",
        "policy_gt_isolation": True,
        "config": {"enabled": True, **dict(config)},
    }
    if v1_1 or v1_2:
        metadata.update(
            {
                "method_version": (
                    "submap_v1.2" if v1_2 else "submap_v1.1"
                ),
                "split_contract": (
                    "vo_anchor_and_rgbd_overlap_joint"
                ),
                "fallback_contract": (
                    "ascent_local_first_persistent_route"
                ),
            }
        )
    if v1_2:
        metadata.update(
            {
                "handoff_enabled": True,
                "exhaustion_recovery_enabled": True,
                "continuity_contract": (
                    "single_connected_handoff_and_one_shot_360_recovery"
                ),
                "route_gateway_replan_enabled": True,
            }
        )
    rows = [metadata]
    order = (
        list(range(episode_count))
        if execution_order is None
        else list(execution_order)
    )
    step_counts = dict(action_steps_by_episode or {})
    for sequence, episode in enumerate(order):
        rows.append(
            {
                "record_type": "submap_episode_reset",
                "episode_sequence": sequence,
            }
        )
        for step in range(step_counts.get(episode, 2)):
            endpoint = {
                "record_type": "submap_action_endpoint",
                "episode_sequence": sequence,
                "action_step": step,
            }
            if v1_1 or v1_2:
                is_last = step == step_counts.get(episode, 2) - 1
                endpoint["decision"] = {
                    "should_split": is_last,
                    "reason": "low_overlap" if is_last else None,
                    "mature": is_last,
                    "anchor_displacement_m": 1.5 if is_last else 0.5,
                    "low_overlap_streak": 3 if is_last else step + 1,
                    "route_action": False,
                }
            rows.append(endpoint)
        rows.append(
            {
                "record_type": "submap_event",
                "episode_sequence": sequence,
                "event": "submap_split",
                "reason": "low_overlap",
            }
        )
        if v1_1 or v1_2:
            rows.extend(
                [
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "remote_route_selected",
                        "candidate_key": (
                            f"semantic:sm{episode}:chair"
                        ),
                        "target_kind": "semantic",
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "remote_route_finished",
                        "outcome": "destination_reached",
                    },
                ]
            )
        if v1_2 and episode == 0:
            rows.extend(
                [
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "remote_route_gateway_replan_paused",
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "handoff_created",
                        "waypoint_local": [1.0, 0.0],
                        "replayed_depth_frames": 4,
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "handoff_action",
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "handoff_completed",
                        "reason": "reached",
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "exhaustion_recovery_requested",
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "submap_exhaustion_recovery",
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "exhaustion_recovery_started",
                    },
                    *[
                        {
                            "record_type": "submap_event",
                            "episode_sequence": sequence,
                            "event": "exhaustion_recovery_scan_turn",
                            "scan_turn_index": index,
                        }
                        for index in range(2, 13)
                    ],
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "exhaustion_recovery_scan_completed",
                    },
                ]
            )
        elif v1_2 and episode == 1:
            rows.extend(
                [
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "exhaustion_recovery_requested",
                    },
                    {
                        "record_type": "submap_event",
                        "episode_sequence": sequence,
                        "event": "exhaustion_recovery_skipped",
                        "reason": "floor_transition_priority",
                    },
                ]
            )
    write_jsonl(path, rows)


def test_paired_summarizer_checks_submap_and_gt_separation(
    tmp_path: Path,
) -> None:
    identities = tmp_path / "identity.csv"
    identity_rows = [
        {
            "runtime_episode_id": str(index),
            "logical_case_id": f"case_{index}",
            "dataset": "hm3d",
            "episode_seed": str(10 + index),
            "source_episode_id": str(index),
            "scene_id": f"hm3d/train/s{index}/s{index}.basis.glb",
            "target_category": "chair",
            "geodesic_distance": "5.0",
        }
        for index in range(2)
    ]
    write_csv(identities, identity_rows, list(identity_rows[0]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "ascent_vo_submap_screen_materialized_v1",
                "dataset": "hm3d",
                "split": "train",
                "episode_count": 2,
                "logical_case_ids": ["case_0", "case_1"],
                "chunks": [
                    {
                        "chunk_id": "c000",
                        "episode_count": 2,
                        "identity_path": str(identities),
                        "identity_sha256": sha256(identities),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "min_action_endpoints": 20,
        "min_path_length_m": 1.5,
        "overlap_threshold": 0.25,
        "low_overlap_consecutive": 3,
        "max_motion_budget_m": 4.0,
        "rotation_weight_m_per_rad": 0.1,
        "gateway_frontier_resolution_radius_m": 1.0,
        "gateway_reached_radius_m": 0.9,
        "provisional_thresholds": False,
    }
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps({"status": "PASS", "config": config}) + "\n",
        encoding="utf-8",
    )
    b1_vo = tmp_path / "b1_vo.jsonl"
    b2_vo = tmp_path / "b2_vo.jsonl"
    b2_submap = tmp_path / "b2_submap.jsonl"
    action_steps = {0: 2, 1: 3}
    make_vo_diagnostics(
        b1_vo,
        "B1",
        2,
        action_steps_by_episode=action_steps,
    )
    make_vo_diagnostics(
        b2_vo,
        "B2",
        2,
        execution_order=[1, 0],
        action_steps_by_episode=action_steps,
    )
    make_submap_diagnostics(
        b2_submap,
        config,
        2,
        execution_order=[1, 0],
        action_steps_by_episode=action_steps,
    )
    inventory = tmp_path / "inventory.csv"
    rows = [
        {
            "priority": 0,
            "stage": "screen",
            "condition": "B1",
            "chunk_id": "c000",
            "vo_diagnostics": b1_vo,
            "submap_diagnostics": "",
        },
        {
            "priority": 0,
            "stage": "screen",
            "condition": "B2",
            "chunk_id": "c000",
            "vo_diagnostics": b2_vo,
            "submap_diagnostics": b2_submap,
        },
    ]
    write_csv(
        inventory,
        rows,
        [
            "priority",
            "stage",
            "condition",
            "chunk_id",
            "vo_diagnostics",
            "submap_diagnostics",
        ],
    )
    registry = tmp_path / "registry.csv"
    write_csv(
        registry,
        [{"lane_id": 0, "inventory": inventory}],
        ["lane_id", "inventory"],
    )
    output = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARIZER),
            "--manifest",
            str(manifest),
            "--registry",
            str(registry),
            "--mode",
            "full",
            "--expected-episodes",
            "2",
            "--source-commit",
            SOURCE,
            "--forward-checkpoint-sha256",
            FORWARD,
            "--turn-checkpoint-sha256",
            TURN,
            "--calibration-json",
            str(calibration),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((output / "summary.json").read_text())
    assert result["technical_status"] == "PASS"
    assert result["paired_valid_episodes"] == 2
    assert result["metrics"]["success_flips"]["0->1"] == 1
    assert result["mechanism"]["split_count"] == 2
    assert result["mechanism"]["submap_event_counts"][
        "submap_split"
    ] == 2
    with (output / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        paired_episode_rows = list(csv.DictReader(handle))
    assert json.loads(
        paired_episode_rows[0]["submap_event_counts_json"]
    )["submap_split"] == 1


def test_v1_1_b2_only_summarizer_matches_frozen_arm_a(
    tmp_path: Path,
) -> None:
    identities = tmp_path / "identity.csv"
    identity_rows = [
        {
            "runtime_episode_id": str(index),
            "logical_case_id": f"case_{index}",
            "dataset": "hm3d",
            "episode_seed": str(10 + index),
            "source_episode_id": str(index),
            "scene_id": f"hm3d/train/s{index}/s{index}.basis.glb",
            "target_category": "chair",
            "geodesic_distance": "5.0",
        }
        for index in range(2)
    ]
    write_csv(identities, identity_rows, list(identity_rows[0]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "ascent_vo_submap_screen_materialized_v1",
                "dataset": "hm3d",
                "split": "train",
                "episode_count": 2,
                "logical_case_ids": ["case_0", "case_1"],
                "chunks": [
                    {
                        "chunk_id": "c000",
                        "episode_count": 2,
                        "identity_path": str(identities),
                        "identity_sha256": sha256(identities),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "min_action_endpoints": 20,
        "min_anchor_displacement_m": 1.5,
        "overlap_threshold": 0.35,
        "low_overlap_consecutive": 3,
        "gateway_frontier_resolution_radius_m": 1.0,
        "gateway_reached_radius_m": 0.9,
        "route_min_progress_m": 0.30,
        "route_max_stagnation_actions": 30,
        "route_max_waypoint_actions": 60,
        "provisional_thresholds": False,
    }
    b2_vo = tmp_path / "b2_vo.jsonl"
    b2_submap = tmp_path / "b2_submap.jsonl"
    action_steps = {0: 2, 1: 3}
    make_vo_diagnostics(
        b2_vo,
        "B2",
        2,
        execution_order=[1, 0],
        action_steps_by_episode=action_steps,
    )
    make_submap_diagnostics(
        b2_submap,
        config,
        2,
        execution_order=[1, 0],
        action_steps_by_episode=action_steps,
        v1_1=True,
    )
    inventory = tmp_path / "inventory.csv"
    write_csv(
        inventory,
        [
            {
                "priority": 0,
                "stage": "screen",
                "condition": "B2",
                "chunk_id": "c000",
                "vo_diagnostics": b2_vo,
                "submap_diagnostics": b2_submap,
            }
        ],
        [
            "priority",
            "stage",
            "condition",
            "chunk_id",
            "vo_diagnostics",
            "submap_diagnostics",
        ],
    )
    registry = tmp_path / "registry.csv"
    write_csv(
        registry,
        [{"lane_id": 0, "inventory": inventory}],
        ["lane_id", "inventory"],
    )
    baseline = tmp_path / "arm_a.csv"
    baseline_rows = [
        {
            "logical_case_id": f"case_{index}",
            "dataset": "hm3d",
            "arm": "A",
            "scene_id": (
                f"/asset/hm3d/train/s{index}/s{index}.basis.glb"
            ),
            "action_steps": action_steps[index],
            "success": float(index == 0),
            "spl": 0.4,
        }
        for index in range(2)
    ]
    write_csv(baseline, baseline_rows, list(baseline_rows[0]))
    output = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            str(V1_1_SUMMARIZER),
            "--manifest",
            str(manifest),
            "--registry",
            str(registry),
            "--mode",
            "full",
            "--expected-dataset",
            "hm3d",
            "--expected-episodes",
            "2",
            "--source-commit",
            SOURCE,
            "--forward-checkpoint-sha256",
            FORWARD,
            "--turn-checkpoint-sha256",
            TURN,
            "--baseline-episodes-csv",
            str(baseline),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((output / "summary.json").read_text())
    assert result["technical_status"] == "PASS"
    assert result["b2_valid_episodes"] == 2
    assert result["historical_b1_matched_episodes"] == 2
    assert result["metrics"]["success_flips"]["0->1"] == 1
    assert result["mechanism"]["route_selected_count"] == 2
    assert result["mechanism"]["legacy_revisit_count"] == 0
    assert result["performance_verdict"] == "SCREEN_POSITIVE"

    v1_2_submap = tmp_path / "v1_2_submap.jsonl"
    make_submap_diagnostics(
        v1_2_submap,
        config,
        2,
        execution_order=[1, 0],
        action_steps_by_episode=action_steps,
        v1_2=True,
    )
    v1_2_inventory = tmp_path / "v1_2_inventory.csv"
    write_csv(
        v1_2_inventory,
        [
            {
                "priority": 0,
                "stage": "screen",
                "condition": "B2",
                "chunk_id": "c000",
                "vo_diagnostics": b2_vo,
                "submap_diagnostics": v1_2_submap,
            }
        ],
        [
            "priority",
            "stage",
            "condition",
            "chunk_id",
            "vo_diagnostics",
            "submap_diagnostics",
        ],
    )
    v1_2_registry = tmp_path / "v1_2_registry.csv"
    write_csv(
        v1_2_registry,
        [{"lane_id": 0, "inventory": v1_2_inventory}],
        ["lane_id", "inventory"],
    )
    v1_2_manifest = tmp_path / "v1_2_manifest.json"
    v1_2_manifest_payload = json.loads(manifest.read_text())
    v1_2_manifest_payload["split"] = "val"
    v1_2_manifest.write_text(
        json.dumps(v1_2_manifest_payload) + "\n", encoding="utf-8"
    )
    v1_2_output = tmp_path / "v1_2_summary"
    subprocess.run(
        [
            sys.executable,
            str(V1_2_SUMMARIZER),
            "--manifest",
            str(v1_2_manifest),
            "--registry",
            str(v1_2_registry),
            "--mode",
            "full",
            "--expected-dataset",
            "hm3d",
            "--expected-split",
            "val",
            "--variant-name",
            "route-leash",
            "--feature-env-key",
            "ASCENT_SUBMAP_ROUTE_GATEWAY_REPLAN_ENABLED",
            "--feature-config-key",
            "route_gateway_replan_enabled",
            "--expected-episodes",
            "2",
            "--source-commit",
            SOURCE,
            "--forward-checkpoint-sha256",
            FORWARD,
            "--turn-checkpoint-sha256",
            TURN,
            "--baseline-episodes-csv",
            str(baseline),
            "--output-dir",
            str(v1_2_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    v1_2_result = json.loads(
        (v1_2_output / "summary.json").read_text()
    )
    assert v1_2_result["technical_status"] == "PASS"
    assert v1_2_result["split"] == "val"
    assert v1_2_result["variant_name"] == "route-leash"
    assert v1_2_result["provenance"]["feature_config_key"] == (
        "route_gateway_replan_enabled"
    )
    assert v1_2_result["performance_verdict"] == "DESCRIPTIVE_ONLY"
    assert "0.02" not in v1_2_result["decision_threshold_note"]
    assert v1_2_result["mechanism_exposure"] == (
        "HANDOFF+RECOVERY+ROUTE"
    )
    assert v1_2_result["mechanism"]["handoff_created_count"] == 1
    assert v1_2_result["mechanism"]["exhaustion_recovery_count"] == 1
    assert v1_2_result["mechanism"][
        "exhaustion_recovery_scan_completed_count"
    ] == 1
    assert v1_2_result["mechanism"]["submap_event_counts"][
        "remote_route_gateway_replan_paused"
    ] == 1
    with (v1_2_output / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        v1_2_episode_rows = list(csv.DictReader(handle))
    first_event_counts = json.loads(
        v1_2_episode_rows[0]["submap_event_counts_json"]
    )
    assert first_event_counts["remote_route_gateway_replan_paused"] == 1

    original_submap_rows = [
        json.loads(line)
        for line in v1_2_submap.read_text(encoding="utf-8").splitlines()
    ]
    for feature_case, feature_value in (
        ("false", False),
        ("missing", None),
    ):
        feature_rows = json.loads(json.dumps(original_submap_rows))
        if feature_value is None:
            feature_rows[0].pop("route_gateway_replan_enabled")
        else:
            feature_rows[0][
                "route_gateway_replan_enabled"
            ] = feature_value
        feature_submap = tmp_path / f"feature_{feature_case}.jsonl"
        write_jsonl(feature_submap, feature_rows)
        feature_inventory = tmp_path / f"feature_{feature_case}.csv"
        write_csv(
            feature_inventory,
            [
                {
                    "priority": 0,
                    "stage": "screen",
                    "condition": "B2",
                    "chunk_id": "c000",
                    "vo_diagnostics": b2_vo,
                    "submap_diagnostics": feature_submap,
                }
            ],
            [
                "priority",
                "stage",
                "condition",
                "chunk_id",
                "vo_diagnostics",
                "submap_diagnostics",
            ],
        )
        feature_registry = tmp_path / f"feature_{feature_case}_registry.csv"
        write_csv(
            feature_registry,
            [{"lane_id": 0, "inventory": feature_inventory}],
            ["lane_id", "inventory"],
        )
        feature_output = tmp_path / f"feature_{feature_case}_summary"
        feature_result = subprocess.run(
            [
                sys.executable,
                str(V1_2_SUMMARIZER),
                "--manifest",
                str(v1_2_manifest),
                "--registry",
                str(feature_registry),
                "--mode",
                "full",
                "--expected-dataset",
                "hm3d",
                "--expected-split",
                "val",
                "--variant-name",
                "route-leash",
                "--feature-env-key",
                "ASCENT_SUBMAP_ROUTE_GATEWAY_REPLAN_ENABLED",
                "--feature-config-key",
                "route_gateway_replan_enabled",
                "--expected-episodes",
                "2",
                "--source-commit",
                SOURCE,
                "--forward-checkpoint-sha256",
                FORWARD,
                "--turn-checkpoint-sha256",
                TURN,
                "--baseline-episodes-csv",
                str(baseline),
                "--output-dir",
                str(feature_output),
            ],
            capture_output=True,
            text=True,
        )
        assert feature_result.returncode == 1
        feature_summary = json.loads(
            (feature_output / "summary.json").read_text()
        )
        assert feature_summary["technical_status"] == "FAIL"
        assert any(
            error.startswith(
                "submap_metadata:route_gateway_replan_enabled:"
            )
            for error in feature_summary["strict_errors"]
        )


def test_pbs_scripts_are_syntactically_valid() -> None:
    for relative in (
        "pbs/run_submap_screen_lane.sh",
        "pbs/run_submap_screen_3shared.pbs",
        "pbs/submit_submap_screen.sh",
        "pbs/run_submap_v1_1_lane.sh",
        "pbs/run_submap_v1_1_3shared.pbs",
        "pbs/submit_submap_v1_1.sh",
        "pbs/run_submap_v1_2_lane.sh",
        "pbs/run_submap_v1_2_3shared.pbs",
        "pbs/submit_submap_v1_2.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / relative)],
            check=True,
            capture_output=True,
            text=True,
        )


def test_v1_1_harness_is_b2_only_fixed_and_placement_gated() -> None:
    worker = (
        ROOT / "pbs" / "run_submap_v1_1_lane.sh"
    ).read_text()
    controller = (
        ROOT / "pbs" / "run_submap_v1_1_3shared.pbs"
    ).read_text()
    submitter = (
        ROOT / "pbs" / "submit_submap_v1_1.sh"
    ).read_text()

    assert "external/ascent_vo_submap_v1_1" in worker
    assert "condition=B2" in worker
    assert "condition=B1" not in worker
    assert "ascent_submaps.min_action_endpoints=20" in worker
    assert "ascent_submaps.min_anchor_displacement_m=1.5" in worker
    assert "ascent_submaps.overlap_threshold=0.35" in worker
    assert "ascent_submaps.low_overlap_consecutive=3" in worker
    assert "ascent_submaps.max_motion_budget_m=" not in worker
    assert "run_lane smoke" in controller
    assert controller.index("run_lane smoke") < controller.index(
        "for lane_row in"
    )
    assert "no_metric_driven_retry=1" in controller
    assert "artifacts/objectnav/submap_v1_1" in submitter
    assert "artifacts/objectnav/submap_v1/manifests" in submitter
    inventory_header = next(
        line
        for line in worker.splitlines()
        if line.startswith("printf 'priority,stage,condition,")
    )
    inventory_row = next(
        line
        for line in worker.splitlines()
        if line.startswith("  printf '%s,screen,B2,")
    )
    assert inventory_header.count(",") == inventory_row.count(",")


def test_v1_2_variant_harness_is_isolated_and_default_safe() -> None:
    worker = (ROOT / "pbs" / "run_submap_v1_2_lane.sh").read_text()
    controller = (
        ROOT / "pbs" / "run_submap_v1_2_3shared.pbs"
    ).read_text()
    submitter = (ROOT / "pbs" / "submit_submap_v1_2.sh").read_text()

    combined = worker + controller + submitter
    assert "SOURCE_ROOT=$PROJECT/external/ascent_vo_submap_v1_2" not in (
        combined
    )
    assert "SOURCE_ROOT=${ASCENT_SUBMAP_V12_SOURCE_ROOT:?required}" in (
        worker + controller
    )
    assert 'SCRIPT_SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.."' in (
        submitter
    )
    assert "RESOURCE_ROOT=$PROJECT/external/ascent" in combined
    assert "$RESOURCE_ROOT/third_party/vlfm" in worker + controller
    assert "$SOURCE_ROOT/third_party/vlfm" not in worker + controller
    assert '"$SOURCE_ROOT/model_api"' in worker
    assert '"$SOURCE_ROOT/scripts"' in worker
    assert 'cd "$RESOURCE_ROOT"' in worker
    assert '"habitat.dataset.split=$DATA_SPLIT"' in worker
    assert '"habitat_baselines.eval.split=$DATA_SPLIT"' in worker
    assert "habitat.dataset.split=train" not in worker
    assert "manifest_split" in (
        ROOT / "scripts" / "check_submap_habitat_loading.py"
    ).read_text()
    assert "manifest_split" in (
        ROOT / "scripts" / "check_submap_task_reset.py"
    ).read_text()
    assert "--expected-split" in controller
    assert '--variant-name "$VARIANT_NAME"' in controller
    assert '--feature-env-key "$FEATURE_ENV_KEY"' in controller
    assert '--feature-config-key "$FEATURE_CONFIG_KEY"' in controller
    assert 'MAIN_SCENES_ROOT=$(manifest_scenes_root "$MANIFEST")' in (
        controller
    )
    assert 'GATE_SCENES_ROOT=$(manifest_scenes_root "$GATE_MANIFEST")' in (
        controller
    )
    assert (
        'preflight_manifest main "$MANIFEST" "$DATA_SPLIT" '
        '"$MAIN_SCENES_ROOT"' in controller
    )
    assert (
        'preflight_manifest placement_gate "$GATE_MANIFEST" train '
        '\\\n    "$GATE_SCENES_ROOT"' in controller
    )
    assert 'run_lane smoke train "$GATE_SCENES_ROOT"' in controller
    assert (
        'run_lane "$MODE" "$DATA_SPLIT" "$MAIN_SCENES_ROOT"'
        in controller
    )
    assert "condition=B2" in worker and "condition=B1" not in worker
    assert '"ascent_submaps.handoff_enabled=true"' in worker
    assert (
        '"ascent_submaps.exhaustion_recovery_enabled=true"' in worker
    )
    assert "ASCENT_SUBMAP_HANDOFF_ENABLED=true" in worker
    assert "ASCENT_SUBMAP_EXHAUSTION_RECOVERY_ENABLED=true" in worker
    assert 'export "$FEATURE_ENV_KEY=true"' in worker
    for variant, feature in (
        (
            "frontier-confirm",
            "ASCENT_SUBMAP_HANDOFF_LIVE_CONFIRMATION_ENABLED",
        ),
        (
            "route-leash",
            "ASCENT_SUBMAP_ROUTE_GATEWAY_REPLAN_ENABLED",
        ),
        (
            "evidence-maturity",
            "ASCENT_SUBMAP_FRONTIER_EVIDENCE_MATURITY_ENABLED",
        ),
    ):
        assert variant in combined
        assert feature in combined
    assert "method_version=submap_v1.2" in worker
    assert "run_lane smoke" in controller
    assert controller.index("run_lane smoke") < controller.index(
        "for lane_row in"
    )
    assert "summarize_submap_v1_2_screen.py" in controller
    assert "no_metric_driven_retry=1" in controller
    assert (
        "technical_retry_policy=one_fixed_retry_per_failed_unit"
        in worker
    )
    assert (
        'if [ "${#FAILED_UNITS[@]}" -gt 0 ]; then' in worker
    )
    assert "technical_retry_exhausted=B2" in worker
    assert (
        "placement_gate_technical_retry=one_fixed_per_failed_unit"
        in controller
    )
    assert '"fixed_technical_retry_per_failed_unit": 1' in submitter
    assert "artifacts/objectnav/submap_v1_2_variants/$VARIANT_NAME" in (
        submitter + controller + worker
    )
    assert "route-exposure" in combined
    assert "route_exposure_requires_exactly_100_episodes" in submitter
    assert "CUSTOM_EXPECTED_CHUNKS" in submitter
    assert "CUSTOM_BASELINE_CSVS" in submitter
    assert "ASCENT_SUBMAP_V12_NO_QSUB" in submitter
    assert '"qsub_executed": False' in submitter
    assert os.access(ROOT / "pbs" / "submit_submap_v1_2.sh", os.X_OK)
    assert "artifacts/objectnav/submap_v1/manifests" in submitter
    assert "/scratch/e1538633/liuyi/submap_v1_2" not in (
        worker + controller + submitter
    )
    v1_2_summarizer = (
        ROOT / "scripts" / "summarize_submap_v1_2_screen.py"
    ).read_text()
    assert '"submap_event_counts_json"' in v1_2_summarizer
    assert '"DESCRIPTIVE_ONLY"' in v1_2_summarizer
    assert "sr_delta >= 0.02" not in v1_2_summarizer


def test_v1_2_variant_submitter_rejects_unknown_variant() -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "pbs" / "submit_submap_v1_2.sh"),
            "unknown-variant",
            "hm3d",
            "smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 20
    assert "invalid_variant=unknown-variant" in result.stdout


def test_frozen_calibration_selection_is_episode_disjoint() -> None:
    calibration = read_csv(
        ROOT
        / "experiments"
        / "submap_v1"
        / "hm3d_calibration30_selection_20260729.csv"
    )
    screen = read_csv(
        ROOT.parent
        / "ascent_vo_oracle"
        / "experiments"
        / "pose_factorial_train300"
        / "selection.csv"
    )
    screen_identities = {
        (row["source_content_file"], row["source_episode_index"])
        for row in screen
        if row["dataset"] == "hm3d"
    }
    calibration_identities = {
        (row["source_content_file"], row["source_episode_index"])
        for row in calibration
    }
    assert len(calibration) == 30
    assert len({row["scene_key"] for row in calibration}) == 30
    assert calibration_identities.isdisjoint(screen_identities)
    assert {
        row["selection_phase"] for row in calibration
    } == {"episode_disjoint_start_separated_3m_metadata_only"}
    assert sum(
        row["cross_floor_stratum"] == "cross_floor_ge_1p5m"
        for row in calibration
    ) == 6
    assert {
        stratum: sum(
            row["distance_stratum"] == stratum for row in calibration
        )
        for stratum in (
            "short_lt_5m",
            "medium_5_to_10m",
            "long_ge_10m",
        )
    } == {
        "short_lt_5m": 9,
        "medium_5_to_10m": 12,
        "long_ge_10m": 9,
    }

    screen_starts: dict[str, list[tuple[float, float]]] = {}
    screen_by_content: dict[str, list[dict[str, str]]] = {}
    for row in screen:
        if row["dataset"] != "hm3d":
            continue
        screen_by_content.setdefault(row["source_content_file"], []).append(
            row
        )
    for content_file, rows in screen_by_content.items():
        with gzip.open(content_file, "rt", encoding="utf-8") as handle:
            episodes = json.load(handle)["episodes"]
        for row in rows:
            start = episodes[int(row["source_episode_index"])][
                "start_position"
            ]
            screen_starts.setdefault(row["scene_key"], []).append(
                (float(start[0]), float(start[2]))
            )
    for row in calibration:
        with gzip.open(
            row["source_content_file"], "rt", encoding="utf-8"
        ) as handle:
            episode = json.load(handle)["episodes"][
                int(row["source_episode_index"])
            ]
        start = episode["start_position"]
        assert all(
            ((float(start[0]) - x) ** 2 + (float(start[2]) - z) ** 2)
            ** 0.5
            >= 3.0
            for x, z in screen_starts.get(row["scene_key"], ())
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
