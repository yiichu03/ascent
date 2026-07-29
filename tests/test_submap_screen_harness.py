from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
CALIBRATOR = ROOT / "scripts" / "calibrate_submap_thresholds.py"
SUMMARIZER = ROOT / "scripts" / "summarize_submap_screen.py"
MATERIALIZER = ROOT / "scripts" / "materialize_submap_screen.py"
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


def test_unified_controller_binds_dataset_and_real_reset_contracts() -> None:
    checker = (
        ROOT / "scripts" / "check_submap_task_reset.py"
    ).read_text(encoding="utf-8")
    controller = (
        ROOT / "pbs" / "run_submap_screen_3shared.pbs"
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
    path: Path, condition: str, episode_count: int
) -> None:
    rows = [vo_metadata()]
    for episode in range(episode_count):
        for step in (1, 2):
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
                "action_steps": 2,
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
) -> None:
    rows = [
        {
            "record_type": "submap_run_metadata",
            "ascent_source_commit": SOURCE,
            "pose_source": "zhao_rgbd_2021",
            "policy_gt_isolation": True,
            "config": {"enabled": True, **dict(config)},
        }
    ]
    for episode in range(episode_count):
        rows.append(
            {
                "record_type": "submap_episode_reset",
                "episode_sequence": episode,
            }
        )
        for step in (0, 1):
            rows.append(
                {
                    "record_type": "submap_action_endpoint",
                    "episode_sequence": episode,
                    "action_step": step,
                }
            )
        rows.append(
            {
                "record_type": "submap_event",
                "episode_sequence": episode,
                "event": "submap_split",
            }
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
    make_vo_diagnostics(b1_vo, "B1", 2)
    make_vo_diagnostics(b2_vo, "B2", 2)
    make_submap_diagnostics(b2_submap, config, 2)
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


def test_pbs_scripts_are_syntactically_valid() -> None:
    for relative in (
        "pbs/run_submap_screen_lane.sh",
        "pbs/run_submap_screen_3shared.pbs",
        "pbs/submit_submap_screen.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / relative)],
            check=True,
            capture_output=True,
            text=True,
        )


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
