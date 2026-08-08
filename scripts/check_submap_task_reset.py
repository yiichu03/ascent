#!/usr/bin/env python3
"""Fail-closed check of the real ASCENT ObjectNav task-reset path.

Dataset deserialization alone does not instantiate Habitat-Sim, load scene
semantic annotations, or reset ASCENT's MultiFloorMap measurement.  This
check deliberately uses a synchronous ``habitat.Env`` so Python exceptions
surface to the controller instead of being stranded in a VectorEnv worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any, Dict

import habitat
import hydra
from habitat import make_dataset
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from hydra.core.global_hydra import GlobalHydra

import ascent.run  # noqa: F401 - register ASCENT config/task extensions


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose_config(
    *,
    dataset: str,
    split: str,
    config_dir: Path,
    data_path: str,
    scene_root: Path,
    scene_dataset_config: Path,
    gpu_device_id: int,
):
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
    ):
        config = patch_config(
            hydra.compose(config_name=f"eval_ascent_{dataset}.yaml")
        )
    with read_write(config):
        config.habitat.dataset.data_path = data_path
        config.habitat.dataset.split = split
        config.habitat.dataset.scenes_dir = str(scene_root)
        config.habitat.dataset.content_scenes = ["*"]
        config.habitat.environment.iterator_options.cycle = False
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.group_by_scene = False
        config.habitat.simulator.scene_dataset = str(
            scene_dataset_config
        )
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
            gpu_device_id
        )
    return config


def check_chunk(
    *,
    dataset_name: str,
    split: str,
    chunk: Dict[str, Any],
    config_dir: Path,
    scene_root: Path,
    scene_dataset_config: Path,
    gpu_device_id: int,
) -> Dict[str, Any]:
    config = compose_config(
        dataset=dataset_name,
        split=split,
        config_dir=config_dir,
        data_path=chunk["data_path"],
        scene_root=scene_root,
        scene_dataset_config=scene_dataset_config,
        gpu_device_id=gpu_device_id,
    )
    dataset = make_dataset(
        config.habitat.dataset.type, config=config.habitat.dataset
    )
    expected = int(chunk["episode_count"])
    if len(dataset.episodes) != expected:
        raise ValueError(
            f"{chunk['chunk_id']} loaded {len(dataset.episodes)}, "
            f"expected {expected}"
        )
    expected_ids = [str(index) for index in range(expected)]
    if [episode.episode_id for episode in dataset.episodes] != expected_ids:
        raise ValueError(f"{chunk['chunk_id']} runtime ID rewrite changed")
    for episode in dataset.episodes:
        if (
            Path(episode.scene_dataset_config).resolve()
            != scene_dataset_config
        ):
            raise ValueError(
                f"{chunk['chunk_id']} episode uses unexpected "
                "scene-dataset config"
            )

    reset_rows = []
    with habitat.Env(config=config.habitat, dataset=dataset) as env:
        for expected_id in expected_ids:
            observations = env.reset()
            episode = env.current_episode
            runtime_id = str(episode.episode_id)
            if runtime_id != expected_id:
                raise ValueError(
                    f"{chunk['chunk_id']} reset order changed: "
                    f"{runtime_id} != {expected_id}"
                )
            semantic_objects = [
                obj
                for obj in env.sim.semantic_scene.objects
                if obj is not None
            ]
            if not semantic_objects:
                raise ValueError(
                    f"{chunk['chunk_id']} episode {runtime_id} loaded "
                    "without semantic objects"
                )
            reset_rows.append(
                {
                    "runtime_episode_id": runtime_id,
                    "scene_id": str(episode.scene_id),
                    "target_category": str(episode.object_category),
                    "semantic_object_count": len(semantic_objects),
                    "observation_keys": sorted(observations),
                }
            )
    return {
        "chunk_id": chunk["chunk_id"],
        "episode_count": expected,
        "reset_count": len(reset_rows),
        "min_semantic_object_count": min(
            row["semantic_object_count"] for row in reset_rows
        ),
        "max_semantic_object_count": max(
            row["semantic_object_count"] for row in reset_rows
        ),
        "episodes": reset_rows,
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "ascent_vo_submap_screen_materialized_v1":
        raise ValueError("unexpected materialized manifest schema")
    scene_root = args.scene_root.resolve()
    dataset_name = str(manifest.get("dataset"))
    if dataset_name not in {"hm3d", "mp3d"}:
        raise ValueError(f"unsupported dataset {dataset_name!r}")
    manifest_split = str(manifest.get("split"))
    if manifest_split not in {"train", "val"}:
        raise ValueError(f"unsupported split {manifest_split!r}")
    scene_dataset_config = Path(
        manifest["scene_dataset_config"]
    ).resolve()
    expected_config_hash = manifest["scene_dataset_config_sha256"]
    if (
        not scene_dataset_config.is_file()
        or sha256(scene_dataset_config) != expected_config_hash
    ):
        raise ValueError("scene-dataset config is missing or changed")

    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    results = []
    errors = []
    for chunk in manifest["chunks"]:
        try:
            results.append(
                check_chunk(
                    dataset_name=dataset_name,
                    split=manifest_split,
                    chunk=chunk,
                    config_dir=args.config_dir.resolve(),
                    scene_root=scene_root,
                    scene_dataset_config=scene_dataset_config,
                    gpu_device_id=args.gpu_device_id,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            break
        finally:
            GlobalHydra.instance().clear()

    reset_count = sum(row["reset_count"] for row in results)
    output = {
        "schema": "ascent_vo_submap_task_reset_check_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "dataset": dataset_name,
        "split": manifest_split,
        "scene_dataset_config": str(scene_dataset_config),
        "scene_dataset_config_sha256": expected_config_hash,
        "expected_episode_count": int(manifest["episode_count"]),
        "reset_episode_count": reset_count,
        "chunks": results,
        "errors": errors,
        "status": (
            "PASS"
            if not errors
            and reset_count == int(manifest["episode_count"])
            else "FAIL"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if output["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
