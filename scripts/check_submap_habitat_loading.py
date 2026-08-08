#!/usr/bin/env python3
"""Validate materialized submap-screen episodes through Habitat's loader."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Dict

import hydra
from habitat import make_dataset
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from hydra.core.global_hydra import GlobalHydra

import ascent.run  # noqa: F401 - register ASCENT's Hydra search path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    if manifest.get("schema") != "ascent_vo_submap_screen_materialized_v1":
        raise ValueError("unexpected materialized manifest schema")
    manifest_split = str(manifest.get("split"))
    if manifest_split not in {"train", "val"}:
        raise ValueError(f"unsupported split {manifest_split!r}")
    scene_root = args.scene_root.resolve()
    scene_dataset_config = Path(
        manifest["scene_dataset_config"]
    ).resolve()
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    results = []
    for chunk in manifest["chunks"]:
        dataset = chunk["dataset"]
        GlobalHydra.instance().clear()
        with hydra.initialize_config_dir(
            version_base=None,
            config_dir=str(args.config_dir.resolve()),
        ):
            config = patch_config(
                hydra.compose(
                    config_name=f"eval_ascent_{dataset}.yaml"
                )
            )
        with read_write(config):
            config.habitat.dataset.data_path = chunk["data_path"]
            config.habitat.dataset.split = manifest_split
            config.habitat.dataset.scenes_dir = str(scene_root)
            config.habitat.dataset.content_scenes = ["*"]
        loaded = make_dataset(
            config.habitat.dataset.type,
            config=config.habitat.dataset,
        )
        expected_count = int(chunk["episode_count"])
        if len(loaded.episodes) != expected_count:
            raise ValueError(
                f"{chunk['chunk_id']} loaded {len(loaded.episodes)}, "
                f"expected {expected_count}"
            )
        runtime_ids = [episode.episode_id for episode in loaded.episodes]
        if runtime_ids != [str(index) for index in range(expected_count)]:
            raise ValueError(
                f"{chunk['chunk_id']} runtime ID rewrite changed"
            )
        if any(
            Path(episode.scene_dataset_config).resolve()
            != scene_dataset_config
            for episode in loaded.episodes
        ):
            raise ValueError(
                f"{chunk['chunk_id']} scene-dataset config changed"
            )
        with Path(chunk["identity_path"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            identities = list(csv.DictReader(handle))
        if [row["runtime_episode_id"] for row in identities] != runtime_ids:
            raise ValueError(
                f"{chunk['chunk_id']} identity/runtime IDs disagree"
            )
        with gzip.open(
            chunk["content_file"], "rt", encoding="utf-8"
        ) as handle:
            declared = json.load(handle)["episodes"]
        for index, (runtime, source) in enumerate(
            zip(loaded.episodes, declared)
        ):
            expected_scene = str(
                (scene_root / source["scene_id"]).resolve()
            )
            if (
                str(Path(runtime.scene_id).resolve()) != expected_scene
                or runtime.object_category != source["object_category"]
            ):
                raise ValueError(
                    f"{chunk['chunk_id']} order mismatch at {index}"
                )
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "episode_count": expected_count,
                "first_runtime_episode_id": runtime_ids[0],
                "last_runtime_episode_id": runtime_ids[-1],
                "status": "PASS",
            }
        )
    GlobalHydra.instance().clear()
    output = {
        "schema": "ascent_vo_submap_habitat_loading_v1",
        "chunk_count": len(results),
        "episode_count": sum(row["episode_count"] for row in results),
        "split": manifest_split,
        "scene_dataset_config": str(scene_dataset_config),
        "chunks": results,
        "status": "PASS",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
