#!/usr/bin/env python3
"""Strict, read-only preflight for ASCENT-VO submap datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument(
        "--expected-dataset", choices=("hm3d", "mp3d")
    )
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-chunks", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    errors = []
    if manifest.get("schema") != "ascent_vo_submap_screen_materialized_v1":
        errors.append("schema")
    dataset = manifest.get("dataset")
    if dataset not in {"hm3d", "mp3d"}:
        errors.append("dataset")
    elif (
        args.expected_dataset is not None
        and dataset != args.expected_dataset
    ):
        errors.append("expected_dataset")
    if manifest.get("split") != "train":
        errors.append("split")
    if int(manifest.get("episode_count", -1)) != args.expected_episodes:
        errors.append("episode_count")
    chunks = manifest.get("chunks", [])
    if len(chunks) != args.expected_chunks:
        errors.append("chunk_count")
    logical_ids = list(manifest.get("logical_case_ids", []))
    if (
        len(logical_ids) != args.expected_episodes
        or len(logical_ids) != len(set(logical_ids))
    ):
        errors.append("logical_identity")
    scene_root = args.scene_root.resolve()
    scene_dataset_raw = manifest.get("scene_dataset_config")
    scene_dataset_config = (
        Path(scene_dataset_raw).resolve()
        if isinstance(scene_dataset_raw, str) and scene_dataset_raw
        else None
    )
    if (
        scene_dataset_config is None
        or not Path(scene_dataset_raw).is_absolute()
        or not scene_dataset_config.is_file()
    ):
        errors.append("scene_dataset_config")
    elif (
        sha256(scene_dataset_config)
        != manifest.get("scene_dataset_config_sha256")
    ):
        errors.append("scene_dataset_config_hash")
    elif scene_dataset_config.parent != (
        scene_root / str(dataset)
    ).resolve():
        errors.append("scene_dataset_config_root")
    validated = []
    total = 0
    for chunk in chunks:
        chunk_errors = []
        for path_key, hash_key in (
            ("data_path", "data_path_sha256"),
            ("content_file", "content_file_sha256"),
            ("identity_path", "identity_sha256"),
        ):
            path = Path(chunk[path_key]).resolve()
            if not path.is_file():
                chunk_errors.append(f"missing_{path_key}")
            elif sha256(path) != chunk[hash_key]:
                chunk_errors.append(f"hash_{path_key}")
        if chunk_errors:
            errors.extend(
                f"{chunk['chunk_id']}:{item}" for item in chunk_errors
            )
            continue
        with gzip.open(
            chunk["content_file"], "rt", encoding="utf-8"
        ) as handle:
            episodes = json.load(handle)["episodes"]
        with Path(chunk["identity_path"]).open(
            newline="", encoding="utf-8"
        ) as handle:
            identities = list(csv.DictReader(handle))
        expected = int(chunk["episode_count"])
        if len(episodes) != expected or len(identities) != expected:
            errors.append(f"{chunk['chunk_id']}:coverage")
        runtime_ids = [row["runtime_episode_id"] for row in identities]
        if runtime_ids != [str(index) for index in range(expected)]:
            errors.append(f"{chunk['chunk_id']}:runtime_ids")
        declared_ids = [str(episode["episode_id"]) for episode in episodes]
        if declared_ids != [
            row["logical_case_id"] for row in identities
        ]:
            errors.append(f"{chunk['chunk_id']}:declared_identity")
        if scene_dataset_config is not None and any(
            Path(str(episode.get("scene_dataset_config", ""))).resolve()
            != scene_dataset_config
            for episode in episodes
        ):
            errors.append(
                f"{chunk['chunk_id']}:episode_scene_dataset_config"
            )
        scene_paths = [
            (scene_root / episode["scene_id"]).resolve()
            for episode in episodes
        ]
        if any(not path.is_file() for path in scene_paths):
            errors.append(f"{chunk['chunk_id']}:scene_assets")
        if dataset == "hm3d":
            companion_paths = [
                path.with_name(path.name.replace(".basis.glb", suffix))
                for path in scene_paths
                for suffix in (
                    ".basis.navmesh",
                    ".semantic.glb",
                    ".semantic.txt",
                )
            ]
        else:
            companion_paths = [
                companion
                for path in scene_paths
                for companion in (
                    path.with_suffix(".navmesh"),
                    path.with_name(f"{path.stem}_semantic.ply"),
                    path.with_suffix(".house"),
                )
            ]
        if any(not path.is_file() for path in companion_paths):
            errors.append(f"{chunk['chunk_id']}:companion_assets")
        total += expected
        validated.append(
            {
                "chunk_id": chunk["chunk_id"],
                "episode_count": expected,
                "scene_count": len(
                    {str(episode["scene_id"]) for episode in episodes}
                ),
                "status": "PASS" if not chunk_errors else "FAIL",
            }
        )
    if total != args.expected_episodes:
        errors.append("total")
    output = {
        "schema": "ascent_vo_submap_screen_preflight_v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "dataset": dataset,
        "scene_dataset_config": (
            str(scene_dataset_config)
            if scene_dataset_config is not None
            else None
        ),
        "scene_dataset_config_sha256": manifest.get(
            "scene_dataset_config_sha256"
        ),
        "expected_episodes": args.expected_episodes,
        "expected_chunks": args.expected_chunks,
        "validated_episode_count": total,
        "chunks": validated,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
