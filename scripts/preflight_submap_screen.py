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
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-chunks", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    errors = []
    if manifest.get("schema") != "ascent_vo_submap_screen_materialized_v1":
        errors.append("schema")
    if manifest.get("dataset") != "hm3d":
        errors.append("dataset")
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
        missing_scenes = [
            str((scene_root / episode["scene_id"]).resolve())
            for episode in episodes
            if not (scene_root / episode["scene_id"]).is_file()
        ]
        if missing_scenes:
            errors.append(f"{chunk['chunk_id']}:scene_assets")
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
