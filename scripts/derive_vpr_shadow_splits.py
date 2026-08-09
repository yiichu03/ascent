#!/usr/bin/env python3
"""Freeze scene-disjoint v1.4 VPR roles without reading outcomes or GT poses."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "ascent_v1_4_vpr_shadow_split_v1"
SPLIT_SALT = "ascent_v1.4_vpr_shadow_scene_split_20260809_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hm3d-manifest", type=Path, required=True)
    parser.add_argument("--mp3d-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hm3d-cal-scenes", type=int, default=25)
    parser.add_argument("--sentinel-scenes", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_rank(scene_id: str) -> str:
    return hashlib.sha256(
        f"{SPLIT_SALT}\0{scene_id}".encode("utf-8")
    ).hexdigest()


def load_source(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    logical_ids = [str(value) for value in manifest["logical_case_ids"]]
    entries: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        content_path = Path(chunk["content_file"]).resolve()
        if sha256(content_path) != chunk["content_file_sha256"]:
            raise ValueError(f"source chunk hash mismatch: {content_path}")
        with gzip.open(content_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        episodes = payload.get("episodes") or []
        if len(episodes) != int(chunk["episode_count"]):
            raise ValueError(f"source chunk count mismatch: {content_path}")
        for source_episode_index, episode in enumerate(episodes):
            entries.append(
                {
                    "logical_case_id": str(episode["episode_id"]),
                    "scene_id": str(episode["scene_id"]),
                    "source_chunk_id": str(chunk["chunk_id"]),
                    "source_episode_index": int(source_episode_index),
                }
            )
    observed_ids = [entry["logical_case_id"] for entry in entries]
    if observed_ids != logical_ids:
        raise ValueError("materialized episode order differs from source manifest")
    if len(entries) != int(manifest["episode_count"]):
        raise ValueError("source manifest episode count mismatch")
    source = {
        "dataset": str(manifest["dataset"]),
        "source_manifest_name": manifest_path.name,
        "source_manifest_sha256": sha256(manifest_path),
        "source_episode_count": len(entries),
    }
    return source, entries


def make_split(
    *,
    source: Mapping[str, Any],
    role: str,
    entries: Sequence[Mapping[str, Any]],
    rule: str,
) -> dict[str, Any]:
    scenes = sorted({str(entry["scene_id"]) for entry in entries})
    return {
        "schema": SCHEMA,
        "dataset": source["dataset"],
        "role": role,
        "episode_count": len(entries),
        "scene_count": len(scenes),
        "scene_ids": scenes,
        "selection_rule": rule,
        "selection_inputs": "scene_id_and_source_order_only",
        "outcome_fields_read": False,
        "evaluation_gt_read": False,
        "split_salt": SPLIT_SALT,
        "source_manifest_name": source["source_manifest_name"],
        "source_manifest_sha256": source["source_manifest_sha256"],
        "episodes": [dict(entry) for entry in entries],
    }


def derive(
    hm3d_manifest: Path,
    mp3d_manifest: Path,
    *,
    hm3d_cal_scenes: int,
    sentinel_scenes: int,
) -> dict[str, dict[str, Any]]:
    hm3d_source, hm3d = load_source(hm3d_manifest)
    mp3d_source, mp3d = load_source(mp3d_manifest)
    if hm3d_source["dataset"] != "hm3d" or mp3d_source["dataset"] != "mp3d":
        raise ValueError("source dataset labels must be hm3d and mp3d")

    hm3d_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in hm3d:
        hm3d_by_scene[entry["scene_id"]].append(entry)
    paired_scenes = [
        scene for scene, rows in hm3d_by_scene.items() if len(rows) == 2
    ]
    if len(paired_scenes) < hm3d_cal_scenes:
        raise ValueError("not enough two-episode HM3D scenes for exact Cal50")
    ranked_paired = sorted(paired_scenes, key=lambda scene: (scene_rank(scene), scene))
    cal_scene_ids = set(ranked_paired[:hm3d_cal_scenes])
    hm3d_cal = [entry for entry in hm3d if entry["scene_id"] in cal_scene_ids]
    hm3d_test = [entry for entry in hm3d if entry["scene_id"] not in cal_scene_ids]
    if len(hm3d_cal) != 2 * hm3d_cal_scenes:
        raise AssertionError("HM3D calibration split is not exactly two episodes/scene")
    if set(entry["scene_id"] for entry in hm3d_cal).intersection(
        entry["scene_id"] for entry in hm3d_test
    ):
        raise AssertionError("HM3D calibration/test scene leakage")

    sentinel_scene_ids = ranked_paired[:sentinel_scenes]
    sentinel = []
    for scene in sentinel_scene_ids:
        sentinel.append(hm3d_by_scene[scene][0])
    sentinel_ids = {row["logical_case_id"] for row in sentinel}
    sentinel = [row for row in hm3d if row["logical_case_id"] in sentinel_ids]

    outputs = {
        "hm3d_sentinel5.json": make_split(
            source=hm3d_source,
            role="engineering_sentinel",
            entries=sentinel,
            rule=(
                "first source-order episode from each of the first five "
                "hash-ranked HM3D-Cal50 scenes"
            ),
        ),
        "hm3d_cal50.json": make_split(
            source=hm3d_source,
            role="threshold_calibration_only",
            entries=hm3d_cal,
            rule=(
                "all episodes from the first 25 hash-ranked scenes having "
                "exactly two source episodes"
            ),
        ),
        "hm3d_test100.json": make_split(
            source=hm3d_source,
            role="locked_in_distribution_test",
            entries=hm3d_test,
            rule="scene-disjoint complement of HM3D-Cal50 in frozen HM3D train-150",
        ),
        "mp3d_test150.json": make_split(
            source=mp3d_source,
            role="locked_cross_dataset_transfer_test",
            entries=mp3d,
            rule="all episodes in frozen MP3D train-150; no threshold calibration",
        ),
    }
    expected = {
        "hm3d_sentinel5.json": (sentinel_scenes, sentinel_scenes),
        "hm3d_cal50.json": (2 * hm3d_cal_scenes, hm3d_cal_scenes),
        "hm3d_test100.json": (
            len(hm3d) - 2 * hm3d_cal_scenes,
            len(hm3d_by_scene) - hm3d_cal_scenes,
        ),
        "mp3d_test150.json": (len(mp3d), len({x["scene_id"] for x in mp3d})),
    }
    for name, payload in outputs.items():
        episodes, scenes = expected[name]
        if payload["episode_count"] != episodes or payload["scene_count"] != scenes:
            raise AssertionError(f"unexpected split dimensions for {name}")
    return outputs


def encoded(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_or_verify(
    output_dir: Path,
    outputs: Mapping[str, Mapping[str, Any]],
    *,
    verify_only: bool,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if not verify_only:
        output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, payload in outputs.items():
        raw = encoded(payload)
        digest = hashlib.sha256(raw).hexdigest()
        path = output_dir / name
        if verify_only:
            if not path.is_file() or path.read_bytes() != raw:
                raise ValueError(f"frozen split differs: {path}")
        else:
            with path.open("xb") as handle:
                handle.write(raw)
        hashes[name] = digest
    registry = {
        "schema": "ascent_v1_4_vpr_shadow_split_registry_v1",
        "split_salt": SPLIT_SALT,
        "files": hashes,
    }
    registry_path = output_dir / "split_registry.json"
    registry_raw = encoded(registry)
    if verify_only:
        if not registry_path.is_file() or registry_path.read_bytes() != registry_raw:
            raise ValueError(f"frozen registry differs: {registry_path}")
    else:
        with registry_path.open("xb") as handle:
            handle.write(registry_raw)
    return registry


def main() -> int:
    args = parse_args()
    outputs = derive(
        args.hm3d_manifest,
        args.mp3d_manifest,
        hm3d_cal_scenes=args.hm3d_cal_scenes,
        sentinel_scenes=args.sentinel_scenes,
    )
    registry = write_or_verify(
        args.output_dir, outputs, verify_only=args.verify_only
    )
    print(json.dumps(registry, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
