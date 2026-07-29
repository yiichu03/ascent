#!/usr/bin/env python3
"""Materialize fixed one-copy ASCENT episodes for submap B1/B2 runs."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


IDENTITY_FIELDS = (
    "runtime_episode_id",
    "logical_case_id",
    "dataset",
    "episode_seed",
    "source_episode_id",
    "scene_id",
    "target_category",
    "geodesic_distance",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_json(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def write_gzip_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with path.open("wb") as stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=stream, mtime=0
        ) as gzip_stream:
            gzip_stream.write(raw)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_selection(
    path: Path, *, dataset: str, start_index: int, episodes: int
) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("dataset") == dataset
        ]
    rows.sort(key=lambda row: int(row["dataset_case_index"]))
    selected = rows[start_index : start_index + episodes]
    if len(selected) != episodes:
        raise ValueError(
            f"requested {episodes} {dataset} rows at {start_index}, "
            f"found {len(selected)}"
        )
    logical_ids = [row["logical_case_id"] for row in selected]
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("logical_case_id is not unique")
    return selected


def verified_episode(
    row: Mapping[str, str],
    payload: Mapping[str, Any],
    verified_hashes: Dict[str, str],
) -> Dict[str, Any]:
    source = Path(row["source_content_file"]).resolve()
    expected_hash = row["source_content_sha256"]
    actual_hash = verified_hashes.get(str(source))
    if actual_hash is None:
        actual_hash = sha256(source)
        verified_hashes[str(source)] = actual_hash
    if actual_hash != expected_hash:
        raise ValueError(f"source hash mismatch for {source}")
    episode = payload["episodes"][int(row["source_episode_index"])]
    checks = {
        "episode_id": row["source_episode_id"],
        "scene_id": row["scene_id"],
        "object_category": row["target_category"],
    }
    for key, expected in checks.items():
        if str(episode.get(key)) != str(expected):
            raise ValueError(
                f"{row['logical_case_id']} {key} mismatch: "
                f"{episode.get(key)!r} != {expected!r}"
            )
    distance = float((episode.get("info") or {})["geodesic_distance"])
    if abs(distance - float(row["geodesic_distance"])) > 1e-7:
        raise ValueError(
            f"{row['logical_case_id']} geodesic distance mismatch"
        )
    return dict(episode)


def materialize_chunk(
    rows: Sequence[Mapping[str, str]],
    *,
    output_root: Path,
    chunk_id: str,
    verified_hashes: Dict[str, str],
) -> Dict[str, Any]:
    dataset = rows[0]["dataset"]
    chunk_root = output_root / chunk_id
    chunk_root.mkdir(parents=True, exist_ok=False)
    data_dir = chunk_root / dataset / "v1" / "train"
    content_dir = data_dir / "content"
    content_dir.mkdir(parents=True)

    root_payload = read_gzip_json(Path(rows[0]["source_root_file"]))
    root_payload["episodes"] = []
    root_payload["goals_by_category"] = {}
    data_path = data_dir / "train.json.gz"
    write_gzip_json(data_path, root_payload)

    combined = copy.deepcopy(root_payload)
    combined_episodes: List[Dict[str, Any]] = []
    combined_goals: Dict[str, Any] = {}
    identity_rows: List[Dict[str, Any]] = []
    rows_by_source: Dict[str, List[Mapping[str, str]]] = defaultdict(
        list
    )
    for row in rows:
        rows_by_source[row["source_content_file"]].append(row)

    runtime_id = 0
    for source_path in sorted(rows_by_source):
        source = read_gzip_json(Path(source_path))
        for key, value in source.items():
            if key in {"episodes", "goals_by_category"}:
                continue
            if key in combined and combined[key] != value:
                raise ValueError(
                    f"inconsistent metadata {key!r} in {source_path}"
                )
            combined[key] = copy.deepcopy(value)
        goals = source.get("goals_by_category", {})
        if not isinstance(goals, Mapping):
            raise ValueError(f"invalid goals_by_category in {source_path}")
        for key, value in goals.items():
            if key in combined_goals and combined_goals[key] != value:
                raise ValueError(f"conflicting goal definition {key!r}")
            combined_goals[key] = copy.deepcopy(value)

        for row in sorted(
            rows_by_source[source_path],
            key=lambda item: int(item["dataset_case_index"]),
        ):
            episode = copy.deepcopy(
                verified_episode(row, source, verified_hashes)
            )
            episode["episode_id"] = row["logical_case_id"]
            combined_episodes.append(episode)
            identity_rows.append(
                {
                    "runtime_episode_id": runtime_id,
                    "logical_case_id": row["logical_case_id"],
                    "dataset": dataset,
                    "episode_seed": row["episode_seed"],
                    "source_episode_id": row["source_episode_id"],
                    "scene_id": row["scene_id"],
                    "target_category": row["target_category"],
                    "geodesic_distance": row["geodesic_distance"],
                }
            )
            runtime_id += 1

    combined["episodes"] = combined_episodes
    combined["goals_by_category"] = combined_goals
    content_path = content_dir / "submap_screen_chunk.json.gz"
    write_gzip_json(content_path, combined)
    identity_path = chunk_root / "episode_identity.csv"
    write_csv(identity_path, identity_rows, IDENTITY_FIELDS)
    return {
        "chunk_id": chunk_id,
        "dataset": dataset,
        "split": "train",
        "episode_count": len(rows),
        "data_path": str(data_path.resolve()),
        "data_path_sha256": sha256(data_path),
        "content_file": str(content_path.resolve()),
        "content_file_sha256": sha256(content_path),
        "identity_path": str(identity_path.resolve()),
        "identity_sha256": sha256(identity_path),
    }


def materialize(args: argparse.Namespace) -> Dict[str, Any]:
    selection = args.selection.resolve()
    rows = load_selection(
        selection,
        dataset=args.dataset,
        start_index=args.start_index,
        episodes=args.episodes,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    verified_hashes: Dict[str, str] = {}
    chunks = []
    for chunk_index, offset in enumerate(
        range(0, len(rows), args.chunk_size)
    ):
        chunk_rows = rows[offset : offset + args.chunk_size]
        chunk_id = f"{args.label}_c{chunk_index:03d}"
        chunks.append(
            materialize_chunk(
                chunk_rows,
                output_root=output_root,
                chunk_id=chunk_id,
                verified_hashes=verified_hashes,
            )
        )
    manifest = {
        "schema": "ascent_vo_submap_screen_materialized_v1",
        "dataset": args.dataset,
        "split": "train",
        "label": args.label,
        "selection_path": str(selection),
        "selection_sha256": sha256(selection),
        "selection_start_index": args.start_index,
        "episode_count": len(rows),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "logical_case_ids": [row["logical_case_id"] for row in rows],
        "chunks": chunks,
    }
    manifest_path = output_root / "chunk_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("hm3d", "mp3d"), required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--label", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.chunk_size < 1 or args.start_index < 0:
        raise SystemExit("episodes/chunk-size must be positive; start-index nonnegative")
    result = materialize(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
