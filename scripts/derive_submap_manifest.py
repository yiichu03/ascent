#!/usr/bin/env python3
"""Derive relocation-safe or chunk-sharded submap-screen manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


PATH_HASH_KEYS = (
    ("data_path", "data_path_sha256"),
    ("content_file", "content_file_sha256"),
    ("identity_path", "identity_sha256"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def relative_to_chunk(path: Path, chunk_id: str) -> Path:
    parts = path.parts
    positions = [index for index, part in enumerate(parts) if part == chunk_id]
    if len(positions) != 1:
        raise ValueError(
            f"{path} must contain chunk id {chunk_id!r} exactly once"
        )
    return Path(*parts[positions[0] + 1 :])


def selected_indices(raw: str | None, count: int) -> list[int]:
    if raw is None:
        return list(range(count))
    values = [int(value) for value in raw.split(",") if value != ""]
    if not values or values != sorted(set(values)):
        raise ValueError("chunk indices must be nonempty, unique, and sorted")
    if values[0] < 0 or values[-1] >= count:
        raise ValueError("chunk index out of range")
    return values


def identity_ids(path: Path) -> Iterable[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield row["logical_case_id"]


def derive(
    source_manifest: Path,
    materialized_root: Path,
    output_manifest: Path,
    *,
    chunk_indices: str | None,
    variant: str,
) -> Dict[str, Any]:
    source_manifest = source_manifest.resolve()
    materialized_root = materialized_root.resolve()
    output_manifest = output_manifest.resolve()
    source = read_json(source_manifest)
    if source.get("schema") != "ascent_vo_submap_screen_materialized_v1":
        raise ValueError("unexpected source schema")
    source_chunks = list(source.get("chunks", []))
    indices = selected_indices(chunk_indices, len(source_chunks))
    chunks = []
    logical_ids = []
    for index in indices:
        original = dict(source_chunks[index])
        chunk_id = str(original["chunk_id"])
        for path_key, hash_key in PATH_HASH_KEYS:
            relative = relative_to_chunk(
                Path(str(original[path_key])), chunk_id
            )
            relocated = materialized_root / chunk_id / relative
            if not relocated.is_file():
                raise ValueError(f"missing relocated {path_key}: {relocated}")
            if sha256(relocated) != original[hash_key]:
                raise ValueError(f"hash mismatch for {relocated}")
            original[path_key] = str(relocated)
        ids = list(identity_ids(Path(original["identity_path"])))
        if len(ids) != int(original["episode_count"]):
            raise ValueError(f"identity count mismatch for {chunk_id}")
        logical_ids.extend(ids)
        chunks.append(original)
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("derived logical identities are not unique")

    output = dict(source)
    output.update(
        {
            "label": f"{source['label']}__{variant}",
            "episode_count": len(logical_ids),
            "chunk_count": len(chunks),
            "logical_case_ids": logical_ids,
            "chunks": chunks,
            "derivation": {
                "schema": "ascent_vo_submap_manifest_derivation_v1",
                "variant": variant,
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": sha256(source_manifest),
                "materialized_root": str(materialized_root),
                "source_chunk_indices": indices,
            },
        }
    )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": sha256(output_manifest),
        "episode_count": len(logical_ids),
        "chunk_count": len(chunks),
        "logical_case_ids": logical_ids,
        "source_chunk_indices": indices,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--chunk-indices")
    parser.add_argument("--variant", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = derive(
        args.source_manifest,
        args.materialized_root,
        args.output_manifest,
        chunk_indices=args.chunk_indices,
        variant=args.variant,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
