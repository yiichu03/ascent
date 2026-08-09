#!/usr/bin/env python3
"""Materialize one frozen VPR-shadow role without reading outcomes or GT tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from materialize_submap_screen import materialize


SCHEMA = "ascent_v1_4_vpr_shadow_prepared_input_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty VPR-shadow selection")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("source selection rows do not share one schema")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_chunk_by_id(source: Mapping[str, Any]) -> dict[str, str]:
    logical_ids = [str(value) for value in source["logical_case_ids"]]
    output: dict[str, str] = {}
    offset = 0
    for chunk in source["chunks"]:
        count = int(chunk["episode_count"])
        for logical_id in logical_ids[offset : offset + count]:
            if logical_id in output:
                raise ValueError(f"duplicate source identity {logical_id}")
            output[logical_id] = str(chunk["chunk_id"])
        offset += count
    if offset != len(logical_ids):
        raise ValueError("source chunk counts do not cover logical identities")
    return output


def _verify_prepared(output_root: Path) -> dict[str, Any]:
    audit_path = output_root / "preparation_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != SCHEMA:
        raise ValueError("unexpected prepared-input schema")
    checks = {
        output_root / "selection.csv": audit["selection_sha256"],
        output_root / "materialized/chunk_manifest.json": (
            audit["materialized_manifest_sha256"]
        ),
        Path(audit["split_manifest"]): audit["split_manifest_sha256"],
        Path(audit["source_manifest"]): audit["source_manifest_sha256"],
    }
    for path, expected in checks.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"prepared-input hash mismatch: {path}")
    manifest = json.loads(
        (output_root / "materialized/chunk_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        int(manifest["episode_count"]) != int(audit["episode_count"])
        or int(manifest["chunk_count"]) != int(audit["chunk_count"])
    ):
        raise ValueError("prepared-input dimensions drifted")
    for chunk in manifest["chunks"]:
        for path_key, hash_key in (
            ("data_path", "data_path_sha256"),
            ("content_file", "content_file_sha256"),
            ("identity_path", "identity_sha256"),
        ):
            path = Path(chunk[path_key])
            if not path.is_file() or sha256(path) != chunk[hash_key]:
                raise ValueError(f"materialized chunk drift: {path}")
    return audit


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    if args.verify_only:
        return _verify_prepared(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)

    source_path = args.source_manifest.resolve()
    split_path = args.split_manifest.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("schema") != "ascent_v1_4_vpr_shadow_split_v1":
        raise ValueError("unexpected VPR-shadow split schema")
    if split.get("source_manifest_sha256") != sha256(source_path):
        raise ValueError("split is not bound to the source manifest")
    if split.get("dataset") != source.get("dataset"):
        raise ValueError("split/source dataset mismatch")
    if int(split.get("episode_count", -1)) < 1:
        raise ValueError("empty VPR-shadow role")
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")

    selection_path = Path(source["selection_path"]).resolve()
    if sha256(selection_path) != source["selection_sha256"]:
        raise ValueError("source selection hash mismatch")
    source_rows = read_csv(selection_path)
    by_id: dict[str, dict[str, str]] = {}
    for row in source_rows:
        logical_id = str(row.get("logical_case_id", ""))
        if logical_id in by_id:
            raise ValueError(f"duplicate selection identity {logical_id}")
        by_id[logical_id] = row

    source_chunk_by_id = _source_chunk_by_id(source)
    selected: list[dict[str, str]] = []
    for split_row in split["episodes"]:
        logical_id = str(split_row["logical_case_id"])
        row = by_id.get(logical_id)
        if row is None:
            raise ValueError(f"split identity missing from source selection: {logical_id}")
        if row.get("dataset") != split["dataset"]:
            raise ValueError(f"selection dataset mismatch: {logical_id}")
        if str(row.get("scene_id")) != str(split_row["scene_id"]):
            raise ValueError(f"selection scene mismatch: {logical_id}")
        if source_chunk_by_id.get(logical_id) != str(split_row["source_chunk_id"]):
            raise ValueError(f"selection source chunk mismatch: {logical_id}")
        selected.append(dict(row))
    expected_ids = [str(row["logical_case_id"]) for row in split["episodes"]]
    if [row["logical_case_id"] for row in selected] != expected_ids:
        raise AssertionError("prepared selection order differs from frozen split")

    output_root.mkdir(parents=True)
    filtered_selection = output_root / "selection.csv"
    write_csv(filtered_selection, selected)
    materialized_root = output_root / "materialized"
    result = materialize(
        SimpleNamespace(
            selection=filtered_selection,
            output_root=materialized_root,
            scene_dataset_config=Path(source["scene_dataset_config"]),
            dataset=split["dataset"],
            episodes=len(selected),
            start_index=0,
            chunk_size=args.chunk_size,
            label=args.label,
        )
    )
    materialized_manifest = materialized_root / "chunk_manifest.json"
    audit = {
        "schema": SCHEMA,
        "dataset": split["dataset"],
        "role": split["role"],
        "selection_inputs": "frozen_split_identity_and_source_dataset_rows_only",
        "outcome_fields_read": False,
        "evaluation_gt_trajectory_read": False,
        "split_manifest": str(split_path),
        "split_manifest_sha256": sha256(split_path),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256(source_path),
        "source_selection_sha256": source["selection_sha256"],
        "selection_sha256": sha256(filtered_selection),
        "materialized_manifest": str(materialized_manifest),
        "materialized_manifest_sha256": sha256(materialized_manifest),
        "episode_count": len(selected),
        "chunk_count": int(result["chunk_count"]),
        "logical_case_ids": expected_ids,
    }
    with (output_root / "preparation_audit.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = prepare(args)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
