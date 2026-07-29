from __future__ import annotations

import csv
import gzip
import json
from types import SimpleNamespace

from scripts.materialize_submap_screen import (
    IDENTITY_FIELDS,
    materialize,
    read_gzip_json,
    sha256,
    write_gzip_json,
)


def test_materializer_is_one_copy_deterministic_and_identity_bound(
    tmp_path,
) -> None:
    source_root = tmp_path / "source_root.json.gz"
    source_content = tmp_path / "source_content.json.gz"
    base = {"episodes": [], "goals_by_category": {}, "config": "synthetic"}
    episode = {
        "episode_id": "7",
        "scene_id": "hm3d/train/scene/scene.basis.glb",
        "object_category": "chair",
        "info": {"geodesic_distance": 4.25},
    }
    write_gzip_json(source_root, base)
    write_gzip_json(
        source_content,
        {
            **base,
            "episodes": [episode],
            "goals_by_category": {"scene.glb_chair": [{"position": [0, 0, 0]}]},
        },
    )
    selection = tmp_path / "selection.csv"
    fields = [
        "dataset_case_index",
        "logical_case_id",
        "dataset",
        "source_root_file",
        "source_content_file",
        "source_content_sha256",
        "source_episode_index",
        "source_episode_id",
        "scene_id",
        "target_category",
        "geodesic_distance",
        "episode_seed",
    ]
    with selection.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset_case_index": 0,
                "logical_case_id": "hm3d_train_0000",
                "dataset": "hm3d",
                "source_root_file": source_root,
                "source_content_file": source_content,
                "source_content_sha256": sha256(source_content),
                "source_episode_index": 0,
                "source_episode_id": 7,
                "scene_id": episode["scene_id"],
                "target_category": "chair",
                "geodesic_distance": 4.25,
                "episode_seed": 123,
            }
        )

    output = tmp_path / "materialized"
    result = materialize(
        SimpleNamespace(
            selection=selection,
            output_root=output,
            dataset="hm3d",
            start_index=0,
            episodes=1,
            chunk_size=1,
            label="smoke",
        )
    )
    assert result["episode_count"] == 1
    assert result["chunk_count"] == 1
    chunk = result["chunks"][0]
    content = read_gzip_json(chunk["content_file"])
    assert len(content["episodes"]) == 1
    assert content["episodes"][0]["episode_id"] == "hm3d_train_0000"
    assert content["goals_by_category"]
    with open(chunk["identity_path"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == IDENTITY_FIELDS
    assert rows[0]["runtime_episode_id"] == "0"
    assert rows[0]["logical_case_id"] == "hm3d_train_0000"

    raw_first = gzip.open(chunk["content_file"], "rb").read()
    assert raw_first
