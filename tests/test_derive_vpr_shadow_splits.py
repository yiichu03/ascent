from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/derive_vpr_shadow_splits.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, dataset: str, scene_counts: list[int]) -> Path:
    chunks = []
    logical_ids = []
    offset = 0
    for chunk_index, counts in enumerate((scene_counts[: len(scene_counts) // 2], scene_counts[len(scene_counts) // 2 :])):
        episodes = []
        for local_scene_index, count in enumerate(counts):
            scene_index = local_scene_index + (0 if chunk_index == 0 else len(scene_counts) // 2)
            for _ in range(count):
                episode_id = f"{dataset}_train_{offset:04d}"
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "scene_id": f"{dataset}/scene_{scene_index:03d}/scene.glb",
                    }
                )
                logical_ids.append(episode_id)
                offset += 1
        content = tmp_path / f"{dataset}_chunk_{chunk_index}.json.gz"
        with content.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                stream.write(json.dumps({"episodes": episodes}).encode())
        chunks.append(
            {
                "chunk_id": f"{dataset}_c{chunk_index}",
                "content_file": str(content),
                "content_file_sha256": _sha(content),
                "episode_count": len(episodes),
            }
        )
    manifest = tmp_path / f"{dataset}_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "episode_count": len(logical_ids),
                "logical_case_ids": logical_ids,
                "chunks": chunks,
            }
        )
    )
    return manifest


def test_split_is_scene_disjoint_deterministic_and_verifiable(tmp_path: Path) -> None:
    hm3d = _source(tmp_path, "hm3d", [2] * 4 + [1] * 2)
    mp3d = _source(tmp_path, "mp3d", [3] * 3 + [2] * 2)
    output = tmp_path / "splits"
    command = [
        sys.executable,
        str(SCRIPT),
        "--hm3d-manifest",
        str(hm3d),
        "--mp3d-manifest",
        str(mp3d),
        "--output-dir",
        str(output),
        "--hm3d-cal-scenes",
        "2",
        "--sentinel-scenes",
        "2",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run(
        [*command, "--verify-only"], check=True, capture_output=True, text=True
    )

    cal = json.loads((output / "hm3d_cal50.json").read_text())
    test = json.loads((output / "hm3d_test100.json").read_text())
    sentinel = json.loads((output / "hm3d_sentinel5.json").read_text())
    transfer = json.loads((output / "mp3d_test150.json").read_text())
    assert (cal["episode_count"], cal["scene_count"]) == (4, 2)
    assert set(cal["scene_ids"]).isdisjoint(test["scene_ids"])
    assert (sentinel["episode_count"], sentinel["scene_count"]) == (2, 2)
    assert transfer["episode_count"] == 13
    assert all(not payload["evaluation_gt_read"] for payload in (cal, test, transfer))

    registry = json.loads((output / "split_registry.json").read_text())
    for name, digest in registry["files"].items():
        assert _sha(output / name) == digest
