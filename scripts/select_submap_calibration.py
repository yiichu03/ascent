#!/usr/bin/env python3
"""Select deterministic HM3D train calibration episodes disjoint by start."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


FIELDS = (
    "manifest_version",
    "selection_seed",
    "global_case_index",
    "dataset_case_index",
    "logical_case_id",
    "dataset",
    "split",
    "chunk_id",
    "source_root_file",
    "source_content_file",
    "source_content_sha256",
    "source_episode_index",
    "source_episode_id",
    "scene_id",
    "scene_key",
    "target_category",
    "geodesic_distance",
    "euclidean_distance",
    "distance_stratum",
    "start_height",
    "target_viewpoint_height",
    "vertical_delta",
    "cross_floor_stratum",
    "selection_phase",
    "episode_seed",
    "arm_order",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(seed: str, *values: object) -> str:
    text = ":".join([seed, *(str(value) for value in values)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_gzip(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def scene_key(scene_id: str) -> str:
    name = Path(scene_id.replace("\\", "/")).name
    for suffix in (".basis.glb", ".glb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def goal_heights(payload: Mapping[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for goals in (payload.get("goals_by_category") or {}).values():
        if not isinstance(goals, list):
            continue
        for goal in goals:
            if not isinstance(goal, Mapping) or goal.get("object_id") is None:
                continue
            heights = []
            for view in goal.get("view_points") or ():
                position = (
                    (view.get("agent_state") or {}).get("position")
                    if isinstance(view, Mapping)
                    else None
                )
                if isinstance(position, list) and len(position) >= 2:
                    heights.append(float(position[1]))
            if not heights:
                position = goal.get("position")
                if isinstance(position, list) and len(position) >= 2:
                    heights.append(float(position[1]))
            if heights:
                output[str(goal["object_id"])] = float(
                    statistics.median(heights)
                )
    return output


def target_height(
    episode: Mapping[str, Any], heights: Mapping[str, float]
) -> Optional[float]:
    info = episode.get("info") or {}
    direct = info.get("best_viewpoint_position")
    if isinstance(direct, list) and len(direct) >= 2:
        return float(direct[1])
    object_id = info.get("closest_goal_object_id")
    return None if object_id is None else heights.get(str(object_id))


def classify_distance(distance: float) -> str:
    if distance < 5.0:
        return "short_lt_5m"
    if distance < 10.0:
        return "medium_5_to_10m"
    return "long_ge_10m"


def candidate_from_episode(
    *,
    heights: Mapping[str, float],
    content_file: Path,
    episode_index: int,
    episode: Mapping[str, Any],
    root_file: Path,
    seed: str,
    forbidden_starts: Iterable[tuple[float, float]],
) -> Optional[Dict[str, Any]]:
    info = episode.get("info") or {}
    start = episode.get("start_position")
    distance = info.get("geodesic_distance")
    if (
        not isinstance(start, list)
        or len(start) < 2
        or not isinstance(distance, (int, float))
    ):
        return None
    scene = str(episode.get("scene_id") or "")
    category = str(episode.get("object_category") or "")
    episode_id_value = episode.get("episode_id")
    episode_id = (
        "" if episode_id_value is None else str(episode_id_value)
    )
    if not scene or not category or episode_id_value is None:
        return None
    start_xz = (float(start[0]), float(start[2]))
    if any(
        math.hypot(
            start_xz[0] - forbidden[0],
            start_xz[1] - forbidden[1],
        )
        < 3.0
        for forbidden in forbidden_starts
    ):
        return None
    target = target_height(episode, heights)
    start_y = float(start[1])
    delta = None if target is None else float(target - start_y)
    if delta is None:
        vertical = "unknown"
    elif abs(delta) >= 1.5:
        vertical = "cross_floor_ge_1p5m"
    elif abs(delta) <= 0.75:
        vertical = "same_floor_le_0p75m"
    else:
        vertical = "vertical_ambiguous_0p75_to_1p5m"
    return {
        "score": stable_digest(seed, content_file.name, episode_index),
        "source_root_file": str(root_file.resolve()),
        "source_content_file": str(content_file.resolve()),
        "source_episode_index": episode_index,
        "source_episode_id": episode_id,
        "scene_id": scene,
        "scene_key": scene_key(scene),
        "target_category": category,
        "geodesic_distance": float(distance),
        "euclidean_distance": info.get("euclidean_distance", ""),
        "distance_stratum": classify_distance(float(distance)),
        "start_height": start_y,
        "target_viewpoint_height": "" if target is None else target,
        "vertical_delta": "" if delta is None else delta,
        "cross_floor_stratum": vertical,
    }


def choose(
    *,
    screen_selection: Path,
    episodes: int,
    cross_floor: int,
    seed: str,
) -> list[Dict[str, Any]]:
    with screen_selection.open(newline="", encoding="utf-8") as handle:
        screen = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("dataset") == "hm3d"
        ]
    if not screen:
        raise ValueError("screen selection has no HM3D rows")
    excluded_identities = {
        (str(Path(row["source_content_file"]).resolve()), row["source_episode_index"])
        for row in screen
    }
    forbidden_starts_by_scene: Dict[
        str, list[tuple[float, float]]
    ] = {}
    screen_by_content: Dict[str, list[Dict[str, str]]] = {}
    for row in screen:
        screen_by_content.setdefault(
            str(Path(row["source_content_file"]).resolve()), []
        ).append(row)
    for source, rows in screen_by_content.items():
        payload = read_gzip(Path(source))
        for row in rows:
            episode = payload["episodes"][int(row["source_episode_index"])]
            start = episode["start_position"]
            forbidden_starts_by_scene.setdefault(
                row["scene_key"], []
            ).append((float(start[0]), float(start[2])))
    root_file = Path(screen[0]["source_root_file"]).resolve()
    content_files = sorted(
        (root_file.parent / "content").glob("*.json.gz"),
        key=lambda path: stable_digest(seed, path.name),
    )
    same: list[Dict[str, Any]] = []
    cross: list[Dict[str, Any]] = []
    for content_file in content_files:
        content_scene_key = (
            content_file.name[: -len(".json.gz")]
            if content_file.name.endswith(".json.gz")
            else content_file.stem
        )
        payload = read_gzip(content_file)
        heights = goal_heights(payload)
        per_scene = []
        for index, episode in enumerate(payload.get("episodes") or ()):
            if (
                str(content_file.resolve()),
                str(index),
            ) in excluded_identities:
                continue
            if not isinstance(episode, Mapping):
                continue
            candidate = candidate_from_episode(
                heights=heights,
                content_file=content_file,
                episode_index=index,
                episode=episode,
                root_file=root_file,
                seed=seed,
                forbidden_starts=forbidden_starts_by_scene.get(
                    content_scene_key, ()
                ),
            )
            if candidate is not None:
                per_scene.append(candidate)
        if not per_scene:
            continue
        per_scene.sort(key=lambda row: row["score"])
        cross_candidates = [
            row
            for row in per_scene
            if row["cross_floor_stratum"] == "cross_floor_ge_1p5m"
        ]
        same_candidates = [
            row
            for row in per_scene
            if row["cross_floor_stratum"] == "same_floor_le_0p75m"
        ]
        if cross_candidates:
            cross.append(cross_candidates[0])
        if same_candidates:
            same.append(same_candidates[0])
        if (
            len(cross) >= max(cross_floor * 2, cross_floor + 4)
            and len(same) >= max(episodes * 2, episodes + 10)
        ):
            break
    cross.sort(key=lambda row: row["score"])
    same.sort(key=lambda row: row["score"])
    selected = list(cross[:cross_floor])
    used_scenes = {row["scene_key"] for row in selected}
    for row in same:
        if row["scene_key"] in used_scenes:
            continue
        selected.append(row)
        used_scenes.add(row["scene_key"])
        if len(selected) == episodes:
            break
    if len(selected) != episodes:
        raise RuntimeError(
            f"found only {len(selected)} disjoint calibration scenes"
        )
    selected.sort(key=lambda row: row["score"])
    for index, row in enumerate(selected):
        content_path = Path(row["source_content_file"])
        row.update(
            {
                "manifest_version": "ascent_vo_submap_calibration_v1",
                "selection_seed": seed,
                "global_case_index": index,
                "dataset_case_index": index,
                "logical_case_id": f"hm3d_submap_calibration_{index:04d}",
                "dataset": "hm3d",
                "split": "train",
                "chunk_id": "hm3d_submap_calibration_c00",
                "source_content_sha256": sha256(content_path),
                "selection_phase": (
                    "episode_disjoint_start_separated_3m_metadata_only"
                ),
                "episode_seed": int(row["score"][:8], 16),
                "arm_order": "CAL",
            }
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-selection", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--cross-floor", type=int, default=6)
    parser.add_argument("--seed", default="20260729")
    args = parser.parse_args()
    if args.episodes < 5:
        parser.error("episodes must be at least 5")
    if args.cross_floor < 0 or args.cross_floor > args.episodes:
        parser.error("invalid cross-floor count")
    rows = choose(
        screen_selection=args.screen_selection.resolve(),
        episodes=args.episodes,
        cross_floor=args.cross_floor,
        seed=str(args.seed),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in FIELDS}
            for row in rows
        )
    print(
        json.dumps(
            {
                "output": str(args.output_csv.resolve()),
                "sha256": sha256(args.output_csv),
                "episodes": len(rows),
                "scene_count": len({row["scene_key"] for row in rows}),
                "cross_floor_count": sum(
                    row["cross_floor_stratum"]
                    == "cross_floor_ge_1p5m"
                    for row in rows
                ),
                "distance_strata": {
                    stratum: sum(
                        row["distance_stratum"] == stratum for row in rows
                    )
                    for stratum in sorted(
                        {row["distance_stratum"] for row in rows}
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
