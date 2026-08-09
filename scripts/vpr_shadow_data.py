"""Strict readers and causal query construction for VPR shadow artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


KEYFRAME_CONTRACT = "ascent_v1.4_vpr_shadow_keyframes_v1"
GRAPH_EDGE_EVENTS = {
    "submap_split",
    "submap_exhaustion_recovery",
    "submap_revisit",
}
FLOOR_CHANGE_REASON = "floor_change"


@dataclass(frozen=True)
class ShadowFrame:
    capture_root: Path
    env: int
    episode_sequence: int
    action_step: int
    admission_index: int
    submap_id: str
    floor_id: int
    world_pose_vo: np.ndarray
    local_pose: np.ndarray
    tf_camera_to_submap: np.ndarray
    rgb_path: Path
    depth_path: Path
    min_depth: float
    max_depth: float
    fx: float
    fy: float

    @property
    def frame_id(self) -> str:
        return (
            f"e{self.episode_sequence}:env{self.env}:{self.submap_id}:"
            f"k{self.admission_index}:s{self.action_step}"
        )

    @property
    def calibration(self) -> dict[str, float]:
        return {
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "fx": self.fx,
            "fy": self.fy,
        }


@dataclass
class EpisodeGraph:
    direct_edges: set[frozenset[str]]
    submap_first_step: dict[str, int]
    submap_last_step: dict[str, int]
    submap_by_step: dict[int, str]
    floor_by_submap: dict[str, int]


@dataclass(frozen=True)
class QueryEvent:
    query_frame: ShadowFrame
    query_window: tuple[ShadowFrame, ...]
    candidate_frames: dict[str, tuple[ShadowFrame, ...]]

    @property
    def event_id(self) -> str:
        return f"query:{self.query_frame.frame_id}"


def resolve_submap_identity_floors(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, int]], set[tuple[int, int, str]]]:
    """Resolve immutable writer-floor identity from endpoint/split evidence.

    A policy call captures its VPR frame before the map-floor update, while the
    same call's endpoint is logged after that update.  Consequently, the final
    endpoint of an old submap may carry the destination floor.  We accept that
    mismatch only when both halves of the existing v1.2 transition contract are
    present: the endpoint requests a floor-change split and a matching split
    event follows at the same episode/step/source submap.
    """

    endpoints: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    floor_split_keys: set[tuple[int, int, str]] = set()
    for row in rows:
        sequence_value = row.get("episode_sequence")
        if sequence_value is None:
            continue
        sequence = int(sequence_value)
        if (
            row.get("record_type") == "submap_event"
            and row.get("event") == "submap_split"
            and row.get("reason") == FLOOR_CHANGE_REASON
        ):
            key = (
                sequence,
                int(row["step"]),
                str(row["source_submap_id"]),
            )
            if key in floor_split_keys:
                raise ValueError(f"duplicate floor-change split event: {key}")
            floor_split_keys.add(key)
        elif row.get("record_type") == "submap_action_endpoint":
            endpoints[(sequence, str(row["submap_id"]))].append(row)

    floors: dict[int, dict[str, int]] = defaultdict(dict)
    matched_split_keys: set[tuple[int, int, str]] = set()
    transition_keys: set[tuple[int, int, str]] = set()
    for (sequence, submap_id), values in endpoints.items():
        values = sorted(values, key=lambda row: int(row["action_step"]))
        steps = [int(row["action_step"]) for row in values]
        if len(steps) != len(set(steps)):
            raise ValueError(
                f"duplicate submap endpoint step: {sequence}:{submap_id}"
            )
        stable_floors: set[int] = set()
        transition_rows: list[Mapping[str, Any]] = []
        for row in values:
            step = int(row["action_step"])
            key = (sequence, step, submap_id)
            decision = row.get("decision")
            endpoint_claims_transition = bool(
                isinstance(decision, Mapping)
                and decision.get("floor_changed") is True
                and decision.get("should_split") is True
                and decision.get("reason") == FLOOR_CHANGE_REASON
            )
            split_confirms_transition = key in floor_split_keys
            if endpoint_claims_transition != split_confirms_transition:
                raise ValueError(f"incomplete floor-change transition: {key}")
            if endpoint_claims_transition:
                transition_rows.append(row)
                transition_keys.add(key)
                matched_split_keys.add(key)
            else:
                stable_floors.add(int(row["floor_id"]))

        if len(stable_floors) != 1:
            raise ValueError(
                "submap lacks one stable writer floor: "
                f"{sequence}:{submap_id}:{sorted(stable_floors)}"
            )
        identity_floor = next(iter(stable_floors))
        if len(transition_rows) > 1:
            raise ValueError(
                f"multiple floor-change endpoints: {sequence}:{submap_id}"
            )
        if transition_rows:
            transition = transition_rows[0]
            if int(transition["action_step"]) != max(steps):
                raise ValueError(
                    f"non-terminal floor-change endpoint: {sequence}:{submap_id}"
                )
            if int(transition["floor_id"]) == identity_floor:
                raise ValueError(
                    f"floor-change endpoint kept writer floor: {sequence}:{submap_id}"
                )
        floors[sequence][submap_id] = identity_floor

    unmatched = floor_split_keys - matched_split_keys
    if unmatched:
        raise ValueError(f"floor-change split lacks endpoint: {sorted(unmatched)[0]}")
    return {key: dict(value) for key, value in floors.items()}, transition_keys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL {path}:{line_number}")
            rows.append(value)
    return rows


def load_keyframes(
    manifest_path: Path, *, verify_hashes: bool = True
) -> tuple[dict[str, Any], list[ShadowFrame]]:
    manifest_path = Path(manifest_path).resolve()
    rows = read_jsonl(manifest_path)
    if not rows or rows[0].get("record_type") != "vpr_shadow_run_metadata":
        raise ValueError("missing VPR shadow metadata record")
    metadata = rows[0]
    if metadata.get("contract") != KEYFRAME_CONTRACT:
        raise ValueError("unexpected VPR shadow keyframe contract")
    if metadata.get("policy_gt_isolation") is not True:
        raise ValueError("keyframe stream does not assert GT isolation")
    if metadata.get("planner_write_access") is not False:
        raise ValueError("keyframe stream is not policy-passive")
    capture_root = manifest_path.parent
    frames: list[ShadowFrame] = []
    seen_ids: set[str] = set()
    for row in rows[1:]:
        if row.get("record_type") != "vpr_shadow_keyframe":
            continue
        rgb_path = (capture_root / str(row["rgb_path"])).resolve()
        depth_path = (capture_root / str(row["depth_path"])).resolve()
        if capture_root not in rgb_path.parents or capture_root not in depth_path.parents:
            raise ValueError("keyframe path escapes capture root")
        if not rgb_path.is_file() or not depth_path.is_file():
            raise ValueError("keyframe image pair is incomplete")
        if verify_hashes:
            if sha256(rgb_path) != row["rgb_file_sha256"]:
                raise ValueError(f"RGB keyframe hash mismatch: {rgb_path}")
            if sha256(depth_path) != row["depth_file_sha256"]:
                raise ValueError(f"depth keyframe hash mismatch: {depth_path}")
        frame = ShadowFrame(
            capture_root=capture_root,
            env=int(row["env"]),
            episode_sequence=int(row["episode_sequence"]),
            action_step=int(row["action_step"]),
            admission_index=int(row["admission_index"]),
            submap_id=str(row["submap_id"]),
            floor_id=int(row["floor_id"]),
            world_pose_vo=_array(row["world_pose_vo"], (3,), "world_pose_vo"),
            local_pose=_array(row["local_pose"], (3,), "local_pose"),
            tf_camera_to_submap=_array(
                row["tf_camera_to_submap"], (4, 4), "tf_camera_to_submap"
            ),
            rgb_path=rgb_path,
            depth_path=depth_path,
            min_depth=float(row["min_depth"]),
            max_depth=float(row["max_depth"]),
            fx=float(row["fx"]),
            fy=float(row["fy"]),
        )
        if frame.frame_id in seen_ids:
            raise ValueError(f"duplicate keyframe id {frame.frame_id}")
        seen_ids.add(frame.frame_id)
        frames.append(frame)
    frames.sort(
        key=lambda frame: (
            frame.episode_sequence,
            frame.action_step,
            frame.submap_id,
            frame.admission_index,
        )
    )
    return metadata, frames


def load_submap_graphs(path: Path) -> dict[int, EpisodeGraph]:
    direct_edges: dict[int, set[frozenset[str]]] = defaultdict(set)
    first: dict[int, dict[str, int]] = defaultdict(dict)
    last: dict[int, dict[str, int]] = defaultdict(dict)
    by_step: dict[int, dict[int, str]] = defaultdict(dict)
    rows = read_jsonl(path)
    floor_by_submap, _ = resolve_submap_identity_floors(rows)
    sequences: set[int] = set()
    for row in rows:
        sequence = row.get("episode_sequence")
        if sequence is None:
            continue
        sequence = int(sequence)
        sequences.add(sequence)
        if row.get("record_type") == "submap_action_endpoint":
            step = int(row["action_step"])
            submap_id = str(row["submap_id"])
            by_step[sequence][step] = submap_id
            first[sequence][submap_id] = min(
                step, first[sequence].get(submap_id, step)
            )
            last[sequence][submap_id] = max(
                step, last[sequence].get(submap_id, step)
            )
        elif (
            row.get("record_type") == "submap_event"
            and row.get("event") in GRAPH_EDGE_EVENTS
        ):
            source = row.get("source_submap_id")
            destination = row.get("destination_submap_id")
            if source and destination and source != destination:
                direct_edges[sequence].add(
                    frozenset((str(source), str(destination)))
                )
    return {
        sequence: EpisodeGraph(
            direct_edges=set(direct_edges[sequence]),
            submap_first_step=dict(first[sequence]),
            submap_last_step=dict(last[sequence]),
            submap_by_step=dict(by_step[sequence]),
            floor_by_submap=dict(floor_by_submap.get(sequence, {})),
        )
        for sequence in sequences
    }


def build_query_events(
    frames: Sequence[ShadowFrame],
    graphs: Mapping[int, EpisodeGraph],
    *,
    query_window_size: int = 3,
    min_query_frames: int = 2,
    min_step_separation: int = 30,
) -> list[QueryEvent]:
    by_episode: dict[int, list[ShadowFrame]] = defaultdict(list)
    for frame in frames:
        by_episode[frame.episode_sequence].append(frame)
    events: list[QueryEvent] = []
    for sequence, episode_frames in sorted(by_episode.items()):
        graph = graphs.get(sequence)
        if graph is None:
            raise ValueError(f"no submap diagnostics for episode sequence {sequence}")
        by_submap: dict[str, list[ShadowFrame]] = defaultdict(list)
        for frame in sorted(episode_frames, key=lambda value: value.action_step):
            writer_floor = graph.floor_by_submap.get(frame.submap_id)
            if writer_floor is None:
                raise ValueError(
                    f"keyframe submap absent from floor binding: {frame.submap_id}"
                )
            if frame.floor_id != writer_floor:
                raise ValueError(
                    "keyframe differs from immutable writer floor: "
                    f"{frame.frame_id}:{frame.floor_id}:{writer_floor}"
                )
            by_submap[frame.submap_id].append(frame)
            current_frames = by_submap[frame.submap_id]
            if len(current_frames) < min_query_frames:
                continue
            current_start = graph.submap_first_step.get(frame.submap_id)
            if current_start is None:
                raise ValueError(f"keyframe submap absent from diagnostics: {frame.submap_id}")
            candidates: dict[str, tuple[ShadowFrame, ...]] = {}
            for candidate_id, candidate_values in by_submap.items():
                if candidate_id == frame.submap_id:
                    continue
                candidate_start = graph.submap_first_step.get(candidate_id)
                if candidate_start is None or candidate_start >= current_start:
                    continue
                if frozenset((candidate_id, frame.submap_id)) in graph.direct_edges:
                    continue
                eligible = tuple(
                    value
                    for value in candidate_values
                    if value.floor_id == frame.floor_id
                    and value.action_step
                    <= frame.action_step - int(min_step_separation)
                )
                if eligible:
                    candidates[candidate_id] = eligible
            if candidates:
                events.append(
                    QueryEvent(
                        query_frame=frame,
                        query_window=tuple(current_frames[-query_window_size:]),
                        candidate_frames=candidates,
                    )
                )
    return events


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array
