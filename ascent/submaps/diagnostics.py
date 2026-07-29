"""Append-only diagnostics for policy-visible submap state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class SubmapDiagnosticsWriter:
    """Write one immutable JSONL stream without accepting evaluation GT."""

    _FORBIDDEN_KEYS = {
        "gps",
        "compass",
        "ground_truth",
        "gt_pose",
        "gt_start_aligned_pose",
    }

    def __init__(self, path: Path, metadata: Mapping[str, Any]) -> None:
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved
        self._stream = resolved.open("x", encoding="utf-8", buffering=1)
        self._write({"record_type": "submap_run_metadata", **dict(metadata)})

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def record_episode_reset(self, *, env: int, episode_sequence: int) -> None:
        self._write(
            {
                "record_type": "submap_episode_reset",
                "env": int(env),
                "episode_sequence": int(episode_sequence),
            }
        )

    def record_action_endpoint(
        self,
        *,
        env: int,
        episode_sequence: int,
        action_step: int,
        submap_id: str,
        floor_id: int,
        world_pose_vo: Sequence[float],
        local_pose: Sequence[float],
        overlap: Optional[float],
        decision: Mapping[str, Any],
    ) -> None:
        self._write(
            {
                "record_type": "submap_action_endpoint",
                "env": int(env),
                "episode_sequence": int(episode_sequence),
                "action_step": int(action_step),
                "submap_id": str(submap_id),
                "floor_id": int(floor_id),
                "world_pose_vo": _finite_pose(world_pose_vo),
                "local_pose": _finite_pose(local_pose),
                "overlap": None if overlap is None else float(overlap),
                "decision": dict(decision),
            }
        )

    def record_event(
        self,
        *,
        env: int,
        episode_sequence: int,
        event: Mapping[str, Any],
    ) -> None:
        self._write(
            {
                "record_type": "submap_event",
                "env": int(env),
                "episode_sequence": int(episode_sequence),
                **dict(event),
            }
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        lowered = {str(key).lower() for key in record}
        forbidden = lowered.intersection(self._FORBIDDEN_KEYS)
        if forbidden:
            raise ValueError(
                "evaluation-only pose key reached submap diagnostics: "
                f"{sorted(forbidden)}"
            )
        self._stream.write(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        )


def _finite_pose(values: Sequence[float]) -> list[float]:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError(f"invalid finite SE(2) pose {pose}")
    return [float(value) for value in pose]
