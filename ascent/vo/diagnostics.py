"""Append-only separation of VO belief and evaluation-only GT pose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ascent.vo.pose_provider import PoseUpdate, wrap_angle


class VODiagnosticsWriter:
    def __init__(self, path: Path, metadata: Mapping[str, Any]) -> None:
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._stream = path.open("x", encoding="utf-8", buffering=1)
        self._write({"record_type": "run_metadata", **dict(metadata)})

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def record_step(
        self,
        *,
        dataset: str,
        scene_id: str,
        episode_id: str,
        seed: int,
        action_step: int,
        update: PoseUpdate,
        gt_pose: Mapping[str, Any] | None,
    ) -> None:
        record = {
            "record_type": "vo_step",
            "dataset": dataset,
            "scene_id": scene_id,
            "episode_id": str(episode_id),
            "seed": int(seed),
            "action_step": int(action_step),
            **update.as_dict(),
        }
        if gt_pose is not None:
            gt = np.array(
                [
                    float(gt_pose["x"]),
                    float(gt_pose["y"]),
                    float(gt_pose["yaw"]),
                ]
            )
            estimated = np.asarray(update.pose_after)
            record["gt_start_aligned_pose"] = {
                key: float(value) for key, value in gt_pose.items()
            }
            if update.status == "terminal_no_policy_successor_then_autoreset":
                record["localization_error_available"] = False
            else:
                record["localization_error_available"] = True
                record["translation_error"] = float(
                    np.linalg.norm(estimated[:2] - gt[:2])
                )
                record["absolute_yaw_error"] = abs(
                    wrap_angle(float(estimated[2] - gt[2]))
                )
        self._write(record)

    def record_error(
        self,
        *,
        dataset: str,
        scene_id: str,
        episode_id: str,
        seed: int,
        action_step: int,
        action: int,
        error: BaseException,
    ) -> None:
        self._write(
            {
                "record_type": "vo_technical_error",
                "dataset": dataset,
                "scene_id": scene_id,
                "episode_id": str(episode_id),
                "seed": int(seed),
                "action_step": int(action_step),
                "action": int(action),
                "finite": False,
                "status": "technical_failure",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    def record_episode_end(
        self,
        *,
        dataset: str,
        scene_id: str,
        episode_id: str,
        seed: int,
        action_steps: int,
        native_metrics: Mapping[str, Any],
    ) -> None:
        """Record Habitat evaluation metrics outside the policy data path."""

        metrics = {}
        for key, value in native_metrics.items():
            if isinstance(value, (bool, int, float, np.number)):
                scalar = float(value)
                if not np.isfinite(scalar):
                    raise ValueError(
                        f"non-finite native metric {key}={value}"
                    )
                metrics[str(key)] = scalar
        self._write(
            {
                "record_type": "episode_end",
                "dataset": dataset,
                "scene_id": scene_id,
                "episode_id": str(episode_id),
                "seed": int(seed),
                "action_steps": int(action_steps),
                "native_metrics": metrics,
                "evaluation_pose_source": "habitat_ground_truth",
                "policy_pose_source": "zhao_rgbd_2021",
                "gt_policy_isolation": True,
            }
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        self._stream.write(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        )
