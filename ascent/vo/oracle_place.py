"""Evaluation-only same-place oracle with a sealed policy output.

Habitat pose values remain private to this module.  The policy receives only an
``OracleSamePlaceEvent`` naming one historical submap.  VO consistency is a
method-side gate and does not correct or replace the Zhao pose estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from ascent.submaps.place_memory import OracleSamePlaceEvent


@dataclass(frozen=True)
class OraclePlaceConfig:
    enabled: bool = False
    physical_planar_radius_m: float = 0.75
    physical_height_radius_m: float = 0.75
    min_step_separation: int = 30
    min_excursion_m: float = 2.0
    vo_consistent_radius_m: float = 1.5

    def __post_init__(self) -> None:
        if self.physical_planar_radius_m <= 0.0:
            raise ValueError("physical_planar_radius_m must be positive")
        if self.physical_height_radius_m <= 0.0:
            raise ValueError("physical_height_radius_m must be positive")
        if self.min_step_separation < 1:
            raise ValueError("min_step_separation must be positive")
        if self.min_excursion_m <= self.physical_planar_radius_m:
            raise ValueError("min_excursion_m must exceed the return radius")
        if self.vo_consistent_radius_m <= 0.0:
            raise ValueError("vo_consistent_radius_m must be positive")


@dataclass
class _HistoricalEndpoint:
    action_step: int
    physical_xy: np.ndarray
    physical_height: float
    vo_xy: np.ndarray
    submap_id: str
    max_excursion_m: float = 0.0


@dataclass(frozen=True)
class OracleObservation:
    policy_events: Tuple[OracleSamePlaceEvent, ...]
    diagnostic: Dict[str, Any]


class OraclePlaceDiagnosticsWriter:
    """Append-only evaluation evidence; never passed to the policy."""

    def __init__(self, path: Path, metadata: Mapping[str, Any]) -> None:
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved
        self._stream = resolved.open("x", encoding="utf-8", buffering=1)
        self._write(
            {"record_type": "oracle_place_run_metadata", **dict(metadata)}
        )

    def record(self, record: Mapping[str, Any]) -> None:
        self._write(dict(record))

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def _write(self, record: Mapping[str, Any]) -> None:
        self._stream.write(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        )


class SamePlaceOracle:
    """Online implementation of the frozen physical-return audit definition."""

    def __init__(self, num_envs: int, config: OraclePlaceConfig) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not config.enabled:
            raise ValueError("same-place oracle must be explicitly enabled")
        self.config = config
        self._history: List[List[_HistoricalEndpoint]] = [
            [] for _ in range(num_envs)
        ]
        self._event_sequence = [0 for _ in range(num_envs)]
        self._emitted_pairs: List[set[Tuple[str, str]]] = [
            set() for _ in range(num_envs)
        ]

    def reset(self, env: int) -> None:
        self._history[env] = []
        self._event_sequence[env] = 0
        self._emitted_pairs[env] = set()

    @staticmethod
    def _physical_pose(gt_pose: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
        required = {"x", "y", "height"}
        missing = required.difference(gt_pose)
        if missing:
            raise ValueError(
                f"same-place oracle missing evaluation pose fields: {sorted(missing)}"
            )
        xy = np.asarray(
            [float(gt_pose["x"]), float(gt_pose["y"])], dtype=np.float64
        )
        height = float(gt_pose["height"])
        if not np.isfinite(xy).all() or not np.isfinite(height):
            raise ValueError("same-place oracle received non-finite evaluation pose")
        return xy, height

    @staticmethod
    def _vo_xy(vo_pose: Sequence[float]) -> np.ndarray:
        pose = np.asarray(vo_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("same-place oracle received invalid Zhao pose")
        return pose[:2].copy()

    def observe(
        self,
        *,
        env: int,
        dataset: str,
        scene_id: str,
        episode_id: str,
        action_step: int,
        gt_pose: Mapping[str, Any],
        vo_pose: Sequence[float],
        current_submap_id: str,
    ) -> OracleObservation:
        physical_xy, physical_height = self._physical_pose(gt_pose)
        vo_xy = self._vo_xy(vo_pose)
        current_submap = str(current_submap_id)
        if not current_submap:
            raise ValueError("same-place oracle requires current submap identity")

        history = self._history[env]
        for endpoint in history:
            endpoint.max_excursion_m = max(
                endpoint.max_excursion_m,
                float(np.linalg.norm(physical_xy - endpoint.physical_xy)),
            )

        eligible: List[Tuple[float, int, _HistoricalEndpoint, float]] = []
        physical_return_count = 0
        for endpoint in history:
            planar_distance = float(
                np.linalg.norm(physical_xy - endpoint.physical_xy)
            )
            height_distance = abs(physical_height - endpoint.physical_height)
            step_separation = int(action_step) - endpoint.action_step
            if (
                planar_distance <= self.config.physical_planar_radius_m
                and height_distance <= self.config.physical_height_radius_m
                and step_separation >= self.config.min_step_separation
                and endpoint.max_excursion_m >= self.config.min_excursion_m
                and endpoint.submap_id != current_submap
            ):
                physical_return_count += 1
                vo_distance = float(np.linalg.norm(vo_xy - endpoint.vo_xy))
                if vo_distance < self.config.vo_consistent_radius_m:
                    eligible.append(
                        (
                            planar_distance,
                            -endpoint.action_step,
                            endpoint,
                            vo_distance,
                        )
                    )

        events: List[OracleSamePlaceEvent] = []
        match_diagnostics: List[Dict[str, Any]] = []
        if eligible:
            # Preserve every distinct same-place identity candidate.  The
            # policy-side graph gate, which never sees metric GT, rejects
            # sequential/direct aliases.  Selecting only one physically
            # nearest endpoint here could otherwise hide a simultaneously
            # valid graph-non-direct revisit behind a direct neighbor.
            by_submap: Dict[
                str, Tuple[float, int, _HistoricalEndpoint, float]
            ] = {}
            for candidate in eligible:
                endpoint = candidate[2]
                previous = by_submap.get(endpoint.submap_id)
                if previous is None or (candidate[0], candidate[1]) < (
                    previous[0],
                    previous[1],
                ):
                    by_submap[endpoint.submap_id] = candidate
            for planar_distance, _, endpoint, vo_distance in sorted(
                by_submap.values(), key=lambda item: (item[0], item[1])
            ):
                pair = (current_submap, endpoint.submap_id)
                emitted = pair not in self._emitted_pairs[env]
                if emitted:
                    self._event_sequence[env] += 1
                    events.append(
                        OracleSamePlaceEvent(
                            event_sequence=self._event_sequence[env],
                            reference_submap_id=endpoint.submap_id,
                        )
                    )
                    self._emitted_pairs[env].add(pair)
                match_diagnostics.append(
                    {
                        "reference_submap_id": endpoint.submap_id,
                        "reference_action_step": endpoint.action_step,
                        "physical_planar_distance_m": float(planar_distance),
                        "physical_height_distance_m": abs(
                            physical_height - endpoint.physical_height
                        ),
                        "vo_planar_distance_m": float(vo_distance),
                        "reference_max_excursion_m": float(
                            endpoint.max_excursion_m
                        ),
                        "policy_event_emitted": emitted,
                    }
                )

        history.append(
            _HistoricalEndpoint(
                action_step=int(action_step),
                physical_xy=physical_xy,
                physical_height=physical_height,
                vo_xy=vo_xy,
                submap_id=current_submap,
            )
        )
        diagnostic = {
            "record_type": "oracle_place_step",
            "dataset": str(dataset),
            "scene_id": str(scene_id),
            "episode_id": str(episode_id),
            "env": int(env),
            "action_step": int(action_step),
            "current_submap_id": current_submap,
            "history_size": len(history),
            "physical_return_candidate_count": int(physical_return_count),
            "vo_consistent_candidate_count": len(eligible),
            "vo_consistent_submap_candidate_count": len(match_diagnostics),
            "matches": match_diagnostics,
            "policy_event_count": len(events),
        }
        return OracleObservation(
            policy_events=tuple(events), diagnostic=diagnostic
        )
