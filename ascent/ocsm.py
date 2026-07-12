"""Minimal Object-Conditioned Search Memory (OCSM) v0.

OCSM stores outcomes of frontier search attempts and calibrates the already-sorted
ASCENT frontier candidates.  It does not produce actions, alter ValueMap values, or
handle stair/cross-floor execution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OCSMConfig:
    enabled: bool = False
    association_radius_m: float = 0.5
    low_gain_area_m2: float = 0.5
    hard_streak: int = 2
    suppression_radius_m: float = 0.5

    @classmethod
    def from_env(cls) -> "OCSMConfig":
        return cls(
            enabled=_env_flag("ASCENT_OCSM_ENABLED"),
            association_radius_m=float(
                os.environ.get("ASCENT_OCSM_ASSOCIATION_RADIUS_M", "0.5")
            ),
            low_gain_area_m2=float(
                os.environ.get("ASCENT_OCSM_LOW_GAIN_AREA_M2", "0.5")
            ),
            hard_streak=int(os.environ.get("ASCENT_OCSM_HARD_STREAK", "2")),
            suppression_radius_m=float(
                os.environ.get("ASCENT_OCSM_SUPPRESSION_RADIUS_M", "0.5")
            ),
        )

    def __post_init__(self) -> None:
        if self.association_radius_m <= 0:
            raise ValueError("association_radius_m must be positive")
        if self.low_gain_area_m2 < 0:
            raise ValueError("low_gain_area_m2 must be non-negative")
        if self.hard_streak < 2:
            raise ValueError("hard_streak must be at least 2")
        if self.suppression_radius_m <= 0:
            raise ValueError("suppression_radius_m must be positive")


@dataclass
class SearchAttempt:
    attempt_id: int
    env: int
    target: str
    floor_index: int
    start_frontier: np.ndarray
    start_step: int
    start_explored_pixels: int
    start_local_explored_pixels: int
    pixels_per_meter: float
    start_target_present: bool
    start_up_stair_present: bool
    start_down_stair_present: bool
    start_frontiers: np.ndarray
    min_robot_distance_m: float = float("inf")
    last_association_distance_m: Optional[float] = None
    target_gain: bool = False
    stair_gain: bool = False


@dataclass
class MemoryEntry:
    target: str
    floor_index: int
    location: np.ndarray
    low_gain_streak: int = 0
    completed_attempts: int = 0


class ObjectConditionedSearchMemory:
    """Per-environment frontier-attempt memory and candidate calibration."""

    def __init__(self, num_envs: int, config: OCSMConfig):
        if not config.enabled:
            raise ValueError("OCSM should only be instantiated when enabled")
        self.config = config
        self._active: List[Optional[SearchAttempt]] = [None] * num_envs
        self._entries: List[List[MemoryEntry]] = [[] for _ in range(num_envs)]
        self._attempt_seq: List[int] = [0] * num_envs
        self._last_event: List[Dict[str, Any]] = [{} for _ in range(num_envs)]
        self._last_calibration: List[Dict[str, Any]] = [{} for _ in range(num_envs)]

    def reset(self, env: int) -> None:
        self._active[env] = None
        self._entries[env] = []
        self._attempt_seq[env] = 0
        self._last_event[env] = {"event": "reset"}
        self._last_calibration[env] = {}

    @staticmethod
    def _xy(value: Sequence[float]) -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < 2:
            raise ValueError("frontier must contain two coordinates")
        return arr[:2].copy()

    @staticmethod
    def _frontier_array(frontiers: Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(frontiers, dtype=float)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)
        return arr.reshape(-1, 2).copy()

    @staticmethod
    def _distance(a: Sequence[float], b: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    @staticmethod
    def local_explored_pixels(obstacle_map: Any, frontier: Sequence[float], radius_m: float) -> int:
        """Count explored pixels near a metric frontier using ASCENT's map transform."""
        frontier_px = obstacle_map._xy_to_px(np.atleast_2d(np.asarray(frontier, dtype=float)))[0]
        cx, cy = int(frontier_px[0]), int(frontier_px[1])
        radius_px = max(1, int(round(radius_m * obstacle_map.pixels_per_meter)))
        explored = np.asarray(obstacle_map.explored_area, dtype=bool)
        y0, y1 = max(0, cy - radius_px), min(explored.shape[0], cy + radius_px + 1)
        x0, x1 = max(0, cx - radius_px), min(explored.shape[1], cx + radius_px + 1)
        if y0 >= y1 or x0 >= x1:
            return 0
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px**2
        return int(np.count_nonzero(explored[y0:y1, x0:x1] & circle))

    def _matching_entry(
        self, env: int, target: str, floor_index: int, location: Sequence[float]
    ) -> Tuple[Optional[MemoryEntry], Optional[float]]:
        matches = []
        for entry in self._entries[env]:
            if entry.target != target or entry.floor_index != floor_index:
                continue
            distance = self._distance(entry.location, location)
            if distance <= self.config.suppression_radius_m:
                matches.append((distance, entry))
        if not matches:
            return None, None
        distance, entry = min(matches, key=lambda item: item[0])
        return entry, float(distance)

    def start_attempt(
        self,
        env: int,
        target: str,
        floor_index: int,
        frontier: Sequence[float],
        step: int,
        obstacle_map: Any,
        target_present: bool,
        up_stair_present: bool,
        down_stair_present: bool,
        frontiers: Sequence[Sequence[float]],
        robot_xy: Sequence[float],
    ) -> None:
        self._attempt_seq[env] += 1
        frontier_xy = self._xy(frontier)
        self._active[env] = SearchAttempt(
            attempt_id=self._attempt_seq[env],
            env=env,
            target=target,
            floor_index=int(floor_index),
            start_frontier=frontier_xy,
            start_step=int(step),
            start_explored_pixels=int(np.count_nonzero(obstacle_map.explored_area)),
            start_local_explored_pixels=self.local_explored_pixels(
                obstacle_map, frontier_xy, self.config.association_radius_m
            ),
            pixels_per_meter=float(obstacle_map.pixels_per_meter),
            start_target_present=bool(target_present),
            start_up_stair_present=bool(up_stair_present),
            start_down_stair_present=bool(down_stair_present),
            start_frontiers=self._frontier_array(frontiers),
            min_robot_distance_m=self._distance(robot_xy, frontier_xy),
            last_association_distance_m=0.0,
        )
        self._last_event[env] = {
            "event": "attempt_started",
            "attempt_id": self._attempt_seq[env],
            "target": target,
            "floor_index": int(floor_index),
            "start_frontier": frontier_xy.tolist(),
            "step": int(step),
        }

    def _frontier_novelty(
        self, attempt: SearchAttempt, current_frontiers: Sequence[Sequence[float]]
    ) -> Dict[str, Any]:
        current = self._frontier_array(current_frontiers)
        if len(current) == 0:
            distances: List[Optional[float]] = []
        elif len(attempt.start_frontiers) == 0:
            distances = [None] * len(current)
        else:
            distances = [
                float(np.min(np.linalg.norm(attempt.start_frontiers - point, axis=1)))
                for point in current
            ]
        novel_count = sum(
            distance is None or distance > self.config.association_radius_m
            for distance in distances
        )
        start_persists = any(
            self._distance(attempt.start_frontier, point)
            <= self.config.association_radius_m
            for point in current
        )
        return {
            "start_count": int(len(attempt.start_frontiers)),
            "end_count": int(len(current)),
            "count_delta": int(len(current) - len(attempt.start_frontiers)),
            "distance_to_start_set_m": distances,
            "novel_count": int(novel_count),
            "attempt_frontier_persists": bool(start_persists),
            "used_for_low_gain": False,
        }

    def finish_attempt(
        self,
        env: int,
        outcome: str,
        reason: str,
        step: int,
        obstacle_map: Any,
        current_frontiers: Sequence[Sequence[float]],
    ) -> Optional[Dict[str, Any]]:
        attempt = self._active[env]
        if attempt is None:
            return None
        explored_pixels = int(np.count_nonzero(obstacle_map.explored_area))
        explored_delta_pixels = max(0, explored_pixels - attempt.start_explored_pixels)
        explored_delta_m2 = explored_delta_pixels / (attempt.pixels_per_meter**2)
        low_gain = bool(
            outcome == "completed"
            and explored_delta_m2 < self.config.low_gain_area_m2
            and not attempt.target_gain
            and not attempt.stair_gain
        )
        entry, entry_distance = self._matching_entry(
            env, attempt.target, attempt.floor_index, attempt.start_frontier
        )
        streak_before = entry.low_gain_streak if entry is not None else 0
        streak_after = streak_before
        memory_update = "none"
        if outcome == "completed":
            if entry is None:
                entry = MemoryEntry(
                    target=attempt.target,
                    floor_index=attempt.floor_index,
                    location=attempt.start_frontier.copy(),
                )
                self._entries[env].append(entry)
                entry_distance = 0.0
            entry.completed_attempts += 1
            if low_gain:
                entry.low_gain_streak += 1
                memory_update = "increment_low_gain_streak"
            else:
                entry.low_gain_streak = 0
                memory_update = "clear_streak_on_positive_gain"
            streak_after = entry.low_gain_streak

        event = {
            "event": "attempt_finished",
            "attempt_id": attempt.attempt_id,
            "target": attempt.target,
            "floor_index": attempt.floor_index,
            "start_frontier": attempt.start_frontier.tolist(),
            "start_step": attempt.start_step,
            "end_step": int(step),
            "duration_steps": int(step - attempt.start_step),
            "outcome": outcome,
            "reason": reason,
            "explored_area_delta_pixels": explored_delta_pixels,
            "explored_area_delta_m2": float(explored_delta_m2),
            "target_evidence_delta": bool(attempt.target_gain),
            "new_stair": bool(attempt.stair_gain),
            "low_gain": low_gain,
            "low_gain_threshold_m2": self.config.low_gain_area_m2,
            "streak_before": int(streak_before),
            "streak_after": int(streak_after),
            "memory_update": memory_update,
            "memory_association_distance_m": entry_distance,
            "min_robot_distance_m": float(attempt.min_robot_distance_m),
            "last_frontier_association_distance_m": attempt.last_association_distance_m,
            "frontier_novelty": self._frontier_novelty(attempt, current_frontiers),
        }
        self._active[env] = None
        self._last_event[env] = event
        return event

    def observe(
        self,
        env: int,
        target: str,
        floor_index: int,
        step: int,
        robot_xy: Sequence[float],
        obstacle_map: Any,
        target_present: bool,
        up_stair_present: bool,
        down_stair_present: bool,
        frontiers: Sequence[Sequence[float]],
        arrival_radius_m: float,
    ) -> Optional[Dict[str, Any]]:
        attempt = self._active[env]
        if attempt is None:
            return None
        if attempt.target != target:
            return self.finish_attempt(
                env, "interrupted", "target_change", step, obstacle_map, frontiers
            )
        if attempt.floor_index != int(floor_index):
            outcome = "completed" if attempt.stair_gain else "interrupted"
            reason = "new_stair_floor_change" if attempt.stair_gain else "floor_change"
            return self.finish_attempt(env, outcome, reason, step, obstacle_map, frontiers)

        robot_distance = self._distance(robot_xy, attempt.start_frontier)
        attempt.min_robot_distance_m = min(attempt.min_robot_distance_m, robot_distance)
        if not attempt.start_target_present and target_present:
            attempt.target_gain = True
        if (
            (not attempt.start_up_stair_present and up_stair_present)
            or (not attempt.start_down_stair_present and down_stair_present)
        ):
            attempt.stair_gain = True

        if attempt.target_gain:
            return self.finish_attempt(
                env, "completed", "target_evidence_found", step, obstacle_map, frontiers
            )
        if robot_distance <= arrival_radius_m:
            return self.finish_attempt(
                env, "completed", "arrived", step, obstacle_map, frontiers
            )

        frontier_persists = any(
            self._distance(attempt.start_frontier, frontier)
            <= self.config.association_radius_m
            for frontier in self._frontier_array(frontiers)
        )
        if not frontier_persists:
            current_local = self.local_explored_pixels(
                obstacle_map, attempt.start_frontier, self.config.association_radius_m
            )
            if current_local > attempt.start_local_explored_pixels:
                return self.finish_attempt(
                    env,
                    "completed",
                    "frontier_disappeared_after_local_exploration",
                    step,
                    obstacle_map,
                    frontiers,
                )
        return None

    def register_selection(
        self,
        env: int,
        target: str,
        floor_index: int,
        selected_frontier: Sequence[float],
        step: int,
        robot_xy: Sequence[float],
        obstacle_map: Any,
        target_present: bool,
        up_stair_present: bool,
        down_stair_present: bool,
        frontiers: Sequence[Sequence[float]],
    ) -> Optional[Dict[str, Any]]:
        selected = self._xy(selected_frontier)
        attempt = self._active[env]
        if attempt is not None:
            association_distance = self._distance(selected, attempt.start_frontier)
            attempt.last_association_distance_m = association_distance
            if (
                attempt.target == target
                and attempt.floor_index == int(floor_index)
                and association_distance <= self.config.association_radius_m
            ):
                return None
            self.finish_attempt(
                env,
                "interrupted",
                "replanning_switch",
                step,
                obstacle_map,
                frontiers,
            )
        self.start_attempt(
            env,
            target,
            floor_index,
            selected,
            step,
            obstacle_map,
            target_present,
            up_stair_present,
            down_stair_present,
            frontiers,
            robot_xy,
        )
        return self._last_event[env]

    def mark_execution_failure(
        self,
        env: int,
        reason: str,
        step: int,
        obstacle_map: Any,
        current_frontiers: Sequence[Sequence[float]],
    ) -> Optional[Dict[str, Any]]:
        return self.finish_attempt(
            env, "execution_failure", reason, step, obstacle_map, current_frontiers
        )

    def end_for_stair_mode(
        self,
        env: int,
        step: int,
        obstacle_map: Any,
        current_frontiers: Sequence[Sequence[float]],
        reason: str,
    ) -> Optional[Dict[str, Any]]:
        attempt = self._active[env]
        if attempt is None:
            return None
        outcome = "completed" if attempt.stair_gain else "interrupted"
        final_reason = f"new_stair_{reason}" if attempt.stair_gain else reason
        return self.finish_attempt(
            env, outcome, final_reason, step, obstacle_map, current_frontiers
        )

    def calibrate_candidates(
        self,
        env: int,
        target: str,
        floor_index: int,
        sorted_pts: np.ndarray,
        sorted_values: Sequence[float],
    ) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
        points = self._frontier_array(sorted_pts)
        values = list(sorted_values)
        records = []
        normal_indices: List[int] = []
        soft_indices: List[int] = []
        hard_indices: List[int] = []
        for idx, point in enumerate(points):
            entry, distance = self._matching_entry(env, target, int(floor_index), point)
            streak = entry.low_gain_streak if entry is not None else 0
            if streak >= self.config.hard_streak:
                status = "hard-suppressed"
                hard_indices.append(idx)
            elif streak > 0:
                status = "soft"
                soft_indices.append(idx)
            else:
                status = "normal"
                normal_indices.append(idx)
            records.append(
                {
                    "frontier": point.tolist(),
                    "original_rank": idx,
                    "original_value": values[idx],
                    "status": status,
                    "low_gain_streak": int(streak),
                    "memory_distance_m": distance,
                    "calibrated_rank": None,
                }
            )

        kept_indices = normal_indices + soft_indices
        fallback = bool(len(points) > 0 and len(kept_indices) == 0)
        if fallback:
            kept_indices = list(range(len(points)))
        for calibrated_rank, original_idx in enumerate(kept_indices):
            records[original_idx]["calibrated_rank"] = calibrated_rank
        calibrated_pts = np.asarray([points[i] for i in kept_indices], dtype=float)
        if calibrated_pts.size == 0:
            calibrated_pts = np.empty((0, 2), dtype=float)
        calibrated_values = [values[i] for i in kept_indices]
        trace = {
            "enabled": True,
            "target": target,
            "floor_index": int(floor_index),
            "fallback_to_original": fallback,
            "association_radius_m": self.config.association_radius_m,
            "suppression_radius_m": self.config.suppression_radius_m,
            "hard_streak": self.config.hard_streak,
            "candidates": records,
            "original_order": points.tolist(),
            "calibrated_order": calibrated_pts.tolist(),
        }
        self._last_calibration[env] = trace
        return calibrated_pts, calibrated_values, trace

    def trace(self, env: int) -> Dict[str, Any]:
        attempt = self._active[env]
        active = None
        if attempt is not None:
            active = {
                "attempt_id": attempt.attempt_id,
                "target": attempt.target,
                "floor_index": attempt.floor_index,
                "start_frontier": attempt.start_frontier.tolist(),
                "start_step": attempt.start_step,
                "min_robot_distance_m": attempt.min_robot_distance_m,
                "association_distance_m": attempt.last_association_distance_m,
                "target_gain": attempt.target_gain,
                "stair_gain": attempt.stair_gain,
            }
        return {
            "enabled": True,
            "config": asdict(self.config),
            "active_attempt": active,
            "last_event": self._last_event[env],
            "candidate_calibration": self._last_calibration[env],
            "memory_entries": [
                {
                    "target": entry.target,
                    "floor_index": entry.floor_index,
                    "location": entry.location.tolist(),
                    "low_gain_streak": entry.low_gain_streak,
                    "completed_attempts": entry.completed_attempts,
                }
                for entry in self._entries[env]
            ],
        }
