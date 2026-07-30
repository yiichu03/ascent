"""Bounded no-progress watchdog for ASCENT's singleton-frontier fast path."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional

import numpy as np


ENABLED_ENV = "ASCENT_SINGLETON_FRONTIER_WATCHDOG"
POSITION_TOLERANCE_ENV = "ASCENT_SINGLETON_WATCHDOG_POSITION_TOLERANCE_M"
PATIENCE_STEPS_ENV = "ASCENT_SINGLETON_WATCHDOG_PATIENCE_STEPS"
MIN_PROGRESS_ENV = "ASCENT_SINGLETON_WATCHDOG_MIN_PROGRESS_M"
STOP_GUARD_ENV = "ASCENT_SINGLETON_WATCHDOG_STOP_GUARD"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _env_positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive float, got {value!r}")
    return value


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass
class SingletonFrontierWatchdog:
    """Track bounded progress toward one spatially stable singleton frontier.

    The watchdog intentionally does not classify scene holes or alter frontier
    generation. It only asks whether a singleton target stayed in the same
    small spatial neighborhood while the robot failed to make one meaningful
    best-so-far distance improvement for a bounded number of consecutive
    policy steps.
    """

    enabled: bool = False
    position_tolerance_m: float = 0.25
    patience_steps: int = 80
    min_progress_m: float = 0.2
    stop_guard_enabled: bool = False
    anchor_frontier: Optional[np.ndarray] = None
    progress_reference_distance_m: float = float("inf")
    best_distance_m: float = float("inf")
    no_progress_steps: int = 0
    last_policy_step: Optional[int] = None
    floor_index: Optional[int] = None

    @classmethod
    def from_environment(cls) -> "SingletonFrontierWatchdog":
        return cls(
            enabled=_env_enabled(ENABLED_ENV, False),
            position_tolerance_m=_env_positive_float(
                POSITION_TOLERANCE_ENV, 0.25
            ),
            patience_steps=_env_positive_int(PATIENCE_STEPS_ENV, 80),
            min_progress_m=_env_positive_float(MIN_PROGRESS_ENV, 0.2),
            stop_guard_enabled=_env_enabled(STOP_GUARD_ENV, False),
        )

    def reset_tracking(self) -> None:
        self.anchor_frontier = None
        self.progress_reference_distance_m = float("inf")
        self.best_distance_m = float("inf")
        self.no_progress_steps = 0
        self.last_policy_step = None
        self.floor_index = None

    def reset_trace(self, reason: str, frontier_count: int) -> Dict[str, Any]:
        self.reset_tracking()
        return {
            "enabled": self.enabled,
            "frontier_count": int(frontier_count),
            "status": reason,
            "triggered": False,
            "position_tolerance_m": self.position_tolerance_m,
            "patience_steps": self.patience_steps,
            "min_progress_m": self.min_progress_m,
            "stop_guard_enabled": self.stop_guard_enabled,
        }

    def observe(
        self,
        frontier: np.ndarray,
        robot_xy: np.ndarray,
        *,
        policy_step: int,
        floor_index: Optional[int],
    ) -> Dict[str, Any]:
        frontier = np.asarray(frontier, dtype=float).reshape(2)
        robot_xy = np.asarray(robot_xy, dtype=float).reshape(2)
        policy_step = int(policy_step)

        if not self.enabled:
            return self.reset_trace("disabled", frontier_count=1)

        consecutive_step = (
            self.last_policy_step is not None
            and policy_step == self.last_policy_step + 1
        )
        same_floor = self.anchor_frontier is not None and self.floor_index == floor_index
        anchor_offset_m = (
            float(np.linalg.norm(frontier - self.anchor_frontier))
            if self.anchor_frontier is not None
            else float("inf")
        )
        stable_frontier = (
            consecutive_step
            and same_floor
            and anchor_offset_m <= self.position_tolerance_m
        )

        if not stable_frontier:
            self.anchor_frontier = frontier.copy()
            current_distance_m = float(np.linalg.norm(frontier - robot_xy))
            self.progress_reference_distance_m = current_distance_m
            self.best_distance_m = current_distance_m
            self.no_progress_steps = 0
            self.last_policy_step = policy_step
            self.floor_index = floor_index
            return self._trace(
                status="tracking_started",
                frontier=frontier,
                current_distance_m=current_distance_m,
                anchor_offset_m=0.0,
                meaningful_progress=False,
                triggered=False,
            )

        current_distance_m = float(np.linalg.norm(self.anchor_frontier - robot_xy))
        self.best_distance_m = min(self.best_distance_m, current_distance_m)
        improvement_m = (
            self.progress_reference_distance_m - self.best_distance_m
        )
        meaningful_progress = improvement_m >= self.min_progress_m
        if meaningful_progress:
            self.progress_reference_distance_m = self.best_distance_m
            self.no_progress_steps = 0
            status = "meaningful_progress"
        else:
            self.no_progress_steps += 1
            status = "no_progress"

        self.last_policy_step = policy_step
        triggered = self.no_progress_steps >= self.patience_steps
        if triggered:
            status = "triggered"

        trace = self._trace(
            status=status,
            frontier=frontier,
            current_distance_m=current_distance_m,
            anchor_offset_m=anchor_offset_m,
            meaningful_progress=meaningful_progress,
            triggered=triggered,
        )
        if triggered:
            self.reset_tracking()
        return trace

    def _trace(
        self,
        *,
        status: str,
        frontier: np.ndarray,
        current_distance_m: float,
        anchor_offset_m: float,
        meaningful_progress: bool,
        triggered: bool,
    ) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "frontier_count": 1,
            "status": status,
            "triggered": bool(triggered),
            "frontier": frontier.tolist(),
            "anchor_frontier": (
                self.anchor_frontier.tolist()
                if self.anchor_frontier is not None
                else []
            ),
            "anchor_offset_m": float(anchor_offset_m),
            "current_distance_m": float(current_distance_m),
            "progress_reference_distance_m": float(
                self.progress_reference_distance_m
            ),
            "best_distance_m": float(self.best_distance_m),
            "no_progress_steps": int(self.no_progress_steps),
            "position_tolerance_m": self.position_tolerance_m,
            "patience_steps": self.patience_steps,
            "min_progress_m": self.min_progress_m,
            "stop_guard_enabled": self.stop_guard_enabled,
            "meaningful_progress": bool(meaningful_progress),
        }
