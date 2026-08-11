"""Place-conditioned, task-level residual search memory.

This module deliberately has no Habitat or ground-truth dependency.  It accepts
only an opaque same-place association naming a historical submap, then uses the
existing VO-anchored submap graph to query compact frontier-attempt outcomes.
Dense obstacle, value, and object maps remain submap-local and are never fused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ascent.submaps.geometry import transform_points_xy
from ascent.submaps.types import SubmapBundle, SubmapGraph


class SearchBranchStatus(str, Enum):
    UNRESOLVED = "unresolved"
    PRODUCTIVE = "productive"
    CONSUMED = "consumed"
    TEMP_BLOCKED = "temp_blocked"


@dataclass(frozen=True)
class OracleSamePlaceEvent:
    """The entire policy-visible output of the evaluation-only oracle."""

    event_sequence: int
    reference_submap_id: str

    def __post_init__(self) -> None:
        if int(self.event_sequence) < 1:
            raise ValueError("event_sequence must be positive")
        if not str(self.reference_submap_id):
            raise ValueError("reference_submap_id must be non-empty")


@dataclass(frozen=True)
class PlaceMemoryConfig:
    enabled: bool = False
    low_gain_area_m2: float = 0.5
    shadow_low_gain_area_m2: float = 1.0
    branch_association_radius_m: float = 0.5
    branch_match_radius_m: float = 1.0
    branch_match_margin_m: float = 0.25
    persistent_frontier_observations: int = 2
    minimum_arrival_observations: int = 2
    minimum_repeat_arrival_observations: int = 3
    minimum_excursion_start_distance_m: float = 1.4

    def __post_init__(self) -> None:
        if self.low_gain_area_m2 <= 0.0:
            raise ValueError("low_gain_area_m2 must be positive")
        if self.shadow_low_gain_area_m2 < self.low_gain_area_m2:
            raise ValueError(
                "shadow_low_gain_area_m2 must cover the decision threshold"
            )
        if self.branch_association_radius_m <= 0.0:
            raise ValueError("branch_association_radius_m must be positive")
        if self.branch_match_radius_m <= 0.0:
            raise ValueError("branch_match_radius_m must be positive")
        if not 0.0 < self.branch_match_margin_m < self.branch_match_radius_m:
            raise ValueError("branch_match_margin_m must be inside match radius")
        if self.persistent_frontier_observations < 2:
            raise ValueError("persistent frontier evidence needs at least two views")
        if self.minimum_arrival_observations < 2:
            raise ValueError("a settled excursion needs at least two arrival views")
        if (
            self.minimum_repeat_arrival_observations
            < self.minimum_arrival_observations
        ):
            raise ValueError(
                "a repeat cutoff cannot use fewer views than excursion settlement"
            )
        if self.minimum_excursion_start_distance_m <= 0.0:
            raise ValueError("minimum excursion distance must be positive")


@dataclass
class SearchAttempt:
    attempt_id: int
    target: str
    source_submap_id: str
    floor_id: int
    frontier_local: np.ndarray
    start_step: int
    start_explored_pixels: int
    pixels_per_meter: float
    start_target_present: bool
    start_up_stair_present: bool
    start_down_stair_present: bool
    start_frontiers: np.ndarray
    start_robot_distance_m: float
    min_robot_distance_m: float
    historical_branch_ids: Tuple[str, ...] = ()
    reached: bool = False
    arrival_observations: int = 0
    first_arrival_step: Optional[int] = None
    target_gain: bool = False
    exit_gain: bool = False
    previous_novel_frontiers: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    novel_frontier_streak: int = 0


@dataclass
class SearchBranchRecord:
    branch_id: str
    target: str
    source_submap_id: str
    floor_id: int
    local_xy: np.ndarray
    status: SearchBranchStatus
    attempts: int
    last_start_step: int
    last_end_step: int
    action_cost: int
    coverage_delta_m2: float
    shadow_low_gain: bool
    reached: bool
    target_gain: bool
    exit_gain: bool
    reason: str
    provisional_low_gain: bool
    qualified_excursions: int
    independent_revisit_confirmations: int
    last_arrival_observations: int
    last_start_robot_distance_m: float


@dataclass(frozen=True)
class BranchMatch:
    status: SearchBranchStatus
    distance_m: float
    branch_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RerankDecision:
    base_frontier: np.ndarray
    base_value: float
    final_frontier: np.ndarray
    final_value: float
    changed: bool
    reason: str
    place_id: Optional[str]
    base_status: Optional[str]
    alternative_status: Optional[str]
    intervention: Optional[str]
    base_branch_ids: Tuple[str, ...] = ()
    final_branch_ids: Tuple[str, ...] = ()


def _frontier_array(frontiers: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(frontiers, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    array = array.reshape(-1, 2)
    if not np.isfinite(array).all():
        raise ValueError("frontier coordinates must be finite")
    return array.copy()


def _xy(point: Sequence[float]) -> np.ndarray:
    array = np.asarray(point, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array[:2]).all():
        raise ValueError("frontier must contain finite XY")
    return array[:2].copy()


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(_xy(first) - _xy(second)))


class PlaceConditionedResidualMemory:
    """Episode-local search outcomes shared only by oracle-aliased submaps."""

    def __init__(
        self, num_envs: int, config: PlaceMemoryConfig
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not config.enabled:
            raise ValueError("place memory must be explicitly enabled")
        self.config = config
        self._submap_to_place: List[Dict[str, str]] = [
            {} for _ in range(num_envs)
        ]
        self._place_members: List[Dict[str, set[str]]] = [
            {} for _ in range(num_envs)
        ]
        self._branches: List[List[SearchBranchRecord]] = [
            [] for _ in range(num_envs)
        ]
        self._active: List[Optional[SearchAttempt]] = [
            None for _ in range(num_envs)
        ]
        self._attempt_sequence = [0 for _ in range(num_envs)]
        self._branch_sequence = [0 for _ in range(num_envs)]
        self._place_sequence = [0 for _ in range(num_envs)]
        self._events: List[List[Dict[str, Any]]] = [
            [] for _ in range(num_envs)
        ]

    def reset(self, env: int) -> None:
        self._submap_to_place[env] = {}
        self._place_members[env] = {}
        self._branches[env] = []
        self._active[env] = None
        self._attempt_sequence[env] = 0
        self._branch_sequence[env] = 0
        self._place_sequence[env] = 0
        self._events[env] = [{"event": "place_memory_reset"}]

    def drain_events(self, env: int) -> List[Dict[str, Any]]:
        events = [dict(event) for event in self._events[env]]
        self._events[env] = []
        return events

    def place_for_submap(self, env: int, submap_id: str) -> Optional[str]:
        return self._submap_to_place[env].get(str(submap_id))

    def branches(self, env: int) -> Tuple[SearchBranchRecord, ...]:
        return tuple(self._branches[env])

    def active_attempt(self, env: int) -> Optional[SearchAttempt]:
        return self._active[env]

    def accept_oracle_event(
        self,
        *,
        env: int,
        event: OracleSamePlaceEvent,
        current_submap_id: str,
        graph: SubmapGraph,
        step: int,
    ) -> bool:
        """Associate two task identities without receiving any metric oracle data."""

        if type(event) is not OracleSamePlaceEvent:
            raise TypeError("place memory accepts only OracleSamePlaceEvent")
        current_id = str(current_submap_id)
        reference_id = str(event.reference_submap_id)
        nodes = graph.nodes
        rejection: Optional[str] = None
        if current_id not in nodes or reference_id not in nodes:
            rejection = "unknown_submap"
        elif current_id == reference_id:
            rejection = "same_submap"
        elif nodes[current_id].floor_id != nodes[reference_id].floor_id:
            rejection = "different_policy_floor"
        else:
            try:
                graph.edge_between(current_id, reference_id)
            except KeyError:
                pass
            else:
                rejection = "graph_direct"
        if rejection is not None:
            self._events[env].append(
                {
                    "event": "place_association_rejected",
                    "step": int(step),
                    "oracle_event_sequence": int(event.event_sequence),
                    "current_submap_id": current_id,
                    "reference_submap_id": reference_id,
                    "reason": rejection,
                }
            )
            return False

        current_place = self._submap_to_place[env].get(current_id)
        reference_place = self._submap_to_place[env].get(reference_id)
        if reference_place is None and current_place is None:
            self._place_sequence[env] += 1
            place_id = f"env{env}:place{self._place_sequence[env]:04d}"
            self._place_members[env][place_id] = {reference_id, current_id}
        elif reference_place is None:
            place_id = str(current_place)
            self._place_members[env][place_id].add(reference_id)
        elif current_place is None:
            place_id = str(reference_place)
            self._place_members[env][place_id].add(current_id)
        elif current_place == reference_place:
            place_id = str(current_place)
        else:
            place_id = str(reference_place)
            merged_id = str(current_place)
            for member in self._place_members[env].pop(merged_id):
                self._place_members[env][place_id].add(member)
                self._submap_to_place[env][member] = place_id
        for member in self._place_members[env][place_id]:
            self._submap_to_place[env][member] = place_id
        self._events[env].append(
            {
                "event": "place_association_accepted",
                "step": int(step),
                "oracle_event_sequence": int(event.event_sequence),
                "place_id": place_id,
                "current_submap_id": current_id,
                "reference_submap_id": reference_id,
                "member_count": len(self._place_members[env][place_id]),
            }
        )
        # Identity association makes historical outcomes queryable; it is not
        # itself evidence that any historical search branch is exhausted.
        # A provisional low-gain branch must be selected and independently
        # tested again from the aliased submap before it can become CONSUMED.
        for record in self._branches[env]:
            if (
                record.source_submap_id
                not in self._place_members[env][place_id]
                or record.source_submap_id == current_id
                or record.status is not SearchBranchStatus.UNRESOLVED
                or not record.provisional_low_gain
            ):
                continue
            self._events[env].append(
                {
                    "event": "search_branch_revisit_available",
                    "step": int(step),
                    "oracle_event_sequence": int(event.event_sequence),
                    "place_id": place_id,
                    "branch_id": record.branch_id,
                    "source_submap_id": record.source_submap_id,
                    "reason": "same_place_identity_only",
                    "qualified_excursions": int(
                        record.qualified_excursions
                    ),
                    "action_cost": int(record.action_cost),
                    "coverage_delta_m2": float(
                        record.coverage_delta_m2
                    ),
                    "last_arrival_observations": int(
                        record.last_arrival_observations
                    ),
                    "last_start_robot_distance_m": float(
                        record.last_start_robot_distance_m
                    ),
                }
            )
        return True

    def _excursion_qualification(
        self, attempt: SearchAttempt
    ) -> Tuple[bool, str]:
        if not attempt.reached:
            return False, "frontier_not_reached"
        if (
            attempt.start_robot_distance_m
            < self.config.minimum_excursion_start_distance_m
        ):
            return False, "start_inside_excursion_distance"
        if (
            attempt.arrival_observations
            < self.config.minimum_arrival_observations
        ):
            return False, "insufficient_arrival_observations"
        return True, "qualified"

    def _selected_frontier_is_novel(
        self, attempt: SearchAttempt, selected: np.ndarray
    ) -> bool:
        if len(attempt.start_frontiers) == 0:
            return False
        return bool(
            float(
                np.min(
                    np.linalg.norm(
                        attempt.start_frontiers - selected,
                        axis=1,
                    )
                )
            )
            > self.config.branch_association_radius_m
        )

    @staticmethod
    def _coverage_pixels(obstacle_map: Any) -> int:
        return int(np.count_nonzero(np.asarray(obstacle_map.explored_area)))

    def _coverage_delta_m2(
        self, attempt: SearchAttempt, obstacle_map: Any
    ) -> float:
        delta = max(
            0, self._coverage_pixels(obstacle_map) - attempt.start_explored_pixels
        )
        return float(delta / (attempt.pixels_per_meter**2))

    def _repeat_trial_branch_ids(
        self,
        *,
        env: int,
        source_submap_id: str,
        branch_ids: Sequence[str],
    ) -> Tuple[str, ...]:
        requested = {str(branch_id) for branch_id in branch_ids}
        if not requested:
            return ()
        return tuple(
            sorted(
                record.branch_id
                for record in self._branches[env]
                if record.branch_id in requested
                and record.source_submap_id != str(source_submap_id)
                and record.status is SearchBranchStatus.UNRESOLVED
                and record.provisional_low_gain
            )
        )

    def start_attempt(
        self,
        *,
        env: int,
        target: str,
        source_submap_id: str,
        floor_id: int,
        frontier: Sequence[float],
        step: int,
        obstacle_map: Any,
        robot_xy: Sequence[float],
        target_present: bool,
        up_stair_present: bool,
        down_stair_present: bool,
        frontiers: Sequence[Sequence[float]],
        historical_branch_ids: Sequence[str] = (),
    ) -> None:
        self._attempt_sequence[env] += 1
        selected = _xy(frontier)
        start_robot_distance_m = _distance(robot_xy, selected)
        attempt = SearchAttempt(
            attempt_id=self._attempt_sequence[env],
            target=str(target),
            source_submap_id=str(source_submap_id),
            floor_id=int(floor_id),
            frontier_local=selected,
            start_step=int(step),
            start_explored_pixels=self._coverage_pixels(obstacle_map),
            pixels_per_meter=float(obstacle_map.pixels_per_meter),
            start_target_present=bool(target_present),
            start_up_stair_present=bool(up_stair_present),
            start_down_stair_present=bool(down_stair_present),
            start_frontiers=_frontier_array(frontiers),
            start_robot_distance_m=start_robot_distance_m,
            min_robot_distance_m=start_robot_distance_m,
            historical_branch_ids=self._repeat_trial_branch_ids(
                env=env,
                source_submap_id=source_submap_id,
                branch_ids=historical_branch_ids,
            ),
        )
        self._active[env] = attempt
        self._events[env].append(
            {
                "event": "search_attempt_started",
                "step": int(step),
                "attempt_id": attempt.attempt_id,
                "place_id": self.place_for_submap(
                    env, attempt.source_submap_id
                ),
                "source_submap_id": attempt.source_submap_id,
                "target": attempt.target,
                "frontier_local": selected.tolist(),
                "start_robot_distance_m": start_robot_distance_m,
                "minimum_excursion_start_distance_m": (
                    self.config.minimum_excursion_start_distance_m
                ),
                "historical_branch_ids": list(
                    attempt.historical_branch_ids
                ),
                "repeat_trial": bool(attempt.historical_branch_ids),
            }
        )

    def register_selection(
        self,
        *,
        env: int,
        target: str,
        source_submap_id: str,
        floor_id: int,
        selected_frontier: Sequence[float],
        step: int,
        obstacle_map: Any,
        robot_xy: Sequence[float],
        target_present: bool,
        up_stair_present: bool,
        down_stair_present: bool,
        frontiers: Sequence[Sequence[float]],
        historical_branch_ids: Sequence[str] = (),
    ) -> None:
        selected = _xy(selected_frontier)
        repeat_branch_ids = self._repeat_trial_branch_ids(
            env=env,
            source_submap_id=source_submap_id,
            branch_ids=historical_branch_ids,
        )
        active = self._active[env]
        if active is not None:
            same_context = bool(
                active.target == str(target)
                and active.source_submap_id == str(source_submap_id)
                and active.floor_id == int(floor_id)
            )
            same_branch = bool(
                same_context
                and _distance(active.frontier_local, selected)
                <= self.config.branch_association_radius_m
            )
            if same_branch:
                if repeat_branch_ids:
                    active.historical_branch_ids = tuple(
                        sorted(
                            set(active.historical_branch_ids).union(
                                repeat_branch_ids
                            )
                        )
                    )
                return
            qualified, _ = self._excursion_qualification(active)
            if same_context and self._selected_frontier_is_novel(
                active, selected
            ):
                active.exit_gain = True
                outcome = "completed"
                reason = "selected_new_frontier"
            elif same_context and qualified:
                outcome = "completed"
                reason = "independent_branch_departure"
            else:
                outcome = "interrupted"
                reason = "replanning_switch"
            self.finish_attempt(
                env=env,
                outcome=outcome,
                reason=reason,
                step=step,
                obstacle_map=obstacle_map,
            )
        self.start_attempt(
            env=env,
            target=target,
            source_submap_id=source_submap_id,
            floor_id=floor_id,
            frontier=selected,
            step=step,
            obstacle_map=obstacle_map,
            robot_xy=robot_xy,
            target_present=target_present,
            up_stair_present=up_stair_present,
            down_stair_present=down_stair_present,
            frontiers=frontiers,
            historical_branch_ids=repeat_branch_ids,
        )

    def _novel_frontiers(
        self, attempt: SearchAttempt, current: np.ndarray
    ) -> np.ndarray:
        if len(current) == 0:
            return np.empty((0, 2), dtype=np.float64)
        if len(attempt.start_frontiers) == 0:
            return current.copy()
        return np.asarray(
            [
                point
                for point in current
                if float(
                    np.min(
                        np.linalg.norm(attempt.start_frontiers - point, axis=1)
                    )
                )
                > self.config.branch_association_radius_m
            ],
            dtype=np.float64,
        ).reshape(-1, 2)

    def observe(
        self,
        *,
        env: int,
        target: str,
        source_submap_id: str,
        floor_id: int,
        step: int,
        robot_xy: Sequence[float],
        obstacle_map: Any,
        target_present: bool,
        up_stair_present: bool,
        down_stair_present: bool,
        frontiers: Sequence[Sequence[float]],
        arrival_radius_m: float,
    ) -> Optional[SearchBranchStatus]:
        attempt = self._active[env]
        if attempt is None:
            return None
        if attempt.source_submap_id != str(source_submap_id):
            self._events[env].append(
                {
                    "event": "search_attempt_dropped",
                    "step": int(step),
                    "attempt_id": attempt.attempt_id,
                    "reason": "submap_changed_without_settlement",
                }
            )
            self._active[env] = None
            return SearchBranchStatus.UNRESOLVED
        if attempt.target != str(target) or attempt.floor_id != int(floor_id):
            return self.finish_attempt(
                env=env,
                outcome="interrupted",
                reason="target_or_floor_change",
                step=step,
                obstacle_map=obstacle_map,
            )

        robot_distance = _distance(robot_xy, attempt.frontier_local)
        attempt.min_robot_distance_m = min(
            attempt.min_robot_distance_m, robot_distance
        )
        if robot_distance <= float(arrival_radius_m):
            if not attempt.reached:
                attempt.first_arrival_step = int(step)
            attempt.reached = True
            attempt.arrival_observations += 1
        if not attempt.start_target_present and bool(target_present):
            attempt.target_gain = True
        if (
            (not attempt.start_up_stair_present and bool(up_stair_present))
            or (
                not attempt.start_down_stair_present
                and bool(down_stair_present)
            )
        ):
            attempt.exit_gain = True

        current = _frontier_array(frontiers)
        novel = self._novel_frontiers(attempt, current)
        if len(novel) == 0:
            attempt.novel_frontier_streak = 0
        elif len(attempt.previous_novel_frontiers) == 0:
            attempt.novel_frontier_streak = 1
        else:
            persistent = any(
                float(
                    np.min(
                        np.linalg.norm(
                            attempt.previous_novel_frontiers - point, axis=1
                        )
                    )
                )
                <= self.config.branch_association_radius_m
                for point in novel
            )
            attempt.novel_frontier_streak = (
                attempt.novel_frontier_streak + 1 if persistent else 1
            )
        attempt.previous_novel_frontiers = novel
        if (
            attempt.novel_frontier_streak
            >= self.config.persistent_frontier_observations
        ):
            attempt.exit_gain = True

        if attempt.target_gain:
            return self.finish_attempt(
                env=env,
                outcome="completed",
                reason="target_evidence_found",
                step=step,
                obstacle_map=obstacle_map,
            )
        if attempt.exit_gain:
            return self.finish_attempt(
                env=env,
                outcome="completed",
                reason="persistent_new_exit",
                step=step,
                obstacle_map=obstacle_map,
            )
        return None

    def finish_attempt(
        self,
        *,
        env: int,
        outcome: str,
        reason: str,
        step: int,
        obstacle_map: Any,
    ) -> Optional[SearchBranchStatus]:
        attempt = self._active[env]
        if attempt is None:
            return None
        coverage = self._coverage_delta_m2(attempt, obstacle_map)
        excursion_qualified, qualification_reason = (
            self._excursion_qualification(attempt)
        )
        provisional_low_gain = False
        if outcome == "execution_failure":
            status = SearchBranchStatus.TEMP_BLOCKED
        elif outcome != "completed":
            status = SearchBranchStatus.UNRESOLVED
        elif attempt.target_gain or attempt.exit_gain or (
            coverage >= self.config.low_gain_area_m2
        ):
            status = SearchBranchStatus.PRODUCTIVE
        elif excursion_qualified:
            # A completed low-gain excursion is deliberately provisional.
            # Same-place identity only makes it queryable.  Suppression
            # authority requires another matched, independent low-gain trial.
            status = SearchBranchStatus.UNRESOLVED
            provisional_low_gain = True
        else:
            status = SearchBranchStatus.UNRESOLVED
        shadow_low_gain = bool(
            outcome == "completed"
            and coverage < self.config.shadow_low_gain_area_m2
            and not attempt.target_gain
            and not attempt.exit_gain
        )
        record = self._upsert_branch(
            env=env,
            attempt=attempt,
            status=status,
            end_step=int(step),
            coverage_delta_m2=coverage,
            shadow_low_gain=shadow_low_gain,
            reason=str(reason),
            provisional_low_gain=provisional_low_gain,
            excursion_qualified=excursion_qualified,
        )
        self._events[env].append(
            {
                "event": "search_attempt_finished",
                "step": int(step),
                "attempt_id": attempt.attempt_id,
                "branch_id": record.branch_id,
                "place_id": self.place_for_submap(
                    env, attempt.source_submap_id
                ),
                "source_submap_id": attempt.source_submap_id,
                "status": status.value,
                "outcome": str(outcome),
                "reason": str(reason),
                "reached": bool(attempt.reached),
                "first_arrival_step": attempt.first_arrival_step,
                "arrival_observations": int(attempt.arrival_observations),
                "minimum_arrival_observations": (
                    self.config.minimum_arrival_observations
                ),
                "start_robot_distance_m": float(
                    attempt.start_robot_distance_m
                ),
                "minimum_excursion_start_distance_m": (
                    self.config.minimum_excursion_start_distance_m
                ),
                "excursion_qualified": excursion_qualified,
                "excursion_qualification_reason": qualification_reason,
                "provisional_low_gain": provisional_low_gain,
                "coverage_delta_m2": float(coverage),
                "low_gain_threshold_m2": self.config.low_gain_area_m2,
                "shadow_low_gain_threshold_m2": (
                    self.config.shadow_low_gain_area_m2
                ),
                "shadow_low_gain": shadow_low_gain,
                "target_gain": bool(attempt.target_gain),
                "exit_gain": bool(attempt.exit_gain),
                "action_cost": int(step) - attempt.start_step,
            }
        )
        self._settle_repeat_trial(
            env=env,
            attempt=attempt,
            record=record,
            status=status,
            outcome=str(outcome),
            step=int(step),
            provisional_low_gain=provisional_low_gain,
            excursion_qualified=excursion_qualified,
        )
        self._active[env] = None
        return status

    def _settle_repeat_trial(
        self,
        *,
        env: int,
        attempt: SearchAttempt,
        record: SearchBranchRecord,
        status: SearchBranchStatus,
        outcome: str,
        step: int,
        provisional_low_gain: bool,
        excursion_qualified: bool,
    ) -> None:
        if not attempt.historical_branch_ids:
            return
        requested = set(attempt.historical_branch_ids)
        historical = [
            item
            for item in self._branches[env]
            if item.branch_id in requested
            and item.source_submap_id != attempt.source_submap_id
        ]
        if not historical:
            return

        if (
            status is SearchBranchStatus.PRODUCTIVE
            or record.status is SearchBranchStatus.PRODUCTIVE
        ):
            record.status = SearchBranchStatus.PRODUCTIVE
            record.provisional_low_gain = False
            for item in historical:
                item.status = SearchBranchStatus.PRODUCTIVE
                item.provisional_low_gain = False
            self._events[env].append(
                {
                    "event": "search_branch_revisit_productive",
                    "step": int(step),
                    "attempt_id": attempt.attempt_id,
                    "source_submap_id": attempt.source_submap_id,
                    "historical_branch_ids": sorted(requested),
                    "reason": "repeat_trial_found_new_information",
                    "coverage_delta_m2": float(record.coverage_delta_m2),
                    "target_gain": bool(attempt.target_gain),
                    "exit_gain": bool(attempt.exit_gain),
                }
            )
            return

        if not (
            outcome == "completed"
            and excursion_qualified
            and attempt.arrival_observations
            >= self.config.minimum_repeat_arrival_observations
            and provisional_low_gain
        ):
            self._events[env].append(
                {
                    "event": "search_branch_revisit_inconclusive",
                    "step": int(step),
                    "attempt_id": attempt.attempt_id,
                    "source_submap_id": attempt.source_submap_id,
                    "historical_branch_ids": sorted(requested),
                    "reason": "repeat_trial_not_qualified_low_gain",
                    "outcome": str(outcome),
                    "excursion_qualified": bool(excursion_qualified),
                    "arrival_observations": int(
                        attempt.arrival_observations
                    ),
                    "minimum_repeat_arrival_observations": (
                        self.config.minimum_repeat_arrival_observations
                    ),
                }
            )
            return

        # The same branch has now produced two independent, qualified low-gain
        # excursions in aliased submaps.  Only this second task-value
        # observation—not the identity association—grants suppression authority.
        lineage = {item.branch_id: item for item in historical}
        lineage[record.branch_id] = record
        for item in lineage.values():
            item.status = SearchBranchStatus.CONSUMED
            item.provisional_low_gain = False
            item.independent_revisit_confirmations += 1
            self._events[env].append(
                {
                    "event": "search_branch_consumed",
                    "step": int(step),
                    "attempt_id": attempt.attempt_id,
                    "place_id": self.place_for_submap(
                        env, attempt.source_submap_id
                    ),
                    "branch_id": item.branch_id,
                    "source_submap_id": item.source_submap_id,
                    "confirmation_source_submap_id": (
                        attempt.source_submap_id
                    ),
                    "historical_branch_ids": sorted(requested),
                    "reason": "matched_repeat_low_gain_excursion",
                    "qualified_excursions": int(item.qualified_excursions),
                    "action_cost": int(item.action_cost),
                    "coverage_delta_m2": float(item.coverage_delta_m2),
                    "last_arrival_observations": int(
                        attempt.arrival_observations
                    ),
                    "last_start_robot_distance_m": float(
                        attempt.start_robot_distance_m
                    ),
                }
            )

    def _upsert_branch(
        self,
        *,
        env: int,
        attempt: SearchAttempt,
        status: SearchBranchStatus,
        end_step: int,
        coverage_delta_m2: float,
        shadow_low_gain: bool,
        reason: str,
        provisional_low_gain: bool,
        excursion_qualified: bool,
    ) -> SearchBranchRecord:
        matches = [
            record
            for record in self._branches[env]
            if record.target == attempt.target
            and record.source_submap_id == attempt.source_submap_id
            and record.floor_id == attempt.floor_id
            and _distance(record.local_xy, attempt.frontier_local)
            <= self.config.branch_association_radius_m
        ]
        if matches:
            record = min(
                matches,
                key=lambda item: _distance(
                    item.local_xy, attempt.frontier_local
                ),
            )
            if status in {
                SearchBranchStatus.PRODUCTIVE,
                SearchBranchStatus.CONSUMED,
            } or record.status in {
                SearchBranchStatus.UNRESOLVED,
                SearchBranchStatus.TEMP_BLOCKED,
            }:
                record.status = status
            if status is SearchBranchStatus.PRODUCTIVE:
                record.provisional_low_gain = False
            elif (
                provisional_low_gain
                and record.status is not SearchBranchStatus.PRODUCTIVE
            ):
                record.provisional_low_gain = True
            if excursion_qualified:
                record.qualified_excursions += 1
            record.attempts += 1
            record.last_start_step = attempt.start_step
            record.last_end_step = int(end_step)
            record.action_cost += int(end_step) - attempt.start_step
            record.coverage_delta_m2 = max(
                record.coverage_delta_m2, float(coverage_delta_m2)
            )
            record.shadow_low_gain = bool(shadow_low_gain)
            record.reached = bool(record.reached or attempt.reached)
            record.target_gain = bool(
                record.target_gain or attempt.target_gain
            )
            record.exit_gain = bool(record.exit_gain or attempt.exit_gain)
            record.reason = str(reason)
            if provisional_low_gain or not record.provisional_low_gain:
                record.last_arrival_observations = int(
                    attempt.arrival_observations
                )
                record.last_start_robot_distance_m = float(
                    attempt.start_robot_distance_m
                )
            return record
        self._branch_sequence[env] += 1
        record = SearchBranchRecord(
            branch_id=f"env{env}:branch{self._branch_sequence[env]:05d}",
            target=attempt.target,
            source_submap_id=attempt.source_submap_id,
            floor_id=attempt.floor_id,
            local_xy=attempt.frontier_local.copy(),
            status=status,
            attempts=1,
            last_start_step=attempt.start_step,
            last_end_step=int(end_step),
            action_cost=int(end_step) - attempt.start_step,
            coverage_delta_m2=float(coverage_delta_m2),
            shadow_low_gain=bool(shadow_low_gain),
            reached=bool(attempt.reached),
            target_gain=bool(attempt.target_gain),
            exit_gain=bool(attempt.exit_gain),
            reason=str(reason),
            provisional_low_gain=bool(provisional_low_gain),
            qualified_excursions=(1 if excursion_qualified else 0),
            independent_revisit_confirmations=0,
            last_arrival_observations=int(attempt.arrival_observations),
            last_start_robot_distance_m=float(
                attempt.start_robot_distance_m
            ),
        )
        self._branches[env].append(record)
        return record

    def interrupt_active(
        self,
        *,
        env: int,
        reason: str,
        step: int,
        obstacle_map: Any,
    ) -> Optional[SearchBranchStatus]:
        attempt = self._active[env]
        if attempt is None:
            return None
        qualified, _ = self._excursion_qualification(attempt)
        return self.finish_attempt(
            env=env,
            outcome="completed" if qualified else "interrupted",
            reason=(
                f"qualified_boundary:{reason}" if qualified else reason
            ),
            step=step,
            obstacle_map=obstacle_map,
        )

    def mark_execution_failure(
        self,
        *,
        env: int,
        reason: str,
        step: int,
        obstacle_map: Any,
    ) -> Optional[SearchBranchStatus]:
        return self.finish_attempt(
            env=env,
            outcome="execution_failure",
            reason=reason,
            step=step,
            obstacle_map=obstacle_map,
        )

    def _place_branch_groups(
        self,
        *,
        env: int,
        target: str,
        active: SubmapBundle,
        graph: SubmapGraph,
    ) -> List[Tuple[np.ndarray, List[SearchBranchRecord]]]:
        place_id = self.place_for_submap(env, active.submap_id)
        if place_id is None:
            return []
        members = self._place_members[env][place_id]
        projected: List[Tuple[np.ndarray, SearchBranchRecord]] = []
        for record in self._branches[env]:
            if (
                record.target != str(target)
                or record.source_submap_id not in members
                or record.source_submap_id == active.submap_id
                or record.floor_id != active.floor_id
            ):
                continue
            source = graph.nodes.get(record.source_submap_id)
            if source is None:
                continue
            point = transform_points_xy(
                record.local_xy,
                source.anchor_pose_world,
                active.anchor_pose_world,
            )
            projected.append((_xy(point), record))

        groups: List[Tuple[np.ndarray, List[SearchBranchRecord]]] = []
        cluster_radius = self.config.branch_association_radius_m
        for point, record in projected:
            for index, (center, records) in enumerate(groups):
                if _distance(center, point) <= cluster_radius:
                    new_records = records + [record]
                    new_center = np.mean(
                        np.asarray(
                            [
                                transform_points_xy(
                                    item.local_xy,
                                    graph.nodes[
                                        item.source_submap_id
                                    ].anchor_pose_world,
                                    active.anchor_pose_world,
                                )
                                for item in new_records
                            ]
                        ),
                        axis=0,
                    )
                    groups[index] = (_xy(new_center), new_records)
                    break
            else:
                groups.append((point.copy(), [record]))
        return groups

    @staticmethod
    def _group_status(
        records: Sequence[SearchBranchRecord],
    ) -> SearchBranchStatus:
        statuses = {record.status for record in records}
        if statuses == {SearchBranchStatus.CONSUMED}:
            return SearchBranchStatus.CONSUMED
        if SearchBranchStatus.PRODUCTIVE in statuses:
            return SearchBranchStatus.PRODUCTIVE
        if SearchBranchStatus.UNRESOLVED in statuses:
            return SearchBranchStatus.UNRESOLVED
        return SearchBranchStatus.TEMP_BLOCKED

    def match_historical_branch(
        self,
        *,
        env: int,
        target: str,
        active: SubmapBundle,
        graph: SubmapGraph,
        frontier: Sequence[float],
    ) -> Tuple[Optional[BranchMatch], str]:
        groups = self._place_branch_groups(
            env=env, target=target, active=active, graph=graph
        )
        candidate = _xy(frontier)
        matches = sorted(
            (
                (_distance(candidate, center), records)
                for center, records in groups
                if _distance(candidate, center)
                <= self.config.branch_match_radius_m
            ),
            key=lambda item: item[0],
        )
        if not matches:
            return None, "untried"
        if (
            len(matches) > 1
            and matches[1][0] - matches[0][0]
            < self.config.branch_match_margin_m
        ):
            return None, "ambiguous"
        distance, records = matches[0]
        return (
            BranchMatch(
                status=self._group_status(records),
                distance_m=float(distance),
                branch_ids=tuple(sorted(item.branch_id for item in records)),
            ),
            "matched",
        )

    def _residual_candidates(
        self,
        *,
        env: int,
        target: str,
        active: SubmapBundle,
        graph: SubmapGraph,
        base: np.ndarray,
        sorted_frontiers: Sequence[Sequence[float]],
        sorted_values: Sequence[float],
        topk: int,
        excluded_branch_ids: Sequence[str] = (),
    ) -> List[Tuple[np.ndarray, float, str, Tuple[str, ...]]]:
        points = _frontier_array(sorted_frontiers)
        values = [float(value) for value in sorted_values]
        candidates: List[
            Tuple[np.ndarray, float, str, Tuple[str, ...]]
        ] = []
        excluded = set(str(item) for item in excluded_branch_ids)
        limit = min(len(points), max(2, int(topk)))
        for point, value in zip(points[:limit], values[:limit]):
            # A frontier micro-refresh is still the same search branch and
            # cannot serve as the residual opportunity that justifies a
            # cutoff.
            if (
                _distance(point, base)
                <= self.config.branch_association_radius_m
            ):
                continue
            match, match_reason = self.match_historical_branch(
                env=env,
                target=target,
                active=active,
                graph=graph,
                frontier=point,
            )
            if match is None and match_reason == "untried":
                candidates.append((point.copy(), value, "untried", ()))
            elif (
                match is not None
                and match.status is SearchBranchStatus.UNRESOLVED
                and not excluded.intersection(match.branch_ids)
            ):
                candidates.append(
                    (point.copy(), value, "unresolved", match.branch_ids)
                )
        return candidates

    def _repeat_cutoff_ready(
        self,
        *,
        env: int,
        target: str,
        active: SubmapBundle,
        base: np.ndarray,
        base_match: BranchMatch,
        obstacle_map: Any,
    ) -> Tuple[bool, str, float]:
        """Return whether the current repeat has enough task evidence to stop.

        The first historical low-gain excursion remains only provisional.  A
        second aliased-submap trial can gain suppression authority while it is
        still the live ASCENT choice, but only after physical travel, a
        multi-frame arrival, and no target, exit, or coverage gain.  This keeps
        settlement inside the useful decision window instead of waiting until
        ASCENT has already departed for another branch.
        """

        attempt = self._active[env]
        if attempt is None:
            return False, "no_active_attempt", 0.0
        coverage = self._coverage_delta_m2(attempt, obstacle_map)
        if (
            attempt.target != str(target)
            or attempt.source_submap_id != active.submap_id
            or attempt.floor_id != active.floor_id
        ):
            return False, "active_context_mismatch", coverage
        if (
            _distance(attempt.frontier_local, base)
            > self.config.branch_association_radius_m
        ):
            return False, "active_branch_mismatch", coverage
        if not set(base_match.branch_ids).issubset(
            attempt.historical_branch_ids
        ):
            return False, "not_a_matched_repeat", coverage
        if any(
            record.target == attempt.target
            and record.source_submap_id == attempt.source_submap_id
            and record.floor_id == attempt.floor_id
            and record.status is SearchBranchStatus.PRODUCTIVE
            and _distance(record.local_xy, attempt.frontier_local)
            <= self.config.branch_association_radius_m
            for record in self._branches[env]
        ):
            return False, "current_branch_already_productive", coverage
        qualified, qualification_reason = self._excursion_qualification(
            attempt
        )
        if not qualified:
            return False, qualification_reason, coverage
        if (
            attempt.arrival_observations
            < self.config.minimum_repeat_arrival_observations
        ):
            return False, "insufficient_repeat_arrival_observations", coverage
        if attempt.target_gain or attempt.exit_gain:
            return False, "repeat_found_discrete_information", coverage
        if coverage >= self.config.low_gain_area_m2:
            return False, "repeat_found_coverage", coverage
        return True, "qualified_repeat_low_gain", coverage

    def arbitrate(
        self,
        *,
        env: int,
        target: str,
        active: SubmapBundle,
        graph: SubmapGraph,
        base_frontier: Sequence[float],
        base_value: float,
        sorted_frontiers: Sequence[Sequence[float]],
        sorted_values: Sequence[float],
        topk: int,
        step: int,
        obstacle_map: Any,
    ) -> RerankDecision:
        base = _xy(base_frontier)
        place_id = self.place_for_submap(env, active.submap_id)
        if place_id is None:
            return RerankDecision(
                base, float(base_value), base.copy(), float(base_value), False,
                "no_place_association", None, None, None, None,
            )
        base_match, base_match_reason = self.match_historical_branch(
            env=env,
            target=target,
            active=active,
            graph=graph,
            frontier=base,
        )
        candidates: Optional[
            List[Tuple[np.ndarray, float, str, Tuple[str, ...]]]
        ] = None
        if (
            base_match is not None
            and base_match.status is SearchBranchStatus.UNRESOLVED
        ):
            candidates = self._residual_candidates(
                env=env,
                target=target,
                active=active,
                graph=graph,
                base=base,
                sorted_frontiers=sorted_frontiers,
                sorted_values=sorted_values,
                topk=topk,
                excluded_branch_ids=base_match.branch_ids,
            )
            ready, cutoff_reason, coverage = self._repeat_cutoff_ready(
                env=env,
                target=target,
                active=active,
                base=base,
                base_match=base_match,
                obstacle_map=obstacle_map,
            )
            # Do not settle the live repeat unless ASCENT already exposes a
            # live residual alternative.  Without one, exact v1.2 behavior is
            # the only safe fallback.
            if ready and candidates:
                attempt = self._active[env]
                assert attempt is not None
                self._events[env].append(
                    {
                        "event": "search_branch_repeat_cutoff",
                        "step": int(step),
                        "attempt_id": attempt.attempt_id,
                        "place_id": place_id,
                        "source_submap_id": attempt.source_submap_id,
                        "historical_branch_ids": list(
                            attempt.historical_branch_ids
                        ),
                        "reason": cutoff_reason,
                        "arrival_observations": int(
                            attempt.arrival_observations
                        ),
                        "minimum_repeat_arrival_observations": (
                            self.config.minimum_repeat_arrival_observations
                        ),
                        "start_robot_distance_m": float(
                            attempt.start_robot_distance_m
                        ),
                        "coverage_delta_m2": float(coverage),
                        "low_gain_threshold_m2": (
                            self.config.low_gain_area_m2
                        ),
                        "residual_alternative_count": len(candidates),
                    }
                )
                self.finish_attempt(
                    env=env,
                    outcome="completed",
                    reason="matched_repeat_low_gain_cutoff",
                    step=int(step),
                    obstacle_map=obstacle_map,
                )
                base_match, base_match_reason = (
                    self.match_historical_branch(
                        env=env,
                        target=target,
                        active=active,
                        graph=graph,
                        frontier=base,
                    )
                )
                if (
                    base_match is None
                    or base_match.status is not SearchBranchStatus.CONSUMED
                ):
                    raise RuntimeError(
                        "qualified repeat cutoff did not consume its historical lineage"
                    )
        if base_match is None or base_match.status is not SearchBranchStatus.CONSUMED:
            reason = (
                f"base_{base_match_reason}"
                if base_match is None
                else f"base_{base_match.status.value}"
            )
            decision = RerankDecision(
                base,
                float(base_value),
                base.copy(),
                float(base_value),
                False,
                reason,
                place_id,
                None if base_match is None else base_match.status.value,
                None,
                None,
                base_branch_ids=(
                    () if base_match is None else base_match.branch_ids
                ),
                final_branch_ids=(
                    () if base_match is None else base_match.branch_ids
                ),
            )
            self._events[env].append(
                {
                    "event": "place_rerank_evaluated",
                    "step": int(step),
                    "place_id": place_id,
                    "decision_changed": False,
                    "reason": decision.reason,
                    "base_frontier": base.tolist(),
                    "base_status": decision.base_status,
                    "base_branch_ids": list(decision.base_branch_ids),
                }
            )
            return decision

        if candidates is None:
            candidates = self._residual_candidates(
                env=env,
                target=target,
                active=active,
                graph=graph,
                base=base,
                sorted_frontiers=sorted_frontiers,
                sorted_values=sorted_values,
                topk=topk,
                excluded_branch_ids=base_match.branch_ids,
            )
        if not candidates:
            decision = RerankDecision(
                base,
                float(base_value),
                base.copy(),
                float(base_value),
                False,
                "no_live_residual_alternative",
                place_id,
                base_match.status.value,
                None,
                None,
                base_branch_ids=base_match.branch_ids,
                final_branch_ids=base_match.branch_ids,
            )
            self._events[env].append(
                {
                    "event": "place_rerank_evaluated",
                    "step": int(step),
                    "place_id": place_id,
                    "decision_changed": False,
                    "reason": decision.reason,
                    "base_frontier": base.tolist(),
                    "base_status": base_match.status.value,
                }
            )
            return decision

        # Candidates retain ASCENT/ValueMap order.  The memory changes only
        # eligibility; it never installs a second ranker or a sticky route.
        selected = candidates[0]
        active_attempt = self._active[env]
        final, final_value, alternative_status, final_branch_ids = selected
        intervention = (
            "continued"
            if active_attempt is not None
            and active_attempt.source_submap_id == active.submap_id
            and _distance(active_attempt.frontier_local, final)
            <= self.config.branch_association_radius_m
            else "started"
        )
        decision = RerankDecision(
            base,
            float(base_value),
            final.copy(),
            float(final_value),
            True,
            "consumed_to_residual",
            place_id,
            base_match.status.value,
            alternative_status,
            intervention,
            base_branch_ids=base_match.branch_ids,
            final_branch_ids=final_branch_ids,
        )
        self._events[env].append(
            {
                "event": "place_rerank_evaluated",
                "step": int(step),
                "place_id": place_id,
                "decision_changed": True,
                "reason": decision.reason,
                "intervention": intervention,
                "base_frontier": base.tolist(),
                "base_value": float(base_value),
                "base_status": base_match.status.value,
                "base_branch_ids": list(base_match.branch_ids),
                "base_match_distance_m": base_match.distance_m,
                "final_frontier": final.tolist(),
                "final_value": float(final_value),
                "alternative_status": alternative_status,
                "final_branch_ids": list(final_branch_ids),
            }
        )
        return decision

    def trace(self, env: int) -> Dict[str, Any]:
        return {
            "enabled": True,
            "place_count": len(self._place_members[env]),
            "associated_submap_count": len(self._submap_to_place[env]),
            "branch_count": len(self._branches[env]),
            "active_attempt_id": (
                None
                if self._active[env] is None
                else self._active[env].attempt_id
            ),
            "status_counts": {
                status.value: sum(
                    record.status is status for record in self._branches[env]
                )
                for status in SearchBranchStatus
            },
            "provisional_low_gain_count": sum(
                record.provisional_low_gain
                for record in self._branches[env]
            ),
            "independent_revisit_confirmation_count": sum(
                record.independent_revisit_confirmations
                for record in self._branches[env]
            ),
        }
