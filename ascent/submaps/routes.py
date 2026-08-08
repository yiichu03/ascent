"""Persistent, read-only routes through the submap gateway graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ascent.submaps.types import SubmapGraph
from ascent.submaps.geometry import pose_to_matrix


@dataclass(frozen=True)
class RouteWaypoint:
    """One immutable waypoint expressed in its owning submap frame."""

    owner_submap_id: str
    local_xy: np.ndarray
    kind: str
    edge_id: Optional[str] = None

    def __post_init__(self) -> None:
        point = np.asarray(self.local_xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid route waypoint: {point}")
        object.__setattr__(self, "local_xy", point.copy())


@dataclass
class RemoteRoute:
    """A route selected once and advanced without mutating submap topology."""

    route_id: str
    target_kind: str
    candidate_key: str
    destination_submap_id: str
    waypoints: Tuple[RouteWaypoint, ...]
    selected_step: int
    frontier_id: Optional[str] = None
    cursor: int = 0
    waypoint_action_count: int = 0
    waypoint_stagnation_count: int = 0
    best_waypoint_distance_m: float = np.inf
    execution_submap_id: Optional[str] = None
    awaiting_local_replan: bool = False
    local_replan_exposed: bool = False
    owner_to_execution_transform: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        if self.target_kind not in {"semantic", "frontier"}:
            raise ValueError(f"Unsupported route target kind {self.target_kind!r}")
        if not self.waypoints:
            raise ValueError("A remote route requires at least one waypoint")
        if self.target_kind == "frontier" and not self.frontier_id:
            raise ValueError("A frontier route requires frontier_id")
        transform = np.asarray(
            self.owner_to_execution_transform, dtype=np.float64
        )
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("owner_to_execution_transform must be finite 3x3")
        self.owner_to_execution_transform = transform.copy()

    @classmethod
    def build(
        cls,
        *,
        graph: SubmapGraph,
        route_id: str,
        active_submap_id: str,
        destination_submap_id: str,
        destination_local_xy: np.ndarray,
        target_kind: str,
        candidate_key: str,
        selected_step: int,
        frontier_id: Optional[str] = None,
    ) -> "RemoteRoute":
        path = graph.shortest_path(active_submap_id, destination_submap_id)
        if not path:
            raise ValueError(
                "No gateway path from "
                f"{active_submap_id} to {destination_submap_id}"
            )
        waypoints = []
        for source_submap_id, next_submap_id in zip(path, path[1:]):
            edge = graph.edge_between(source_submap_id, next_submap_id)
            waypoints.append(
                RouteWaypoint(
                    owner_submap_id=source_submap_id,
                    local_xy=edge.endpoint_for(source_submap_id)[:2],
                    kind="gateway",
                    edge_id=edge.edge_id,
                )
            )
        waypoints.append(
            RouteWaypoint(
                owner_submap_id=destination_submap_id,
                local_xy=np.asarray(destination_local_xy, dtype=np.float64),
                kind=f"destination_{target_kind}",
            )
        )
        return cls(
            route_id=str(route_id),
            target_kind=str(target_kind),
            candidate_key=str(candidate_key),
            destination_submap_id=str(destination_submap_id),
            waypoints=tuple(waypoints),
            selected_step=int(selected_step),
            frontier_id=frontier_id,
            execution_submap_id=str(active_submap_id),
        )

    @property
    def complete(self) -> bool:
        return self.cursor >= len(self.waypoints)

    @property
    def current_waypoint(self) -> RouteWaypoint:
        if self.complete:
            raise RuntimeError(f"Route {self.route_id} is already complete")
        return self.waypoints[self.cursor]

    def advance(self) -> None:
        if self.complete:
            raise RuntimeError(f"Route {self.route_id} is already complete")
        self.cursor += 1
        self.waypoint_action_count = 0
        self.waypoint_stagnation_count = 0
        self.best_waypoint_distance_m = np.inf

    def pause_for_local_replan(self) -> None:
        """Pause after one gateway without completing the remote route."""

        if self.complete:
            raise RuntimeError(
                f"Route {self.route_id} cannot pause after completion"
            )
        self.awaiting_local_replan = True
        self.local_replan_exposed = False

    def expose_local_replan(self) -> bool:
        """Record the first native-planner handoff for the current pause."""

        if not self.awaiting_local_replan or self.local_replan_exposed:
            return False
        self.local_replan_exposed = True
        return True

    def resume_after_local_replan(self) -> None:
        """Resume the next route segment after local evidence is exhausted."""

        if not self.awaiting_local_replan:
            raise RuntimeError(
                f"Route {self.route_id} is not awaiting local replanning"
            )
        self.awaiting_local_replan = False
        self.local_replan_exposed = False

    def project_current_waypoint(self) -> np.ndarray:
        """Project the current point using only the route's local alignment."""

        waypoint = self.current_waypoint
        homogeneous = np.array(
            [waypoint.local_xy[0], waypoint.local_xy[1], 1.0],
            dtype=np.float64,
        )
        return (
            self.owner_to_execution_transform @ homogeneous
        )[:2]

    def rebase_execution_submap(
        self, graph: SubmapGraph, new_execution_submap_id: str
    ) -> None:
        """Carry route alignment across a lifecycle split using its gateway."""

        new_id = str(new_execution_submap_id)
        old_id = self.execution_submap_id
        if old_id == new_id:
            return
        if old_id is None:
            raise RuntimeError("Route has no execution submap")
        edge = graph.edge_between(old_id, new_id)
        old_endpoint = edge.endpoint_for(old_id)
        new_endpoint = edge.endpoint_for(new_id)
        new_from_old = (
            pose_to_matrix(new_endpoint)
            @ np.linalg.inv(pose_to_matrix(old_endpoint))
        )
        self.owner_to_execution_transform = (
            new_from_old @ self.owner_to_execution_transform
        )
        self.execution_submap_id = new_id

    def reach_current_waypoint(
        self,
        graph: SubmapGraph,
        actual_execution_xy: np.ndarray,
    ) -> None:
        """Advance and re-anchor translation when a gateway is reached."""

        waypoint = self.current_waypoint
        if waypoint.kind != "gateway":
            self.advance()
            return
        if waypoint.edge_id is None:
            raise RuntimeError("Gateway waypoint is missing edge_id")
        actual = np.asarray(actual_execution_xy, dtype=np.float64)
        if actual.shape != (2,) or not np.isfinite(actual).all():
            raise ValueError(f"Invalid gateway arrival point {actual}")

        edge = graph.edges[waypoint.edge_id]
        owner_id = waypoint.owner_submap_id
        next_owner_id = edge.other(owner_id)
        owner_endpoint = edge.endpoint_for(owner_id)
        next_endpoint = edge.endpoint_for(next_owner_id)

        aligned_execution_from_owner = (
            self.owner_to_execution_transform.copy()
        )
        rotation = aligned_execution_from_owner[:2, :2]
        aligned_execution_from_owner[:2, 2] = (
            actual - rotation @ owner_endpoint[:2]
        )
        owner_from_next = (
            pose_to_matrix(owner_endpoint)
            @ np.linalg.inv(pose_to_matrix(next_endpoint))
        )
        self.owner_to_execution_transform = (
            aligned_execution_from_owner @ owner_from_next
        )
        self.advance()

    def observe_distance(
        self,
        distance_m: float,
        *,
        min_progress_m: float,
        max_stagnation_actions: int,
        max_waypoint_actions: int,
    ) -> Optional[str]:
        """Update a bounded no-progress guard and return a failure reason."""

        distance = float(distance_m)
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError(f"Invalid waypoint distance {distance!r}")
        self.waypoint_action_count += 1
        if (
            not np.isfinite(self.best_waypoint_distance_m)
            or distance
            <= self.best_waypoint_distance_m - float(min_progress_m)
        ):
            self.best_waypoint_distance_m = distance
            self.waypoint_stagnation_count = 0
        else:
            self.best_waypoint_distance_m = min(
                self.best_waypoint_distance_m, distance
            )
            self.waypoint_stagnation_count += 1

        if self.waypoint_stagnation_count >= int(max_stagnation_actions):
            return "no_progress"
        if self.waypoint_action_count >= int(max_waypoint_actions):
            return "waypoint_action_limit"
        return None
