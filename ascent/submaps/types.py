"""Typed state for submap-local ASCENT memory."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ascent.submaps.geometry import as_pose


class SubmapState(str, Enum):
    ACTIVE = "active"
    FROZEN = "frozen"


class FrontierStatus(str, Enum):
    ACTIVE = "active"
    SELECTED = "selected"
    ATTEMPTED = "attempted"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"
    DISABLED = "disabled"


@dataclass
class MapPayload:
    """References to one ASCENT obstacle/value/object map triplet."""

    obstacle_map: Any
    value_map: Any
    object_map: Any

    @property
    def identity(self) -> Tuple[int, int, int]:
        return (id(self.obstacle_map), id(self.value_map), id(self.object_map))


@dataclass
class FrontierRecord:
    frontier_id: str
    source_submap_id: str
    local_xy: np.ndarray
    creation_step: int
    score: float = 0.0
    status: FrontierStatus = FrontierStatus.ACTIVE
    lineage: Optional[str] = None
    resolved_by: Optional[str] = None
    superseded_by: Optional[str] = None
    selection_count: int = 0
    attempt_count: int = 0
    last_update_step: int = 0

    def __post_init__(self) -> None:
        point = np.asarray(self.local_xy, dtype=np.float64)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid frontier point: {point}")
        self.local_xy = point.copy()

    @property
    def eligible(self) -> bool:
        return self.status in {
            FrontierStatus.ACTIVE,
            FrontierStatus.SELECTED,
        }


@dataclass
class GatewayEdge:
    edge_id: str
    source_submap_id: str
    destination_submap_id: str
    source_local_pose: np.ndarray
    destination_local_pose: np.ndarray
    creation_step: int
    source_floor_id: int
    destination_floor_id: int
    relative_transform: np.ndarray
    confidence: float = 1.0
    kind: str = "sequential"
    boundary_frames: Tuple[Dict[str, Any], ...] = ()
    traversal_count: int = 0
    last_outcome: Optional[str] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        self.source_local_pose = as_pose(self.source_local_pose)
        self.destination_local_pose = as_pose(self.destination_local_pose)
        transform = np.asarray(self.relative_transform, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("Gateway relative_transform must be a finite 3x3 matrix")
        self.relative_transform = transform.copy()
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Gateway confidence must be in [0, 1]")

    def other(self, submap_id: str) -> str:
        if submap_id == self.source_submap_id:
            return self.destination_submap_id
        if submap_id == self.destination_submap_id:
            return self.source_submap_id
        raise KeyError(f"Submap {submap_id} is not incident to {self.edge_id}")

    def endpoint_for(self, submap_id: str) -> np.ndarray:
        if submap_id == self.source_submap_id:
            return self.source_local_pose.copy()
        if submap_id == self.destination_submap_id:
            return self.destination_local_pose.copy()
        raise KeyError(f"Submap {submap_id} is not incident to {self.edge_id}")


@dataclass
class SubmapBundle:
    submap_id: str
    floor_id: int
    anchor_pose_world: np.ndarray
    creation_step: int
    payload: MapPayload
    state: SubmapState = SubmapState.ACTIVE
    freeze_step: Optional[int] = None
    accepted_action_endpoints: int = 0
    path_length_m: float = 0.0
    accumulated_rotation_rad: float = 0.0
    last_world_pose: Optional[np.ndarray] = None
    latest_overlap: Optional[float] = None
    low_overlap_streak: int = 0
    frozen_frontiers_local: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    parent_submap_id: Optional[str] = None
    reference_submap_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.anchor_pose_world = as_pose(self.anchor_pose_world)
        if self.last_world_pose is None:
            self.last_world_pose = self.anchor_pose_world.copy()
        else:
            self.last_world_pose = as_pose(self.last_world_pose)

    def require_active(self) -> None:
        if self.state is not SubmapState.ACTIVE:
            raise RuntimeError(f"Submap {self.submap_id} is frozen and cannot be written")

    def freeze(self, step: int, frontiers_local: np.ndarray) -> None:
        self.require_active()
        frontiers = np.asarray(frontiers_local, dtype=np.float64)
        if frontiers.size == 0:
            frontiers = np.empty((0, 2), dtype=np.float64)
        if frontiers.ndim != 2 or frontiers.shape[1] != 2:
            raise ValueError(f"Invalid frontier array shape {frontiers.shape}")
        if not np.isfinite(frontiers).all():
            raise ValueError("Frozen frontiers must be finite")
        self.frozen_frontiers_local = frontiers.copy()
        self.frozen_frontiers_local.setflags(write=False)
        self.anchor_pose_world.setflags(write=False)
        for value in self.metadata.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self.state = SubmapState.FROZEN
        self.freeze_step = int(step)


class SubmapGraph:
    """Small online undirected graph with explicit gateway endpoints."""

    def __init__(self) -> None:
        self._nodes: Dict[str, SubmapBundle] = {}
        self._edges: Dict[str, GatewayEdge] = {}
        self._adjacency: Dict[str, List[str]] = {}

    @property
    def nodes(self) -> Dict[str, SubmapBundle]:
        return dict(self._nodes)

    @property
    def edges(self) -> Dict[str, GatewayEdge]:
        return dict(self._edges)

    def add_node(self, bundle: SubmapBundle) -> None:
        if bundle.submap_id in self._nodes:
            raise ValueError(f"Duplicate submap id {bundle.submap_id}")
        self._nodes[bundle.submap_id] = bundle
        self._adjacency[bundle.submap_id] = []

    def get_node(self, submap_id: str) -> SubmapBundle:
        return self._nodes[submap_id]

    def add_edge(self, edge: GatewayEdge) -> None:
        if edge.edge_id in self._edges:
            raise ValueError(f"Duplicate gateway edge id {edge.edge_id}")
        if edge.source_submap_id not in self._nodes:
            raise KeyError(edge.source_submap_id)
        if edge.destination_submap_id not in self._nodes:
            raise KeyError(edge.destination_submap_id)
        self._edges[edge.edge_id] = edge
        self._adjacency[edge.source_submap_id].append(edge.edge_id)
        self._adjacency[edge.destination_submap_id].append(edge.edge_id)

    def shortest_path(self, source: str, destination: str) -> List[str]:
        if source not in self._nodes or destination not in self._nodes:
            raise KeyError(f"Unknown graph endpoint: {source}, {destination}")
        if source == destination:
            return [source]

        queue = deque([source])
        parents: Dict[str, Tuple[str, str]] = {}
        visited = {source}
        while queue:
            current = queue.popleft()
            for edge_id in self._adjacency[current]:
                edge = self._edges[edge_id]
                if not edge.enabled:
                    continue
                neighbor = edge.other(current)
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parents[neighbor] = (current, edge_id)
                if neighbor == destination:
                    queue.clear()
                    break
                queue.append(neighbor)

        if destination not in parents:
            return []
        reverse_path = [destination]
        cursor = destination
        while cursor != source:
            cursor = parents[cursor][0]
            reverse_path.append(cursor)
        return list(reversed(reverse_path))

    def edge_between(self, first: str, second: str) -> GatewayEdge:
        for edge_id in self._adjacency.get(first, []):
            edge = self._edges[edge_id]
            if edge.enabled and edge.other(first) == second:
                return edge
        raise KeyError(f"No enabled gateway between {first} and {second}")

    def next_gateway(self, source: str, destination: str) -> Optional[GatewayEdge]:
        path = self.shortest_path(source, destination)
        if len(path) < 2:
            return None
        return self.edge_between(path[0], path[1])
