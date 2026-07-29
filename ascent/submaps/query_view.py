"""Read-only query adapter over immutable local submaps and mutable topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ascent.submaps.frontier_registry import FrontierRegistry
from ascent.submaps.geometry import transform_points_xy
from ascent.submaps.types import FrontierRecord, GatewayEdge, SubmapGraph


@dataclass(frozen=True)
class RemoteFrontierPlan:
    frontier_id: str
    source_submap_id: str
    next_gateway_edge_id: str
    next_gateway_local_xy: np.ndarray
    graph_hops: int


class SubmapQueryView:
    """A transient view; all returned arrays are copies."""

    def __init__(
        self,
        graph: SubmapGraph,
        registry: FrontierRegistry,
        active_submap_id: str,
    ) -> None:
        graph.get_node(active_submap_id)
        self._graph = graph
        self._registry = registry
        self.active_submap_id = active_submap_id

    def project_points_to_active(
        self, source_submap_id: str, points_local: np.ndarray
    ) -> np.ndarray:
        source = self._graph.get_node(source_submap_id)
        active = self._graph.get_node(self.active_submap_id)
        return transform_points_xy(
            points_local,
            source.anchor_pose_world,
            active.anchor_pose_world,
        )

    def remote_frontier_plans(self) -> List[RemoteFrontierPlan]:
        candidates: List[Tuple[int, float, FrontierRecord, GatewayEdge]] = []
        for frontier in self._registry.eligible():
            if frontier.source_submap_id == self.active_submap_id:
                continue
            path = self._graph.shortest_path(
                self.active_submap_id, frontier.source_submap_id
            )
            if len(path) < 2:
                continue
            edge = self._graph.edge_between(path[0], path[1])
            candidates.append(
                (len(path) - 1, -frontier.score, frontier, edge)
            )

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2].creation_step,
                item[2].frontier_id,
            )
        )
        return [
            RemoteFrontierPlan(
                frontier_id=frontier.frontier_id,
                source_submap_id=frontier.source_submap_id,
                next_gateway_edge_id=edge.edge_id,
                next_gateway_local_xy=edge.endpoint_for(
                    self.active_submap_id
                )[:2],
                graph_hops=hops,
            )
            for hops, _, frontier, edge in candidates
        ]

    def best_remote_frontier_plan(self) -> Optional[RemoteFrontierPlan]:
        plans = self.remote_frontier_plans()
        return plans[0] if plans else None
