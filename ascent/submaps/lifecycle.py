"""Rule-based lifecycle for immutable local ASCENT submaps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ascent.submaps.frontier_registry import FrontierRegistry
from ascent.submaps.geometry import as_pose, pose_to_matrix, relative_pose, wrap_angle
from ascent.submaps.query_view import SubmapQueryView
from ascent.submaps.types import (
    GatewayEdge,
    MapPayload,
    SubmapBundle,
    SubmapGraph,
)


@dataclass(frozen=True)
class SubmapLifecycleConfig:
    enabled: bool = False
    min_action_endpoints: int = 20
    min_path_length_m: float = 1.5
    overlap_threshold: float = 0.25
    low_overlap_consecutive: int = 3
    max_motion_budget_m: float = 8.0
    rotation_weight_m_per_rad: float = 0.10
    gateway_frontier_resolution_radius_m: float = 1.0
    gateway_reached_radius_m: float = 0.9
    provisional_thresholds: bool = True

    def __post_init__(self) -> None:
        if self.min_action_endpoints < 1:
            raise ValueError("min_action_endpoints must be positive")
        if self.min_path_length_m < 0.0:
            raise ValueError("min_path_length_m cannot be negative")
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be in [0, 1]")
        if self.low_overlap_consecutive < 1:
            raise ValueError("low_overlap_consecutive must be positive")
        if self.max_motion_budget_m <= 0.0:
            raise ValueError("max_motion_budget_m must be positive")
        if self.rotation_weight_m_per_rad < 0.0:
            raise ValueError("rotation_weight_m_per_rad cannot be negative")
        if self.gateway_frontier_resolution_radius_m < 0.0:
            raise ValueError("gateway frontier radius cannot be negative")
        if self.gateway_reached_radius_m <= 0.0:
            raise ValueError("gateway reached radius must be positive")


@dataclass(frozen=True)
class SplitDecision:
    should_split: bool
    reason: Optional[str]
    mature: bool
    motion_budget_m: float
    low_overlap_streak: int
    floor_changed: bool


@dataclass
class _EnvironmentMemory:
    graph: SubmapGraph = field(default_factory=SubmapGraph)
    registry: FrontierRegistry = field(default_factory=FrontierRegistry)
    active_submap_id: Optional[str] = None
    next_submap_index: int = 0
    next_edge_index: int = 0
    pending_decision: Optional[SplitDecision] = None
    events: List[Dict[str, object]] = field(default_factory=list)


class SubmapManager:
    """Owns submap identities, local frames, lifecycle and topology."""

    def __init__(self, num_envs: int, config: SubmapLifecycleConfig):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.config = config
        self._environments = [_EnvironmentMemory() for _ in range(num_envs)]

    def reset(self, env: int) -> None:
        self._environments[env] = _EnvironmentMemory()

    def _new_submap_id(self, env: int) -> str:
        memory = self._environments[env]
        result = f"env{env}:sm{memory.next_submap_index:04d}"
        memory.next_submap_index += 1
        return result

    def _new_edge_id(self, env: int) -> str:
        memory = self._environments[env]
        result = f"env{env}:gw{memory.next_edge_index:04d}"
        memory.next_edge_index += 1
        return result

    def start(
        self,
        env: int,
        world_pose: Sequence[float],
        floor_id: int,
        payload: MapPayload,
        step: int,
    ) -> SubmapBundle:
        memory = self._environments[env]
        if memory.active_submap_id is not None:
            raise RuntimeError(f"Environment {env} already has an active submap")
        pose = as_pose(world_pose)
        bundle = SubmapBundle(
            submap_id=self._new_submap_id(env),
            floor_id=int(floor_id),
            anchor_pose_world=pose,
            creation_step=int(step),
            payload=payload,
            last_world_pose=pose,
        )
        memory.graph.add_node(bundle)
        memory.active_submap_id = bundle.submap_id
        memory.events.append(
            {
                "event": "submap_created",
                "step": int(step),
                "submap_id": bundle.submap_id,
                "floor_id": int(floor_id),
                "reason": "episode_start",
            }
        )
        return bundle

    def active_bundle(self, env: int) -> SubmapBundle:
        memory = self._environments[env]
        if memory.active_submap_id is None:
            raise RuntimeError(f"Environment {env} has no active submap")
        return memory.graph.get_node(memory.active_submap_id)

    def has_active(self, env: int) -> bool:
        return self._environments[env].active_submap_id is not None

    def graph(self, env: int) -> SubmapGraph:
        return self._environments[env].graph

    def registry(self, env: int) -> FrontierRegistry:
        return self._environments[env].registry

    def events(self, env: int) -> Tuple[Dict[str, object], ...]:
        return tuple(self._environments[env].events)

    def query_view(self, env: int) -> SubmapQueryView:
        memory = self._environments[env]
        if memory.active_submap_id is None:
            raise RuntimeError(f"Environment {env} has no active submap")
        return SubmapQueryView(
            memory.graph, memory.registry, memory.active_submap_id
        )

    def local_pose(self, env: int, world_pose: Sequence[float]) -> np.ndarray:
        return relative_pose(self.active_bundle(env).anchor_pose_world, world_pose)

    def observe_action_endpoint(
        self,
        env: int,
        world_pose: Sequence[float],
        floor_id: int,
        overlap: Optional[float],
        allow_nonfloor_split: bool = True,
    ) -> SplitDecision:
        bundle = self.active_bundle(env)
        bundle.require_active()
        pose = as_pose(world_pose)
        previous = as_pose(bundle.last_world_pose)
        bundle.path_length_m += float(np.linalg.norm(pose[:2] - previous[:2]))
        bundle.accumulated_rotation_rad += abs(
            wrap_angle(float(pose[2] - previous[2]))
        )
        bundle.last_world_pose = pose
        bundle.accepted_action_endpoints += 1

        if overlap is not None:
            overlap_value = float(overlap)
            if not 0.0 <= overlap_value <= 1.0:
                raise ValueError("overlap must be in [0, 1]")
            bundle.latest_overlap = overlap_value
            if overlap_value < self.config.overlap_threshold:
                bundle.low_overlap_streak += 1
            else:
                bundle.low_overlap_streak = 0

        floor_changed = int(floor_id) != bundle.floor_id
        mature = (
            bundle.accepted_action_endpoints >= self.config.min_action_endpoints
            and bundle.path_length_m >= self.config.min_path_length_m
        )
        motion_budget = (
            bundle.path_length_m
            + self.config.rotation_weight_m_per_rad
            * bundle.accumulated_rotation_rad
        )

        reason = None
        if self.config.enabled:
            if floor_changed:
                reason = "floor_change"
            elif allow_nonfloor_split and mature and (
                bundle.low_overlap_streak >= self.config.low_overlap_consecutive
            ):
                reason = "low_overlap"
            elif (
                allow_nonfloor_split
                and mature
                and motion_budget >= self.config.max_motion_budget_m
            ):
                reason = "motion_budget"

        decision = SplitDecision(
            should_split=reason is not None,
            reason=reason,
            mature=mature,
            motion_budget_m=motion_budget,
            low_overlap_streak=bundle.low_overlap_streak,
            floor_changed=floor_changed,
        )
        self._environments[env].pending_decision = decision
        return decision

    def commit_split(
        self,
        env: int,
        world_pose: Sequence[float],
        floor_id: int,
        new_payload: MapPayload,
        step: int,
        frontiers_local: np.ndarray,
        frontier_scores: Optional[Sequence[float]] = None,
        boundary_frames: Sequence[Dict[str, object]] = (),
    ) -> Tuple[SubmapBundle, GatewayEdge, List[str]]:
        memory = self._environments[env]
        decision = memory.pending_decision
        if decision is None or not decision.should_split:
            raise RuntimeError("commit_split requires a pending positive split decision")
        old_bundle = self.active_bundle(env)
        old_bundle.require_active()
        pose_world = as_pose(world_pose)
        source_local_pose = relative_pose(old_bundle.anchor_pose_world, pose_world)

        old_bundle.freeze(step, frontiers_local)
        records = memory.registry.register_submap_frontiers(
            old_bundle.submap_id,
            old_bundle.frozen_frontiers_local,
            creation_step=int(step),
            scores=frontier_scores,
        )

        new_bundle = SubmapBundle(
            submap_id=self._new_submap_id(env),
            floor_id=int(floor_id),
            anchor_pose_world=pose_world,
            creation_step=int(step),
            payload=new_payload,
            last_world_pose=pose_world,
            parent_submap_id=old_bundle.submap_id,
        )
        memory.graph.add_node(new_bundle)

        relative_transform = (
            np.linalg.inv(pose_to_matrix(old_bundle.anchor_pose_world))
            @ pose_to_matrix(new_bundle.anchor_pose_world)
        )
        edge = GatewayEdge(
            edge_id=self._new_edge_id(env),
            source_submap_id=old_bundle.submap_id,
            destination_submap_id=new_bundle.submap_id,
            source_local_pose=source_local_pose,
            destination_local_pose=np.zeros(3, dtype=np.float64),
            creation_step=int(step),
            source_floor_id=old_bundle.floor_id,
            destination_floor_id=new_bundle.floor_id,
            relative_transform=relative_transform,
            confidence=1.0,
            kind="floor_transition" if decision.floor_changed else "sequential",
            boundary_frames=tuple(dict(frame) for frame in boundary_frames[-4:]),
        )
        memory.graph.add_edge(edge)
        memory.active_submap_id = new_bundle.submap_id
        resolved = memory.registry.resolve_near_gateway(
            old_bundle.submap_id,
            source_local_pose[:2],
            new_bundle.submap_id,
            self.config.gateway_frontier_resolution_radius_m,
            int(step),
        )
        memory.pending_decision = None
        memory.events.append(
            {
                "event": "submap_split",
                "step": int(step),
                "source_submap_id": old_bundle.submap_id,
                "destination_submap_id": new_bundle.submap_id,
                "edge_id": edge.edge_id,
                "reason": decision.reason,
                "registered_frontiers": len(records),
                "resolved_gateway_frontiers": len(resolved),
            }
        )
        return new_bundle, edge, resolved

    def commit_revisit(
        self,
        env: int,
        world_pose: Sequence[float],
        floor_id: int,
        new_payload: MapPayload,
        step: int,
        gateway_edge_id: str,
        frontiers_local: np.ndarray,
        frontier_scores: Optional[Sequence[float]] = None,
        boundary_frames: Sequence[Dict[str, object]] = (),
    ) -> Tuple[SubmapBundle, GatewayEdge, List[str]]:
        """Freeze the current writer and create a fresh revisit writer.

        The historical destination remains frozen. The new active bundle is
        associated with it through a read-only reference/revisit edge.
        """

        memory = self._environments[env]
        old_bundle = self.active_bundle(env)
        old_bundle.require_active()
        existing_edge = memory.graph.edges[gateway_edge_id]
        if (
            existing_edge.source_submap_id != old_bundle.submap_id
            and existing_edge.destination_submap_id != old_bundle.submap_id
        ):
            raise ValueError(
                f"gateway {gateway_edge_id} is not incident to active submap"
            )
        reference_submap_id = existing_edge.other(old_bundle.submap_id)
        reference_bundle = memory.graph.get_node(reference_submap_id)
        pose_world = as_pose(world_pose)
        source_local_pose = relative_pose(old_bundle.anchor_pose_world, pose_world)

        old_bundle.freeze(step, frontiers_local)
        memory.registry.register_submap_frontiers(
            old_bundle.submap_id,
            old_bundle.frozen_frontiers_local,
            creation_step=int(step),
            scores=frontier_scores,
        )
        new_bundle = SubmapBundle(
            submap_id=self._new_submap_id(env),
            floor_id=int(floor_id),
            anchor_pose_world=pose_world,
            creation_step=int(step),
            payload=new_payload,
            last_world_pose=pose_world,
            parent_submap_id=old_bundle.submap_id,
            reference_submap_id=reference_submap_id,
        )
        memory.graph.add_node(new_bundle)

        reference_endpoint = existing_edge.endpoint_for(reference_submap_id)
        relative_transform = (
            np.linalg.inv(pose_to_matrix(reference_bundle.anchor_pose_world))
            @ pose_to_matrix(new_bundle.anchor_pose_world)
        )
        revisit_edge = GatewayEdge(
            edge_id=self._new_edge_id(env),
            source_submap_id=reference_submap_id,
            destination_submap_id=new_bundle.submap_id,
            source_local_pose=reference_endpoint,
            destination_local_pose=np.zeros(3, dtype=np.float64),
            creation_step=int(step),
            source_floor_id=reference_bundle.floor_id,
            destination_floor_id=int(floor_id),
            relative_transform=relative_transform,
            confidence=float(existing_edge.confidence),
            kind="revisit",
            boundary_frames=tuple(dict(frame) for frame in boundary_frames[-4:]),
        )
        memory.graph.add_edge(revisit_edge)
        existing_edge.traversal_count += 1
        existing_edge.last_outcome = "gateway_reached"
        memory.active_submap_id = new_bundle.submap_id
        resolved = memory.registry.resolve_near_gateway(
            old_bundle.submap_id,
            source_local_pose[:2],
            new_bundle.submap_id,
            self.config.gateway_frontier_resolution_radius_m,
            int(step),
        )
        memory.pending_decision = None
        memory.events.append(
            {
                "event": "submap_revisit",
                "step": int(step),
                "source_submap_id": old_bundle.submap_id,
                "destination_submap_id": new_bundle.submap_id,
                "reference_submap_id": reference_submap_id,
                "traversed_edge_id": gateway_edge_id,
                "revisit_edge_id": revisit_edge.edge_id,
                "resolved_gateway_frontiers": len(resolved),
            }
        )
        return new_bundle, revisit_edge, resolved
