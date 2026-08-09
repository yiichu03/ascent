"""Submap-local task memory for ASCENT with estimated visual odometry poses."""

from ascent.submaps.frontier_registry import FrontierRegistry
from ascent.submaps.diagnostics import SubmapDiagnosticsWriter
from ascent.submaps.geometry import (
    matrix_to_pose,
    pose_to_matrix,
    relative_pose,
    transform_points_xy,
    wrap_angle,
)
from ascent.submaps.lifecycle import (
    SplitDecision,
    SubmapLifecycleConfig,
    SubmapManager,
)
from ascent.submaps.overlap import ViewOverlapConfig, estimate_view_overlap
from ascent.submaps.topology import transfer_stair_topology
from ascent.submaps.continuity import (
    BoundaryHandoff,
    DepthGeometryFrame,
    ExhaustionRecovery,
    replay_depth_geometry,
    select_connected_handoff_waypoint,
    transform_camera_to_destination,
    waypoint_in_robot_component,
)
from ascent.submaps.query_view import SubmapQueryView
from ascent.submaps.routes import RemoteRoute, RouteWaypoint
from ascent.submaps.vpr_shadow import (
    VPRShadowKeyframeConfig,
    VPRShadowKeyframeWriter,
)
from ascent.submaps.types import (
    FrontierRecord,
    FrontierStatus,
    GatewayEdge,
    MapPayload,
    SubmapBundle,
    SubmapGraph,
    SubmapState,
)

__all__ = [
    "BoundaryHandoff",
    "DepthGeometryFrame",
    "ExhaustionRecovery",
    "FrontierRecord",
    "FrontierRegistry",
    "FrontierStatus",
    "GatewayEdge",
    "MapPayload",
    "RemoteRoute",
    "RouteWaypoint",
    "SplitDecision",
    "SubmapDiagnosticsWriter",
    "SubmapBundle",
    "SubmapGraph",
    "SubmapLifecycleConfig",
    "SubmapManager",
    "SubmapQueryView",
    "SubmapState",
    "ViewOverlapConfig",
    "VPRShadowKeyframeConfig",
    "VPRShadowKeyframeWriter",
    "estimate_view_overlap",
    "matrix_to_pose",
    "pose_to_matrix",
    "relative_pose",
    "replay_depth_geometry",
    "select_connected_handoff_waypoint",
    "transform_camera_to_destination",
    "transform_points_xy",
    "transfer_stair_topology",
    "waypoint_in_robot_component",
    "wrap_angle",
]
