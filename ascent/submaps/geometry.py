"""Small, dependency-free SE(2) helpers used by submap memory."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def as_pose(pose: Iterable[float]) -> np.ndarray:
    """Validate and copy an ``(x, y, yaw)`` pose."""

    result = np.asarray(pose, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite SE(2) pose with shape (3,), got {result}")
    result = result.copy()
    result[2] = wrap_angle(result[2])
    return result


def pose_to_matrix(pose: Iterable[float]) -> np.ndarray:
    """Convert ``(x, y, yaw)`` to a homogeneous 3x3 transform."""

    x, y, yaw = as_pose(pose)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array(
        [
            [cosine, -sine, x],
            [sine, cosine, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def matrix_to_pose(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous 3x3 transform to ``(x, y, yaw)``."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(
            f"Expected a finite homogeneous transform with shape (3, 3), got {matrix}"
        )
    return np.array(
        [
            matrix[0, 2],
            matrix[1, 2],
            wrap_angle(np.arctan2(matrix[1, 0], matrix[0, 0])),
        ],
        dtype=np.float64,
    )


def relative_pose(anchor_pose: Iterable[float], world_pose: Iterable[float]) -> np.ndarray:
    """Express ``world_pose`` in the local frame defined by ``anchor_pose``."""

    transform = np.linalg.inv(pose_to_matrix(anchor_pose)) @ pose_to_matrix(world_pose)
    return matrix_to_pose(transform)


def transform_pose(
    pose: Iterable[float],
    source_anchor_world: Iterable[float],
    destination_anchor_world: Iterable[float],
) -> np.ndarray:
    """Transform a pose from one anchored submap frame to another."""

    world_from_source = pose_to_matrix(source_anchor_world)
    world_from_destination = pose_to_matrix(destination_anchor_world)
    source_from_pose = pose_to_matrix(pose)
    destination_from_pose = (
        np.linalg.inv(world_from_destination) @ world_from_source @ source_from_pose
    )
    return matrix_to_pose(destination_from_pose)


def transform_points_xy(
    points: np.ndarray,
    source_anchor_world: Iterable[float],
    destination_anchor_world: Iterable[float],
) -> np.ndarray:
    """Transform one or more XY points between anchored submap frames."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (2,):
            raise ValueError(f"Expected an XY point, got shape {array.shape}")
        array = array.reshape(1, 2)
        squeeze = True
    elif array.ndim == 2 and array.shape[1] == 2:
        squeeze = False
    else:
        raise ValueError(f"Expected points with shape (2,) or (N, 2), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Cannot transform non-finite XY points")

    destination_from_source = (
        np.linalg.inv(pose_to_matrix(destination_anchor_world))
        @ pose_to_matrix(source_anchor_world)
    )
    homogeneous = np.concatenate(
        [array, np.ones((array.shape[0], 1), dtype=np.float64)], axis=1
    )
    transformed = (destination_from_source @ homogeneous.T).T[:, :2]
    return transformed[0] if squeeze else transformed
