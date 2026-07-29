"""Read-only RGB-D view overlap against one active local obstacle map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from vlfm.utils.geometry_utils import get_point_cloud, transform_points


@dataclass(frozen=True)
class ViewOverlapConfig:
    """Sampling controls for the action-endpoint overlap diagnostic."""

    sample_stride: int = 16
    ray_samples: int = 4
    explored_dilation_cells: int = 3
    min_valid_samples: int = 32
    min_explored_cells: int = 400

    def __post_init__(self) -> None:
        if self.sample_stride < 1:
            raise ValueError("sample_stride must be positive")
        if self.ray_samples < 1:
            raise ValueError("ray_samples must be positive")
        if self.explored_dilation_cells < 0:
            raise ValueError("explored_dilation_cells cannot be negative")
        if self.min_valid_samples < 1:
            raise ValueError("min_valid_samples must be positive")
        if self.min_explored_cells < 0:
            raise ValueError("min_explored_cells cannot be negative")


def estimate_view_overlap(
    *,
    normalized_depth: np.ndarray,
    tf_camera_to_local: np.ndarray,
    min_depth: float,
    max_depth: float,
    fx: float,
    fy: float,
    obstacle_map: object,
    config: ViewOverlapConfig,
) -> Optional[float]:
    """Estimate how much of the current RGB-D frustum was already explored.

    The function is deliberately read-only. It samples several points along
    valid depth rays, projects them into the active submap, and reports the
    fraction falling in a slightly dilated copy of ``explored_area``. ``None``
    means that the active map or current frame is too immature for a decision.
    """

    depth = np.asarray(normalized_depth, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError("normalized_depth must be a finite HxW array")
    transform = np.asarray(tf_camera_to_local, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("tf_camera_to_local must be a finite 4x4 matrix")
    if max_depth <= min_depth:
        raise ValueError("max_depth must be larger than min_depth")

    explored = np.asarray(getattr(obstacle_map, "explored_area"), dtype=bool)
    if explored.ndim != 2:
        raise ValueError("obstacle_map.explored_area must be 2-D")
    if int(np.count_nonzero(explored)) < config.min_explored_cells:
        return None

    sample_mask = np.zeros(depth.shape, dtype=bool)
    sample_mask[:: config.sample_stride, :: config.sample_stride] = True
    sample_mask &= (depth > 0.0) & (depth < 1.0)
    if int(np.count_nonzero(sample_mask)) < config.min_valid_samples:
        return None

    scaled_depth = depth * float(max_depth - min_depth) + float(min_depth)
    endpoints_camera = get_point_cloud(scaled_depth, sample_mask, fx, fy)
    if len(endpoints_camera) < config.min_valid_samples:
        return None

    # Full-ray sampling is less brittle than terminal depth points, which are
    # often located exactly on a frontier.
    fractions = np.linspace(
        1.0 / config.ray_samples, 1.0, config.ray_samples, dtype=np.float64
    )
    ray_points_camera = (
        endpoints_camera[:, None, :] * fractions[None, :, None]
    ).reshape(-1, 3)
    ray_points_local = transform_points(transform, ray_points_camera)
    xy = ray_points_local[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < config.min_valid_samples:
        return None

    pixels = np.asarray(obstacle_map._xy_to_px(xy), dtype=np.int64)
    height, width = explored.shape
    in_bounds = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[in_bounds]
    if len(pixels) < config.min_valid_samples:
        return None

    if config.explored_dilation_cells > 0:
        radius = int(config.explored_dilation_cells)
        kernel_size = 2 * radius + 1
        known = cv2.dilate(
            explored.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
    else:
        known = explored

    covered = known[pixels[:, 1], pixels[:, 0]]
    return float(np.mean(covered))
