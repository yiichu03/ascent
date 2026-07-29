"""Transfer only stair/gateway topology across local map frames."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ascent.submaps.geometry import transform_points_xy


_MASK_ATTRIBUTES = (
    "_up_stair_map",
    "_down_stair_map",
    "_disabled_stair_map",
    "stair_boundary",
    "stair_boundary_goal",
)
_XY_ATTRIBUTES = (
    "_up_stair_frontiers",
    "_down_stair_frontiers",
    "_potential_stair_centroid",
)
_PIXEL_ATTRIBUTES = (
    "_up_stair_start",
    "_up_stair_end",
    "_down_stair_start",
    "_down_stair_end",
    "_potential_stair_centroid_px",
)
_SCALAR_ATTRIBUTES = (
    "_has_up_stair",
    "_has_down_stair",
    "_explored_up_stair",
    "_explored_down_stair",
    "_look_for_downstair_flag",
)


def transfer_stair_topology(
    *,
    source_obstacle_map: object,
    destination_obstacle_map: object,
    source_anchor_world: np.ndarray,
    destination_anchor_world: np.ndarray,
) -> None:
    """Copy gateway evidence without copying obstacle/free/explored cells."""

    for attribute in _MASK_ATTRIBUTES:
        source = np.asarray(getattr(source_obstacle_map, attribute))
        destination = np.asarray(getattr(destination_obstacle_map, attribute))
        destination.fill(0)
        yx = np.argwhere(source)
        if len(yx) == 0:
            continue
        source_xy = source_obstacle_map._px_to_xy(yx[:, ::-1])
        destination_xy = transform_points_xy(
            source_xy, source_anchor_world, destination_anchor_world
        )
        pixels = destination_obstacle_map._xy_to_px(destination_xy)
        valid = _valid_pixels(pixels, destination.shape)
        pixels = pixels[valid]
        destination[pixels[:, 1], pixels[:, 0]] = True

    for attribute in _XY_ATTRIBUTES:
        points = _as_xy(getattr(source_obstacle_map, attribute))
        transformed = (
            np.empty((0, 2), dtype=np.float64)
            if len(points) == 0
            else transform_points_xy(
                points, source_anchor_world, destination_anchor_world
            )
        )
        setattr(destination_obstacle_map, attribute, transformed)
        pixel_attribute = f"{attribute}_px"
        if hasattr(destination_obstacle_map, pixel_attribute):
            pixels = (
                np.empty((0, 2), dtype=np.int64)
                if len(transformed) == 0
                else destination_obstacle_map._xy_to_px(transformed)
            )
            setattr(destination_obstacle_map, pixel_attribute, pixels)

    for attribute in _PIXEL_ATTRIBUTES:
        value = np.asarray(getattr(source_obstacle_map, attribute))
        original_shape = value.shape
        if value.size == 0:
            setattr(destination_obstacle_map, attribute, np.array([]))
            continue
        pixels = value.reshape(-1, 2)
        source_xy = source_obstacle_map._px_to_xy(pixels)
        destination_xy = transform_points_xy(
            source_xy, source_anchor_world, destination_anchor_world
        )
        destination_pixels = destination_obstacle_map._xy_to_px(
            destination_xy
        )
        setattr(
            destination_obstacle_map,
            attribute,
            destination_pixels.reshape(original_shape),
        )

    for attribute in _SCALAR_ATTRIBUTES:
        setattr(
            destination_obstacle_map,
            attribute,
            bool(getattr(source_obstacle_map, attribute)),
        )

    disabled = _as_xy(
        list(getattr(source_obstacle_map, "_disabled_frontiers", set()))
    )
    if len(disabled) == 0:
        destination_obstacle_map._disabled_frontiers = set()
    else:
        transformed = transform_points_xy(
            disabled, source_anchor_world, destination_anchor_world
        )
        destination_obstacle_map._disabled_frontiers = {
            tuple(point) for point in transformed
        }


def _as_xy(points: Iterable[object]) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return result.reshape(-1, 2)


def _valid_pixels(pixels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
