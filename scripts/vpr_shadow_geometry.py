"""Deterministic RGB-D geometric verification for v1.4 VPR shadow matches."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


RANSAC_INLIER_THRESHOLD_M = 0.15
RANSAC_ITERATIONS = 512
PAIR_MIN_INLIERS = 30
PAIR_MIN_INLIER_RATIO = 0.30
PAIR_MAX_RMSE_M = 0.10
SUBMAP_MIN_QUERY_SUPPORT = 2
SUBMAP_MAX_TRANSLATION_DISAGREEMENT_M = 0.35
SUBMAP_MAX_ROTATION_DISAGREEMENT_RAD = math.radians(15.0)


@dataclass(frozen=True)
class RigidVerification:
    match_count: int
    depth_match_count: int
    inlier_count: int
    inlier_ratio: float
    rmse_m: float | None
    transform_target_from_source: np.ndarray | None

    @property
    def pair_supported(self) -> bool:
        return (
            self.transform_target_from_source is not None
            and self.inlier_count >= PAIR_MIN_INLIERS
            and self.inlier_ratio >= PAIR_MIN_INLIER_RATIO
            and self.rmse_m is not None
            and self.rmse_m <= PAIR_MAX_RMSE_M
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "match_count": int(self.match_count),
            "depth_match_count": int(self.depth_match_count),
            "inlier_count": int(self.inlier_count),
            "inlier_ratio": float(self.inlier_ratio),
            "rmse_m": None if self.rmse_m is None else float(self.rmse_m),
            "pair_supported": bool(self.pair_supported),
            "transform_target_from_source": (
                None
                if self.transform_target_from_source is None
                else self.transform_target_from_source.tolist()
            ),
        }


def decode_normalized_depth(path: Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.ndim != 2 or encoded.dtype != np.uint16:
        raise ValueError(f"expected uint16 depth PNG: {path}")
    return encoded.astype(np.float32) / 65535.0


def lift_matched_keypoints(
    *,
    keypoints_source: np.ndarray,
    keypoints_target: np.ndarray,
    matches: np.ndarray,
    depth_source: np.ndarray,
    depth_target: np.ndarray,
    source_calibration: Mapping[str, float],
    target_calibration: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    matches = np.asarray(matches, dtype=np.int64)
    if matches.ndim != 2 or matches.shape[1] != 2:
        raise ValueError("matches must have shape Nx2")
    source_pixels = np.asarray(keypoints_source, dtype=np.float64)[matches[:, 0]]
    target_pixels = np.asarray(keypoints_target, dtype=np.float64)[matches[:, 1]]
    source_points, source_valid = _lift(
        source_pixels, depth_source, source_calibration
    )
    target_points, target_valid = _lift(
        target_pixels, depth_target, target_calibration
    )
    valid = source_valid & target_valid
    return source_points[valid], target_points[valid]


def _lift(
    pixels: np.ndarray,
    normalized_depth: np.ndarray,
    calibration: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(normalized_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must be two-dimensional")
    fx = float(calibration["fx"])
    fy = float(calibration["fy"])
    min_depth = float(calibration["min_depth"])
    max_depth = float(calibration["max_depth"])
    if not (fx > 0.0 and fy > 0.0 and 0.0 <= min_depth < max_depth):
        raise ValueError("invalid RGB-D calibration")
    height, width = depth.shape
    # Preserve the exact VLFM projection center used by get_point_cloud().
    cx = float(width // 2)
    cy = float(height // 2)
    values = np.zeros(len(pixels), dtype=np.float64)
    valid = np.zeros(len(pixels), dtype=bool)
    for index, (u, v) in enumerate(pixels):
        x = int(round(float(u)))
        y = int(round(float(v)))
        if not (0 <= x < width and 0 <= y < height):
            continue
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        patch = depth[y0:y1, x0:x1]
        candidates = patch[
            np.isfinite(patch) & (patch > 0.0) & (patch < 1.0)
        ]
        if not len(candidates):
            continue
        values[index] = float(np.median(candidates))
        valid[index] = True
    metric = values * (max_depth - min_depth) + min_depth
    # Match ASCENT/VLFM's camera basis used by tf_camera_to_submap:
    # +X forward, +Y left, +Z up.  LightGlue keypoints are image (u, v).
    points = np.column_stack(
        (
            metric,
            -(pixels[:, 0] - cx) * metric / fx,
            -(pixels[:, 1] - cy) * metric / fy,
        )
    )
    valid &= np.isfinite(points).all(axis=1)
    return points, valid


def verify_rigid_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    *,
    match_count: int,
    seed_key: str,
    inlier_threshold_m: float = RANSAC_INLIER_THRESHOLD_M,
    iterations: int = RANSAC_ITERATIONS,
) -> RigidVerification:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source/target correspondences must both be Nx3")
    count = len(source)
    if count < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        return RigidVerification(match_count, count, 0, 0.0, None, None)
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_rmse = math.inf
    for _ in range(int(iterations)):
        indices = rng.choice(count, size=3, replace=False)
        transform = _kabsch(source[indices], target[indices])
        if transform is None:
            continue
        errors = _errors(transform, source, target)
        inliers = errors <= float(inlier_threshold_m)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 3:
            continue
        rmse = float(np.sqrt(np.mean(errors[inliers] ** 2)))
        if (
            best_inliers is None
            or inlier_count > int(np.count_nonzero(best_inliers))
            or (
                inlier_count == int(np.count_nonzero(best_inliers))
                and rmse < best_rmse
            )
        ):
            best_inliers = inliers
            best_rmse = rmse
    if best_inliers is None:
        return RigidVerification(match_count, count, 0, 0.0, None, None)
    transform = _kabsch(source[best_inliers], target[best_inliers])
    if transform is None:
        return RigidVerification(match_count, count, 0, 0.0, None, None)
    errors = _errors(transform, source, target)
    inliers = errors <= float(inlier_threshold_m)
    if int(np.count_nonzero(inliers)) >= 3:
        refined = _kabsch(source[inliers], target[inliers])
        if refined is not None:
            transform = refined
            errors = _errors(transform, source, target)
            inliers = errors <= float(inlier_threshold_m)
    inlier_count = int(np.count_nonzero(inliers))
    rmse = (
        float(np.sqrt(np.mean(errors[inliers] ** 2)))
        if inlier_count
        else None
    )
    return RigidVerification(
        match_count=int(match_count),
        depth_match_count=count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_count / count),
        rmse_m=rmse,
        transform_target_from_source=transform,
    )


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)
    if (
        np.linalg.matrix_rank(source_centered, tol=1e-6) < 2
        or np.linalg.matrix_rank(target_centered, tol=1e-6) < 2
    ):
        return None
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = np.mean(target, axis=0) - rotation @ np.mean(source, axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _errors(
    transform: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    predicted = (transform[:3, :3] @ source.T).T + transform[:3, 3]
    return np.linalg.norm(predicted - target, axis=1)


def submap_transform_from_pair(
    *,
    target_submap_from_target_camera: np.ndarray,
    target_camera_from_source_camera: np.ndarray,
    source_submap_from_source_camera: np.ndarray,
) -> np.ndarray:
    target_from_target_camera = np.asarray(
        target_submap_from_target_camera, dtype=np.float64
    )
    target_camera_from_source = np.asarray(
        target_camera_from_source_camera, dtype=np.float64
    )
    source_from_source_camera = np.asarray(
        source_submap_from_source_camera, dtype=np.float64
    )
    for transform in (
        target_from_target_camera,
        target_camera_from_source,
        source_from_source_camera,
    ):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("submap consistency transforms must be finite 4x4")
    return (
        target_from_target_camera
        @ target_camera_from_source
        @ np.linalg.inv(source_from_source_camera)
    )


def aggregate_multiframe_support(
    pair_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    supported = [
        (index, record)
        for index, record in enumerate(pair_records)
        if record["pair_supported"]
    ]
    best_support = 0
    best_indices: list[int] = []
    for _, anchor in supported:
        anchor_transform = np.asarray(
            anchor["submap_transform_target_from_source"], dtype=np.float64
        )
        compatible: dict[str, tuple[int, Mapping[str, Any]]] = {}
        for original_index, candidate in supported:
            transform = np.asarray(
                candidate["submap_transform_target_from_source"],
                dtype=np.float64,
            )
            translation_error, rotation_error = transform_disagreement(
                anchor_transform, transform
            )
            if (
                translation_error
                > SUBMAP_MAX_TRANSLATION_DISAGREEMENT_M
                or rotation_error
                > SUBMAP_MAX_ROTATION_DISAGREEMENT_RAD
            ):
                continue
            query_id = str(candidate["query_frame_id"])
            incumbent = compatible.get(query_id)
            if incumbent is None or int(candidate["inlier_count"]) > int(
                incumbent[1]["inlier_count"]
            ):
                compatible[query_id] = (original_index, candidate)
        if len(compatible) > best_support:
            best_support = len(compatible)
            best_indices = sorted(index for index, _ in compatible.values())
    return {
        "supported_pair_count": len(supported),
        "distinct_query_frame_support": int(best_support),
        "supporting_pair_indices": best_indices,
        "geometry_pass": best_support >= SUBMAP_MIN_QUERY_SUPPORT,
    }


def transform_disagreement(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    relative = np.linalg.inv(first) @ second
    translation = float(np.linalg.norm(relative[:3, 3]))
    cosine = float(np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    rotation = float(math.acos(cosine))
    return translation, rotation


def geometry_contract() -> dict[str, Any]:
    return {
        "ransac_inlier_threshold_m": RANSAC_INLIER_THRESHOLD_M,
        "ransac_iterations": RANSAC_ITERATIONS,
        "pair_min_inliers": PAIR_MIN_INLIERS,
        "pair_min_inlier_ratio": PAIR_MIN_INLIER_RATIO,
        "pair_max_rmse_m": PAIR_MAX_RMSE_M,
        "submap_min_distinct_query_frame_support": SUBMAP_MIN_QUERY_SUPPORT,
        "submap_max_translation_disagreement_m": (
            SUBMAP_MAX_TRANSLATION_DISAGREEMENT_M
        ),
        "submap_max_rotation_disagreement_rad": (
            SUBMAP_MAX_ROTATION_DISAGREEMENT_RAD
        ),
    }
