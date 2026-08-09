from __future__ import annotations

import math

import numpy as np

from scripts.vpr_shadow_geometry import (
    aggregate_multiframe_support,
    lift_matched_keypoints,
    submap_transform_from_pair,
    transform_disagreement,
    verify_rigid_correspondences,
)


def _transform(yaw: float, translation: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = [
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ]
    value[:3, 3] = translation
    return value


def test_rigid_ransac_recovers_transform_with_outliers() -> None:
    rng = np.random.default_rng(7)
    source = rng.uniform([-2.0, -1.0, 1.0], [2.0, 1.0, 4.0], size=(100, 3))
    expected = _transform(0.25, (0.8, -0.3, 0.1))
    target = (expected[:3, :3] @ source.T).T + expected[:3, 3]
    target += rng.normal(0.0, 0.01, size=target.shape)
    target[:25] = rng.uniform(-4.0, 4.0, size=(25, 3))

    result = verify_rigid_correspondences(
        source, target, match_count=120, seed_key="fixture"
    )
    assert result.pair_supported
    assert result.inlier_count >= 70
    assert result.rmse_m is not None and result.rmse_m < 0.03
    assert result.transform_target_from_source is not None
    translation, rotation = transform_disagreement(
        expected, result.transform_target_from_source
    )
    assert translation < 0.02
    assert rotation < math.radians(1.0)


def test_depth_lifting_uses_ascent_forward_left_up_camera_basis() -> None:
    keypoints = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 0.0]])
    matches = np.column_stack((np.arange(3), np.arange(3)))
    depth = np.full((3, 3), 0.5, dtype=np.float32)
    calibration = {"min_depth": 0.0, "max_depth": 2.0, "fx": 1.0, "fy": 1.0}
    source, target = lift_matched_keypoints(
        keypoints_source=keypoints,
        keypoints_target=keypoints,
        matches=matches,
        depth_source=depth,
        depth_target=depth,
        source_calibration=calibration,
        target_calibration=calibration,
    )
    expected = np.array([[1.0, 0.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0]])
    np.testing.assert_allclose(source, expected)
    np.testing.assert_allclose(target, expected)


def test_degenerate_correspondences_fail_closed() -> None:
    line = np.column_stack((np.arange(40), np.zeros(40), np.ones(40)))
    result = verify_rigid_correspondences(
        line, line.copy(), match_count=40, seed_key="line"
    )
    assert not result.pair_supported
    assert result.transform_target_from_source is None


def test_pair_transform_is_normalized_to_submap_frames() -> None:
    target_submap_from_target_camera = _transform(0.4, (1.0, 2.0, 0.8))
    target_camera_from_source_camera = _transform(-0.2, (0.3, 0.0, 0.0))
    source_submap_from_source_camera = _transform(0.1, (-1.0, 0.5, 0.8))
    result = submap_transform_from_pair(
        target_submap_from_target_camera=target_submap_from_target_camera,
        target_camera_from_source_camera=target_camera_from_source_camera,
        source_submap_from_source_camera=source_submap_from_source_camera,
    )
    expected = (
        target_submap_from_target_camera
        @ target_camera_from_source_camera
        @ np.linalg.inv(source_submap_from_source_camera)
    )
    np.testing.assert_allclose(result, expected)


def test_multiframe_support_requires_two_consistent_query_frames() -> None:
    base = _transform(0.2, (0.5, -0.1, 0.0))
    close = base @ _transform(math.radians(3.0), (0.05, 0.0, 0.0))
    far = base @ _transform(math.radians(40.0), (1.0, 0.0, 0.0))
    records = [
        {
            "query_frame_id": "q0",
            "pair_supported": True,
            "inlier_count": 50,
            "submap_transform_target_from_source": base.tolist(),
        },
        {
            "query_frame_id": "q1",
            "pair_supported": True,
            "inlier_count": 45,
            "submap_transform_target_from_source": close.tolist(),
        },
        {
            "query_frame_id": "q2",
            "pair_supported": True,
            "inlier_count": 60,
            "submap_transform_target_from_source": far.tolist(),
        },
    ]
    aggregate = aggregate_multiframe_support(records)
    assert aggregate["geometry_pass"] is True
    assert aggregate["distinct_query_frame_support"] == 2

    duplicate_query = [dict(records[0]), dict(records[1], query_frame_id="q0")]
    assert aggregate_multiframe_support(duplicate_query)["geometry_pass"] is False
