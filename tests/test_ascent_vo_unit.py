from __future__ import annotations

import ast
import hashlib
import inspect
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from ascent.vo import pose_provider as pose_provider_module
from ascent.vo.diagnostics import VODiagnosticsWriter
from ascent.vo.habitat_extensions import _VOFixedLookAction
from ascent.vo.pose_provider import (
    LOOK_DOWN,
    LOOK_UP,
    MOVE_FORWARD,
    STOP,
    TURN_LEFT,
    TURN_RIGHT,
    PoseUpdate,
    VOInferenceError,
    ZhaoRGBDPoseProvider,
    compose_zhao_delta,
    wrap_angle,
)
from ascent.vo.zhao_model import (
    DEPTH_INVALID_POLICY,
    DepthValidityStats,
    FORWARD_CHECKPOINT_SHA256,
    TURN_CHECKPOINT_SHA256,
    NormalizedDepthTopDown,
    ZhaoActionModels,
    ZhaoCheckpointError,
    ZhaoFramePreprocessor,
    ZhaoVisualOdometryCNN,
    _load_checkpoint,
)
from scripts.run_vo_gt_equivalence import (
    _sensor_local_transforms,
    _sensor_transforms,
    assert_fixed_look_transforms,
    assert_forward_sensor_transforms,
    build_sequence_plan,
    gt_local_zhao_delta,
    inverse_native_actions,
    select_runtime_episode,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = (
    REPO_ROOT.parent / "PointNav-VO" / "pretrained_ckpts" / "vo"
)


def _observation(value: int = 0, *, extra: dict | None = None) -> dict:
    output = {
        "vo_rgb": np.full((8, 10, 3), value, dtype=np.uint8),
        "vo_depth": np.full((8, 10, 1), value / 255.0, dtype=np.float32),
        "objectgoal": np.array([1], dtype=np.int64),
    }
    if extra is not None:
        output.update(extra)
    return output


class _FakePreprocessor:
    def prepare(self, rgb: np.ndarray, depth: np.ndarray):
        if not np.isfinite(depth).all():
            raise ValueError("non-finite fake depth")
        return (np.array(rgb, copy=True), np.array(depth, copy=True))


class _FakeModels:
    def __init__(self, deltas: dict[int, np.ndarray]) -> None:
        self.deltas = deltas
        self.calls: list[int] = []

    def estimate(self, action: int, previous, current) -> np.ndarray:
        self.calls.append(action)
        return np.asarray(self.deltas[action], dtype=np.float64)


def _fake_provider(monkeypatch: pytest.MonkeyPatch) -> tuple[
    ZhaoRGBDPoseProvider, _FakeModels
]:
    models = _FakeModels(
        {
            MOVE_FORWARD: np.array([0.0, -0.25, 0.0]),
            TURN_LEFT: np.array([0.0, 0.0, np.pi / 6.0]),
            TURN_RIGHT: np.array([0.0, 0.0, -np.pi / 6.0]),
        }
    )
    preprocessor = _FakePreprocessor()
    monkeypatch.setattr(
        pose_provider_module,
        "ZhaoActionModels",
        lambda checkpoint_dir, device: models,
    )
    monkeypatch.setattr(
        pose_provider_module,
        "ZhaoFramePreprocessor",
        lambda **kwargs: preprocessor,
    )
    provider = ZhaoRGBDPoseProvider(
        num_envs=1,
        device=torch.device("cpu"),
        checkpoint_dir=Path("/not/used"),
        source_min_depth=0.5,
        source_max_depth=5.0,
        source_hfov_degrees=79.0,
    )
    return provider, models


def test_se2_coordinate_convention_and_wrap() -> None:
    np.testing.assert_allclose(
        compose_zhao_delta(
            np.zeros(3), np.array([0.0, -0.25, 0.0])
        ),
        np.array([0.25, 0.0, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compose_zhao_delta(
            np.zeros(3), np.array([0.10, 0.0, 0.0])
        ),
        np.array([0.0, -0.10, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        compose_zhao_delta(
            np.array([0.0, 0.0, np.pi / 2.0]),
            np.array([0.0, -0.25, 0.0]),
        ),
        np.array([0.0, 0.25, np.pi / 2.0]),
        atol=1e-12,
    )
    assert wrap_angle(np.pi) == pytest.approx(-np.pi)
    assert wrap_angle(5.0 * np.pi / 2.0) == pytest.approx(np.pi / 2.0)


def test_gt_equivalence_helpers_cover_five_bounded_sequences() -> None:
    actions = (
        [TURN_LEFT] * 399
        + [MOVE_FORWARD] * 19
        + [TURN_LEFT] * 7
        + [TURN_RIGHT] * 8
    )
    plans = build_sequence_plan(actions)
    labels = {
        step.sequence
        for plan in plans
        for step in plan
        if step.measured
    }
    assert labels == {
        "ordinary_same_floor",
        "turn_heavy",
        "stair_ascent",
        "stair_descent",
        "floor_revisit",
    }
    assert max(map(len, plans)) < 1200
    assert inverse_native_actions([TURN_LEFT, TURN_RIGHT]) == [
        TURN_LEFT,
        TURN_RIGHT,
    ]
    inverse_forward = inverse_native_actions([MOVE_FORWARD])
    assert inverse_forward == (
        [TURN_LEFT] * 6
        + [MOVE_FORWARD]
        + [TURN_RIGHT] * 6
    )


def test_gt_equivalence_plan_supports_recorded_descent_with_looks() -> None:
    actions = [TURN_LEFT] * 219 + [
        MOVE_FORWARD,
        LOOK_DOWN,
        TURN_RIGHT,
        MOVE_FORWARD,
        LOOK_UP,
    ] + [TURN_LEFT] * 57
    plans = build_sequence_plan(
        actions,
        ordinary_end_step=60,
        stair_setup_end_step=281,
        stair_transition_start_step=220,
        stair_transition_end_step=281,
        recorded_transition_direction="descent",
    )
    measured = {
        label: [
            step.action
            for plan in plans
            for step in plan
            if step.measured and step.sequence == label
        ]
        for label in (
            "ordinary_same_floor",
            "turn_heavy",
            "stair_ascent",
            "stair_descent",
            "floor_revisit",
        )
    }
    transition_motion = [
        action
        for action in actions[219:281]
        if action not in (LOOK_UP, LOOK_DOWN)
    ]
    assert measured["stair_descent"] == transition_motion
    assert measured["stair_ascent"] == inverse_native_actions(
        transition_motion
    )
    assert LOOK_UP not in measured["floor_revisit"]
    assert LOOK_DOWN not in measured["floor_revisit"]
    assert max(map(len, plans)) < 1200


def test_gt_equivalence_selects_habitat_rewritten_runtime_identity() -> None:
    episodes = [
        types.SimpleNamespace(
            episode_id=str(index),
            scene_id=(
                "/dataset/hm3d/val/00877-4ok3usBNeis/"
                "4ok3usBNeis.basis.glb"
            ),
            object_category=(
                "toilet" if index in {3, 5, 6} else "bed"
            ),
        )
        for index in range(10)
    ]
    selected = select_runtime_episode(
        episodes,
        runtime_episode_id="6",
        expected_scene_id=(
            "hm3d/val/00877-4ok3usBNeis/"
            "4ok3usBNeis.basis.glb"
        ),
        expected_target_category="toilet",
    )
    assert selected.episode_id == "6"
    with pytest.raises(RuntimeError, match="runtime episode target"):
        select_runtime_episode(
            episodes,
            runtime_episode_id="6",
            expected_scene_id=(
                "hm3d/val/00877-4ok3usBNeis/"
                "4ok3usBNeis.basis.glb"
            ),
            expected_target_category="chair",
        )


def test_gt_equivalence_fixed_look_uses_camera_extrinsics() -> None:
    def transform(
        translation: tuple[float, float, float],
        rotation: tuple[float, float, float, float],
    ) -> dict[str, list[float]]:
        return {
            "translation": list(translation),
            "rotation": list(rotation),
        }

    identity = (0.0, 0.0, 0.0, 1.0)
    pitched = (-0.258819, 0.0, 0.0, 0.965926)
    previous = {
        key: transform((0.0, 1.25, 0.0), identity)
        for key in ("rgb", "depth", "vo_rgb", "vo_depth")
    }
    current = {
        "rgb": transform((0.0, 1.25, 0.0), pitched),
        "depth": transform((0.0, 1.25, 0.0), pitched),
        "vo_rgb": transform((0.0, 1.25, 0.0), identity),
        "vo_depth": transform((0.0, 1.25, 0.0), identity),
    }
    assert_fixed_look_transforms(previous, current)

    current["vo_rgb"] = transform((0.0, 1.25, 0.01), identity)
    with pytest.raises(
        RuntimeError,
        match="fixed vo_rgb transform changed during look action",
    ):
        assert_fixed_look_transforms(previous, current)


def test_gt_equivalence_forward_camera_gate_uses_extrinsics() -> None:
    coincident = {
        key: {
            "translation": [0.0, 1.25, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        }
        for key in ("rgb", "depth", "vo_rgb", "vo_depth")
    }
    assert_forward_sensor_transforms(coincident)

    coincident["vo_depth"] = {
        "translation": [0.0, 1.25, 0.01],
        "rotation": [0.0, 0.0, 0.0, 1.0],
    }
    with pytest.raises(
        RuntimeError,
        match="depth/vo_depth transforms diverged at zero pitch",
    ):
        assert_forward_sensor_transforms(coincident)


def test_gt_equivalence_reads_habitat_sensor_absolute_transforms() -> None:
    import magnum as mn

    class Node:
        translation = mn.Vector3(0.0, 1.25, 0.0)
        rotation = mn.Quaternion()

        @staticmethod
        def absolute_transformation():
            return mn.Matrix4.identity_init()

    wrapper = types.SimpleNamespace(node=Node())
    agent = types.SimpleNamespace(
        _sensors={
            key: wrapper
            for key in ("rgb", "depth", "vo_rgb", "vo_depth")
        }
    )
    env = types.SimpleNamespace(
        sim=types.SimpleNamespace(agents=[agent])
    )
    transforms = _sensor_transforms(env)
    local_transforms = _sensor_local_transforms(env)
    assert set(transforms) == {"rgb", "depth", "vo_rgb", "vo_depth"}
    for transform in transforms.values():
        assert transform["translation"] == [0.0, 0.0, 0.0]
        assert _quaternion_is_identity(transform["rotation"])
    for transform in local_transforms.values():
        assert transform["translation"] == [0.0, 1.25, 0.0]
        assert _quaternion_is_identity(transform["rotation"])


def _quaternion_is_identity(rotation: list[float]) -> bool:
    return bool(
        np.allclose(rotation, [0.0, 0.0, 0.0, 1.0])
        or np.allclose(rotation, [0.0, 0.0, 0.0, -1.0])
    )


def test_gt_local_delta_round_trips_through_production_composition() -> None:
    previous = np.array([1.2, -0.4, 0.7])
    current = np.array([1.6, 0.3, -2.8])
    delta = gt_local_zhao_delta(previous, current)
    reconstructed = compose_zhao_delta(previous, delta)
    np.testing.assert_allclose(
        reconstructed[:2], current[:2], rtol=0.0, atol=1e-12
    )
    assert wrap_angle(reconstructed[2] - current[2]) == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize(
    "pose,delta",
    [
        (np.zeros(2), np.zeros(3)),
        (np.zeros(3), np.zeros(2)),
        (np.array([0.0, np.nan, 0.0]), np.zeros(3)),
        (np.zeros(3), np.array([0.0, np.inf, 0.0])),
    ],
)
def test_se2_fails_closed(pose: np.ndarray, delta: np.ndarray) -> None:
    with pytest.raises(VOInferenceError):
        compose_zhao_delta(pose, delta)


def test_provider_uses_exactly_one_model_call_per_native_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, models = _fake_provider(monkeypatch)
    reset_obs = _observation(1)
    reset_update = provider.reset_batch([reset_obs])[0]
    assert reset_update.status == "episode_reset"
    assert reset_update.vo_inferences == 0
    provider.inject_estimated_pose([reset_obs])
    np.testing.assert_array_equal(
        reset_obs["estimated_pose"], np.zeros(3, dtype=np.float32)
    )
    assert "vo_rgb" not in reset_obs and "vo_depth" not in reset_obs

    forward_obs = _observation(2)
    forward = provider.update_batch(
        [forward_obs], [MOVE_FORWARD], [False]
    )[0]
    assert forward.vo_inferences == 1
    assert len(forward.local_deltas) == 1
    np.testing.assert_allclose(forward.pose_after, [0.25, 0.0, 0.0])

    left_obs = _observation(3)
    left = provider.update_batch([left_obs], [TURN_LEFT], [False])[0]
    assert left.vo_inferences == 1
    assert len(left.local_deltas) == 1
    assert left.pose_after[2] == pytest.approx(np.pi / 6.0)

    rotated_forward_obs = _observation(4)
    rotated_forward = provider.update_batch(
        [rotated_forward_obs], [MOVE_FORWARD], [False]
    )[0]
    np.testing.assert_allclose(
        rotated_forward.pose_after,
        [0.25 + 0.25 * np.cos(np.pi / 6.0), 0.125, np.pi / 6.0],
        atol=1e-12,
    )
    assert models.calls == [MOVE_FORWARD, TURN_LEFT, MOVE_FORWARD]


@pytest.mark.parametrize("action", [LOOK_UP, LOOK_DOWN, STOP])
def test_provider_identity_actions_never_call_zhao(
    monkeypatch: pytest.MonkeyPatch, action: int
) -> None:
    provider, models = _fake_provider(monkeypatch)
    provider.reset_batch([_observation(1)])
    update = provider.update_batch([_observation(2)], [action], [False])[0]
    assert update.status == "identity"
    assert update.vo_inferences == 0
    assert update.pose_after == (0.0, 0.0, 0.0)
    assert models.calls == []


def test_provider_autoreset_never_infers_across_episodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, models = _fake_provider(monkeypatch)
    provider.reset_batch([_observation(1)])
    provider.update_batch([_observation(2)], [MOVE_FORWARD], [False])
    reset_after_done = _observation(3)
    update = provider.update_batch(
        [reset_after_done], [MOVE_FORWARD], [True]
    )[0]
    assert update.status == "terminal_no_policy_successor_then_autoreset"
    assert update.vo_inferences == 0
    assert update.next_episode_reset
    assert models.calls == [MOVE_FORWARD]
    # The provider was already reset from the VectorEnv reset observation.
    provider.inject_estimated_pose([reset_after_done])
    np.testing.assert_array_equal(
        reset_after_done["estimated_pose"], np.zeros(3, dtype=np.float32)
    )


@pytest.mark.parametrize(
    "forbidden_key",
    ("gps", "heading", "base_explorer", "frontier_sensor"),
)
def test_provider_rejects_missing_nonfinite_and_gt_visible_inputs(
    monkeypatch: pytest.MonkeyPatch,
    forbidden_key: str,
) -> None:
    provider, _ = _fake_provider(monkeypatch)
    provider.reset_batch([_observation(1)])
    with pytest.raises(VOInferenceError, match="missing auxiliary"):
        provider.update_batch(
            [{"vo_rgb": np.zeros((8, 10, 3), dtype=np.uint8)}],
            [MOVE_FORWARD],
            [False],
        )

    bad = _observation(2)
    bad["vo_depth"][0, 0, 0] = np.nan
    with pytest.raises(VOInferenceError, match="failed closed"):
        provider.update_batch([bad], [MOVE_FORWARD], [False])

    visible_gt = _observation(3, extra={forbidden_key: np.array([123.0])})
    provider.reset_batch([visible_gt])
    with pytest.raises(VOInferenceError, match="forbidden keys"):
        provider.inject_estimated_pose([visible_gt])


def test_gt_poison_changes_only_diagnostics_not_policy_pose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy_observations = []
    diagnostic_paths = []
    for index, poisoned_x in enumerate((0.25, 12345.0)):
        provider, _ = _fake_provider(monkeypatch)
        provider.reset_batch([_observation(1)])
        observation = _observation(2)
        update = provider.update_batch(
            [observation], [MOVE_FORWARD], [False]
        )[0]
        provider.inject_estimated_pose([observation])
        policy_observations.append(
            {
                key: np.array(value, copy=True)
                for key, value in observation.items()
            }
        )
        diagnostics_path = tmp_path / f"poison_{index}.jsonl"
        writer = VODiagnosticsWriter(
            diagnostics_path, {"run": index}
        )
        writer.record_step(
            dataset="hm3d",
            scene_id="scene",
            episode_id="episode",
            seed=0,
            action_step=1,
            update=update,
            gt_pose={
                "x": poisoned_x,
                "y": -9876.0,
                "yaw": 2.75,
                "height": 42.0,
            },
        )
        writer.close()
        diagnostic_paths.append(diagnostics_path)

    assert policy_observations[0].keys() == policy_observations[1].keys()
    for key in policy_observations[0]:
        np.testing.assert_array_equal(
            policy_observations[0][key],
            policy_observations[1][key],
        )
    assert (
        diagnostic_paths[0].read_bytes()
        != diagnostic_paths[1].read_bytes()
    )


def test_provider_api_has_no_gt_or_info_input() -> None:
    names = set(inspect.signature(ZhaoRGBDPoseProvider.update_batch).parameters)
    assert names == {"self", "observations", "actions", "dones"}
    source = inspect.getsource(pose_provider_module)
    assert "get_agent_state" not in source
    assert "start_position" not in source
    assert "start_rotation" not in source


def test_preprocessor_calibrates_intrinsics_depth_and_modalities() -> None:
    preprocessor = ZhaoFramePreprocessor(
        device=torch.device("cpu"),
        source_min_depth=0.5,
        source_max_depth=5.0,
        source_hfov_degrees=79.0,
    )
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640, 1), 0.5, dtype=np.float32)
    prepared = preprocessor.prepare(rgb, depth)
    assert prepared.rgb.shape == (192, 341, 3)
    assert prepared.depth.shape == (192, 341, 1)
    assert prepared.discretized_depth.shape == (192, 341, 10)
    assert prepared.top_down_view.shape == (192, 341, 1)
    assert torch.isfinite(prepared.top_down_view).all()
    torch.testing.assert_close(
        prepared.discretized_depth.sum(dim=-1),
        torch.ones((192, 341)),
    )
    # Source normalized 0.5 -> 2.75 m -> checkpoint normalized depth.
    expected_depth = (2.75 - 0.1) / (10.0 - 0.1)
    assert float(prepared.depth.mean()) == pytest.approx(
        expected_depth, abs=1e-6
    )
    # The released checkpoint receives numeric 70, which remains positive.
    assert float(preprocessor.top_down._intrinsic[0, 0]) > 0.0
    assert prepared.depth_validity.as_dict() == {
        "policy": DEPTH_INVALID_POLICY,
        "total_pixels": 480 * 640,
        "finite_pixels": 480 * 640,
        "invalid_pixels": 0,
        "nan_pixels": 0,
        "positive_inf_pixels": 0,
        "negative_inf_pixels": 0,
        "invalid_fraction": 0.0,
        "all_invalid": False,
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def test_finite_depth_preprocessing_is_bit_identical_to_v9_golden() -> None:
    preprocessor = ZhaoFramePreprocessor(
        device=torch.device("cpu"),
        source_min_depth=0.5,
        source_max_depth=5.0,
        source_hfov_degrees=79.0,
    )
    height, width = 480, 640
    rgb = (
        np.arange(height * width * 3, dtype=np.uint32) % 256
    ).astype(np.uint8).reshape(height, width, 3)
    rng = np.random.default_rng(20260725)
    cases = {
        "all_zero": (
            np.zeros((height, width, 1), dtype=np.float32),
            (
                "e9fe2d66dc53cf34fce9bb6fb55a9b7c3b449c257a901c495336b71d2981f952",
                "b2689c6d15049bce1362471afd7414181c850b80c1ddfa57dedd11a5cdf279c4",
                "bc00407ef356fe81744f7b7099e45e61acdd1bd3641e27533932d61da5ebf514",
            ),
        ),
        "all_one": (
            np.ones((height, width, 1), dtype=np.float32),
            (
                "7fded57a7c48717c6e2322c29c7b9dfa5c7694360335b8d881e3472ab7e2949f",
                "a831da50d8fe5ab6e450893a468b3651d5700450e62da1d661d7f53366bb9581",
                "3aa8fe593a25b7a2d208b51c326d58271b84523b17bd178f5eee70655454a319",
            ),
        ),
        "constant_half": (
            np.full((height, width, 1), 0.5, dtype=np.float32),
            (
                "b4b8406734caf4fe349333b41be337dda294d94423b693bf16d14777d98c922d",
                "74d7830f2ea4612706c801511a5841206e7f32ec8baea7d2b44918eb83680ded",
                "6e19eefedbc850dc7b090bbf0e780664d39c2c1ff9b203c266e2220d44117b53",
            ),
        ),
        "uniform_random": (
            rng.random((height, width, 1), dtype=np.float32),
            (
                "e69d7af653e899f0bffe54faa7dada260ce144eace46003f50e82e986df54566",
                "b4181d373c52d8e2b26a9662d40c38ddfcf5a6c91c9820139aff862932a4de7c",
                "08f8f9021b96eb5018fca89307801e2a085583027b8f0de0b1a3de7240df20c4",
            ),
        ),
        "boundary_pattern": (
            np.linspace(
                0.0,
                1.0,
                height * width,
                dtype=np.float32,
            ).reshape(height, width, 1),
            (
                "149a135868d49a87b892802214a9eae1b1053302a46f8ba844f0a9742d262cf7",
                "631524aee8189488613888179322c28d93d0551dcfc840590e94bbb7406798fd",
                "f1195226e63d79bbfb679ed1a50b2e1076796d750cf19a95a3ddd58454f527ae",
            ),
        ),
    }
    for name, (depth, expected) in cases.items():
        prepared = preprocessor.prepare(rgb, depth)
        assert (
            _tensor_sha256(prepared.rgb)
            == "586adb62a57140a49e696101e546f23512a21406a23c927ce4ebddc924955e2c"
        ), name
        assert (
            _tensor_sha256(prepared.depth),
            _tensor_sha256(prepared.discretized_depth),
            _tensor_sha256(prepared.top_down_view),
        ) == expected, name
        assert prepared.depth_validity.invalid_pixels == 0


def test_preprocessor_maps_sparse_nonfinite_depth_to_checkpoint_zero() -> None:
    preprocessor = ZhaoFramePreprocessor(
        device=torch.device("cpu"),
        source_min_depth=0.5,
        source_max_depth=5.0,
        source_hfov_degrees=70.0,
    )
    rgb = np.zeros((192, 341, 3), dtype=np.uint8)
    depth = np.full((192, 341, 1), 0.5, dtype=np.float32)
    locations = ((10, 20), (80, 170), (170, 320))
    depth[locations[0]] = np.nan
    depth[locations[1]] = np.inf
    depth[locations[2]] = -np.inf
    prepared = preprocessor.prepare(rgb, depth)

    assert torch.isfinite(prepared.depth).all()
    assert torch.isfinite(prepared.discretized_depth).all()
    assert torch.isfinite(prepared.top_down_view).all()
    for row, column in locations:
        assert float(prepared.depth[row, column, 0]) == 0.0
        assert float(prepared.discretized_depth[row, column, 0]) == 1.0
    assert prepared.depth_validity.as_dict() == {
        "policy": DEPTH_INVALID_POLICY,
        "total_pixels": 192 * 341,
        "finite_pixels": 192 * 341 - 3,
        "invalid_pixels": 3,
        "nan_pixels": 1,
        "positive_inf_pixels": 1,
        "negative_inf_pixels": 1,
        "invalid_fraction": 3 / (192 * 341),
        "all_invalid": False,
    }


def test_preprocessor_all_invalid_depth_uses_released_zero_contract() -> None:
    preprocessor = ZhaoFramePreprocessor(
        device=torch.device("cpu"),
        source_min_depth=0.5,
        source_max_depth=5.0,
        source_hfov_degrees=70.0,
    )
    rgb = np.zeros((192, 341, 3), dtype=np.uint8)
    depth = np.full((192, 341, 1), np.nan, dtype=np.float32)
    prepared = preprocessor.prepare(rgb, depth)
    torch.testing.assert_close(
        prepared.depth, torch.zeros_like(prepared.depth)
    )
    torch.testing.assert_close(
        prepared.discretized_depth[..., 0],
        torch.ones_like(prepared.discretized_depth[..., 0]),
    )
    torch.testing.assert_close(
        prepared.discretized_depth[..., 1:],
        torch.zeros_like(prepared.discretized_depth[..., 1:]),
    )
    torch.testing.assert_close(
        prepared.top_down_view, torch.zeros_like(prepared.top_down_view)
    )
    assert prepared.depth_validity.all_invalid
    assert prepared.depth_validity.invalid_fraction == 1.0


def test_preprocessor_fails_closed_on_bad_camera_or_finite_depth() -> None:
    with pytest.raises(ValueError, match="no wider"):
        ZhaoFramePreprocessor(
            device=torch.device("cpu"),
            source_min_depth=0.5,
            source_max_depth=5.0,
            source_hfov_degrees=60.0,
        )
    preprocessor = ZhaoFramePreprocessor(
        device=torch.device("cpu"),
        source_min_depth=0.5,
        source_max_depth=5.0,
        source_hfov_degrees=79.0,
    )
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    depth = np.zeros((12, 16, 1), dtype=np.float32)
    depth.fill(1.1)
    with pytest.raises(ValueError, match=r"outside \[0,1\]"):
        preprocessor.prepare(rgb, depth)


def test_released_top_down_zero_depth_is_finite() -> None:
    generator = NormalizedDepthTopDown(
        min_depth=0.1,
        max_depth=10.0,
        height=192,
        width=341,
        hfov_numeric=70.0,
    )
    output = generator.generate(torch.zeros((192, 341, 1)))
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_top_down_projection_matches_released_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointnav_root = REPO_ROOT.parent / "PointNav-VO"
    for name, path in (
        ("pointnav_vo", pointnav_root / "pointnav_vo"),
        ("pointnav_vo.utils", pointnav_root / "pointnav_vo" / "utils"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
    released_module = importlib.import_module(
        "pointnav_vo.utils.geometry_utils"
    )
    NormalizedDepth2TopDownViewHabitatTorch = (
        released_module.NormalizedDepth2TopDownViewHabitatTorch
    )

    released = NormalizedDepth2TopDownViewHabitatTorch(
        min_depth=0.1,
        max_depth=10.0,
        vis_size_h=192,
        vis_size_w=341,
        hfov_rad=70.0,
    )
    ported = NormalizedDepthTopDown(
        min_depth=0.1,
        max_depth=10.0,
        height=192,
        width=341,
        hfov_numeric=70.0,
    )
    depth = torch.rand(
        (192, 341, 1), generator=torch.Generator().manual_seed(19)
    )
    depth[:7] = 0
    depth[-5:] = 0
    depth[:, :11] = 0
    depth[:, -13:] = 0
    expected = released.gen_top_down_view(depth.clone())
    actual = ported.generate(depth.clone())
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_fixed_look_rotates_main_sensors_but_not_vo_sensors() -> None:
    import magnum as mn

    class Node:
        def __init__(self) -> None:
            self.rotation = mn.Quaternion()

    class SensorWrapper:
        def __init__(self) -> None:
            self.node = Node()

    class Agent:
        def __init__(self) -> None:
            self._sensors = {
                key: SensorWrapper()
                for key in ("rgb", "depth", "vo_rgb", "vo_depth")
            }

    class Sim:
        def __init__(self) -> None:
            self.agents = [Agent()]

    action = object.__new__(_VOFixedLookAction)
    action._sim = Sim()
    action._tilt_angle = 30
    action.look_sign = 1.0
    before = {
        key: wrapper.node.rotation
        for key, wrapper in action._sim.agents[0]._sensors.items()
    }
    action.step(task=object())
    after = {
        key: wrapper.node.rotation
        for key, wrapper in action._sim.agents[0]._sensors.items()
    }
    assert after["vo_rgb"] == before["vo_rgb"]
    assert after["vo_depth"] == before["vo_depth"]
    assert after["rgb"] != before["rgb"]
    assert after["depth"] != before["depth"]


def test_config_keeps_internal_heading_behind_policy_boundary() -> None:
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    from habitat.config.default import patch_config
    from habitat.config.default_structured_configs import register_hydra_plugin
    from habitat_baselines.config.default_structured_configs import (
        HabitatBaselinesConfigPlugin,
    )

    import ascent.run  # noqa: F401 - registers ASCENT's search path
    from ascent.vo.habitat_extensions import configure_gt_isolated_vo

    GlobalHydra.instance().clear()
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str((REPO_ROOT / "experiments").resolve()),
    ):
        config = patch_config(
            hydra.compose(config_name="eval_ascent_hm3d.yaml")
        )
    from habitat.config import read_write

    with read_write(config):
        config.habitat.gym.obs_keys = [
            "rgb",
            "depth",
            "gps",
            "compass",
            "heading",
        ]
    agent = config.habitat.simulator.agents.main_agent
    original_rgb = dict(agent.sim_sensors.rgb_sensor)
    original_depth = dict(agent.sim_sensors.depth_sensor)
    configure_gt_isolated_vo(config)
    assert float(config.habitat.simulator.turn_angle) == 30.0
    assert config.habitat.task.actions.stop.type == "StopAction"
    assert config.habitat.task.actions.move_forward.type == "MoveForwardAction"
    assert config.habitat.task.actions.turn_left.type == "TurnLeftAction"
    assert config.habitat.task.actions.turn_right.type == "TurnRightAction"
    assert config.habitat.task.actions.look_up.type == "VOFixedLookUpAction"
    assert config.habitat.task.actions.look_down.type == "VOFixedLookDownAction"
    assert not {"gps_sensor", "compass_sensor"}.intersection(
        config.habitat.task.lab_sensors
    )
    assert {
        "heading_sensor",
        "base_explorer",
        "frontier_sensor",
    }.issubset(config.habitat.task.lab_sensors)
    assert "gt_start_aligned_pose" in config.habitat.task.measurements
    assert config.habitat.gym.obs_keys == [
        "rgb",
        "depth",
        "objectgoal",
        "vo_rgb",
        "vo_depth",
    ]

    for original, auxiliary, expected_type in (
        (
            original_rgb,
            dict(agent.sim_sensors.vo_rgb_sensor),
            "VOHabitatSimRGBSensor",
        ),
        (
            original_depth,
            dict(agent.sim_sensors.vo_depth_sensor),
            "VOHabitatSimDepthSensor",
        ),
    ):
        original.pop("type")
        assert auxiliary.pop("type") == expected_type
        assert auxiliary == original
    GlobalHydra.instance().clear()


def _pose_update(
    status: str = "ok", *, pose_after: tuple[float, float, float] = (1, 2, 0.3)
) -> PoseUpdate:
    validity = DepthValidityStats(
        total_pixels=100,
        finite_pixels=99,
        invalid_pixels=1,
        nan_pixels=1,
        positive_inf_pixels=0,
        negative_inf_pixels=0,
    )
    return PoseUpdate(
        env=0,
        action=MOVE_FORWARD,
        status=status,
        pose_before=(0.0, 0.0, 0.0),
        pose_after=pose_after,
        local_deltas=((0.0, -0.25, 0.0),) if status == "ok" else (),
        vo_inferences=1 if status == "ok" else 0,
        next_episode_reset="autoreset" in status,
        previous_depth_validity=validity,
        current_depth_validity=validity,
        current_frame_role=(
            "next_episode_reset"
            if "autoreset" in status
            else "action_successor"
        ),
    )


def test_diagnostics_are_append_only_and_gt_is_evaluation_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "vo.jsonl"
    writer = VODiagnosticsWriter(path, {"provider": "test"})
    writer.record_step(
        dataset="hm3d",
        scene_id="scene",
        episode_id="ep",
        seed=100,
        action_step=1,
        update=_pose_update(),
        gt_pose={"x": 1.1, "y": 2.2, "yaw": 0.4, "height": 0.0},
    )
    writer.record_error(
        dataset="hm3d",
        scene_id="scene",
        episode_id="ep",
        seed=100,
        action_step=2,
        action=TURN_LEFT,
        error=VOInferenceError("synthetic"),
    )
    writer.record_episode_end(
        dataset="hm3d",
        scene_id="scene",
        episode_id="ep",
        seed=100,
        action_steps=2,
        native_metrics={
            "success": np.float32(1.0),
            "spl": 0.5,
            "non_scalar": [1, 2, 3],
        },
    )
    writer.close()
    records = [
        json.loads(line) for line in path.read_text().splitlines()
    ]
    assert [record["record_type"] for record in records] == [
        "run_metadata",
        "vo_step",
        "vo_technical_error",
        "episode_end",
    ]
    assert records[1]["translation_error"] > 0
    assert records[1]["finite"] is True
    assert records[1]["depth_invalid_policy"] == DEPTH_INVALID_POLICY
    assert records[1]["current_depth_validity"]["invalid_pixels"] == 1
    assert records[1]["current_depth_validity"]["invalid_fraction"] == 0.01
    assert records[1]["current_frame_role"] == "action_successor"
    assert records[2]["finite"] is False
    assert records[3]["native_metrics"] == {
        "spl": 0.5,
        "success": 1.0,
    }
    assert records[3]["evaluation_pose_source"] == "habitat_ground_truth"
    assert records[3]["policy_pose_source"] == "zhao_rgbd_2021"
    with pytest.raises(FileExistsError):
        VODiagnosticsWriter(path, {"provider": "must_not_overwrite"})


def test_terminal_no_successor_has_no_misleading_localization_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "terminal.jsonl"
    writer = VODiagnosticsWriter(path, {"provider": "test"})
    writer.record_step(
        dataset="hm3d",
        scene_id="scene",
        episode_id="ep",
        seed=100,
        action_step=500,
        update=_pose_update(
            "terminal_no_policy_successor_then_autoreset",
            pose_after=(3.0, 4.0, 0.2),
        ),
        gt_pose={"x": 9.0, "y": 9.0, "yaw": 1.0, "height": 0.0},
    )
    writer.close()
    record = json.loads(path.read_text().splitlines()[1])
    assert record["localization_error_available"] is False
    assert "translation_error" not in record
    assert "absolute_yaw_error" not in record


def test_official_checkpoints_load_strictly_and_infer_finite() -> None:
    if not CHECKPOINT_DIR.is_dir():
        pytest.fail(f"missing official checkpoint directory {CHECKPOINT_DIR}")
    models = ZhaoActionModels(CHECKPOINT_DIR, torch.device("cpu"))
    preprocessor = ZhaoFramePreprocessor(
        device=torch.device("cpu"),
        source_min_depth=0.5,
        source_max_depth=5.0,
        source_hfov_degrees=79.0,
    )
    rng = np.random.default_rng(11)
    previous = preprocessor.prepare(
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        rng.random((480, 640, 1), dtype=np.float32),
    )
    current = preprocessor.prepare(
        rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        rng.random((480, 640, 1), dtype=np.float32),
    )
    for action in (MOVE_FORWARD, TURN_LEFT, TURN_RIGHT):
        delta = models.estimate(action, previous, current)
        assert delta.shape == (3,)
        assert np.isfinite(delta).all()
    assert set(models.models) == {MOVE_FORWARD, TURN_LEFT, TURN_RIGHT}


def test_checkpoint_serialized_training_contract_is_30_degrees() -> None:
    forward = _load_checkpoint(
        CHECKPOINT_DIR / "act_forward.pth",
        FORWARD_CHECKPOINT_SHA256,
    )
    turn = _load_checkpoint(
        CHECKPOINT_DIR / "act_left_right_inv_joint.pth",
        TURN_CHECKPOINT_SHA256,
    )
    for checkpoint in (forward, turn):
        config = checkpoint["config"]
        assert float(config.TASK_CONFIG.SIMULATOR.TURN_ANGLE) == 30.0
        assert int(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH) == 341
        assert int(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT) == 192
        assert float(config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV) == 70.0
        assert float(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH) == 0.1
        assert float(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH) == 10.0
        assert bool(
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        )
        assert (
            str(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NOISE_MODEL)
            == "RedwoodDepthNoiseModel"
        )
        assert not hasattr(config.VO.TRAIN, "loss_weight_fixed")
        assert {
            key: float(value)
            for key, value in config.VO.TRAIN.loss_weight_multiplier.items()
        } == {"dx": 0.0, "dz": 0.0, "dyaw": 0.0}


def test_inference_port_is_numerically_identical_to_released_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pointnav_root = REPO_ROOT.parent / "PointNav-VO" / "pointnav_vo"
    if not pointnav_root.is_dir():
        pytest.fail(f"missing official PointNav-VO source {pointnav_root}")
    for name, path in (
        ("pointnav_vo", pointnav_root),
        ("pointnav_vo.vo", pointnav_root / "vo"),
        ("pointnav_vo.vo.models", pointnav_root / "vo" / "models"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
    released_module = importlib.import_module(
        "pointnav_vo.vo.models.vo_cnn"
    )
    released_model = (
        released_module.VisualOdometryCNNDiscretizedDepthTopDownView(
            observation_space=[
                "rgb",
                "depth",
                "discretized_depth",
                "top_down_view",
            ],
            observation_size=(341, 192),
            normalize_visual_inputs=True,
            output_dim=3,
            discretized_depth_channels=10,
        )
    )
    ported_model = ZhaoVisualOdometryCNN()
    state = _load_checkpoint(
        CHECKPOINT_DIR / "act_forward.pth",
        FORWARD_CHECKPOINT_SHA256,
    )["model_states"][MOVE_FORWARD]
    released_model.load_state_dict(state, strict=True)
    ported_model.load_state_dict(state, strict=True)
    released_model.eval()
    ported_model.eval()
    generator = torch.Generator().manual_seed(7)
    inputs = {
        "rgb": torch.randint(
            0, 256, (1, 192, 341, 6), generator=generator
        ).float(),
        "depth": torch.rand(
            (1, 192, 341, 2), generator=generator
        ),
        "discretized_depth": torch.randint(
            0, 2, (1, 192, 341, 20), generator=generator
        ).float(),
        "top_down_view": torch.rand(
            (1, 192, 341, 2), generator=generator
        ),
    }
    with torch.inference_mode():
        released = released_model(inputs)
        ported = ported_model(inputs)
    torch.testing.assert_close(ported, released, rtol=0.0, atol=0.0)


def test_checkpoint_missing_and_hash_mismatch_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ZhaoCheckpointError, match="missing"):
        _load_checkpoint(
            CHECKPOINT_DIR / "does_not_exist.pth",
            FORWARD_CHECKPOINT_SHA256,
        )
    monkeypatch.setattr(
        "ascent.vo.zhao_model._sha256", lambda path: "0" * 64
    )
    with pytest.raises(ZhaoCheckpointError, match="SHA mismatch"):
        _load_checkpoint(
            CHECKPOINT_DIR / "act_left_right_inv_joint.pth",
            TURN_CHECKPOINT_SHA256,
        )


def test_policy_pose_source_has_no_direct_gt_observation_reads() -> None:
    source = (REPO_ROOT / "ascent" / "ascent_policy.py").read_text()
    assert 'observations["estimated_pose"]' in source
    for key in ("gps", "compass", "heading"):
        assert f'observations["{key}"]' not in source
        assert f"observations['{key}']" not in source
    assert "gt_start_aligned_pose" not in source

    trainer_source = (
        REPO_ROOT / "ascent" / "ascent_trainer.py"
    ).read_text()
    provider_update = trainer_source.index(
        "self._vo_pose_provider.update_batch"
    )
    gt_pop = trainer_source.index(
        'infos[i].pop("gt_start_aligned_pose", None)'
    )
    policy_extra = trainer_source.index(
        "self._agent.actor_critic.get_extra", gt_pop
    )
    next_batch = trainer_source.index("batch = batch_obs", gt_pop)
    assert provider_update < gt_pop < policy_extra < next_batch

    tree = ast.parse(trainer_source)
    policy_act_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "act"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "actor_critic"
    ]
    assert len(policy_act_calls) == 1
    assert all(
        keyword.arg != "current_episodes_info"
        for keyword in policy_act_calls[0].keywords
    )


def test_submap_policy_diagnostics_are_not_habitat_episode_scalars() -> None:
    from ascent.ascent_trainer import extract_scalars_from_info

    scalars = extract_scalars_from_info(
        {
            "success": 1.0,
            "spl": 0.5,
            "submap_overlap_before_update": None,
            "submap_split_reason": None,
            "submap_count": 3,
            "visual_payload": [],
        }
    )
    assert scalars == {"success": 1.0, "spl": 0.5}
