#!/usr/bin/env python3
"""Bounded Habitat gate for ASCENT-VO's sensor/action/pose boundary.

This is an engineering control, not a VO-quality experiment.  It runs the
same native Habitat actions once with the original action/sensor boundary and
once with the fixed auxiliary RGB-D camera.  A GT-derived local delta is then
composed through the production SE(2) function to verify conventions without
making GT available to the navigation policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import habitat
import hydra
import magnum as mn
import numpy as np
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    register_hydra_plugin,
)
from habitat.tasks.nav.nav import HeadingSensor
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from hydra.core.global_hydra import GlobalHydra

import ascent.run  # noqa: F401 - register ASCENT config search paths
from ascent.vo.habitat_extensions import (
    POLICY_VISIBLE_RAW_OBSERVATION_KEYS,
    configure_gt_isolated_vo,
)
from ascent.vo.pose_provider import (
    LOOK_DOWN,
    LOOK_UP,
    MOVE_FORWARD,
    TURN_LEFT,
    TURN_RIGHT,
    VO_DEPTH_KEY,
    VO_RGB_KEY,
    compose_zhao_delta,
    wrap_angle,
)


SCHEMA = "ascent_vo_gt_equivalence_v1"
_HYDRA_PLUGIN_REGISTERED = False
ACTION_NAMES = {
    MOVE_FORWARD: "move_forward",
    TURN_LEFT: "turn_left",
    TURN_RIGHT: "turn_right",
    LOOK_UP: "look_up",
    LOOK_DOWN: "look_down",
}
FORBIDDEN_POLICY_KEYS = {
    "gps",
    "compass",
    "heading",
    "base_explorer",
    "frontier_sensor",
    "gt_start_aligned_pose",
}
IGNORED_COMPARISON_KEYS = FORBIDDEN_POLICY_KEYS | {
    VO_RGB_KEY,
    VO_DEPTH_KEY,
}


@dataclass(frozen=True)
class PlannedStep:
    sequence: str
    action: int
    measured: bool = True


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open(
            "x", encoding="utf-8", buffering=1
        )

    def write(self, record: Mapping[str, Any]) -> None:
        self._stream.write(
            json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        )

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inverse_native_actions(actions: Sequence[int]) -> list[int]:
    """Express the exact primitive inverse without adding a back action."""

    output: list[int] = []
    for action in reversed(actions):
        if action == TURN_LEFT:
            output.append(TURN_RIGHT)
        elif action == TURN_RIGHT:
            output.append(TURN_LEFT)
        elif action == MOVE_FORWARD:
            output.extend(
                [TURN_LEFT] * 6
                + [MOVE_FORWARD]
                + [TURN_RIGHT] * 6
            )
        else:
            raise ValueError(
                f"cannot invert non-motion action {action}"
            )
    return output


def build_sequence_plan(
    recorded_actions: Sequence[int],
    *,
    ordinary_end_step: int = 60,
    stair_setup_end_step: int = 399,
    stair_transition_start_step: int = 400,
    stair_transition_end_step: int = 433,
    recorded_transition_direction: str = "ascent",
) -> list[list[PlannedStep]]:
    if not (
        1 <= ordinary_end_step <= len(recorded_actions)
        and 1 <= stair_setup_end_step <= len(recorded_actions)
        and 1 <= stair_transition_start_step
        <= stair_transition_end_step
        <= len(recorded_actions)
    ):
        raise ValueError(
            "invalid 1-indexed action windows for trace of "
            f"{len(recorded_actions)} steps"
        )
    if recorded_transition_direction not in {"ascent", "descent"}:
        raise ValueError(
            "recorded_transition_direction must be ascent or descent"
        )
    setup_actions = list(recorded_actions[:stair_setup_end_step])
    unsupported = sorted(set(setup_actions) - set((1, 2, 3, 4, 5)))
    if unsupported:
        raise ValueError(
            f"stair setup contains unsupported actions {unsupported}"
        )
    if setup_actions.count(LOOK_UP) != setup_actions.count(LOOK_DOWN):
        raise ValueError(
            "stair setup must finish at zero camera pitch"
        )
    recorded_transition = list(
        recorded_actions[
            stair_transition_start_step - 1 : stair_transition_end_step
        ]
    )
    unsupported = sorted(
        set(recorded_transition) - set((1, 2, 3, 4, 5))
    )
    if unsupported:
        raise ValueError(
            "recorded stair transition contains unsupported actions "
            f"{unsupported}"
        )
    transition_motion = [
        action
        for action in recorded_transition
        if action not in (LOOK_UP, LOOK_DOWN)
    ]
    if MOVE_FORWARD not in transition_motion:
        raise ValueError(
            "recorded stair transition has no forward motion"
        )
    ordinary_actions = list(recorded_actions[:ordinary_end_step])
    unsupported = sorted(
        set(ordinary_actions) - set((1, 2, 3, 4, 5))
    )
    if unsupported:
        raise ValueError(
            f"ordinary sequence contains unsupported actions {unsupported}"
        )
    if ordinary_actions.count(LOOK_UP) != ordinary_actions.count(LOOK_DOWN):
        raise ValueError(
            "ordinary sequence must finish at zero camera pitch"
        )
    ordinary = [
        PlannedStep("ordinary_same_floor", action)
        for action in ordinary_actions
    ]
    turn_heavy = [
        *[
            PlannedStep("turn_heavy", TURN_LEFT)
            for _ in range(12)
        ],
        PlannedStep("turn_heavy", LOOK_UP),
        PlannedStep("turn_heavy", LOOK_DOWN),
    ]
    setup = [
        PlannedStep("stair_setup", action, measured=False)
        for action in setup_actions
    ]
    if recorded_transition_direction == "ascent":
        ascent_actions = transition_motion
        descent_actions = inverse_native_actions(transition_motion)
    else:
        ascent_actions = inverse_native_actions(transition_motion)
        descent_actions = transition_motion
    stair_cycle = [
        *setup,
        *[
            PlannedStep("stair_ascent", action)
            for action in ascent_actions
        ],
        *[
            PlannedStep("stair_descent", action)
            for action in descent_actions
        ],
        *[
            PlannedStep("floor_revisit", action)
            for action in ascent_actions + descent_actions
        ],
    ]
    return [ordinary, turn_heavy, stair_cycle]


def _rotation_coefficients(rotation: Any) -> list[float]:
    return [
        *[float(value) for value in rotation.imag],
        float(rotation.real),
    ]


def _sensor_rotation_coefficients(rotation: Any) -> list[float]:
    return [
        float(rotation.vector.x),
        float(rotation.vector.y),
        float(rotation.vector.z),
        float(rotation.scalar),
    ]


def _equivalent_quaternion(
    first: Sequence[float],
    second: Sequence[float],
    *,
    atol: float = 1e-6,
) -> bool:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return bool(
        np.allclose(a, b, rtol=0.0, atol=atol)
        or np.allclose(a, -b, rtol=0.0, atol=atol)
    )


def _start_aligned_pose(
    *,
    origin: np.ndarray,
    start_rotation: Any,
    state: Any,
) -> np.ndarray:
    relative = quaternion_rotate_vector(
        start_rotation.inverse(),
        np.asarray(state.position, dtype=np.float64) - origin,
    )
    yaw = HeadingSensor._quat_to_xy_heading(
        None, state.rotation.inverse() * start_rotation
    )[0]
    pose = np.array(
        [-relative[2], -relative[0], yaw, relative[1]],
        dtype=np.float64,
    )
    if not np.isfinite(pose).all():
        raise RuntimeError(f"non-finite GT reference pose {pose}")
    return pose


def gt_local_zhao_delta(
    previous_pose: Sequence[float],
    current_pose: Sequence[float],
) -> np.ndarray:
    previous = np.asarray(previous_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    if previous.shape != (3,) or current.shape != (3,):
        raise ValueError(
            f"bad GT pose shapes {previous.shape} and {current.shape}"
        )
    delta_xy = current[:2] - previous[:2]
    yaw = float(previous[2])
    world_to_local = np.array(
        [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]]
    )
    local_forward, local_left = world_to_local @ delta_xy
    return np.array(
        [
            -local_left,
            -local_forward,
            wrap_angle(float(current[2] - previous[2])),
        ],
        dtype=np.float64,
    )


def _value_digest(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        encoded = json.dumps(
            {
                str(key): _value_digest(item)
                for key, item in sorted(value.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return {
            "kind": "mapping",
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    array = np.asarray(value)
    if array.dtype == object:
        encoded = repr(value).encode()
        return {
            "kind": "repr",
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(str(tuple(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return {
        "kind": "array",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": digest.hexdigest(),
    }


def _observation_digests(
    observations: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(key): _value_digest(value)
        for key, value in sorted(observations.items())
        if key not in IGNORED_COMPARISON_KEYS
    }


def _assert_vo_policy_boundary(config, observations: Mapping[str, Any]) -> None:
    configured = tuple(config.habitat.gym.obs_keys or ())
    expected = tuple(POLICY_VISIBLE_RAW_OBSERVATION_KEYS)
    if configured != expected:
        raise RuntimeError(
            "VO Gym observation boundary mismatch "
            f"{configured} != {expected}"
        )
    missing = set(configured).difference(observations)
    if missing:
        raise RuntimeError(
            f"VO raw observation is missing configured keys {sorted(missing)}"
        )
    exposed = FORBIDDEN_POLICY_KEYS.intersection(configured)
    if exposed:
        raise RuntimeError(
            f"VO policy boundary exposes forbidden keys {sorted(exposed)}"
        )
    post_provider_keys = (
        set(configured).difference((VO_RGB_KEY, VO_DEPTH_KEY))
        | {"estimated_pose"}
    )
    exposed_after_provider = FORBIDDEN_POLICY_KEYS.intersection(
        post_provider_keys
    )
    if exposed_after_provider:
        raise RuntimeError(
            "VO post-provider policy boundary exposes forbidden keys "
            f"{sorted(exposed_after_provider)}"
        )


def _sensor_rotations(env: habitat.Env) -> dict[str, list[float]]:
    return {
        name: _sensor_rotation_coefficients(wrapper.node.rotation)
        for name, wrapper in env.sim.agents[0]._sensors.items()
    }


def _sensor_transforms(
    env: habitat.Env,
) -> dict[str, dict[str, list[float]]]:
    transforms = {}
    for name, wrapper in env.sim.agents[0]._sensors.items():
        transform = wrapper.node.absolute_transformation()
        rotation = mn.Quaternion.from_matrix(transform.rotation())
        transforms[name] = {
            "translation": [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            "rotation": _sensor_rotation_coefficients(rotation),
        }
    return transforms


def assert_fixed_look_transforms(
    previous: Mapping[str, Mapping[str, Sequence[float]]],
    current: Mapping[str, Mapping[str, Sequence[float]]],
) -> None:
    """Check the camera extrinsics that define a fixed-forward VO view.

    Habitat advances simulation time and renders a fresh frame for a LOOK
    action, so byte-identical RGB-D frames are not a valid camera-pose
    invariant.  The auxiliary camera's absolute transform is the relevant
    contract; the main camera must rotate while the auxiliary camera stays
    fixed.
    """

    for auxiliary in (VO_RGB_KEY, VO_DEPTH_KEY):
        if not np.allclose(
            previous[auxiliary]["translation"],
            current[auxiliary]["translation"],
            rtol=0.0,
            atol=1e-6,
        ) or not _equivalent_quaternion(
            previous[auxiliary]["rotation"],
            current[auxiliary]["rotation"],
        ):
            raise RuntimeError(
                f"fixed {auxiliary} transform changed during look action"
            )
    for main in ("rgb", "depth"):
        if _equivalent_quaternion(
            previous[main]["rotation"],
            current[main]["rotation"],
        ):
            raise RuntimeError(
                f"main {main} transform did not rotate during look action"
            )


def _compose_config(
    *,
    config_name: str,
    dataset_path: Path,
    content_scene: str,
    split: str,
    scenes_dir: Path,
    seed: int,
    max_episode_steps: int,
    vo_enabled: bool,
    gpu_device_id: int,
):
    global _HYDRA_PLUGIN_REGISTERED
    GlobalHydra.instance().clear()
    if not _HYDRA_PLUGIN_REGISTERED:
        register_hydra_plugin(HabitatBaselinesConfigPlugin)
        _HYDRA_PLUGIN_REGISTERED = True
    config_dir = Path(__file__).resolve().parents[1] / "experiments"
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(config_dir)
    ):
        config = patch_config(
            hydra.compose(config_name=config_name)
        )
    with read_write(config):
        config.habitat.seed = int(seed)
        config.habitat.dataset.data_path = str(dataset_path.resolve())
        config.habitat.dataset.content_scenes = [content_scene]
        config.habitat.dataset.split = split
        config.habitat.dataset.scenes_dir = str(scenes_dir.resolve())
        config.habitat.environment.max_episode_steps = int(
            max_episode_steps
        )
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(
            gpu_device_id
        )
        config.habitat_baselines.eval.video_option = []
        config.habitat_baselines.num_environments = 1
    if vo_enabled:
        configure_gt_isolated_vo(config)
    return config


def select_runtime_episode(
    episodes: Sequence[Any],
    *,
    runtime_episode_id: str,
    expected_scene_id: str,
    expected_target_category: str,
) -> Any:
    matching = [
        episode
        for episode in episodes
        if str(episode.episode_id) == str(runtime_episode_id)
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"equivalence dataset must contain exactly one runtime episode "
            f"with id {runtime_episode_id}, got {len(matching)} matches "
            f"among {len(episodes)} episodes"
        )
    episode = matching[0]
    runtime_scene = str(episode.scene_id).replace("\\", "/")
    expected_scene = str(expected_scene_id).replace("\\", "/")
    if not (
        runtime_scene == expected_scene
        or runtime_scene.endswith("/" + expected_scene.lstrip("/"))
    ):
        raise RuntimeError(
            f"runtime episode scene mismatch {runtime_scene} != "
            f"{expected_scene}"
        )
    if str(episode.object_category) != str(expected_target_category):
        raise RuntimeError(
            f"runtime episode target mismatch "
            f"{episode.object_category} != {expected_target_category}"
        )
    return episode


def _run_plan(
    *,
    role: str,
    config,
    runtime_episode_id: str,
    expected_scene_id: str,
    expected_target_category: str,
    plans: Sequence[Sequence[PlannedStep]],
    writer: JsonlWriter,
) -> list[dict[str, Any]]:
    dataset = habitat.make_dataset(
        id_dataset=config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    dataset.episodes = [
        select_runtime_episode(
            dataset.episodes,
            runtime_episode_id=runtime_episode_id,
            expected_scene_id=expected_scene_id,
            expected_target_category=expected_target_category,
        )
    ]
    records: list[dict[str, Any]] = []
    with habitat.Env(config=config.habitat, dataset=dataset) as env:
        for reset_index, plan in enumerate(plans):
            observations = env.reset()
            if role == "vo":
                _assert_vo_policy_boundary(config, observations)
            if role == "vo":
                for key in (VO_RGB_KEY, VO_DEPTH_KEY):
                    if key not in observations:
                        raise RuntimeError(
                            f"VO observation is missing {key}"
                        )
                if not np.array_equal(
                    observations["rgb"], observations[VO_RGB_KEY]
                ) or not np.array_equal(
                    observations["depth"], observations[VO_DEPTH_KEY]
                ):
                    raise RuntimeError(
                        "auxiliary VO sensors differ at forward-facing reset"
                    )
            state = env.sim.get_agent_state()
            origin = np.asarray(state.position, dtype=np.float64).copy()
            start_rotation = state.rotation
            previous_pose = np.zeros(3, dtype=np.float64)
            reference_pose = np.zeros(3, dtype=np.float64)
            pitch_level = 0
            previous_sensor_transforms = _sensor_transforms(env)
            reset_record = {
                "record_type": "reset",
                "role": role,
                "reset_index": reset_index,
                "scene_id": env.current_episode.scene_id,
                "episode_id": str(env.current_episode.episode_id),
                "position": [
                    float(value) for value in state.position
                ],
                "rotation": _rotation_coefficients(state.rotation),
                "observation_digests": _observation_digests(
                    observations
                ),
                "sensor_rotations": _sensor_rotations(env),
                "sensor_transforms": previous_sensor_transforms,
            }
            writer.write(reset_record)
            records.append(reset_record)

            for plan_step, planned in enumerate(plan, start=1):
                before_elapsed = int(env._elapsed_steps)
                observations = env.step(ACTION_NAMES[planned.action])
                after_elapsed = int(env._elapsed_steps)
                if after_elapsed != before_elapsed + 1:
                    raise RuntimeError(
                        f"{role} action accounting mismatch "
                        f"{before_elapsed}->{after_elapsed}"
                    )
                if env.episode_over:
                    raise RuntimeError(
                        f"{role} episode ended during bounded equivalence "
                        f"at reset={reset_index} step={plan_step}"
                    )
                if role == "vo":
                    _assert_vo_policy_boundary(config, observations)
                state = env.sim.get_agent_state()
                gt_pose_4d = _start_aligned_pose(
                    origin=origin,
                    start_rotation=start_rotation,
                    state=state,
                )
                gt_pose = gt_pose_4d[:3]
                local_delta = gt_local_zhao_delta(
                    previous_pose, gt_pose
                )
                reference_pose = compose_zhao_delta(
                    reference_pose, local_delta
                )
                if not np.allclose(
                    reference_pose,
                    gt_pose,
                    rtol=0.0,
                    atol=2e-5,
                ):
                    raise RuntimeError(
                        f"{role} GT-provider composition mismatch at "
                        f"reset={reset_index} step={plan_step}: "
                        f"{reference_pose} != {gt_pose}"
                    )
                previous_pose = gt_pose

                if planned.action == LOOK_UP:
                    pitch_level += 1
                elif planned.action == LOOK_DOWN:
                    pitch_level -= 1
                if role == "vo":
                    for key in (VO_RGB_KEY, VO_DEPTH_KEY):
                        if key not in observations:
                            raise RuntimeError(
                                f"VO observation lost auxiliary key {key}"
                            )
                    sensor_rotations = _sensor_rotations(env)
                    sensor_transforms = _sensor_transforms(env)
                    if pitch_level == 0:
                        if not np.array_equal(
                            observations["rgb"],
                            observations[VO_RGB_KEY],
                        ) or not np.array_equal(
                            observations["depth"],
                            observations[VO_DEPTH_KEY],
                        ):
                            raise RuntimeError(
                                "forward-facing main and VO sensors diverged"
                            )
                        for main, auxiliary in (
                            ("rgb", VO_RGB_KEY),
                            ("depth", VO_DEPTH_KEY),
                        ):
                            if not _equivalent_quaternion(
                                sensor_rotations[main],
                                sensor_rotations[auxiliary],
                            ):
                                raise RuntimeError(
                                    f"{main}/{auxiliary} rotations diverged "
                                    "at zero pitch"
                                )
                    if planned.action in (LOOK_UP, LOOK_DOWN):
                        assert_fixed_look_transforms(
                            previous_sensor_transforms,
                            sensor_transforms,
                        )
                    previous_sensor_transforms = sensor_transforms
                else:
                    sensor_rotations = _sensor_rotations(env)
                    sensor_transforms = _sensor_transforms(env)

                record = {
                    "record_type": "step",
                    "role": role,
                    "reset_index": reset_index,
                    "plan_step": plan_step,
                    "sequence": planned.sequence,
                    "measured": planned.measured,
                    "action": planned.action,
                    "action_name": ACTION_NAMES[planned.action],
                    "elapsed_steps": after_elapsed,
                    "position": [
                        float(value) for value in state.position
                    ],
                    "rotation": _rotation_coefficients(state.rotation),
                    "gt_start_aligned_pose": [
                        float(value) for value in gt_pose_4d
                    ],
                    "gt_local_zhao_delta": [
                        float(value) for value in local_delta
                    ],
                    "gt_provider_pose": [
                        float(value) for value in reference_pose
                    ],
                    "observation_digests": _observation_digests(
                        observations
                    ),
                    "sensor_rotations": sensor_rotations,
                    "sensor_transforms": sensor_transforms,
                    "camera_pitch_level": pitch_level,
                }
                writer.write(record)
                records.append(record)
            if pitch_level != 0:
                raise RuntimeError(
                    f"{role} plan ended at nonzero camera pitch "
                    f"{pitch_level}"
                )
    return records


def _compare_runs(
    baseline: Sequence[Mapping[str, Any]],
    vo: Sequence[Mapping[str, Any]],
) -> None:
    if len(baseline) != len(vo):
        raise RuntimeError(
            f"record length mismatch {len(baseline)} != {len(vo)}"
        )
    for index, (first, second) in enumerate(zip(baseline, vo)):
        identity_fields = (
            "record_type",
            "reset_index",
            "plan_step",
            "sequence",
            "measured",
            "action",
            "action_name",
            "elapsed_steps",
        )
        for field in identity_fields:
            if first.get(field) != second.get(field):
                raise RuntimeError(
                    f"record {index} field {field} mismatch: "
                    f"{first.get(field)} != {second.get(field)}"
                )
        if not np.allclose(
            first["position"],
            second["position"],
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"record {index} native endpoint position mismatch"
            )
        if not _equivalent_quaternion(
            first["rotation"], second["rotation"]
        ):
            raise RuntimeError(
                f"record {index} native endpoint rotation mismatch"
            )
        if first["observation_digests"] != second[
            "observation_digests"
        ]:
            raise RuntimeError(
                f"record {index} non-pose observation mismatch"
            )
        baseline_sensors = first["sensor_rotations"]
        vo_sensors = second["sensor_rotations"]
        for sensor in ("rgb", "depth"):
            if not _equivalent_quaternion(
                baseline_sensors[sensor], vo_sensors[sensor]
            ):
                raise RuntimeError(
                    f"record {index} main {sensor} rotation mismatch"
                )


def _sequence_summaries(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_floor_height_change: float = 2.0,
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for sequence in (
        "ordinary_same_floor",
        "turn_heavy",
        "stair_ascent",
        "stair_descent",
        "floor_revisit",
    ):
        selected = [
            record
            for record in records
            if record.get("record_type") == "step"
            and record.get("sequence") == sequence
        ]
        if not selected:
            raise RuntimeError(f"missing sequence {sequence}")
        poses = np.asarray(
            [record["gt_start_aligned_pose"] for record in selected],
            dtype=np.float64,
        )
        first = poses[0]
        last = poses[-1]
        summaries[sequence] = {
            "actions": len(selected),
            "start_pose": [float(value) for value in first],
            "end_pose": [float(value) for value in last],
            "min_height": float(poses[:, 3].min()),
            "max_height": float(poses[:, 3].max()),
            "height_delta": float(last[3] - first[3]),
            "planar_endpoint_delta": float(
                np.linalg.norm(last[:2] - first[:2])
            ),
        }
    ordinary = summaries["ordinary_same_floor"]
    if ordinary["max_height"] - ordinary["min_height"] > 0.25:
        raise RuntimeError(
            "ordinary sequence is not bounded to one floor"
        )
    ascent = summaries["stair_ascent"]
    if ascent["height_delta"] < minimum_floor_height_change:
        raise RuntimeError(
            f"stair ascent gained only {ascent['height_delta']:.3f} m"
        )
    descent = summaries["stair_descent"]
    if descent["height_delta"] > -minimum_floor_height_change:
        raise RuntimeError(
            f"stair descent lost only {descent['height_delta']:.3f} m"
        )
    revisit = summaries["floor_revisit"]
    if (
        revisit["max_height"] - revisit["min_height"]
        < minimum_floor_height_change
    ):
        raise RuntimeError(
            "floor revisit did not traverse two height levels"
        )
    if abs(revisit["height_delta"]) > 0.35:
        raise RuntimeError(
            f"floor revisit height closure is "
            f"{revisit['height_delta']:.3f} m"
        )
    if revisit["planar_endpoint_delta"] > 0.75:
        raise RuntimeError(
            f"floor revisit planar closure is "
            f"{revisit['planar_endpoint_delta']:.3f} m"
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", default="eval_ascent_hm3d.yaml"
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--content-scene", default="hm3d_r0_006")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--runtime-episode-id", required=True)
    parser.add_argument("--expected-scene-id", required=True)
    parser.add_argument("--expected-target-category", required=True)
    parser.add_argument(
        "--split", choices=("train", "val"), required=True
    )
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--action-trace", type=Path, required=True)
    parser.add_argument(
        "--expected-action-trace-sha256", required=True
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--ordinary-end-step", type=int, default=60)
    parser.add_argument(
        "--stair-setup-end-step", type=int, default=399
    )
    parser.add_argument(
        "--stair-transition-start-step", type=int, default=400
    )
    parser.add_argument(
        "--stair-transition-end-step", type=int, default=433
    )
    parser.add_argument(
        "--recorded-transition-direction",
        choices=("ascent", "descent"),
        default="ascent",
    )
    parser.add_argument(
        "--minimum-floor-height-change", type=float, default=2.0
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    writer = JsonlWriter(args.output_jsonl)
    try:
        for path in (
            args.dataset_path,
            args.scenes_dir,
            args.action_trace,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        action_sha = sha256(args.action_trace)
        if action_sha != args.expected_action_trace_sha256:
            raise RuntimeError(
                f"action trace SHA mismatch {action_sha}"
            )
        recorded_actions = [
            int(line)
            for line in args.action_trace.read_text().splitlines()
            if line.strip()
        ]
        if args.minimum_floor_height_change <= 0.0:
            raise ValueError(
                "minimum floor height change must be positive"
            )
        plans = build_sequence_plan(
            recorded_actions,
            ordinary_end_step=args.ordinary_end_step,
            stair_setup_end_step=args.stair_setup_end_step,
            stair_transition_start_step=(
                args.stair_transition_start_step
            ),
            stair_transition_end_step=args.stair_transition_end_step,
            recorded_transition_direction=(
                args.recorded_transition_direction
            ),
        )
        required_steps = max(max(map(len, plans)), 1)
        if required_steps >= args.max_episode_steps:
            raise RuntimeError(
                f"max episode steps {args.max_episode_steps} does not "
                f"cover largest plan {required_steps}"
            )
        writer.write(
            {
                "record_type": "run_metadata",
                "schema": SCHEMA,
                "config_name": args.config_name,
                "dataset_path": str(args.dataset_path.resolve()),
                "content_scene": args.content_scene,
                "episode_id": str(args.episode_id),
                "runtime_episode_id": str(args.runtime_episode_id),
                "expected_scene_id": args.expected_scene_id,
                "expected_target_category": (
                    args.expected_target_category
                ),
                "split": args.split,
                "scenes_dir": str(args.scenes_dir.resolve()),
                "action_trace": str(args.action_trace.resolve()),
                "action_trace_sha256": action_sha,
                "seed": args.seed,
                "gpu_device_id": args.gpu_device_id,
                "max_episode_steps": args.max_episode_steps,
                "ordinary_end_step": args.ordinary_end_step,
                "stair_setup_end_step": args.stair_setup_end_step,
                "stair_transition_start_step": (
                    args.stair_transition_start_step
                ),
                "stair_transition_end_step": (
                    args.stair_transition_end_step
                ),
                "recorded_transition_direction": (
                    args.recorded_transition_direction
                ),
                "minimum_floor_height_change": (
                    args.minimum_floor_height_change
                ),
                "sequence_count": 5,
                "scientific_role": "engineering_equivalence_only",
                "not_vo_performance": True,
            }
        )
        common = dict(
            config_name=args.config_name,
            dataset_path=args.dataset_path,
            content_scene=args.content_scene,
            split=args.split,
            scenes_dir=args.scenes_dir,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            gpu_device_id=args.gpu_device_id,
        )
        baseline = _run_plan(
            role="baseline",
            config=_compose_config(**common, vo_enabled=False),
            runtime_episode_id=args.runtime_episode_id,
            expected_scene_id=args.expected_scene_id,
            expected_target_category=args.expected_target_category,
            plans=plans,
            writer=writer,
        )
        vo = _run_plan(
            role="vo",
            config=_compose_config(**common, vo_enabled=True),
            runtime_episode_id=args.runtime_episode_id,
            expected_scene_id=args.expected_scene_id,
            expected_target_category=args.expected_target_category,
            plans=plans,
            writer=writer,
        )
        _compare_runs(baseline, vo)
        summaries = _sequence_summaries(
            vo,
            minimum_floor_height_change=(
                args.minimum_floor_height_change
            ),
        )
        writer.write(
            {
                "record_type": "gate_result",
                "schema": SCHEMA,
                "status": "pass",
                "sequence_summaries": summaries,
                "native_endpoint_equivalence": True,
                "one_step_accounting": True,
                "non_pose_observation_equivalence": True,
                "gt_provider_composition_equivalence": True,
                "fixed_look_auxiliary_sensor": True,
                "forbidden_policy_observations": [],
            }
        )
        return 0
    except BaseException as exc:
        writer.write(
            {
                "record_type": "technical_failure",
                "schema": SCHEMA,
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
