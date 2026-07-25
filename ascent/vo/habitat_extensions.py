"""Habitat VO sensors, fixed-look actions, and evaluation-only GT."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from habitat import registry
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import (
    MeasurementConfig,
)
from habitat.core.embodied_task import Measure
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSimDepthSensor,
    HabitatSimRGBSensor,
)
from habitat.tasks.nav.nav import HeadingSensor, NavigationMovementAgentAction
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
import magnum as mn
from omegaconf import OmegaConf, open_dict

from ascent.vo.pose_provider import (
    VO_DEPTH_KEY,
    VO_RGB_KEY,
)

POLICY_VISIBLE_RAW_OBSERVATION_KEYS = (
    "rgb",
    "depth",
    "objectgoal",
    VO_RGB_KEY,
    VO_DEPTH_KEY,
)


@registry.register_sensor
class VOHabitatSimRGBSensor(HabitatSimRGBSensor):
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return VO_RGB_KEY


@registry.register_sensor
class VOHabitatSimDepthSensor(HabitatSimDepthSensor):
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return VO_DEPTH_KEY


class _VOFixedLookAction(NavigationMovementAgentAction):
    look_sign: float

    def _move_main_cameras(self, amount: float) -> None:
        if len(self._sim.agents) != 1:
            raise RuntimeError("VO ObjectNav supports one simulator agent")
        for sensor_name, sensor_wrapper in self._sim.agents[0]._sensors.items():
            if sensor_name in (VO_RGB_KEY, VO_DEPTH_KEY):
                continue
            sensor_wrapper.node.rotation = (
                sensor_wrapper.node.rotation
                * mn.Quaternion.rotation(
                    mn.Deg(amount), mn.Vector3.x_axis()
                )
            )

    def step(self, *args: Any, task, **kwargs: Any) -> None:
        self._move_main_cameras(self.look_sign * self._tilt_angle)
        return None


@registry.register_task_action
class VOFixedLookUpAction(_VOFixedLookAction):
    look_sign = 1.0


@registry.register_task_action
class VOFixedLookDownAction(_VOFixedLookAction):
    look_sign = -1.0


@registry.register_measure
class GTStartAlignedPoseMeasure(Measure):
    """Evaluation-only pose; never appears in the policy observation."""

    cls_uuid = "gt_start_aligned_pose"

    def __init__(self, sim, config, *args: Any, **kwargs: Any) -> None:
        self._sim = sim
        self._origin = None
        self._start_rotation = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return GTStartAlignedPoseMeasure.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any) -> None:
        self._origin = np.asarray(episode.start_position, dtype=np.float64)
        self._start_rotation = quaternion_from_coeff(
            episode.start_rotation
        )
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any) -> None:
        if self._origin is None or self._start_rotation is None:
            raise RuntimeError("GT diagnostic measure was not reset")
        state = self._sim.get_agent_state()
        relative = quaternion_rotate_vector(
            self._start_rotation.inverse(),
            np.asarray(state.position, dtype=np.float64) - self._origin,
        )
        yaw = HeadingSensor._quat_to_xy_heading(
            None, state.rotation.inverse() * self._start_rotation
        )[0]
        self._metric = {
            "x": float(-relative[2]),
            "y": float(-relative[0]),
            "yaw": float(yaw),
            "height": float(relative[1]),
        }


@dataclass
class GTStartAlignedPoseMeasurementConfig(MeasurementConfig):
    type: str = GTStartAlignedPoseMeasure.__name__


def configure_gt_isolated_vo(config) -> None:
    """Mutate a composed ASCENT config before any environment is created."""

    if not bool(config.ascent_vo.enabled):
        raise RuntimeError("ASCENT-VO config is not enabled")
    if float(config.habitat.simulator.turn_angle) != 30.0:
        raise RuntimeError(
            "ASCENT-VO requires Zhao's native 30-degree turn granularity"
        )

    agent_config = get_agent_config(config.habitat.simulator)
    rgb_config = OmegaConf.create(
        deepcopy(OmegaConf.to_container(agent_config.sim_sensors.rgb_sensor))
    )
    depth_config = OmegaConf.create(
        deepcopy(
            OmegaConf.to_container(agent_config.sim_sensors.depth_sensor)
        )
    )
    rgb_config.type = VOHabitatSimRGBSensor.__name__
    depth_config.type = VOHabitatSimDepthSensor.__name__

    with read_write(config):
        with open_dict(agent_config.sim_sensors):
            agent_config.sim_sensors.vo_rgb_sensor = rgb_config
            agent_config.sim_sensors.vo_depth_sensor = depth_config

        config.habitat.task.actions.look_up.type = VOFixedLookUpAction.__name__
        config.habitat.task.actions.look_down.type = (
            VOFixedLookDownAction.__name__
        )

        with open_dict(config.habitat.task.lab_sensors):
            for sensor_name in ("gps_sensor", "compass_sensor"):
                config.habitat.task.lab_sensors.pop(sensor_name, None)

        with open_dict(config.habitat.task.measurements):
            config.habitat.task.measurements.gt_start_aligned_pose = (
                GTStartAlignedPoseMeasurementConfig()
            )

        # ASCENT's inherited FrontierSensor needs HeadingSensor during
        # Habitat's internal sensor update, but ASCENT's policy does not
        # consume either output.  HabGymWrapper is therefore the explicit
        # boundary: only the camera, goal, and auxiliary VO frames leave the
        # environment.  The VO provider consumes the auxiliary frames and
        # injects estimated_pose before batch_obs reaches the policy.
        config.habitat.gym.obs_keys = list(
            POLICY_VISIBLE_RAW_OBSERVATION_KEYS
        )

    remaining = {
        name
        for name in config.habitat.task.lab_sensors
        if name in ("gps_sensor", "compass_sensor")
    }
    if remaining:
        raise RuntimeError(
            f"GT pose lab sensors survived VO isolation: {sorted(remaining)}"
        )
    required_internal = {"heading_sensor", "base_explorer", "frontier_sensor"}
    missing_internal = required_internal.difference(
        config.habitat.task.lab_sensors
    )
    if missing_internal:
        raise RuntimeError(
            "missing internal ASCENT compatibility sensors: "
            f"{sorted(missing_internal)}"
        )
