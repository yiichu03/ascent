import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ascent.singleton_frontier_watchdog import SingletonFrontierWatchdog


class SingletonFrontierWatchdogStateTest(unittest.TestCase):
    def test_default_environment_is_disabled(self):
        names = [
            "ASCENT_SINGLETON_FRONTIER_WATCHDOG",
            "ASCENT_SINGLETON_WATCHDOG_POSITION_TOLERANCE_M",
            "ASCENT_SINGLETON_WATCHDOG_PATIENCE_STEPS",
            "ASCENT_SINGLETON_WATCHDOG_MIN_PROGRESS_M",
        ]
        clean_env = {key: value for key, value in os.environ.items() if key not in names}
        with patch.dict(os.environ, clean_env, clear=True):
            watchdog = SingletonFrontierWatchdog.from_environment()

        trace = watchdog.observe(
            np.array([5.0, 0.0]),
            np.array([0.0, 0.0]),
            policy_step=1,
            floor_index=0,
        )
        self.assertFalse(trace["enabled"])
        self.assertFalse(trace["triggered"])
        self.assertEqual(trace["status"], "disabled")
        self.assertEqual(trace["patience_steps"], 80)
        self.assertEqual(trace["min_progress_m"], 0.2)

    def test_stable_no_progress_triggers_after_patience(self):
        watchdog = SingletonFrontierWatchdog(
            enabled=True,
            position_tolerance_m=0.25,
            patience_steps=30,
            min_progress_m=0.3,
        )
        frontier = np.array([5.0, 0.0])
        robot = np.array([0.0, 0.0])

        trace = watchdog.observe(
            frontier, robot, policy_step=0, floor_index=0
        )
        self.assertEqual(trace["status"], "tracking_started")
        for step in range(1, 30):
            trace = watchdog.observe(
                frontier, robot, policy_step=step, floor_index=0
            )
            self.assertFalse(trace["triggered"])

        trace = watchdog.observe(
            frontier, robot, policy_step=30, floor_index=0
        )
        self.assertTrue(trace["triggered"])
        self.assertEqual(trace["no_progress_steps"], 30)

    def test_cumulative_meaningful_progress_resets_patience(self):
        watchdog = SingletonFrontierWatchdog(
            enabled=True,
            position_tolerance_m=0.25,
            patience_steps=3,
            min_progress_m=0.3,
        )
        frontier = np.array([5.0, 0.0])
        positions = [
            np.array([0.0, 0.0]),
            np.array([0.1, 0.0]),
            np.array([0.2, 0.0]),
            np.array([0.31, 0.0]),
        ]
        traces = [
            watchdog.observe(
                frontier, robot, policy_step=step, floor_index=0
            )
            for step, robot in enumerate(positions)
        ]

        self.assertFalse(traces[2]["triggered"])
        self.assertEqual(traces[3]["status"], "meaningful_progress")
        self.assertEqual(traces[3]["no_progress_steps"], 0)

    def test_frontier_motion_step_gap_and_floor_change_reset_tracking(self):
        watchdog = SingletonFrontierWatchdog(
            enabled=True,
            position_tolerance_m=0.25,
            patience_steps=2,
            min_progress_m=0.3,
        )
        robot = np.array([0.0, 0.0])
        watchdog.observe(
            np.array([5.0, 0.0]), robot, policy_step=0, floor_index=0
        )
        trace = watchdog.observe(
            np.array([5.3, 0.0]), robot, policy_step=1, floor_index=0
        )
        self.assertEqual(trace["status"], "tracking_started")
        trace = watchdog.observe(
            np.array([5.3, 0.0]), robot, policy_step=3, floor_index=0
        )
        self.assertEqual(trace["status"], "tracking_started")
        trace = watchdog.observe(
            np.array([5.3, 0.0]), robot, policy_step=4, floor_index=1
        )
        self.assertEqual(trace["status"], "tracking_started")


class SingletonFrontierPlannerIntegrationTest(unittest.TestCase):
    def test_singleton_fast_path_disables_frontier_on_trigger(self):
        from ascent.llm_planner import Ascent_LLM_Planner

        planner = Ascent_LLM_Planner.__new__(Ascent_LLM_Planner)
        planner._singleton_frontier_watchdogs = [
            SingletonFrontierWatchdog(
                enabled=True,
                position_tolerance_m=0.25,
                patience_steps=2,
                min_progress_m=0.3,
            )
        ]
        planner.last_singleton_watchdog_trace = [{}]
        obstacle = SimpleNamespace(_disabled_frontiers=set())
        frontier = np.array([[5.0, 0.0]])
        observations = [{"robot_xy": np.array([0.0, 0.0])}]

        for step in range(3):
            selected, value = planner._get_best_frontier_with_llm(
                observations,
                [obstacle],
                [None],
                [None],
                [[]],
                [[]],
                [[]],
                frontier,
                env=0,
                cur_floor_index=[0],
                num_steps=[step],
            )
            np.testing.assert_array_equal(selected, frontier[0])
            self.assertEqual(value, 1.0)

        self.assertIn(tuple(frontier[0]), obstacle._disabled_frontiers)
        self.assertTrue(
            planner.last_singleton_watchdog_trace[0]["triggered"]
        )


if __name__ == "__main__":
    unittest.main()
