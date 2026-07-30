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
            "ASCENT_SINGLETON_WATCHDOG_STOP_GUARD",
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
        self.assertFalse(trace["stop_guard_enabled"])

    def test_stop_guard_is_explicitly_enabled_from_environment(self):
        with patch.dict(
            os.environ,
            {"ASCENT_SINGLETON_WATCHDOG_STOP_GUARD": "1"},
            clear=True,
        ):
            watchdog = SingletonFrontierWatchdog.from_environment()

        self.assertTrue(watchdog.stop_guard_enabled)

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
    @staticmethod
    def _run_until_trigger(*, stop_guard_enabled):
        from ascent.llm_planner import Ascent_LLM_Planner

        planner = Ascent_LLM_Planner.__new__(Ascent_LLM_Planner)
        planner._singleton_frontier_watchdogs = [
            SingletonFrontierWatchdog(
                enabled=True,
                position_tolerance_m=0.25,
                patience_steps=2,
                min_progress_m=0.3,
                stop_guard_enabled=stop_guard_enabled,
            )
        ]
        planner.last_singleton_watchdog_trace = [{}]
        planner.pending_singleton_watchdog_frontier = [None]
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
            if value != 1.0:
                raise AssertionError(f"unexpected value: {value}")

        return planner, obstacle, frontier

    def test_singleton_fast_path_disables_frontier_on_trigger(self):
        planner, obstacle, frontier = self._run_until_trigger(
            stop_guard_enabled=True
        )

        self.assertIn(tuple(frontier[0]), obstacle._disabled_frontiers)
        self.assertTrue(
            planner.last_singleton_watchdog_trace[0]["triggered"]
        )
        self.assertEqual(
            planner.pending_singleton_watchdog_frontier[0],
            tuple(frontier[0]),
        )

    def test_guard_disabled_preserves_original_trigger_behavior(self):
        planner, obstacle, frontier = self._run_until_trigger(
            stop_guard_enabled=False
        )

        self.assertIn(tuple(frontier[0]), obstacle._disabled_frontiers)
        self.assertTrue(
            planner.last_singleton_watchdog_trace[0]["triggered"]
        )
        self.assertIsNone(
            planner.pending_singleton_watchdog_frontier[0]
        )


class SingletonFrontierStopGuardTest(unittest.TestCase):
    @staticmethod
    def _make_policy_for_no_frontier(frontier, *, stair_action=None):
        from ascent.ascent_policy import Ascent_Policy, torch

        obstacle = SimpleNamespace(
            _disabled_frontiers={tuple(frontier)},
            _this_floor_explored=False,
            _reinitialize_flag=True,
            _floor_num_steps=100,
            _explored_up_stair=stair_action is None,
            _explored_down_stair=True,
        )
        planner = SimpleNamespace(
            pending_singleton_watchdog_frontier=[tuple(frontier)],
            last_singleton_watchdog_trace=[{}],
        )

        def clear_pending(env):
            pending = planner.pending_singleton_watchdog_frontier[env]
            planner.pending_singleton_watchdog_frontier[env] = None
            return pending

        def choose_frontier(*args, **kwargs):
            planner.last_singleton_watchdog_trace[0] = {
                "status": "tracking_started",
                "triggered": False,
            }
            return frontier, 1.0

        planner.clear_pending_singleton_watchdog_frontier = clear_pending
        planner._get_best_frontier_with_llm = choose_frontier
        map_controller = SimpleNamespace(
            _obstacle_map=[obstacle],
            _value_map=[None],
            _object_map=[None],
            _obstacle_map_list=[[]],
            _value_map_list=[[]],
            _object_map_list=[[]],
            _cur_floor_index=[0],
            _frontier_stick_step=[0],
        )
        policy = Ascent_Policy.__new__(Ascent_Policy)
        policy.llm_planner = planner
        policy._map_controller = map_controller
        policy._observations_cache = [
            {"frontier_sensor": np.asarray([frontier])}
        ]
        policy._last_explore_trace = [{}]
        policy._num_steps = [100]
        policy._last_frontier_distance = [1.0]
        policy.cur_frontier = [None]
        policy.topk = 3
        policy._pointnav_stop_radius = 0.25
        policy._stop_action = torch.tensor([[0]])
        policy._pointnav = lambda *args, **kwargs: torch.tensor([[1]])
        policy._navigate_stair_if_unexplored_floor = (
            lambda *args, **kwargs: stair_action
        )
        return policy, planner, obstacle, torch

    def test_immediate_stop_is_replaced_by_restored_frontier_action(self):
        frontier = np.array([5.0, 0.0])
        policy, planner, obstacle, torch = (
            self._make_policy_for_no_frontier(frontier)
        )

        action = policy._explore(
            observations={},
            env=0,
            masks=torch.tensor([[1]]),
        )

        self.assertEqual(action.item(), 1)
        self.assertNotIn(tuple(frontier), obstacle._disabled_frontiers)
        self.assertFalse(obstacle._this_floor_explored)
        self.assertTrue(
            policy._last_explore_trace[0]["stop_guard_suppressed"]
        )
        self.assertEqual(
            policy._last_explore_trace[0]["no_frontier_action"],
            "watchdog_stop_guard_restore",
        )
        self.assertIsNone(
            planner.pending_singleton_watchdog_frontier[0]
        )

    def test_stair_fallback_keeps_watchdog_deletion(self):
        frontier = np.array([5.0, 0.0])
        from ascent.ascent_policy import torch

        stair_action = torch.tensor([[2]])
        policy, planner, obstacle, torch = (
            self._make_policy_for_no_frontier(
                frontier,
                stair_action=stair_action,
            )
        )

        action = policy._explore(
            observations={},
            env=0,
            masks=torch.tensor([[1]]),
        )

        self.assertEqual(action.item(), 2)
        self.assertIn(tuple(frontier), obstacle._disabled_frontiers)
        self.assertTrue(obstacle._this_floor_explored)
        self.assertEqual(
            policy._last_explore_trace[0]["no_frontier_action"],
            "stair_navigation",
        )
        self.assertIsNone(
            planner.pending_singleton_watchdog_frontier[0]
        )

    def test_restores_only_pending_frontier_still_in_sensor_output(self):
        from ascent.ascent_policy import Ascent_Policy

        frontier = np.array([5.0, 0.0])
        obstacle = SimpleNamespace(
            _disabled_frontiers={tuple(frontier)},
            _this_floor_explored=True,
        )
        planner = SimpleNamespace(
            pending_singleton_watchdog_frontier=[tuple(frontier)]
        )

        def clear_pending(env):
            pending = planner.pending_singleton_watchdog_frontier[env]
            planner.pending_singleton_watchdog_frontier[env] = None
            return pending

        planner.clear_pending_singleton_watchdog_frontier = clear_pending
        policy = Ascent_Policy.__new__(Ascent_Policy)
        policy.llm_planner = planner
        policy._map_controller = SimpleNamespace(
            _obstacle_map=[obstacle]
        )

        restored = policy._restore_watchdog_frontier_before_stop(
            0,
            [frontier],
            previous_this_floor_explored=False,
        )

        np.testing.assert_array_equal(restored, frontier)
        self.assertNotIn(tuple(frontier), obstacle._disabled_frontiers)
        self.assertFalse(obstacle._this_floor_explored)
        self.assertIsNone(
            planner.pending_singleton_watchdog_frontier[0]
        )

    def test_does_not_restore_stale_pending_frontier(self):
        from ascent.ascent_policy import Ascent_Policy

        pending_frontier = np.array([5.0, 0.0])
        obstacle = SimpleNamespace(
            _disabled_frontiers={tuple(pending_frontier)},
            _this_floor_explored=True,
        )
        planner = SimpleNamespace(
            pending_singleton_watchdog_frontier=[
                tuple(pending_frontier)
            ]
        )

        def clear_pending(env):
            pending = planner.pending_singleton_watchdog_frontier[env]
            planner.pending_singleton_watchdog_frontier[env] = None
            return pending

        planner.clear_pending_singleton_watchdog_frontier = clear_pending
        policy = Ascent_Policy.__new__(Ascent_Policy)
        policy.llm_planner = planner
        policy._map_controller = SimpleNamespace(
            _obstacle_map=[obstacle]
        )

        restored = policy._restore_watchdog_frontier_before_stop(
            0,
            [np.array([4.0, 0.0])],
            previous_this_floor_explored=False,
        )

        self.assertIsNone(restored)
        self.assertIn(
            tuple(pending_frontier), obstacle._disabled_frontiers
        )
        self.assertTrue(obstacle._this_floor_explored)


if __name__ == "__main__":
    unittest.main()
