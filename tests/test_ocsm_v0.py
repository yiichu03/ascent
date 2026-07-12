import ast
import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ascent.ocsm import OCSMConfig, ObjectConditionedSearchMemory


BASELINE_COMMIT = "ea0d4a367f162aab4978acb702a8eab05a9cf1f0"
ASCENT_ROOT = Path(__file__).resolve().parents[1]


class StripAnnotations(ast.NodeTransformer):
    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        node.decorator_list = []
        node.returns = None
        node.type_comment = None
        return node


def compile_planner_method(source):
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Ascent_LLM_Planner"
    )
    method_node = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_get_best_frontier_with_llm"
    )
    method_node = StripAnnotations().visit(copy.deepcopy(method_node))
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, "<planner>", "exec"), namespace)
    return namespace["_get_best_frontier_with_llm"]


def compile_policy_explore(source):
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Ascent_Policy"
    )
    method_node = next(
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_explore"
    )
    method_node = StripAnnotations().visit(copy.deepcopy(method_node))
    module = ast.fix_missing_locations(ast.Module(body=[method_node], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, "<policy_explore>", "exec"), namespace)
    return namespace["_explore"]


class FakeAction:
    def __init__(self, value):
        self.value = int(value)

    def item(self):
        return self.value

    def fill_(self, value):
        self.value = int(value)
        return self

    def to(self, device):
        return self


def make_explore_policy(case):
    obstacle = SimpleNamespace(
        _disabled_frontiers=set(),
        _reinitialize_flag=True,
        _floor_num_steps=100,
        _explored_up_stair=case != "stair",
        _up_stair_frontiers=np.array([(1.0, 1.0)]),
        _explored_down_stair=True,
        _down_stair_frontiers=np.empty((0, 2)),
        _this_floor_explored=False,
        _frontier_stick_step=0,
    )
    controller = SimpleNamespace(
        _obstacle_map=[obstacle],
        _value_map=[object()],
        _object_map=[SimpleNamespace(has_object=lambda target: False)],
        _obstacle_map_list=[[]],
        _value_map_list=[[]],
        _object_map_list=[[]],
        _cur_floor_index=[0],
        _frontier_stick_step=[0],
        _target_object=["chair"],
    )
    frontiers = (
        np.array([(1.0, 2.0), (3.0, 4.0)])
        if case == "active"
        else np.empty((0, 2))
    )
    planner = SimpleNamespace(
        last_frontier_disable_event=[None],
        _get_best_frontier_with_llm=lambda *args, **kwargs: (
            np.array([3.0, 4.0]), 0.75
        ),
    )
    policy = SimpleNamespace(
        _ocsm=None,
        _instrumentation_enabled=False,
        _observations_cache=[{"frontier_sensor": frontiers, "robot_xy": np.zeros(2)}],
        _map_controller=controller,
        llm_planner=planner,
        topk=3,
        _num_steps=[20],
        _last_frontier_distance=[0.0],
        cur_frontier=[np.array([])],
        _pointnav_stop_radius=0.5,
        _stop_action=FakeAction(0),
        navigation_calls=[],
        pointnav_calls=[],
    )
    policy._navigate_stair_if_unexplored_floor = lambda observations, env, direction: (
        policy.navigation_calls.append(direction)
        or (FakeAction(4) if case == "stair" and direction == "up" else None)
    )
    policy._pointnav = lambda observations, goal, stop, env, stop_radius: (
        policy.pointnav_calls.append(np.asarray(goal).tolist()) or FakeAction(2)
    )
    policy._handle_stairwell_reinitialization = lambda env, masks: FakeAction(5)
    return policy


def explore_signature(method, policy):
    action = method(policy, {}, 0, SimpleNamespace(device="cpu"))
    return {
        "action": action.value,
        "frontier": np.asarray(policy.cur_frontier[0]).tolist(),
        "this_floor_explored": policy._map_controller._obstacle_map[0]._this_floor_explored,
        "navigation_calls": policy.navigation_calls,
        "pointnav_calls": policy.pointnav_calls,
    }


class PlannerObstacle:
    def __init__(self):
        self._finish_first_explore = True
        self._neighbor_search = False
        self._best_frontier_selection_count = {}
        self._disabled_frontiers = set()


def make_planner(case, ocsm=None, instrumentation=False):
    planner = SimpleNamespace(
        _instrumentation_enabled=instrumentation,
        _ocsm=ocsm,
        _target_object=["chair"],
        _last_value=[-999.0],
        _last_frontier=[np.array([-1.0, -1.0])],
        _force_frontier=[np.array([1.0, 2.0])],
        last_frontier_disable_event=[None],
        last_decision_trace=[{}],
        last_llm_trace=[{}],
        last_multi_floor_trace=[{}],
    )
    planner._sort_frontiers_by_value = lambda *args: (
        np.asarray(args[2]), [0.9, 0.4][: len(args[2])]
    )
    planner._try_force_frontier = lambda points, values, env: (
        (points[0], values[0])
        if case == "force" and np.array_equal(points[0], [1.0, 2.0])
        else (None, None)
    )
    planner._try_nearby_frontier = lambda *args: (None, None, False)
    planner._decide_frontier_with_llm = lambda *args: (
        np.asarray(args[2])[-1], -100 if case == "sentinel" else args[3][-1]
    )
    planner._handle_frontier_stick_and_disable = lambda *args: None
    return planner


def planner_signature(method, planner, frontiers):
    obstacle = PlannerObstacle()
    result = method(
        planner,
        [{"robot_xy": np.array([0.0, 0.0])}],
        [obstacle],
        [object()],
        [object()],
        [[]],
        [[]],
        [[]],
        np.asarray(frontiers),
    )
    return {
        "frontier": np.asarray(result[0]).tolist(),
        "value": float(result[1]),
        "last_value": float(planner._last_value[0]),
        "last_frontier": np.asarray(planner._last_frontier[0]).tolist(),
        "finish_first_explore": obstacle._finish_first_explore,
        "neighbor_search": obstacle._neighbor_search,
        "trace": json.loads(
            json.dumps(
                planner.last_decision_trace[0],
                sort_keys=True,
                default=lambda value: value.tolist()
                if isinstance(value, np.ndarray)
                else value.item(),
            )
        ),
    }


class FakeObstacleMap:
    def __init__(self, size=80, pixels_per_meter=10):
        self.explored_area = np.zeros((size, size), dtype=bool)
        self.pixels_per_meter = pixels_per_meter
        self.origin = size // 2

    def _xy_to_px(self, points):
        points = np.asarray(points, dtype=float)
        return np.column_stack(
            (
                self.origin + points[:, 0] * self.pixels_per_meter,
                self.origin + points[:, 1] * self.pixels_per_meter,
            )
        ).astype(int)

    def add_explored_pixels(self, count):
        self.explored_area.flat[:count] = True

    def explore_at(self, xy):
        x, y = self._xy_to_px(np.atleast_2d(xy))[0]
        self.explored_area[y, x] = True


def make_memory(num_envs=1):
    return ObjectConditionedSearchMemory(
        num_envs,
        OCSMConfig(
            enabled=True,
            association_radius_m=0.5,
            low_gain_area_m2=0.5,
            hard_streak=2,
            suppression_radius_m=0.5,
        ),
    )


def start(memory, obstacle, env=0, target="chair", frontier=(0.0, 0.0), step=0):
    memory.start_attempt(
        env=env,
        target=target,
        floor_index=0,
        frontier=frontier,
        step=step,
        obstacle_map=obstacle,
        target_present=False,
        up_stair_present=False,
        down_stair_present=False,
        frontiers=np.array([frontier, (2.0, 0.0)]),
        robot_xy=(3.0, 0.0),
    )


class OCSMV0Test(unittest.TestCase):
    def test_default_off_and_configurable_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = OCSMConfig.from_env()
        self.assertFalse(config.enabled)
        self.assertEqual(config.association_radius_m, 0.5)
        self.assertEqual(config.low_gain_area_m2, 0.5)
        self.assertEqual(config.hard_streak, 2)
        self.assertEqual(config.suppression_radius_m, 0.5)

    def test_association_distance_and_replanning_switch(self):
        memory = make_memory()
        obstacle = FakeObstacleMap()
        start(memory, obstacle)
        memory.register_selection(
            0, "chair", 0, (0.5, 0.0), 1, (3.0, 0.0), obstacle,
            False, False, False, np.array([(0.5, 0.0), (2.0, 0.0)]),
        )
        self.assertEqual(memory.trace(0)["active_attempt"]["attempt_id"], 1)
        self.assertEqual(memory.trace(0)["active_attempt"]["association_distance_m"], 0.5)
        memory.register_selection(
            0, "chair", 0, (0.51, 0.0), 2, (3.0, 0.0), obstacle,
            False, False, False, np.array([(0.51, 0.0), (2.0, 0.0)]),
        )
        self.assertEqual(memory.trace(0)["active_attempt"]["attempt_id"], 2)
        self.assertEqual(memory.trace(0)["last_event"]["event"], "attempt_started")
        self.assertEqual(memory.trace(0)["memory_entries"], [])

    def test_area_threshold_below_equal_above(self):
        expected_low = {49: True, 50: False, 51: False}
        for pixels, low_gain in expected_low.items():
            memory = make_memory()
            obstacle = FakeObstacleMap()
            start(memory, obstacle)
            obstacle.add_explored_pixels(pixels)
            event = memory.finish_attempt(
                0, "completed", "test", 5, obstacle, np.array([(2.0, 0.0)])
            )
            self.assertAlmostEqual(event["explored_area_delta_m2"], pixels / 100)
            self.assertEqual(event["low_gain"], low_gain)
            self.assertEqual(event["streak_after"], 1 if low_gain else 0)

    def test_soft_hard_calibration_and_fallback(self):
        memory = make_memory()
        obstacle = FakeObstacleMap()
        points = np.array([(0.0, 0.0), (2.0, 0.0)])
        for attempt_index in range(2):
            start(memory, obstacle, step=attempt_index * 2)
            memory.finish_attempt(
                0, "completed", "low_gain", attempt_index * 2 + 1, obstacle, points
            )
            calibrated, values, trace = memory.calibrate_candidates(
                0, "chair", 0, points, [0.9, 0.8]
            )
            if attempt_index == 0:
                self.assertEqual(calibrated.tolist(), [[2.0, 0.0], [0.0, 0.0]])
                self.assertEqual(values, [0.8, 0.9])
                self.assertEqual(trace["candidates"][0]["status"], "soft")
            else:
                self.assertEqual(calibrated.tolist(), [[2.0, 0.0]])
                self.assertEqual(values, [0.8])
                self.assertEqual(trace["candidates"][0]["status"], "hard-suppressed")
        calibrated, values, trace = memory.calibrate_candidates(
            0, "chair", 0, np.array([(0.0, 0.0)]), [0.9]
        )
        self.assertTrue(trace["fallback_to_original"])
        self.assertEqual(calibrated.tolist(), [[0.0, 0.0]])
        self.assertEqual(values, [0.9])

    def test_positive_gain_clears_local_streak(self):
        memory = make_memory()
        obstacle = FakeObstacleMap()
        for index in range(2):
            start(memory, obstacle, step=index * 2)
            memory.finish_attempt(0, "completed", "low", index * 2 + 1, obstacle, [])
        self.assertEqual(memory.trace(0)["memory_entries"][0]["low_gain_streak"], 2)
        start(memory, obstacle, step=5)
        obstacle.add_explored_pixels(50)
        event = memory.finish_attempt(0, "completed", "positive", 6, obstacle, [])
        self.assertFalse(event["low_gain"])
        self.assertEqual(event["streak_after"], 0)

    def test_failure_and_interruption_never_update_streak(self):
        for outcome, reason in (
            ("execution_failure", "no_progress_proxy"),
            ("interrupted", "replanning_switch"),
        ):
            memory = make_memory()
            obstacle = FakeObstacleMap()
            start(memory, obstacle)
            event = memory.finish_attempt(0, outcome, reason, 3, obstacle, [])
            self.assertFalse(event["low_gain"])
            self.assertEqual(event["memory_update"], "none")
            self.assertEqual(memory.trace(0)["memory_entries"], [])

    def test_target_and_stair_gain_lifecycle(self):
        memory = make_memory()
        obstacle = FakeObstacleMap()
        start(memory, obstacle)
        event = memory.observe(
            0, "chair", 0, 1, (3.0, 0.0), obstacle, True, False, False,
            np.array([(0.0, 0.0)]), 0.9,
        )
        self.assertEqual(event["reason"], "target_evidence_found")
        self.assertFalse(event["low_gain"])

        start(memory, obstacle, step=2)
        event = memory.observe(
            0, "chair", 0, 3, (3.0, 0.0), obstacle, False, True, False,
            np.array([(0.0, 0.0)]), 0.9,
        )
        self.assertIsNone(event)
        self.assertTrue(memory.trace(0)["active_attempt"]["stair_gain"])
        event = memory.end_for_stair_mode(0, 4, obstacle, [], "stair_mode")
        self.assertEqual(event["outcome"], "completed")
        self.assertTrue(event["new_stair"])

    def test_disappearance_requires_local_exploration(self):
        memory = make_memory()
        obstacle = FakeObstacleMap()
        start(memory, obstacle)
        event = memory.observe(
            0, "chair", 0, 1, (3.0, 0.0), obstacle, False, False, False,
            np.array([(2.0, 0.0)]), 0.9,
        )
        self.assertIsNone(event)
        memory.register_selection(
            0, "chair", 0, (2.0, 0.0), 2, (3.0, 0.0), obstacle,
            False, False, False, np.array([(2.0, 0.0)]),
        )
        self.assertEqual(memory.trace(0)["memory_entries"], [])

        memory.reset(0)
        obstacle = FakeObstacleMap()
        start(memory, obstacle)
        obstacle.explore_at((0.0, 0.0))
        event = memory.observe(
            0, "chair", 0, 1, (3.0, 0.0), obstacle, False, False, False,
            np.array([(2.0, 0.0)]), 0.9,
        )
        self.assertEqual(event["reason"], "frontier_disappeared_after_local_exploration")
        self.assertEqual(event["outcome"], "completed")

    def test_env_target_and_reset_isolation(self):
        memory = make_memory(num_envs=2)
        obstacles = [FakeObstacleMap(), FakeObstacleMap()]
        start(memory, obstacles[0], env=0, target="chair")
        memory.finish_attempt(0, "completed", "low", 1, obstacles[0], [])
        start(memory, obstacles[1], env=1, target="chair")
        memory.finish_attempt(1, "completed", "low", 1, obstacles[1], [])

        _, _, different_target = memory.calibrate_candidates(
            0, "bed", 0, np.array([(0.0, 0.0)]), [0.9]
        )
        self.assertEqual(different_target["candidates"][0]["status"], "normal")
        memory.reset(0)
        self.assertEqual(memory.trace(0)["memory_entries"], [])
        self.assertEqual(len(memory.trace(1)["memory_entries"]), 1)

    def test_trace_is_json_parseable_and_novelty_is_diagnostic_only(self):
        memory = make_memory()
        obstacle = FakeObstacleMap()
        start(memory, obstacle)
        event = memory.finish_attempt(
            0, "interrupted", "test", 1, obstacle,
            np.array([(0.1, 0.0), (2.0, 0.0), (4.0, 0.0)]),
        )
        self.assertFalse(event["frontier_novelty"]["used_for_low_gain"])
        json.loads(json.dumps(memory.trace(0)))

    def test_ocsm_off_matches_frozen_planner_path(self):
        baseline_source = subprocess.check_output(
            ["git", "-C", str(ASCENT_ROOT), "show", f"{BASELINE_COMMIT}:ascent/llm_planner.py"],
            text=True,
        )
        current_source = (ASCENT_ROOT / "ascent/llm_planner.py").read_text()
        baseline_method = compile_planner_method(baseline_source)
        current_method = compile_planner_method(current_source)
        for case, frontiers in (
            ("single", [(1.0, 2.0)]),
            ("force", [(1.0, 2.0), (3.0, 4.0)]),
            ("llm", [(1.0, 2.0), (3.0, 4.0)]),
            ("sentinel", [(1.0, 2.0), (3.0, 4.0)]),
        ):
            for instrumentation in (False, True):
                expected = planner_signature(
                    baseline_method,
                    make_planner(case, instrumentation=instrumentation),
                    frontiers,
                )
                actual = planner_signature(
                    current_method,
                    make_planner(case, instrumentation=instrumentation),
                    frontiers,
                )
                self.assertEqual(actual, expected, (case, instrumentation))

    def test_ocsm_off_matches_frozen_explore_path(self):
        baseline_source = subprocess.check_output(
            ["git", "-C", str(ASCENT_ROOT), "show", f"{BASELINE_COMMIT}:ascent/ascent_policy.py"],
            text=True,
        )
        current_source = (ASCENT_ROOT / "ascent/ascent_policy.py").read_text()
        baseline_method = compile_policy_explore(baseline_source)
        current_method = compile_policy_explore(current_source)
        for case in ("active", "stop", "stair"):
            expected = explore_signature(baseline_method, make_explore_policy(case))
            actual = explore_signature(current_method, make_explore_policy(case))
            self.assertEqual(actual, expected, case)

    def test_hard_force_is_cancelled_and_sentinel_is_preserved(self):
        memory = make_memory()
        obstacle_map = FakeObstacleMap()
        for index in range(2):
            start(memory, obstacle_map, frontier=(1.0, 2.0), step=index * 2)
            memory.finish_attempt(
                0, "completed", "low", index * 2 + 1, obstacle_map,
                np.array([(1.0, 2.0), (3.0, 4.0)]),
            )
        current_method = compile_planner_method(
            (ASCENT_ROOT / "ascent/llm_planner.py").read_text()
        )
        planner = make_planner("force", ocsm=memory, instrumentation=True)
        obstacle = PlannerObstacle()
        result = current_method(
            planner,
            [{"robot_xy": np.array([0.0, 0.0])}],
            [obstacle], [object()], [object()], [[]], [[]], [[]],
            np.array([(1.0, 2.0), (3.0, 4.0)]),
            cur_floor_index=[0],
        )
        self.assertEqual(np.asarray(result[0]).tolist(), [3.0, 4.0])
        self.assertNotEqual(planner.last_decision_trace[0]["selection_source"], "force_frontier")

        planner = make_planner("sentinel", ocsm=memory, instrumentation=True)
        result = current_method(
            planner,
            [{"robot_xy": np.array([0.0, 0.0])}],
            [PlannerObstacle()], [object()], [object()], [[]], [[]], [[]],
            np.array([(1.0, 2.0), (3.0, 4.0)]),
            cur_floor_index=[0],
        )
        self.assertEqual(result[1], -100)


if __name__ == "__main__":
    unittest.main()
