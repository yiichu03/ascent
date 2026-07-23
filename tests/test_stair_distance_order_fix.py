import ast
from pathlib import Path
import unittest

import numpy as np


POLICY_PATH = Path(__file__).parents[1] / "ascent" / "ascent_policy.py"


def _load_distance_helper():
    source = POLICY_PATH.read_text()
    module = ast.parse(source)
    helper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_min_stair_map_l1_distance_px"
    )
    isolated = ast.Module(body=[helper], type_ignores=[])
    namespace = {"np": np}
    exec(compile(isolated, str(POLICY_PATH), "exec"), namespace)
    return namespace["_min_stair_map_l1_distance_px"]


class StairDistanceOrderFixTest(unittest.TestCase):
    def test_coordinate_order_fix_compares_xy_with_xy(self):
        helper = _load_distance_helper()
        stair_map = np.zeros((6, 10), dtype=bool)
        stair_map[1, 8] = True
        robot_px_xy = np.array([7, 1])

        legacy_distance = helper(
            stair_map,
            robot_px_xy,
            correct_coordinate_order=False,
        )
        corrected_distance = helper(
            stair_map,
            robot_px_xy,
            correct_coordinate_order=True,
        )

        self.assertEqual(legacy_distance, 13)
        self.assertEqual(corrected_distance, 1)

    def test_distance_helper_rejects_empty_stair_map(self):
        helper = _load_distance_helper()

        with self.assertRaisesRegex(ValueError, "at least one stair pixel"):
            helper(
                np.zeros((4, 4), dtype=bool),
                np.array([1, 1]),
                correct_coordinate_order=True,
            )


if __name__ == "__main__":
    unittest.main()
