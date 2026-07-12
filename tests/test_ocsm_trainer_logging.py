import ast
from pathlib import Path
import unittest


ASCENT_ROOT = Path(__file__).resolve().parents[1]


def load_extract_scalars(fake_habitat_extractor):
    source = (ASCENT_ROOT / "ascent/ascent_trainer.py").read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "extract_scalars_from_info"
    )
    function.decorator_list = []
    function.returns = None
    for arg in function.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"extract_scalars_from_info_habitat": fake_habitat_extractor}
    exec(compile(module, "<extract_scalars_from_info>", "exec"), namespace)
    return namespace["extract_scalars_from_info"]


class OCSMTrainerLoggingTest(unittest.TestCase):
    def test_decision_trace_is_not_sent_to_episode_scalar_extraction(self):
        captured = {}

        def fake_habitat_extractor(info):
            captured.update(info)
            return {"success": float(info["success"])}

        extract_scalars = load_extract_scalars(fake_habitat_extractor)
        result = extract_scalars(
            {
                "success": 1,
                "num_steps": 43,
                "decision_trace": {
                    "ocsm": {
                        "active_attempt": None,
                        "memory_entries": [{"memory_distance_m": None}],
                    }
                },
                "render_payload": [1, 2, 3],
            }
        )
        self.assertEqual(result, {"success": 1.0})
        self.assertEqual(captured, {"success": 1, "num_steps": 43})


if __name__ == "__main__":
    unittest.main()
