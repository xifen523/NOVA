import inspect
import unittest

from nova.evaluation import kitti, nuscenes, v2x
from nova.evaluation.types import SplitGroup


class EvaluationWrapperTests(unittest.TestCase):
    def test_three_plans_are_disabled_and_dataset_specific(self):
        plans = (v2x.build_evaluation_plan(), kitti.build_evaluation_plan(), nuscenes.build_evaluation_plan())
        self.assertEqual({plan.dataset for plan in plans}, {"V2X-Seq-SPD", "KITTI", "nuScenes"})
        for plan in plans:
            self.assertFalse(plan.execution_enabled)
            self.assertIn("UNRESOLVED_EVALUATOR", plan.command)
            self.assertTrue(plan.required_input_schema)
        parsed = nuscenes.parse_results(
            {"AMOTA": "1.001", "AMOTP": "2", "MOTA": "3", "MOTP": "4", "MT": "5"},
            SplitGroup.BASE,
        )
        self.assertEqual(parsed.bundle.by_name("AMOTA").raw_value.as_tuple().exponent, -3)

    def test_wrappers_have_no_subprocess_or_prediction_reader(self):
        source = "\n".join(inspect.getsource(module) for module in (v2x, kitti, nuscenes))
        self.assertNotIn("subprocess", source)
        self.assertNotIn("read_text", source)
        self.assertNotIn("open(", source)


if __name__ == "__main__":
    unittest.main()
