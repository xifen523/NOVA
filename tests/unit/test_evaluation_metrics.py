import unittest
from decimal import Decimal

from nova.evaluation.common import parse_metric_bundle
from nova.evaluation.types import MetricValue, SplitGroup


class EvaluationMetricTests(unittest.TestCase):
    def test_raw_precision_and_two_decimal_display_are_separate(self):
        metric = MetricValue.from_value(
            "AMOTA", "48.87365677481718", 2, "validation", "nuScenes",
            "synthetic evaluator", "synthetic evidence",
        )
        self.assertEqual(metric.raw_value, Decimal("48.87365677481718"))
        self.assertEqual(metric.display_value, "48.87")

    def test_display_is_computed_directly_without_double_rounding(self):
        metric = MetricValue.from_value(
            "metric", "2.3449", 2, "synthetic", "synthetic", "synthetic", "synthetic"
        )
        self.assertEqual(metric.display_value, "2.34")

    def test_missing_is_rejected_and_extra_is_reported(self):
        with self.assertRaisesRegex(ValueError, "missing metrics"):
            parse_metric_bundle(
                {"AMOTA": "1.0"}, ("AMOTA", "MOTA"), "nuScenes", "val",
                SplitGroup.BASE, "external", "synthetic",
            )
        parsed = parse_metric_bundle(
            {"AMOTA": "1.0", "MOTA": "2.0", "EXTRA": "3.0"},
            ("AMOTA", "MOTA"), "nuScenes", "val", SplitGroup.NOVEL,
            "external", "synthetic",
        )
        self.assertEqual(parsed.extra_metrics, ("EXTRA",))
        self.assertEqual(parsed.bundle.split_group, SplitGroup.NOVEL)


if __name__ == "__main__":
    unittest.main()
