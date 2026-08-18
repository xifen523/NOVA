import unittest
from pathlib import Path

from nova.evaluation.paper_targets import PaperTarget, load_paper_targets
from nova.evaluation.types import SplitGroup


TARGET_PATH = Path("configs/paper_targets/main_results.yaml")


class PaperTargetTests(unittest.TestCase):
    def test_targets_have_provenance_and_do_not_mix_base_novel(self):
        targets = load_paper_targets(TARGET_PATH)
        self.assertEqual(len(targets), 6)
        for target in targets:
            self.assertIsInstance(target, PaperTarget)
            self.assertEqual(target.metrics.split_group, target.split_group)
            self.assertTrue(target.source_url.startswith("https://arxiv.org/"))
            self.assertEqual(target.verification_status, "paper-reported")
        by_dataset = {}
        for target in targets:
            by_dataset.setdefault(target.dataset, set()).add(target.split_group)
        for groups in by_dataset.values():
            self.assertEqual(groups, {SplitGroup.BASE, SplitGroup.NOVEL})

    def test_nuscenes_score_a_ten_display_values_match_registry_import(self):
        targets = [
            target for target in load_paper_targets(TARGET_PATH)
            if target.dataset == "nuScenes" and target.method == "NOVA"
        ]
        self.assertEqual({target.verification_status for target in targets}, {"paper-reported"})
        self.assertEqual({target.exact_historical_cli for target in targets}, {"UNKNOWN"})
        values = tuple(
            metric.display_value for target in targets for metric in target.metrics.metrics
        )
        self.assertEqual(
            values,
            ("48.87", "100.09", "43.90", "29.94", "50.51",
             "22.41", "138.98", "19.60", "55.04", "14.93"),
        )


if __name__ == "__main__":
    unittest.main()
