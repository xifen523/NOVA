import unittest
from pathlib import Path

from nova.evaluation.conflicts import load_known_conflicts


class KnownConflictTests(unittest.TestCase):
    def test_open3dtrack_conflicts_never_pass_release_acceptance(self):
        conflicts = load_known_conflicts(Path("configs/paper_targets/known_conflicts.yaml"))
        self.assertEqual(len(conflicts), 2)
        for conflict in conflicts:
            self.assertEqual(conflict.status, "KNOWN_CONFLICT")
            self.assertFalse(conflict.release_acceptance_target)
            self.assertFalse(conflict.passes_release_acceptance())
            self.assertTrue(conflict.material_conflict)
            self.assertEqual(
                set(conflict.paper_metrics), set(conflict.verified_local_metrics)
            )
            self.assertEqual(
                set(conflict.paper_metrics), set(conflict.absolute_differences)
            )


if __name__ == "__main__":
    unittest.main()
