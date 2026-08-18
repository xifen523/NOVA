import unittest

import numpy as np

from nova.tracking.assignment import assign_detections


class AssignmentTests(unittest.TestCase):
    def test_square_and_tie_are_deterministic(self):
        result = assign_detections([[1.0, 2.0], [2.0, 1.0]], [[True, True], [True, True]])
        self.assertEqual(result.matches, ((0, 0), (1, 1)))
        tie_first = assign_detections([[1.0, 1.0], [1.0, 1.0]], [[True, True], [True, True]])
        tie_second = assign_detections([[1.0, 1.0], [1.0, 1.0]], [[True, True], [True, True]])
        self.assertEqual(tie_first, tie_second)
        self.assertEqual(tie_first.matches, ((0, 0), (1, 1)))

    def test_rectangular_and_empty_matrices(self):
        result = assign_detections([[1.0, 3.0, 2.0]], [[True, True, True]])
        self.assertEqual(result.matches, ((0, 0),))
        self.assertEqual(result.unmatched_detections, (1, 2))
        empty = assign_detections(np.empty((0, 2)), np.empty((0, 2), dtype=bool))
        self.assertEqual(empty.unmatched_detections, (0, 1))

    def test_masks_all_invalid_and_nonfinite_costs(self):
        masked = assign_detections([[1.0, 0.0]], [[False, True]])
        self.assertEqual(masked.matches, ((0, 1),))
        invalid = assign_detections([[1.0]], [[False]])
        self.assertEqual(invalid.matches, ())
        permitted_masked_nonfinite = assign_detections([[float("inf")]], [[False]])
        self.assertEqual(permitted_masked_nonfinite.matches, ())
        with self.assertRaises(ValueError):
            assign_detections([[float("nan")]], [[True]])


if __name__ == "__main__":
    unittest.main()
