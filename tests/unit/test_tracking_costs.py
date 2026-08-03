import unittest

import numpy as np

from nova.tracking.costs import association_cost, association_cost_matrix


class TrackingCostTests(unittest.TestCase):
    def test_language_extremes_and_default_distance_weight(self):
        self.assertEqual(association_cost(1.0, 0.0), 0.0)
        self.assertEqual(association_cost(0.0, 0.0), 1.0)
        self.assertAlmostEqual(association_cost(0.8, 2.0), 0.4)

    def test_invalid_probability_distance_and_nonfinite_are_rejected(self):
        for probability in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                association_cost(probability, 1.0)
        for distance in (-1.0, float("nan"), float("inf")):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                association_cost(0.5, distance)

    def test_matrix_shape_and_values(self):
        costs = association_cost_matrix([[1.0, 0.0]], [[2.0, 3.0]], 0.1)
        self.assertTrue(np.allclose(costs, [[0.2, 1.3]]))
        with self.assertRaises(ValueError):
            association_cost_matrix([[0.5]], [[1.0, 2.0]])


if __name__ == "__main__":
    unittest.main()
