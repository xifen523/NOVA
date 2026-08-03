from __future__ import annotations

import unittest

import torch

from nova.models.geometry_encoder import GeometryEncoder


class GeometryEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.encoder = GeometryEncoder(9, 32, 16)

    def test_rank_three_shape(self):
        output = self.encoder(torch.randn(2, 3, 9))
        self.assertEqual(tuple(output.shape), (2, 3, 16))

    def test_rank_two_shape(self):
        output = self.encoder(torch.randn(4, 9))
        self.assertEqual(tuple(output.shape), (4, 16))

    def test_wrong_last_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            self.encoder(torch.randn(2, 8))

    def test_non_finite_input_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                geometry = torch.zeros(2, 9)
                geometry[0, 0] = value
                with self.assertRaises(ValueError):
                    self.encoder(geometry)

    def test_cpu_backward_populates_parameter_gradients(self):
        geometry = torch.randn(3, 9, requires_grad=True)
        self.encoder(geometry).square().mean().backward()
        self.assertIsNotNone(geometry.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in self.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()

