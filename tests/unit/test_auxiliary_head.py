from __future__ import annotations

import unittest

import torch

from nova.models.auxiliary_head import AuxiliaryQualityHead, auxiliary_quality_loss


class AuxiliaryQualityHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.head = AuxiliaryQualityHead(hidden_dim=16, bottleneck_dim=8)

    def test_output_shape_is_batch(self):
        output = self.head(torch.randn(5, 16))
        self.assertEqual(tuple(output.shape), (5,))

    def test_output_is_probability(self):
        output = self.head(torch.randn(5, 16))
        self.assertTrue(((output >= 0) & (output <= 1)).all().item())

    def test_mse_is_finite(self):
        prediction = self.head(torch.randn(5, 16))
        loss = auxiliary_quality_loss(prediction, torch.linspace(0, 1, 5))
        self.assertTrue(torch.isfinite(loss).item())

    def test_cpu_backward(self):
        hidden = torch.randn(5, 16, requires_grad=True)
        prediction = self.head(hidden)
        loss = auxiliary_quality_loss(prediction, torch.linspace(0, 1, 5))
        loss.backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(all(parameter.grad is not None for parameter in self.head.parameters()))


if __name__ == "__main__":
    unittest.main()

