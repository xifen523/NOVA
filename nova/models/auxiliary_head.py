"""Auxiliary box-quality head and its MSE objective."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class AuxiliaryQualityHead(nn.Module):
    """Return one sigmoid quality value per rank-2 hidden-state row."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim != 2 or hidden_state.shape[-1] != self.hidden_dim:
            raise ValueError(
                "hidden_state must have [batch, {0}] shape".format(self.hidden_dim)
            )
        if not torch.isfinite(hidden_state).all().item():
            raise ValueError("hidden_state must contain only finite values")
        return self.network(hidden_state).squeeze(-1)


def auxiliary_quality_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute finite mean-squared error without embedding policy in the head."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if not torch.is_floating_point(prediction) or not torch.is_floating_point(target):
        raise TypeError("prediction and target must be floating-point tensors")
    if not torch.isfinite(prediction).all().item() or not torch.isfinite(target).all().item():
        raise ValueError("prediction and target must be finite")
    return functional.mse_loss(prediction, target)
