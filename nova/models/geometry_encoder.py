"""Nine-dimensional geometry encoder used by NOVA."""

from __future__ import annotations

import torch
from torch import nn


class GeometryEncoder(nn.Module):
    """Map geometry from 9 features through 1024 to LM hidden width."""

    def __init__(
        self,
        input_dim: int,
        geometry_hidden_dim: int,
        language_hidden_dim: int,
    ) -> None:
        super().__init__()
        if input_dim != 9:
            raise ValueError("GeometryEncoder input_dim must equal 9")
        if geometry_hidden_dim <= 0 or language_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        self.input_dim = input_dim
        self.geometry_hidden_dim = geometry_hidden_dim
        self.language_hidden_dim = language_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, geometry_hidden_dim),
            nn.LayerNorm(geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, geometry_hidden_dim),
            nn.GELU(),
            nn.Linear(geometry_hidden_dim, language_hidden_dim),
        )

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if not isinstance(geometry, torch.Tensor):
            raise TypeError("geometry must be a torch.Tensor")
        if geometry.ndim not in (2, 3):
            raise ValueError("geometry must have rank 2 or 3")
        if geometry.shape[-1] != self.input_dim:
            raise ValueError(
                "geometry last dimension must be {0}".format(self.input_dim)
            )
        if not torch.is_floating_point(geometry):
            raise TypeError("geometry must use a floating-point dtype")
        if not torch.isfinite(geometry).all().item():
            raise ValueError("geometry must contain only finite values")
        return self.mlp(geometry)
