"""Reproducibility helpers for CPU and single-GPU execution."""

from __future__ import annotations

import random

import torch


def seed_cpu_randomness(seed: int) -> None:
    """Seed Python and PyTorch CPU generators without touching CUDA state."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    torch.manual_seed(seed)


def seed_everything(seed: int) -> None:
    """Seed Python and Torch; seed CUDA generators only when CUDA is available."""

    seed_cpu_randomness(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
