"""Frozen nine-dimensional geometry construction."""

from __future__ import annotations

from typing import Sequence

import torch

from nova.data.types import AssociationSample, Detection3D


POSITION_NORMALIZER_METERS = 10.0
GEOMETRY_DIMENSION = 9


def build_geometry_9d(detection: Detection3D) -> torch.Tensor:
    if not isinstance(detection, Detection3D):
        raise TypeError("detection must be Detection3D")
    x, y, z = detection.position_m
    length, width, height = detection.size_lwh_m
    values = (
        x / POSITION_NORMALIZER_METERS,
        y / POSITION_NORMALIZER_METERS,
        z / POSITION_NORMALIZER_METERS,
        length,
        width,
        height,
        length * width * height,
        detection.yaw_rad,
        detection.score,
    )
    geometry = torch.tensor(values, dtype=torch.float32)
    if geometry.shape != (GEOMETRY_DIMENSION,):
        raise RuntimeError("geometry construction violated the frozen dimension")
    if not torch.isfinite(geometry).all().item():
        raise ValueError("geometry must contain only finite values")
    return geometry


def build_geometry_batch(samples: Sequence[AssociationSample]) -> torch.Tensor:
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("samples must be a non-empty list or tuple")
    if any(not isinstance(sample, AssociationSample) for sample in samples):
        raise TypeError("every sample must be AssociationSample")
    rows = []
    for sample in samples:
        if sample.context is not None:
            rows.extend(sample.context.geometry_features)
        else:
            rows.append(tuple(build_geometry_9d(sample.candidate.detection).tolist()))
    return torch.tensor(rows, dtype=torch.float32)
