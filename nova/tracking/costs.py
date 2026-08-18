"""Experiment-compatible association cost without gate side effects."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


DEFAULT_DISTANCE_WEIGHT = 0.1


def association_cost(
    p_yes: float, distance: float, distance_weight: float = DEFAULT_DISTANCE_WEIGHT
) -> float:
    for name, value in (
        ("p_yes", p_yes), ("distance", distance), ("distance_weight", distance_weight)
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("{0} must be numeric".format(name))
        if not math.isfinite(float(value)):
            raise ValueError("{0} must be finite".format(name))
    if not 0.0 <= float(p_yes) <= 1.0:
        raise ValueError("p_yes must be in [0,1]")
    if distance < 0.0 or distance_weight < 0.0:
        raise ValueError("distance and distance_weight must be non-negative")
    return (1.0 - float(p_yes)) + float(distance_weight) * float(distance)


def association_cost_matrix(
    p_yes: Sequence[Sequence[float]],
    distances: Sequence[Sequence[float]],
    distance_weight: float = DEFAULT_DISTANCE_WEIGHT,
) -> np.ndarray:
    probabilities = np.asarray(p_yes, dtype=np.float64)
    distance_array = np.asarray(distances, dtype=np.float64)
    if probabilities.ndim != 2 or distance_array.ndim != 2:
        raise ValueError("p_yes and distances must be rank-two matrices")
    if probabilities.shape != distance_array.shape:
        raise ValueError("p_yes and distances must have identical shapes")
    if not np.isfinite(probabilities).all() or not np.isfinite(distance_array).all():
        raise ValueError("cost inputs must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("every p_yes must be in [0,1]")
    if np.any(distance_array < 0.0):
        raise ValueError("distances must be non-negative")
    if not isinstance(distance_weight, (int, float)) or not math.isfinite(float(distance_weight)) or distance_weight < 0:
        raise ValueError("distance_weight must be finite and non-negative")
    return (1.0 - probabilities) + float(distance_weight) * distance_array
