"""Narrow deterministic Hungarian assignment over an explicit validity mask."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import AssignmentResult


def assign_detections(
    costs: Sequence[Sequence[float]], valid_mask: Sequence[Sequence[bool]]
) -> AssignmentResult:
    cost_array = np.asarray(costs, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if cost_array.ndim != 2 or mask.ndim != 2 or cost_array.shape != mask.shape:
        raise ValueError("costs and valid_mask must be same-shaped rank-two matrices")
    rows, columns = cost_array.shape
    if rows == 0 or columns == 0:
        return AssignmentResult((), tuple(range(rows)), tuple(range(columns)))
    if np.any(mask & ~np.isfinite(cost_array)):
        raise ValueError("valid assignment costs must be finite")
    if np.any(mask & (cost_array < 0)):
        raise ValueError("valid assignment costs must be non-negative")
    if not mask.any():
        return AssignmentResult((), tuple(range(rows)), tuple(range(columns)))
    finite_valid = cost_array[mask]
    penalty = (float(np.max(finite_valid)) + 1.0) * float(rows + columns + 1)
    solver_costs = np.where(mask, cost_array, penalty)
    row_indices, column_indices = linear_sum_assignment(solver_costs)
    matches = tuple(
        (int(row), int(column))
        for row, column in zip(row_indices.tolist(), column_indices.tolist())
        if mask[row, column]
    )
    matched_rows = {row for row, _ in matches}
    matched_columns = {column for _, column in matches}
    return AssignmentResult(
        matches=matches,
        unmatched_tracks=tuple(index for index in range(rows) if index not in matched_rows),
        unmatched_detections=tuple(index for index in range(columns) if index not in matched_columns),
    )
