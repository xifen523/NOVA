"""Strict helpers for the untyped mapping boundary."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Set, Tuple


def reject_unknown_fields(record: Mapping[str, object], allowed: Set[str]) -> None:
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError("unknown record fields: {0}".format(", ".join(unknown)))


def require_string(record: Mapping[str, object], key: str) -> str:
    if key not in record:
        raise ValueError("required field {0} is missing".format(key))
    value = record[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError("{0} must be a non-empty string".format(key))
    return value


def optional_string(record: Mapping[str, object], key: str) -> Optional[str]:
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TypeError("{0} must be null or a non-empty string".format(key))
    return value


def require_int(record: Mapping[str, object], key: str) -> int:
    if key not in record:
        raise ValueError("required field {0} is missing".format(key))
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{0} must be an integer".format(key))
    return value


def require_float(record: Mapping[str, object], key: str) -> float:
    if key not in record:
        raise ValueError("required field {0} is missing".format(key))
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{0} must be numeric".format(key))
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("{0} must be finite".format(key))
    return result


def require_vector3(
    record: Mapping[str, object], key: str
) -> Tuple[float, float, float]:
    if key not in record:
        raise ValueError("required field {0} is missing".format(key))
    value = record[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise TypeError("{0} must contain exactly three numeric values".format(key))
    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise TypeError("{0} must contain only numeric values".format(key))
        numeric = float(component)
        if not math.isfinite(numeric):
            raise ValueError("{0} must contain only finite values".format(key))
        result.append(numeric)
    return result[0], result[1], result[2]
