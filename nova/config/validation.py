"""Reusable strict validation helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Type


class ConfigValidationError(ValueError):
    """Raised when configuration violates the frozen schema or safety policy."""


class ReproducibilityWarning(UserWarning):
    """Warns that a valid synthetic configuration is not run-reproducible."""


def reject_unknown(mapping: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigValidationError(
            "unknown field(s) at {0}: {1}".format(path, ", ".join(unknown))
        )


def require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError("{0} must be a mapping".format(path))
    return value


def require_type(value: Any, expected: Type[Any], path: str) -> None:
    if expected is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected is float:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise ConfigValidationError(
            "{0} must be {1}, got {2}".format(path, expected.__name__, type(value).__name__)
        )


def require_positive_int(value: Any, path: str) -> None:
    require_type(value, int, path)
    if value <= 0:
        raise ConfigValidationError("{0} must be positive".format(path))


def require_probability(value: Any, path: str, upper_inclusive: bool = True) -> None:
    require_type(value, float, path)
    upper_ok = value <= 1 if upper_inclusive else value < 1
    if value < 0 or not upper_ok:
        interval = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ConfigValidationError("{0} must be in {1}".format(path, interval))

