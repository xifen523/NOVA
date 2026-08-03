"""Explicit adapter registry without dynamic imports or executable configuration."""

from __future__ import annotations

from typing import Dict, Tuple, Type

from .base import DatasetAdapter
from .class_split import ClassSplit


_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}
_BUILTINS_REGISTERED = False


def _normalize_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("adapter name must be a non-empty string")
    return "_".join(name.strip().lower().split())


def register_dataset_adapter(name: str, adapter_type: Type[DatasetAdapter]) -> None:
    normalized = _normalize_name(name)
    if not isinstance(adapter_type, type) or not issubclass(adapter_type, DatasetAdapter):
        raise TypeError("adapter_type must be a DatasetAdapter subclass")
    if normalized in _REGISTRY:
        raise ValueError("dataset adapter is already registered: {0}".format(normalized))
    _REGISTRY[normalized] = adapter_type


def _ensure_builtins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .kitti import KITTIAdapter
    from .nuscenes import NuScenesAdapter
    from .v2x import V2XAdapter

    register_dataset_adapter("kitti", KITTIAdapter)
    register_dataset_adapter("nuscenes", NuScenesAdapter)
    register_dataset_adapter("v2x", V2XAdapter)
    _BUILTINS_REGISTERED = True


def list_dataset_adapters() -> Tuple[str, ...]:
    _ensure_builtins()
    return tuple(sorted(_REGISTRY))


def build_dataset_adapter(
    name: str, class_split: ClassSplit, real_mode: bool = False
) -> DatasetAdapter:
    _ensure_builtins()
    normalized = _normalize_name(name)
    if normalized not in _REGISTRY:
        raise ValueError("unknown dataset adapter: {0}".format(normalized))
    return _REGISTRY[normalized](class_split=class_split, real_mode=real_mode)
