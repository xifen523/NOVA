"""Common evaluation result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MetricGroup:
    name: str
    raw_metrics: Dict[str, float]


@dataclass(frozen=True)
class MetricBundle:
    dataset: str
    split: str
    evaluator_name: str
    evaluator_version: Optional[str]
    groups: List[MetricGroup] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
