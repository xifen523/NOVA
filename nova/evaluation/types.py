"""Precision-preserving metric contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Tuple


class SplitGroup(str, Enum):
    BASE = "Base"
    NOVEL = "Novel"
    ALL = "All"


@dataclass(frozen=True)
class MetricValue:
    metric_name: str
    raw_value: Decimal
    display_precision: int
    split: str
    dataset: str
    evaluator: str
    source_evidence: str

    def __post_init__(self) -> None:
        for name, value in (
            ("metric_name", self.metric_name), ("split", self.split),
            ("dataset", self.dataset), ("evaluator", self.evaluator),
            ("source_evidence", self.source_evidence),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("{0} must be a non-empty string".format(name))
        if not isinstance(self.raw_value, Decimal) or not self.raw_value.is_finite():
            raise ValueError("raw_value must be a finite Decimal")
        if isinstance(self.display_precision, bool) or not isinstance(self.display_precision, int) or self.display_precision < 0:
            raise ValueError("display_precision must be a non-negative integer")

    @property
    def display_value(self) -> str:
        quantum = Decimal(1).scaleb(-self.display_precision)
        rounded = self.raw_value.quantize(quantum, rounding=ROUND_HALF_UP)
        return format(rounded, ".{0}f".format(self.display_precision))

    @classmethod
    def from_value(
        cls, metric_name: str, raw_value: object, display_precision: int,
        split: str, dataset: str, evaluator: str, source_evidence: str,
    ) -> "MetricValue":
        try:
            decimal_value = Decimal(str(raw_value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("metric value is not decimal-compatible") from exc
        return cls(
            metric_name, decimal_value, display_precision, split, dataset,
            evaluator, source_evidence,
        )


@dataclass(frozen=True)
class MetricBundle:
    dataset: str
    split_group: SplitGroup
    evaluator: str
    metrics: Tuple[MetricValue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.split_group, SplitGroup):
            raise TypeError("split_group must be SplitGroup")
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        names = [metric.metric_name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("metric names must be unique")
        if any(
            metric.dataset != self.dataset or metric.evaluator != self.evaluator
            for metric in self.metrics
        ):
            raise ValueError("bundle context does not match contained metrics")

    def by_name(self, name: str) -> MetricValue:
        for metric in self.metrics:
            if metric.metric_name == name:
                return metric
        raise KeyError("missing metric: {0}".format(name))
