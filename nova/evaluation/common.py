"""Metric parsing and caller-controlled external-evaluator plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from .types import MetricBundle, MetricValue, SplitGroup


@dataclass(frozen=True)
class ParsedMetrics:
    bundle: MetricBundle
    extra_metrics: Tuple[str, ...]


@dataclass(frozen=True)
class EvaluationPlan:
    dataset: str
    evaluator: str
    required_input_schema: Tuple[str, ...]
    external_requirements: Tuple[str, ...]
    command: Tuple[str, ...]
    execution_enabled: bool = False

def parse_metric_bundle(
    raw_metrics: Mapping[str, object],
    required_metrics: Sequence[str],
    dataset: str,
    split: str,
    split_group: SplitGroup,
    evaluator: str,
    source_evidence: str,
    display_precision: int = 2,
) -> ParsedMetrics:
    missing = tuple(name for name in required_metrics if name not in raw_metrics)
    if missing:
        raise ValueError("missing metrics: {0}".format(",".join(missing)))
    required_set = set(required_metrics)
    extras = tuple(sorted(name for name in raw_metrics if name not in required_set))
    metrics = tuple(
        MetricValue.from_value(
            name, raw_metrics[name], display_precision, split, dataset, evaluator,
            source_evidence,
        )
        for name in required_metrics
    )
    return ParsedMetrics(MetricBundle(dataset, split_group, evaluator, metrics), extras)
