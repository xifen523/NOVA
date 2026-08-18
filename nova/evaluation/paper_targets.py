"""Strict loader for public paper-reported result declarations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple

import yaml

from .types import MetricBundle, MetricValue, SplitGroup


@dataclass(frozen=True)
class PaperTarget:
    paper_experiment_id: str
    paper_table: str
    paper_row: str
    dataset: str
    detector: str
    method: str
    split_group: SplitGroup
    metrics: MetricBundle
    paper_version: str
    source_url: str
    metric_display_precision: int
    verification_status: str
    exact_historical_cli: str

    def __post_init__(self) -> None:
        if not self.paper_version or not self.source_url.startswith("https://"):
            raise ValueError("paper target requires a public paper version and URL")
        if self.metric_display_precision < 0 or self.verification_status != "paper-reported":
            raise ValueError("paper target display or verification metadata is invalid")
        if self.metrics.split_group is not self.split_group:
            raise ValueError("paper target split and metric split differ")


def load_paper_targets(path: Path) -> Tuple[PaperTarget, ...]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or set(document) != {"targets"}:
        raise ValueError("paper target root must contain only targets")
    targets = []
    for entry in document["targets"]:
        required = {
            "paper_experiment_id", "paper_table", "paper_row", "dataset", "detector",
            "method", "split_group", "metrics", "paper_version", "source_url",
            "metric_display_precision", "verification_status", "exact_historical_cli",
        }
        if not isinstance(entry, Mapping) or set(entry) != required:
            raise ValueError("paper target has missing or extra fields")
        group = SplitGroup(entry["split_group"])
        metric_values = tuple(
            MetricValue.from_value(
                name, value, entry["metric_display_precision"], group.value,
                entry["dataset"], "paper-reported", entry["source_url"],
            )
            for name, value in entry["metrics"].items()
        )
        bundle = MetricBundle(entry["dataset"], group, "paper-reported", metric_values)
        targets.append(PaperTarget(
            entry["paper_experiment_id"], entry["paper_table"], entry["paper_row"],
            entry["dataset"], entry["detector"], entry["method"], group, bundle,
            entry["paper_version"], entry["source_url"],
            entry["metric_display_precision"], entry["verification_status"],
            entry["exact_historical_cli"],
        ))
    return tuple(targets)
