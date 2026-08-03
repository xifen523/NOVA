"""V2X evaluator schema and disabled command plan."""

from typing import Mapping

from .common import EvaluationPlan, ParsedMetrics, parse_metric_bundle
from .types import SplitGroup


REQUIRED_METRICS = ("sAMOTA", "AMOTA", "AMOTP", "MOTA", "MOTP", "MT")


def build_evaluation_plan() -> EvaluationPlan:
    return EvaluationPlan(
        dataset="V2X-Seq-SPD",
        evaluator="external V2X tracking evaluator",
        required_input_schema=("sequence_id", "frame_id", "track_id", "category", "box", "score"),
        external_requirements=("authoritative dataset root", "official evaluator command", "prediction artifact"),
        command=("UNRESOLVED_EVALUATOR", "--predictions", "UNRESOLVED", "--output", "UNRESOLVED"),
    )


def parse_results(raw_metrics: Mapping[str, object], group: SplitGroup) -> ParsedMetrics:
    return parse_metric_bundle(
        raw_metrics, REQUIRED_METRICS, "V2X-Seq-SPD", "evaluation", group,
        "external V2X tracking evaluator", "caller-provided parsed result",
    )
