"""nuScenes evaluator schema and disabled command plan."""

from typing import Mapping

from .common import EvaluationPlan, ParsedMetrics, parse_metric_bundle
from .types import SplitGroup


REQUIRED_METRICS = ("AMOTA", "AMOTP", "MOTA", "MOTP", "MT")


def build_evaluation_plan() -> EvaluationPlan:
    return EvaluationPlan(
        dataset="nuScenes",
        evaluator="official nuScenes tracking evaluator (external)",
        required_input_schema=("sample_token", "tracking_id", "tracking_name", "translation", "size", "rotation", "tracking_score"),
        external_requirements=("nuScenes devkit", "authoritative dataset root", "prediction artifact"),
        command=("UNRESOLVED_EVALUATOR", "--result_path", "UNRESOLVED", "--output_dir", "UNRESOLVED"),
    )


def parse_results(raw_metrics: Mapping[str, object], group: SplitGroup) -> ParsedMetrics:
    return parse_metric_bundle(
        raw_metrics, REQUIRED_METRICS, "nuScenes", "validation", group,
        "official nuScenes tracking evaluator (external)",
        "caller-provided parsed result",
    )
