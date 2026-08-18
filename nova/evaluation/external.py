"""Strict caller-configured external evaluator execution and parsing."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from . import kitti, nuscenes, v2x
from .common import ParsedMetrics
from .types import SplitGroup


@dataclass(frozen=True)
class ExternalEvaluationResult:
    parsed: ParsedMetrics
    command: tuple
    evaluator_name: str
    evaluator_version: str
    stdout: str
    stderr: str
    return_code: int


def _parse_key_value(text: str) -> Mapping[str, object]:
    metrics = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            metrics[key] = value
    if not metrics:
        raise ValueError("EVALUATION_PARSE=FAIL: no key=value metrics found")
    return metrics


def _raw_metrics(text: str, parser: str) -> Mapping[str, object]:
    if parser == "key_value":
        return _parse_key_value(text)
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("EVALUATION_PARSE=FAIL: evaluator output is not JSON") from exc
    if not isinstance(document, Mapping):
        raise ValueError("EVALUATION_PARSE=FAIL: evaluator JSON must be an object")
    metrics = document.get("metrics", document)
    if not isinstance(metrics, Mapping):
        raise ValueError("EVALUATION_PARSE=FAIL: metrics object is missing")
    return metrics


def _dataset_parser(dataset: str):
    if dataset == "v2x":
        return v2x.parse_results
    if dataset == "kitti":
        return kitti.parse_results
    if dataset == "nuscenes":
        return nuscenes.parse_results
    raise ValueError("unsupported evaluator dataset")


def run_external_evaluator(
    *,
    input_path: Path,
    command: Sequence[str],
    dataset: str,
    split_group: SplitGroup,
    evaluator_name: str,
    evaluator_version: str,
    output_parser: str = "json",
    metrics_path: Optional[Path] = None,
    ground_truth_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> ExternalEvaluationResult:
    if not command:
        raise ValueError("external evaluator command is required")
    if not evaluator_name.strip() or not evaluator_version.strip():
        raise ValueError("evaluator name and version are required")
    replacements = {
        "{input}": str(input_path),
        "{prediction}": str(input_path),
        "{ground_truth}": str(ground_truth_path) if ground_truth_path is not None else "",
        "{output}": str(output_path) if output_path is not None else "",
    }
    rendered = []
    for part in command:
        value = part
        for placeholder, replacement in replacements.items():
            value = value.replace(placeholder, replacement)
        rendered.append(value)
    if not any("{input}" in part or "{prediction}" in part for part in command):
        rendered.append(str(input_path))
    completed = subprocess.run(rendered, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "external evaluator failed with code {0}: {1}".format(
                completed.returncode, completed.stderr.strip()
            )
        )
    if metrics_path is not None:
        if metrics_path.is_symlink() or not metrics_path.is_file():
            raise ValueError("EVALUATION_PARSE=FAIL: metrics file was not produced")
        result_text = metrics_path.read_text(encoding="utf-8")
    else:
        result_text = completed.stdout
    parsed = _dataset_parser(dataset)(_raw_metrics(result_text, output_parser), split_group)
    return ExternalEvaluationResult(
        parsed=parsed,
        command=tuple(rendered),
        evaluator_name=evaluator_name,
        evaluator_version=evaluator_version,
        stdout=completed.stdout,
        stderr=completed.stderr,
        return_code=completed.returncode,
    )
