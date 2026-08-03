"""Run and strictly parse a caller-supplied benchmark evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from nova.config import load_config
from nova.evaluation.external import run_external_evaluator
from nova.evaluation.types import SplitGroup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an external benchmark evaluator")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", "--prediction", dest="prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("v2x", "kitti", "nuscenes"))
    parser.add_argument("--split-group", choices=("Base", "Novel", "All"), default="All")
    parser.add_argument("--evaluator-name")
    parser.add_argument("--evaluator-version")
    parser.add_argument("--evaluator-command", nargs="+")
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--output-parser", choices=("json", "key_value"), default="json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _bundle_document(result) -> dict:
    bundle = result.parsed.bundle
    return {
        "schema": "nova.metric-bundle.v1",
        "dataset": bundle.dataset,
        "split_group": bundle.split_group.value,
        "evaluator": result.evaluator_name,
        "evaluator_version": result.evaluator_version,
        "command": list(result.command),
        "return_code": result.return_code,
        "metrics": {
            metric.metric_name: {
                "raw": str(metric.raw_value),
                "display": metric.display_value,
            }
            for metric in bundle.metrics
        },
        "extra_metrics": list(result.parsed.extra_metrics),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if not args.prediction.exists() or args.prediction.is_symlink():
            raise FileNotFoundError("prediction input was not found")
        if args.ground_truth is not None and (
            not args.ground_truth.exists() or args.ground_truth.is_symlink()
        ):
            raise FileNotFoundError("ground-truth input was not found")
        config = load_config(args.config) if args.config else None
        dataset = args.dataset or (config.dataset.name if config else None)
        evaluator_name = args.evaluator_name or (
            config.evaluation.evaluator_name if config else None
        )
        evaluator_version = args.evaluator_version or (
            config.evaluation.evaluator_version if config else None
        )
        evaluator_command = args.evaluator_command or (
            config.evaluation.evaluator_command if config else None
        )
        output_parser = (
            args.output_parser
            if args.output_parser != "json" or config is None
            else config.evaluation.output_parser
        )
        if not dataset:
            raise ValueError("--dataset or --config is required")
        evaluator_ready = bool(evaluator_name and evaluator_version and evaluator_command)
        if args.dry_run:
            document = {
                "status": "PASS" if evaluator_ready else "EXTERNAL_EVALUATOR_REQUIRED",
                "execution": "dry-run",
                "dataset": dataset,
                "evaluator": evaluator_name,
                "evaluator_version": evaluator_version,
                "command": evaluator_command or [],
                "prediction": str(args.prediction),
                "ground_truth": str(args.ground_truth) if args.ground_truth else None,
            }
        else:
            if not evaluator_ready:
                raise RuntimeError("EXTERNAL_EVALUATOR_REQUIRED")
            result = run_external_evaluator(
                input_path=args.prediction,
                command=evaluator_command,
                dataset=dataset,
                split_group=SplitGroup(args.split_group),
                evaluator_name=evaluator_name,
                evaluator_version=evaluator_version,
                output_parser=output_parser,
                metrics_path=args.metrics_file,
                ground_truth_path=args.ground_truth,
                output_path=args.metrics_file or args.output,
            )
            document = _bundle_document(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(document, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
