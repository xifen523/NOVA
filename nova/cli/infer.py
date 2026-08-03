"""Run association scoring or prepared-sequence tracking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from nova.data.prepared import read_association_jsonl
from nova.config import load_config
from nova.inference import infer_associations, infer_detection_sequence, validate_tracking_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NOVA inference")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("real", "heuristic"), default="real")
    parser.add_argument("--mode", choices=("association", "tracking"), default="association")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--vocab-alignment", choices=("strict", "verified-overlap"), default="strict")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-compatible-input", action="store_true")
    parser.add_argument("--legacy-input", action="store_true")
    parser.add_argument("--output-format", choices=("generic", "v2x", "kitti", "nuscenes"), default="generic")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = load_config(args.config) if args.config else None
        if args.backend == "real" and not args.model and config is None:
            raise ValueError("Model path or --config is required for real inference.")
        if not args.input.is_file():
            raise FileNotFoundError("Prepared detector input was not found.")
        if args.dry_run or args.check_only:
            if args.mode == "association":
                samples = len(read_association_jsonl(args.input, legacy_input=args.legacy_input))
                frames = None
            else:
                if config is None:
                    raise ValueError("--config is required for tracking mode")
                document = json.loads(args.input.read_text(encoding="utf-8"))
                validate_tracking_document(
                    document, config, allow_compatible_input=args.allow_compatible_input
                )
                frames = len(document["frames"])
                samples = sum(len(frame.get("detections", [])) for frame in document["frames"])
            print(json.dumps({
                "status": "PASS", "samples": samples, "frames": frames,
                "execution": "check-only",
            }, sort_keys=True))
            return 0
        if args.mode == "tracking":
            if args.config is None:
                raise ValueError("--config is required for tracking mode")
            summary = infer_detection_sequence(
                args.input, args.output, args.config, backend=args.backend,
                model_path=args.model, checkpoint=args.checkpoint, device=args.device,
                checkpoint_sha256=args.checkpoint_sha256,
                allow_download=args.allow_download, vocab_alignment=args.vocab_alignment,
                allow_compatible_input=args.allow_compatible_input,
                output_format=args.output_format,
            )
        else:
            summary = infer_associations(
                args.input, args.output, config=config, backend=args.backend, model_path=args.model,
                checkpoint=args.checkpoint, device=args.device, allow_download=args.allow_download,
                checkpoint_sha256=args.checkpoint_sha256,
                vocab_alignment=args.vocab_alignment,
                legacy_input=args.legacy_input,
            )
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
