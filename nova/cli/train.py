"""Train NOVA on prepared association JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from nova.config import load_config
from nova.data.prepared import read_association_jsonl
from nova.training import train_prepared_associations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NOVA association components")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=("real", "tiny"), default="real")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--vocab-alignment", choices=("strict", "verified-overlap"), default="strict")
    parser.add_argument("--legacy-input", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        config = load_config(args.config, for_training=True)
        model = args.model or config.model.base_model
        if args.dry_run:
            rows = read_association_jsonl(
                args.input,
                require_target=True,
                require_auxiliary=(
                    config.model.auxiliary_head.enabled
                    and config.training.auxiliary_loss_weight > 0
                ),
                legacy_input=args.legacy_input,
            )
            if args.backend == "real" and not model:
                raise ValueError("Model path is required for real training.")
            print(json.dumps({"status": "PASS", "samples": len(rows), "execution": "dry-run"}, sort_keys=True))
            return 0
        summary = train_prepared_associations(
            args.input, args.output_dir, config=config, backend=args.backend, model_path=model,
            checkpoint=args.checkpoint, device=args.device, steps=args.steps,
            checkpoint_sha256=args.checkpoint_sha256,
            allow_download=args.allow_download, vocab_alignment=args.vocab_alignment,
            legacy_input=args.legacy_input,
        )
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
