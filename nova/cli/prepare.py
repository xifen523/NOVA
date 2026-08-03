"""Convert adapter-compatible JSON into NOVA association JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from nova.config import load_config
from nova.data import build_dataset_adapter
from nova.data.prepared import prepare_associations, write_jsonl
from nova.inference import validate_tracking_document
from nova.preprocessing.association import ClassPromptPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NOVA association records")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--legacy-input", action="store_true")
    parser.add_argument("--migration-report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if not args.input.is_file():
            raise FileNotFoundError("Prepared detector input was not found: {0}".format(args.input))
        document = json.loads(args.input.read_text(encoding="utf-8"))
        config = None
        if args.config:
            config = load_config(args.config)
            if config.dataset.name != document.get("dataset"):
                raise ValueError("config dataset does not match input dataset")
        if document.get("schema") == "nova.detection-sequence.v1":
            if config is None:
                raise ValueError("--config is required to prepare a detection sequence")
            split, warnings = validate_tracking_document(
                document, config, allow_compatible_input=False
            )
            adapter = build_dataset_adapter(
                config.dataset.name, split, real_mode=config.dataset.real_mode
            )
            records = [
                record
                for frame in document["frames"]
                for record in frame.get("detections", [])
            ]
            for record in records:
                adapter.build_detection(record)
            if not args.dry_run:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(json.dumps({
                "status": "PASS", "mode": "detection-sequence",
                "frames": len(document["frames"]), "records": len(records),
                "compatibility_warnings": len(warnings),
                "output_written": not args.dry_run,
            }, sort_keys=True))
            return 0
        policy = ClassPromptPolicy(
            mode=config.prompting.mode if config else "hybrid",
            novel_label=config.prompting.novel_label if config else "Unknown",
            mask_novel_classes=config.prompting.mask_novel_classes if config else True,
            preserve_base_classes=config.prompting.preserve_base_classes if config else True,
        )
        rows = prepare_associations(
            document,
            history_length=config.dataset.history_length if config else None,
            auxiliary_loss_enabled=(
                config.model.auxiliary_head.enabled
                and config.training.auxiliary_loss_weight > 0
            ) if config else True,
            class_prompt_policy=policy,
            legacy_input=args.legacy_input,
        )
        if not args.dry_run:
            write_jsonl(args.output, rows)
            if args.legacy_input:
                report_path = args.migration_report or args.output.with_suffix(args.output.suffix + ".migration.json")
                report_path.write_text(json.dumps({
                    "source": str(args.input),
                    "output": str(args.output),
                    "records": len(rows),
                    "migration": "explicit legacy document to nova.association.v2",
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PASS", "records": len(rows), "output_written": not args.dry_run}, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
