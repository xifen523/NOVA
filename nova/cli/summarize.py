"""Structural diagnostics for NOVA prediction files; no benchmark claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def summarize_predictions(path: Path, dataset: str) -> dict:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("{") and "\n{" not in text.strip():
        document = json.loads(text)
        if document.get("schema") == "nova.association-prediction.v1":
            return {"dataset": dataset, "association_predictions": 1, "matched": int(bool(document.get("matched")))}
        frames = document.get("frames")
        if not isinstance(frames, list):
            raise ValueError("unrecognized prediction document")
        track_ids = {
            (frame.get("sequence_id"), track.get("track_id"))
            for frame in frames for track in frame.get("tracks", [])
        }
        return {
            "dataset": dataset,
            "frames": len(frames),
            "emitted_tracks": sum(len(frame.get("tracks", [])) for frame in frames),
            "unique_tracks": len(track_ids),
            "diagnostic_only": True,
        }
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("prediction input is empty")
    return {
        "dataset": dataset,
        "association_predictions": len(rows),
        "matched": sum(bool(row.get("matched")) for row in rows),
        "diagnostic_only": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize NOVA prediction structure")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", choices=("v2x", "kitti", "nuscenes"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if not args.input.is_file():
            raise FileNotFoundError("prediction input was not found")
        summary = summarize_predictions(args.input, args.dataset)
        if args.output and not args.dry_run:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
