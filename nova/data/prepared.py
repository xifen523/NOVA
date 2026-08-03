"""Portable, explicit-mode association and detection formats."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional

from nova.preprocessing.association import ClassPromptPolicy

from .class_split import ClassSplit
from .registry import build_dataset_adapter
from .types import AssociationMode, SamplingType


ASSOCIATION_SCHEMA = "nova.association.v2"
LEGACY_ASSOCIATION_SCHEMA = "nova.association.v1"
SEQUENCE_SCHEMA = "nova.detection-sequence.v1"


class PreparedDataError(ValueError):
    pass


def _mode(value: object) -> AssociationMode:
    try:
        return AssociationMode(str(value))
    except ValueError as exc:
        raise PreparedDataError("mode must be training, inference, or evaluation") from exc


def prepare_associations(
    document: Mapping[str, object],
    *,
    history_length: Optional[int] = None,
    auxiliary_loss_enabled: bool = True,
    class_prompt_policy: Optional[ClassPromptPolicy] = None,
    legacy_input: bool = False,
) -> List[dict]:
    """Convert one validated adapter document into explicit v2 association rows."""

    dataset = document.get("dataset")
    split_document = document.get("class_split")
    history_records = document.get("history")
    candidates = document.get("candidates")
    raw_mode = document.get("mode")
    if raw_mode is None:
        if not legacy_input:
            raise PreparedDataError("mode is required; use --legacy-input for explicit migration")
        raw_mode = "training"
    mode = _mode(raw_mode)
    if not isinstance(dataset, str) or not isinstance(split_document, Mapping):
        raise PreparedDataError("dataset and class_split are required")
    if not isinstance(history_records, list):
        raise PreparedDataError("history must be a list")
    if not isinstance(candidates, list) or not candidates:
        raise PreparedDataError("candidates must be a non-empty list")
    if history_length is None:
        requested = document.get("history_length", len(history_records))
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
            raise PreparedDataError("history_length must be a non-negative integer")
        history_length = requested
    split = ClassSplit.from_mapping(split_document)
    adapter = build_dataset_adapter(dataset, split, real_mode=split.authoritative)
    history_track_id = document.get("history_track_id")
    if history_track_id is None and history_records:
        history_track_id = history_records[0].get("track_id")
    if not isinstance(history_track_id, str) or not history_track_id.strip():
        raise PreparedDataError("history_track_id is required")
    history = adapter.build_history(history_track_id, history_records)
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise PreparedDataError("every candidate must be an object")
        expected = candidate.get("expected_match")
        quality = candidate.get("auxiliary_quality")
        source_id = candidate.get("candidate_source_id")
        sampling_raw = candidate.get("sampling_type")
        sampling_type = None
        if mode is AssociationMode.TRAINING:
            if not isinstance(expected, bool):
                raise PreparedDataError("training candidates require Boolean expected_match")
            if not isinstance(source_id, str) or not source_id.strip():
                raise PreparedDataError("training candidates require candidate_source_id")
            if not isinstance(sampling_raw, str):
                raise PreparedDataError("training candidates require sampling_type")
            try:
                sampling_type = SamplingType(sampling_raw)
            except ValueError as exc:
                raise PreparedDataError("sampling_type is invalid") from exc
            if auxiliary_loss_enabled and quality is None:
                raise PreparedDataError(
                    "auxiliary_quality is required when auxiliary loss is enabled"
                )
        elif mode is AssociationMode.INFERENCE:
            if expected is not None:
                raise PreparedDataError("inference candidates must not contain expected_match")
            quality = None
        elif expected is not None and not isinstance(expected, bool):
            raise PreparedDataError("evaluation expected_match must be Boolean or null")
        if quality is not None:
            if isinstance(quality, bool) or not isinstance(quality, (int, float)):
                raise PreparedDataError("auxiliary_quality must be numeric or null")
            if not math.isfinite(float(quality)) or not 0.0 <= float(quality) <= 1.0:
                raise PreparedDataError("auxiliary_quality must be in [0,1]")
        sample = adapter.build_association_sample(
            history,
            candidate,
            expected_match=expected if isinstance(expected, bool) else None,
            auxiliary_quality=float(quality) if quality is not None else None,
            candidate_source_id=source_id if isinstance(source_id, str) else None,
            mode=mode,
            sampling_type=sampling_type,
            history_length=history_length,
            class_prompt_policy=class_prompt_policy,
        )
        detection = sample.candidate.detection
        if sample.context is None:
            raise RuntimeError("association serializer did not produce a context")
        rows.append({
            "schema": ASSOCIATION_SCHEMA,
            "mode": mode.value,
            "dataset": dataset,
            "sequence_id": detection.frame.sequence_id,
            "frame_id": detection.frame.frame_id,
            "timestamp": detection.frame.timestamp,
            "history_track_id": history.track_id,
            "candidate_source_id": sample.candidate.candidate_source_id,
            "detection_id": detection.detection_id,
            "category": detection.category,
            "coordinate_frame": detection.coordinate_frame.value,
            "prompt": sample.context.prompt,
            "geometry": [list(row) for row in sample.context.geometry_features],
            "box_count": sample.context.box_count,
            "box_roles": list(sample.context.box_roles),
            "box_timestamps": list(sample.context.timestamps),
            "prompt_metadata": dict(sample.context.token_metadata),
            "target": sample.target_token,
            "auxiliary_quality": sample.candidate.auxiliary_quality,
            "sampling_type": sample.sampling_type.value if sample.sampling_type else None,
        })
    return rows


def _migrate_legacy_row(row: Mapping[str, object]) -> dict:
    geometry = row.get("geometry")
    if not isinstance(geometry, list) or len(geometry) != 9:
        raise PreparedDataError("legacy geometry must contain nine values")
    target = row.get("target")
    migrated = dict(row)
    migrated.update({
        "schema": ASSOCIATION_SCHEMA,
        "mode": "training" if target in {"Yes", "No"} else "inference",
        "history_track_id": row.get("track_id"),
        "candidate_source_id": row.get("candidate_source_id", row.get("detection_id")),
        "geometry": [geometry],
        "box_count": 1,
        "box_roles": ["candidate"],
        "box_timestamps": [row.get("timestamp")],
        "prompt_metadata": {"legacy_migration": True, "history_serialized": 0},
        "auxiliary_quality": row.get("quality"),
        "sampling_type": row.get("sampling_type"),
    })
    return migrated


def validate_association_row(
    row: Mapping[str, object],
    *,
    require_target: bool = False,
    require_auxiliary: bool = False,
    legacy_input: bool = False,
) -> dict:
    if row.get("schema") == LEGACY_ASSOCIATION_SCHEMA:
        if not legacy_input:
            raise PreparedDataError("legacy association rows require --legacy-input")
        row = _migrate_legacy_row(row)
    if row.get("schema") != ASSOCIATION_SCHEMA:
        raise PreparedDataError("unsupported association schema")
    mode = _mode(row.get("mode"))
    required_strings = (
        "dataset", "sequence_id", "history_track_id", "candidate_source_id",
        "detection_id", "category", "prompt", "coordinate_frame",
    )
    for name in required_strings:
        if not isinstance(row.get(name), str) or not str(row[name]).strip():
            raise PreparedDataError("{0} must be a non-empty string".format(name))
    geometry = row.get("geometry")
    box_count = row.get("box_count")
    roles = row.get("box_roles")
    timestamps = row.get("box_timestamps")
    if isinstance(box_count, bool) or not isinstance(box_count, int) or box_count <= 0:
        raise PreparedDataError("box_count must be a positive integer")
    if not isinstance(geometry, list) or len(geometry) != box_count:
        raise PreparedDataError("geometry rows must equal box_count")
    parsed_geometry = []
    for geometry_row in geometry:
        if not isinstance(geometry_row, list) or len(geometry_row) != 9:
            raise PreparedDataError("each geometry row must contain nine values")
        try:
            numeric = [float(value) for value in geometry_row]
        except (TypeError, ValueError) as exc:
            raise PreparedDataError("geometry must contain numeric values") from exc
        if not all(math.isfinite(value) for value in numeric):
            raise PreparedDataError("geometry must be finite")
        parsed_geometry.append(numeric)
    if not isinstance(roles, list) or len(roles) != box_count or roles[-1] != "candidate":
        raise PreparedDataError("box_roles must align and end with candidate")
    if not isinstance(timestamps, list) or len(timestamps) != box_count:
        raise PreparedDataError("box_timestamps must align with geometry")
    if str(row["prompt"]).count("<box>") != box_count:
        raise PreparedDataError("prompt box-token count must equal box_count")
    target = row.get("target")
    if target not in {None, "Yes", "No"}:
        raise PreparedDataError("target must be Yes, No, or null")
    if mode is AssociationMode.TRAINING or require_target:
        if target not in {"Yes", "No"}:
            raise PreparedDataError("target must be Yes or No for training")
        if row.get("sampling_type") not in {item.value for item in SamplingType}:
            raise PreparedDataError("training rows require valid sampling_type")
    if mode is AssociationMode.INFERENCE and target is not None:
        raise PreparedDataError("inference rows must not carry target supervision")
    quality = row.get("auxiliary_quality")
    if quality is not None:
        try:
            quality = float(quality)
        except (TypeError, ValueError) as exc:
            raise PreparedDataError("auxiliary_quality must be numeric") from exc
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise PreparedDataError("auxiliary_quality must be in [0,1]")
    if require_auxiliary and quality is None:
        raise PreparedDataError("auxiliary_quality is required for auxiliary training")
    result = dict(row)
    result["geometry"] = parsed_geometry
    result["auxiliary_quality"] = quality
    result["mode"] = mode.value
    return result


def iter_association_jsonl(
    path: Path,
    *,
    require_target: bool = False,
    require_auxiliary: bool = False,
    legacy_input: bool = False,
) -> Iterator[dict]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Prepared association input was not found: {0}".format(path))
    found = False
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            found = True
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreparedDataError("invalid JSON on line {0}".format(line_number)) from exc
            if not isinstance(document, Mapping):
                raise PreparedDataError("association JSONL rows must be objects")
            yield validate_association_row(
                document,
                require_target=require_target,
                require_auxiliary=require_auxiliary,
                legacy_input=legacy_input,
            )
    if not found:
        raise PreparedDataError("prepared association input is empty")


def read_association_jsonl(
    path: Path,
    *,
    require_target: bool = False,
    require_auxiliary: bool = False,
    legacy_input: bool = False,
) -> List[dict]:
    return list(iter_association_jsonl(
        path,
        require_target=require_target,
        require_auxiliary=require_auxiliary,
        legacy_input=legacy_input,
    ))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
