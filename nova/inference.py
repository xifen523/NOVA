"""History-aware association scoring and configuration-consistent tracking."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from nova.config import ExperimentConfig, load_config
from nova.contracts.model import AssociationModelInput
from nova.data import ClassSplit, build_dataset_adapter
from nova.data.prepared import read_association_jsonl, write_jsonl
from nova.data.types import (
    AssociationCandidate,
    FrameMetadata,
    SerializedAssociationContext,
    TrackObservation,
    TrajectoryHistory,
)
from nova.models.loading import load_nova_model
from nova.models.tokenizer import resolve_token_ids
from nova.preprocessing.association import ClassPromptPolicy, serialize_association_context
from nova.tracking import (
    GateConfig,
    LifecycleConfig,
    LifecycleTracker,
    NuScenesPolicy,
    V2XKITTIPolicy,
    class_compatible,
    format_tracking_document,
)
from nova.tracking.types import TrackingFrameInput


def _runtime(
    config: ExperimentConfig,
    model_path: Optional[str],
    checkpoint: Optional[str],
    checkpoint_sha256: Optional[str],
    device: str,
    allow_download: bool,
    vocab_alignment: str,
):
    effective_model = model_path or config.model.base_model
    model, tokenizer, report = load_nova_model(
        effective_model,
        checkpoint_path=checkpoint or config.assets.checkpoint_path,
        checkpoint_sha256=checkpoint_sha256 or (
            config.assets.checkpoint_sha256 if checkpoint is None else None
        ),
        revision=config.model.revision,
        device=device,
        local_files_only=not allow_download,
        allow_download=allow_download,
        dtype=config.inference.dtype,
        vocab_alignment=vocab_alignment,
        model_config=config.model,
    )
    token_ids = resolve_token_ids(tokenizer, config.model.tokens)
    return model, tokenizer, token_ids, report


def _score_context_chunk(
    model,
    tokenizer,
    token_ids,
    contexts: Sequence[SerializedAssociationContext],
    device: str,
) -> torch.Tensor:
    encoded = tokenizer([context.prompt for context in contexts], padding=True, return_tensors="pt")
    geometry_rows = [row for context in contexts for row in context.geometry_features]
    inputs = AssociationModelInput(
        input_ids=encoded.input_ids.to(device),
        attention_mask=encoded.attention_mask.to(device),
        geometry=torch.tensor(
            geometry_rows,
            dtype=model.geometry_encoder.mlp[0].weight.dtype,
            device=device,
        ),
        box_counts=torch.tensor(
            [context.box_count for context in contexts], dtype=torch.long, device=device
        ),
        box_token_id=token_ids.box,
    )
    model.eval()
    with torch.no_grad():
        output = model(inputs, token_ids)
    return output.p_yes.detach().cpu()


def score_serialized_contexts(
    model,
    tokenizer,
    token_ids,
    contexts: Sequence[SerializedAssociationContext],
    *,
    device: str,
    pair_batch_size: int,
) -> Tuple[torch.Tensor, int]:
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be positive")
    outputs = []
    chunks = 0
    for start in range(0, len(contexts), pair_batch_size):
        outputs.append(_score_context_chunk(
            model, tokenizer, token_ids, contexts[start:start + pair_batch_size], device
        ))
        chunks += 1
    return (torch.cat(outputs) if outputs else torch.empty(0), chunks)


def _effective_pair_batch_size(config: ExperimentConfig) -> int:
    """Both public limits constrain the number of pairs sent in one forward."""

    return min(config.inference.pair_batch_size, config.inference.max_pairs_per_forward)


def _effective_class_gate(config: ExperimentConfig) -> bool:
    return config.inference.class_gate and config.inference.class_gate_policy == "exact_alias"


def _class_value(values: Mapping[str, float], category: str, split: ClassSplit, default):
    canonical = split.canonicalize(category)
    normalized = "_".join(canonical.strip().lower().split())
    for key, value in values.items():
        if "_".join(key.strip().lower().split()) == normalized:
            return value
    return default


def _row_context(row: Mapping[str, object]) -> SerializedAssociationContext:
    return SerializedAssociationContext(
        prompt=str(row["prompt"]),
        geometry_features=tuple(tuple(float(value) for value in item) for item in row["geometry"]),
        box_count=int(row["box_count"]),
        box_roles=tuple(str(value) for value in row["box_roles"]),
        token_metadata=dict(row.get("prompt_metadata", {})),
        timestamps=tuple(int(value) for value in row["box_timestamps"]),
    )


def infer_associations(
    input_path: Path,
    output_path: Path,
    *,
    config: Optional[ExperimentConfig] = None,
    backend: str = "real",
    model_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    device: str = "cpu",
    allow_download: bool = False,
    vocab_alignment: str = "strict",
    legacy_input: bool = False,
) -> Dict[str, object]:
    rows = read_association_jsonl(input_path, legacy_input=legacy_input)
    contexts = [_row_context(row) for row in rows]
    if backend == "real":
        if config is None:
            raise ValueError("--config is required for real inference")
        model, tokenizer, token_ids, report = _runtime(
            config, model_path, checkpoint, checkpoint_sha256,
            device, allow_download, vocab_alignment
        )
        probabilities, chunks = score_serialized_contexts(
            model,
            tokenizer,
            token_ids,
            contexts,
            device=device,
            pair_batch_size=_effective_pair_batch_size(config),
        )
        probability_values = probabilities.tolist()
        runtime = report.architecture
        threshold = config.inference.yes_threshold
    elif backend == "heuristic":
        probability_values = [
            float(1.0 / (1.0 + math.exp(-(2.0 * float(row.get("auxiliary_quality") or 0.5) - 1.0))))
            for row in rows
        ]
        runtime = "explicit-heuristic-smoke-backend"
        threshold = config.inference.yes_threshold if config else 0.5
        chunks = math.ceil(len(rows) / (_effective_pair_batch_size(config) if config else max(len(rows), 1)))
    else:
        raise ValueError("backend must be real or heuristic")
    predictions = []
    for row, probability in zip(rows, probability_values):
        predictions.append({
            "schema": "nova.association-prediction.v1",
            "sequence_id": row["sequence_id"],
            "frame_id": row["frame_id"],
            "track_id": row["history_track_id"],
            "detection_id": row["detection_id"],
            "p_yes": float(probability),
            "matched": bool(probability >= threshold),
            "box_count": row["box_count"],
        })
    write_jsonl(output_path, predictions)
    return {
        "samples": len(predictions),
        "backend": runtime,
        "output": str(output_path),
        "model_chunks": chunks,
        "box_counts": sorted(set(int(row["box_count"]) for row in rows)),
    }


def _configured_split(config: ExperimentConfig) -> ClassSplit:
    return ClassSplit.from_mapping(asdict(config.dataset.class_split))


def validate_tracking_document(
    document: Mapping[str, object],
    config: ExperimentConfig,
    *,
    allow_compatible_input: bool,
) -> Tuple[ClassSplit, List[str]]:
    warnings = []
    if document.get("schema") != "nova.detection-sequence.v1" or not isinstance(document.get("frames"), list):
        raise ValueError("Prepared detector input must use nova.detection-sequence.v1")
    checks = (
        ("dataset", document.get("dataset"), config.dataset.name),
        ("coordinate_frame", document.get("coordinate_frame"), config.dataset.coordinate_frame),
    )
    for name, actual, expected in checks:
        if actual != expected:
            message = "input {0} does not match config".format(name)
            if not allow_compatible_input:
                raise ValueError(message)
            warnings.append(message + ": {0!r} != {1!r}".format(actual, expected))
    split_document = document.get("class_split")
    if not isinstance(split_document, Mapping):
        raise ValueError("input class_split is required")
    split = ClassSplit.from_mapping(split_document)
    configured = _configured_split(config)
    if split != configured:
        if not allow_compatible_input:
            raise ValueError("input class split does not match config")
        warnings.append("input class split differs from config")
    frames = document["frames"]
    prior_sequence = None
    prior_key = None
    seen_sequences = set()
    for frame_document in frames:
        if not isinstance(frame_document, Mapping):
            raise ValueError("every frame must be an object")
        sequence_id = frame_document.get("sequence_id")
        frame_id = frame_document.get("frame_id")
        timestamp = frame_document.get("timestamp")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("sequence_id must be non-empty")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ValueError("frame_id must be an integer")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("timestamp must be an integer")
        if prior_sequence is None:
            seen_sequences.add(sequence_id)
        elif sequence_id != prior_sequence:
            if sequence_id in seen_sequences:
                raise ValueError("sequence frames must be contiguous")
            seen_sequences.add(sequence_id)
            prior_key = None
        key = (timestamp, frame_id)
        if prior_key is not None and key <= prior_key:
            raise ValueError("frame_id and timestamp must be strictly monotonic")
        prior_sequence = sequence_id
        prior_key = key
    return split, warnings


def _new_tracker(config: ExperimentConfig, split: ClassSplit) -> LifecycleTracker:
    policy = (
        NuScenesPolicy(config.inference.emission_mode == "active_and_lost")
        if config.dataset.name == "nuscenes"
        else V2XKITTIPolicy(config.inference.emission_mode == "active_and_lost")
    )
    return LifecycleTracker(
        split,
        GateConfig(
            detection_threshold=config.inference.detection_threshold,
            distance_gate=config.inference.distance_gate,
            yes_threshold=config.inference.yes_threshold,
            class_gate=_effective_class_gate(config),
            class_gate_policy=config.inference.class_gate_policy,
            detection_threshold_by_class=config.inference.detection_threshold_by_class,
            distance_gate_by_class=config.inference.distance_gate_by_class,
        ),
        LifecycleConfig(
            config.inference.max_age,
            config.inference.min_hits,
            config.inference.birth_threshold,
            config.inference.lost_score_decay,
        ),
        policy,
        distance_weight=config.inference.distance_weight,
        score_mode=config.inference.score_mode,
    )


def infer_detection_sequence(
    input_path: Path,
    output_path: Path,
    config_path: Path,
    *,
    backend: str = "real",
    model_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    device: str = "cpu",
    allow_download: bool = False,
    vocab_alignment: str = "strict",
    allow_compatible_input: bool = False,
    output_format: str = "generic",
) -> Dict[str, object]:
    document = json.loads(input_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    split, compatibility_warnings = validate_tracking_document(
        document, config, allow_compatible_input=allow_compatible_input
    )
    adapter = build_dataset_adapter(config.dataset.name, split, real_mode=config.dataset.real_mode)
    tracker = _new_tracker(config, split)
    runtime = None
    if backend == "real":
        runtime = _runtime(
            config, model_path, checkpoint, checkpoint_sha256,
            device, allow_download, vocab_alignment,
        )
    elif backend != "heuristic":
        raise ValueError("backend must be real or heuristic")
    policy = ClassPromptPolicy(
        mode=config.prompting.mode,
        novel_label=config.prompting.novel_label,
        mask_novel_classes=config.prompting.mask_novel_classes,
        preserve_base_classes=config.prompting.preserve_base_classes,
    )
    frames_out = []
    current_sequence = None
    counters = {
        "total_pairs": 0,
        "pairs_rejected_detection": 0,
        "pairs_rejected_distance": 0,
        "pairs_rejected_class": 0,
        "pairs_sent_to_model": 0,
        "model_chunks": 0,
    }
    observed_box_counts = set()
    for frame_document in document["frames"]:
        if current_sequence is not None and frame_document["sequence_id"] != current_sequence:
            tracker = _new_tracker(config, split)
        current_sequence = frame_document["sequence_id"]
        records = frame_document.get("detections", [])
        detections = tuple(adapter.build_detection(record) for record in records)
        if detections:
            frame = detections[0].frame
        else:
            frame = FrameMetadata(
                dataset_name=adapter.name,
                sequence_id=str(frame_document["sequence_id"]),
                frame_id=int(frame_document["frame_id"]),
                timestamp=int(frame_document["timestamp"]),
                sample_token=frame_document.get("sample_token"),
            )
        active = tuple(
            track for _, track in sorted(tracker.tracks.items())
            if track.status.value != "removed"
        )
        distances = np.zeros((len(active), len(detections)), dtype=np.float64)
        probabilities = np.zeros((len(active), len(detections)), dtype=np.float64)
        eligible_indices = []
        contexts = []
        counters["total_pairs"] += len(active) * len(detections)
        for track_index, track in enumerate(active):
            previous = track.latest_detection
            observations = tuple(
                TrackObservation(str(track.track_id), detection)
                for detection in track.history
            )
            history = TrajectoryHistory(str(track.track_id), observations)
            for detection_index, detection in enumerate(detections):
                distance = math.dist(previous.position_m, detection.position_m)
                distances[track_index, detection_index] = distance
                rejected = False
                detection_threshold = _class_value(
                    config.inference.detection_threshold_by_class,
                    detection.category,
                    split,
                    config.inference.detection_threshold,
                )
                distance_gate = _class_value(
                    config.inference.distance_gate_by_class,
                    track.category,
                    split,
                    config.inference.distance_gate,
                )
                if detection.score < detection_threshold:
                    counters["pairs_rejected_detection"] += 1
                    rejected = True
                if distance_gate is not None and distance > distance_gate:
                    counters["pairs_rejected_distance"] += 1
                    rejected = True
                if _effective_class_gate(config) and not class_compatible(
                    track, detection, split, config.inference.class_gate_policy
                ):
                    counters["pairs_rejected_class"] += 1
                    rejected = True
                if rejected:
                    continue
                context = serialize_association_context(
                    history,
                    AssociationCandidate(detection, candidate_source_id=detection.detection_id),
                    history_length=config.dataset.history_length,
                    class_prompt_policy=policy,
                    class_split=split,
                )
                contexts.append(context)
                eligible_indices.append((track_index, detection_index))
                observed_box_counts.add(context.box_count)
        counters["pairs_sent_to_model"] += len(contexts)
        if contexts and backend == "real":
            model, tokenizer, token_ids, _ = runtime
            flat, chunk_count = score_serialized_contexts(
                model,
                tokenizer,
                token_ids,
                contexts,
                device=device,
                pair_batch_size=_effective_pair_batch_size(config),
            )
            counters["model_chunks"] += chunk_count
            for (track_index, detection_index), value in zip(eligible_indices, flat.tolist()):
                probabilities[track_index, detection_index] = value
        elif contexts:
            chunk_count = math.ceil(len(contexts) / _effective_pair_batch_size(config))
            counters["model_chunks"] += chunk_count
            for track_index, detection_index in eligible_indices:
                probabilities[track_index, detection_index] = math.exp(
                    -distances[track_index, detection_index]
                    / max(
                        _class_value(
                            config.inference.distance_gate_by_class,
                            active[track_index].category,
                            split,
                            config.inference.distance_gate or 10.0,
                        ),
                        1e-6,
                    )
                )
        output = tracker.process_frame(
            TrackingFrameInput(frame, detections), probabilities, distances
        )
        frames_out.append({
            "sequence_id": frame.sequence_id,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "tracks": [
                {
                    "track_id": item.track_id,
                    "category": item.category,
                    "box_xyz_lwh_yaw": list(item.box_xyz_lwh_yaw),
                    "score": item.score,
                    "status": item.status.value,
                }
                for item in output.tracks
            ],
        })
    result = {
        "schema": "nova.tracking-output.v1",
        "dataset": document["dataset"],
        "coordinate_frame": document.get("coordinate_frame"),
        "frames": frames_out,
        "runtime_manifest": {
            "history_length": config.dataset.history_length,
            "observed_box_counts": sorted(observed_box_counts),
            "score_mode": config.inference.score_mode,
            "class_gate_policy": config.inference.class_gate_policy,
            "compatibility_warnings": compatibility_warnings,
            **counters,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = format_tracking_document(result, output_format)
    output_path.write_text(json.dumps(exported, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "frames": len(frames_out),
        "output": str(output_path),
        "backend": backend,
        **counters,
        "history_length": config.dataset.history_length,
        "observed_box_counts": sorted(observed_box_counts),
        "compatibility_warnings": len(compatibility_warnings),
        "output_format": output_format,
    }
