"""Pure, explicitly configurable association gates and rejection reasons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from nova.data.class_split import ClassSplit
from nova.data.types import Detection3D

from .types import TrackState


@dataclass(frozen=True)
class GateConfig:
    detection_threshold: Optional[float] = 0.1
    distance_gate: Optional[float] = None
    yes_threshold: Optional[float] = 0.5
    class_gate: bool = False
    class_gate_policy: str = "exact_alias"
    detection_threshold_by_class: Mapping[str, float] = None
    distance_gate_by_class: Mapping[str, float] = None

    def __post_init__(self) -> None:
        for name, value in (
            ("detection_threshold", self.detection_threshold),
            ("distance_gate", self.distance_gate),
            ("yes_threshold", self.yes_threshold),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
            ):
                raise ValueError("{0} must be null or finite and non-negative".format(name))
        if self.detection_threshold is not None and self.detection_threshold > 1:
            raise ValueError("detection_threshold must be in [0,1]")
        if self.yes_threshold is not None and self.yes_threshold > 1:
            raise ValueError("yes_threshold must be in [0,1]")
        if not isinstance(self.class_gate, bool):
            raise TypeError("class_gate must be bool")
        if self.class_gate_policy not in {"exact_alias", "base_novel", "disabled"}:
            raise ValueError("class_gate_policy is invalid")
        object.__setattr__(self, "detection_threshold_by_class", dict(self.detection_threshold_by_class or {}))
        object.__setattr__(self, "distance_gate_by_class", dict(self.distance_gate_by_class or {}))


def _category_value(values: Mapping[str, float], category: str, split: ClassSplit, default):
    canonical = split.canonicalize(category)
    normalized = "_".join(canonical.strip().lower().split())
    for key, value in values.items():
        if "_".join(key.strip().lower().split()) == normalized:
            return value
    return default


@dataclass(frozen=True)
class GateDecision:
    mask: np.ndarray
    rejection_reasons: Tuple[Tuple[Tuple[str, ...], ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.mask, np.ndarray) or self.mask.ndim != 2 or self.mask.dtype != np.bool_:
            raise TypeError("mask must be a rank-two Boolean ndarray")


def _class_compatible(
    track: TrackState,
    detection: Detection3D,
    split: ClassSplit,
    policy: str = "exact_alias",
) -> bool:
    if policy == "base_novel":
        return split.classify(track.category) == split.classify(detection.category)
    return split.canonicalize(track.category) == split.canonicalize(detection.category)


def class_compatible(
    track: TrackState,
    detection: Detection3D,
    split: ClassSplit,
    policy: str = "exact_alias",
) -> bool:
    """Apply the explicit canonical-equality or Base/Novel gate policy."""

    return _class_compatible(track, detection, split, policy)


def build_pair_gate(
    tracks: Sequence[TrackState],
    detections: Sequence[Detection3D],
    p_yes: Sequence[Sequence[float]],
    distances: Sequence[Sequence[float]],
    config: GateConfig,
    class_split: ClassSplit,
) -> GateDecision:
    if not isinstance(config, GateConfig) or not isinstance(class_split, ClassSplit):
        raise TypeError("config and class_split must be typed contracts")
    probabilities = np.asarray(p_yes, dtype=np.float64)
    distance_array = np.asarray(distances, dtype=np.float64)
    shape = (len(tracks), len(detections))
    if probabilities.shape != shape or distance_array.shape != shape:
        raise ValueError("gate matrices must have shape [tracks,detections]")
    if not np.isfinite(probabilities).all() or not np.isfinite(distance_array).all():
        raise ValueError("gate matrices must be finite")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or np.any(distance_array < 0):
        raise ValueError("invalid probability or distance")
    mask = np.ones(shape, dtype=np.bool_)
    reason_rows = []
    for track_index, track in enumerate(tracks):
        reason_columns = []
        for detection_index, detection in enumerate(detections):
            reasons = []
            detection_threshold = _category_value(
                config.detection_threshold_by_class, detection.category, class_split,
                config.detection_threshold,
            )
            distance_gate = _category_value(
                config.distance_gate_by_class, track.category, class_split,
                config.distance_gate,
            )
            if detection_threshold is not None and detection.score < detection_threshold:
                reasons.append("DETECTION_THRESHOLD")
            if distance_gate is not None and distance_array[track_index, detection_index] > distance_gate:
                reasons.append("DISTANCE_GATE")
            if config.yes_threshold is not None and probabilities[track_index, detection_index] < config.yes_threshold:
                reasons.append("YES_THRESHOLD")
            if config.class_gate and not _class_compatible(
                track, detection, class_split, config.class_gate_policy
            ):
                reasons.append("CLASS_GATE")
            if reasons:
                mask[track_index, detection_index] = False
            reason_columns.append(tuple(reasons))
        reason_rows.append(tuple(reason_columns))
    return GateDecision(mask=mask, rejection_reasons=tuple(reason_rows))
