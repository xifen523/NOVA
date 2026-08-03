"""Explicit immutable data contracts used after adapter validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

import torch


class CoordinateFrame(str, Enum):
    V2X_EGO_LIDAR = "v2x_ego_lidar"
    KITTI_LIDAR = "kitti_lidar"
    NUSCENES_GLOBAL = "nuscenes_global"


class ClassMembership(str, Enum):
    BASE = "base"
    NOVEL = "novel"
    UNKNOWN = "unknown"


class AssociationMode(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


class SamplingType(str, Enum):
    POSITIVE_EXACT = "positive_exact"
    POSITIVE_JITTERED = "positive_jittered"
    NEGATIVE_HARD = "negative_hard"
    NEGATIVE_LOCAL = "negative_local"
    NEGATIVE_RANDOM = "negative_random"


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{0} must be a non-empty string".format(name))


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("{0} must be numeric".format(name))
    if not math.isfinite(float(value)):
        raise ValueError("{0} must be finite".format(name))


def _vector3(value: Tuple[float, float, float], name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("{0} must be a three-value tuple".format(name))
    for component in value:
        _finite(component, name)


@dataclass(frozen=True)
class FrameMetadata:
    """Required non-negative frame/time ids plus an optional non-empty token."""

    dataset_name: str
    sequence_id: str
    frame_id: int
    timestamp: int
    sample_token: Optional[str] = None

    def __post_init__(self) -> None:
        _nonempty(self.dataset_name, "dataset_name")
        _nonempty(self.sequence_id, "sequence_id")
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise TypeError("frame_id must be an integer")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, int):
            raise TypeError("timestamp must be an integer")
        if self.frame_id < 0 or self.timestamp < 0:
            raise ValueError("frame_id and timestamp must be non-negative")
        if self.sample_token is not None:
            _nonempty(self.sample_token, "sample_token")


@dataclass(frozen=True)
class SequenceMetadata:
    """Sequence identity and coordinate frame; expected count may be omitted."""

    dataset_name: str
    sequence_id: str
    coordinate_frame: CoordinateFrame
    expected_frame_count: Optional[int] = None

    def __post_init__(self) -> None:
        _nonempty(self.dataset_name, "dataset_name")
        _nonempty(self.sequence_id, "sequence_id")
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("coordinate_frame must be a CoordinateFrame")
        if self.expected_frame_count is not None and (
            isinstance(self.expected_frame_count, bool)
            or not isinstance(self.expected_frame_count, int)
            or self.expected_frame_count < 0
        ):
            raise ValueError("expected_frame_count must be a non-negative integer")


@dataclass(frozen=True)
class Detection3D:
    """Finite xyz/lwh meters, yaw radians, score [0,1], class, and frame."""

    detection_id: str
    frame: FrameMetadata
    position_m: Tuple[float, float, float]
    size_lwh_m: Tuple[float, float, float]
    yaw_rad: float
    score: float
    category: str
    coordinate_frame: CoordinateFrame
    class_membership: ClassMembership

    def __post_init__(self) -> None:
        _nonempty(self.detection_id, "detection_id")
        if not isinstance(self.frame, FrameMetadata):
            raise TypeError("frame must be FrameMetadata")
        _vector3(self.position_m, "position_m")
        _vector3(self.size_lwh_m, "size_lwh_m")
        if any(float(value) <= 0.0 for value in self.size_lwh_m):
            raise ValueError("size_lwh_m components must be positive")
        _finite(self.yaw_rad, "yaw_rad")
        _finite(self.score, "score")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be in the closed interval zero to one")
        _nonempty(self.category, "category")
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("coordinate_frame must be a CoordinateFrame")
        if not isinstance(self.class_membership, ClassMembership):
            raise TypeError("class_membership must be a ClassMembership")


@dataclass(frozen=True)
class TrackObservation:
    """One typed detection assigned to one required track identity."""

    track_id: str
    detection: Detection3D

    def __post_init__(self) -> None:
        _nonempty(self.track_id, "track_id")
        if not isinstance(self.detection, Detection3D):
            raise TypeError("detection must be Detection3D")


@dataclass(frozen=True)
class TrajectoryHistory:
    """Possibly empty tuple; populated observations are strictly time ordered."""

    track_id: str
    observations: Tuple[TrackObservation, ...]

    def __post_init__(self) -> None:
        _nonempty(self.track_id, "track_id")
        if not isinstance(self.observations, tuple):
            raise TypeError("observations must be a tuple")
        previous_key: Optional[Tuple[int, int]] = None
        expected_context: Optional[Tuple[str, str, CoordinateFrame]] = None
        for observation in self.observations:
            if not isinstance(observation, TrackObservation):
                raise TypeError("every observation must be TrackObservation")
            if observation.track_id != self.track_id:
                raise ValueError("observation track_id does not match history")
            detection = observation.detection
            key = (detection.frame.timestamp, detection.frame.frame_id)
            if previous_key is not None and key <= previous_key:
                raise ValueError("history observations must be strictly ordered")
            previous_key = key
            context = (
                detection.frame.dataset_name,
                detection.frame.sequence_id,
                detection.coordinate_frame,
            )
            if expected_context is not None and context != expected_context:
                raise ValueError("history observations must share dataset, sequence, and frame")
            expected_context = context


@dataclass(frozen=True)
class AssociationCandidate:
    """Detection with optional Boolean target and optional quality in [0,1]."""

    detection: Detection3D
    expected_match: Optional[bool] = None
    auxiliary_quality: Optional[float] = None
    candidate_source_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.detection, Detection3D):
            raise TypeError("detection must be Detection3D")
        if self.expected_match is not None and not isinstance(self.expected_match, bool):
            raise TypeError("expected_match must be bool or None")
        if self.auxiliary_quality is not None:
            _finite(self.auxiliary_quality, "auxiliary_quality")
            if not 0.0 <= float(self.auxiliary_quality) <= 1.0:
                raise ValueError("auxiliary_quality must be in the closed interval zero to one")
        if self.candidate_source_id is not None:
            _nonempty(self.candidate_source_id, "candidate_source_id")


@dataclass(frozen=True)
class SerializedAssociationContext:
    """One prompt and its exactly aligned oldest-to-newest geometry sequence."""

    prompt: str
    geometry_features: Tuple[Tuple[float, ...], ...]
    box_count: int
    box_roles: Tuple[str, ...]
    token_metadata: Mapping[str, Any]
    timestamps: Tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty(self.prompt, "prompt")
        if isinstance(self.box_count, bool) or not isinstance(self.box_count, int):
            raise TypeError("box_count must be an integer")
        if self.box_count <= 0:
            raise ValueError("candidate context must contain at least one box")
        if len(self.geometry_features) != self.box_count:
            raise ValueError("geometry row count must equal box_count")
        if len(self.box_roles) != self.box_count or len(self.timestamps) != self.box_count:
            raise ValueError("box roles and timestamps must equal box_count")
        if self.prompt.count("<box>") != self.box_count:
            raise ValueError("prompt box-token count must equal box_count")
        if not self.box_roles or self.box_roles[-1] != "candidate":
            raise ValueError("candidate must be the final serialized box")
        for row in self.geometry_features:
            if not isinstance(row, tuple) or len(row) != 9:
                raise ValueError("each serialized geometry row must contain nine values")
            for value in row:
                _finite(value, "geometry_features")
        if any(
            self.timestamps[index] >= self.timestamps[index + 1]
            for index in range(max(0, len(self.timestamps) - 2))
        ):
            raise ValueError("serialized history timestamps must be strictly ordered")
        if self.box_count > 1 and self.timestamps[-1] <= self.timestamps[-2]:
            raise ValueError("candidate timestamp must follow serialized history")
        if not isinstance(self.token_metadata, Mapping):
            raise TypeError("token_metadata must be a mapping")


@dataclass(frozen=True)
class AssociationSample:
    """Explicit-mode history/candidate pair ready for shared serialization."""

    history: TrajectoryHistory
    candidate: AssociationCandidate
    semantic_prompt: str
    target_token: Optional[str]
    dataset_name: str
    coordinate_frame: CoordinateFrame
    context: Optional[SerializedAssociationContext] = None
    mode: AssociationMode = AssociationMode.INFERENCE
    sampling_type: Optional[SamplingType] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.history, TrajectoryHistory):
            raise TypeError("history must be TrajectoryHistory")
        if not isinstance(self.candidate, AssociationCandidate):
            raise TypeError("candidate must be AssociationCandidate")
        _nonempty(self.semantic_prompt, "semantic_prompt")
        _nonempty(self.dataset_name, "dataset_name")
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("coordinate_frame must be a CoordinateFrame")
        if self.target_token not in {None, "Yes", "No"}:
            raise ValueError("target_token must be Yes, No, or None")
        if not isinstance(self.mode, AssociationMode):
            raise TypeError("mode must be AssociationMode")
        if self.sampling_type is not None and not isinstance(self.sampling_type, SamplingType):
            raise TypeError("sampling_type must be SamplingType or None")
        if self.mode is AssociationMode.TRAINING:
            if self.candidate.expected_match is None or self.target_token is None:
                raise ValueError("training samples require an explicit expected_match")
            if self.sampling_type is None:
                raise ValueError("training samples require sampling_type")
        elif self.mode is AssociationMode.INFERENCE and self.candidate.expected_match is not None:
            raise ValueError("inference samples must not carry expected_match supervision")
        if self.context is not None:
            if not isinstance(self.context, SerializedAssociationContext):
                raise TypeError("context must be SerializedAssociationContext")
            if self.context.prompt != self.semantic_prompt:
                raise ValueError("semantic_prompt must equal serialized context prompt")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        candidate = self.candidate.detection
        if candidate.frame.dataset_name != self.dataset_name or candidate.coordinate_frame != self.coordinate_frame:
            raise ValueError("association sample dataset, sequence, or coordinate frame mismatch")
        if self.history.observations:
            last = self.history.observations[-1].detection
            if (
                last.frame.dataset_name != self.dataset_name
                or last.frame.sequence_id != candidate.frame.sequence_id
                or last.coordinate_frame != self.coordinate_frame
            ):
                raise ValueError("association sample dataset, sequence, or coordinate frame mismatch")


@dataclass(frozen=True)
class PreparedBatch:
    """Non-empty typed samples with flattened multi-box geometry."""

    samples: Tuple[AssociationSample, ...]
    geometry: torch.Tensor
    box_counts: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple) or not self.samples:
            raise ValueError("samples must be a non-empty tuple")
        if not isinstance(self.geometry, torch.Tensor):
            raise TypeError("geometry must be a torch.Tensor")
        if self.geometry.ndim != 2 or self.geometry.shape[1] != 9:
            raise ValueError("geometry must have shape [total_boxes, 9]")
        if not torch.is_floating_point(self.geometry):
            raise TypeError("geometry must have floating-point dtype")
        if not torch.isfinite(self.geometry).all().item():
            raise ValueError("geometry must contain only finite values")
        counts = self.box_counts or tuple(
            sample.context.box_count if sample.context is not None else 1
            for sample in self.samples
        )
        if len(counts) != len(self.samples) or any(count <= 0 for count in counts):
            raise ValueError("box_counts must contain one positive count per sample")
        if sum(counts) != self.geometry.shape[0]:
            raise ValueError("sum(box_counts) must equal geometry rows")
        object.__setattr__(self, "box_counts", tuple(counts))
