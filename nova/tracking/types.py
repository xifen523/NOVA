"""Tracking contracts independent of dataset runners and serialization."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from nova.data.types import CoordinateFrame, Detection3D, FrameMetadata


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class TrackState:
    """Controlled mutable state for one sequence-local track."""

    track_id: int
    sequence_id: str
    coordinate_frame: CoordinateFrame
    category: str
    history: List[Detection3D] = field(default_factory=list)
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 0
    misses: int = 0
    score: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if not isinstance(self.coordinate_frame, CoordinateFrame):
            raise TypeError("coordinate_frame must be CoordinateFrame")
        if not isinstance(self.category, str) or not self.category:
            raise ValueError("category must be a non-empty string")
        if not isinstance(self.history, list) or any(
            not isinstance(item, Detection3D) for item in self.history
        ):
            raise TypeError("history must be a list of Detection3D")
        if not isinstance(self.status, TrackStatus):
            raise TypeError("status must be TrackStatus")
        for name, value in (("hits", self.hits), ("misses", self.misses)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("{0} must be a non-negative integer".format(name))
        if not isinstance(self.score, (int, float)) or not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")

    @property
    def latest_detection(self) -> Detection3D:
        if not self.history:
            raise ValueError("track history is empty")
        return self.history[-1]


@dataclass(frozen=True)
class AssociationScore:
    track_index: int
    detection_index: int
    p_yes: float
    distance: float
    cost: float

    def __post_init__(self) -> None:
        if self.track_index < 0 or self.detection_index < 0:
            raise ValueError("association indices must be non-negative")
        for name, value in (("p_yes", self.p_yes), ("distance", self.distance), ("cost", self.cost)):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("{0} must be finite".format(name))
        if not 0.0 <= float(self.p_yes) <= 1.0:
            raise ValueError("p_yes must be in [0,1]")
        if self.distance < 0.0 or self.cost < 0.0:
            raise ValueError("distance and cost must be non-negative")


@dataclass(frozen=True)
class AssignmentResult:
    matches: Tuple[Tuple[int, int], ...]
    unmatched_tracks: Tuple[int, ...]
    unmatched_detections: Tuple[int, ...]


@dataclass(frozen=True)
class LifecycleDecision:
    track_id: int
    previous_status: Optional[TrackStatus]
    new_status: TrackStatus
    action: str


@dataclass(frozen=True)
class TrackingFrameInput:
    frame: FrameMetadata
    detections: Tuple[Detection3D, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, FrameMetadata):
            raise TypeError("frame must be FrameMetadata")
        if not isinstance(self.detections, tuple):
            raise TypeError("detections must be a tuple")
        for detection in self.detections:
            if not isinstance(detection, Detection3D):
                raise TypeError("every detection must be Detection3D")
            if detection.frame != self.frame:
                raise ValueError("detection metadata must equal frame metadata")


@dataclass(frozen=True)
class TrackingFrameOutput:
    frame: FrameMetadata
    tracks: Tuple["TrackOutput", ...]
    assignment: AssignmentResult
    decisions: Tuple[LifecycleDecision, ...]
