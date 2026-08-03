"""In-memory tracking output only; no submission serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .types import TrackState, TrackStatus


@dataclass(frozen=True)
class TrackOutput:
    track_id: int
    frame_id: int
    category: str
    box_xyz_lwh_yaw: Tuple[float, float, float, float, float, float, float]
    score: float
    status: TrackStatus


def track_to_output(track: TrackState, frame_id: int) -> TrackOutput:
    detection = track.latest_detection
    x, y, z = detection.position_m
    length, width, height = detection.size_lwh_m
    return TrackOutput(
        track_id=track.track_id,
        frame_id=frame_id,
        category=track.category,
        box_xyz_lwh_yaw=(x, y, z, length, width, height, detection.yaw_rad),
        score=float(track.score),
        status=track.status,
    )
