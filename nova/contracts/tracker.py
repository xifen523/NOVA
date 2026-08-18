"""Tracker contracts only; lifecycle and association are not implemented."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrackStatus(str, Enum):
    ACTIVE = "active"
    LOST = "lost"
    DEAD = "dead"


@dataclass(frozen=True)
class TrackState:
    track_id: int
    age: int
    status: TrackStatus


@dataclass(frozen=True)
class TrackerConfigContract:
    history_length: int = 3
    max_age: int = 3
    yes_threshold: float = 0.5
    birth_threshold: Optional[float] = None
    detection_threshold: float = 0.1
    distance_gate: Optional[float] = None
    class_gate: str = "dataset_specific"
    score_mode: str = "detection"
    emission_mode: str = "dataset_specific"
    association_distance_weight: float = 0.1

