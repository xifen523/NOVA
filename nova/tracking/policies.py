"""Shared lifecycle configuration and explicit dataset emission policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .types import TrackState, TrackStatus


@dataclass(frozen=True)
class LifecycleConfig:
    max_age: int = 3
    min_hits: int = 1
    birth_threshold: float = 0.1
    lost_score_decay: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.max_age, bool) or not isinstance(self.max_age, int) or self.max_age < 0:
            raise ValueError("max_age must be a non-negative integer")
        if isinstance(self.min_hits, bool) or not isinstance(self.min_hits, int) or self.min_hits <= 0:
            raise ValueError("min_hits must be a positive integer")
        if not 0.0 <= self.birth_threshold <= 1.0:
            raise ValueError("birth_threshold must be in [0,1]")
        if not 0.0 <= self.lost_score_decay <= 1.0:
            raise ValueError("lost_score_decay must be in [0,1]")


@dataclass(frozen=True)
class V2XKITTIPolicy:
    """Select whether still-retained missed tracks are emitted."""

    emit_lost: bool = False

    @property
    def emission_mode(self) -> str:
        return "active_and_lost" if self.emit_lost else "active_only"

    def select(self, tracks: Tuple[TrackState, ...]) -> Tuple[TrackState, ...]:
        allowed = {TrackStatus.ACTIVE}
        if self.emit_lost:
            allowed.add(TrackStatus.LOST)
        return tuple(track for track in tracks if track.status in allowed)


@dataclass(frozen=True)
class NuScenesPolicy:
    """Lost emission is opt-in and never inherited by other datasets."""

    emit_lost: bool = False
    min_hits_for_lost: int = 1
    min_lost_score: float = 0.0

    @property
    def emission_mode(self) -> str:
        return "active_and_lost" if self.emit_lost else "active_only"

    def select(self, tracks: Tuple[TrackState, ...]) -> Tuple[TrackState, ...]:
        selected = []
        for track in tracks:
            if track.status is TrackStatus.ACTIVE:
                selected.append(track)
            elif (
                self.emit_lost
                and track.status is TrackStatus.LOST
                and track.hits >= self.min_hits_for_lost
                and track.score >= self.min_lost_score
            ):
                selected.append(track)
        return tuple(selected)
