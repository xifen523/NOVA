"""Shared state machine and synthetic frame orchestrator."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from nova.data.class_split import ClassSplit
from nova.data.types import Detection3D

from .assignment import assign_detections
from .costs import association_cost_matrix
from .gates import GateConfig, build_pair_gate
from .output import track_to_output
from .policies import LifecycleConfig, NuScenesPolicy, V2XKITTIPolicy
from .types import (
    LifecycleDecision, TrackState, TrackStatus, TrackingFrameInput, TrackingFrameOutput,
)


def birth_track(track_id: int, detection: Detection3D, config: LifecycleConfig) -> Tuple[TrackState, LifecycleDecision]:
    if detection.score < config.birth_threshold:
        raise ValueError("detection does not satisfy birth_threshold")
    status = TrackStatus.ACTIVE if config.min_hits <= 1 else TrackStatus.TENTATIVE
    track = TrackState(
        track_id=track_id,
        sequence_id=detection.frame.sequence_id,
        coordinate_frame=detection.coordinate_frame,
        category=detection.category,
        history=[detection],
        status=status,
        hits=1,
        misses=0,
        score=float(detection.score),
    )
    return track, LifecycleDecision(track_id, None, status, "BIRTH")


def update_matched_track(
    track: TrackState, detection: Detection3D, config: LifecycleConfig
) -> LifecycleDecision:
    if track.status is TrackStatus.REMOVED:
        raise ValueError("removed tracks cannot be updated")
    if detection.frame.sequence_id != track.sequence_id or detection.coordinate_frame != track.coordinate_frame:
        raise ValueError("matched detection context differs from track")
    previous = track.status
    track.history.append(detection)
    track.hits += 1
    track.misses = 0
    track.score = float(detection.score)
    track.category = detection.category
    track.status = TrackStatus.ACTIVE if track.hits >= config.min_hits else TrackStatus.TENTATIVE
    return LifecycleDecision(track.track_id, previous, track.status, "UPDATE")


def mark_missed(track: TrackState, config: LifecycleConfig) -> LifecycleDecision:
    if track.status is TrackStatus.REMOVED:
        return LifecycleDecision(track.track_id, TrackStatus.REMOVED, TrackStatus.REMOVED, "UNCHANGED")
    previous = track.status
    track.misses += 1
    track.score *= config.lost_score_decay
    track.status = TrackStatus.REMOVED if track.misses > config.max_age else TrackStatus.LOST
    action = "REMOVE" if track.status is TrackStatus.REMOVED else "MISS"
    return LifecycleDecision(track.track_id, previous, track.status, action)


Policy = Union[V2XKITTIPolicy, NuScenesPolicy]


class LifecycleTracker:
    """Dataset-neutral tracker receiving model probabilities and distances."""

    def __init__(
        self,
        class_split: ClassSplit,
        gate_config: GateConfig,
        lifecycle_config: LifecycleConfig,
        policy: Policy,
        distance_weight: float = 0.1,
        score_mode: str = "detection",
    ) -> None:
        self.class_split = class_split
        self.gate_config = gate_config
        self.lifecycle_config = lifecycle_config
        self.policy = policy
        self.distance_weight = distance_weight
        if score_mode not in {"detection", "association", "decayed_detection"}:
            raise ValueError("score_mode is invalid")
        self.score_mode = score_mode
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0

    def process_frame(
        self,
        frame_input: TrackingFrameInput,
        p_yes: Sequence[Sequence[float]],
        distances: Sequence[Sequence[float]],
    ) -> TrackingFrameOutput:
        active_tracks = tuple(
            track for _, track in sorted(self.tracks.items())
            if track.status is not TrackStatus.REMOVED
        )
        detections = frame_input.detections
        expected_shape = (len(active_tracks), len(detections))
        probability_array = np.asarray(p_yes, dtype=np.float64)
        distance_array = np.asarray(distances, dtype=np.float64)
        if probability_array.shape != expected_shape or distance_array.shape != expected_shape:
            raise ValueError("probability and distance matrices do not match frame cardinality")
        gate = build_pair_gate(
            active_tracks, detections, probability_array, distance_array,
            self.gate_config, self.class_split,
        )
        costs = association_cost_matrix(probability_array, distance_array, self.distance_weight)
        assignment = assign_detections(costs, gate.mask)
        decisions: List[LifecycleDecision] = []
        for track_index, detection_index in assignment.matches:
            decisions.append(update_matched_track(
                active_tracks[track_index], detections[detection_index], self.lifecycle_config
            ))
            if self.score_mode == "association":
                active_tracks[track_index].score = float(
                    probability_array[track_index, detection_index]
                )
        for track_index in assignment.unmatched_tracks:
            decisions.append(mark_missed(active_tracks[track_index], self.lifecycle_config))
        for detection_index in assignment.unmatched_detections:
            detection = detections[detection_index]
            if detection.score >= self.lifecycle_config.birth_threshold:
                track, decision = birth_track(self.next_track_id, detection, self.lifecycle_config)
                self.tracks[track.track_id] = track
                self.next_track_id += 1
                decisions.append(decision)
        retained = tuple(
            track for _, track in sorted(self.tracks.items())
            if track.status is not TrackStatus.REMOVED
        )
        emitted = self.policy.select(retained)
        outputs = tuple(track_to_output(track, frame_input.frame.frame_id) for track in emitted)
        return TrackingFrameOutput(frame_input.frame, outputs, assignment, tuple(decisions))
