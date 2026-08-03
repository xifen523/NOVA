"""Typed tracking primitives with explicit dataset emission policies."""

from .assignment import assign_detections
from .costs import association_cost, association_cost_matrix
from .gates import GateConfig, GateDecision, build_pair_gate, class_compatible
from .exporters import format_tracking_document
from .lifecycle import LifecycleTracker, birth_track, mark_missed, update_matched_track
from .policies import LifecycleConfig, NuScenesPolicy, V2XKITTIPolicy
from .types import (
    AssignmentResult,
    AssociationScore,
    LifecycleDecision,
    TrackState,
    TrackStatus,
    TrackingFrameInput,
    TrackingFrameOutput,
)

__all__ = [
    "AssignmentResult", "AssociationScore", "GateConfig", "GateDecision",
    "LifecycleConfig", "LifecycleDecision", "LifecycleTracker", "NuScenesPolicy",
    "TrackState", "TrackStatus", "TrackingFrameInput", "TrackingFrameOutput",
    "V2XKITTIPolicy", "assign_detections", "association_cost",
    "association_cost_matrix", "birth_track", "build_pair_gate", "mark_missed",
    "class_compatible", "format_tracking_document", "update_matched_track",
]
