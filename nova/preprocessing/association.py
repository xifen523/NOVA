"""Paper-aligned association prompting shared by training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from nova.data.class_split import ClassSplit
from nova.data.types import (
    AssociationCandidate,
    ClassMembership,
    Detection3D,
    SerializedAssociationContext,
    TrajectoryHistory,
)

from .geometry import build_geometry_9d


@dataclass(frozen=True)
class ClassPromptPolicy:
    mode: str = "hybrid"
    novel_label: str = "Unknown"
    mask_novel_classes: bool = True
    preserve_base_classes: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"hybrid", "raw"}:
            raise ValueError("prompt policy mode must be hybrid or raw")
        if not isinstance(self.novel_label, str) or not self.novel_label.strip():
            raise ValueError("novel_label must be a non-empty string")
        if not isinstance(self.mask_novel_classes, bool) or not isinstance(self.preserve_base_classes, bool):
            raise TypeError("prompt policy switches must be Boolean")


def prompt_category(
    detection: Detection3D,
    class_split: ClassSplit,
    policy: ClassPromptPolicy,
) -> tuple[str, Dict[str, str]]:
    """Return the displayed category and auditable original/canonical metadata."""

    canonical = class_split.canonicalize(detection.category)
    membership = class_split.classify(detection.category)
    if policy.mode == "raw":
        displayed = canonical
    elif membership is ClassMembership.BASE and policy.preserve_base_classes:
        displayed = canonical
    elif membership is ClassMembership.NOVEL and policy.mask_novel_classes:
        displayed = policy.novel_label
    elif membership is ClassMembership.UNKNOWN:
        displayed = policy.novel_label
    else:
        displayed = canonical
    return displayed, {
        "original_category": detection.category,
        "canonical_category": canonical,
        "class_membership": membership.value,
        "prompt_category": displayed,
    }


def serialize_association_context(
    history: TrajectoryHistory,
    candidate: AssociationCandidate,
    *,
    history_length: int,
    class_prompt_policy: ClassPromptPolicy,
    class_split: ClassSplit,
    temporal_order: str = "oldest_to_newest",
) -> SerializedAssociationContext:
    """Serialize recent history boxes followed by one candidate box.

    No synthetic history is fabricated. A zero history length therefore produces
    exactly one ``<box>`` and one geometry row for the candidate.
    """

    if not isinstance(history, TrajectoryHistory):
        raise TypeError("history must be TrajectoryHistory")
    if not isinstance(candidate, AssociationCandidate):
        raise TypeError("candidate must be AssociationCandidate")
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 0:
        raise ValueError("history_length must be a non-negative integer")
    if temporal_order != "oldest_to_newest":
        raise ValueError("temporal_order must be oldest_to_newest")
    selected = history.observations[-history_length:] if history_length else ()
    candidate_detection = candidate.detection
    if selected:
        last = selected[-1].detection
        if (
            candidate_detection.frame.sequence_id != last.frame.sequence_id
            or (candidate_detection.frame.timestamp, candidate_detection.frame.frame_id)
            <= (last.frame.timestamp, last.frame.frame_id)
        ):
            raise ValueError("candidate must occur strictly after selected history")

    lines = ["Task: 3D Association.", "History:"]
    geometries = []
    roles = []
    timestamps = []
    prompt_boxes = []
    if not selected:
        lines.append("(none)")
    for index, observation in enumerate(selected):
        detection = observation.detection
        displayed, metadata = prompt_category(detection, class_split, class_prompt_policy)
        lines.append(
            "<box> Role:history_{0} Class:{1} Frame:{2}".format(
                index, displayed, detection.frame.frame_id
            )
        )
        geometries.append(tuple(float(value) for value in build_geometry_9d(detection).tolist()))
        roles.append("history_{0}".format(index))
        timestamps.append(detection.frame.timestamp)
        prompt_boxes.append(metadata)

    displayed, candidate_metadata = prompt_category(
        candidate_detection, class_split, class_prompt_policy
    )
    lines.extend((
        "Candidate:",
        "<box> Role:candidate Class:{0} Frame:{1}".format(
            displayed, candidate_detection.frame.frame_id
        ),
        "Question: Is this the same object?",
        "Answer:",
    ))
    geometries.append(tuple(float(value) for value in build_geometry_9d(candidate_detection).tolist()))
    roles.append("candidate")
    timestamps.append(candidate_detection.frame.timestamp)
    prompt_boxes.append(candidate_metadata)
    return SerializedAssociationContext(
        prompt="\n".join(lines),
        geometry_features=tuple(geometries),
        box_count=len(geometries),
        box_roles=tuple(roles),
        token_metadata={
            "temporal_order": temporal_order,
            "history_requested": history_length,
            "history_serialized": len(selected),
            "boxes": tuple(prompt_boxes),
        },
        timestamps=tuple(timestamps),
    )
