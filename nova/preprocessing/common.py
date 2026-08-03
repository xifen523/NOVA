"""Dataset-neutral pure constructors for validated association data."""

from __future__ import annotations

from typing import Mapping, Optional, Tuple

from nova.data.class_split import ClassSplit
from nova.data.types import (
    ClassMembership,
    CoordinateFrame,
    Detection3D,
    FrameMetadata,
)

from .validation import optional_string, require_int, require_string


def build_frame_metadata(
    dataset_name: str, record: Mapping[str, object]
) -> FrameMetadata:
    return FrameMetadata(
        dataset_name=dataset_name,
        sequence_id=require_string(record, "sequence_id"),
        frame_id=require_int(record, "frame_id"),
        timestamp=require_int(record, "timestamp"),
        sample_token=optional_string(record, "sample_token"),
    )


def build_detection(
    dataset_name: str,
    coordinate_frame: CoordinateFrame,
    class_split: ClassSplit,
    record: Mapping[str, object],
    position_m: Tuple[float, float, float],
    size_lwh_m: Tuple[float, float, float],
    yaw_rad: float,
    score: float,
) -> Detection3D:
    category = require_string(record, "category")
    membership: ClassMembership = class_split.classify(category)
    return Detection3D(
        detection_id=require_string(record, "detection_id"),
        frame=build_frame_metadata(dataset_name, record),
        position_m=position_m,
        size_lwh_m=size_lwh_m,
        yaw_rad=yaw_rad,
        score=score,
        category=category,
        coordinate_frame=coordinate_frame,
        class_membership=membership,
    )


def build_semantic_prompt(track_id: str, detection: Detection3D) -> str:
    return (
        "Determine whether candidate <box> {0} in frame {1} belongs to track {2}. "
        "Answer Yes or No."
    ).format(detection.detection_id, detection.frame.frame_id, track_id)


def target_from_expected(expected_match: Optional[bool]) -> Optional[str]:
    if expected_match is None:
        return None
    if not isinstance(expected_match, bool):
        raise TypeError("expected_match must be bool or None")
    return "Yes" if expected_match else "No"
