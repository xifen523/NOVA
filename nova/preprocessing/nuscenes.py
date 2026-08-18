"""Pure field mapping for nuScenes global records without a devkit."""

from __future__ import annotations

from typing import Mapping, Tuple

from .validation import reject_unknown_fields, require_float, require_string, require_vector3


ALLOWED_FIELDS = {
    "coordinate_frame", "sequence_id", "frame_id", "timestamp", "sample_token",
    "detection_id", "track_id", "category", "score", "translation", "size_wlh", "yaw",
}


def extract_nuscenes_geometry(
    record: Mapping[str, object]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float]:
    reject_unknown_fields(record, ALLOWED_FIELDS)
    if require_string(record, "coordinate_frame") != "nuscenes_global":
        raise ValueError("nuScenes records must declare nuscenes_global")
    width, length, height = require_vector3(record, "size_wlh")
    return (
        require_vector3(record, "translation"),
        (length, width, height),
        require_float(record, "yaw"),
        require_float(record, "score"),
    )
