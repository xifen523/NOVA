"""Pure field mapping for already transformed V2X ego-lidar records."""

from __future__ import annotations

from typing import Mapping, Tuple

from .validation import reject_unknown_fields, require_float, require_string, require_vector3


ALLOWED_FIELDS = {
    "coordinate_frame", "sequence_id", "frame_id", "timestamp", "sample_token",
    "detection_id", "track_id", "category", "score", "ego_xyz", "lwh", "ego_yaw",
}


def extract_v2x_geometry(
    record: Mapping[str, object]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float]:
    reject_unknown_fields(record, ALLOWED_FIELDS)
    if require_string(record, "coordinate_frame") != "v2x_ego_lidar":
        raise ValueError("V2X records must declare v2x_ego_lidar")
    return (
        require_vector3(record, "ego_xyz"),
        require_vector3(record, "lwh"),
        require_float(record, "ego_yaw"),
        require_float(record, "score"),
    )
