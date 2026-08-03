"""Pure field mapping for already calibrated KITTI lidar records."""

from __future__ import annotations

from typing import Mapping, Tuple

from .validation import reject_unknown_fields, require_float, require_string, require_vector3


ALLOWED_FIELDS = {
    "coordinate_frame", "sequence_id", "frame_id", "timestamp", "sample_token",
    "detection_id", "track_id", "category", "score", "lidar_xyz", "lwh", "lidar_yaw",
}


def extract_kitti_geometry(
    record: Mapping[str, object]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], float, float]:
    reject_unknown_fields(record, ALLOWED_FIELDS)
    if require_string(record, "coordinate_frame") != "kitti_lidar":
        raise ValueError("KITTI records must declare kitti_lidar")
    return (
        require_vector3(record, "lidar_xyz"),
        require_vector3(record, "lwh"),
        require_float(record, "lidar_yaw"),
        require_float(record, "score"),
    )
