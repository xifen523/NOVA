"""Synthetic/in-memory V2X adapter."""

from __future__ import annotations

from typing import Mapping

from nova.preprocessing.common import build_detection
from nova.preprocessing.v2x import extract_v2x_geometry

from .base import DatasetAdapter
from .types import CoordinateFrame, Detection3D


class V2XAdapter(DatasetAdapter):
    name = "v2x"
    coordinate_frame = CoordinateFrame.V2X_EGO_LIDAR

    def build_detection(self, record: Mapping[str, object]) -> Detection3D:
        position, size_lwh, yaw, score = extract_v2x_geometry(record)
        return build_detection(
            self.name,
            self.coordinate_frame,
            self.class_split,
            record,
            position,
            size_lwh,
            yaw,
            score,
        )
