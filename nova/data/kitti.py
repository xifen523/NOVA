"""Synthetic/in-memory KITTI adapter."""

from __future__ import annotations

from typing import Mapping

from nova.preprocessing.common import build_detection
from nova.preprocessing.kitti import extract_kitti_geometry

from .base import DatasetAdapter
from .types import CoordinateFrame, Detection3D


class KITTIAdapter(DatasetAdapter):
    name = "kitti"
    coordinate_frame = CoordinateFrame.KITTI_LIDAR

    def build_detection(self, record: Mapping[str, object]) -> Detection3D:
        position, size_lwh, yaw, score = extract_kitti_geometry(record)
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
