"""Synthetic/in-memory nuScenes adapter."""

from __future__ import annotations

from typing import Mapping

from nova.preprocessing.common import build_detection
from nova.preprocessing.nuscenes import extract_nuscenes_geometry

from .base import DatasetAdapter
from .types import CoordinateFrame, Detection3D


class NuScenesAdapter(DatasetAdapter):
    name = "nuscenes"
    coordinate_frame = CoordinateFrame.NUSCENES_GLOBAL

    def build_detection(self, record: Mapping[str, object]) -> Detection3D:
        position, size_lwh, yaw, score = extract_nuscenes_geometry(record)
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
