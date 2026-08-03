"""Dataset-shaped prepared intermediates; official evaluators remain external."""

from __future__ import annotations

from typing import Mapping


def format_tracking_document(document: Mapping[str, object], output_format: str) -> dict:
    if output_format == "generic":
        return dict(document)
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError("generic tracking frames are required")
    if output_format == "v2x":
        return {
            "schema": "nova.v2x-prepared-intermediate.v1",
            "status": "PREPARED_INTERMEDIATE_OUTPUT",
            "frames": frames,
        }
    if output_format == "kitti":
        rows = []
        for frame in frames:
            for track in frame.get("tracks", []):
                rows.append({
                    "sequence_id": frame["sequence_id"],
                    "frame_id": frame["frame_id"],
                    "track_id": track["track_id"],
                    "category": track["category"],
                    "box_xyz_lwh_yaw": track["box_xyz_lwh_yaw"],
                    "score": track["score"],
                })
        return {
            "schema": "nova.kitti-prepared-intermediate.v1",
            "status": "PREPARED_INTERMEDIATE_OUTPUT",
            "rows": rows,
        }
    if output_format == "nuscenes":
        results = {}
        for frame in frames:
            token = frame.get("sample_token") or "{0}:{1}".format(
                frame["sequence_id"], frame["frame_id"]
            )
            results[token] = [
                {
                    "tracking_id": str(track["track_id"]),
                    "tracking_name": track["category"],
                    "box_xyz_lwh_yaw": track["box_xyz_lwh_yaw"],
                    "tracking_score": track["score"],
                }
                for track in frame.get("tracks", [])
            ]
        return {
            "schema": "nova.nuscenes-prepared-intermediate.v1",
            "status": "PREPARED_INTERMEDIATE_OUTPUT",
            "results": results,
        }
    raise ValueError("output_format must be generic, v2x, kitti, or nuscenes")
