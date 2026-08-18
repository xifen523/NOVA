"""Paper-fidelity dataset tracking-export tests."""

import pytest

from nova.tracking.exporters import format_tracking_document


DOCUMENT = {
    "schema": "nova.tracking-output.v1",
    "frames": [{
        "sequence_id": "sequence",
        "frame_id": 1,
        "tracks": [{
            "track_id": 2,
            "category": "car",
            "box_xyz_lwh_yaw": [0, 0, 0, 4, 2, 1, 0],
            "score": 0.8,
        }],
    }],
}


@pytest.mark.parametrize("name", ["v2x", "kitti", "nuscenes"])
def test_dataset_exporters_are_explicit_prepared_intermediates(name):
    exported = format_tracking_document(DOCUMENT, name)
    assert exported["status"] == "PREPARED_INTERMEDIATE_OUTPUT"
    assert "prepared-intermediate" in exported["schema"]


def test_unknown_export_format_is_rejected():
    with pytest.raises(ValueError):
        format_tracking_document(DOCUMENT, "official-but-unknown")
