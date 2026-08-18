import unittest

from nova.data.types import (
    ClassMembership, CoordinateFrame, Detection3D, FrameMetadata,
)
from nova.tracking.lifecycle import birth_track
from nova.tracking.output import TrackOutput, track_to_output
from nova.tracking.policies import LifecycleConfig


class TrackingOutputTests(unittest.TestCase):
    def test_output_contains_only_in_memory_contract_fields(self):
        detection = Detection3D(
            "det", FrameMetadata("v2x", "seq", 3, 3), (1.0, 2.0, 3.0),
            (4.0, 2.0, 1.0), 0.25, 0.75, "synthetic",
            CoordinateFrame.V2X_EGO_LIDAR, ClassMembership.UNKNOWN,
        )
        track, _ = birth_track(7, detection, LifecycleConfig())
        output = track_to_output(track, 3)
        self.assertIsInstance(output, TrackOutput)
        self.assertEqual(output.track_id, 7)
        self.assertEqual(output.box_xyz_lwh_yaw, (1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.25))
        self.assertFalse(hasattr(output, "submission_path"))


if __name__ == "__main__":
    unittest.main()
