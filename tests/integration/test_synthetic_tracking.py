import copy
import unittest

import numpy as np

from nova.data import ClassSplit
from nova.data.types import (
    ClassMembership, CoordinateFrame, Detection3D, FrameMetadata,
)
from nova.tracking import GateConfig, LifecycleConfig, LifecycleTracker, V2XKITTIPolicy
from nova.tracking.types import TrackingFrameInput


def frame(frame_id, with_detection=True):
    metadata = FrameMetadata("v2x", "synthetic-sequence", frame_id, frame_id)
    if not with_detection:
        return TrackingFrameInput(metadata, ())
    detection = Detection3D(
        "synthetic-det-{0}".format(frame_id), metadata, (float(frame_id), 0.0, 0.0),
        (2.0, 1.0, 1.0), 0.0, 0.9, "synthetic_base",
        CoordinateFrame.V2X_EGO_LIDAR, ClassMembership.BASE,
    )
    return TrackingFrameInput(metadata, (detection,))


class SyntheticTrackingIntegrationTests(unittest.TestCase):
    def test_three_frames_stable_id_and_no_input_mutation(self):
        tracker = LifecycleTracker(
            ClassSplit(("synthetic_base",), (), mapping_status="SYNTHETIC_EXAMPLE"),
            GateConfig(0.1, 5.0, 0.5, True), LifecycleConfig(max_age=2),
            V2XKITTIPolicy(),
        )
        first = frame(1)
        second = frame(2)
        third = frame(3, with_detection=False)
        second_before = copy.deepcopy(second)
        output1 = tracker.process_frame(first, np.empty((0, 1)), np.empty((0, 1)))
        output2 = tracker.process_frame(second, [[0.9]], [[1.0]])
        output3 = tracker.process_frame(third, np.empty((1, 0)), np.empty((1, 0)))
        self.assertEqual(output1.tracks[0].track_id, 0)
        self.assertEqual(output2.tracks[0].track_id, 0)
        self.assertEqual(output3.tracks, ())
        self.assertEqual(second, second_before)


if __name__ == "__main__":
    unittest.main()
