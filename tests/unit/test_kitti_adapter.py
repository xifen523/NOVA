import unittest

from nova.data import ClassSplit, CoordinateFrame, KITTIAdapter, TrajectoryHistory


def record(frame_id=1, timestamp=1, coordinate_frame="kitti_lidar"):
    return {
        "coordinate_frame": coordinate_frame, "sequence_id": "synthetic-kitti-sequence",
        "frame_id": frame_id, "timestamp": timestamp,
        "detection_id": "synthetic-kitti-{0}".format(frame_id), "track_id": "synthetic-track",
        "category": "synthetic_object", "score": 0.7,
        "lidar_xyz": [20.0, 3.0, -1.0], "lwh": [3.0, 2.0, 1.0], "lidar_yaw": -0.2,
    }


class KITTIAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = KITTIAdapter(
            ClassSplit(("synthetic_object",), (), mapping_status="SYNTHETIC_EXAMPLE")
        )

    def test_record_history_and_geometry(self):
        detection = self.adapter.build_detection(record())
        self.assertEqual(detection.coordinate_frame, CoordinateFrame.KITTI_LIDAR)
        history = self.adapter.build_history("synthetic-track", [record()])
        self.assertIsInstance(history, TrajectoryHistory)
        sample = self.adapter.build_association_sample(history, record(2, 2))
        self.assertEqual(self.adapter.build_geometry_features(sample).shape[-1], 9)

    def test_wrong_coordinate_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.build_detection(record(coordinate_frame="v2x_ego_lidar"))

    def test_out_of_order_candidate_is_rejected(self):
        history = self.adapter.build_history("synthetic-track", [record(2, 2)])
        with self.assertRaises(ValueError):
            self.adapter.build_association_sample(history, record(1, 1))


if __name__ == "__main__":
    unittest.main()
