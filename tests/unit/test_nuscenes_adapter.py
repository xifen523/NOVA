import unittest

from nova.data import ClassSplit, CoordinateFrame, NuScenesAdapter


def record(frame_id=1, timestamp=1, coordinate_frame="nuscenes_global"):
    return {
        "coordinate_frame": coordinate_frame, "sequence_id": "invented-nusc-scene",
        "frame_id": frame_id, "timestamp": timestamp,
        "sample_token": "invented-token-{0}".format(frame_id),
        "detection_id": "synthetic-nusc-{0}".format(frame_id), "track_id": "synthetic-track",
        "category": "synthetic_object", "score": 0.6,
        "translation": [30.0, -4.0, 2.0], "size_wlh": [2.0, 4.0, 1.5], "yaw": 1.0,
    }


class NuScenesAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = NuScenesAdapter(
            ClassSplit(("synthetic_object",), (), mapping_status="SYNTHETIC_EXAMPLE")
        )

    def test_native_wlh_is_reordered_and_geometry_is_nine_dimensional(self):
        detection = self.adapter.build_detection(record())
        self.assertEqual(detection.coordinate_frame, CoordinateFrame.NUSCENES_GLOBAL)
        self.assertEqual(detection.size_lwh_m, (4.0, 2.0, 1.5))
        history = self.adapter.build_history("synthetic-track", [record()])
        sample = self.adapter.build_association_sample(history, record(2, 2))
        self.assertEqual(self.adapter.build_geometry_features(sample).shape, (2, 9))

    def test_wrong_coordinate_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.build_detection(record(coordinate_frame="kitti_lidar"))

    def test_unknown_fields_are_rejected(self):
        source = record()
        source["untrusted_field"] = "value"
        with self.assertRaises(ValueError):
            self.adapter.build_detection(source)


if __name__ == "__main__":
    unittest.main()
