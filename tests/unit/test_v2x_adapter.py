import copy
import unittest

from nova.data import ClassMembership, ClassSplit, Detection3D, TrajectoryHistory, V2XAdapter


def record(frame_id=1, timestamp=1, coordinate_frame="v2x_ego_lidar"):
    return {
        "coordinate_frame": coordinate_frame, "sequence_id": "synthetic-v2x-sequence",
        "frame_id": frame_id, "timestamp": timestamp,
        "detection_id": "synthetic-v2x-{0}".format(frame_id), "track_id": "synthetic-track",
        "category": "synthetic_auto", "score": 0.8,
        "ego_xyz": [10.0, 2.0, 1.0], "lwh": [4.0, 2.0, 1.5], "ego_yaw": 0.1,
    }


class V2XAdapterTests(unittest.TestCase):
    def setUp(self):
        split = ClassSplit(
            ("synthetic_vehicle",), (), (("synthetic_auto", "synthetic_vehicle"),),
            mapping_status="SYNTHETIC_EXAMPLE",
        )
        self.adapter = V2XAdapter(split)

    def test_record_history_sample_and_no_mutation(self):
        source = record()
        before = copy.deepcopy(source)
        detection = self.adapter.build_detection(source)
        self.assertIsInstance(detection, Detection3D)
        self.assertEqual(detection.class_membership, ClassMembership.BASE)
        self.assertEqual(source, before)
        history = self.adapter.build_history("synthetic-track", [source])
        self.assertIsInstance(history, TrajectoryHistory)
        sample = self.adapter.build_association_sample(history, record(2, 2))
        self.assertEqual(self.adapter.build_geometry_features(sample).shape, (2, 9))

    def test_wrong_coordinate_frame_is_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.build_detection(record(coordinate_frame="kitti_lidar"))

    def test_foreign_dataset_metadata_is_rejected(self):
        history = self.adapter.build_history("synthetic-track", [record()])
        foreign = record(2, 2)
        foreign["sequence_id"] = "different-sequence"
        with self.assertRaises(ValueError):
            self.adapter.build_association_sample(history, foreign)


if __name__ == "__main__":
    unittest.main()
