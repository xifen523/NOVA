import unittest

import torch

from nova.data import ClassSplit, Detection3D, V2XAdapter
from nova.preprocessing.geometry import build_geometry_9d, build_geometry_batch


def record(frame_id, timestamp, position):
    return {
        "coordinate_frame": "v2x_ego_lidar", "sequence_id": "synthetic-sequence",
        "frame_id": frame_id, "timestamp": timestamp,
        "detection_id": "synthetic-det-{0}".format(frame_id),
        "track_id": "synthetic-track", "category": "synthetic_base", "score": 0.75,
        "ego_xyz": position, "lwh": [4.0, 2.0, 1.5], "ego_yaw": 0.25,
    }


class GeometryFeatureTests(unittest.TestCase):
    def setUp(self):
        split = ClassSplit(("synthetic_base",), (), mapping_status="SYNTHETIC_EXAMPLE")
        self.adapter = V2XAdapter(split)
        history = self.adapter.build_history(
            "synthetic-track", [record(1, 1, [0.0, 0.0, 0.0])]
        )
        self.sample = self.adapter.build_association_sample(
            history, record(2, 2, [10.0, -20.0, 5.0])
        )

    def test_frozen_order_and_multi_box_shape(self):
        geometry = self.adapter.build_geometry_features(self.sample)
        expected = torch.tensor([1.0, -2.0, 0.5, 4.0, 2.0, 1.5, 12.0, 0.25, 0.75])
        self.assertEqual(geometry.shape, (2, 9))
        self.assertTrue(torch.equal(geometry[-1], expected))

    def test_batch_shape_and_determinism(self):
        first = build_geometry_batch([self.sample, self.sample])
        second = build_geometry_batch([self.sample, self.sample])
        self.assertEqual(first.shape, (4, 9))
        self.assertTrue(torch.equal(first, second))

    def test_wrong_type_and_empty_batch_are_rejected(self):
        with self.assertRaises(TypeError):
            build_geometry_9d("not-a-detection")
        with self.assertRaises(ValueError):
            build_geometry_batch([])

    def test_contract_rejects_nonfinite_before_geometry(self):
        kwargs = dict(self.sample.candidate.detection.__dict__)
        kwargs["yaw_rad"] = float("inf")
        with self.assertRaises(ValueError):
            Detection3D(**kwargs)


if __name__ == "__main__":
    unittest.main()
