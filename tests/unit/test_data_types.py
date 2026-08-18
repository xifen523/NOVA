import unittest

import torch

from nova.data.types import (
    AssociationCandidate,
    AssociationSample,
    ClassMembership,
    CoordinateFrame,
    Detection3D,
    FrameMetadata,
    PreparedBatch,
    TrackObservation,
    TrajectoryHistory,
)


def detection(frame_id=1, timestamp=1, position=(1.0, 2.0, 3.0)):
    return Detection3D(
        detection_id="synthetic-det-{0}".format(frame_id),
        frame=FrameMetadata("v2x", "synthetic-sequence", frame_id, timestamp),
        position_m=position,
        size_lwh_m=(4.0, 2.0, 1.0),
        yaw_rad=0.25,
        score=0.75,
        category="synthetic_base",
        coordinate_frame=CoordinateFrame.V2X_EGO_LIDAR,
        class_membership=ClassMembership.BASE,
    )


class DataTypeTests(unittest.TestCase):
    def test_missing_required_field_is_rejected(self):
        with self.assertRaises(TypeError):
            FrameMetadata(dataset_name="v2x", sequence_id="s", frame_id=1)

    def test_nan_inf_and_invalid_dimension_are_rejected(self):
        with self.assertRaises(ValueError):
            detection(position=(float("nan"), 0.0, 0.0))
        with self.assertRaises(TypeError):
            detection(position=(1.0, 2.0))
        with self.assertRaises(ValueError):
            Detection3D(
                detection_id="d", frame=FrameMetadata("v2x", "s", 1, 1),
                position_m=(1.0, 2.0, 3.0), size_lwh_m=(1.0, float("inf"), 1.0),
                yaw_rad=0.0, score=0.5, category="c",
                coordinate_frame=CoordinateFrame.V2X_EGO_LIDAR,
                class_membership=ClassMembership.UNKNOWN,
            )

    def test_empty_history_is_valid_for_history_length_zero_association(self):
        history = TrajectoryHistory("synthetic-track", ())
        sample = AssociationSample(
            history=history,
            candidate=AssociationCandidate(detection()),
            semantic_prompt="prompt",
            target_token=None,
            dataset_name="v2x",
            coordinate_frame=CoordinateFrame.V2X_EGO_LIDAR,
        )
        self.assertEqual(sample.history.observations, ())

    def test_time_order_is_strict(self):
        observations = (
            TrackObservation("synthetic-track", detection(2, 2)),
            TrackObservation("synthetic-track", detection(1, 1)),
        )
        with self.assertRaises(ValueError):
            TrajectoryHistory("synthetic-track", observations)

    def test_prepared_batch_shape_and_finite_contract(self):
        first = detection()
        history = TrajectoryHistory(
            "synthetic-track", (TrackObservation("synthetic-track", first),)
        )
        sample = AssociationSample(
            history=history,
            candidate=AssociationCandidate(detection(2, 2)),
            semantic_prompt="prompt",
            target_token="Yes",
            dataset_name="v2x",
            coordinate_frame=CoordinateFrame.V2X_EGO_LIDAR,
        )
        PreparedBatch((sample,), torch.zeros(1, 9))
        with self.assertRaises(ValueError):
            PreparedBatch((sample,), torch.zeros(1, 8))


if __name__ == "__main__":
    unittest.main()
