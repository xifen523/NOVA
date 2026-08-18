import unittest

from nova.data.types import (
    ClassMembership, CoordinateFrame, Detection3D, FrameMetadata,
)
from nova.tracking.lifecycle import birth_track, mark_missed, update_matched_track
from nova.tracking.policies import LifecycleConfig, NuScenesPolicy, V2XKITTIPolicy
from nova.tracking.types import TrackStatus


def detection(frame_id=1, score=0.8):
    return Detection3D(
        "det-{0}".format(frame_id), FrameMetadata("v2x", "seq", frame_id, frame_id),
        (float(frame_id), 0.0, 0.0), (2.0, 1.0, 1.0), 0.0, score, "base",
        CoordinateFrame.V2X_EGO_LIDAR, ClassMembership.BASE,
    )


class LifecycleTests(unittest.TestCase):
    def test_birth_min_hits_confirmation_miss_and_removal(self):
        config = LifecycleConfig(max_age=1, min_hits=2, birth_threshold=0.5)
        track, birth = birth_track(0, detection(), config)
        self.assertEqual(birth.action, "BIRTH")
        self.assertEqual(track.status, TrackStatus.TENTATIVE)
        update_matched_track(track, detection(2), config)
        self.assertEqual(track.status, TrackStatus.ACTIVE)
        mark_missed(track, config)
        self.assertEqual(track.status, TrackStatus.LOST)
        mark_missed(track, config)
        self.assertEqual(track.status, TrackStatus.REMOVED)

    def test_lost_score_decay(self):
        config = LifecycleConfig(max_age=3, lost_score_decay=0.5)
        track, _ = birth_track(0, detection(score=0.8), config)
        mark_missed(track, config)
        self.assertAlmostEqual(track.score, 0.4)

    def test_dataset_emission_policies_are_explicit(self):
        config = LifecycleConfig(max_age=3)
        track, _ = birth_track(0, detection(), config)
        self.assertEqual(V2XKITTIPolicy().select((track,)), (track,))
        mark_missed(track, config)
        self.assertEqual(V2XKITTIPolicy().select((track,)), ())
        self.assertEqual(NuScenesPolicy(emit_lost=False).select((track,)), ())
        self.assertEqual(NuScenesPolicy(emit_lost=True).select((track,)), (track,))


if __name__ == "__main__":
    unittest.main()
