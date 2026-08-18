import unittest

from nova.data import ClassSplit
from nova.data.types import (
    ClassMembership, CoordinateFrame, Detection3D, FrameMetadata,
)
from nova.tracking.gates import GateConfig, build_pair_gate
from nova.tracking.types import TrackState, TrackStatus


def detection(category="base_a", score=0.5):
    return Detection3D(
        "det", FrameMetadata("v2x", "seq", 2, 2), (1.0, 0.0, 0.0),
        (2.0, 1.0, 1.0), 0.0, score, category,
        CoordinateFrame.V2X_EGO_LIDAR, ClassMembership.UNKNOWN,
    )


def track(category="base_a"):
    return TrackState(
        0, "seq", CoordinateFrame.V2X_EGO_LIDAR, category,
        [detection(category)], TrackStatus.ACTIVE, 1, 0, 0.5,
    )


class TrackingGateTests(unittest.TestCase):
    def setUp(self):
        self.split = ClassSplit(
            ("base_a", "base_b"), ("novel_a",),
            (("alias_a", "base_a"),), mapping_status="SYNTHETIC_EXAMPLE",
        )

    def test_threshold_boundaries_are_inclusive(self):
        decision = build_pair_gate(
            [track()], [detection(score=0.5)], [[0.5]], [[2.0]],
            GateConfig(0.5, 2.0, 0.5, False), self.split,
        )
        self.assertTrue(decision.mask[0, 0])
        self.assertEqual(decision.rejection_reasons[0][0], ())

    def test_each_gate_reports_rejection_reason_and_can_be_disabled(self):
        decision = build_pair_gate(
            [track()], [detection(score=0.4)], [[0.4]], [[2.1]],
            GateConfig(0.5, 2.0, 0.5, False), self.split,
        )
        self.assertEqual(
            decision.rejection_reasons[0][0],
            ("DETECTION_THRESHOLD", "DISTANCE_GATE", "YES_THRESHOLD"),
        )
        disabled = build_pair_gate(
            [track()], [detection(score=0.0)], [[0.0]], [[100.0]],
            GateConfig(None, None, None, False), self.split,
        )
        self.assertTrue(disabled.mask[0, 0])

    def test_class_gate_uses_exact_canonical_or_alias_equality(self):
        base = build_pair_gate(
            [track("base_a")], [detection("base_b")], [[1.0]], [[0.0]],
            GateConfig(None, None, None, True), self.split,
        )
        novel = build_pair_gate(
            [track("base_a")], [detection("novel_a")], [[1.0]], [[0.0]],
            GateConfig(None, None, None, True), self.split,
        )
        unknown_same = build_pair_gate(
            [track("mystery")], [detection("mystery")], [[1.0]], [[0.0]],
            GateConfig(None, None, None, True), self.split,
        )
        unknown_other = build_pair_gate(
            [track("mystery")], [detection("other")], [[1.0]], [[0.0]],
            GateConfig(None, None, None, True), self.split,
        )
        alias = build_pair_gate(
            [track("base_a")], [detection("alias_a")], [[1.0]], [[0.0]],
            GateConfig(None, None, None, True), self.split,
        )
        self.assertFalse(base.mask[0, 0])
        self.assertTrue(alias.mask[0, 0])
        self.assertFalse(novel.mask[0, 0])
        self.assertTrue(unknown_same.mask[0, 0])
        self.assertFalse(unknown_other.mask[0, 0])


if __name__ == "__main__":
    unittest.main()
