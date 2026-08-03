"""Paper-fidelity negative-mining and positive-jitter tests."""

import pytest

from nova.data.sampling import jitter_positive_detection, select_negative_candidate
from nova.data.types import (
    ClassMembership,
    CoordinateFrame,
    Detection3D,
    FrameMetadata,
    SamplingType,
)


def detection(name, x):
    return Detection3D(
        name,
        FrameMetadata("v2x", "sequence", 2, 2),
        (float(x), 0.0, 0.0),
        (4.0, 2.0, 1.5),
        0.1,
        0.8,
        "car",
        CoordinateFrame.V2X_EGO_LIDAR,
        ClassMembership.BASE,
    )


def candidates():
    return (("track-a", detection("self", 0)), ("track-b", detection("near", 1)), ("track-c", detection("far", 9)))


def test_hard_negative_is_different_identity_and_near_top_k():
    result = select_negative_candidate("track-a", detection("anchor", 0), candidates(), strategy="hard", seed=4, hard_k=1)
    assert result.source_id == "track-b"
    assert result.sampling_type is SamplingType.NEGATIVE_HARD


def test_local_negative_requires_explicit_nonempty_radius():
    result = select_negative_candidate("track-a", detection("anchor", 0), candidates(), strategy="local", seed=4, local_radius_m=2.0)
    assert result.source_id == "track-b"
    with pytest.raises(ValueError, match="empty"):
        select_negative_candidate("track-a", detection("anchor", 0), candidates(), strategy="local", seed=4, local_radius_m=0.5)


def test_random_negative_is_deterministic():
    first = select_negative_candidate("track-a", detection("anchor", 0), candidates(), strategy="random", seed=19)
    second = select_negative_candidate("track-a", detection("anchor", 0), candidates(), strategy="random", seed=19)
    assert first.source_id == second.source_id
    assert first.sampling_type is SamplingType.NEGATIVE_RANDOM


def test_empty_negative_set_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        select_negative_candidate("track-a", detection("anchor", 0), (("track-a", detection("same", 1)),), strategy="hard", seed=1)


def test_positive_jitter_is_deterministic_and_auditable():
    first = jitter_positive_detection(detection("positive", 2), standard_deviation=0.02, seed=7)
    second = jitter_positive_detection(detection("positive", 2), standard_deviation=0.02, seed=7)
    assert first.jittered_geometry_7d == second.jittered_geometry_7d
    assert first.noise_7d == second.noise_7d
    assert first.sampling_type is SamplingType.POSITIVE_JITTERED


def test_zero_jitter_preserves_geometry():
    result = jitter_positive_detection(detection("positive", 2), standard_deviation=0.0, seed=7)
    assert result.original_geometry_7d == result.jittered_geometry_7d


def test_invalid_sampling_parameters_fail_closed():
    with pytest.raises(ValueError):
        select_negative_candidate("track-a", detection("anchor", 0), candidates(), strategy="hard", seed=1, hard_k=0)
    with pytest.raises(ValueError):
        jitter_positive_detection(detection("positive", 2), standard_deviation=-1.0, seed=1)
