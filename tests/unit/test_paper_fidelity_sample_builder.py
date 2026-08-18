"""Paper-fidelity typed sample-builder tests."""

import pytest

from nova.data import AssociationSampleBuilder, ClassSplit, V2XAdapter, allocate_sampling_counts
from nova.preprocessing.association import ClassPromptPolicy


def record(frame, track, x):
    return {
        "coordinate_frame": "v2x_ego_lidar", "sequence_id": "sequence",
        "frame_id": frame, "timestamp": frame, "detection_id": "det-{0}-{1}".format(frame, track),
        "track_id": track, "category": "car", "score": 0.8,
        "ego_xyz": [x, 0.0, 0.0], "lwh": [4.0, 2.0, 1.5], "ego_yaw": 0.0,
    }


def builder():
    adapter = V2XAdapter(ClassSplit(("car",), (), mapping_status="SYNTHETIC_EXAMPLE"))
    history = adapter.build_history("track-a", [record(1, "track-a", 0.0)])
    return adapter, AssociationSampleBuilder(
        adapter, history, history_length=3, prompt_policy=ClassPromptPolicy(), seed=7
    )


def test_sampling_count_allocation_is_exact_and_deterministic():
    ratios = {"positive": 0.5, "hard": 0.3, "random": 0.2}
    assert allocate_sampling_counts(10, ratios) == {"positive": 5, "hard": 3, "random": 2}
    assert sum(allocate_sampling_counts(7, ratios).values()) == 7


def test_sample_builder_integrates_positive_jitter_and_hard_negative():
    adapter, sample_builder = builder()
    positive = adapter.build_detection(record(2, "track-a", 1.0))
    jittered = sample_builder.positive_jittered(
        positive,
        source_id="track-a",
        standard_deviation=0.02,
        auxiliary_quality_after_jitter=0.9,
    )
    negative = sample_builder.negative(
        positive,
        (("track-b", adapter.build_detection(record(2, "track-b", 1.5))),),
        strategy="hard",
        auxiliary_quality=0.0,
    )
    assert jittered.target_token == "Yes" and "jitter" in jittered.metadata
    assert negative.target_token == "No"
    assert negative.candidate.candidate_source_id == "track-b"


def test_jittered_and_negative_quality_must_be_explicit():
    adapter, sample_builder = builder()
    positive = adapter.build_detection(record(2, "track-a", 1.0))
    with pytest.raises(ValueError, match="GT-derived"):
        sample_builder.positive_jittered(
            positive, source_id="track-a", standard_deviation=0.02,
            auxiliary_quality_after_jitter=None,
        )
    with pytest.raises(ValueError, match="quality"):
        sample_builder.negative(
            positive,
            (("track-b", adapter.build_detection(record(2, "track-b", 1.5))),),
            strategy="hard", auxiliary_quality=None,
        )
