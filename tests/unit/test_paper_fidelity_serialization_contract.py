"""Paper-fidelity history serialization and data-contract tests."""

import copy
import json
from pathlib import Path

import pytest

from nova.data import (
    AssociationMode,
    ClassSplit,
    SamplingType,
    V2XAdapter,
)
from nova.data.prepared import PreparedDataError, prepare_associations, validate_association_row
from nova.preprocessing.association import ClassPromptPolicy, serialize_association_context


FIXTURE = Path("tests/fixtures/synthetic_v2x.json")


def record(frame, category="synthetic_vehicle", track="track-a"):
    return {
        "coordinate_frame": "v2x_ego_lidar",
        "sequence_id": "sequence",
        "frame_id": frame,
        "timestamp": frame,
        "detection_id": "det-{0}".format(frame),
        "track_id": track,
        "category": category,
        "score": 0.8,
        "ego_xyz": [float(frame), 0.0, 0.0],
        "lwh": [4.0, 2.0, 1.5],
        "ego_yaw": 0.1,
    }


@pytest.fixture
def adapter():
    return V2XAdapter(ClassSplit(
        ("synthetic_vehicle",),
        ("synthetic_cart",),
        (("synthetic_auto", "synthetic_vehicle"),),
        mapping_status="SYNTHETIC_EXAMPLE",
    ))


@pytest.mark.parametrize("length,expected", [(0, 1), (1, 2), (3, 4), (5, 6)])
def test_history_lengths_change_prompt_geometry_and_box_count(adapter, length, expected):
    history = adapter.build_history("track-a", [record(index) for index in range(1, 6)])
    candidate_record = record(6)
    candidate_record["candidate_source_id"] = "track-a"
    sample = adapter.build_association_sample(
        history,
        candidate_record,
        mode=AssociationMode.INFERENCE,
        history_length=length,
    )
    assert sample.context.box_count == expected
    assert sample.context.prompt.count("<box>") == expected
    assert len(sample.context.geometry_features) == expected
    assert sample.context.box_roles[-1] == "candidate"


def test_history_longer_than_limit_keeps_recent_oldest_to_newest(adapter):
    history = adapter.build_history("track-a", [record(index) for index in range(1, 6)])
    candidate_record = record(6)
    sample = adapter.build_association_sample(history, candidate_record, history_length=3)
    assert sample.context.timestamps == (3, 4, 5, 6)
    assert sample.context.box_roles == ("history_0", "history_1", "history_2", "candidate")


def test_training_requires_explicit_label_source_and_sampling(adapter):
    history = adapter.build_history("track-a", [record(1)])
    with pytest.raises(ValueError, match="expected_match"):
        adapter.build_association_sample(history, record(2), mode=AssociationMode.TRAINING)


def test_negative_candidate_may_have_different_identity(adapter):
    history = adapter.build_history("track-a", [record(1)])
    negative = record(2, track="track-b")
    sample = adapter.build_association_sample(
        history,
        negative,
        expected_match=False,
        auxiliary_quality=0.0,
        candidate_source_id="track-b",
        mode=AssociationMode.TRAINING,
        sampling_type=SamplingType.NEGATIVE_HARD,
    )
    assert sample.target_token == "No"
    assert sample.candidate.candidate_source_id == "track-b"


def test_inference_rejects_training_supervision(adapter):
    history = adapter.build_history("track-a", [record(1)])
    with pytest.raises(ValueError, match="must not carry"):
        adapter.build_association_sample(history, record(2), expected_match=True)


def test_hybrid_prompt_preserves_base_masks_novel_and_handles_unknown(adapter):
    history = adapter.build_history("track-a", [
        record(1, "synthetic_auto"), record(2, "synthetic_cart")
    ])
    candidate_record = record(3, "not_registered")
    sample = adapter.build_association_sample(history, candidate_record, history_length=2)
    boxes = sample.context.token_metadata["boxes"]
    assert boxes[0]["prompt_category"] == "synthetic_vehicle"
    assert boxes[1]["prompt_category"] == "Unknown"
    assert boxes[2]["prompt_category"] == "Unknown"


def test_raw_prompt_disables_novel_mask(adapter):
    history = adapter.build_history("track-a", [record(1, "synthetic_cart")])
    sample = adapter.build_association_sample(
        history,
        record(2, "synthetic_cart"),
        class_prompt_policy=ClassPromptPolicy(mode="raw"),
    )
    assert "Class:synthetic_cart" in sample.semantic_prompt


def test_preparation_rejects_missing_label_in_training():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["candidates"][0].pop("expected_match")
    with pytest.raises(PreparedDataError, match="expected_match"):
        prepare_associations(document, history_length=3)


def test_preparation_never_uses_detector_score_as_auxiliary_target():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["candidates"][0].pop("auxiliary_quality")
    with pytest.raises(PreparedDataError, match="auxiliary_quality"):
        prepare_associations(document, history_length=3, auxiliary_loss_enabled=True)


def test_prepared_row_has_four_boxes_for_two_available_history_items():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    row = prepare_associations(document, history_length=3)[0]
    assert row["box_count"] == 3
    assert row["box_roles"] == ["history_0", "history_1", "candidate"]
    assert validate_association_row(row, require_target=True, require_auxiliary=True)["target"] == "Yes"
