"""Paper-fidelity configuration and training-runtime tests."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from nova.config import ConfigValidationError, load_config, load_config_text
from nova.data.prepared import prepare_associations, write_jsonl
from nova.training.engine import (
    TinyAssociationBackend,
    build_optimizer_parameter_groups,
    build_supervised_token_ids,
    train_prepared_associations,
    validate_training_rows_against_config,
)
from nova.models.tokenizer import ResolvedTokenIds
from nova.config.schema import TokenMode


CONFIG = Path("configs/examples/synthetic_v2x.yaml")
FIXTURE = Path("tests/fixtures/synthetic_v2x.json")


class BoundaryTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        if text == " Yes":
            return [91]
        if text == " No":
            return [92]
        values = [ord(value) % 67 + 3 for value in text]
        return ([1] if add_special_tokens else []) + values


@pytest.mark.parametrize("prompt", [
    "Is this the same object? Answer:",
    "History: <box> Candidate: <box> Answer:",
    "History: <box> <box> <box> Candidate: <box> Class:Unknown Answer:",
])
def test_target_mask_uses_explicit_token_boundary_for_prompts_boxes_and_unknown(prompt):
    tokenizer = BoundaryTokenizer()
    for target, expected in (("Yes", 71), ("No", 72)):
        token_ids = ResolvedTokenIds(
            box=99,
            positive=(71,),
            negative=(72,),
            mode=TokenMode.SPACE,
        )
        ids, labels = build_supervised_token_ids(tokenizer, prompt, target, token_ids)
        assert ids[-1] == expected
        assert labels[-1] == expected
        assert all(value == -100 for value in labels[:-1])
        assert [index for index, value in enumerate(labels) if value != -100] == [len(ids) - 1]


def test_optimizer_has_three_disjoint_configured_learning_rates():
    config = load_config(CONFIG, for_training=True)
    groups = build_optimizer_parameter_groups(TinyAssociationBackend(), config)
    assert {group["name"] for group in groups} == {"language_model", "geometry_encoder", "auxiliary_head"}
    assert {group["name"]: group["lr"] for group in groups} == {
        "language_model": config.training.learning_rate_language_model,
        "geometry_encoder": config.training.learning_rate_geometry,
        "auxiliary_head": config.training.learning_rate_auxiliary_head,
    }


def test_training_config_requires_explicit_sampling_ratios():
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace("positive_exact_ratio: 1.0", "positive_exact_ratio: null", 1)
    with pytest.raises((ConfigValidationError, TypeError)):
        load_config_text(text, for_training=True)


def _prepared(tmp_path):
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = prepare_associations(document, history_length=3)
    path = tmp_path / "prepared.jsonl"
    write_jsonl(path, rows)
    return path


def test_tiny_training_uses_batch_accumulation_and_writes_effective_config(tmp_path):
    config = load_config(CONFIG, for_training=True)
    output = tmp_path / "train"
    summary = train_prepared_associations(
        _prepared(tmp_path), output, config=config, backend="tiny", steps=2
    )
    assert summary["steps"] == 2
    assert summary["batch_size"] == config.training.batch_size
    assert summary["gradient_accumulation_steps"] == config.training.gradient_accumulation_steps
    effective = json.loads((output / "effective_config.json").read_text(encoding="utf-8"))
    assert effective["model"]["lora"]["rank"] == config.model.lora.rank
    assert effective["training"]["auxiliary_loss_weight"] == 1.0
    assert effective["runtime_binding"]["history_length"] == 3
    assert effective["runtime_binding"]["observed_sampling_counts"]["positive_exact"] == 1
    assert effective["runtime_binding"]["effective_auxiliary_loss_weight"] == 1.0


def test_training_rejects_disabled_sampling_kind_and_history_mismatch(tmp_path):
    config = load_config(CONFIG, for_training=True)
    rows = prepare_associations(json.loads(FIXTURE.read_text(encoding="utf-8")), history_length=3)
    disabled = dict(rows[0])
    disabled["sampling_type"] = "negative_hard"
    with pytest.raises(ValueError, match="configured ratio is zero"):
        validate_training_rows_against_config([disabled], config)
    wrong_history = dict(rows[0])
    wrong_history["prompt_metadata"] = dict(wrong_history["prompt_metadata"])
    wrong_history["prompt_metadata"]["history_requested"] = 2
    with pytest.raises(ValueError, match="history_requested"):
        validate_training_rows_against_config([wrong_history], config)


def test_auxiliary_head_and_training_weights_both_affect_effective_loss(tmp_path):
    config = load_config(CONFIG, for_training=True)
    config = replace(
        config,
        model=replace(
            config.model,
            auxiliary_head=replace(config.model.auxiliary_head, loss_weight=0.25),
        ),
        training=replace(config.training, auxiliary_loss_weight=0.4),
    )
    summary = train_prepared_associations(
        _prepared(tmp_path), tmp_path / "weighted", config=config, backend="tiny", steps=1
    )
    assert summary["effective_auxiliary_loss_weight"] == pytest.approx(0.1)


def test_tiny_checkpoint_resume_advances_global_step(tmp_path):
    config = load_config(CONFIG, for_training=True)
    prepared = _prepared(tmp_path)
    first_output = tmp_path / "first"
    first = train_prepared_associations(prepared, first_output, config=config, backend="tiny", steps=1)
    assert first["steps"] == 1
    resume_dir = first_output / "checkpoint-epoch-1"
    resumed_config = replace(
        config,
        training=replace(config.training, resume_from=str(resume_dir)),
    )
    second = train_prepared_associations(
        prepared, tmp_path / "second", config=resumed_config, backend="tiny", steps=2
    )
    assert second["steps"] == 2


def test_auxiliary_quality_counts_are_reported(tmp_path):
    config = load_config(CONFIG, for_training=True)
    summary = train_prepared_associations(
        _prepared(tmp_path), tmp_path / "train", config=config, backend="tiny", steps=1
    )
    assert summary["samples_with_auxiliary_target"] == 1
    assert summary["samples_without_auxiliary_target"] == 0
