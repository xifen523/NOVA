"""Final experiment-qualified configuration and release-interface contracts."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nova.config import load_config
from nova.data import ClassSplit
from nova.inference import _effective_class_gate, _effective_pair_batch_size
from nova.tracking import class_compatible
from nova.training import train_prepared_associations


ROOT = Path(__file__).parents[2]
REPRODUCTION_CONFIGS = (
    ROOT / "configs/reproduction/v2x_gd_nova.yaml",
    ROOT / "configs/reproduction/kitti_gd_nova.yaml",
    ROOT / "configs/reproduction/nuscenes_fnp_nova_scorea.yaml",
)


@pytest.mark.parametrize("path", REPRODUCTION_CONFIGS)
def test_reproduction_configs_are_strict_complete_and_checkpoint_aligned(path):
    config = load_config(path)
    assert config.dataset.real_mode is True
    assert config.dataset.history_length == 3
    assert config.model.geometry.input_dim == 9
    assert config.model.tokens.token_mode.value == "space"
    assert config.reproduction.iou_threshold == 0.25
    assert config.reproduction.known_unresolved_fields
    assert config.reproduction.checkpoint_identifier != "UNRESOLVED"
    assert config.reproduction.geometry_definition != "UNRESOLVED"


def test_both_pair_limits_and_class_gate_controls_affect_runtime():
    config = load_config(REPRODUCTION_CONFIGS[0])
    config = replace(
        config,
        inference=replace(
            config.inference,
            pair_batch_size=64,
            max_pairs_per_forward=7,
            class_gate=True,
            class_gate_policy="disabled",
        ),
    )
    assert _effective_pair_batch_size(config) == 7
    assert _effective_class_gate(config) is False
    assert _effective_class_gate(
        replace(config, inference=replace(config.inference, class_gate_policy="exact_alias"))
    ) is True


def test_base_novel_gate_is_distinct_from_exact_alias_gate():
    config = load_config(REPRODUCTION_CONFIGS[1])
    split = ClassSplit.from_mapping({
        "base_classes": config.dataset.class_split.base_classes,
        "novel_classes": config.dataset.class_split.novel_classes,
        "aliases": config.dataset.class_split.aliases,
        "unknown_policy": config.dataset.class_split.unknown_policy,
        "authoritative": config.dataset.class_split.authoritative,
        "mapping_status": config.dataset.class_split.mapping_status,
    })
    track = SimpleNamespace(category="car")
    detection = SimpleNamespace(category="cyclist")
    assert class_compatible(track, detection, split, "base_novel") is True
    assert class_compatible(track, detection, split, "exact_alias") is False


def test_training_enabled_is_an_execution_gate(tmp_path):
    config = load_config(REPRODUCTION_CONFIGS[0])
    with pytest.raises(ValueError, match="training.enabled"):
        train_prepared_associations(
            tmp_path / "not-read.jsonl", tmp_path / "out", config=config, backend="tiny"
        )


def test_weight_manifest_and_tested_environment_are_parseable():
    manifest = yaml.safe_load((ROOT / "weights/manifest.template.yaml").read_text(encoding="utf-8"))
    required = {
        "name", "dataset", "base_model", "base_model_revision",
        "base_model_tree_sha256", "tokenizer_sha256", "checkpoint_filename",
        "checkpoint_sha256", "checkpoint_format", "expected_vocab_size",
        "box_token", "positive_token", "negative_token", "geometry_dimension",
        "history_length", "config", "download_url", "license",
    }
    assert set(manifest) == required
    assert manifest["geometry_dimension"] == 9
    environment = yaml.safe_load((ROOT / "environment/environment.yml").read_text(encoding="utf-8"))
    assert environment["dependencies"][0] == "python=3.10.18"
