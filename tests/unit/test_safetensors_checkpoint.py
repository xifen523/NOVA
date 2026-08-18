from __future__ import annotations

import hashlib
import warnings

import pytest
import torch
from torch import nn
from types import SimpleNamespace

from nova.config import ModelConfig, ReproducibilityWarning
from nova.models.association_model import AssociationModel
from nova.runtime.checkpoints import CheckpointPolicyError, inspect_checkpoint, load_checkpoint


safetensors = pytest.importorskip("safetensors.torch")


class SyntheticLanguageModel(nn.Module):
    def __init__(self, vocabulary_size=16, hidden_dim=12):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels=None):
        return SimpleNamespace(logits=self.output(inputs_embeds), loss=None)


def _model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ReproducibilityWarning)
        return AssociationModel(ModelConfig(), 12, SyntheticLanguageModel(hidden_dim=12))


def test_flat_safetensors_components_load_and_hash_is_enforced(tmp_path):
    model = _model()
    state = {}
    for prefix, module in (("geo_encoder", model.geometry_encoder), ("score_head", model.auxiliary_head.network)):
        for name, tensor in module.state_dict().items():
            state[prefix + "." + name] = tensor.detach().clone().contiguous()
    checkpoint = tmp_path / "geo_components.safetensors"
    safetensors.save_file(state, str(checkpoint))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    inspection = inspect_checkpoint(checkpoint, expected_sha256=digest)
    assert inspection.files == ("geo_components.safetensors",)
    report = load_checkpoint(model, checkpoint, components_only=True)
    assert report.loaded_components == ("geometry_encoder", "auxiliary_head")
    with pytest.raises(CheckpointPolicyError, match="SHA256 mismatch"):
        inspect_checkpoint(checkpoint, expected_sha256="0" * 64)


def test_component_checkpoint_missing_and_unexpected_keys_fail_closed(tmp_path):
    model = _model()
    checkpoint = tmp_path / "geo_components.safetensors"
    safetensors.save_file({
        "geo_encoder.unexpected": torch.zeros(1),
        "score_head.unexpected": torch.zeros(1),
    }, str(checkpoint))
    with pytest.raises(RuntimeError):
        load_checkpoint(model, checkpoint, components_only=True)
    with pytest.raises(CheckpointPolicyError, match="not found"):
        inspect_checkpoint(tmp_path / "missing.safetensors")
