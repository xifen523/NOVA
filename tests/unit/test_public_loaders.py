from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from nova.config import ReproducibilityWarning
from nova.models.loading import ModelLoadError, load_nova_model
from nova.models.tokenizer import TokenizerLoadError, load_tokenizer
from nova.runtime.checkpoints import CheckpointFormat, CheckpointInspection, CheckpointLoadReport


class FakeTokenizer:
    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.eos_token_id = 6
        self.eos_token = "<eos>"
        self.pad_token_id = None
        self._pad_token = None
        self.unk_token_id = 0
        self.padding_side = "left"
        self.mapping = {"<box>": 7, "Yes": 1, "No": 2}

    @property
    def pad_token(self):
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        if value == self.eos_token:
            self.pad_token_id = self.eos_token_id

    def __len__(self):
        return self.size

    def add_special_tokens(self, document):
        assert document == {"additional_special_tokens": ["<box>"]}
        return 0

    def convert_tokens_to_ids(self, token):
        return self.mapping.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        return [self.mapping.get(text, self.unk_token_id)]


class FakeLanguageModel(nn.Module):
    def __init__(self, vocab: int = 10, hidden: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden)

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, size):
        hidden = self.embedding.embedding_dim
        self.embedding = nn.Embedding(size, hidden)
        self.lm_head = nn.Linear(hidden, size, bias=False)
        return self.embedding


def test_standard_tokenizer_path_is_offline_safe_and_idempotent():
    tokenizer = FakeTokenizer()
    factory = mock.Mock(return_value=tokenizer)
    transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory))
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        loaded, first = load_tokenizer("local-model")
        _, second = load_tokenizer("local-model")
    assert loaded is tokenizer
    assert first.token_ids == second.token_ids
    assert first.padding_side == "right"
    assert first.pad_token_id == first.eos_token_id
    assert factory.call_args.kwargs["local_files_only"] is True
    assert factory.call_args.kwargs["trust_remote_code"] is False


def test_remote_tokenizer_access_requires_explicit_opt_in():
    with pytest.raises(TokenizerLoadError, match="explicit allow_download"):
        load_tokenizer("remote/model", local_files_only=False)


def test_invalid_tokenizer_path_reports_standard_failure():
    factory = mock.Mock(side_effect=OSError("not found"))
    transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory))
    with mock.patch.dict(sys.modules, {"transformers": transformers}):
        with pytest.raises(TokenizerLoadError, match="standard tokenizer loading failed"):
            load_tokenizer("definitely-not-a-local-directory")


def _checkpoint_loader(model, _path, **kwargs):
    report = CheckpointLoadReport(
        "PEFT_ADAPTER", ("geometry_encoder",) if kwargs.get("components_only") else ("peft_adapter",),
        (), (), (), (),
    )
    return report if kwargs.get("components_only") else (model, report)


def test_strict_vocab_mismatch_rejects_and_verified_policy_resizes_in_memory():
    tokenizer = FakeTokenizer(size=8)
    token_report = SimpleNamespace(
        token_ids=SimpleNamespace(box=7, positive=(1,), negative=(2,)),
        load_method="fixture",
    )
    inspection = CheckpointInspection(CheckpointFormat.PEFT_ADAPTER, 8, ("adapter_model.safetensors",))
    factory = mock.Mock(side_effect=lambda *_args, **_kwargs: FakeLanguageModel())
    transformers = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory))
    patches = (
        mock.patch("nova.models.loading.load_tokenizer", return_value=(tokenizer, token_report)),
        mock.patch("nova.models.loading.inspect_checkpoint", return_value=inspection),
        mock.patch("nova.models.loading.load_checkpoint", side_effect=_checkpoint_loader),
        mock.patch.dict(sys.modules, {"transformers": transformers}),
    )
    with patches[0], patches[1], patches[2], patches[3], warnings.catch_warnings():
        warnings.simplefilter("ignore", ReproducibilityWarning)
        with pytest.raises(ModelLoadError, match="differs from base vocabulary"):
            load_nova_model("local", checkpoint_path="checkpoint", vocab_alignment="strict")
    with mock.patch("nova.models.loading.load_tokenizer", return_value=(tokenizer, token_report)), \
         mock.patch("nova.models.loading.inspect_checkpoint", return_value=inspection), \
         mock.patch("nova.models.loading.load_checkpoint", side_effect=_checkpoint_loader), \
         mock.patch.dict(sys.modules, {"transformers": transformers}), warnings.catch_warnings():
        warnings.simplefilter("ignore", ReproducibilityWarning)
        model, _, report = load_nova_model(
            "local", checkpoint_path="checkpoint", vocab_alignment="verified-overlap"
        )
    assert model.language_model.get_input_embeddings().weight.shape[0] == 8
    assert report.embedding_resize_decision == "RESIZE_IN_MEMORY_TO_VERIFIED_CHECKPOINT_VOCAB"
    assert report.loaded_components == ("peft_adapter", "geometry_encoder")
    assert report.base_vocab_size == 10


def test_adapter_without_saved_embeddings_expands_for_registered_tokenizer_rows():
    tokenizer = FakeTokenizer(size=12)
    token_report = SimpleNamespace(
        token_ids=SimpleNamespace(box=7, positive=(1,), negative=(2,)),
        load_method="fixture",
    )
    inspection = CheckpointInspection(
        CheckpointFormat.PEFT_ADAPTER, None, ("adapter_model.safetensors",)
    )
    transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=mock.Mock(return_value=FakeLanguageModel())
        )
    )
    with mock.patch("nova.models.loading.load_tokenizer", return_value=(tokenizer, token_report)), \
         mock.patch("nova.models.loading.inspect_checkpoint", return_value=inspection), \
         mock.patch("nova.models.loading.load_checkpoint", side_effect=_checkpoint_loader), \
         mock.patch.dict(sys.modules, {"transformers": transformers}), warnings.catch_warnings():
        warnings.simplefilter("ignore", ReproducibilityWarning)
        model, _, report = load_nova_model("local", checkpoint_path="checkpoint")
    assert model.language_model.get_input_embeddings().weight.shape[0] == 12
    assert report.embedding_resize_decision == "EXPAND_IN_MEMORY_TO_TOKENIZER_VOCAB_BEFORE_ADAPTER"
