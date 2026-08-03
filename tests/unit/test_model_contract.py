from __future__ import annotations

import unittest
import warnings
import sys
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from nova.config import ModelConfig, ReproducibilityWarning, TokenMode
from nova.contracts.model import AssociationModelInput, ModelContractError
from nova.models.association_model import AssociationModel, build_language_model
from nova.models.tokenizer import ResolvedTokenIds


class SyntheticLanguageModel(nn.Module):
    def __init__(self, vocabulary_size=16, hidden_dim=12):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels=None):
        logits = self.output(inputs_embeds)
        loss = logits.square().mean() if labels is not None else None
        return SimpleNamespace(logits=logits, loss=loss)


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = ModelConfig()
        self.tokens = ResolvedTokenIds(
            box=7,
            positive=(2,),
            negative=(3,),
            mode=TokenMode.PLAIN,
        )

    def make_input(self):
        return AssociationModelInput(
            input_ids=torch.tensor([[4, 7, 5, 0], [7, 6, 7, 5]]),
            attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            geometry=torch.randn(3, 9),
            box_counts=torch.tensor([1, 2]),
            box_token_id=7,
            labels=None,
        )

    def make_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ReproducibilityWarning)
            return AssociationModel(
                self.config,
                language_hidden_dim=12,
                language_model=SyntheticLanguageModel(hidden_dim=12),
            )

    def test_missing_geometry_is_rejected(self):
        inputs = self.make_input()
        inputs.geometry = None
        with self.assertRaises(ModelContractError):
            inputs.validate()

    def test_box_token_geometry_mismatch_is_rejected(self):
        inputs = self.make_input()
        inputs.box_counts = torch.tensor([2, 1])
        with self.assertRaises(ModelContractError):
            self.make_model()(inputs, self.tokens)

    def test_synthetic_forward_contract(self):
        output = self.make_model()(self.make_input(), self.tokens)
        self.assertEqual(tuple(output.token_logits.shape), (2, 4, 16))
        self.assertEqual(tuple(output.p_yes.shape), (2,))
        self.assertEqual(tuple(output.auxiliary_quality.shape), (3,))
        self.assertTrue(((output.p_yes >= 0) & (output.p_yes <= 1)).all().item())

    def test_language_model_builder_is_offline_and_rejects_remote_code(self):
        factory = mock.Mock(return_value=SyntheticLanguageModel())
        module = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory))
        with mock.patch.dict(sys.modules, {"transformers": module}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ReproducibilityWarning)
                model = build_language_model(self.config, "/models/example")
        self.assertIsInstance(model, SyntheticLanguageModel)
        kwargs = factory.call_args.kwargs
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])


if __name__ == "__main__":
    unittest.main()
