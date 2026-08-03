from __future__ import annotations

import unittest
import warnings

import yaml

from nova.config import (
    ConfigValidationError,
    ReproducibilityWarning,
    TokenMode,
    load_config_text,
)


VALID = """
experiment_name: unit-synthetic
model:
  base_model: Qwen/Qwen2.5-0.5B-Instruct
  revision: null
  trust_remote_code: false
  geometry: {input_dim: 9, hidden_dim: 1024}
  lora:
    rank: 16
    alpha: 32
    dropout: null
    target_modules: [q_proj, k_proj, v_proj, o_proj]
  auxiliary_head: {enabled: true, hidden_dim: 256, loss_weight: 1.0}
  tokens: {box: "<box>", positive: "Yes", negative: "No", token_mode: plain}
training: {enabled: false}
"""


class ConfigSchemaTests(unittest.TestCase):
    def load(self, text=VALID, for_training=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ReproducibilityWarning)
            return load_config_text(text, for_training=for_training)

    def test_valid_yaml_parses(self):
        config = self.load()
        self.assertEqual(config.model.geometry.input_dim, 9)
        self.assertEqual(config.model.tokens.token_mode, TokenMode.PLAIN)
        self.assertIsNone(config.model.lora.dropout)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ConfigValidationError):
            self.load(VALID.replace("model:\n", "model:\n  surprise: 1\n"))

    def test_missing_model_is_rejected(self):
        with self.assertRaises(ConfigValidationError):
            self.load("experiment_name: missing-model\n")

    def test_geometry_width_other_than_nine_is_rejected(self):
        with self.assertRaises(ConfigValidationError):
            self.load(VALID.replace("input_dim: 9", "input_dim: 8"))

    def test_remote_custom_code_is_rejected(self):
        with self.assertRaises(ConfigValidationError):
            self.load(VALID.replace("trust_remote_code: false", "trust_remote_code: true"))

    def test_lora_rank_and_alpha_must_be_positive(self):
        for field in ("rank", "alpha"):
            with self.subTest(field=field):
                with self.assertRaises(ConfigValidationError):
                    self.load(VALID.replace("{0}: {1}".format(field, 16 if field == "rank" else 32), "{0}: 0".format(field)))

    def test_training_requires_explicit_lora_dropout(self):
        with self.assertRaises(ConfigValidationError):
            self.load(VALID, for_training=True)

    def test_null_revision_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_config_text(VALID)
        self.assertTrue(any(item.category is ReproducibilityWarning for item in caught))

    def test_unsafe_yaml_constructor_is_not_allowed(self):
        with self.assertRaises(yaml.YAMLError):
            self.load("!!python/object/apply:builtins.str [unsafe]")


if __name__ == "__main__":
    unittest.main()

