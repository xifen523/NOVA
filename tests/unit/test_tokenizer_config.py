from __future__ import annotations

import unittest
import warnings
from unittest import mock

from nova.config import ModelConfig, ReproducibilityWarning, TokenConfig, TokenMode
from nova.models.tokenizer import build_tokenizer, resolve_token_ids


class SyntheticTokenizer:
    def __init__(self):
        self.mapping = {"<box>": 9, "Yes": 1, "No": 2, " Yes": 3, " No": 4}
        self.registered = []

    def add_special_tokens(self, values):
        self.registered.extend(values["additional_special_tokens"])
        return 0

    def convert_tokens_to_ids(self, token):
        return self.mapping[token]

    def encode(self, text, add_special_tokens=False):
        self.assert_no_special = add_special_tokens
        return [self.mapping[text]]


class TokenizerConfigTests(unittest.TestCase):
    def test_plain_is_default_and_registers_box(self):
        tokenizer = SyntheticTokenizer()
        resolved = resolve_token_ids(tokenizer, TokenConfig())
        self.assertEqual(resolved.mode, TokenMode.PLAIN)
        self.assertEqual(resolved.box, 9)
        self.assertEqual(resolved.positive, (1,))
        self.assertEqual(resolved.negative, (2,))
        self.assertEqual(tokenizer.registered, ["<box>"])

    def test_both_mode_registers_plain_and_space_variants(self):
        resolved = resolve_token_ids(
            SyntheticTokenizer(), TokenConfig(token_mode=TokenMode.BOTH)
        )
        self.assertEqual(resolved.positive, (1, 3))
        self.assertEqual(resolved.negative, (2, 4))

    def test_builder_delegates_to_safe_loader(self):
        tokenizer = SyntheticTokenizer()
        with mock.patch("nova.models.tokenizer.load_tokenizer", return_value=(tokenizer, object())) as loader:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ReproducibilityWarning)
                self.assertIs(build_tokenizer(ModelConfig(), "/models/example"), tokenizer)
        self.assertTrue(loader.call_args.kwargs["local_files_only"])
        self.assertFalse(loader.call_args.kwargs["allow_download"])


if __name__ == "__main__":
    unittest.main()
