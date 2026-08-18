"""Safe tokenizer loading and NOVA decision-token resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from nova.config.schema import ModelConfig, TokenConfig, TokenMode


class TokenizerLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedTokenIds:
    box: int
    positive: Tuple[int, ...]
    negative: Tuple[int, ...]
    mode: TokenMode

    def validate(self) -> None:
        values = (self.box,) + self.positive + self.negative
        if not self.positive or not self.negative:
            raise ValueError("positive and negative token-ID sets must be non-empty")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("token IDs must be non-negative integers")
        if set(self.positive) & set(self.negative):
            raise ValueError("positive and negative token-ID sets must not overlap")


@dataclass(frozen=True)
class TokenizerReport:
    source: str
    load_method: str
    vocabulary_size: int
    padding_side: str
    pad_token_id: int
    eos_token_id: int
    token_ids: ResolvedTokenIds
    fallback_status: str = "NOT_APPLICABLE"


@dataclass(frozen=True)
class TokenizerEquivalenceReport:
    status: str
    cases_checked: int
    mismatches: Tuple[str, ...]


REFERENCE_PROMPTS = (
    "NOVA association <box> <box> Answer: Yes",
    "Unknown object 123.45 <box> Answer: No",
    "车辆 Unknown 多个框 <box> <box> <box>",
    "Car Pedestrian Cyclist Truck Bus Motorcycle",
)


def validate_tokenizer_equivalence(
    reference: Any,
    candidate: Any,
    prompts: Sequence[str] = REFERENCE_PROMPTS,
) -> TokenizerEquivalenceReport:
    """Compare token IDs plus special/padding/chat-template behavior."""

    mismatches = []
    for index, prompt in enumerate(prompts):
        reference_ids = list(reference.encode(prompt, add_special_tokens=True))
        candidate_ids = list(candidate.encode(prompt, add_special_tokens=True))
        if reference_ids != candidate_ids:
            mismatches.append("prompt[{0}] token IDs".format(index))
    for attribute in (
        "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
        "padding_side", "truncation_side", "chat_template",
    ):
        if getattr(reference, attribute, None) != getattr(candidate, attribute, None):
            mismatches.append(attribute)
    return TokenizerEquivalenceReport(
        "SUPPORTED_FALLBACK" if not mismatches else "BEST_EFFORT_FALLBACK",
        len(prompts),
        tuple(mismatches),
    )


def _unique(values: Iterable[int]) -> Tuple[int, ...]:
    result: List[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def register_box_token(tokenizer: Any, config: TokenConfig) -> int:
    """Idempotently register ``<box>`` and return its non-unknown token ID."""

    config.validate()
    tokenizer.add_special_tokens({"additional_special_tokens": [config.box]})
    box_id = tokenizer.convert_tokens_to_ids(config.box)
    unknown = getattr(tokenizer, "unk_token_id", None)
    if isinstance(box_id, bool) or not isinstance(box_id, int) or box_id < 0:
        raise ValueError("tokenizer did not resolve the box token")
    if unknown is not None and box_id == unknown:
        raise ValueError("box token resolved to the unknown token")
    return box_id


def _single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(ids, Sequence) or len(ids) != 1:
        raise ValueError("decision token variant must resolve to exactly one token")
    value = ids[0]
    unknown = getattr(tokenizer, "unk_token_id", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("resolved token ID must be a non-negative integer")
    if unknown is not None and value == unknown:
        raise ValueError("decision token resolved to the unknown token")
    return value


def resolve_token_ids(tokenizer: Any, config: TokenConfig) -> ResolvedTokenIds:
    box_id = register_box_token(tokenizer, config)
    plain_positive = _single_token_id(tokenizer, config.positive)
    plain_negative = _single_token_id(tokenizer, config.negative)
    if config.token_mode is TokenMode.PLAIN:
        positive, negative = (plain_positive,), (plain_negative,)
    elif config.token_mode is TokenMode.SPACE:
        positive = (_single_token_id(tokenizer, " " + config.positive),)
        negative = (_single_token_id(tokenizer, " " + config.negative),)
    else:
        positive = _unique((plain_positive, _single_token_id(tokenizer, " " + config.positive)))
        negative = _unique((plain_negative, _single_token_id(tokenizer, " " + config.negative)))
    resolved = ResolvedTokenIds(box_id, positive, negative, config.token_mode)
    resolved.validate()
    return resolved


def _fallback_from_tokenizer_json(model_dir: Path) -> Any:
    """Rebuild a local BPE tokenizer after a known fast-tokenizer schema failure.

    This fallback accepts only a regular local ``tokenizer.json`` containing a
    BPE model and the Qwen-style Split+ByteLevel pipeline. It never downloads
    code or data and raises on unsupported layouts.
    """

    from tokenizers import AddedToken, Tokenizer, decoders, normalizers, pre_tokenizers, processors
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast

    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.is_symlink() or not tokenizer_json.is_file():
        raise TokenizerLoadError("fallback requires a regular local tokenizer.json")
    document = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    model = document.get("model", {})
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
        raise TokenizerLoadError("fallback supports only BPE tokenizer JSON")
    merges = []
    for merge in model.get("merges", []):
        if isinstance(merge, str):
            parts = tuple(merge.split())
        elif isinstance(merge, list):
            parts = tuple(merge)
        else:
            raise TokenizerLoadError("unsupported BPE merge entry")
        if len(parts) != 2:
            raise TokenizerLoadError("BPE merges must contain two tokens")
        merges.append(parts)
    backend = Tokenizer(BPE(vocab=model["vocab"], merges=merges))
    backend.normalizer = normalizers.NFC()
    pre = document.get("pre_tokenizer", {})
    entries = pre.get("pretokenizers", [])
    if pre.get("type") != "Sequence" or len(entries) != 2:
        raise TokenizerLoadError("unsupported pre-tokenizer layout")
    split_cfg, byte_cfg = entries
    pattern = split_cfg.get("pattern", {}).get("Regex")
    if split_cfg.get("type") != "Split" or not isinstance(pattern, str):
        raise TokenizerLoadError("fallback requires a regex Split pre-tokenizer")
    split = pre_tokenizers.Split(
        pattern=pattern,
        behavior=str(split_cfg.get("behavior", "isolated")).lower(),
        invert=bool(split_cfg.get("invert", False)),
    )
    if byte_cfg.get("type") != "ByteLevel":
        raise TokenizerLoadError("fallback requires a ByteLevel pre-tokenizer")
    byte = pre_tokenizers.ByteLevel(
        add_prefix_space=bool(byte_cfg.get("add_prefix_space", False)),
        trim_offsets=bool(byte_cfg.get("trim_offsets", True)),
        use_regex=bool(byte_cfg.get("use_regex", True)),
    )
    backend.pre_tokenizer = pre_tokenizers.Sequence([split, byte])
    decoder = document.get("decoder", {})
    processor = document.get("post_processor", {})
    if decoder.get("type") != "ByteLevel" or processor.get("type") != "ByteLevel":
        raise TokenizerLoadError("fallback requires ByteLevel decoder and processor")
    backend.decoder = decoders.ByteLevel(
        add_prefix_space=bool(decoder.get("add_prefix_space", True)),
        trim_offsets=bool(decoder.get("trim_offsets", True)),
        use_regex=bool(decoder.get("use_regex", True)),
    )
    backend.post_processor = processors.ByteLevel(
        add_prefix_space=bool(processor.get("add_prefix_space", False)),
        trim_offsets=bool(processor.get("trim_offsets", False)),
        use_regex=bool(processor.get("use_regex", True)),
    )
    for item in document.get("added_tokens", []):
        token = AddedToken(
            item["content"], single_word=bool(item.get("single_word", False)),
            lstrip=bool(item.get("lstrip", False)), rstrip=bool(item.get("rstrip", False)),
            normalized=bool(item.get("normalized", False)), special=bool(item.get("special", False)),
        )
        (backend.add_special_tokens if token.special else backend.add_tokens)([token])
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token=config.get("eos_token"), pad_token=config.get("pad_token"),
        bos_token=config.get("bos_token"), unk_token=config.get("unk_token"),
        clean_up_tokenization_spaces=bool(config.get("clean_up_tokenization_spaces", False)),
    )


def load_tokenizer(
    model_name_or_path: str,
    *,
    revision: Optional[str] = None,
    local_files_only: bool = True,
    allow_download: bool = False,
    token_config: Optional[TokenConfig] = None,
    padding_side: str = "right",
    allow_local_json_fallback: bool = True,
) -> Tuple[Any, TokenizerReport]:
    """Load a tokenizer safely; remote access requires ``allow_download=True``."""

    if not model_name_or_path or not str(model_name_or_path).strip():
        raise TokenizerLoadError("model_name_or_path is required")
    if allow_download and local_files_only:
        local_files_only = False
    if not allow_download and not local_files_only:
        raise TokenizerLoadError("remote download requires explicit allow_download=True")
    if padding_side not in {"left", "right"}:
        raise TokenizerLoadError("padding_side must be left or right")
    from transformers import AutoTokenizer

    method = "AutoTokenizer"
    fallback_status = "NOT_APPLICABLE"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision, local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except Exception as standard_error:
        path = Path(model_name_or_path)
        if not allow_local_json_fallback or not path.is_dir():
            raise TokenizerLoadError("standard tokenizer loading failed") from standard_error
        try:
            tokenizer = _fallback_from_tokenizer_json(path)
            method = "audited-local-json-fallback"
            fallback_status = "BEST_EFFORT_FALLBACK"
        except Exception as fallback_error:
            raise TokenizerLoadError(
                "standard tokenizer loading and the constrained local fallback both failed"
            ) from fallback_error
    tokenizer.padding_side = padding_side
    if tokenizer.eos_token_id is None:
        raise TokenizerLoadError("tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ids = resolve_token_ids(tokenizer, token_config or TokenConfig())
    report = TokenizerReport(
        source=str(model_name_or_path), load_method=method, vocabulary_size=len(tokenizer),
        padding_side=tokenizer.padding_side, pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id), token_ids=ids,
        fallback_status=fallback_status,
    )
    return tokenizer, report


def build_tokenizer(
    config: ModelConfig,
    model_name_or_path: Optional[str] = None,
    *,
    local_files_only: bool = True,
    allow_download: bool = False,
) -> Any:
    config.validate(for_training=False)
    tokenizer, _ = load_tokenizer(
        model_name_or_path or config.base_model,
        revision=config.revision,
        local_files_only=local_files_only,
        allow_download=allow_download,
        token_config=config.tokens,
    )
    return tokenizer
