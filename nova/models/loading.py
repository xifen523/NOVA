"""Public model loader with explicit offline and vocabulary-alignment policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from nova.config.schema import ModelConfig, TokenConfig
from nova.runtime.checkpoints import CheckpointLoadReport, inspect_checkpoint, load_checkpoint

from .association_model import AssociationModel
from .tokenizer import ResolvedTokenIds, TokenizerReport, load_tokenizer


class ModelLoadError(RuntimeError):
    pass


class VocabAlignmentPolicy(str, Enum):
    STRICT = "strict"
    VERIFIED_OVERLAP = "verified-overlap"


@dataclass(frozen=True)
class LoadReport:
    architecture: str
    hidden_size: int
    base_vocab_size: int
    tokenizer_vocab_size: int
    checkpoint_format: str
    loaded_components: Tuple[str, ...]
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    shape_mismatches: Tuple[str, ...]
    embedding_resize_decision: str
    input_embedding_rows_before: int
    input_embedding_rows_after: int
    output_embedding_rows_before: Optional[int]
    output_embedding_rows_after: Optional[int]
    embeddings_tied_before: bool
    embeddings_tied_after: bool
    tokenizer_method: str
    warnings: Tuple[str, ...]


def resolve_torch_dtype(name: Optional[str], device: str) -> Optional[torch.dtype]:
    if name is None or name == "auto":
        return None
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in mapping:
        raise ModelLoadError("dtype must be auto, float32, float16, or bfloat16")
    if device == "cpu" and name == "float16":
        raise ModelLoadError("float16 CPU execution is not supported; use float32 or bfloat16")
    return mapping[name]


def _embedding_state(model: Any) -> Tuple[int, Optional[int], bool]:
    input_embedding = model.get_input_embeddings()
    output_getter = getattr(model, "get_output_embeddings", None)
    output_embedding = output_getter() if callable(output_getter) else getattr(model, "lm_head", None)
    input_rows = int(input_embedding.weight.shape[0])
    output_rows = int(output_embedding.weight.shape[0]) if output_embedding is not None else None
    tied = bool(
        output_embedding is not None
        and input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr()
    )
    return input_rows, output_rows, tied


def load_nova_model(
    model_name_or_path: str,
    *,
    checkpoint_path: Optional[str] = None,
    revision: Optional[str] = None,
    device: str = "cpu",
    local_files_only: bool = True,
    allow_download: bool = False,
    dtype: Optional[str] = "auto",
    training: bool = False,
    checkpoint_sha256: Optional[str] = None,
    vocab_alignment: str = "strict",
    allow_legacy_torch_checkpoint: bool = False,
    model_config: Optional[ModelConfig] = None,
) -> Tuple[AssociationModel, Any, LoadReport]:
    """Load a causal LM, NOVA heads, and an optional adapter/checkpoint.

    ``strict`` rejects checkpoint embedding mismatches. ``verified-overlap`` is
    opt-in and is accepted only when the checkpoint vocabulary equals the loaded
    tokenizer size and all NOVA decision tokens lie inside that vocabulary.
    Source model and checkpoint files are never modified.
    """

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ModelLoadError("CUDA was requested but is not available")
    if device != "cpu" and not device.startswith("cuda"):
        raise ModelLoadError("device must be cpu or cuda[:index]")
    try:
        policy = VocabAlignmentPolicy(vocab_alignment)
    except ValueError as exc:
        raise ModelLoadError("vocab_alignment must be strict or verified-overlap") from exc
    config = model_config or ModelConfig(
        base_model=model_name_or_path,
        revision=revision,
        trust_remote_code=False,
        lora=ModelConfig().lora,
        geometry=ModelConfig().geometry,
        auxiliary_head=ModelConfig().auxiliary_head,
        tokens=TokenConfig(),
    )
    tokenizer, tokenizer_report = load_tokenizer(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        allow_download=allow_download,
        token_config=config.tokens,
    )
    if allow_download and local_files_only:
        local_files_only = False
    if not allow_download and not local_files_only:
        raise ModelLoadError("remote download requires explicit allow_download=True")
    from transformers import AutoModelForCausalLM

    requested_dtype = resolve_torch_dtype(dtype, device)
    kwargs = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    if requested_dtype is not None:
        kwargs["torch_dtype"] = requested_dtype
    language_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    input_rows_before, output_rows_before, tied_before = _embedding_state(language_model)
    base_vocab = input_rows_before
    hidden_size = int(language_model.config.hidden_size)
    checkpoint_report: Optional[CheckpointLoadReport] = None
    resize_decision = "KEEP_BASE_VOCAB"
    warnings = []
    tokenizer_vocab = len(tokenizer)
    decision_ids = tokenizer_report.token_ids
    all_decision_ids = (decision_ids.box,) + decision_ids.positive + decision_ids.negative
    if checkpoint_path is not None:
        inspection = inspect_checkpoint(Path(checkpoint_path), expected_sha256=checkpoint_sha256)
        checkpoint_vocab = inspection.embedding_rows
        if checkpoint_vocab is not None and checkpoint_vocab != base_vocab:
            if policy is VocabAlignmentPolicy.STRICT:
                raise ModelLoadError(
                    "checkpoint vocabulary {0} differs from base vocabulary {1}; "
                    "use verified-overlap only after token mapping review".format(checkpoint_vocab, base_vocab)
                )
            if checkpoint_vocab != tokenizer_vocab or max(all_decision_ids) >= checkpoint_vocab:
                raise ModelLoadError("verified-overlap preconditions were not satisfied")
            language_model.resize_token_embeddings(checkpoint_vocab)
            resize_decision = "RESIZE_IN_MEMORY_TO_VERIFIED_CHECKPOINT_VOCAB"
            warnings.append(
                "Base embeddings were resized in memory after verified token mapping; source files were unchanged."
            )
        elif checkpoint_vocab is not None and max(all_decision_ids) >= checkpoint_vocab:
            raise ModelLoadError("checkpoint embeddings do not contain every configured NOVA token")
        elif checkpoint_vocab is None and tokenizer_vocab > base_vocab:
            language_model.resize_token_embeddings(tokenizer_vocab)
            resize_decision = "EXPAND_IN_MEMORY_TO_TOKENIZER_VOCAB_BEFORE_ADAPTER"
            warnings.append(
                "Embeddings were expanded in memory before loading an adapter without saved embedding rows; source files were unchanged."
            )
        language_model, checkpoint_report = load_checkpoint(
            language_model,
            Path(checkpoint_path),
            expected_sha256=checkpoint_sha256,
            training=training,
            allow_legacy_torch=allow_legacy_torch_checkpoint,
        )
    elif tokenizer_vocab > base_vocab:
        language_model.resize_token_embeddings(tokenizer_vocab)
        resize_decision = "EXPAND_IN_MEMORY_TO_TOKENIZER_VOCAB"
        warnings.append(
            "Embeddings were expanded in memory to include registered NOVA tokens; source files were unchanged."
        )
    input_rows_after, output_rows_after, tied_after = _embedding_state(language_model)
    if max(all_decision_ids) >= input_rows_after:
        raise ModelLoadError("input embeddings do not contain every configured NOVA token")
    if output_rows_after is not None and max(decision_ids.positive + decision_ids.negative) >= output_rows_after:
        raise ModelLoadError("output embeddings do not contain every configured decision token")
    if tied_before and not tied_after:
        warnings.append(
            "The base model used tied embeddings; the loaded adapter exposes separately saved input/output modules. Both row counts were validated."
        )
    wrapper = AssociationModel(config, hidden_size, language_model=language_model)
    if checkpoint_path is not None:
        component_report = load_checkpoint(
            wrapper,
            Path(checkpoint_path),
            expected_sha256=checkpoint_sha256,
            components_only=True,
            allow_legacy_torch=allow_legacy_torch_checkpoint,
        )
        if checkpoint_report is None:
            checkpoint_report = component_report
        else:
            checkpoint_report = CheckpointLoadReport(
                checkpoint_report.checkpoint_format,
                checkpoint_report.loaded_components + component_report.loaded_components,
                checkpoint_report.missing_keys + component_report.missing_keys,
                checkpoint_report.unexpected_keys + component_report.unexpected_keys,
                checkpoint_report.shape_mismatches + component_report.shape_mismatches,
                checkpoint_report.warnings + component_report.warnings,
                checkpoint_report.skipped_keys + component_report.skipped_keys,
                checkpoint_report.aligned_keys + component_report.aligned_keys,
            )
    wrapper.to(device)
    wrapper.train(training)
    checkpoint_report = checkpoint_report or CheckpointLoadReport.empty()
    architecture = language_model.__class__.__name__
    report = LoadReport(
        architecture=architecture,
        hidden_size=hidden_size,
        base_vocab_size=base_vocab,
        tokenizer_vocab_size=tokenizer_vocab,
        checkpoint_format=checkpoint_report.checkpoint_format,
        loaded_components=checkpoint_report.loaded_components,
        missing_keys=checkpoint_report.missing_keys,
        unexpected_keys=checkpoint_report.unexpected_keys,
        shape_mismatches=checkpoint_report.shape_mismatches,
        embedding_resize_decision=resize_decision,
        input_embedding_rows_before=input_rows_before,
        input_embedding_rows_after=input_rows_after,
        output_embedding_rows_before=output_rows_before,
        output_embedding_rows_after=output_rows_after,
        embeddings_tied_before=tied_before,
        embeddings_tied_after=tied_after,
        tokenizer_method=tokenizer_report.load_method,
        warnings=tuple(warnings) + checkpoint_report.warnings,
    )
    return wrapper, tokenizer, report


def prepare_model_input(
    tokenizer: Any,
    prompts: Tuple[str, ...],
    geometries: torch.Tensor,
    token_ids: ResolvedTokenIds,
    *,
    box_counts: Tuple[int, ...],
    device: str,
    labels: Optional[torch.Tensor] = None,
):
    """Tokenize prompts and preserve each history-aware multi-box count."""

    from nova.contracts.model import AssociationModelInput

    encoded = tokenizer(list(prompts), padding=True, return_tensors="pt")
    if len(box_counts) != len(prompts):
        raise ValueError("box_counts must contain one entry per prompt")
    model_input = AssociationModelInput(
        input_ids=encoded.input_ids.to(device),
        attention_mask=encoded.attention_mask.to(device),
        geometry=geometries.to(device),
        box_counts=torch.tensor(box_counts, dtype=torch.long, device=device),
        box_token_id=token_ids.box,
        labels=labels.to(device) if labels is not None else None,
    )
    model_input.validate()
    return model_input
