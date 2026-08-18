"""Configuration-driven tiny and real association training engines."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from nova.config.schema import ExperimentConfig
from nova.contracts.model import AssociationModelInput
from nova.data.prepared import PreparedDataError, read_association_jsonl
from nova.models.auxiliary_head import auxiliary_quality_loss
from nova.models.loading import load_nova_model
from nova.models.tokenizer import ResolvedTokenIds, resolve_token_ids
from nova.utils.reproducibility import seed_everything


class _Rows(Dataset):
    def __init__(self, rows: Sequence[dict]) -> None:
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class TinyAssociationBackend(nn.Module):
    """Small explicit CI backend with the same three optimizer groups."""

    def __init__(self) -> None:
        super().__init__()
        self.geometry_encoder = nn.Sequential(nn.Linear(9, 16), nn.GELU())
        self.language_model = nn.Linear(16, 2)
        self.auxiliary_head = nn.Sequential(nn.Linear(16, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, geometry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.geometry_encoder(geometry)
        return self.language_model(hidden), self.auxiliary_head(hidden).squeeze(-1)


def _candidate_geometry(row: Mapping[str, object]) -> List[float]:
    geometry = row["geometry"]
    if not isinstance(geometry, list) or not geometry:
        raise ValueError("prepared geometry is empty")
    return list(geometry[-1])


def _collate_rows(rows: Sequence[dict]) -> dict:
    return {
        "rows": list(rows),
        "candidate_geometry": torch.tensor([_candidate_geometry(row) for row in rows], dtype=torch.float32),
        "targets": torch.tensor([1 if row["target"] == "Yes" else 0 for row in rows], dtype=torch.long),
        "quality": torch.tensor([
            float(row["auxiliary_quality"]) if row.get("auxiliary_quality") is not None else 0.0
            for row in rows
        ], dtype=torch.float32),
        "quality_mask": torch.tensor([row.get("auxiliary_quality") is not None for row in rows], dtype=torch.bool),
    }


def build_optimizer_parameter_groups(model: nn.Module, config: ExperimentConfig) -> List[dict]:
    """Return disjoint language, geometry, and auxiliary parameter groups."""

    groups = []
    for name, module, learning_rate in (
        ("language_model", model.language_model, config.training.learning_rate_language_model),
        ("geometry_encoder", model.geometry_encoder, config.training.learning_rate_geometry),
        ("auxiliary_head", model.auxiliary_head, config.training.learning_rate_auxiliary_head),
    ):
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if parameters:
            groups.append({"name": name, "params": parameters, "lr": learning_rate})
    identities = [id(parameter) for group in groups for parameter in group["params"]]
    if len(identities) != len(set(identities)):
        raise RuntimeError("optimizer parameter groups overlap")
    return groups


def _scheduler(optimizer: torch.optim.Optimizer, config: ExperimentConfig, total_steps: int):
    warmup_steps = int(total_steps * config.training.warmup_ratio)

    def scale(index: int) -> float:
        if warmup_steps and index < warmup_steps:
            return float(index + 1) / float(warmup_steps)
        progress = float(index - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        if config.training.scheduler == "constant":
            return 1.0
        if config.training.scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _effective_auxiliary_loss_weight(config: ExperimentConfig) -> float:
    """Resolve the two documented auxiliary-loss controls without leaving either dead."""

    if not config.model.auxiliary_head.enabled:
        return 0.0
    return (
        config.training.auxiliary_loss_weight
        * config.model.auxiliary_head.loss_weight
    )


def _sampling_ratios(config: ExperimentConfig) -> Dict[str, float]:
    return {
        "positive_exact": float(config.sampling.positive_exact_ratio or 0.0),
        "positive_jittered": float(config.sampling.jittered_positive_ratio or 0.0),
        "negative_hard": float(config.sampling.hard_negative_ratio or 0.0),
        "negative_local": float(config.sampling.local_negative_ratio or 0.0),
        "negative_random": float(config.sampling.random_negative_ratio or 0.0),
    }


def validate_training_rows_against_config(
    rows: Sequence[Mapping[str, object]],
    config: ExperimentConfig,
    *,
    legacy_input: bool = False,
) -> Dict[str, object]:
    """Bind prepared-data provenance to the effective training configuration.

    Sampling ratios describe upstream construction, so the training runtime does
    not silently re-sample an already prepared dataset.  It does, however, reject
    sample kinds disabled by the configuration and verifies the serializer's
    requested history length for every non-legacy row.
    """

    ratios = _sampling_ratios(config)
    counts = {name: 0 for name in ratios}
    history_serialized = []
    for index, row in enumerate(rows):
        sampling_type = str(row.get("sampling_type"))
        if sampling_type not in counts:
            raise PreparedDataError(
                "row {0} has an unsupported sampling_type".format(index + 1)
            )
        if ratios[sampling_type] <= 0.0:
            raise PreparedDataError(
                "row {0} uses sampling_type {1!r}, but its configured ratio is zero".format(
                    index + 1, sampling_type
                )
            )
        counts[sampling_type] += 1
        metadata = row.get("prompt_metadata")
        if not isinstance(metadata, Mapping):
            raise PreparedDataError("row {0} lacks prompt_metadata".format(index + 1))
        if not legacy_input:
            requested = metadata.get("history_requested")
            if requested != config.dataset.history_length:
                raise PreparedDataError(
                    "row {0} history_requested does not match dataset.history_length".format(
                        index + 1
                    )
                )
        serialized = metadata.get("history_serialized")
        if isinstance(serialized, bool) or not isinstance(serialized, int) or serialized < 0:
            raise PreparedDataError(
                "row {0} has invalid history_serialized provenance".format(index + 1)
            )
        if serialized > config.dataset.history_length:
            raise PreparedDataError(
                "row {0} serializes more history than configured".format(index + 1)
            )
        if int(row["box_count"]) != serialized + 1:
            raise PreparedDataError(
                "row {0} box_count is inconsistent with serialized history".format(index + 1)
            )
        history_serialized.append(serialized)
        if sampling_type == "positive_jittered":
            if not config.sampling.positive_jitter_enabled:
                raise PreparedDataError("jittered positives are disabled by configuration")
            if config.sampling.positive_jitter_std is None:
                raise PreparedDataError("positive_jitter_std is required for jittered positives")
    return {
        "history_length": config.dataset.history_length,
        "observed_history_serialized": sorted(set(history_serialized)),
        "configured_sampling_ratios": ratios,
        "observed_sampling_counts": counts,
        "positive_jitter_enabled": config.sampling.positive_jitter_enabled,
        "positive_jitter_std": config.sampling.positive_jitter_std,
        "effective_auxiliary_loss_weight": _effective_auxiliary_loss_weight(config),
    }


def _effective_config(
    config: ExperimentConfig,
    backend: str,
    steps_override: Optional[int],
    runtime_binding: Mapping[str, object],
) -> dict:
    document = asdict(config)
    document["backend"] = backend
    document["max_steps_override"] = steps_override
    document["effective_max_steps"] = steps_override or config.training.max_steps
    document["runtime_binding"] = dict(runtime_binding)
    return document


def _write_effective_config(
    output_dir: Path,
    config: ExperimentConfig,
    backend: str,
    steps: Optional[int],
    runtime_binding: Mapping[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "effective_config.json").write_text(
        json.dumps(
            _effective_config(config, backend, steps, runtime_binding),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _tiny_train(
    rows: List[dict],
    output_dir: Path,
    config: ExperimentConfig,
    steps_override: Optional[int],
) -> Dict[str, object]:
    seed_everything(config.training.seed)
    model = TinyAssociationBackend()
    groups = build_optimizer_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.training.weight_decay)
    loader = DataLoader(
        _Rows(rows),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=_collate_rows,
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    batches_per_epoch = max(len(loader), 1)
    updates_per_epoch = math.ceil(batches_per_epoch / config.training.gradient_accumulation_steps)
    configured_steps = config.training.max_steps or updates_per_epoch * config.training.epochs
    max_steps = steps_override or configured_steps
    scheduler = _scheduler(optimizer, config, max_steps)
    start_epoch = 0
    global_step = 0
    resume_path = Path(config.training.resume_from) if config.training.resume_from else None
    if resume_path is not None:
        model_path = resume_path / "tiny_model.safetensors"
        state_path = resume_path / "tiny_training_state.json"
        unsafe_path = resume_path / "tiny_training_state.pt"
        if unsafe_path.exists() and not (model_path.is_file() and state_path.is_file()):
            raise RuntimeError(
                "legacy pickle resume state is not accepted; start from a safetensors+JSON checkpoint"
            )
        if any(path.is_symlink() or not path.is_file() for path in (model_path, state_path)):
            raise FileNotFoundError("tiny safetensors+JSON resume state was not found")
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("resume_scope") != "weights_and_counters":
            raise RuntimeError("tiny resume metadata has an unsupported resume_scope")
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        scheduler.last_epoch = global_step
        for group, base_lr, scale in zip(
            optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas
        ):
            group["lr"] = base_lr * scale(global_step)
    losses = []
    optimizer.zero_grad()
    for epoch in range(start_epoch, config.training.epochs):
        for batch_index, batch in enumerate(loader):
            logits, prediction = model(batch["candidate_geometry"])
            language = functional.cross_entropy(logits, batch["targets"])
            if batch["quality_mask"].any().item():
                auxiliary = functional.mse_loss(
                    prediction[batch["quality_mask"]], batch["quality"][batch["quality_mask"]]
                )
            else:
                auxiliary = prediction.sum() * 0.0
            loss = (
                config.training.language_loss_weight * language
                + _effective_auxiliary_loss_weight(config) * auxiliary
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError("training loss became non-finite")
            (loss / config.training.gradient_accumulation_steps).backward()
            should_update = (
                (batch_index + 1) % config.training.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                losses.append(float(loss.detach()))
                if global_step >= max_steps:
                    break
        if (epoch + 1) % config.training.checkpoint_frequency == 0 or global_step >= max_steps:
            checkpoint_dir = output_dir / "checkpoint-epoch-{0}".format(epoch + 1)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from safetensors.torch import save_file

            save_file(
                {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
                str(checkpoint_dir / "tiny_model.safetensors"),
            )
            (checkpoint_dir / "tiny_training_state.json").write_text(
                json.dumps({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "resume_scope": "weights_and_counters",
                    "optimizer_state_restored": False,
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if global_step >= max_steps:
            break
    if not losses:
        raise RuntimeError("training performed no optimizer updates")
    quality_values = [float(row["auxiliary_quality"]) for row in rows if row.get("auxiliary_quality") is not None]
    summary = {
        "backend": "tiny",
        "steps": global_step,
        "epochs_completed": epoch + 1,
        "final_loss": losses[-1],
        "samples": len(rows),
        "samples_with_auxiliary_target": len(quality_values),
        "samples_without_auxiliary_target": len(rows) - len(quality_values),
        "auxiliary_quality_min": min(quality_values) if quality_values else None,
        "auxiliary_quality_max": max(quality_values) if quality_values else None,
        "optimizer_groups": {
            group["name"]: group.get("initial_lr", group["lr"]) for group in groups
        },
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "batch_size": config.training.batch_size,
        "effective_auxiliary_loss_weight": _effective_auxiliary_loss_weight(config),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_supervised_token_ids(
    tokenizer: Any,
    prompt: str,
    target: str,
    token_ids: ResolvedTokenIds,
) -> Tuple[List[int], List[int]]:
    """Use the exact decision IDs shared with inference, without re-tokenizing targets."""

    if target not in {"Yes", "No"}:
        raise ValueError("target must be Yes or No")
    token_ids.validate()
    target_ids = token_ids.positive if target == "Yes" else token_ids.negative
    if len(target_ids) != 1:
        raise ValueError(
            "training requires one configured decision token per target; token_mode=both is inference-only"
        )
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    target_ids = list(target_ids)
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if input_ids[: len(prompt_ids)] != prompt_ids or labels[len(prompt_ids):] != target_ids:
        raise RuntimeError("explicit target boundary validation failed")
    return input_ids, labels


def _real_collate(tokenizer: Any, token_ids: ResolvedTokenIds, rows: Sequence[dict]) -> dict:
    encoded = [
        build_supervised_token_ids(tokenizer, row["prompt"], row["target"], token_ids)
        for row in rows
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_length = max(len(ids) for ids, _ in encoded)
    input_ids, labels, attention = [], [], []
    geometry, counts = [], []
    quality, quality_mask = [], []
    for row, (ids, row_labels) in zip(rows, encoded):
        padding = max_length - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        labels.append(row_labels + [-100] * padding)
        attention.append([1] * len(ids) + [0] * padding)
        geometry.extend(row["geometry"])
        counts.append(int(row["box_count"]))
        quality.append(float(row["auxiliary_quality"]) if row.get("auxiliary_quality") is not None else 0.0)
        quality_mask.append(row.get("auxiliary_quality") is not None)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "geometry": torch.tensor(geometry, dtype=torch.float32),
        "box_counts": torch.tensor(counts, dtype=torch.long),
        "quality": torch.tensor(quality, dtype=torch.float32),
        "quality_mask": torch.tensor(quality_mask, dtype=torch.bool),
    }


def _candidate_indices(box_counts: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cumsum(box_counts, dim=0)
    return cumulative - 1


def _save_real_checkpoint(model: nn.Module, output_dir: Path, epoch: int, global_step: int) -> None:
    checkpoint_dir = output_dir / "checkpoint-epoch-{0}".format(epoch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.language_model, "save_pretrained"):
        model.language_model.save_pretrained(checkpoint_dir, safe_serialization=True)
    from safetensors.torch import save_file

    components = {}
    for prefix, module in (("geo_encoder", model.geometry_encoder), ("score_head", model.auxiliary_head.network)):
        for name, value in module.state_dict().items():
            components[prefix + "." + name] = value.detach().cpu().contiguous()
    save_file(components, str(checkpoint_dir / "geo_components.safetensors"))
    (checkpoint_dir / "training_state.json").write_text(
        json.dumps({
            "epoch": epoch,
            "global_step": global_step,
            "resume_scope": "weights_and_counters; optimizer state intentionally not deserialized",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _real_train(
    rows: List[dict],
    output_dir: Path,
    config: ExperimentConfig,
    model_path: str,
    checkpoint: Optional[str],
    checkpoint_sha256: Optional[str],
    device: str,
    steps_override: Optional[int],
    allow_download: bool,
    vocab_alignment: str,
) -> Dict[str, object]:
    seed_everything(config.training.seed)
    resume = config.training.resume_from
    load_checkpoint_path = resume or checkpoint or config.assets.checkpoint_path
    model, tokenizer, load_report = load_nova_model(
        model_path,
        checkpoint_path=load_checkpoint_path,
        checkpoint_sha256=(
            None if resume else checkpoint_sha256 or config.assets.checkpoint_sha256
        ),
        device=device,
        local_files_only=not allow_download,
        allow_download=allow_download,
        dtype=config.inference.dtype,
        training=True,
        vocab_alignment=vocab_alignment,
        model_config=config.model,
    )
    if load_checkpoint_path is None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("PEFT is required for real training") from exc
        model.language_model = get_peft_model(model.language_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.model.lora.rank,
            lora_alpha=config.model.lora.alpha,
            lora_dropout=float(config.model.lora.dropout or 0.0),
            target_modules=list(config.model.lora.target_modules),
            modules_to_save=["embed_tokens", "lm_head"],
        ))
    token_ids = resolve_token_ids(tokenizer, config.model.tokens)
    groups = build_optimizer_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.training.weight_decay)
    loader = DataLoader(
        _Rows(rows),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=lambda batch: _real_collate(tokenizer, token_ids, batch),
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    updates_per_epoch = math.ceil(max(len(loader), 1) / config.training.gradient_accumulation_steps)
    configured_steps = config.training.max_steps or updates_per_epoch * config.training.epochs
    max_steps = steps_override or configured_steps
    scheduler = _scheduler(optimizer, config, max_steps)
    start_epoch = 0
    global_step = 0
    if resume:
        state_path = Path(resume) / "training_state.json"
        if state_path.is_symlink() or not state_path.is_file():
            raise FileNotFoundError("resume training_state.json was not found")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        scheduler.last_epoch = global_step
        for group, base_lr, scale in zip(
            optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas
        ):
            group["lr"] = base_lr * scale(global_step)
    model.train()
    optimizer.zero_grad()
    losses = []
    for epoch in range(start_epoch, config.training.epochs):
        for batch_index, batch in enumerate(loader):
            dtype = model.geometry_encoder.mlp[0].weight.dtype
            inputs = AssociationModelInput(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                geometry=batch["geometry"].to(device=device, dtype=dtype),
                box_counts=batch["box_counts"].to(device),
                box_token_id=token_ids.box,
                labels=batch["labels"].to(device),
                auxiliary_target=batch["quality"].to(device),
            )
            output = model(inputs, token_ids)
            language = output.loss_components.get("language")
            if language is None:
                raise RuntimeError("language model did not return a training loss")
            candidate_scores = output.auxiliary_quality[_candidate_indices(inputs.box_counts)]
            quality_mask = batch["quality_mask"].to(device)
            if quality_mask.any().item():
                auxiliary = auxiliary_quality_loss(
                    candidate_scores[quality_mask], inputs.auxiliary_target[quality_mask]
                )
            else:
                auxiliary = candidate_scores.sum() * 0.0
            loss = (
                config.training.language_loss_weight * language
                + _effective_auxiliary_loss_weight(config) * auxiliary
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError("training loss became non-finite")
            (loss / config.training.gradient_accumulation_steps).backward()
            should_update = (
                (batch_index + 1) % config.training.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                losses.append(float(loss.detach().cpu()))
                if global_step >= max_steps:
                    break
        if (epoch + 1) % config.training.checkpoint_frequency == 0 or global_step >= max_steps:
            _save_real_checkpoint(model, output_dir, epoch + 1, global_step)
        if global_step >= max_steps:
            break
    if not losses:
        raise RuntimeError("training performed no optimizer updates")
    summary = {
        "backend": "real",
        "steps": global_step,
        "epochs_completed": epoch + 1,
        "final_loss": losses[-1],
        "samples": len(rows),
        "box_counts": sorted(set(int(row["box_count"]) for row in rows)),
        "optimizer_groups": {
            group["name"]: group.get("initial_lr", group["lr"]) for group in groups
        },
        "lora": asdict(config.model.lora),
        "load_architecture": load_report.architecture,
        "resume_scope": "weights_and_counters" if resume else None,
        "effective_auxiliary_loss_weight": _effective_auxiliary_loss_weight(config),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def train_prepared_associations(
    input_path: Path,
    output_dir: Path,
    *,
    config: ExperimentConfig,
    backend: str = "real",
    model_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    checkpoint_sha256: Optional[str] = None,
    device: str = "cpu",
    steps: Optional[int] = None,
    allow_download: bool = False,
    vocab_alignment: str = "strict",
    legacy_input: bool = False,
) -> Dict[str, object]:
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive")
    if not config.training.enabled:
        raise ValueError("training.enabled must be true to start a training runtime")
    config.validate(for_training=True)
    rows = read_association_jsonl(
        input_path,
        require_target=True,
        require_auxiliary=(
            config.model.auxiliary_head.enabled
            and config.training.auxiliary_loss_weight > 0
        ),
        legacy_input=legacy_input,
    )
    runtime_binding = validate_training_rows_against_config(
        rows, config, legacy_input=legacy_input
    )
    _write_effective_config(output_dir, config, backend, steps, runtime_binding)
    if backend == "tiny":
        return _tiny_train(rows, output_dir, config, steps)
    if backend != "real":
        raise ValueError("backend must be real or tiny")
    effective_model = model_path or config.model.base_model
    if not effective_model:
        raise ValueError("Model path is required for real training.")
    return _real_train(
        rows,
        output_dir,
        config,
        effective_model,
        checkpoint,
        checkpoint_sha256,
        device,
        steps,
        allow_download,
        vocab_alignment,
    )
