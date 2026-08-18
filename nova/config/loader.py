"""Safe YAML loader with unknown-field rejection at every level."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from .schema import (
    AuxiliaryHeadConfig,
    AssetConfig,
    ClassSplitConfig,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    GeometryConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    PromptingConfig,
    ReproductionConfig,
    SamplingConfig,
    TokenConfig,
    TokenMode,
    TrainingConfig,
)
from .validation import ConfigValidationError, reject_unknown, require_mapping


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return require_mapping(root.get(name, {}), name)


def _build_config(document: Any, for_training: Optional[bool]) -> ExperimentConfig:
    root = require_mapping(document, "root")
    reject_unknown(
        root,
        {
            "experiment_name", "model", "dataset", "prompting", "sampling",
            "training", "inference", "evaluation", "assets", "reproduction",
        },
        "root",
    )
    if "model" not in root:
        raise ConfigValidationError("required field root.model is missing")

    model_data = _section(root, "model")
    reject_unknown(
        model_data,
        {
            "base_model", "revision", "trust_remote_code", "language_hidden_dim",
            "geometry", "lora", "auxiliary_head", "tokens",
        },
        "model",
    )
    geometry_data = _section(model_data, "geometry")
    reject_unknown(geometry_data, {"input_dim", "hidden_dim"}, "model.geometry")
    lora_data = _section(model_data, "lora")
    reject_unknown(
        lora_data,
        {"rank", "alpha", "dropout", "target_modules"},
        "model.lora",
    )
    auxiliary_data = _section(model_data, "auxiliary_head")
    reject_unknown(
        auxiliary_data,
        {"enabled", "hidden_dim", "loss_weight"},
        "model.auxiliary_head",
    )
    token_data = _section(model_data, "tokens")
    reject_unknown(
        token_data,
        {"box", "positive", "negative", "token_mode"},
        "model.tokens",
    )
    token_kwargs = dict(token_data)
    if "token_mode" in token_kwargs:
        try:
            token_kwargs["token_mode"] = TokenMode(token_kwargs["token_mode"])
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                "model.tokens.token_mode must be plain, space, or both"
            ) from exc

    model_kwargs = {
        key: value
        for key, value in model_data.items()
        if key not in {"geometry", "lora", "auxiliary_head", "tokens"}
    }
    model = ModelConfig(
        geometry=GeometryConfig(**dict(geometry_data)),
        lora=LoRAConfig(**dict(lora_data)),
        auxiliary_head=AuxiliaryHeadConfig(**dict(auxiliary_data)),
        tokens=TokenConfig(**token_kwargs),
        **model_kwargs
    )

    dataset_data = _section(root, "dataset")
    reject_unknown(
        dataset_data,
        {"name", "history_length", "coordinate_frame", "real_mode", "class_split"},
        "dataset",
    )
    class_split_data = _section(dataset_data, "class_split")
    reject_unknown(
        class_split_data,
        {
            "base_classes", "novel_classes", "aliases", "unknown_policy",
            "authoritative", "mapping_status",
        },
        "dataset.class_split",
    )
    dataset_kwargs = {
        key: value for key, value in dataset_data.items() if key != "class_split"
    }
    training_data = _section(root, "training")
    reject_unknown(
        training_data,
        {
            "enabled", "batch_size", "gradient_accumulation_steps",
            "learning_rate_geometry", "learning_rate_language_model",
            "learning_rate_auxiliary_head", "epochs", "max_steps", "weight_decay",
            "warmup_ratio", "scheduler", "gradient_clip_norm", "language_loss_weight",
            "auxiliary_loss_weight", "checkpoint_frequency", "resume_from",
            "num_workers", "seed",
        },
        "training",
    )
    prompting_data = _section(root, "prompting")
    reject_unknown(
        prompting_data,
        {"mode", "novel_label", "mask_novel_classes", "preserve_base_classes"},
        "prompting",
    )
    sampling_data = _section(root, "sampling")
    reject_unknown(
        sampling_data,
        {
            "positive_exact_ratio", "jittered_positive_ratio", "hard_negative_ratio",
            "local_negative_ratio", "random_negative_ratio", "positive_jitter_enabled",
            "positive_jitter_std", "hard_k", "local_radius_m",
        },
        "sampling",
    )
    inference_data = _section(root, "inference")
    reject_unknown(
        inference_data,
        {
            "yes_threshold", "birth_threshold", "detection_threshold",
            "detection_threshold_by_class", "distance_gate", "distance_gate_by_class",
            "distance_weight", "class_gate", "max_age", "min_hits",
            "lost_score_decay", "emission_mode", "score_mode",
            "class_gate_policy", "pair_batch_size", "max_pairs_per_forward", "dtype",
        },
        "inference",
    )
    evaluation_data = _section(root, "evaluation")
    reject_unknown(
        evaluation_data,
        {
            "enabled", "split", "evaluator_name", "evaluator_version",
            "evaluator_command", "output_parser",
        },
        "evaluation",
    )
    asset_data = _section(root, "assets")
    reject_unknown(
        asset_data,
        {"checkpoint_path", "checkpoint_sha256", "detector_asset_path", "detector_asset_sha256"},
        "assets",
    )
    reproduction_data = _section(root, "reproduction")
    reject_unknown(
        reproduction_data,
        {
            "dataset_version", "split", "detector", "coordinate_convention",
            "history_policy", "geometry_definition", "prompt_policy", "token_policy",
            "association_cost", "gate_policy", "lifecycle_policy", "iou_threshold",
            "model_identifier", "checkpoint_identifier", "evaluator", "evaluator_status",
            "known_unresolved_fields",
        },
        "reproduction",
    )

    config = ExperimentConfig(
        experiment_name=root.get("experiment_name", "synthetic-default"),
        model=model,
        dataset=DatasetConfig(
            class_split=ClassSplitConfig(**dict(class_split_data)),
            **dataset_kwargs
        ),
        prompting=PromptingConfig(**dict(prompting_data)),
        sampling=SamplingConfig(**dict(sampling_data)),
        training=TrainingConfig(**dict(training_data)),
        inference=InferenceConfig(**dict(inference_data)),
        evaluation=EvaluationConfig(**dict(evaluation_data)),
        assets=AssetConfig(**dict(asset_data)),
        reproduction=ReproductionConfig(**dict(reproduction_data)),
    )
    config.validate(for_training=for_training)
    return config


def load_config(path: Union[str, Path], for_training: Optional[bool] = None) -> ExperimentConfig:
    """Load one local YAML file without includes, interpolation, or code execution."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return _build_config(document, for_training=for_training)


def load_config_text(text: str, for_training: Optional[bool] = None) -> ExperimentConfig:
    """Load YAML text for tests and caller-owned configuration sources."""

    document = yaml.safe_load(text)
    return _build_config(document, for_training=for_training)
