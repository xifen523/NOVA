"""Strict dataclass schema for NOVA experiments."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .validation import (
    ConfigValidationError,
    ReproducibilityWarning,
    require_positive_int,
    require_probability,
    require_type,
)


FROZEN_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class TokenMode(str, Enum):
    PLAIN = "plain"
    SPACE = "space"
    BOTH = "both"


@dataclass(frozen=True)
class GeometryConfig:
    input_dim: int = 9
    hidden_dim: int = 1024

    def validate(self) -> None:
        require_type(self.input_dim, int, "model.geometry.input_dim")
        if self.input_dim != 9:
            raise ConfigValidationError("model.geometry.input_dim must equal 9")
        require_positive_int(self.hidden_dim, "model.geometry.hidden_dim")


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: Optional[float] = None
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    def validate(self, for_training: bool = False) -> None:
        require_positive_int(self.rank, "model.lora.rank")
        require_positive_int(self.alpha, "model.lora.alpha")
        if self.dropout is None:
            if for_training:
                raise ConfigValidationError(
                    "model.lora.dropout must be explicit before training"
                )
        else:
            require_probability(
                self.dropout, "model.lora.dropout", upper_inclusive=False
            )
        if not isinstance(self.target_modules, list) or not self.target_modules:
            raise ConfigValidationError("model.lora.target_modules must be a non-empty list")
        if any(not isinstance(item, str) or not item for item in self.target_modules):
            raise ConfigValidationError("every LoRA target module must be a non-empty string")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ConfigValidationError("LoRA target modules must be unique")


@dataclass(frozen=True)
class AuxiliaryHeadConfig:
    enabled: bool = True
    hidden_dim: int = 256
    loss_weight: float = 1.0

    def validate(self) -> None:
        require_type(self.enabled, bool, "model.auxiliary_head.enabled")
        require_positive_int(self.hidden_dim, "model.auxiliary_head.hidden_dim")
        require_type(self.loss_weight, float, "model.auxiliary_head.loss_weight")
        if self.loss_weight < 0:
            raise ConfigValidationError("auxiliary loss weight must be non-negative")


@dataclass(frozen=True)
class TokenConfig:
    box: str = "<box>"
    positive: str = "Yes"
    negative: str = "No"
    token_mode: TokenMode = TokenMode.PLAIN

    def validate(self) -> None:
        for name, value in (
            ("box", self.box),
            ("positive", self.positive),
            ("negative", self.negative),
        ):
            if not isinstance(value, str) or not value:
                raise ConfigValidationError("model.tokens.{0} must be a non-empty string".format(name))
        if not isinstance(self.token_mode, TokenMode):
            raise ConfigValidationError("model.tokens.token_mode must be a TokenMode")
        if len({self.box, self.positive, self.negative}) != 3:
            raise ConfigValidationError("box, positive, and negative tokens must be distinct")


@dataclass(frozen=True)
class ModelConfig:
    base_model: str = FROZEN_BASE_MODEL
    revision: Optional[str] = None
    trust_remote_code: bool = False
    language_hidden_dim: Optional[int] = None
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    auxiliary_head: AuxiliaryHeadConfig = field(default_factory=AuxiliaryHeadConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)

    def validate(self, for_training: bool = False) -> None:
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ConfigValidationError("model.base_model must be a non-empty path or model ID")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ConfigValidationError("model.revision must be null or a non-empty string")
        if self.revision is None:
            warnings.warn(
                "model.revision is unresolved; synthetic tests are allowed but real model loading is not reproducible",
                ReproducibilityWarning,
                stacklevel=2,
            )
        require_type(self.trust_remote_code, bool, "model.trust_remote_code")
        if self.trust_remote_code:
            raise ConfigValidationError("trust_remote_code=true is not supported")
        if self.language_hidden_dim is not None:
            require_positive_int(self.language_hidden_dim, "model.language_hidden_dim")
        self.geometry.validate()
        self.lora.validate(for_training=for_training)
        self.auxiliary_head.validate()
        self.tokens.validate()


@dataclass(frozen=True)
class ClassSplitConfig:
    """Named class partitions; examples are non-authoritative by default."""

    base_classes: List[str] = field(default_factory=list)
    novel_classes: List[str] = field(default_factory=list)
    aliases: Dict[str, str] = field(default_factory=dict)
    unknown_policy: str = "as_unknown"
    authoritative: bool = False
    mapping_status: str = "REQUIRES_AUTHORITATIVE_CLASS_MAP"

    def validate(self, real_mode: bool = False) -> None:
        for field_name, values in (
            ("base_classes", self.base_classes),
            ("novel_classes", self.novel_classes),
        ):
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ConfigValidationError(
                    "dataset.class_split.{0} must be a list of non-empty strings".format(
                        field_name
                    )
                )
        if not isinstance(self.aliases, dict) or any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.aliases.items()
        ):
            raise ConfigValidationError(
                "dataset.class_split.aliases must map non-empty strings to non-empty strings"
            )
        if self.unknown_policy not in {"as_unknown", "reject"}:
            raise ConfigValidationError(
                "dataset.class_split.unknown_policy must be as_unknown or reject"
            )
        require_type(
            self.authoritative, bool, "dataset.class_split.authoritative"
        )
        allowed_statuses = {
            "SYNTHETIC_EXAMPLE",
            "REQUIRES_AUTHORITATIVE_CLASS_MAP",
            "AUTHORITATIVE",
        }
        if self.mapping_status not in allowed_statuses:
            raise ConfigValidationError(
                "dataset.class_split.mapping_status is not an allowed status"
            )
        normalize = lambda value: "_".join(value.strip().lower().split())
        base = [normalize(value) for value in self.base_classes]
        novel = [normalize(value) for value in self.novel_classes]
        if len(base) != len(set(base)) or len(novel) != len(set(novel)):
            raise ConfigValidationError("class names must be unique after normalization")
        if set(base) & set(novel):
            raise ConfigValidationError("base and novel class sets must be disjoint")
        known = set(base) | set(novel)
        normalized_aliases = [normalize(value) for value in self.aliases]
        if len(normalized_aliases) != len(set(normalized_aliases)):
            raise ConfigValidationError("class aliases must be unique after normalization")
        if any(normalize(target) not in known for target in self.aliases.values()):
            raise ConfigValidationError("every class alias must resolve to a known class")
        if self.authoritative != (self.mapping_status == "AUTHORITATIVE"):
            raise ConfigValidationError(
                "authoritative must be true exactly when mapping_status is AUTHORITATIVE"
            )
        if real_mode and not self.authoritative:
            raise ConfigValidationError(
                "real dataset mode requires an authoritative class map"
            )


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "unconfigured"
    history_length: int = 3
    coordinate_frame: str = "unconfigured"
    real_mode: bool = False
    class_split: ClassSplitConfig = field(default_factory=ClassSplitConfig)

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ConfigValidationError("dataset.name must be a non-empty string")
        require_type(self.history_length, int, "dataset.history_length")
        if self.history_length < 0:
            raise ConfigValidationError("dataset.history_length must be non-negative")
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame:
            raise ConfigValidationError("dataset.coordinate_frame must be a non-empty string")
        require_type(self.real_mode, bool, "dataset.real_mode")
        self.class_split.validate(real_mode=self.real_mode)


@dataclass(frozen=True)
class PromptingConfig:
    mode: str = "hybrid"
    novel_label: str = "Unknown"
    mask_novel_classes: bool = True
    preserve_base_classes: bool = True

    def validate(self) -> None:
        if self.mode not in {"hybrid", "raw"}:
            raise ConfigValidationError("prompting.mode must be hybrid or raw")
        if not isinstance(self.novel_label, str) or not self.novel_label.strip():
            raise ConfigValidationError("prompting.novel_label must be non-empty")
        require_type(self.mask_novel_classes, bool, "prompting.mask_novel_classes")
        require_type(self.preserve_base_classes, bool, "prompting.preserve_base_classes")


@dataclass(frozen=True)
class SamplingConfig:
    positive_exact_ratio: Optional[float] = None
    jittered_positive_ratio: Optional[float] = None
    hard_negative_ratio: Optional[float] = None
    local_negative_ratio: Optional[float] = None
    random_negative_ratio: Optional[float] = None
    positive_jitter_enabled: bool = True
    positive_jitter_std: Optional[float] = None
    hard_k: int = 3
    local_radius_m: Optional[float] = None

    def validate(self, for_training: bool = False) -> None:
        ratios = (
            self.positive_exact_ratio,
            self.jittered_positive_ratio,
            self.hard_negative_ratio,
            self.local_negative_ratio,
            self.random_negative_ratio,
        )
        if for_training and any(value is None for value in ratios):
            raise ConfigValidationError("all sampling ratios must be explicit for training")
        present = [float(value) for value in ratios if value is not None]
        for value in present:
            require_probability(value, "sampling ratio")
        if present and abs(sum(present) - 1.0) > 1e-8:
            raise ConfigValidationError("explicit sampling ratios must sum to 1")
        require_type(self.positive_jitter_enabled, bool, "sampling.positive_jitter_enabled")
        if self.positive_jitter_enabled and for_training and self.positive_jitter_std is None:
            raise ConfigValidationError("sampling.positive_jitter_std must be explicit")
        if self.positive_jitter_std is not None:
            require_type(self.positive_jitter_std, float, "sampling.positive_jitter_std")
            if self.positive_jitter_std < 0:
                raise ConfigValidationError("sampling.positive_jitter_std must be non-negative")
        require_positive_int(self.hard_k, "sampling.hard_k")
        if self.local_negative_ratio not in {None, 0, 0.0}:
            if self.local_radius_m is None:
                raise ConfigValidationError("sampling.local_radius_m is required for local negatives")
        if self.local_radius_m is not None:
            require_type(self.local_radius_m, float, "sampling.local_radius_m")
            if self.local_radius_m < 0:
                raise ConfigValidationError("sampling.local_radius_m must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    enabled: bool = False
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate_geometry: float = 2e-4
    learning_rate_language_model: float = 5e-5
    learning_rate_auxiliary_head: float = 2e-4
    epochs: int = 10
    max_steps: Optional[int] = None
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    scheduler: str = "cosine"
    gradient_clip_norm: float = 1.0
    language_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 1.0
    checkpoint_frequency: int = 1
    resume_from: Optional[str] = None
    num_workers: int = 0
    seed: int = 0

    def validate(self) -> None:
        require_type(self.enabled, bool, "training.enabled")
        require_positive_int(self.batch_size, "training.batch_size")
        require_positive_int(
            self.gradient_accumulation_steps,
            "training.gradient_accumulation_steps",
        )
        require_type(self.learning_rate_geometry, float, "training.learning_rate_geometry")
        require_type(
            self.learning_rate_language_model,
            float,
            "training.learning_rate_language_model",
        )
        require_type(self.learning_rate_auxiliary_head, float, "training.learning_rate_auxiliary_head")
        if (
            self.learning_rate_geometry <= 0
            or self.learning_rate_language_model <= 0
            or self.learning_rate_auxiliary_head <= 0
        ):
            raise ConfigValidationError("training learning rates must be positive")
        require_positive_int(self.epochs, "training.epochs")
        if self.max_steps is not None:
            require_positive_int(self.max_steps, "training.max_steps")
        for name, value in (
            ("weight_decay", self.weight_decay),
            ("gradient_clip_norm", self.gradient_clip_norm),
            ("language_loss_weight", self.language_loss_weight),
            ("auxiliary_loss_weight", self.auxiliary_loss_weight),
        ):
            require_type(value, float, "training.{0}".format(name))
            if value < 0:
                raise ConfigValidationError("training.{0} must be non-negative".format(name))
        require_probability(self.warmup_ratio, "training.warmup_ratio")
        if self.scheduler not in {"linear", "cosine", "constant"}:
            raise ConfigValidationError("training.scheduler is invalid")
        require_positive_int(self.checkpoint_frequency, "training.checkpoint_frequency")
        if self.resume_from is not None and (
            not isinstance(self.resume_from, str) or not self.resume_from.strip()
        ):
            raise ConfigValidationError("training.resume_from must be null or a non-empty path")
        require_type(self.num_workers, int, "training.num_workers")
        if self.num_workers < 0:
            raise ConfigValidationError("training.num_workers must be non-negative")
        require_type(self.seed, int, "training.seed")
        if self.seed < 0:
            raise ConfigValidationError("training.seed must be non-negative")


@dataclass(frozen=True)
class InferenceConfig:
    yes_threshold: float = 0.5
    birth_threshold: float = 0.1
    detection_threshold: float = 0.1
    detection_threshold_by_class: Dict[str, float] = field(default_factory=dict)
    distance_gate: Optional[float] = None
    distance_gate_by_class: Dict[str, float] = field(default_factory=dict)
    distance_weight: float = 0.1
    class_gate: bool = False
    max_age: int = 3
    min_hits: int = 1
    lost_score_decay: float = 1.0
    emission_mode: str = "active_only"
    score_mode: str = "detection"
    class_gate_policy: str = "exact_alias"
    pair_batch_size: int = 32
    max_pairs_per_forward: int = 32
    dtype: str = "auto"

    def validate(self) -> None:
        require_probability(self.yes_threshold, "inference.yes_threshold")
        require_probability(self.birth_threshold, "inference.birth_threshold")
        require_probability(self.detection_threshold, "inference.detection_threshold")
        if not isinstance(self.detection_threshold_by_class, dict):
            raise ConfigValidationError("inference.detection_threshold_by_class must be a mapping")
        for category, value in self.detection_threshold_by_class.items():
            if not isinstance(category, str) or not category.strip():
                raise ConfigValidationError("detection threshold class names must be non-empty")
            require_probability(value, "inference.detection_threshold_by_class")
        if self.distance_gate is not None:
            require_type(self.distance_gate, float, "inference.distance_gate")
            if self.distance_gate < 0:
                raise ConfigValidationError("inference.distance_gate must be non-negative")
        if not isinstance(self.distance_gate_by_class, dict):
            raise ConfigValidationError("inference.distance_gate_by_class must be a mapping")
        for category, value in self.distance_gate_by_class.items():
            if not isinstance(category, str) or not category.strip():
                raise ConfigValidationError("distance gate class names must be non-empty")
            require_type(value, float, "inference.distance_gate_by_class")
            if value < 0:
                raise ConfigValidationError("class distance gates must be non-negative")
        require_type(self.distance_weight, float, "inference.distance_weight")
        if self.distance_weight < 0:
            raise ConfigValidationError("inference.distance_weight must be non-negative")
        require_type(self.class_gate, bool, "inference.class_gate")
        require_type(self.max_age, int, "inference.max_age")
        if self.max_age < 0:
            raise ConfigValidationError("inference.max_age must be non-negative")
        require_positive_int(self.min_hits, "inference.min_hits")
        require_probability(self.lost_score_decay, "inference.lost_score_decay")
        if self.emission_mode not in {"active_only", "active_and_lost"}:
            raise ConfigValidationError("inference.emission_mode is invalid")
        if self.score_mode not in {"detection", "association", "decayed_detection"}:
            raise ConfigValidationError("inference.score_mode is invalid")
        if self.class_gate_policy not in {"exact_alias", "base_novel", "disabled"}:
            raise ConfigValidationError("inference.class_gate_policy is invalid")
        require_positive_int(self.pair_batch_size, "inference.pair_batch_size")
        require_positive_int(self.max_pairs_per_forward, "inference.max_pairs_per_forward")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ConfigValidationError("inference.dtype is invalid")


@dataclass(frozen=True)
class ReproductionConfig:
    dataset_version: str = "UNRESOLVED"
    split: str = "UNRESOLVED"
    detector: str = "UNRESOLVED"
    coordinate_convention: str = "UNRESOLVED"
    history_policy: str = "UNRESOLVED"
    geometry_definition: str = "UNRESOLVED"
    prompt_policy: str = "UNRESOLVED"
    token_policy: str = "UNRESOLVED"
    association_cost: str = "UNRESOLVED"
    gate_policy: str = "UNRESOLVED"
    lifecycle_policy: str = "UNRESOLVED"
    iou_threshold: float = 0.25
    model_identifier: str = "UNRESOLVED"
    checkpoint_identifier: str = "UNRESOLVED"
    evaluator: str = "UNRESOLVED"
    evaluator_status: str = "EXTERNAL_EVALUATOR_REQUIRED"
    known_unresolved_fields: List[str] = field(default_factory=list)

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if name in {"iou_threshold", "known_unresolved_fields"}:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError("reproduction.{0} must be non-empty".format(name))
        require_probability(self.iou_threshold, "reproduction.iou_threshold")
        if any(not isinstance(item, str) or not item.strip() for item in self.known_unresolved_fields):
            raise ConfigValidationError("known_unresolved_fields must contain non-empty strings")


@dataclass(frozen=True)
class AssetConfig:
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    detector_asset_path: Optional[str] = None
    detector_asset_sha256: Optional[str] = None

    def validate(self) -> None:
        for name, value in (
            ("checkpoint_path", self.checkpoint_path),
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("detector_asset_path", self.detector_asset_path),
            ("detector_asset_sha256", self.detector_asset_sha256),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigValidationError("assets.{0} must be null or a non-empty string".format(name))


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    split: str = "validation"
    evaluator_name: Optional[str] = None
    evaluator_version: Optional[str] = None
    evaluator_command: List[str] = field(default_factory=list)
    output_parser: str = "json"

    def validate(self) -> None:
        require_type(self.enabled, bool, "evaluation.enabled")
        if not isinstance(self.split, str) or not self.split:
            raise ConfigValidationError("evaluation.split must be a non-empty string")
        for name, value in (
            ("evaluator_name", self.evaluator_name),
            ("evaluator_version", self.evaluator_version),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigValidationError("evaluation.{0} must be null or non-empty".format(name))
        if not isinstance(self.evaluator_command, list) or any(
            not isinstance(item, str) or not item for item in self.evaluator_command
        ):
            raise ConfigValidationError("evaluation.evaluator_command must be a string list")
        if self.enabled and (not self.evaluator_name or not self.evaluator_command):
            raise ConfigValidationError("enabled evaluation requires evaluator_name and command")
        if self.output_parser not in {"json", "key_value"}:
            raise ConfigValidationError("evaluation.output_parser is invalid")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    experiment_name: str = "synthetic-default"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    assets: AssetConfig = field(default_factory=AssetConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)

    def validate(self, for_training: Optional[bool] = None) -> None:
        if not isinstance(self.experiment_name, str) or not self.experiment_name:
            raise ConfigValidationError("experiment_name must be a non-empty string")
        self.dataset.validate()
        self.prompting.validate()
        self.training.validate()
        self.inference.validate()
        self.evaluation.validate()
        self.assets.validate()
        self.reproduction.validate()
        training_mode = self.training.enabled if for_training is None else for_training
        self.sampling.validate(for_training=training_mode)
        self.model.validate(for_training=training_mode)
