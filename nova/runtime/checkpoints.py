"""Safe checkpoint inspection and loading for NOVA components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from .local_compatibility import read_safetensors_metadata

from .assets import AllowedOperation, AssetRecord


class CheckpointPolicyError(ValueError):
    pass


class CheckpointLoadError(RuntimeError):
    def __init__(self, message: str, report: "CheckpointLoadReport") -> None:
        super().__init__(message)
        self.report = report


class CheckpointFormat(str, Enum):
    SAFETENSORS = "SAFETENSORS"
    TORCH_STATE_DICT = "TORCH_STATE_DICT"
    PEFT_ADAPTER = "PEFT_ADAPTER"
    GEOMETRY_COMPONENTS = "GEOMETRY_COMPONENTS"
    DETECTOR_PICKLE = "DETECTOR_PICKLE"
    UNTRUSTED_PICKLE = "UNTRUSTED_PICKLE"


@dataclass(frozen=True)
class CheckpointDecision:
    checkpoint_format: CheckpointFormat
    policy_ready: bool
    execution_enabled: bool
    reason: str


@dataclass(frozen=True)
class CheckpointInspection:
    checkpoint_format: CheckpointFormat
    embedding_rows: Optional[int]
    files: Tuple[str, ...]
    tree_sha256: Optional[str] = None
    file_sha256: Tuple[Tuple[str, int, str], ...] = ()


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_format: str
    loaded_components: Tuple[str, ...]
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    shape_mismatches: Tuple[str, ...]
    warnings: Tuple[str, ...]
    skipped_keys: Tuple[str, ...] = ()
    aligned_keys: Tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> "CheckpointLoadReport":
        return cls("NONE", (), (), (), (), ())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: Optional[str]) -> None:
    if expected is not None:
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected.lower()):
            raise CheckpointPolicyError("expected SHA256 must contain 64 hexadecimal characters")
        if _sha256(path) != expected.lower():
            raise CheckpointPolicyError("checkpoint SHA256 mismatch")


def _checkpoint_files(path: Path) -> Tuple[Path, ...]:
    if path.is_symlink() or not path.exists():
        raise CheckpointPolicyError("checkpoint path was not found or is a symlink")
    if path.is_file():
        return (path,)
    root = path.resolve()
    for child in path.rglob("*"):
        if child.is_symlink():
            raise CheckpointPolicyError(
                "checkpoint child symlinks are forbidden: {0}".format(
                    child.relative_to(path).as_posix()
                )
            )
        resolved_child = child.resolve()
        try:
            resolved_child.relative_to(root)
        except ValueError as exc:
            raise CheckpointPolicyError("checkpoint child escapes checkpoint root") from exc
    allowed = ("adapter_config.json", "adapter_model.safetensors", "geo_components.safetensors", "geo_components.pt")
    files = []
    for name in allowed:
        candidate = path / name
        if not candidate.exists():
            continue
        if candidate.is_symlink():
            raise CheckpointPolicyError("checkpoint child symlinks are forbidden: {0}".format(name))
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise CheckpointPolicyError("checkpoint child escapes checkpoint root") from exc
        if not candidate.is_file():
            raise CheckpointPolicyError("checkpoint child is not a regular file")
        files.append(candidate)
    return tuple(files)


def checkpoint_tree_manifest(path: Path) -> Tuple[Tuple[Tuple[str, int, str], ...], str]:
    """Return per-file size/hash records and a deterministic directory tree hash."""

    _checkpoint_files(path)
    root = path if path.is_dir() else path.parent
    files = (
        tuple(sorted((item for item in path.rglob("*") if item.is_file()), key=lambda value: value.as_posix()))
        if path.is_dir()
        else (path,)
    )
    records = tuple(
        (item.relative_to(root).as_posix(), item.stat().st_size, _sha256(item))
        for item in sorted(files, key=lambda value: value.name)
    )
    digest = hashlib.sha256()
    for name, size, file_hash in records:
        digest.update("{0}\t{1}\t{2}\n".format(name, size, file_hash).encode("utf-8"))
    return records, digest.hexdigest()


def inspect_checkpoint(path: Path, *, expected_sha256: Optional[str] = None) -> CheckpointInspection:
    files = _checkpoint_files(path)
    if not files:
        raise CheckpointPolicyError("checkpoint contains no supported files")
    records, tree_hash = checkpoint_tree_manifest(path)
    if expected_sha256 is not None:
        if path.is_file():
            _verify_hash(files[0], expected_sha256)
        elif tree_hash != expected_sha256.lower():
            raise CheckpointPolicyError("checkpoint directory tree SHA256 mismatch")
    adapter = next((item for item in files if item.name == "adapter_model.safetensors"), None)
    embedding_rows = None
    if adapter is not None:
        metadata = read_safetensors_metadata(adapter)
        for key in (
            "base_model.model.model.embed_tokens.weight",
            "base_model.model.lm_head.weight",
        ):
            shape = metadata.shapes.get(key)
            if shape is not None:
                embedding_rows = int(shape[0])
                break
    checkpoint_format = CheckpointFormat.PEFT_ADAPTER if adapter is not None else classify_checkpoint(str(files[0]))
    return CheckpointInspection(
        checkpoint_format,
        embedding_rows,
        tuple(item.name for item in files),
        tree_hash,
        records,
    )


def _state_compatibility(module, state: Dict[str, torch.Tensor]):
    expected = module.state_dict()
    missing = tuple(sorted(set(expected) - set(state)))
    unexpected = tuple(sorted(set(state) - set(expected)))
    mismatches = tuple(sorted(
        "{0}: expected {1}, found {2}".format(
            key, tuple(expected[key].shape), tuple(state[key].shape)
        )
        for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    ))
    mismatch_names = {item.split(":", 1)[0] for item in mismatches}
    aligned = tuple(sorted(
        key for key in set(expected) & set(state) if key not in mismatch_names
    ))
    skipped = tuple(sorted(set(unexpected) | mismatch_names))
    return missing, unexpected, mismatches, aligned, skipped


def _load_component_state(path: Path, allow_legacy_torch: bool) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
        return dict(state)
    if not allow_legacy_torch:
        raise CheckpointPolicyError(
            "legacy .pt component loading is disabled; pass allow_legacy_torch_checkpoint after reviewing provenance"
        )
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(key, str) for key in state):
        raise CheckpointPolicyError("legacy component checkpoint must contain a string-keyed state dictionary")
    return state


def _component_groups(state: dict) -> Tuple[dict, dict]:
    """Accept legacy nested dictionaries or portable flat safetensors keys."""

    geometry = state.get("geo_encoder", state.get("geometry_encoder"))
    head = state.get("score_head", state.get("auxiliary_head"))
    if isinstance(geometry, dict) and isinstance(head, dict):
        return geometry, head
    geometry = {
        key[len("geo_encoder."):]: value
        for key, value in state.items() if key.startswith("geo_encoder.")
    }
    head = {
        key[len("score_head."):]: value
        for key, value in state.items() if key.startswith("score_head.")
    }
    if not geometry or not head:
        raise CheckpointPolicyError(
            "component checkpoint must contain geo_encoder and score_head state"
        )
    return geometry, head


def load_checkpoint(
    model,
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
    training: bool = False,
    components_only: bool = False,
    allow_legacy_torch: bool = False,
):
    """Load PEFT or NOVA component state without accepting arbitrary pickle by default."""

    inspection = inspect_checkpoint(path, expected_sha256=expected_sha256)
    loaded = []
    missing = []
    unexpected = []
    warnings = []
    shape_mismatches = []
    skipped = []
    aligned = []
    checkpoint_dir = path if path.is_dir() else path.parent
    if not components_only and "adapter_model.safetensors" in inspection.files:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise CheckpointPolicyError("PEFT is required to load adapter checkpoints") from exc
        model = PeftModel.from_pretrained(model, str(checkpoint_dir), is_trainable=training)
        loaded.append("peft_adapter")
    if components_only:
        component = None
        for name in ("geo_components.safetensors", "geo_components.pt"):
            candidate = checkpoint_dir / name
            if candidate.is_file():
                component = candidate
                break
        if component is not None:
            state = _load_component_state(component, allow_legacy_torch)
            geometry, head = _component_groups(state)
            for prefix, module, component_state in (
                ("geometry_encoder", model.geometry_encoder, geometry),
                ("auxiliary_head", model.auxiliary_head.network, head),
            ):
                result = _state_compatibility(module, component_state)
                missing.extend(prefix + "." + key for key in result[0])
                unexpected.extend(prefix + "." + key for key in result[1])
                shape_mismatches.extend(prefix + "." + item for item in result[2])
                aligned.extend(prefix + "." + key for key in result[3])
                skipped.extend(prefix + "." + key for key in result[4])
                module.load_state_dict(
                    {key: component_state[key] for key in result[3]}, strict=False
                )
            loaded.extend(("geometry_encoder", "auxiliary_head"))
            if component.suffix == ".pt":
                warnings.append("Loaded a provenance-reviewed legacy state dictionary by explicit opt-in.")
    report = CheckpointLoadReport(
        inspection.checkpoint_format.value, tuple(loaded), tuple(missing), tuple(unexpected),
        tuple(shape_mismatches), tuple(warnings), tuple(skipped), tuple(aligned),
    )
    if components_only and (missing or unexpected or shape_mismatches):
        raise CheckpointLoadError("component checkpoint is not shape/key compatible", report)
    return (model, report) if not components_only else report


def classify_checkpoint(path: str, *, declared_detector_result: bool = False) -> CheckpointFormat:
    candidate = Path(path)
    suffix = candidate.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return (
            CheckpointFormat.DETECTOR_PICKLE
            if declared_detector_result
            else CheckpointFormat.UNTRUSTED_PICKLE
        )
    if suffix == ".safetensors":
        return CheckpointFormat.SAFETENSORS
    if suffix == ".pt" and "geo" in candidate.name.lower():
        return CheckpointFormat.GEOMETRY_COMPONENTS
    if suffix in {".pt", ".pth", ".bin"}:
        return CheckpointFormat.TORCH_STATE_DICT
    if suffix == "" or candidate.name.startswith("checkpoint-"):
        return CheckpointFormat.PEFT_ADAPTER
    raise CheckpointPolicyError("unsupported checkpoint format")


def inspect_checkpoint_policy(
    record: AssetRecord,
    *,
    hash_verified: bool,
    trusted_local: bool,
    justification: str,
) -> CheckpointDecision:
    checkpoint_format = classify_checkpoint(record.path)
    if checkpoint_format is CheckpointFormat.UNTRUSTED_PICKLE:
        raise CheckpointPolicyError("pickle checkpoints are denied by default")
    if record.allowed_operation not in {
        AllowedOperation.METADATA_ONLY,
        AllowedOperation.CHECKPOINT_LOAD,
    }:
        raise CheckpointPolicyError("asset is not registered as checkpoint material")
    if not hash_verified:
        raise CheckpointPolicyError("checkpoint hash must be verified first")
    if not trusted_local:
        raise CheckpointPolicyError("checkpoint must be explicitly trusted local")
    if not justification.strip():
        raise CheckpointPolicyError("checkpoint use requires a recorded justification")
    return CheckpointDecision(
        checkpoint_format=checkpoint_format,
        policy_ready=True,
        execution_enabled=True,
        reason="checkpoint format and provenance policy passed; public loader may proceed",
    )
