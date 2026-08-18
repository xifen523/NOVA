"""Metadata-only helpers for offline safetensors compatibility reviews."""

from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple


class CompatibilityMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class SafeTensorMetadata:
    tensor_count: int
    header_bytes: int
    dtype_counts: Mapping[str, int]
    shapes: Mapping[str, Tuple[int, ...]]


def read_safetensors_metadata(path: Path, *, max_header_bytes: int = 64 * 1024 * 1024) -> SafeTensorMetadata:
    """Read only a safetensors header; tensor payload bytes are never loaded."""

    if path.is_symlink() or not path.is_file():
        raise CompatibilityMetadataError("safetensors path must be a regular file")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise CompatibilityMetadataError("truncated safetensors prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes < 2 or header_bytes > max_header_bytes:
            raise CompatibilityMetadataError("invalid safetensors header length")
        try:
            header = json.loads(handle.read(header_bytes))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompatibilityMetadataError("invalid safetensors header JSON") from exc
    tensors: Dict[str, Mapping[str, object]] = {
        key: value for key, value in header.items() if key != "__metadata__"
    }
    shapes: Dict[str, Tuple[int, ...]] = {}
    dtypes = Counter()
    for key, value in tensors.items():
        if not isinstance(value, dict) or "shape" not in value or "dtype" not in value:
            raise CompatibilityMetadataError("invalid tensor metadata")
        shapes[key] = tuple(int(item) for item in value["shape"])
        dtypes[str(value["dtype"])] += 1
    return SafeTensorMetadata(len(tensors), header_bytes, dict(dtypes), shapes)


@dataclass(frozen=True)
class AdapterCompatibility:
    compatible: bool
    requires_embedding_resize: bool
    reasons: Tuple[str, ...]


def assess_lora_adapter(
    metadata: SafeTensorMetadata,
    *,
    hidden_size: int,
    base_vocab_size: int,
    adapter_vocab_size: int,
    expected_layers: int,
    rank: int,
) -> AdapterCompatibility:
    reasons = []
    keys = metadata.shapes
    layer_ids = set()
    for key in keys:
        if ".layers." not in key:
            continue
        part = key.split(".layers.", 1)[1].split(".", 1)[0]
        if part.isdigit():
            layer_ids.add(int(part))
    if layer_ids != set(range(expected_layers)):
        reasons.append("layer_coverage_mismatch")
    lora_a = [shape for key, shape in keys.items() if key.endswith("lora_A.weight")]
    if not lora_a or any(shape != (rank, hidden_size) for shape in lora_a):
        reasons.append("lora_A_shape_mismatch")
    embedding = keys.get("base_model.model.model.embed_tokens.weight")
    lm_head = keys.get("base_model.model.lm_head.weight")
    expected_adapter = (adapter_vocab_size, hidden_size)
    if embedding != expected_adapter or lm_head != expected_adapter:
        reasons.append("adapter_embedding_shape_mismatch")
    resize = base_vocab_size != adapter_vocab_size
    return AdapterCompatibility(not reasons, resize, tuple(reasons))
