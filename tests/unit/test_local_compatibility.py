import json
import struct
from pathlib import Path

from nova.runtime.local_compatibility import assess_lora_adapter, read_safetensors_metadata


def write_header(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload)


def test_safetensors_header_and_adapter_resize_are_metadata_only(tmp_path: Path) -> None:
    path = tmp_path / "adapter.safetensors"
    header = {
        "base_model.model.model.embed_tokens.weight": {"dtype": "BF16", "shape": [10, 8], "data_offsets": [0, 0]},
        "base_model.model.lm_head.weight": {"dtype": "BF16", "shape": [10, 8], "data_offsets": [0, 0]},
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {"dtype": "BF16", "shape": [2, 8], "data_offsets": [0, 0]},
    }
    write_header(path, header)
    metadata = read_safetensors_metadata(path)
    decision = assess_lora_adapter(
        metadata, hidden_size=8, base_vocab_size=12, adapter_vocab_size=10,
        expected_layers=1, rank=2,
    )
    assert metadata.tensor_count == 3
    assert decision.compatible
    assert decision.requires_embedding_resize
