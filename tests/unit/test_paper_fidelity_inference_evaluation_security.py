"""Paper-fidelity inference, evaluation, and asset-boundary tests."""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from nova.config import load_config
from nova.data import ClassSplit
from nova.data.types import SerializedAssociationContext
from nova.evaluation.external import run_external_evaluator
from nova.cli.evaluate import main as evaluate_main
from nova.evaluation.types import SplitGroup
from nova.inference import validate_tracking_document, score_serialized_contexts
from nova.models.tokenizer import validate_tokenizer_equivalence
from nova.models.loading import prepare_model_input
from nova.runtime.checkpoints import CheckpointPolicyError, checkpoint_tree_manifest, inspect_checkpoint


CONFIG = Path("configs/examples/synthetic_v2x.yaml")
FIXTURE = Path("tests/fixtures/synthetic_v2x.json")


def tracking_document():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "schema": "nova.detection-sequence.v1",
        "dataset": "v2x",
        "coordinate_frame": "v2x_ego_lidar",
        "class_split": fixture["class_split"],
        "frames": [
            {
                "sequence_id": item["sequence_id"],
                "frame_id": item["frame_id"],
                "timestamp": item["timestamp"],
                "detections": [item],
            }
            for item in fixture["history"]
        ],
    }


def test_tracking_dataset_and_coordinate_mismatch_rejected():
    config = load_config(CONFIG)
    for key, value in (("dataset", "kitti"), ("coordinate_frame", "kitti_lidar")):
        document = tracking_document()
        document[key] = value
        with pytest.raises(ValueError, match="does not match"):
            validate_tracking_document(document, config, allow_compatible_input=False)


def test_tracking_frame_order_must_be_monotonic():
    config = load_config(CONFIG)
    document = tracking_document()
    document["frames"].reverse()
    with pytest.raises(ValueError, match="monotonic"):
        validate_tracking_document(document, config, allow_compatible_input=False)


def test_compatible_input_override_records_warning():
    config = load_config(CONFIG)
    document = tracking_document()
    document["coordinate_frame"] = "compatible-but-different"
    _, warnings = validate_tracking_document(document, config, allow_compatible_input=True)
    assert warnings and "coordinate_frame" in warnings[0]


def context(count):
    return SerializedAssociationContext(
        prompt=" ".join(["<box>"] * count),
        geometry_features=tuple((0.0,) * 9 for _ in range(count)),
        box_count=count,
        box_roles=tuple(["history_{0}".format(index) for index in range(count - 1)] + ["candidate"]),
        token_metadata={},
        timestamps=tuple(range(1, count + 1)),
    )


def test_pair_chunking_is_equivalent(monkeypatch):
    def fake(_model, _tokenizer, _ids, contexts, _device):
        return torch.tensor([item.box_count / 10.0 for item in contexts])
    monkeypatch.setattr("nova.inference._score_context_chunk", fake)
    contexts = [context(1), context(2), context(4), context(3)]
    unchunked, one = score_serialized_contexts(None, None, None, contexts, device="cpu", pair_batch_size=10)
    chunked, many = score_serialized_contexts(None, None, None, contexts, device="cpu", pair_batch_size=2)
    assert torch.equal(unchunked, chunked)
    assert one == 1 and many == 2


class ShapeTokenizer:
    def __call__(self, prompts, padding=True, return_tensors="pt"):
        counts = [prompt.count("<box>") for prompt in prompts]
        width = max(counts) + 1
        ids = []
        masks = []
        for count in counts:
            row = [9] * count + [2] + [0] * (width - count - 1)
            ids.append(row)
            masks.append([1] * (count + 1) + [0] * (width - count - 1))
        return SimpleNamespace(
            input_ids=torch.tensor(ids, dtype=torch.long),
            attention_mask=torch.tensor(masks, dtype=torch.long),
        )


@pytest.mark.parametrize("history_length", [0, 3, 5])
def test_history_length_changes_model_input_shape(history_length):
    item = context(history_length + 1)
    geometry = torch.tensor(item.geometry_features, dtype=torch.float32)
    model_input = prepare_model_input(
        ShapeTokenizer(), (item.prompt,), geometry,
        SimpleNamespace(box=9), box_counts=(item.box_count,), device="cpu",
    )
    assert model_input.box_counts.tolist() == [history_length + 1]
    assert model_input.geometry.shape == (history_length + 1, 9)
    assert int((model_input.input_ids == 9).sum()) == history_length + 1


def test_external_evaluator_success_parses_metric_bundle(tmp_path):
    prediction = tmp_path / "prediction.json"
    prediction.write_text("{}", encoding="utf-8")
    metrics = {"sAMOTA": 1, "AMOTA": 2, "AMOTP": 3, "MOTA": 4, "MOTP": 5, "MT": 6}
    result = run_external_evaluator(
        input_path=prediction,
        command=[sys.executable, "-c", "import json; print(json.dumps({0}))".format(metrics)],
        dataset="v2x",
        split_group=SplitGroup.ALL,
        evaluator_name="fixture evaluator",
        evaluator_version="1",
    )
    assert result.parsed.bundle.by_name("AMOTA").display_value == "2.00"


def test_external_evaluator_failure_and_unknown_output_are_rejected(tmp_path):
    prediction = tmp_path / "prediction.json"
    prediction.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed"):
        run_external_evaluator(
            input_path=prediction,
            command=[sys.executable, "-c", "raise SystemExit(3)"],
            dataset="v2x", split_group=SplitGroup.ALL,
            evaluator_name="fixture", evaluator_version="1",
        )
    with pytest.raises(ValueError, match="EVALUATION_PARSE=FAIL"):
        run_external_evaluator(
            input_path=prediction,
            command=[sys.executable, "-c", "print('unknown')"],
            dataset="v2x", split_group=SplitGroup.ALL,
            evaluator_name="fixture", evaluator_version="1",
        )


def test_evaluate_without_external_adapter_returns_required_status(tmp_path, capsys):
    prediction = tmp_path / "prediction.json"
    prediction.write_text("{}", encoding="utf-8")
    code = evaluate_main([
        "--input", str(prediction), "--output", str(tmp_path / "metrics.json"),
        "--dataset", "v2x",
    ])
    assert code == 2
    assert capsys.readouterr().err.strip() == "EXTERNAL_EVALUATOR_REQUIRED"


def test_checkpoint_child_symlink_and_tree_hash_mismatch_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    target = tmp_path / "outside.safetensors"
    target.write_bytes(b"not-a-real-file")
    (checkpoint / "adapter_model.safetensors").symlink_to(target)
    with pytest.raises(CheckpointPolicyError, match="symlink"):
        inspect_checkpoint(checkpoint)


def test_checkpoint_directory_tree_hash_is_enforced(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    records, digest = checkpoint_tree_manifest(checkpoint)
    assert records[0][0] == "adapter_config.json"
    with pytest.raises(CheckpointPolicyError, match="tree SHA256 mismatch"):
        inspect_checkpoint(checkpoint, expected_sha256="0" * 64)


class Tokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2
    unk_token_id = 0
    padding_side = "right"
    truncation_side = "right"
    chat_template = "fixture"

    def __init__(self, offset=0):
        self.offset = offset

    def encode(self, text, add_special_tokens=True):
        return ([self.bos_token_id] if add_special_tokens else []) + [len(text) + self.offset]


def test_tokenizer_fallback_equivalence_reports_supported_or_best_effort():
    supported = validate_tokenizer_equivalence(Tokenizer(), Tokenizer())
    mismatch = validate_tokenizer_equivalence(Tokenizer(), Tokenizer(offset=1))
    assert supported.status == "SUPPORTED_FALLBACK"
    assert mismatch.status == "BEST_EFFORT_FALLBACK"
    assert mismatch.mismatches
