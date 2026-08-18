import json
import os
import subprocess
import sys
from pathlib import Path


TOOLS = ("prepare_data", "train", "infer", "summarize", "evaluate")
CONFIG = Path("configs/examples/synthetic_v2x.yaml")
FIXTURE = Path("tests/fixtures/synthetic_v2x.json")


def safe_environment():
    environment = dict(os.environ)
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        environment.pop(name, None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    return environment


def run(*arguments):
    return subprocess.run(
        [sys.executable, *arguments], capture_output=True, text=True,
        env=safe_environment(), check=False,
    )


def test_all_help_commands_succeed():
    for tool in TOOLS:
        result = run("tools/" + tool + ".py", "--help")
        assert result.returncode == 0, result.stderr


def test_synthetic_prepare_train_infer_evaluate_chain(tmp_path: Path):
    prepared = tmp_path / "prepared.jsonl"
    prepare = run(
        "tools/prepare_data.py", "--config", str(CONFIG), "--input", str(FIXTURE),
        "--output", str(prepared),
    )
    assert prepare.returncode == 0, prepare.stderr
    assert prepared.is_file()

    train_dir = tmp_path / "training"
    train = run(
        "tools/train.py", "--config", str(CONFIG), "--input", str(prepared),
        "--output-dir", str(train_dir), "--backend", "tiny", "--steps", "2",
    )
    assert train.returncode == 0, train.stderr
    assert json.loads(train.stdout)["steps"] == 2

    predictions = tmp_path / "predictions.jsonl"
    infer = run(
        "tools/infer.py", "--input", str(prepared), "--output", str(predictions),
        "--backend", "heuristic",
    )
    assert infer.returncode == 0, infer.stderr
    assert predictions.is_file()

    metrics = tmp_path / "metrics.json"
    evaluate = run(
        "tools/summarize.py", "--input", str(predictions), "--output", str(metrics),
        "--dataset", "v2x",
    )
    assert evaluate.returncode == 0, evaluate.stderr
    assert json.loads(metrics.read_text(encoding="utf-8"))["association_predictions"] == 1


def test_real_cli_requirements_are_clear(tmp_path: Path):
    prepared = tmp_path / "prepared.jsonl"
    result = run(
        "tools/prepare_data.py", "--input", str(FIXTURE), "--output", str(prepared)
    )
    assert result.returncode == 0
    missing_model = run(
        "tools/infer.py", "--input", str(prepared), "--output", str(tmp_path / "out.jsonl")
    )
    assert missing_model.returncode == 2
    assert "Model path or --config is required for real inference." in missing_model.stderr
    missing_input = run(
        "tools/infer.py", "--input", str(tmp_path / "missing.jsonl"),
        "--output", str(tmp_path / "out.jsonl"), "--backend", "heuristic",
    )
    assert missing_input.returncode == 2
    assert "Prepared detector input was not found" in missing_input.stderr


def test_all_dry_run_paths_validate_without_external_assets(tmp_path: Path):
    prepared = tmp_path / "prepared.jsonl"
    assert run("tools/prepare_data.py", "--input", str(FIXTURE), "--output", str(prepared)).returncode == 0
    assert run("tools/prepare_data.py", "--input", str(FIXTURE), "--output", str(tmp_path / "ignored"), "--dry-run").returncode == 0
    assert run("tools/train.py", "--config", str(CONFIG), "--input", str(prepared), "--output-dir", str(tmp_path / "train"), "--backend", "tiny", "--dry-run").returncode == 0
    assert run("tools/infer.py", "--input", str(prepared), "--output", str(tmp_path / "predictions"), "--backend", "heuristic", "--dry-run").returncode == 0
    diagnostic = tmp_path / "diagnostic.jsonl"
    diagnostic.write_text(json.dumps({"schema": "nova.association-prediction.v1", "matched": True}) + "\n", encoding="utf-8")
    assert run("tools/summarize.py", "--input", str(diagnostic), "--dataset", "v2x", "--dry-run").returncode == 0


def test_prepared_sequence_tracking_runs_with_explicit_heuristic_backend(tmp_path: Path):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    first, second = fixture["history"][:2]
    sequence = {
            "schema": "nova.detection-sequence.v1",
            "dataset": "v2x",
            "coordinate_frame": "v2x_ego_lidar",
        "class_split": fixture["class_split"],
        "frames": [
            {"sequence_id": first["sequence_id"], "frame_id": first["frame_id"], "timestamp": first["timestamp"], "detections": [first]},
            {"sequence_id": second["sequence_id"], "frame_id": second["frame_id"], "timestamp": second["timestamp"], "detections": [second]},
        ],
    }
    input_path = tmp_path / "detections.json"
    output_path = tmp_path / "tracks.json"
    input_path.write_text(json.dumps(sequence), encoding="utf-8")
    result = run(
        "tools/infer.py", "--mode", "tracking", "--backend", "heuristic",
        "--config", str(CONFIG), "--input", str(input_path), "--output", str(output_path),
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["schema"] == "nova.tracking-output.v1"
    assert len(output["frames"]) == 2
