"""Stable, allowlisted run manifests that never expose process environment."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from nova.config.schema import ExperimentConfig


UNRESOLVED = "UNRESOLVED"


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _git_value(arguments: List[str]) -> str:
    result = subprocess.run(
        ["git"] + arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else UNRESOLVED


@dataclass(frozen=True)
class RunManifest:
    command: str
    config_path: str
    resolved_config: Mapping[str, object]
    python_version: str
    torch_version: str
    git_commit: str
    git_tag: str
    dataset: str
    model_id: str
    model_revision: str
    checkpoint_path: str
    checkpoint_sha256: str
    detector_asset_path: str
    detector_asset_sha256: str
    class_map: Mapping[str, object]
    random_seed: int
    offline: bool
    timestamp: str
    known_unresolved_fields: Tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def build_run_manifest(
    command: str,
    config_path: Path,
    config: ExperimentConfig,
    offline: bool,
    timestamp: Optional[str] = None,
) -> RunManifest:
    unresolved = []
    values = {
        "model.revision": config.model.revision,
        "assets.checkpoint_path": config.assets.checkpoint_path,
        "assets.checkpoint_sha256": config.assets.checkpoint_sha256,
        "assets.detector_asset_path": config.assets.detector_asset_path,
        "assets.detector_asset_sha256": config.assets.detector_asset_sha256,
    }
    for name, value in values.items():
        if value is None:
            unresolved.append(name)
    tags = _git_value(["tag", "--points-at", "HEAD"]).splitlines()
    tag = sorted(tags)[0] if tags else UNRESOLVED
    split = config.dataset.class_split
    class_map: Dict[str, object] = {
        "base_classes": list(split.base_classes),
        "novel_classes": list(split.novel_classes),
        "aliases": dict(split.aliases),
        "mapping_status": split.mapping_status,
        "authoritative": split.authoritative,
    }
    return RunManifest(
        command=command,
        config_path=str(config_path),
        resolved_config=_plain(asdict(config)),
        python_version=".".join(str(value) for value in sys.version_info[:3]),
        torch_version=torch.__version__,
        git_commit=_git_value(["rev-parse", "HEAD"]),
        git_tag=tag,
        dataset=config.dataset.name,
        model_id=config.model.base_model,
        model_revision=config.model.revision or UNRESOLVED,
        checkpoint_path=config.assets.checkpoint_path or UNRESOLVED,
        checkpoint_sha256=config.assets.checkpoint_sha256 or UNRESOLVED,
        detector_asset_path=config.assets.detector_asset_path or UNRESOLVED,
        detector_asset_sha256=config.assets.detector_asset_sha256 or UNRESOLVED,
        class_map=class_map,
        random_seed=config.training.seed,
        offline=offline,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        known_unresolved_fields=tuple(sorted(unresolved)),
    )


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")
