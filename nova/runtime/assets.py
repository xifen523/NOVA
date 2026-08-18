"""Exact-path asset allowlists and metadata-only integrity verification."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Tuple


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AssetPolicyError(ValueError):
    """Raised when a path or operation is outside the explicit allowlist."""


class AssetType(str, Enum):
    FILE = "FILE"
    DIRECTORY = "DIRECTORY"


class HashAlgorithm(str, Enum):
    SHA256 = "SHA256"
    SORTED_SHA256SUM_MANIFEST = "SHA256(SORTED_SHA256SUM_MANIFEST)"


class AllowedOperation(str, Enum):
    METADATA_ONLY = "METADATA_ONLY"
    CHECKPOINT_LOAD = "CHECKPOINT_LOAD"
    DETECTOR_PARSE = "DETECTOR_PARSE"
    PREDICTION_READ = "PREDICTION_READ"
    EVALUATION_READ = "EVALUATION_READ"


@dataclass(frozen=True)
class AssetRecord:
    p0_id: str
    dataset: str
    path: str
    asset_type: AssetType
    expected_size: int
    expected_sha256: str
    hash_algorithm: HashAlgorithm
    expected_file_count: int
    allowed_operation: AllowedOperation = AllowedOperation.METADATA_ONLY

    def validate(self) -> None:
        candidate = Path(self.path)
        if not candidate.is_absolute() or os.path.normpath(self.path) != self.path:
            raise AssetPolicyError("asset paths must be normalized absolute paths")
        if not self.p0_id or not self.dataset:
            raise AssetPolicyError("p0_id and dataset must be explicit")
        if self.expected_size < 0 or self.expected_file_count < 1:
            raise AssetPolicyError("expected size/count must be non-negative and non-zero")
        if not _SHA256.fullmatch(self.expected_sha256):
            raise AssetPolicyError("expected_sha256 must be lowercase 64-hex")
        if self.asset_type is AssetType.FILE and self.expected_file_count != 1:
            raise AssetPolicyError("file assets must have expected_file_count=1")
        if self.asset_type is AssetType.FILE and self.hash_algorithm is not HashAlgorithm.SHA256:
            raise AssetPolicyError("file assets require SHA256")
        if (
            self.asset_type is AssetType.DIRECTORY
            and self.hash_algorithm is not HashAlgorithm.SORTED_SHA256SUM_MANIFEST
        ):
            raise AssetPolicyError("directory assets require the sorted-manifest hash")


@dataclass(frozen=True)
class AssetVerification:
    path: str
    actual_size: int
    actual_file_count: int
    actual_sha256: str
    passed: bool
    reasons: Tuple[str, ...]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_metadata(path: Path) -> Tuple[int, int, str]:
    children = sorted(path.iterdir(), key=lambda item: item.name)
    files = []
    for child in children:
        if child.is_symlink() or not child.is_file():
            raise AssetPolicyError("directory assets may contain only immediate regular files")
        files.append(child)
    lines = []
    size = 0
    for child in files:
        size += child.stat().st_size
        # The normalized form matches sha256sum output with the literal
        # two-space separator and directory prefix removed.
        lines.append("{0}{1}\n".format(_hash_file(child), child.name))
    tree_hash = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    return size, len(files), tree_hash


def verify_asset_metadata(record: AssetRecord) -> AssetVerification:
    """Hash bytes without deserializing, parsing, importing, or executing them."""

    record.validate()
    path = Path(record.path)
    reasons = []
    if path.is_symlink() or not path.exists():
        return AssetVerification(record.path, 0, 0, "", False, ("missing_or_symlink",))
    if record.asset_type is AssetType.FILE:
        if not path.is_file():
            return AssetVerification(record.path, 0, 0, "", False, ("type_mismatch",))
        actual_size = path.stat().st_size
        actual_count = 1
        actual_hash = _hash_file(path)
    else:
        if not path.is_dir():
            return AssetVerification(record.path, 0, 0, "", False, ("type_mismatch",))
        actual_size, actual_count, actual_hash = _directory_metadata(path)
    if actual_size != record.expected_size:
        reasons.append("size_mismatch")
    if actual_count != record.expected_file_count:
        reasons.append("file_count_mismatch")
    if actual_hash != record.expected_sha256:
        reasons.append("hash_mismatch")
    return AssetVerification(
        record.path,
        actual_size,
        actual_count,
        actual_hash,
        not reasons,
        tuple(reasons),
    )


class AssetAllowlist:
    """An immutable exact-path registry; globbing and path guessing are impossible."""

    def __init__(self, records: Iterable[AssetRecord]):
        registry: Dict[str, AssetRecord] = {}
        for record in records:
            record.validate()
            if record.path in registry:
                raise AssetPolicyError("duplicate allowlist path: {0}".format(record.path))
            registry[record.path] = record
        self._records = registry

    def require(self, path: str, operation: AllowedOperation) -> AssetRecord:
        if not Path(path).is_absolute() or os.path.normpath(path) != path:
            raise AssetPolicyError("requested path must be normalized and absolute")
        record = self._records.get(path)
        if record is None:
            raise AssetPolicyError("asset path is not allowlisted")
        if operation is not record.allowed_operation:
            raise AssetPolicyError("operation is not allowlisted for this asset")
        return record

    @property
    def records(self) -> Tuple[AssetRecord, ...]:
        return tuple(self._records[path] for path in sorted(self._records))
