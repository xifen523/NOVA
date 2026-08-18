"""Deterministic integrity hashes for public release trees."""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Mapping


TREE_SHA256_METHOD = (
    "SHA256 of UTF-8 relative-path-sorted lines formatted as file_sha256, "
    "two spaces, relative_path, newline"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


class ReleaseHashError(ValueError):
    """Raised when release input cannot produce a trustworthy tree hash."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(relative: str) -> str:
    logical = PurePosixPath(relative)
    if (
        not relative
        or logical.is_absolute()
        or ".." in logical.parts
        or logical.as_posix() != relative
        or relative.startswith("./")
    ):
        raise ReleaseHashError(f"invalid relative path: {relative!r}")
    return relative


def iter_release_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield ordinary files in UTF-8 byte order, rejecting polluted trees."""

    root = root.resolve()
    if not root.is_dir():
        raise ReleaseHashError(f"release root is not a directory: {root}")
    discovered: list[tuple[str, Path]] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            candidate = current_path / dirname
            if candidate.is_symlink():
                raise ReleaseHashError(f"symlink in release tree: {candidate.relative_to(root)}")
            if dirname in _FORBIDDEN_DIR_NAMES or dirname.endswith(".egg-info"):
                raise ReleaseHashError(
                    f"forbidden directory in release tree: {candidate.relative_to(root)}"
                )
        for filename in filenames:
            candidate = current_path / filename
            relative = candidate.relative_to(root).as_posix()
            _validate_relative_path(relative)
            if candidate.is_symlink() or not candidate.is_file():
                raise ReleaseHashError(f"non-regular file in release tree: {relative}")
            discovered.append((relative, candidate))
    yield from sorted(discovered, key=lambda item: item[0].encode("utf-8"))


def tree_hash_lines(entries: Iterable[tuple[str, str]]) -> bytes:
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for relative, digest in entries:
        relative = _validate_relative_path(relative)
        if relative in seen:
            raise ReleaseHashError(f"duplicate manifest path: {relative}")
        if not _SHA256.fullmatch(digest):
            raise ReleaseHashError(f"invalid SHA-256 for {relative}")
        seen.add(relative)
        normalized.append((relative, digest))
    normalized.sort(key=lambda item: item[0].encode("utf-8"))
    return "".join(f"{digest}  {relative}\n" for relative, digest in normalized).encode("utf-8")


def tree_sha256(root: Path) -> str:
    entries = ((relative, sha256_file(path)) for relative, path in iter_release_files(root))
    return hashlib.sha256(tree_hash_lines(entries)).hexdigest()


def manifest_rows(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "classification": "PUBLIC_RELEASE",
        }
        for relative, path in iter_release_files(root)
    ]


def write_manifest(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "sha256", "classification"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def tree_sha256_from_manifest(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or set(rows[0]) != {"path", "size_bytes", "sha256", "classification"}:
        raise ReleaseHashError("manifest has an invalid header or no file rows")
    entries = ((row["path"], row["sha256"]) for row in rows)
    return hashlib.sha256(tree_hash_lines(entries)).hexdigest()


def metadata_tree_sha256(path: Path) -> str:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    digest = values.get("TREE_SHA256", "")
    if not _SHA256.fullmatch(digest):
        raise ReleaseHashError("metadata TREE_SHA256 is missing or invalid")
    if values.get("TREE_SHA256_METHOD") != TREE_SHA256_METHOD:
        raise ReleaseHashError("metadata TREE_SHA256_METHOD does not match")
    return digest
