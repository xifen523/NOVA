from pathlib import Path

import pytest

from nova.release_hash import (
    TREE_SHA256_METHOD,
    ReleaseHashError,
    manifest_rows,
    metadata_tree_sha256,
    tree_hash_lines,
    tree_sha256,
    tree_sha256_from_manifest,
    write_manifest,
)


def test_directory_manifest_and_metadata_tree_hashes_match(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "z.txt").write_text("z\n", encoding="utf-8")
    (release / "nested").mkdir()
    (release / "nested" / "a.txt").write_text("a\n", encoding="utf-8")

    manifest = tmp_path / "manifest.tsv"
    write_manifest(manifest, manifest_rows(release))
    expected = tree_sha256(release)
    metadata = tmp_path / "metadata.txt"
    metadata.write_text(
        f"TREE_SHA256={expected}\nTREE_SHA256_METHOD={TREE_SHA256_METHOD}\n",
        encoding="utf-8",
    )

    assert expected == tree_sha256_from_manifest(manifest)
    assert expected == metadata_tree_sha256(metadata)


def test_tree_hash_lines_use_two_spaces_and_no_dot_prefix() -> None:
    digest = "0" * 64
    assert tree_hash_lines([("file.txt", digest)]) == f"{digest}  file.txt\n".encode()
    with pytest.raises(ReleaseHashError, match="invalid relative path"):
        tree_hash_lines([("./file.txt", digest)])


def test_tree_hash_rejects_forbidden_directory(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    with pytest.raises(ReleaseHashError, match="forbidden directory"):
        tree_sha256(tmp_path)
