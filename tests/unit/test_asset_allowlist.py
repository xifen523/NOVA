import hashlib
import tempfile
import unittest
from pathlib import Path

from nova.runtime.assets import (
    AllowedOperation,
    AssetAllowlist,
    AssetPolicyError,
    AssetRecord,
    AssetType,
    HashAlgorithm,
    verify_asset_metadata,
)


def file_record(path: Path, digest: str) -> AssetRecord:
    return AssetRecord(
        p0_id="P0", dataset="synthetic", path=str(path),
        asset_type=AssetType.FILE, expected_size=path.stat().st_size,
        expected_sha256=digest, hash_algorithm=HashAlgorithm.SHA256,
        expected_file_count=1,
    )


class AssetAllowlistTests(unittest.TestCase):
    def test_unknown_and_non_absolute_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"safe-metadata-fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            allowlist = AssetAllowlist([file_record(path, digest)])
            with self.assertRaisesRegex(AssetPolicyError, "not allowlisted"):
                allowlist.require(str(Path(directory) / "other.bin"), AllowedOperation.METADATA_ONLY)
            with self.assertRaisesRegex(AssetPolicyError, "absolute"):
                allowlist.require("asset.bin", AllowedOperation.METADATA_ONLY)

    def test_operation_is_exact_and_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"fixture")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            allowlist = AssetAllowlist([file_record(path, digest)])
            with self.assertRaisesRegex(AssetPolicyError, "operation"):
                allowlist.require(str(path), AllowedOperation.CHECKPOINT_LOAD)

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.bin"
            path.write_bytes(b"fixture")
            record = file_record(path, "0" * 64)
            result = verify_asset_metadata(record)
            self.assertFalse(result.passed)
            self.assertIn("hash_mismatch", result.reasons)

    def test_directory_manifest_hash_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.txt").write_bytes(b"b")
            (root / "a.txt").write_bytes(b"a")
            lines = "".join(
                "{0}{1}\n".format(hashlib.sha256(value).hexdigest(), name)
                for name, value in (("a.txt", b"a"), ("b.txt", b"b"))
            )
            tree_hash = hashlib.sha256(lines.encode("utf-8")).hexdigest()
            record = AssetRecord(
                p0_id="P0", dataset="synthetic", path=str(root),
                asset_type=AssetType.DIRECTORY, expected_size=2,
                expected_sha256=tree_hash,
                hash_algorithm=HashAlgorithm.SORTED_SHA256SUM_MANIFEST,
                expected_file_count=2,
            )
            self.assertTrue(verify_asset_metadata(record).passed)


if __name__ == "__main__":
    unittest.main()
