import unittest

from nova.runtime.assets import (
    AssetRecord,
    AssetType,
    HashAlgorithm,
)
from nova.runtime.checkpoints import (
    CheckpointFormat,
    CheckpointPolicyError,
    classify_checkpoint,
    inspect_checkpoint_policy,
)


def record(path: str) -> AssetRecord:
    return AssetRecord(
        p0_id="P0", dataset="synthetic", path=path,
        asset_type=AssetType.FILE, expected_size=1,
        expected_sha256="a" * 64, hash_algorithm=HashAlgorithm.SHA256,
        expected_file_count=1,
    )


class CheckpointPolicyTests(unittest.TestCase):
    def test_formats_are_classified_without_loading(self):
        self.assertIs(classify_checkpoint("/tmp/model.safetensors"), CheckpointFormat.SAFETENSORS)
        self.assertIs(classify_checkpoint("/tmp/geo_components.pt"), CheckpointFormat.GEOMETRY_COMPONENTS)
        self.assertIs(classify_checkpoint("/tmp/state.pth"), CheckpointFormat.TORCH_STATE_DICT)
        self.assertIs(classify_checkpoint("/tmp/checkpoint-epoch-8"), CheckpointFormat.PEFT_ADAPTER)
        self.assertIs(
            classify_checkpoint("/tmp/result.pkl", declared_detector_result=True),
            CheckpointFormat.DETECTOR_PICKLE,
        )

    def test_pickle_is_denied_by_default(self):
        with self.assertRaisesRegex(CheckpointPolicyError, "pickle"):
            inspect_checkpoint_policy(
                record("/tmp/detector.pkl"), hash_verified=True,
                trusted_local=True, justification="fixture",
            )

    def test_torch_material_requires_hash_trust_and_reason(self):
        candidate = record("/tmp/geometry.pt")
        with self.assertRaisesRegex(CheckpointPolicyError, "hash"):
            inspect_checkpoint_policy(
                candidate, hash_verified=False, trusted_local=True,
                justification="geometry adapter",
            )
        decision = inspect_checkpoint_policy(
            candidate, hash_verified=True, trusted_local=True,
            justification="geometry adapter",
        )
        self.assertTrue(decision.policy_ready)
        self.assertTrue(decision.execution_enabled)


if __name__ == "__main__":
    unittest.main()
