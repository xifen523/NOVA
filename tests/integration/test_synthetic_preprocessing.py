import json
import unittest
from pathlib import Path

import torch

from nova.data import ClassSplit, build_dataset_adapter
from nova.models.geometry_encoder import GeometryEncoder


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class SyntheticPreprocessingIntegrationTests(unittest.TestCase):
    def test_all_fixtures_reach_geometry_encoder_on_cpu(self):
        encoder = GeometryEncoder(9, 16, 8).cpu()
        for fixture_path in sorted(FIXTURE_ROOT.glob("synthetic_*.json")):
            with self.subTest(fixture=fixture_path.name):
                document = json.loads(fixture_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    document["notices"],
                    ["SYNTHETIC_TEST_DATA_ONLY", "NOT_DERIVED_FROM_NUSCENES_KITTI_OR_V2X"],
                )
                split = ClassSplit.from_mapping(document["class_split"])
                adapter = build_dataset_adapter(document["dataset"], split)
                track_id = document["history"][0]["track_id"]
                history = adapter.build_history(track_id, document["history"])
                sample = adapter.build_association_sample(
                    history, document["candidates"][0]
                )
                geometry = adapter.build_geometry_features(sample)
                expected = torch.tensor(document["expected_geometry"], dtype=torch.float32)
                self.assertEqual(geometry.shape, (3, 9))
                self.assertTrue(torch.allclose(geometry[-1], expected))
                with torch.no_grad():
                    output = encoder(geometry)
                self.assertEqual(output.shape, (3, 8))
                self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
