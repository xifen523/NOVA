import dataclasses
import unittest
from pathlib import Path

from nova.config import ClassSplitConfig, ConfigValidationError, load_config, load_config_text


CONFIG_PATH = Path("configs/examples/synthetic_v2x.yaml")


class TrackingConfigTests(unittest.TestCase):
    def test_three_synthetic_configs_parse_with_tracking_fields(self):
        for name in ("v2x", "kitti", "nuscenes"):
            with self.subTest(name=name):
                config = load_config(Path("configs/examples") / ("synthetic_" + name + ".yaml"))
                self.assertGreaterEqual(config.inference.distance_weight, 0.0)
                self.assertGreaterEqual(config.inference.min_hits, 1)

    def test_unknown_tracking_field_is_rejected(self):
        text = CONFIG_PATH.read_text(encoding="utf-8").replace(
            "inference: {", "inference: {unreviewed_gate: 1, "
        )
        with self.assertRaises(ConfigValidationError):
            load_config_text(text)

    def test_real_mode_needs_authoritative_class_map(self):
        config = load_config(CONFIG_PATH)
        with self.assertRaises(ConfigValidationError):
            dataclasses.replace(config.dataset, real_mode=True).validate()
        authoritative = ClassSplitConfig(
            base_classes=["synthetic_vehicle"], novel_classes=[], aliases={},
            unknown_policy="reject", authoritative=True, mapping_status="AUTHORITATIVE",
        )
        real_dataset = dataclasses.replace(
            config.dataset, real_mode=True, class_split=authoritative
        )
        real_config = dataclasses.replace(config, dataset=real_dataset)
        real_config.validate()


if __name__ == "__main__":
    unittest.main()
