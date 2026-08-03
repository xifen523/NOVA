import unittest

from nova.data import ClassSplit
from nova.data.registry import (
    build_dataset_adapter,
    list_dataset_adapters,
    register_dataset_adapter,
)
from nova.data.v2x import V2XAdapter


class DatasetRegistryTests(unittest.TestCase):
    def setUp(self):
        self.split = ClassSplit((), (), mapping_status="SYNTHETIC_EXAMPLE")

    def test_builtin_adapters_are_listed_and_buildable(self):
        self.assertEqual(list_dataset_adapters(), ("kitti", "nuscenes", "v2x"))
        for name in list_dataset_adapters():
            self.assertEqual(build_dataset_adapter(name, self.split).name, name)

    def test_unknown_adapter_is_rejected(self):
        with self.assertRaises(ValueError):
            build_dataset_adapter("invented", self.split)

    def test_duplicate_registration_is_rejected(self):
        with self.assertRaises(ValueError):
            register_dataset_adapter("V2X", V2XAdapter)


if __name__ == "__main__":
    unittest.main()
