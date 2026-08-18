import unittest

from nova.data.class_split import ClassSplit, UnknownClassPolicy
from nova.data.types import ClassMembership


class ClassSplitTests(unittest.TestCase):
    def test_overlap_and_normalized_duplicates_are_rejected(self):
        with self.assertRaises(ValueError):
            ClassSplit(("Synthetic Car",), ("synthetic_car",))
        with self.assertRaises(ValueError):
            ClassSplit(("Synthetic Car", " synthetic  car "), ())

    def test_alias_normalization_is_stable(self):
        split = ClassSplit(
            ("synthetic_car",),
            ("synthetic_cart",),
            (("Synthetic Auto", "Synthetic Car"),),
            mapping_status="SYNTHETIC_EXAMPLE",
        )
        self.assertEqual(split.classify(" synthetic  AUTO "), ClassMembership.BASE)
        self.assertEqual(split.classify("synthetic_cart"), ClassMembership.NOVEL)

    def test_unknown_policy_is_deterministic(self):
        permissive = ClassSplit((), (), mapping_status="SYNTHETIC_EXAMPLE")
        self.assertEqual(permissive.classify("invented"), ClassMembership.UNKNOWN)
        strict = ClassSplit(
            (), (), unknown_policy=UnknownClassPolicy.REJECT,
            mapping_status="SYNTHETIC_EXAMPLE",
        )
        with self.assertRaises(ValueError):
            strict.classify("invented")

    def test_real_mode_requires_authoritative_map(self):
        split = ClassSplit((), (), mapping_status="REQUIRES_AUTHORITATIVE_CLASS_MAP")
        with self.assertRaises(ValueError):
            split.validate_for_real_mode()


if __name__ == "__main__":
    unittest.main()
