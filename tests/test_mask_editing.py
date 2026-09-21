import unittest

import numpy as np

from core.mask_editing import enclosed_mask_region


class EnclosedMaskTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((30, 30), dtype=np.uint8)
        self.mask[5:25, 5:25] = 255
        self.mask[10:20, 10:20] = 0

    def test_thick_outline_fills_all_interior_pixels_without_mutation(self):
        original = self.mask.copy()
        region = enclosed_mask_region(self.mask, (15, 15))
        self.assertEqual(int(region.sum()), 100)
        np.testing.assert_array_equal(self.mask, original)

    def test_open_outline_is_rejected_even_with_protection(self):
        self.mask[0:11, 15] = 0
        protected = np.zeros_like(self.mask)
        protected[0:10] = 255
        self.assertIsNone(enclosed_mask_region(
            self.mask, (15, 15), protected_mask=protected
        ))

    def test_fill_and_subtract_exclude_protected_pixels(self):
        protected = np.zeros_like(self.mask)
        protected[12:18, 12:18] = 255
        region = enclosed_mask_region(
            self.mask, (10, 10), protected_mask=protected
        )
        self.assertEqual(int(region.sum()), 64)
        self.mask[10:20, 10:20] = 255
        region = enclosed_mask_region(
            self.mask, (10, 10), filled=True, protected_mask=protected
        )
        self.assertEqual(int(region.sum()), 400 - 36)
        self.assertFalse(region[15, 15])

    def test_wrong_seed_class_and_protected_seed_are_rejected(self):
        self.assertIsNone(enclosed_mask_region(self.mask, (5, 5)))
        self.assertIsNone(enclosed_mask_region(self.mask, (15, 15), filled=True))
        self.assertIsNone(enclosed_mask_region(self.mask, (-1, 2)))
        self.assertIsNone(enclosed_mask_region(
            self.mask, (15, 15), protected_mask=np.ones_like(self.mask)
        ))

    def test_diagonal_leak_is_not_enclosed(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        np.fill_diagonal(mask, 0)
        self.assertIsNone(enclosed_mask_region(mask, (2, 2)))

    def test_bad_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            enclosed_mask_region(self.mask, (15, 15), protected_mask=np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            enclosed_mask_region(np.zeros((2, 2, 3)), (1, 1))


if __name__ == "__main__":
    unittest.main()
