import unittest

import cv2
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

    def test_diagonal_neighbors_do_not_connect_across_thin_boundaries(self):
        mask = np.ones((5, 5), dtype=np.uint8)
        np.fill_diagonal(mask, 0)
        expected = np.zeros_like(mask, dtype=bool)
        expected[2, 2] = True
        np.testing.assert_array_equal(enclosed_mask_region(mask, (2, 2)), expected)

    def diamond_outline(self):
        mask = np.zeros((51, 51), dtype=np.uint8)
        points = np.array([(25, 5), (45, 25), (25, 45), (5, 25)], dtype=np.int32)
        cv2.polylines(mask, [points], True, 255, 1)
        y, x = np.indices(mask.shape)
        interior = np.abs(x - 25) + np.abs(y - 25) < 20
        return mask, interior

    def test_one_pixel_diagonal_outline_fills_and_subtracts_exact_interior(self):
        outline, expected = self.diamond_outline()
        for filled in (False, True):
            with self.subTest(filled=filled):
                mask = 255 - outline if filled else outline.copy()
                original = mask.copy()
                region = enclosed_mask_region(mask, (25, 25), filled=filled)
                np.testing.assert_array_equal(region, expected)
                np.testing.assert_array_equal(mask, original)

    def test_one_pixel_gap_is_rejected_even_when_gap_is_protected(self):
        outline, _ = self.diamond_outline()
        outline[5, 25] = 0
        protected = np.zeros_like(outline)
        protected[5, 25] = 255
        for filled in (False, True):
            for protection in (None, protected):
                with self.subTest(filled=filled, protected=protection is not None):
                    mask = 255 - outline if filled else outline
                    self.assertIsNone(enclosed_mask_region(
                        mask, (25, 25), filled=filled, protected_mask=protection
                    ))

    def test_thin_diagonal_outline_preserves_protected_interior(self):
        outline, interior = self.diamond_outline()
        protected = np.zeros_like(outline)
        protected[22:29, 22:29] = 255
        expected = interior & (protected == 0)
        for filled in (False, True):
            with self.subTest(filled=filled):
                mask = 255 - outline if filled else outline
                region = enclosed_mask_region(
                    mask, (25, 15), filled=filled, protected_mask=protected
                )
                np.testing.assert_array_equal(region, expected)
                self.assertIsNone(enclosed_mask_region(
                    mask, (25, 25), filled=filled, protected_mask=protected
                ))

    def test_brush_widths_fill_up_to_outline_without_an_unfilled_rim(self):
        shapes = {
            "square": [(10, 10), (40, 10), (40, 40), (10, 40)],
            "diamond": [(25, 5), (45, 25), (25, 45), (5, 25)],
        }
        for name, vertices in shapes.items():
            for width in (1, 2, 7):
                with self.subTest(shape=name, width=width):
                    points = np.array(vertices, dtype=np.int32)
                    outline = np.zeros((51, 51), dtype=np.uint8)
                    cv2.polylines(outline, [points], True, 255, width)
                    solid = np.zeros_like(outline)
                    cv2.fillPoly(solid, [points], 255)
                    expected = (solid > 0) & (outline == 0)
                    np.testing.assert_array_equal(
                        enclosed_mask_region(outline, (25, 25)), expected
                    )

    def test_bad_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            enclosed_mask_region(self.mask, (15, 15), protected_mask=np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            enclosed_mask_region(np.zeros((2, 2, 3)), (1, 1))


if __name__ == "__main__":
    unittest.main()
