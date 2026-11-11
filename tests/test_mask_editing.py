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

    def open_outline_and_neighbor(self):
        outline = np.zeros((30, 30), dtype=np.uint8)
        outline[5, 5:24] = 255
        outline[24, 5:24] = 255
        outline[5:25, 5] = 255
        neighbor = np.zeros_like(outline, dtype=bool)
        neighbor[5:25, 24:28] = True
        expected = np.zeros_like(outline, dtype=bool)
        expected[6:24, 6:24] = True
        return outline, neighbor, expected

    def test_neighbor_completes_outline_at_exact_pixel_edge_without_mutation(self):
        outline, neighbor, expected = self.open_outline_and_neighbor()
        originals = outline.copy(), neighbor.copy()
        self.assertIsNone(enclosed_mask_region(outline, (15, 15)))
        region = enclosed_mask_region(outline, (15, 15), boundary_mask=neighbor)
        np.testing.assert_array_equal(region, expected)
        np.testing.assert_array_equal(outline, originals[0])
        np.testing.assert_array_equal(neighbor, originals[1])
        self.assertFalse(np.any(region & neighbor))

    def test_neighbor_fill_excludes_intruding_pixels_and_protected_interior(self):
        outline, neighbor, expected = self.open_outline_and_neighbor()
        neighbor[12:18, 18:25] = True
        protection = np.zeros_like(outline)
        protection[8:11, 8:11] = 255
        region = enclosed_mask_region(outline, (15, 15), boundary_mask=neighbor,
                                      protected_mask=protection)
        np.testing.assert_array_equal(region, expected & ~neighbor & (protection == 0))
        self.assertIsNone(enclosed_mask_region(outline, (20, 15), boundary_mask=neighbor))
        self.assertIsNone(enclosed_mask_region(outline, (9, 9), boundary_mask=neighbor,
                                              protected_mask=protection))

    def test_neighbor_cannot_close_an_actual_gap_or_repurpose_protection(self):
        outline, neighbor, _ = self.open_outline_and_neighbor()
        outline[5, 15] = 0
        protection = np.zeros_like(outline)
        protection[5, 15] = 255
        self.assertIsNone(enclosed_mask_region(outline, (15, 15), boundary_mask=neighbor,
                                              protected_mask=protection))

    def test_neighbor_does_not_make_an_edge_connected_region_fillable(self):
        outline, neighbor, _ = self.open_outline_and_neighbor()
        self.assertIsNone(enclosed_mask_region(outline, (2, 2), boundary_mask=neighbor))

    def test_neighbor_fill_requires_selected_outline_to_border_region(self):
        neighbor = self.mask > 0
        for focal in (np.zeros_like(self.mask), np.eye(30, dtype=np.uint8)):
            # A distant selected component must not authorize filling a hole
            # belonging entirely to the neighboring instance.
            focal[5:25, 5:25] = 0
            self.assertIsNone(enclosed_mask_region(focal, (15, 15), boundary_mask=neighbor))

    def test_neighbor_diagonal_edge_closes_one_pixel_outline(self):
        outline, expected = self.diamond_outline()
        neighbor = np.zeros_like(outline)
        neighbor[:26, 25:] = outline[:26, 25:]
        outline[neighbor > 0] = 0
        region = enclosed_mask_region(outline, (25, 25), boundary_mask=neighbor)
        np.testing.assert_array_equal(region, expected)

    def test_neighbor_fill_handles_multiple_disconnected_regions_separately(self):
        outline, neighbor, expected = self.open_outline_and_neighbor()
        neighbor[15, 6:25] = True
        expected[15:] = False
        np.testing.assert_array_equal(
            enclosed_mask_region(outline, (15, 10), boundary_mask=neighbor), expected)

    def test_boundary_shape_and_subtract_misuse_are_rejected(self):
        with self.assertRaises(ValueError):
            enclosed_mask_region(self.mask, (15, 15), boundary_mask=np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            enclosed_mask_region(self.mask, (5, 5), filled=True,
                                 boundary_mask=np.zeros_like(self.mask))


if __name__ == "__main__":
    unittest.main()
