#!/usr/bin/env python3
"""Focused regression tests for bbox resizing and protected brush zones."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from gui.canvas import ImageCanvas
from gui.toolbar import AnnotationToolbar


class EditingToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_bbox_corner_shrinks_inward_without_flipping(self):
        original = QRectF(10, 20, 100, 80)
        resized = ImageCanvas._resize_bbox_rect(
            original, "br", QPointF(70, 60), minimum_size=3
        )
        self.assertEqual((resized.x(), resized.y()), (10.0, 20.0))
        self.assertEqual((resized.width(), resized.height()), (60.0, 40.0))

        crossed = ImageCanvas._resize_bbox_rect(
            original, "br", QPointF(0, 0), minimum_size=3
        )
        self.assertEqual((crossed.x(), crossed.y()), (10.0, 20.0))
        self.assertEqual((crossed.width(), crossed.height()), (3.0, 3.0))

    def test_bbox_edge_handle_changes_only_one_dimension(self):
        original = QRectF(10, 20, 100, 80)
        resized = ImageCanvas._resize_bbox_rect(
            original, "ml", QPointF(40, 999), minimum_size=3
        )
        self.assertEqual((resized.x(), resized.y()), (40.0, 20.0))
        self.assertEqual((resized.width(), resized.height()), (70.0, 80.0))

    def test_measurement_line_reports_pixels_and_centimeters(self):
        canvas = ImageCanvas()
        canvas.load_image(np.zeros((100, 100), dtype=np.uint8))
        canvas.set_pixels_per_cm(5.0)
        canvas._update_measurement_line(QPointF(10, 10), QPointF(13, 14))

        self.assertAlmostEqual(canvas.get_measurement_pixel_length(), 5.0)
        label = canvas.measurement_label_item.toPlainText()
        self.assertIn("5.0 px", label)
        self.assertIn("1.000 cm", label)

        canvas.clear_measurement_line()
        self.assertIsNone(canvas.measurement_line_item)
        self.assertIsNone(canvas.measurement_label_item)

    def test_bbox_dimension_label_reports_x_and_y_distances(self):
        canvas = ImageCanvas()
        canvas.load_image(np.zeros((100, 100), dtype=np.uint8))
        canvas.set_pixels_per_cm(10.0)
        canvas._update_bbox_dimension_label(QRectF(5, 10, 30, 20))

        label = canvas.bbox_dimension_label_item.toPlainText()
        self.assertIn("Width (x): 30.0 px | 3.000 cm", label)
        self.assertIn("Height (y): 20.0 px | 2.000 cm", label)

    def test_polygon_preview_reports_active_segment_length(self):
        canvas = ImageCanvas()
        canvas.load_image(np.zeros((100, 100), dtype=np.uint8))
        canvas.set_pixels_per_cm(5.0)
        canvas._add_no_draw_polygon_point(QPointF(10, 10))
        canvas._update_polygon_dimension_label(QPointF(13, 14))

        label = canvas.polygon_dimension_label_item.toPlainText()
        self.assertIn("Segment: 5.0 px | 1.000 cm", label)

    def test_toolbar_selects_named_imaging_setup(self):
        toolbar = AnnotationToolbar()
        toolbar.set_imaging_setups(
            ["Setup B", "Setup A"], current_setup="Setup B"
        )

        self.assertEqual(toolbar.imaging_setup_combo.currentData(), "Setup B")
        self.assertEqual(
            [
                toolbar.imaging_setup_combo.itemData(index)
                for index in range(toolbar.imaging_setup_combo.count())
            ],
            ["", "Setup A", "Setup B"],
        )

        toolbar.set_measurement_scale(25.0, setup_name="Setup B")
        self.assertIn("Shared setup: 25.000 px/cm", toolbar.measurement_scale_label.text())

    def test_toolbar_can_reapply_displayed_setup_to_batch_selection(self):
        toolbar = AnnotationToolbar()
        toolbar.set_imaging_setups(["Setup A"], current_setup="Setup A")
        emitted_setups = []
        toolbar.apply_imaging_setup_requested.connect(emitted_setups.append)

        toolbar.apply_imaging_setup_btn.click()

        self.assertEqual(emitted_setups, ["Setup A"])

    def make_canvas(self, initial_value=0):
        canvas = ImageCanvas()
        canvas.current_image = np.zeros((100, 100), dtype=np.uint8)
        canvas.bee_mask = np.zeros((100, 100), dtype=np.int32)
        canvas.selected_mask_idx = 1
        canvas.selected_instance_category = "bee"
        canvas.editing_instance_id = 1
        canvas.editing_instance_category = "bee"
        canvas.editing_mask = np.full((100, 100), initial_value, dtype=np.uint8)
        canvas.annotation_metadata[1] = {"mask_id": 1, "category": "bee"}
        canvas.brush_size = 9
        return canvas

    def add_square_zone(self, canvas):
        for x, y in ((40, 40), (60, 40), (60, 60), (40, 60)):
            canvas._add_no_draw_polygon_point(QPointF(x, y))
        self.assertTrue(canvas._finish_no_draw_polygon())
        self.assertEqual(canvas.no_draw_zone_mask[50, 50], 255)

    def test_no_draw_zone_blocks_brush(self):
        canvas = self.make_canvas(initial_value=0)
        self.add_square_zone(canvas)
        canvas.current_tool = "brush"
        canvas.draw_on_mask(QPointF(10, 50), QPointF(90, 50))
        self.assertEqual(canvas.editing_mask[50, 25], 255)
        self.assertEqual(canvas.editing_mask[50, 50], 0)

    def test_no_draw_zone_blocks_eraser(self):
        canvas = self.make_canvas(initial_value=255)
        self.add_square_zone(canvas)
        canvas.current_tool = "eraser"
        canvas.draw_on_mask(QPointF(10, 50), QPointF(90, 50))
        self.assertEqual(canvas.editing_mask[50, 25], 0)
        self.assertEqual(canvas.editing_mask[50, 50], 255)

    def test_fill_respects_protection_and_supports_undo_redo(self):
        canvas = self.make_canvas()
        canvas.editing_mask[20:80, 20:80] = 255
        canvas.editing_mask[30:70, 30:70] = 0
        self.add_square_zone(canvas)
        original = canvas.editing_mask.copy()
        self.assertTrue(canvas.fill_enclosed_region_at(QPointF(35, 35)))
        self.assertEqual(canvas.editing_mask[35, 35], 255)
        self.assertEqual(canvas.editing_mask[50, 50], 0)
        filled = canvas.editing_mask.copy()
        canvas.undo()
        np.testing.assert_array_equal(canvas.editing_mask, original)
        canvas.redo()
        np.testing.assert_array_equal(canvas.editing_mask, filled)

    def test_subtract_preserves_protected_pixels_and_other_components(self):
        canvas = self.make_canvas()
        canvas.editing_mask[20:80, 20:80] = 255
        canvas.editing_mask[5:10, 5:10] = 255
        self.add_square_zone(canvas)
        original = canvas.editing_mask.copy()
        self.assertTrue(canvas.subtract_enclosed_region_at(QPointF(25, 25)))
        self.assertEqual(canvas.editing_mask[25, 25], 0)
        self.assertEqual(canvas.editing_mask[50, 50], 255)
        self.assertEqual(canvas.editing_mask[7, 7], 255)
        canvas.undo()
        np.testing.assert_array_equal(canvas.editing_mask, original)

    def test_space_toggle_preserves_class_instance_and_bbox_switches(self):
        canvas = self.make_canvas(initial_value=255)
        canvas.set_annotation_type_visibility('bee', False)
        canvas.set_instance_visible(2, 'pollen', False)
        canvas.set_annotation_overlay_visibility(True, False)
        hidden = canvas.hidden_instance_keys.copy()
        canvas.toggle_annotation_overlays()
        self.assertFalse(canvas.show_segmentations)
        self.assertFalse(canvas.show_bboxes)
        canvas.toggle_annotation_overlays()
        self.assertTrue(canvas.show_segmentations)
        self.assertFalse(canvas.show_bboxes)
        self.assertFalse(canvas.annotation_type_visibility['bee'])
        self.assertEqual(canvas.hidden_instance_keys, hidden)

    def test_individual_switch_cannot_override_hidden_class(self):
        canvas = self.make_canvas()
        canvas.set_annotation_type_visibility('bee', False)
        canvas.set_instance_visible(1, 'bee', True)
        self.assertFalse(canvas.is_annotation_instance_visible(1, 'bee'))
        canvas.set_annotation_type_visibility('bee', True)
        self.assertTrue(canvas.is_annotation_instance_visible(1, 'bee'))

    def test_loading_next_image_preserves_zoom_and_position(self):
        canvas = ImageCanvas()
        canvas.resize(500, 500)
        canvas.load_image(np.zeros((1000, 1000), dtype=np.uint8))
        canvas.show()
        self.app.processEvents()
        canvas.scale(3, 3)
        canvas.centerOn(QPointF(700, 650))
        before = canvas.get_view_state()
        canvas.load_image(np.ones((1000, 1000), dtype=np.uint8), preserve_view=True)
        after = canvas.get_view_state()
        self.assertEqual(before, after)

    def test_can_switch_instances_after_erasing_active_mask_entirely(self):
        canvas = ImageCanvas()
        canvas.resize(500, 500)
        canvas.load_image(np.zeros((100, 100), dtype=np.uint8))
        canvas.bee_mask = np.zeros((100, 100), dtype=np.int32)
        canvas.bee_mask[10:30, 10:30] = 1
        canvas.bee_mask[60:80, 60:80] = 2
        canvas.annotation_metadata = {
            1: {"mask_id": 1, "category": "bee"},
            2: {"mask_id": 2, "category": "bee"},
        }
        canvas.set_tool("eraser")
        canvas.start_editing_instance(1, category="bee")
        canvas.show()
        self.app.processEvents()

        # Reproduce the end of an eraser stroke that removes every pixel.
        canvas.editing_mask[:] = 0
        canvas._editing_started_with_pixels = True
        canvas.is_drawing = True
        erased_pos = canvas.mapFromScene(QPointF(20, 20))
        QTest.mouseRelease(
            canvas.viewport(), Qt.MouseButton.LeftButton, pos=erased_pos
        )
        self.app.processEvents()
        self.assertEqual(canvas.editing_instance_id, -1)
        self.assertNotIn(1, canvas.annotation_metadata)

        # Match the click + double-click event stream produced by a real
        # triple-click. The double-click event may not be followed by a release
        # on every platform, so it must count its tap immediately.
        target_pos = canvas.mapFromScene(QPointF(70, 70))
        QTest.mouseClick(
            canvas.viewport(), Qt.MouseButton.LeftButton, pos=target_pos, delay=30
        )
        QTest.mouseDClick(
            canvas.viewport(), Qt.MouseButton.LeftButton, pos=target_pos, delay=30
        )
        QTest.mouseClick(
            canvas.viewport(), Qt.MouseButton.LeftButton, pos=target_pos, delay=30
        )
        self.app.processEvents()

        self.assertEqual(canvas.editing_instance_id, 2)
        self.assertEqual(canvas.selected_mask_idx, 2)
        self.assertEqual(canvas.editing_instance_category, "bee")


if __name__ == "__main__":
    unittest.main()
