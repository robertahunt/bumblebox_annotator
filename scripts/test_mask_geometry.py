"""Geometry caching must preserve pixels, visibility, and every mutation path."""

import gc
import os
import unittest
import weakref
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QGraphicsPathItem, QMessageBox

from core.categories import BROOD_CATEGORIES
from core.mask_geometry import MaskGeometry, MaskGeometryCache, measure_mask
from gui.canvas import ImageCanvas
from gui.main_window import MainWindow


def sample_mask():
    mask = np.zeros((64, 64), np.uint8)
    mask[7:32, 9:40] = 255
    mask[12:20, 14:25] = 0
    mask[45:55, 48:60] = 255
    return mask


def reference_geometry(mask):
    y, x = np.nonzero(mask > 0)
    if not len(x):
        return MaskGeometry()
    return MaskGeometry(len(x), (int(x.min()), int(y.min()), int(x.max() - x.min() + 1),
                                int(y.max() - y.min() + 1)), (float(x.mean()), float(y.mean())))


class MaskGeometryTests(unittest.TestCase):
    def test_measurements_match_full_pixel_calculation(self):
        rng = np.random.default_rng(42)
        for mask in (sample_mask(), sample_mask() > 0, sample_mask()[::2, ::2],
                     np.zeros((10, 20), np.uint8), np.ones((10, 20), np.uint8),
                     np.eye(30, dtype=np.uint8), np.ones((1, 1), np.uint8),
                     rng.integers(-2, 3, (50, 60), dtype=np.int32)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                self.assertEqual(measure_mask(mask), reference_geometry(mask))

    def test_cache_is_small_and_does_not_retain_source_arrays(self):
        cache = MaskGeometryCache()
        source = np.zeros((64, 64), np.int32)
        source[sample_mask() > 0] = 1_000_000_001
        ref = weakref.ref(source)
        self.assertEqual(cache.instance_ids('bee', source), (1_000_000_001,))
        geometry = cache.geometry('bee', source, 1_000_000_001)
        self.assertEqual(geometry, reference_geometry(sample_mask()))
        with patch('core.mask_geometry.measure_mask', side_effect=AssertionError('Cache miss')):
            self.assertIs(cache.geometry('bee', source, 1_000_000_001), geometry)
        self.assertTrue(all(not isinstance(value, np.ndarray) for value in vars(geometry).values()))
        del source
        gc.collect()
        self.assertIsNone(ref())

    def test_category_invalidation_and_array_replacement(self):
        cache = MaskGeometryCache()
        source = (sample_mask() > 0).astype(np.int32)
        saved = cache.geometry('hive', source, 1)
        cache.geometry('nectar', source, 1)
        cache.invalidate('nectar')
        self.assertIs(cache.geometry('hive', source, 1), saved)
        empty = np.zeros_like(source)
        self.assertEqual(cache.geometry('hive', empty, 1), MaskGeometry())
        self.assertEqual(cache.instance_ids('hive', empty), ())
        empty[0, 0] = 2
        cache.invalidate('hive')
        self.assertEqual(cache.instance_ids('hive', empty), (2,))
        self.assertEqual(cache.geometry('hive', empty, 2).bbox, (0, 0, 1, 1))


class CanvasGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def canvas(self):
        canvas = ImageCanvas()
        self.addCleanup(canvas.close)
        self.addCleanup(canvas._viz_update_timer.stop)
        canvas.load_image(np.zeros((64, 64, 3), np.uint8))
        canvas.set_annotations([
            dict(mask_id=1, category='nectar', mask=sample_mask()),
            dict(mask_id=2, category='hive', mask=np.full((64, 64), 255, np.uint8)),
        ])
        canvas.set_annotation_type_visibility('nectar', True)
        canvas.set_annotation_type_visibility('hive', True)
        canvas.set_show_bboxes(True)
        return canvas

    def assert_geometry(self, canvas, instance_id, category, mask):
        actual = canvas.get_instance_geometry(instance_id, category)
        expected = reference_geometry(mask)
        self.assertEqual(actual.area, expected.area)
        self.assertEqual(actual.bbox, expected.bbox)
        if expected.centroid is None:
            self.assertIsNone(actual.centroid)
        else:
            np.testing.assert_allclose(actual.centroid, expected.centroid, rtol=0, atol=1e-10)
        if np.any(mask):
            self.assertEqual(canvas.get_instance_bbox_cached(instance_id, category), reference_geometry(mask).bbox)

    def test_repeated_visibility_and_sidebar_queries_reuse_geometry(self):
        canvas = self.canvas()
        # Warm all stored masks. Changing visibility must not change their geometry.
        for entry in canvas.get_instance_entries():
            canvas.get_instance_geometry(entry['id'], entry['category'])
        with patch('core.mask_geometry.measure_mask', side_effect=AssertionError('Unexpected scan')):
            for _ in range(2):
                canvas.set_annotation_type_visibility('nectar', False)
                canvas.set_annotation_type_visibility('nectar', True)
                canvas.set_instance_visible(1, 'nectar', False)
                canvas.set_instance_visible(1, 'nectar', True)
                canvas.toggle_annotation_overlays()
                canvas.toggle_annotation_overlays()
                canvas.get_instance_entries()

    def test_brush_erase_undo_redo_and_commit_measure_live_pixels(self):
        canvas = self.canvas()
        hive_geometry = canvas.get_instance_geometry(2, 'hive')
        canvas.start_editing_instance(1, category='nectar')
        self.assert_geometry(canvas, 1, 'nectar', canvas.editing_mask)
        for tool, start, end in [('brush', (20, 28), (52, 35)), ('eraser', (20, 25), (35, 25))]:
            canvas.set_tool(tool)
            canvas.brush_size = 7
            before = canvas.editing_mask.copy()
            self.assertTrue(canvas._start_brush_stroke_at(QPointF(*start)))
            canvas.draw_on_mask(QPointF(*start), QPointF(*end))
            canvas._finish_brush_history_step()
            canvas.is_drawing = False
            after = canvas.editing_mask.copy()
            self.assert_geometry(canvas, 1, 'nectar', after)
            canvas.undo()
            np.testing.assert_array_equal(canvas.editing_mask, before)
            self.assert_geometry(canvas, 1, 'nectar', before)
            canvas.redo()
            np.testing.assert_array_equal(canvas.editing_mask, after)
            self.assert_geometry(canvas, 1, 'nectar', after)
        canvas.commit_editing()
        self.assert_geometry(canvas, 1, 'nectar', after)
        self.assertIs(canvas.get_instance_geometry(2, 'hive'), hive_geometry)

    def test_fill_subtract_and_external_edit_buffer_replacement(self):
        canvas = self.canvas()
        canvas.start_editing_instance(1, category='nectar')
        self.assertTrue(canvas.fill_enclosed_region_at(QPointF(16, 16)))
        self.assert_geometry(canvas, 1, 'nectar', canvas.editing_mask)
        self.assertTrue(canvas.subtract_enclosed_region_at(QPointF(16, 16)))
        self.assert_geometry(canvas, 1, 'nectar', canvas.editing_mask)
        # SAM2/refinement can replace the buffer without calling a cache API.
        canvas.editing_mask = np.eye(64, dtype=np.uint8) * 255
        self.assert_geometry(canvas, 1, 'nectar', canvas.editing_mask)
        canvas.editing_mask[:20] = 0
        self.assert_geometry(canvas, 1, 'nectar', canvas.editing_mask)
        final = canvas.editing_mask.copy()
        canvas.commit_editing()
        self.assert_geometry(canvas, 1, 'nectar', final)

    def test_reclassification_invalidates_source_and_overwritten_target(self):
        canvas = self.canvas()
        canvas.change_instance_category(1, 'hive', old_category='nectar')
        self.assert_geometry(canvas, 1, 'hive', sample_mask())
        self.assert_geometry(canvas, 2, 'hive', (sample_mask() == 0).astype(np.uint8))
        self.assertEqual(canvas.get_instance_geometry(1, 'nectar'), MaskGeometry())
        canvas.start_editing_instance(1, category='hive')
        canvas.change_instance_category(1, 'brood_middle', old_category='hive')
        self.assert_geometry(canvas, 1, 'brood_middle', sample_mask())
        canvas.commit_editing()
        self.assert_geometry(canvas, 1, 'brood_middle', sample_mask())

    def test_add_delete_reassign_and_merge_invalidate_geometry_and_ids(self):
        canvas = self.canvas()
        extra = np.zeros((64, 64), np.uint8)
        extra[2:6, 3:9] = 255
        canvas.add_mask(extra, mask_id=3, category='nectar')
        self.assert_geometry(canvas, 3, 'nectar', extra)
        self.assertTrue(canvas.reassign_instance_id(3, 1))
        self.assert_geometry(canvas, 1, 'nectar', np.maximum(sample_mask(), extra))
        self.assertTrue(canvas.reassign_instance_id(1, 7))
        self.assert_geometry(canvas, 7, 'nectar', np.maximum(sample_mask(), extra))
        canvas.add_mask(extra, mask_id=8, category='nectar')
        self.assert_geometry(canvas, 7, 'nectar', sample_mask())
        self.assert_geometry(canvas, 8, 'nectar', extra)
        canvas.delete_instance(7, category='nectar')
        self.assertEqual(canvas.get_instance_geometry(7, 'nectar'), MaskGeometry())
        self.assertEqual({entry['id'] for entry in canvas.get_instance_entries()}, {2, 8})

    def test_split_updates_both_pieces_and_cached_instance_ids(self):
        canvas = self.canvas()
        self.assertTrue(canvas.split_hovered_segment_at(QPointF(50, 50)))
        new_id = canvas.next_mask_id - 1
        split = np.zeros((64, 64), np.uint8)
        split[45:55, 48:60] = 255
        remaining = sample_mask()
        remaining[split > 0] = 0
        self.assert_geometry(canvas, 1, 'nectar', remaining)
        self.assert_geometry(canvas, new_id, 'nectar', split)
        self.assertEqual({entry['id'] for entry in canvas.get_instance_entries()}, {1, 2, new_id})

    def test_frame_replacement_bbox_metadata_and_zero_mask(self):
        canvas = self.canvas()
        canvas.load_image(np.zeros((32, 48, 3), np.uint8))
        canvas.set_annotations([dict(mask_id=1, category='nectar', bbox=[5, 6, 7, 8], bbox_only=True)])
        self.assertEqual(canvas.get_instance_geometry(1, 'nectar'), MaskGeometry())
        self.assertEqual(canvas.get_instance_bbox_cached(1, 'nectar'), (5, 6, 7, 8))
        canvas.annotation_metadata[1]['bbox'] = [7, 9, 10, 11]
        self.assertEqual(canvas.get_instance_bbox_cached(1, 'nectar'), (7, 9, 10, 11))
        canvas.start_editing_instance(1, category='nectar')
        canvas.editing_mask[12:18, 20:30] = 255
        self.assert_geometry(canvas, 1, 'nectar', canvas.editing_mask)
        canvas.commit_editing()
        canvas.start_editing_instance(1, category='nectar')
        canvas.editing_mask[:] = 0
        canvas.commit_editing()
        self.assertEqual(canvas.get_instance_geometry(1, 'nectar'), MaskGeometry())
        self.assertEqual(canvas.get_instance_entries(), [])
        canvas.clear_image()
        self.assertFalse(canvas._geometry_cache._categories)

    def test_roi_composite_and_contours_match_full_frame_rendering(self):
        canvas = self.canvas()
        order = ('chamber', 'hive', 'pollen', 'nectar') + BROOD_CATEGORIES + ('bee',)
        for hidden in (False, True):
            canvas.set_instance_visible(1, 'nectar', not hidden)
            expected = np.zeros((64, 64, 4), np.uint8)
            for category in order:
                mask = canvas._get_mask_array_by_category(category)
                if mask is None or not canvas.annotation_type_visibility[category]:
                    continue
                for instance_id in np.unique(mask):
                    if instance_id > 0 and canvas.is_instance_enabled(int(instance_id), category):
                        expected[mask == instance_id] = (*canvas.mask_colors[int(instance_id)], canvas.mask_opacity)
            np.testing.assert_array_equal(canvas._cached_overlay, expected)
        canvas.set_annotation_type_visibility('hive', False)
        canvas.set_instance_visible(1, 'nectar', True)
        path_items = [item for item in canvas.mask_items if isinstance(item, QGraphicsPathItem)]
        self.assertEqual(len(path_items), 1)
        path = path_items[0].path()
        actual = {(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())}
        contours, _ = cv2.findContours(sample_mask(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        expected = {tuple(point[0]) for contour in contours if len(contour) >= 3 for point in contour}
        self.assertEqual(actual, expected)


class RefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self):
        with patch.object(MainWindow, 'load_settings'):
            window = MainWindow()

        def close():
            with patch('gui.main_window.QSettings'), patch.object(
                    QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                window.close()
            self.app.processEvents()

        self.addCleanup(close)
        window.canvas.load_image(np.zeros((64, 64, 3), np.uint8))
        window.canvas.set_annotations([
            dict(mask_id=1, category='nectar', mask=sample_mask()),
            dict(mask_id=2, category='hive', mask=sample_mask()),
            dict(mask_id=3, category='brood_early', mask=sample_mask()),
        ])
        window.toolbar.show_nectar_checkbox.setChecked(True)
        window.toolbar.show_hives_checkbox.setChecked(True)
        window.toolbar.brood_visibility_actions['brood_early'].setChecked(True)
        window.update_instance_list_from_canvas()
        return window

    def test_visibility_and_reclassification_refresh_labels_once(self):
        window = self.window()
        canvas = window.canvas
        with patch.object(canvas, 'update_instance_labels', wraps=canvas.update_instance_labels) as labels:
            window.toolbar.brood_visibility_checkbox.click()
            labels.assert_called_once()
        canvas.start_editing_instance(3, category='brood_early')
        with patch.object(canvas, 'update_instance_labels', wraps=canvas.update_instance_labels) as labels, \
                patch.object(window, 'update_instance_list_from_canvas',
                             wraps=window.update_instance_list_from_canvas) as sidebar:
            canvas.change_instance_category(3, 'queen_brood_middle', old_category='brood_early')
            labels.assert_called_once()
            sidebar.assert_called_once()
        self.assertFalse(window._instance_list_update_timer.isActive())

    def test_sidebar_refresh_cancels_queued_duplicate_without_redrawing_labels(self):
        window = self.window()
        window._schedule_instance_list_update()
        with patch.object(window.canvas, 'update_instance_labels') as labels:
            window.update_instance_list_from_canvas()
            labels.assert_not_called()
        self.assertFalse(window._instance_list_update_timer.isActive())

    def test_sidebar_mask_switch_rebuilds_once_and_preserves_pixel_edits(self):
        window = self.window()
        canvas = window.canvas
        canvas.set_tool('brush')
        canvas.start_editing_instance(1, category='nectar')
        canvas.editing_mask[2:5, 2:5] = 255
        edited = canvas.editing_mask.copy()
        index = next(i for i in range(window.instance_list.count())
                     if window.instance_list.item(i).data(Qt.ItemDataRole.UserRole)['id'] == 2)
        window.instance_list.clearSelection()
        with patch.object(canvas, 'rebuild_visualizations', wraps=canvas.rebuild_visualizations) as rebuild, \
                patch.object(canvas, 'update_instance_labels', wraps=canvas.update_instance_labels) as labels:
            window.instance_list.setCurrentRow(index)
            window.on_instance_clicked(window.instance_list.item(index))
            rebuild.assert_called_once()
            labels.assert_called_once()
        self.assertEqual(canvas.editing_instance_id, 2)
        np.testing.assert_array_equal(canvas.nectar_mask == 1, edited > 0)
        self.assertEqual(canvas.get_instance_geometry(1, 'nectar'), reference_geometry(edited))


if __name__ == '__main__':
    unittest.main()
