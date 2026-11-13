"""Local contrast is deterministic, inexpensive on redraw, and display-only."""

import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QGraphicsTextItem

from core.categories import CATEGORY_COLORS
import colorsys

from core.instance_colors import (BEE_PALETTE, MIN_SATURATION, MIN_VALUE, contrasting_instance_colors,
                                  neighboring_boxes, random_vivid_color, vivid_color)
from gui.canvas import ImageCanvas
from gui.toolbar import AnnotationToolbar


def is_vivid(color):
    _, saturation, value = colorsys.rgb_to_hsv(*(c / 255 for c in color))
    return saturation >= MIN_SATURATION and value >= MIN_VALUE


class InstanceColorTests(unittest.TestCase):
    def test_no_grey_bee_colors_for_greyscale_frames(self):
        self.assertTrue(all(is_vivid(c) for c in BEE_PALETTE), BEE_PALETTE)
        self.assertEqual(len(set(BEE_PALETTE)), len(BEE_PALETTE))
        for c in BEE_PALETTE:
            self.assertEqual(vivid_color(c), c)
        for grey in ((60, 60, 60), (235, 235, 245), (128, 128, 128), (0, 0, 0), (255, 255, 255),
                     (200, 180, 160), (20, 40, 30)):
            self.assertTrue(is_vivid(vivid_color(grey)), grey)
            self.assertEqual(vivid_color(grey), vivid_color(grey))
        # A weak tint keeps its hue.
        hue = colorsys.rgb_to_hsv(*(c / 255 for c in (200, 170, 170)))[0]
        self.assertAlmostEqual(colorsys.rgb_to_hsv(*(c / 255 for c in vivid_color((200, 170, 170))))[0], hue,
                               places=2)
        state = np.random.get_state()
        self.assertTrue(all(is_vivid(random_vivid_color()) for _ in range(500)))
        self.assertEqual(random_vivid_color(7), random_vivid_color(7))
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])

    def test_touching_long_bodies_are_neighbors_not_just_close_centroids(self):
        graph = neighboring_boxes({1: (0, 0, 150, 10), 2: (150, 0, 150, 10),
                                   3: (1000, 0, 10, 10)})
        self.assertEqual(graph, {1: {2}, 2: {1}, 3: set()})

    def test_nearby_and_overlapping_boxes(self):
        graph = neighboring_boxes({1: (0, 0, 20, 20), 2: (25, 0, 20, 20),
                                   3: (10, 5, 20, 20)})
        self.assertTrue(all(len(neighbors) == 2 for neighbors in graph.values()))

    def test_dense_cluster_is_distinct_and_order_independent(self):
        boxes = {i: (i, 0, 20, 20) for i in range(1, 9)}
        colors = contrasting_instance_colors(boxes)
        self.assertEqual(len(set(colors.values())), 8)
        self.assertEqual(colors, contrasting_instance_colors(dict(reversed(list(boxes.items())))))

    def test_reuses_colors_at_a_distance_but_breaks_local_conflicts(self):
        preferred = {1: BEE_PALETTE[0], 2: BEE_PALETTE[0]}
        far = contrasting_instance_colors({1: (0, 0, 20, 20), 2: (100, 0, 20, 20)}, preferred=preferred)
        self.assertEqual(far, preferred)
        near = contrasting_instance_colors({1: (0, 0, 20, 20), 2: (20, 0, 20, 20)}, preferred=preferred)
        self.assertNotEqual(near[1], near[2])
        self.assertEqual(near, contrasting_instance_colors(
            {1: (0, 0, 20, 20), 2: (20, 0, 20, 20)}, preferred=near))

    def test_new_neighbor_keeps_existing_colors_fixed(self):
        fixed = contrasting_instance_colors({1: (0, 0, 20, 20), 2: (0, 20, 20, 20)})
        colors = contrasting_instance_colors({1: (0, 0, 20, 20), 2: (0, 20, 20, 20),
                                              3: (20, 0, 20, 20)}, fixed=fixed)
        self.assertEqual({i: colors[i] for i in fixed}, fixed)
        self.assertNotIn(colors[3], fixed.values())

    def test_invalid_and_empty_boxes_are_ignored_and_palette_exhaustion_is_safe(self):
        self.assertEqual(contrasting_instance_colors({1: None, 2: (0, 0, 0, 1),
                                                     3: (0, 0, -1, 1), 4: (float('nan'), 0, 1, 1)}), {})
        colors = contrasting_instance_colors({i: (0, 0, 20, 20) for i in range(1, 30)})
        self.assertEqual(len(colors), 29)
        self.assertTrue(set(colors.values()) <= set(BEE_PALETTE))

    def test_global_rng_is_unchanged(self):
        state = np.random.get_state()
        contrasting_instance_colors({1: (0, 0, 20, 20)})
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])


class CanvasColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def canvas(self):
        canvas = ImageCanvas()
        self.addCleanup(canvas.close)
        self.addCleanup(canvas._viz_update_timer.stop)
        self.addCleanup(canvas._brush_cursor_timer.stop)
        canvas.resize(800, 600)
        canvas.show()
        canvas.set_bee_color_context(('project', 'video'))
        canvas.load_image(np.zeros((200, 300), np.uint8))
        self.app.processEvents()
        annotations = []
        for iid, category, x in ((1, 'bee', 20), (2, 'bee', 55), (3, 'hive', 110)):
            mask = np.zeros((200, 300), np.uint8)
            mask[40:70, x:x + 30] = 255
            annotations.append(dict(mask_id=iid, category=category, mask=mask))
        annotations.append(dict(mask_id=4, category='bee', bbox_only=True, bbox=[20, 75, 30, 30]))
        canvas.set_annotations(annotations, {1: (60, 60, 60), 2: (61, 61, 61), 4: (60, 60, 60)})
        canvas.set_show_bboxes(True)
        return canvas

    def assert_base_colors_match(self, c):
        labels = []
        original_set_html = QGraphicsTextItem.setHtml

        def record_html(item, text):
            labels.append(text)
            original_set_html(item, text)

        # Qt normalizes rich-text CSS when reading toHtml back.
        with patch.object(QGraphicsTextItem, 'setHtml', record_html):
            c.update_instance_labels()
        for iid, x in ((1, 25), (2, 60)):
            color = c.display_color_for_instance(iid, 'bee')
            self.assertEqual(tuple(c._cached_overlay[45, x]), (*color, c.mask_opacity))
            self.assertEqual(c.bbox_items_map[iid].pen().color().getRgb()[:3], color)
            self.assertTrue(any(f'rgb({color[0]}, {color[1]}, {color[2]})' in text for text in labels))

    def test_masks_boxes_labels_match_and_fixed_categories_are_unchanged(self):
        c = self.canvas()
        self.assertNotEqual(c.display_color_for_instance(1), c.display_color_for_instance(2))
        self.assertNotEqual(c.display_color_for_instance(1), c.display_color_for_instance(4))
        self.assert_base_colors_match(c)
        self.assertEqual(c.display_color_for_instance(3), CATEGORY_COLORS['hive'])
        self.assertEqual(c.mask_colors[1], (60, 60, 60))

    def test_grey_saved_colors_display_vivid_without_changing_them(self):
        c = self.canvas()
        c.set_distinct_bee_colors(False)
        for iid in (1, 2, 4):
            self.assertTrue(is_vivid(c.display_color_for_instance(iid, 'bee')))
        self.assertNotEqual(c.display_color_for_instance(1), c.display_color_for_instance(2))
        self.assert_base_colors_match(c)
        self.assertEqual(c.mask_colors[1], (60, 60, 60))
        # Fixed category colours are never adjusted, even the soft brood ones.
        self.assertEqual(c.display_color_for_instance(3), CATEGORY_COLORS['hive'])
        c.annotation_metadata[9] = {'category': 'brood_middle_late'}
        self.assertEqual(c.display_color_for_instance(9, 'brood_middle_late'),
                         CATEGORY_COLORS['brood_middle_late'])

    def test_toggle_refresh_preserve_pending_annotations_history_and_view(self):
        c = self.canvas()
        c.start_editing_instance(1, 'bee')
        c.set_tool('brush')
        c._begin_brush_history_step()
        c.draw_on_mask(QPointF(30, 55), QPointF(45, 75))
        c._finish_brush_history_step()
        c._flush_pending_viz_update()
        original = (c.bee_mask.copy(), c.editing_mask.copy(), copy.deepcopy(c.annotation_metadata),
                    len(c.brush_undo_stack), dict(c.mask_colors), c.get_view_state())
        events = []
        c.annotation_changed.connect(lambda: events.append(True))
        for enabled in (False, True):
            c.set_distinct_bee_colors(enabled)
            color = c.display_color_for_instance(1)
            pixels = c._editing_overlay_cache[c.editing_mask > 0]
            self.assertTrue(np.all(pixels == (*color, c.mask_opacity)))
            self.assertEqual(c.editing_bbox_item.pen().color().getRgb()[:3], color)
            c.refresh_bee_colors()
        np.testing.assert_array_equal(c.bee_mask, original[0])
        np.testing.assert_array_equal(c.editing_mask, original[1])
        self.assertEqual(c.annotation_metadata, original[2])
        self.assertEqual(len(c.brush_undo_stack), original[3])
        self.assertEqual(c.mask_colors, original[4])
        self.assertEqual(c.get_view_state(), original[5])
        self.assertEqual(events, [])

    def test_visibility_and_brushing_do_not_recompute_colors(self):
        c = self.canvas()
        before = dict(c._bee_display_colors)
        with patch('gui.canvas.contrasting_instance_colors', side_effect=AssertionError('recolor')):
            c.set_instance_visible(2, 'bee', False)
            c.set_instance_visible(2, 'bee', True)
            c.set_annotation_type_visibility('bee', False)
            c.set_annotation_type_visibility('bee', True)
            c.toggle_annotation_overlays()
            c.toggle_annotation_overlays()
            c.set_mask_opacity(100)
            c.start_editing_instance(1, 'bee')
            c.set_tool('brush')
            c.draw_on_mask(QPointF(30, 55), QPointF(40, 65))
            c._flush_pending_viz_update()
            c.commit_editing()
        self.assertEqual(c._bee_display_colors, before)

    def test_disabled_uses_original_colors_and_reenable_restores_contrast(self):
        c = self.canvas()
        contrast = dict(c._bee_display_colors)
        c.set_distinct_bee_colors(False)
        # Original colours, except that grey ones are shown vivid on greyscale frames.
        self.assertEqual(c.display_color_for_instance(1), vivid_color((60, 60, 60)))
        c.mask_colors[2] = (0, 170, 255)
        self.assertEqual(c.display_color_for_instance(2), (0, 170, 255))
        c.rebuild_visualizations()
        self.assert_base_colors_match(c)
        c.set_distinct_bee_colors(True)
        self.assertEqual(c._bee_display_colors, contrast)
        self.assert_base_colors_match(c)

    def test_reload_is_stable_and_context_change_clears_preferences(self):
        c = self.canvas()
        colors = dict(c._bee_display_colors)
        annotations = c.get_annotations()
        c.load_image(np.zeros((200, 300), np.uint8))
        c.set_annotations(annotations)
        self.assertEqual(c._bee_display_colors, colors)
        c.set_bee_color_context(('project', 'other-video'))
        self.assertEqual(c._bee_color_preferences, {})
        self.assertEqual(c._bee_display_colors, {})
        c.set_annotations([])
        self.assertEqual(c._bee_display_colors, {})

    def test_refresh_repairs_new_neighbors_without_automatic_recoloring(self):
        c = self.canvas()
        c.set_bee_color_context(('project', 'new-video'))
        c.set_annotations([
            dict(mask_id=1, category='bee', bbox_only=True, bbox=[0, 0, 20, 20]),
            dict(mask_id=13, category='bee', bbox_only=True, bbox=[200, 0, 20, 20]),
        ])
        self.assertEqual(c.display_color_for_instance(1), c.display_color_for_instance(13))
        c.annotation_metadata[13]['bbox'] = [20, 0, 20, 20]
        c.rebuild_visualizations()
        self.assertEqual(c.display_color_for_instance(1), c.display_color_for_instance(13))
        c.refresh_bee_colors()
        self.assertNotEqual(c.display_color_for_instance(1), c.display_color_for_instance(13))

    def test_window_cache_keeps_original_colors(self):
        from types import SimpleNamespace
        from gui.main_window import MainWindow
        c = self.canvas()
        window = SimpleNamespace(canvas=c, current_video_id='video', video_mask_colors={})
        MainWindow._register_canvas_colors(window)
        self.assertEqual(window.video_mask_colors['video'][1], (60, 60, 60))
        c.refresh_bee_colors()
        MainWindow._register_canvas_colors(window)
        self.assertEqual(window.video_mask_colors['video'][1], (60, 60, 60))

    def test_window_loads_disabled_preference(self):
        import tempfile
        from types import SimpleNamespace
        from PyQt6.QtCore import QSettings
        from gui.main_window import MainWindow
        c = self.canvas()
        toolbar = AnnotationToolbar()
        self.addCleanup(toolbar.close)
        toolbar.distinct_bee_colors_changed.connect(c.set_distinct_bee_colors)
        window = SimpleNamespace(toolbar=toolbar, update_recent_projects_menu=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            settings = QSettings(directory + '/display.ini', QSettings.Format.IniFormat)
            settings.setValue('display/distinct_bee_colors', False)
            with patch('gui.main_window.QSettings', return_value=settings):
                MainWindow.load_settings(window)
        self.assertFalse(toolbar.distinct_bee_colors_checkbox.isChecked())
        self.assertFalse(c.distinct_bee_colors_enabled)

    def test_new_mask_and_category_change_do_not_recolor_neighbors(self):
        c = self.canvas()
        colors = dict(c._bee_display_colors)
        mask = np.zeros((200, 300), np.uint8)
        mask[10:40, 20:50] = 255
        c.add_mask(mask, mask_id=5, category='bee')
        self.assertEqual({i: c._bee_display_colors[i] for i in colors}, colors)
        self.assertNotEqual(c.display_color_for_instance(5), c.display_color_for_instance(1))
        c.change_instance_category(5, 'hive')
        self.assertEqual(c.display_color_for_instance(5), CATEGORY_COLORS['hive'])
        self.assertNotIn(5, c._bee_display_colors)

    def test_first_stroke_assigns_color_only_after_pen_lifts(self):
        c = self.canvas()
        colors = dict(c._bee_display_colors)
        c.annotation_metadata[5] = {'category': 'bee', 'mask_id': 5}
        c.start_editing_instance(5, 'bee')
        c.set_tool('brush')
        c.is_drawing = True
        c._is_actively_drawing = True
        c._begin_brush_history_step()
        c.draw_on_mask(QPointF(30, 20), QPointF(40, 20))
        self.assertNotIn(5, c._bee_display_colors)
        event = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(40, 20), QPointF(40, 20),
                            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        c.mouseReleaseEvent(event)
        self.assertIn(5, c._bee_display_colors)
        self.assertEqual({i: c._bee_display_colors[i] for i in colors}, colors)
        pixels = c._editing_overlay_cache[c.editing_mask > 0]
        self.assertTrue(np.all(pixels == (*c.display_color_for_instance(5), c.mask_opacity)))

    def test_toolbar_controls(self):
        c = self.canvas()
        toolbar = AnnotationToolbar()
        self.addCleanup(toolbar.close)
        toolbar.distinct_bee_colors_changed.connect(c.set_distinct_bee_colors)
        toolbar.refresh_bee_colors_requested.connect(c.refresh_bee_colors)
        toolbar.distinct_bee_colors_checkbox.setChecked(False)
        self.assertFalse(c.distinct_bee_colors_enabled)
        self.assertFalse(toolbar.refresh_bee_colors_btn.isEnabled())
        toolbar.distinct_bee_colors_checkbox.setChecked(True)
        with patch('gui.canvas.contrasting_instance_colors', wraps=contrasting_instance_colors) as refresh:
            toolbar.refresh_bee_colors_btn.click()
            self.assertEqual(refresh.call_count, 1)


if __name__ == '__main__':
    unittest.main()
