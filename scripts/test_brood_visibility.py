"""Group visibility controls must summarize, not override, subclass choices."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.categories import STANDARD_BROOD_CATEGORIES, QUEEN_BROOD_CATEGORIES
from gui.main_window import MainWindow
from gui.toolbar import AnnotationToolbar


GROUPS = {'brood': STANDARD_BROOD_CATEGORIES, 'queen_brood': QUEEN_BROOD_CATEGORIES}


class BroodVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def toolbar(self):
        toolbar = AnnotationToolbar()
        self.addCleanup(toolbar.close)
        return toolbar

    def assert_group(self, toolbar, name, checked_categories):
        categories = GROUPS[name]
        self.assertEqual({cat for cat in categories if toolbar.brood_visibility_actions[cat].isChecked()},
                         set(checked_categories))
        state = (Qt.CheckState.Checked if len(checked_categories) == len(categories) else
                 Qt.CheckState.PartiallyChecked if checked_categories else Qt.CheckState.Unchecked)
        self.assertEqual(toolbar.brood_visibility_checkboxes[name].checkState(), state)

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
        annotations = []
        for index, category in enumerate((*STANDARD_BROOD_CATEGORIES, *QUEEN_BROOD_CATEGORIES), 1):
            mask = np.zeros((64, 64), np.uint8)
            mask[8:32, index * 6:index * 6 + 4] = 255
            annotations.append(dict(mask_id=index, category=category, mask=mask))
        window.canvas.set_annotations(annotations)
        window.update_instance_list_from_canvas()
        return window

    def test_defaults_and_two_click_all_on_all_off_are_independent(self):
        toolbar = self.toolbar()
        individual, batches = [], []
        toolbar.annotation_type_visibility_changed.connect(lambda *args: individual.append(args))
        toolbar.annotation_group_visibility_changed.connect(lambda *args: batches.append(args))
        for name, categories in GROUPS.items():
            with self.subTest(group=name):
                other = 'queen_brood' if name == 'brood' else 'brood'
                self.assert_group(toolbar, name, ())
                self.assert_group(toolbar, other, ())
                checkbox = toolbar.brood_visibility_checkboxes[name]
                checkbox.click()
                self.assert_group(toolbar, name, categories)
                self.assert_group(toolbar, other, ())
                self.assertEqual(batches[-1], (categories, True))
                checkbox.click()
                self.assert_group(toolbar, name, ())
                self.assertEqual(batches[-1], (categories, False))
        self.assertEqual(len(batches), 4)
        self.assertFalse(individual)
        self.assertTrue(toolbar.show_bees_checkbox.isChecked())

    def test_manual_menu_changes_recompute_summary_without_changing_siblings(self):
        toolbar = self.toolbar()
        batches, individual = [], []
        toolbar.annotation_group_visibility_changed.connect(lambda *args: batches.append(args))
        toolbar.annotation_type_visibility_changed.connect(lambda *args: individual.append(args))
        for name, categories in GROUPS.items():
            with self.subTest(group=name):
                enabled = []
                for category in categories:
                    toolbar.brood_visibility_actions[category].trigger()
                    enabled.append(category)
                    self.assert_group(toolbar, name, enabled)
                    self.assertEqual(individual[-1], (category, True))
                for category in categories:
                    toolbar.brood_visibility_actions[category].trigger()
                    enabled.remove(category)
                    self.assert_group(toolbar, name, enabled)
                    self.assertEqual(individual[-1], (category, False))
        self.assertFalse(batches)
        self.assertEqual(len(individual), 16)

    def test_partial_click_enables_all_then_next_click_disables_all(self):
        toolbar = self.toolbar()
        for name, categories in GROUPS.items():
            with self.subTest(group=name):
                checkbox = toolbar.brood_visibility_checkboxes[name]
                checkbox.click()
                toolbar.brood_visibility_actions[categories[1]].trigger()
                self.assert_group(toolbar, name, [c for c in categories if c != categories[1]])
                checkbox.click()
                self.assert_group(toolbar, name, categories)
                checkbox.click()
                self.assert_group(toolbar, name, ())

    def test_keyboard_activation_skips_partial_state(self):
        toolbar = self.toolbar()
        toolbar.show()
        self.app.processEvents()
        for name, categories in GROUPS.items():
            with self.subTest(group=name):
                checkbox = toolbar.brood_visibility_checkboxes[name]
                checkbox.setFocus()
                QTest.keyClick(checkbox, Qt.Key.Key_Space)
                self.assert_group(toolbar, name, categories)
                QTest.keyClick(checkbox, Qt.Key.Key_Space)
                self.assert_group(toolbar, name, ())

    def test_group_switch_updates_masks_boxes_and_sidebar_in_one_refresh(self):
        window = self.window()
        toolbar, canvas = window.toolbar, window.canvas
        for name, categories in GROUPS.items():
            with self.subTest(group=name):
                checkbox = toolbar.brood_visibility_checkboxes[name]
                ids = {entry['id'] for entry in canvas.get_instance_entries()
                       if entry['category'] in categories}
                with patch.object(canvas, 'rebuild_visualizations', wraps=canvas.rebuild_visualizations) as redraw, \
                        patch.object(window, 'update_instance_list_from_canvas',
                                     wraps=window.update_instance_list_from_canvas) as sidebar:
                    checkbox.click()
                    redraw.assert_called_once()
                    sidebar.assert_called_once()
                self.assertEqual(set(canvas.bbox_items_map), ids)
                self.assertIsNotNone(canvas._cached_overlay)
                for i in range(window.instance_list.count()):
                    item = window.instance_list.item(i)
                    category = item.data(Qt.ItemDataRole.UserRole)['type']
                    self.assertEqual('[class hidden]' in item.text(), category not in categories)
                checkbox.click()
                self.assertFalse(canvas.bbox_items_map)
                self.assertIsNone(canvas._cached_overlay)

    def test_group_switch_preserves_global_layers_and_hidden_instances(self):
        window = self.window()
        toolbar, canvas = window.toolbar, window.canvas
        hidden_category = STANDARD_BROOD_CATEGORIES[0]
        canvas.set_instance_visible(1, hidden_category, False)
        hidden = canvas.hidden_instance_keys.copy()
        window.on_show_bboxes_changed(False)
        window.on_show_segmentations_changed(False)
        toolbar.brood_visibility_checkbox.click()
        self.assert_group(toolbar, 'brood', STANDARD_BROOD_CATEGORIES)
        self.assertFalse(canvas.show_segmentations)
        self.assertFalse(canvas.show_bboxes)
        self.assertFalse(canvas.bbox_items_map)
        self.assertIsNone(canvas._cached_overlay)
        self.assertEqual(canvas.hidden_instance_keys, hidden)
        window.on_show_segmentations_changed(True)
        window.on_show_bboxes_changed(True)
        self.assertNotIn(1, canvas.bbox_items_map)
        self.assertEqual(set(canvas.bbox_items_map), {2, 3, 4, 5})
        toolbar.brood_visibility_actions[STANDARD_BROOD_CATEGORIES[1]].trigger()
        selection = STANDARD_BROOD_CATEGORIES[:1] + STANDARD_BROOD_CATEGORIES[2:]
        window.on_space_toggle_annotation_overlays()
        window.on_space_toggle_annotation_overlays()
        self.assert_group(toolbar, 'brood', selection)
        self.assertEqual(set(canvas.bbox_items_map), {3, 4, 5})
        self.assertEqual(canvas.hidden_instance_keys, hidden)

    def test_active_edit_is_hidden_without_changing_pixels_selection_or_view(self):
        window = self.window()
        toolbar, canvas = window.toolbar, window.canvas
        toolbar.brood_visibility_checkbox.click()
        canvas.start_editing_instance(1, category=STANDARD_BROOD_CATEGORIES[0])
        original = canvas.editing_mask.copy()
        canvas.scale(2, 2)
        before = canvas.get_view_state()
        toolbar.brood_visibility_checkbox.click()
        self.assertFalse(canvas._is_editing_mask_visible())
        self.assertNotIn(1, canvas.bbox_items_map)
        self.assertEqual(canvas.editing_instance_id, 1)
        np.testing.assert_array_equal(canvas.editing_mask, original)
        toolbar.brood_visibility_checkbox.click()
        self.assertTrue(canvas._is_editing_mask_visible())
        self.assertIn(1, canvas.bbox_items_map)
        self.assertEqual(canvas.get_view_state(), before)
        np.testing.assert_array_equal(canvas.editing_mask, original)

    def test_new_instance_updates_group_summary_without_enabling_siblings(self):
        window = self.window()
        for name, categories in GROUPS.items():
            with self.subTest(group=name):
                window.new_instance(categories[0])
                self.assert_group(window.toolbar, name, categories[:1])
                for category in categories:
                    self.assertEqual(window.canvas.annotation_type_visibility[category], category == categories[0])


if __name__ == '__main__':
    unittest.main()
