"""Avoid redundant editing work without stale geometry, IDs, or sidebar state."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox
from core.annotation import AnnotationManager
from core.mask_geometry import MaskGeometryCache, measure_mask
from gui.canvas import ImageCanvas


def annotations():
    records = []
    for iid, x, y in [(1, 5, 5), (2, 25, 5), (3, 45, 5), (4, 65, 35)]:
        mask = np.zeros((64, 96), np.uint8)
        mask[y:y + 15, x:x + 15] = 255
        records.append(dict(mask_id=iid, category='bee', mask=mask))
    return records


class TargetedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def canvas(self):
        c = ImageCanvas()
        self.addCleanup(c.close)
        c.load_image(np.zeros((64, 96), np.uint8))
        c.set_annotations(annotations())
        return c

    def test_switch_keeps_other_bee_measurements_and_commit_invalidates_overlaps(self):
        c = self.canvas()
        before = {i: c.get_instance_geometry(i, 'bee') for i in range(1, 5)}
        c.start_editing_instance(1, 'bee')
        for i in (2, 3, 4):
            self.assertIs(c.get_instance_geometry(i, 'bee'), before[i])
        # Partial overlap with bee 2; complete overwrite of bee 3.
        c.editing_mask[5:20, 30:60] = 255
        expected = c.bee_mask.copy()
        expected[c.editing_mask > 0] = 1
        c.commit_editing()
        np.testing.assert_array_equal(c.bee_mask, expected)
        for i in range(1, 5):
            self.assertEqual(c.get_instance_geometry(i, 'bee'), measure_mask(expected == i))
        self.assertIs(c.get_instance_geometry(4, 'bee'), before[4])
        self.assertNotIn(3, c._stored_instance_ids('bee', c.bee_mask))
        self.assertNotEqual(c.get_instance_geometry(2, 'bee').area, before[2].area)

    def test_add_and_delete_preserve_unchanged_bee_cache(self):
        c = self.canvas()
        unchanged = c.get_instance_geometry(4, 'bee')
        c.get_instance_geometry(2, 'bee')
        added = np.zeros((64, 96), np.uint8)
        added[10:15, 28:33] = 255
        c.add_mask(added, mask_id=8, category='bee')
        self.assertIs(c.get_instance_geometry(4, 'bee'), unchanged)
        self.assertEqual(c.get_instance_geometry(2, 'bee').area, 200)
        c.delete_instance(8, category='bee')
        self.assertIs(c.get_instance_geometry(4, 'bee'), unchanged)
        self.assertNotIn(8, c._stored_instance_ids('bee', c.bee_mask))

    def test_partial_invalidation_still_detects_array_replacement(self):
        cache = MaskGeometryCache()
        array = np.ones((10, 10), np.int32)
        cache.geometry('bee', array, 1)
        cache.invalidate_instances('bee', (2,))
        self.assertEqual(cache.geometry('bee', np.zeros_like(array), 1).area, 0)

    def test_reclassify_and_rename_keep_unaffected_geometry(self):
        c = self.canvas()
        unchanged = c.get_instance_geometry(4, 'bee')
        c.change_instance_category(1, 'hive', old_category='bee')
        self.assertIs(c.get_instance_geometry(4, 'bee'), unchanged)
        c.change_instance_category(1, 'bee', old_category='hive')
        self.assertIs(c.get_instance_geometry(4, 'bee'), unchanged)
        self.assertTrue(c.reassign_instance_id(1, 9))
        self.assertIs(c.get_instance_geometry(4, 'bee'), unchanged)
        self.assertEqual(c.get_instance_geometry(9, 'bee').area, 225)


class TagMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / 'annotations/json/video/video_annotations.json'
        self.file.parent.mkdir(parents=True)
        self.manager = AnnotationManager()

    def write(self, mapping):
        self.file.write_text(json.dumps({'annotations': [{'category': 'hive', 'mask_id': 99}],
                                         'aruco_tracking': mapping}))

    def test_metadata_read_never_loads_masks_and_reuses_unchanged_file(self):
        self.write({'23': 1})
        with patch('cv2.imread', side_effect=AssertionError('Loaded a mask')):
            result = self.manager.load_aruco_tracking(self.root, 'video')
            self.assertEqual(result, {'23': 1})
            result['23'] = 999
            with patch('json.load', side_effect=AssertionError('Reread unchanged JSON')):
                self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {'23': 1})

    def test_updates_replacement_removal_and_project_change_invalidate_mapping(self):
        self.write({'23': 1})
        self.manager.load_aruco_tracking(self.root, 'video')
        self.manager.update_aruco_tracking(self.root, 'video', 23, 2)
        self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {'23': 2})
        self.manager.remove_aruco_tracking(self.root, 'video', 23)
        self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {})
        replacement = self.file.with_suffix('.replacement')
        replacement.write_text(json.dumps({'aruco_tracking': {'33': 4}}))
        replacement.replace(self.file)
        self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {'33': 4})
        self.assertEqual(self.manager.load_aruco_tracking(self.root / 'other', 'video'), {})
        self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {'33': 4})
        self.file.unlink()
        self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {})
        self.write({'44': 5})
        self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {'44': 5})

    def test_legacy_and_incomplete_json_recover(self):
        for data in ('[]', '{', '{"aruco_tracking": null}'):
            self.file.write_text(data)
            self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {})
        self.write({'23': 1})
        self.assertEqual(self.manager.load_aruco_tracking(self.root, 'video'), {'23': 1})


class SidebarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def window(self):
        from gui.main_window import MainWindow
        with patch.object(MainWindow, 'load_settings'):
            w = MainWindow()

        def cleanup():
            with patch('gui.main_window.QSettings'), patch.object(
                    QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                w.close()
            self.app.processEvents()
        self.addCleanup(cleanup)
        w.canvas.load_image(np.zeros((64, 96), np.uint8))
        w.canvas.set_annotations(annotations())
        w.update_instance_list_from_canvas()
        return w

    def test_edit_only_changes_affected_row_and_preserves_items(self):
        w = self.window()
        items = [w.instance_list.item(i) for i in range(4)]
        w.canvas.start_editing_instance(1, 'bee')
        w.update_instance_list_from_canvas()
        changes = []
        w.instance_list.model().dataChanged.connect(
            lambda first, last, roles: changes.append(first.row()))
        w.canvas.editing_mask[0:2, 0:2] = 255
        with patch('core.mask_geometry.measure_mask', side_effect=AssertionError('Scanned unchanged mask')):
            w.update_instance_list_from_canvas()
        self.assertEqual(set(changes), {0})
        for i, item in enumerate(items):
            self.assertIs(w.instance_list.item(i), item)
        self.assertIn('area: 229', items[0].text())
        changes.clear()
        w.update_instance_list_from_canvas()
        self.assertEqual(changes, [])

    def test_optional_refresh_waits_until_stroke_finishes(self):
        w = self.window()
        w.canvas.is_drawing = True
        with patch.object(w, 'update_instance_list_from_canvas', wraps=w.update_instance_list_from_canvas) as refresh:
            w._do_update_instance_list()
            refresh.assert_not_called()
            self.assertTrue(w._instance_list_update_timer.isActive())
            w.canvas.is_drawing = False
            w._do_update_instance_list()
            refresh.assert_called_once()
        self.assertFalse(w._instance_list_update_timer.isActive())

    def test_optional_counter_waits_until_stroke_finishes(self):
        w = self.window()
        w.canvas.is_drawing = True
        with patch.object(w, '_current_annotation_counts', wraps=w._current_annotation_counts) as counts:
            w._refresh_done_counter_labels()
            counts.assert_not_called()
            self.assertTrue(w.done_counter_update_timer.isActive())
            w.canvas.is_drawing = False
            w._refresh_done_counter_labels()
            counts.assert_called_once()

    def test_rows_survive_insert_delete_and_multiple_selection(self):
        w = self.window()
        first, last = w.instance_list.item(0), w.instance_list.item(3)
        first.setSelected(True)
        last.setSelected(True)
        w.canvas.delete_instance(2, 'bee')
        w.update_instance_list_from_canvas()
        self.assertIs(w.instance_list.item(0), first)
        self.assertIs(w.instance_list.item(2), last)
        self.assertTrue(first.isSelected() and last.isSelected())
        mask = np.zeros((64, 96), np.uint8)
        mask[40:45, 5:10] = 255
        w.canvas.add_mask(mask, mask_id=2, category='bee')
        w.update_instance_list_from_canvas()
        self.assertIs(w.instance_list.item(3), last)
        self.assertTrue(first.isSelected() and last.isSelected())
        self.assertEqual([w.instance_list.item(i).data(Qt.ItemDataRole.UserRole)['id']
                          for i in range(4)], [1, 2, 3, 4])

    def test_tag_row_updates_without_replacing_items_or_loading_masks(self):
        w = self.window()
        old = w.instance_list.item(1)
        with patch.object(w, '_get_instance_to_aruco_map', return_value={2: 23}), \
                patch.object(w.annotation_manager, 'load_video_annotations', side_effect=AssertionError('Loaded masks')):
            w.update_instance_list_from_canvas()
        self.assertIs(w.instance_list.item(1), old)
        self.assertIn('[ArUco: 23]', old.text())

    def test_new_frame_does_not_inherit_previous_multiselection(self):
        w = self.window()
        w.instance_list.item(0).setSelected(True)
        w.instance_list.item(2).setSelected(True)
        w.canvas.load_image(np.zeros((64, 96), np.uint8))
        w.canvas.set_annotations(annotations())
        w.update_instance_list_from_canvas()
        self.assertEqual(w.instance_list.selectedItems(), [])
        self.assertEqual(w.instance_list.currentRow(), -1)


if __name__ == '__main__':
    unittest.main()
