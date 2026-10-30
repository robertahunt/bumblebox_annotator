"""Display-only isolation and exact positioning of cropped editing overlays."""
import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QMenu
from gui.canvas import ImageCanvas


class FocalBeeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_canvas(self, shape=(200, 300)):
        c = ImageCanvas()
        self.addCleanup(c.close)
        self.addCleanup(c._viz_update_timer.stop)
        c.load_image(np.zeros(shape, np.uint8))
        annotations = []
        for iid, category, x in [(1, 'bee', 30), (2, 'bee', 90), (3, 'hive', 150)]:
            mask = np.zeros(shape, np.uint8)
            mask[40:65, x:x + 25] = 255
            annotations.append(dict(mask_id=iid, category=category, mask=mask))
        annotations.append(dict(mask_id=4, category='bee', bbox_only=True, bbox=[210, 40, 20, 20]))
        c.set_annotations(annotations)
        c.start_editing_instance(1, 'bee')
        c.current_tool = 'brush'
        return c

    def test_bulk_visibility_keeps_pending_pixels_history_and_other_categories(self):
        c = self.make_canvas()
        c.set_instance_visible(3, 'hive', False)
        c._begin_brush_history_step()
        c.draw_on_mask(QPointF(40, 60), QPointF(80, 70))
        c._finish_brush_history_step()
        original = (c.bee_mask.copy(), c.hive_mask.copy(), c.editing_mask.copy(),
                    copy.deepcopy(c.annotation_metadata), len(c.brush_undo_stack))
        events = []
        c.annotation_changed.connect(lambda: events.append('changed'))
        with patch.object(c, 'rebuild_visualizations', wraps=c.rebuild_visualizations) as redraw:
            self.assertTrue(c.show_only_bee_mask(1))
            self.assertEqual(redraw.call_count, 1)
            self.assertEqual(c.hidden_instance_keys, {('hive', 3), ('bee', 2), ('bee', 4)})
            self.assertTrue(c.editing_mask_item.isVisible())
            c.show_all_bee_masks()
            self.assertEqual(redraw.call_count, 2)
        self.assertEqual(c.hidden_instance_keys, {('hive', 3)})
        self.assertEqual(events, [])
        for actual, expected in zip((c.bee_mask, c.hive_mask, c.editing_mask), original[:3]):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(c.annotation_metadata, original[3])
        self.assertEqual(len(c.brush_undo_stack), original[4])
        self.assertFalse(c.show_only_bee_mask(999))
        self.assertFalse(c.show_only_bee_mask(3))

    def test_actions_work_and_respect_parent_visibility(self):
        c = self.make_canvas()
        menu = QMenu()
        c.add_bee_visibility_actions(menu, 1, 'bee')
        menu.actions()[0].trigger()
        self.assertFalse(c.is_instance_enabled(2, 'bee'))
        menu2 = QMenu()
        c.add_bee_visibility_actions(menu2, 1, 'bee')
        self.assertTrue(menu2.actions()[1].isEnabled())
        menu2.actions()[1].trigger()
        self.assertTrue(c.is_instance_enabled(2, 'bee'))
        c.set_annotation_type_visibility('bee', False)
        menu3 = QMenu()
        c.add_bee_visibility_actions(menu3, 1, 'bee')
        self.assertFalse(menu3.actions()[0].isEnabled())
        self.assertFalse(c.annotation_type_visibility['bee'])

    def test_brush_menu_uses_editing_bee_instead_of_hovered_bee(self):
        c = self.make_canvas()
        class Event:
            def pos(self): return QPoint(0, 0)
            def globalPosition(self): return QPointF(0, 0)
        with patch.object(c, '_context_instance_at_scene_pos', return_value=(2, 'bee')), \
             patch.object(c, 'add_bee_visibility_actions', wraps=c.add_bee_visibility_actions) as actions, \
             patch('gui.canvas.QMenu.exec'):
            c._show_brush_context_menu(Event())
        self.assertEqual(actions.call_args.args[1:], (1, 'bee'))

    def test_isolation_resets_when_loading_another_frame(self):
        c = self.make_canvas()
        c.show_only_bee_mask(1)
        c.load_image(np.zeros((200, 300), np.uint8))
        self.assertEqual(c.hidden_instance_keys, set())

    def test_sidebar_sync_does_not_trigger_individual_rebuilds_or_drop_selection(self):
        from types import SimpleNamespace
        from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QLabel
        from gui.main_window import MainWindow
        c = self.make_canvas()
        sidebar = QListWidget()
        self.addCleanup(sidebar.close)
        for iid in (1, 2, 4):
            item = QListWidgetItem(f'Bee {iid}')
            item.setData(Qt.ItemDataRole.UserRole, {'id': iid, 'type': 'bee'})
            item.setCheckState(Qt.CheckState.Checked)
            sidebar.addItem(item)
        sidebar.setCurrentRow(0)
        changed = []
        sidebar.itemChanged.connect(lambda item: changed.append(item))
        window = SimpleNamespace(instance_list=sidebar, canvas=c, status_label=QLabel())
        c.bee_visibility_changed.connect(lambda iid: MainWindow.on_bee_visibility_changed(window, iid))
        c.show_only_bee_mask(1)
        self.assertEqual([sidebar.item(i).checkState() for i in range(3)],
                         [Qt.CheckState.Checked, Qt.CheckState.Unchecked, Qt.CheckState.Unchecked])
        self.assertEqual(changed, [])
        self.assertEqual(sidebar.currentRow(), 0)
        self.assertFalse(sidebar.signalsBlocked())
        c.show_all_bee_masks()
        self.assertTrue(all(sidebar.item(i).checkState() == Qt.CheckState.Checked for i in range(3)))

    def assert_overlay_matches_mask(self, c):
        item = c.editing_mask_item
        image = item.pixmap().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.bits().asstring(image.sizeInBytes()), np.uint8)
        pixels = pixels.reshape(image.height(), image.bytesPerLine())[:, :image.width()*4]
        alpha = pixels.reshape(image.height(), image.width(), 4)[:, :, 3]
        actual = np.zeros(c.editing_mask.shape, bool)
        x, y = int(item.offset().x()), int(item.offset().y())
        actual[y:y + image.height(), x:x + image.width()] = alpha > 0
        np.testing.assert_array_equal(actual, c.editing_mask > 0)

    def test_cropped_overlay_tracks_growth_erasure_undo_and_empty_mask(self):
        c = self.make_canvas()
        self.assert_overlay_matches_mask(c)
        c._begin_brush_history_step()
        c.draw_on_mask(QPointF(30, 40), QPointF(5, 10))
        c._finish_brush_history_step()
        c._viz_update_timer.stop()
        c._flush_pending_viz_update()
        self.assert_overlay_matches_mask(c)
        c.undo()
        self.assert_overlay_matches_mask(c)
        c.redo()
        self.assert_overlay_matches_mask(c)
        c.current_tool = 'eraser'
        c.brush_size = 180
        c.draw_on_mask(QPointF(30, 40), QPointF(30, 40))
        c._viz_update_timer.stop()
        c._flush_pending_viz_update()
        self.assert_overlay_matches_mask(c)
        self.assertEqual(c.editing_mask_item.pixmap().width(), 1)

    def test_small_bee_does_not_upload_full_resolution_transparent_margins(self):
        c = self.make_canvas((3040, 4056))
        c.draw_on_mask(QPointF(40, 45), QPointF(45, 50))
        c._viz_update_timer.stop()
        c._flush_pending_viz_update()
        p = c.editing_mask_item.pixmap()
        self.assertLess(p.width() * p.height(), 10000)
        self.assertEqual((int(c.editing_mask_item.offset().x()), int(c.editing_mask_item.offset().y())), (30, 40))
        self.assert_overlay_matches_mask(c)


if __name__ == '__main__':
    unittest.main()
