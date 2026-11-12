#!/usr/bin/env python3
"""Editing toolbar layout: options follow the active tool, occasional actions in More."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt6.QtWidgets import QApplication

from gui.canvas import ImageCanvas
from gui.toolbar import AnnotationToolbar


class ToolOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.toolbar = AnnotationToolbar()
        self.toolbar.resize(1400, 200)
        self.toolbar.show()
        self.app.processEvents()
        self.addCleanup(self.toolbar.close)

    def shown(self, widget):
        return widget.isVisibleTo(self.toolbar)

    def test_options_row_shows_only_the_active_tools_controls(self):
        t = self.toolbar
        brush = [t.brush_size_slider, t.brush_cursor_checkbox]
        self.assertFalse(any(self.shown(w) for w in brush + [t.clear_no_draw_zones_btn]))
        self.assertFalse(any(self.shown(w) for w in t.calibration_widgets))
        self.assertIn('Drag to move', t.tool_hint_label.text())

        for tool in ('brush', 'eraser'):
            t.set_tool(tool)
            self.assertTrue(all(self.shown(w) for w in brush))
            self.assertFalse(self.shown(t.clear_no_draw_zones_btn))
        t.set_tool('no_draw_zone')
        self.assertTrue(self.shown(t.clear_no_draw_zones_btn))
        self.assertFalse(any(self.shown(w) for w in brush))
        t.set_tool('measure')
        self.assertTrue(all(self.shown(w) for w in t.calibration_widgets))
        self.assertFalse(self.shown(t.tool_hint_label))

        # Display settings stay visible for every tool.
        for tool in ('pan', 'brush', 'measure'):
            t.set_tool(tool)
            self.assertTrue(self.shown(t.opacity_slider))
            self.assertTrue(self.shown(t.distinct_bee_colors_checkbox))

    def test_tools_chosen_elsewhere_update_the_options_row(self):
        t = self.toolbar
        t.show_tool_options('sam2_prompt')
        self.assertIn('positive point', t.tool_hint_label.text())
        self.assertFalse(self.shown(t.brush_size_slider))

    def test_calibration_hidden_when_not_allowed(self):
        t = self.toolbar
        t.set_calibration_allowed(False)
        t.set_tool('measure')
        self.assertFalse(any(self.shown(w) for w in t.calibration_widgets))
        self.assertIn('Drag a line', t.tool_hint_label.text())
        t.set_calibration_allowed(True)
        self.assertTrue(all(self.shown(w) for w in t.calibration_widgets))

    def test_more_menu_holds_occasional_and_bulk_actions(self):
        t = self.toolbar
        texts = [a.text() for a in t.more_menu.actions() if not a.isSeparator()]
        self.assertEqual(texts[-1], 'Delete All Bee Instances on This Frame...')
        self.assertIn('Detect ArUco Tags on Bees', texts)
        # Bulk delete is separated from everything else in the menu.
        actions = t.more_menu.actions()
        self.assertTrue(actions[actions.index(t.delete_all_action) - 1].isSeparator())
        emitted = []
        t.delete_all_requested.connect(lambda: emitted.append('delete'))
        t.clear_all_aruco_requested.connect(lambda: emitted.append('clear_aruco'))
        t.tool_changed.connect(emitted.append)
        t.delete_all_action.trigger()
        t.clear_aruco_action.trigger()
        t.calibrate_action.trigger()
        self.assertEqual(emitted, ['delete', 'clear_aruco', 'measure'])
        self.assertTrue(t.measure_btn.isChecked())
        # None of these sit in the tool or options rows any more.
        self.assertFalse(hasattr(t, 'delete_all_btn'))

    def test_keypoint_tool_button_only_when_available(self):
        t = self.toolbar
        self.assertFalse(self.shown(t.keypoint_btn))
        t.set_keypoints_available(True)
        self.assertTrue(self.shown(t.keypoint_btn))
        self.assertTrue(t.check_tool_button('keypoint'))
        self.assertTrue(t.keypoint_btn.isChecked())


class CanvasOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_brush_size_buttons_only_for_brush_and_eraser(self):
        canvas = ImageCanvas()
        canvas.resize(400, 300)
        canvas.show()
        canvas.load_image(np.zeros((100, 100, 3), np.uint8))
        self.app.processEvents()
        overlay = canvas._brush_size_overlay
        self.assertFalse(overlay.isVisibleTo(canvas))
        for tool, visible in (('brush', True), ('eraser', True), ('pan', False),
                              ('measure', False), ('keypoint', False), ('brush', True)):
            canvas.set_tool(tool)
            self.assertEqual(overlay.isVisibleTo(canvas), visible, tool)
        canvas.close()


class DoneCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_counter_is_one_line_until_clicked(self):
        import tempfile
        from unittest.mock import patch
        from PyQt6.QtCore import QPoint, QSettings, Qt
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QMessageBox
        from gui.main_window import MainWindow
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        settings = QSettings(str(Path(temp.name) / 'settings.ini'), QSettings.Format.IniFormat)
        mock = patch('gui.main_window.QSettings', lambda *a, **k: settings)
        mock.start()
        self.addCleanup(mock.stop)
        with patch.object(MainWindow, 'load_settings'):
            window = MainWindow()

        def close():
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                window.close()
            self.app.processEvents()
        self.addCleanup(close)
        window.show()
        window.done_counter_saved_counts_dirty = False
        window._refresh_done_counter_labels()
        self.app.processEvents()
        overlay = window.done_counter_overlay
        visible_labels = lambda: [l for l in window.done_counter_category_labels.values()
                                  if l.isVisibleTo(overlay)]
        self.assertFalse(window.done_counter_title_label.isVisibleTo(overlay))
        self.assertEqual(visible_labels(), [])
        self.assertIn('Instances made', window.done_counter_summary_label.text())
        collapsed_height = overlay.height()

        QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
        self.app.processEvents()
        self.assertTrue(window.done_counter_expanded)
        self.assertTrue(window.done_counter_title_label.isVisibleTo(overlay))
        self.assertGreater(len(visible_labels()), 0)
        self.assertGreater(overlay.height(), collapsed_height)

        QTest.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
        self.app.processEvents()
        self.assertFalse(window.done_counter_expanded)
        self.assertEqual(visible_labels(), [])

        # One header row: Model assistance and the policy summary sit in the status row,
        # left of the save/shared status; undo/redo lead the tool row.
        flow = window.workflow
        controls = flow.editor_controls.layout()
        self.assertGreaterEqual(controls.indexOf(flow.assist_toggle), 0)
        self.assertGreaterEqual(controls.indexOf(window.policy_label), 0)
        self.assertEqual(flow.status_layout.indexOf(flow.editor_controls), 0)
        self.assertLess(flow.status_layout.indexOf(flow.editor_controls),
                        flow.status_layout.indexOf(flow.local_status))
        self.assertEqual(flow.editor.layout().indexOf(window.policy_label), -1)
        first_tools = [window.toolbar.tools_row.itemAt(i).widget() for i in range(2)]
        self.assertEqual([w.accessibleName() for w in first_tools], ['Undo', 'Redo'])


class StatusRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_shared_status_wording(self):
        from gui.workspaces import shared_status_text
        self.assertEqual(shared_status_text({'status': 'current', 'message': 'No newer shared version'}),
                         'Shared copy: up to date')
        self.assertEqual(shared_status_text({'status': 'update', 'changed': False, 'message': ''}),
                         'Shared copy is newer')
        both = shared_status_text({'status': 'update', 'changed': True, 'message':
                                   'Local annotation edits and a newer shared version; review before updating'})
        self.assertEqual(both, 'Shared copy is newer; review before updating')
        # Specific attention messages pass through unchanged.
        msg = 'Choose the shared folder used for this assignment'
        self.assertEqual(shared_status_text({'status': 'attention', 'message': msg}), msg)

    def test_status_labels_stay_on_one_line(self):
        from gui.workspaces import ElidedLabel
        label = ElidedLabel('Shared copy: newer version; you have edits here, so review before updating')
        label.resize(120, 20)
        label.show()
        self.app.processEvents()
        self.assertFalse(label.wordWrap())
        self.assertLessEqual(label.minimumSizeHint().width(), 60)
        self.assertIn('review before updating', label.text())  # full text kept
        label.close()


if __name__ == "__main__":
    unittest.main()
