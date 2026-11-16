#!/usr/bin/env python3
"""The cursor magnifier: a close-up of the image under the cursor in the canvas corner."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRect, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from gui import magnifier
from gui.canvas import ImageCanvas

RED = (255, 0, 0)


class MagnifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_canvas(self):
        """A black 300x200 image with a red 16x16 square centred on (150, 100), shown 1:1."""
        canvas = ImageCanvas()
        canvas.resize(900, 700)
        canvas.show()
        image = np.zeros((200, 300, 3), np.uint8)
        image[92:108, 142:158] = RED
        canvas.load_image(image)
        self.app.processEvents()
        canvas.resetTransform()
        canvas.centerOn(150, 100)
        self.app.processEvents()
        self.addCleanup(canvas.close)
        return canvas

    def hover(self, canvas, x, y):
        self.app.processEvents()  # let a zoom settle before working out where the point is
        QTest.mouseMove(canvas.viewport(), canvas.mapFromScene(QPointF(x, y)))
        self.app.processEvents()

    def red_run(self, canvas, row):
        """How many red pixels the close-up has along a row clear of the cursor mark."""
        image = canvas.magnifier.grab().toImage()
        return sum(image.pixelColor(x, row).getRgb()[:3] == RED for x in range(image.width()))

    def test_follows_the_cursor_in_the_top_left_corner(self):
        canvas = self.make_canvas()
        frame = canvas.magnifier
        self.assertFalse(frame.isVisible())  # nothing to magnify until the cursor is on the canvas
        self.hover(canvas, 150, 100)
        self.assertTrue(frame.isVisibleTo(canvas))
        self.assertIs(frame.parent(), canvas.viewport())
        self.assertEqual(frame.geometry(), QRect(8, 8, 100, 100))
        # The frame is centred on the cursor: 100 px at 3x covers a third as much of the image.
        source = frame.source_rect()
        self.assertAlmostEqual(source.width(), 100 / 3)
        self.assertAlmostEqual(source.center().x(), canvas.mapToScene(frame.cursor_pos).x())
        self.assertAlmostEqual(source.center().y(), canvas.mapToScene(frame.cursor_pos).y())
        # The 16 px square is drawn 3x as wide.
        self.assertEqual(self.red_run(canvas, 30), 48)
        self.assertEqual(self.red_run(canvas, 10), 0)  # above the square

        canvas.leaveEvent(QEvent(QEvent.Type.Leave))
        self.assertFalse(frame.isVisible())

    def test_follows_the_view_zoom_and_shows_edits(self):
        canvas = self.make_canvas()
        self.hover(canvas, 150, 100)
        for _ in range(4):
            canvas.zoom_in()
        self.hover(canvas, 150, 100)
        scale = canvas.transform().m11()  # the close-up is 3x whatever the main view is
        self.assertGreater(scale, 1.4)
        self.assertAlmostEqual(canvas.magnifier.source_rect().width(), 100 / (3 * scale))
        self.assertAlmostEqual(self.red_run(canvas, 20), 16 * 3 * scale, delta=2)

        # Masks drawn on the canvas appear in the close-up without moving the mouse.
        mask = np.zeros((200, 300), np.uint8)
        mask[92:108, 142:158] = 1
        before = canvas.magnifier.grab().toImage().pixelColor(30, 20)
        canvas.set_annotations([{'mask': mask, 'mask_id': 1, 'category': 'bee'}])
        self.app.processEvents()
        self.assertNotEqual(canvas.magnifier.grab().toImage().pixelColor(30, 20), before)

    def test_size_zoom_and_switch(self):
        canvas = self.make_canvas()
        frame = canvas.magnifier
        self.hover(canvas, 150, 100)
        frame.set_frame_size(200)
        self.assertEqual(frame.geometry(), QRect(8, 8, 200, 200))
        frame.set_zoom(4)
        self.assertAlmostEqual(frame.source_rect().width(), 50)
        self.assertEqual(self.red_run(canvas, 70), 64)
        frame.set_active(False)
        self.assertFalse(frame.isVisible())
        self.hover(canvas, 160, 110)
        self.assertFalse(frame.isVisible())
        frame.set_active(True)
        self.assertTrue(frame.isVisibleTo(canvas))

    def test_steps_aside_under_the_cursor_and_lets_clicks_through(self):
        canvas = self.make_canvas()
        frame = canvas.magnifier
        self.assertTrue(frame.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))
        QTest.mouseMove(canvas.viewport(), QPoint(40, 40))
        self.app.processEvents()
        bottom = canvas.viewport().height() - 100 - 8
        self.assertEqual(frame.geometry(), QRect(8, bottom, 100, 100))
        # Still aside while the area being magnified would be under the frame.
        QTest.mouseMove(canvas.viewport(), QPoint(120, 60))
        self.assertEqual(frame.pos(), QPoint(8, bottom))
        QTest.mouseMove(canvas.viewport(), QPoint(200, 150))
        self.assertEqual(frame.pos(), QPoint(8, 8))

    def test_overlays_stay_put_when_the_view_scrolls(self):
        canvas = self.make_canvas()
        canvas.set_tool('brush')
        for _ in range(8):
            canvas.zoom_in()
        self.hover(canvas, 150, 100)
        buttons = canvas._brush_size_overlay
        before = canvas.magnifier.pos(), buttons.pos()
        bar = canvas.horizontalScrollBar()
        start = bar.value()
        bar.setValue(start + 40)
        canvas.verticalScrollBar().setValue(canvas.verticalScrollBar().value() + 25)
        self.app.processEvents()
        self.assertNotEqual(bar.value(), start)
        self.assertEqual((canvas.magnifier.pos(), buttons.pos()), before)

    def test_no_image_means_no_magnifier(self):
        canvas = self.make_canvas()
        self.hover(canvas, 150, 100)
        canvas.clear_image()
        self.assertFalse(canvas.magnifier.isVisible())
        QTest.mouseMove(canvas.viewport(), QPoint(200, 150))
        self.app.processEvents()
        self.assertFalse(canvas.magnifier.isVisible())


class MagnifierMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self, settings):
        from gui.main_window import MainWindow
        with patch('gui.main_window.QSettings', lambda *a, **k: settings), \
                patch.object(MainWindow, 'load_settings'):
            window = MainWindow()

        def close():
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                window.close()
            self.app.processEvents()
        self.addCleanup(close)
        return window

    def test_view_menu_sets_and_remembers_the_magnifier(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        settings = QSettings(str(Path(temp.name) / 'settings.ini'), QSettings.Format.IniFormat)
        mock = patch('gui.main_window.QSettings', lambda *a, **k: settings)
        mock.start()
        self.addCleanup(mock.stop)

        window = self.make_window(settings)
        frame = window.canvas.magnifier
        self.assertTrue(window.magnifier_action.isChecked())
        self.assertTrue(frame.active)
        self.assertEqual((frame.width(), frame.zoom), (magnifier.DEFAULT_SIZE, magnifier.DEFAULT_ZOOM))
        self.assertTrue(window.magnifier_size_actions[100].isChecked())
        self.assertTrue(window.magnifier_zoom_actions[3].isChecked())

        window.magnifier_size_actions[200].trigger()
        window.magnifier_zoom_actions[6].trigger()
        window.magnifier_action.trigger()
        self.assertEqual((frame.width(), frame.height(), frame.zoom, frame.active), (200, 200, 6, False))
        self.assertFalse(window.magnifier_size_actions[100].isChecked())

        # A new window picks the choices up again.
        again = self.make_window(settings)
        again._load_magnifier_settings(settings)
        frame = again.canvas.magnifier
        self.assertEqual((frame.width(), frame.zoom, frame.active), (200, 6, False))
        self.assertFalse(again.magnifier_action.isChecked())
        self.assertTrue(again.magnifier_size_actions[200].isChecked())
        self.assertTrue(again.magnifier_zoom_actions[6].isChecked())


if __name__ == '__main__':
    unittest.main()
