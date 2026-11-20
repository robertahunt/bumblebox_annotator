#!/usr/bin/env python3
"""The cursor views: a close-up and a zoomed-out view of the image around the cursor, in the canvas corner."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSettings, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGraphicsItem, QGraphicsRectItem, QMessageBox

from gui import magnifier
from gui.canvas import ImageCanvas

RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


class CanvasCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_canvas(self, width=900, height=700):
        """A black 300x200 image with a red 16x16 square centred on (150, 100), shown 1:1."""
        canvas = ImageCanvas()
        canvas.resize(width, height)
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

    def move_to(self, canvas, point):
        """Move the mouse to a place in the viewport.

        Qt drops a move to where the mouse already is, even when a new window
        is there now, so the mouse comes from the pixel next to it.
        """
        QTest.mouseMove(canvas.viewport(), point + QPoint(1, 1))
        QTest.mouseMove(canvas.viewport(), point)
        self.app.processEvents()

    def hover(self, canvas, x, y):
        self.app.processEvents()  # let a zoom settle before working out where the point is
        self.move_to(canvas, canvas.mapFromScene(QPointF(x, y)))


class MagnifierTests(CanvasCase):
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
        canvas.overview.set_active(False)  # the close-up on its own
        frame = canvas.magnifier
        self.assertTrue(frame.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))
        self.assertTrue(canvas.overview.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))
        self.move_to(canvas, QPoint(40, 40))
        self.app.processEvents()
        bottom = canvas.viewport().height() - 100 - 8
        self.assertEqual(frame.geometry(), QRect(8, bottom, 100, 100))
        # Still aside while the area being magnified would be under the frame.
        self.move_to(canvas, QPoint(120, 60))
        self.assertEqual(frame.pos(), QPoint(8, bottom))
        self.move_to(canvas, QPoint(200, 150))
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
        self.assertFalse(canvas.overview.isVisible())
        self.move_to(canvas, QPoint(200, 150))
        self.app.processEvents()
        self.assertFalse(canvas.magnifier.isVisible())
        self.assertFalse(canvas.overview.isVisible())

    def test_the_largest_frame_is_600_pixels(self):
        self.assertEqual(max(magnifier.SIZES), 600)
        canvas = self.make_canvas()
        self.hover(canvas, 150, 100)
        canvas.magnifier.set_frame_size(600)
        self.assertEqual(canvas.magnifier.size().width(), 600)
        self.assertAlmostEqual(canvas.magnifier.source_rect().width(), 200)

    def test_half_height_frames_fill_the_left_side_and_follow_the_window(self):
        canvas = self.make_canvas()
        close_up, wide = canvas.magnifier, canvas.overview
        close_up.set_frame_size(magnifier.HALF_HEIGHT)
        wide.set_frame_size(magnifier.HALF_HEIGHT)
        self.move_to(canvas, QPoint(700, 300))
        height = canvas.viewport().height()
        side = (height - 24) // 2
        self.assertEqual(canvas.cursor_views.half_height_side(), side)
        # Two equal squares, one above the other, with the usual 8 px around and between them.
        self.assertEqual(close_up.geometry(), QRect(8, 8, side, side))
        self.assertEqual(wide.geometry(), QRect(8, 16 + side, side, side))
        self.assertLessEqual(height - wide.geometry().bottom() - 1 - 8, 1)

        canvas.resize(900, 500)
        self.app.processEvents()
        self.move_to(canvas, QPoint(700, 300))
        smaller = (canvas.viewport().height() - 24) // 2
        self.assertLess(smaller, side)
        self.assertEqual(close_up.geometry(), QRect(8, 8, smaller, smaller))
        self.assertEqual(wide.geometry(), QRect(8, 16 + smaller, smaller, smaller))

        # With the cursor under them they cross to the right side, still stacked.
        self.move_to(canvas, QPoint(60, 250))
        right = canvas.viewport().width() - smaller - 8
        self.assertEqual((close_up.pos(), wide.pos()), (QPoint(right, 8), QPoint(right, 16 + smaller)))
        # A fixed size again.
        wide.set_frame_size(150)
        self.assertEqual(wide.size().width(), 150)


class ZoomedOutViewTests(CanvasCase):
    """The second frame: the surroundings of the cursor, smaller than the main view shows them."""

    def zoom_in_until(self, canvas, scale):
        for _ in range(80):
            if canvas.transform().m11() >= scale:
                break
            canvas.zoom_in()
        self.assertGreaterEqual(canvas.transform().m11(), scale)

    def pixels(self, frame, colour):
        image = frame.grab().toImage()
        return [(x, y) for y in range(image.height()) for x in range(image.width())
                if image.pixelColor(x, y).getRgb()[:3] == colour]

    def add_square(self, canvas, x, y, side, fixed_size):
        item = QGraphicsRectItem(0, 0, side, side)
        item.setPos(x, y)
        item.setBrush(QBrush(QColor(*GREEN)))
        item.setPen(QPen(Qt.PenStyle.NoPen))
        item.setZValue(50)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, fixed_size)
        canvas.scene.addItem(item)
        self.app.processEvents()
        return item

    def test_sits_below_the_close_up_and_follows_the_cursor(self):
        canvas = self.make_canvas()
        frame = canvas.overview
        self.assertFalse(frame.isVisible())
        self.hover(canvas, 150, 100)
        self.assertTrue(frame.isVisibleTo(canvas))
        self.assertIs(frame.parent(), canvas.viewport())
        self.assertEqual(canvas.magnifier.geometry(), QRect(8, 8, 100, 100))
        self.assertEqual(frame.geometry(), QRect(8, 116, 200, 200))
        canvas.magnifier.set_frame_size(300)
        self.assertEqual(frame.geometry(), QRect(8, 316, 200, 200))
        # Without the close-up it takes the corner itself.
        canvas.magnifier.set_active(False)
        self.assertEqual(frame.geometry(), QRect(8, 8, 200, 200))
        frame.set_active(False)
        self.assertFalse(frame.isVisible())
        frame.set_active(True)
        canvas.leaveEvent(QEvent(QEvent.Type.Leave))
        self.assertFalse(frame.isVisible())

    def test_reaches_several_times_as_far_as_the_main_view(self):
        canvas = self.make_canvas()
        frame = canvas.overview
        self.hover(canvas, 150, 100)
        # The main view already shows the whole 300x200 image: there is no more to show.
        whole = QRectF(0, -50, 300, 300)
        self.assertEqual(frame.source_rect(), whole)

        self.zoom_in_until(canvas, 8)
        canvas.centerOn(150, 100)
        self.hover(canvas, 150, 100)
        seen = frame.main_view_rect()
        self.assertLess(seen.width(), 120)
        frame.set_reach(2)
        source = frame.source_rect()
        self.assertAlmostEqual(source.width(), 2 * seen.width())
        self.assertAlmostEqual(source.height(), source.width())
        self.assertAlmostEqual(source.center().x(), canvas.mapToScene(frame.cursor_pos).x())
        # Taller than the image: all of its height is shown, centred.
        self.assertGreater(source.height(), 200)
        self.assertAlmostEqual(source.center().y(), 100)
        # It does not look past the edge of the image while there is more image to show.
        canvas.centerOn(10, 100)
        self.hover(canvas, 4, 100)
        self.assertAlmostEqual(canvas.mapToScene(frame.cursor_pos).x(), 4, delta=0.5)
        self.assertAlmostEqual(frame.source_rect().left(), 0)
        self.assertAlmostEqual(frame.source_rect().width(), 2 * frame.main_view_rect().width())

        frame.set_reach(1.05)  # only a little more than the main view shows
        self.assertAlmostEqual(frame.source_rect().width(), 1.05 * frame.main_view_rect().width())
        self.assertNotEqual(frame.source_rect(), whole)
        frame.set_reach(magnifier.WHOLE_FRAME)
        self.assertEqual(frame.source_rect(), whole)

    def test_draws_the_scene_small_and_leaves_out_what_keeps_its_screen_size(self):
        canvas = self.make_canvas()
        frame = canvas.overview
        self.hover(canvas, 250, 180)  # clear of everything looked at below
        # The whole 300x200 image in 200 px: two thirds of the size, with a band above and below.
        red = self.pixels(frame, RED)
        self.assertAlmostEqual(len(red), (16 * 2 / 3) ** 2, delta=30)
        self.assertAlmostEqual(sum(x for x, _ in red) / len(red), 150 * 2 / 3, delta=1)
        self.assertAlmostEqual(sum(y for _, y in red) / len(red), (100 + 50) * 2 / 3, delta=1)
        image = frame.grab().toImage()
        self.assertEqual(image.pixelColor(60, 100).getRgb()[:3], (0, 0, 0))
        self.assertEqual(image.pixelColor(100, 15), canvas.backgroundBrush().color())

        # Something that shrinks with the view is drawn; number labels and keypoint
        # markers, which keep their size on screen, would bury a small picture.
        self.add_square(canvas, 30, 30, 30, fixed_size=False)
        green = self.pixels(frame, GREEN)
        self.assertAlmostEqual(len(green), 20 * 20, delta=45)
        self.assertAlmostEqual(sum(x for x, _ in green) / len(green), 45 * 2 / 3, delta=1)
        marker = self.add_square(canvas, 200, 40, 30, fixed_size=True)
        self.assertEqual(len(self.pixels(frame, GREEN)), len(green))
        # The close-up still shows them.
        self.hover(canvas, 205, 45)
        self.assertGreater(len(self.pixels(canvas.magnifier, GREEN)), 500)
        marker.setVisible(False)
        self.app.processEvents()
        self.assertEqual(self.pixels(canvas.magnifier, GREEN), [])

    def test_outlines_the_main_view_and_marks_the_cursor(self):
        canvas = self.make_canvas()
        frame = canvas.overview
        frame.set_reach(2)
        self.zoom_in_until(canvas, 8)
        canvas.centerOn(150, 100)
        self.hover(canvas, 150, 100)
        outline = frame.to_frame(frame.source_rect()).mapRect(frame.main_view_rect())
        self.assertAlmostEqual(outline.width(), 100, delta=1)   # half as far across as the frame reaches
        white = self.pixels(frame, WHITE)
        self.assertGreater(len(white), 2 * (outline.width() + outline.height()) * 0.9)
        xs, ys = [x for x, _ in white], [y for _, y in white]
        self.assertAlmostEqual(min(xs), outline.left(), delta=1.5)
        self.assertAlmostEqual(max(xs), outline.right(), delta=1.5)
        self.assertAlmostEqual(min(ys), outline.top(), delta=1.5)
        self.assertAlmostEqual(max(ys), outline.bottom(), delta=1.5)
        # The cursor mark: ticks either side of where the cursor is.
        cursor = frame.to_frame(frame.source_rect()).map(canvas.mapToScene(frame.cursor_pos))
        near = [(x, y) for x, y in white if abs(x - cursor.x()) <= 10 and abs(y - cursor.y()) <= 10]
        self.assertGreaterEqual(len(near), 20)

    def test_both_frames_step_aside_together(self):
        canvas = self.make_canvas()
        close_up, wide = canvas.magnifier, canvas.overview
        height = canvas.viewport().height()
        self.move_to(canvas, QPoint(40, 40))  # under the close-up
        self.app.processEvents()
        self.assertEqual(close_up.geometry(), QRect(8, height - 316, 100, 100))
        self.assertEqual(wide.geometry(), QRect(8, height - 208, 200, 200))
        self.move_to(canvas, QPoint(150, 200))  # under the zoomed-out view's place
        self.assertEqual(wide.pos(), QPoint(8, height - 208))
        self.move_to(canvas, QPoint(500, 300))
        self.assertEqual((close_up.pos(), wide.pos()), (QPoint(8, 8), QPoint(8, 116)))
        self.move_to(canvas, QPoint(60, height - 60))  # the bottom-left is free again
        self.assertEqual((close_up.pos(), wide.pos()), (QPoint(8, 8), QPoint(8, 116)))

        # A column filling the left side has nowhere to go down there: it crosses to the right.
        close_up.set_frame_size(300)
        wide.set_frame_size(300)
        self.move_to(canvas, QPoint(100, 350))
        right = canvas.viewport().width() - 308
        self.assertEqual((close_up.pos(), wide.pos()), (QPoint(right, 8), QPoint(right, 316)))

    def test_side_by_side_where_the_canvas_is_too_short_to_stack_them(self):
        canvas = self.make_canvas(1400, 700)
        canvas.magnifier.set_frame_size(300)
        canvas.overview.set_frame_size(600)
        self.move_to(canvas, QPoint(1200, 650))
        self.app.processEvents()
        self.assertEqual(canvas.magnifier.geometry(), QRect(8, 8, 300, 300))
        self.assertEqual(canvas.overview.geometry(), QRect(316, 8, 600, 600))


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

    def test_view_menu_sets_and_remembers_the_zoomed_out_view(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        settings = QSettings(str(Path(temp.name) / 'settings.ini'), QSettings.Format.IniFormat)
        mock = patch('gui.main_window.QSettings', lambda *a, **k: settings)
        mock.start()
        self.addCleanup(mock.stop)

        window = self.make_window(settings)
        frame = window.canvas.overview
        self.assertTrue(window.overview_action.isChecked())
        self.assertEqual((frame.width(), frame.reach, frame.active),
                         (magnifier.DEFAULT_OVERVIEW_SIZE, magnifier.DEFAULT_REACH, True))
        self.assertTrue(window.overview_size_actions[200].isChecked())
        self.assertTrue(window.overview_reach_actions[2].isChecked())
        self.assertEqual(window.overview_reach_actions[2].text(), '2\u00d7 the main view')
        self.assertEqual(window.overview_reach_actions[1.05].text(), '1.05\u00d7 the main view')
        self.assertEqual(window.overview_reach_actions[1.1].text(), '1.1\u00d7 the main view')
        self.assertEqual(list(window.overview_reach_actions),
                         [1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75, 2, magnifier.WHOLE_FRAME])
        self.assertEqual(window.overview_reach_actions[magnifier.WHOLE_FRAME].text(), 'Whole frame')
        self.assertIn(600, window.magnifier_size_actions)

        window.overview_reach_actions[1.5].trigger()
        self.assertEqual(frame.reach, 1.5)
        self.assertEqual(settings.value('display/overview_reach', type=float), 1.5)
        window.overview_size_actions[600].trigger()
        window.overview_reach_actions[magnifier.WHOLE_FRAME].trigger()
        window.overview_action.trigger()
        self.assertEqual((frame.width(), frame.height(), frame.reach, frame.active),
                         (600, 600, magnifier.WHOLE_FRAME, False))
        # The close-up keeps its own choices.
        self.assertEqual((window.canvas.magnifier.width(), window.canvas.magnifier.active), (100, True))

        again = self.make_window(settings)
        again._load_magnifier_settings(settings)
        frame = again.canvas.overview
        self.assertEqual((frame.width(), frame.reach, frame.active), (600, magnifier.WHOLE_FRAME, False))
        self.assertFalse(again.overview_action.isChecked())
        self.assertTrue(again.overview_size_actions[600].isChecked())
        self.assertTrue(again.overview_reach_actions[magnifier.WHOLE_FRAME].isChecked())
        self.assertTrue(again.magnifier_action.isChecked())

        # A remembered reach that is no longer offered (3x and up were removed) gives the default.
        settings.setValue('display/overview_reach', 4.0)
        settings.setValue('display/overview', True)
        third = self.make_window(settings)
        third._load_magnifier_settings(settings)
        self.assertEqual(third.canvas.overview.reach, magnifier.DEFAULT_REACH)
        self.assertTrue(third.overview_reach_actions[magnifier.DEFAULT_REACH].isChecked())
        settings.setValue('display/overview_reach', 1.15)
        third._load_magnifier_settings(settings)
        self.assertEqual(third.canvas.overview.reach, 1.15)
        self.assertTrue(third.overview_reach_actions[1.15].isChecked())

    def test_half_height_is_a_size_choice_named_with_its_pixels(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        settings = QSettings(str(Path(temp.name) / 'settings.ini'), QSettings.Format.IniFormat)
        mock = patch('gui.main_window.QSettings', lambda *a, **k: settings)
        mock.start()
        self.addCleanup(mock.stop)

        window = self.make_window(settings)
        views = window.canvas.cursor_views
        side = views.half_height_side()
        for actions in (window.magnifier_size_actions, window.overview_size_actions):
            actions[magnifier.HALF_HEIGHT].trigger()
        self.assertEqual((views.magnifier.width(), views.overview.width()), (side, side))
        # The menu names the size as it is when the menu opens.
        window._label_half_height_sizes()
        for actions in (window.magnifier_size_actions, window.overview_size_actions):
            self.assertEqual(actions[magnifier.HALF_HEIGHT].text(), f'Half the canvas height ({side} px now)')
            self.assertEqual(actions[300].text(), 'Frame 300 px')

        again = self.make_window(settings)
        again._load_magnifier_settings(settings)
        self.assertEqual(again.canvas.magnifier.chosen_size, magnifier.HALF_HEIGHT)
        self.assertEqual(again.canvas.overview.chosen_size, magnifier.HALF_HEIGHT)
        self.assertTrue(again.overview_size_actions[magnifier.HALF_HEIGHT].isChecked())


if __name__ == '__main__':
    unittest.main()
