"""Pointer preview and annotation overlays must not move the image viewport."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPen, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGraphicsTextItem

from gui.canvas import ImageCanvas, OutwardBBoxRectItem


class CanvasViewStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def canvas(self):
        canvas = ImageCanvas()
        canvas.resize(800, 600)
        canvas.show()
        canvas.load_image(np.zeros((1000, 1000), dtype=np.uint8))
        self.app.processEvents()
        self.app.processEvents()
        self.addCleanup(canvas.close)
        self.addCleanup(canvas._brush_cursor_timer.stop)
        self.addCleanup(canvas._viz_update_timer.stop)
        return canvas

    def test_overlays_crossing_image_edges_do_not_shift_fit_or_zoomed_view(self):
        for zoomed in (False, True):
            with self.subTest(zoomed=zoomed):
                canvas = self.canvas()
                if zoomed:
                    canvas.zoom_in()
                    canvas.zoom_in()
                    canvas.centerOn(650, 600)
                    self.app.processEvents()
                canvas.set_tool('brush')
                canvas.set_brush_size(200)
                before = canvas.get_view_state()
                for point in (QPointF(1, 1), QPointF(999, 999), QPointF(500, 1)):
                    canvas._update_brush_cursor_preview(point)
                    self.app.processEvents()
                    self.assertEqual(canvas.scene.sceneRect(), QRectF(0, 0, 1000, 1000))
                    self.assertEqual(canvas.get_view_state(), before)
                canvas.set_brush_cursor_preview_enabled(False)
                box = OutwardBBoxRectItem(-30, -30, 1100, 1100)
                box.setPen(QPen(Qt.GlobalColor.green, 6))
                label = QGraphicsTextItem('Instance near the edge')
                label.setPos(-20, -20)
                canvas.scene.addItem(box)
                canvas.scene.addItem(label)
                self.app.processEvents()
                self.assertEqual(canvas.get_view_state(), before)
                canvas.scene.removeItem(box)
                canvas.scene.removeItem(label)
                self.app.processEvents()
                self.assertEqual(canvas.get_view_state(), before)

    def test_stroke_bbox_updates_and_mask_toggles_do_not_shift_view(self):
        canvas = self.canvas()
        mask = np.zeros((1000, 1000), dtype=np.uint8)
        mask[:50, :50] = 255
        canvas.set_annotations([{'mask_id': 1, 'category': 'bee', 'mask': mask}])
        canvas.set_show_bboxes(True)
        canvas.set_tool('brush')
        canvas.start_editing_instance(1, 'bee')
        canvas.set_brush_cursor_preview_enabled(False)
        self.app.processEvents()
        before = canvas.get_view_state()
        canvas.draw_on_mask(QPointF(10, 10), QPointF(0, 500))
        canvas._flush_pending_viz_update()
        canvas.add_selection_border(canvas.editing_mask)
        canvas.update_instance_labels()
        self.app.processEvents()
        self.assertEqual(canvas.get_view_state(), before)
        for _ in range(2):
            canvas.toggle_annotation_overlays()
            self.app.processEvents()
            self.assertEqual(canvas.get_view_state(), before)

    def test_preview_coalesces_burst_to_latest_point(self):
        canvas = self.canvas()
        canvas.set_tool('brush')
        canvas._queue_brush_cursor_preview(QPointF(100, 100))
        with patch.object(canvas, '_update_brush_cursor_preview',
                          wraps=canvas._update_brush_cursor_preview) as update:
            for i in range(1000):
                canvas._queue_brush_cursor_preview(QPointF(100 + i % 100, 200))
            self.assertEqual(update.call_count, 0)
            self.assertTrue(canvas._brush_cursor_timer.isActive())
            QTest.qWait(40)
            self.assertEqual(update.call_count, 1)
        circle, h_line, v_line = canvas.brush_cursor_items
        self.assertEqual(circle.pos(), QPointF(199, 200))
        self.assertEqual(h_line.pos(), circle.pos())
        self.assertEqual(v_line.pos(), circle.pos())
        self.assertEqual(circle.rect().center(), QPointF(0, 0))
        self.assertEqual(circle.rect().width(), canvas.brush_size)
        canvas.set_brush_size(7)
        self.assertEqual(circle.rect().width(), 7)
        canvas.set_tool('eraser')
        self.assertEqual(circle.pen().color().getRgb(), (255, 80, 80, 235))

    def test_throttling_preview_does_not_drop_stroke_points(self):
        canvas = self.canvas()
        canvas.set_tool('brush')
        canvas.is_drawing = True
        canvas.last_point = QPointF(100, 100)
        with patch.object(canvas, 'draw_on_mask') as draw:
            for index in range(40):
                event = QMouseEvent(QEvent.Type.MouseMove, QPointF(100 + index, 120),
                    QPointF(100 + index, 120), Qt.MouseButton.NoButton,
                    Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
                canvas.mouseMoveEvent(event)
            self.assertEqual(draw.call_count, 40)
        self.assertTrue(canvas._brush_cursor_timer.isActive())

    def test_pending_cursor_is_cancelled_on_leave_disable_and_new_frame(self):
        for action in ('leave', 'disable', 'load', 'clear'):
            with self.subTest(action=action):
                canvas = self.canvas()
                canvas.set_tool('brush')
                canvas._queue_brush_cursor_preview(QPointF(100, 100))
                canvas._queue_brush_cursor_preview(QPointF(200, 100))
                self.assertTrue(canvas._brush_cursor_timer.isActive())
                if action == 'leave':
                    canvas.leaveEvent(QEvent(QEvent.Type.Leave))
                elif action == 'disable':
                    canvas.set_brush_cursor_preview_enabled(False)
                elif action == 'load':
                    canvas.load_image(np.zeros((200, 300), dtype=np.uint8))
                else:
                    canvas.clear_image()
                self.assertFalse(canvas._brush_cursor_timer.isActive())
                QTest.qWait(25)
                self.assertFalse(canvas.brush_cursor_items)
                self.assertNotEqual(canvas.cursor().shape(), Qt.CursorShape.BlankCursor)

    def test_new_image_resets_canvas_extent(self):
        canvas = self.canvas()
        canvas.load_image(np.zeros((200, 300), dtype=np.uint8))
        self.app.processEvents()
        self.assertEqual(canvas.scene.sceneRect(), QRectF(0, 0, 300, 200))
        canvas.clear_image()
        self.assertTrue(canvas.scene.sceneRect().isEmpty())

    def test_next_frame_preserves_view_after_queued_layout_updates(self):
        canvas = self.canvas()
        canvas.zoom_in()
        canvas.zoom_in()
        canvas.centerOn(650, 600)
        canvas.set_tool('brush')
        canvas.set_brush_size(100)
        canvas._update_brush_cursor_preview(QPointF(1, 1))
        self.app.processEvents()
        before = canvas.get_view_state()
        canvas.load_image(np.ones((1000, 1000), dtype=np.uint8), preserve_view=True)
        self.app.processEvents()
        self.app.processEvents()
        self.assertEqual(canvas.get_view_state(), before)

    def wheel(self, angle=QPoint(), pixels=QPoint()):
        return QWheelEvent(QPointF(100, 100), QPointF(100, 100), pixels, angle,
                           Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                           Qt.ScrollPhase.NoScrollPhase, False)

    def test_empty_or_horizontal_wheel_event_cannot_zoom(self):
        canvas = self.canvas()
        canvas.zoom_in()
        self.app.processEvents()
        before = canvas.get_view_state()
        for event in (self.wheel(), self.wheel(QPoint(120, 0)), self.wheel(pixels=QPoint(20, 0))):
            canvas.wheelEvent(event)
            self.app.processEvents()
            self.assertTrue(event.isAccepted())
            self.assertEqual(canvas.get_view_state(), before)

    def test_real_vertical_scroll_still_zooms(self):
        canvas = self.canvas()
        for event in (self.wheel(QPoint(0, 120)), self.wheel(pixels=QPoint(0, 20))):
            before = canvas.transform().m11()
            canvas.wheelEvent(event)
            self.assertGreater(canvas.transform().m11(), before)
        before = canvas.transform().m11()
        canvas.wheelEvent(self.wheel(QPoint(0, -120)))
        self.assertLess(canvas.transform().m11(), before)


if __name__ == '__main__':
    unittest.main()
