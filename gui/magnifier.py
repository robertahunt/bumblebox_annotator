"""A close-up of the image under the cursor, drawn in a corner of the canvas."""

import math

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

SIZES = (100, 150, 200, 300)  # side of the square frame, in screen pixels
ZOOMS = (2, 3, 4, 6, 8)       # enlargement relative to the main view
DEFAULT_SIZE = 100
DEFAULT_ZOOM = 3
MARGIN = 8


class CursorMagnifier(QWidget):
    """The scene around the cursor, enlarged, with a mark at the exact cursor position.

    It shows what the canvas shows (image, masks, boxes, keypoints), follows the
    main view's zoom, and lets clicks through to the canvas. Its home is the
    top-left corner of the viewport; while the cursor is under that corner it
    steps to the bottom-left, so it never covers what is being worked on.
    """

    def __init__(self, canvas):
        super().__init__(canvas.viewport())
        self.canvas = canvas
        self.active = True
        self.zoom = DEFAULT_ZOOM
        self.cursor_pos = None  # viewport position being magnified, None off the canvas
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Opaque, so redrawing the close-up never asks the canvas underneath to redraw.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(DEFAULT_SIZE, DEFAULT_SIZE)
        self.hide()

    def set_active(self, active):
        self.active = bool(active)
        self._refresh()

    def set_frame_size(self, side):
        self.setFixedSize(int(side), int(side))
        self._refresh()

    def set_zoom(self, zoom):
        self.zoom = float(zoom)
        self._refresh()

    def track(self, viewport_pos):
        """Follow the cursor to a position in the canvas viewport."""
        self.cursor_pos = QPoint(viewport_pos)
        self._refresh()

    def clear(self):
        """The cursor left the canvas, or there is no image."""
        self.cursor_pos = None
        self.hide()

    def view_painted(self, region):
        """The canvas redrew (an edit, a zoom, a scroll, a new frame): match it."""
        if self.isVisible() and not self.geometry().contains(region.boundingRect()):
            self.update()

    def place(self):
        """Keep to the top-left corner, or the bottom-left while the cursor is under the top-left."""
        side = self.width()
        spot = QRect(MARGIN, MARGIN, side, side)
        # Wide enough that the area being magnified is never under the frame.
        reach = math.ceil(side / (2 * self.zoom)) + MARGIN
        if self.cursor_pos is not None and spot.adjusted(-reach, -reach, reach, reach).contains(self.cursor_pos):
            spot.moveTop(max(MARGIN, self.canvas.viewport().height() - side - MARGIN))
        if self.pos() != spot.topLeft():
            self.move(spot.topLeft())

    def source_rect(self):
        """The scene area shown in the frame."""
        scale = self.canvas.transform().m11() * self.zoom
        if self.cursor_pos is None or scale <= 0:
            return QRectF()
        centre = self.canvas.mapToScene(self.cursor_pos)
        span = self.width() / scale
        return QRectF(centre.x() - span / 2, centre.y() - span / 2, span, span)

    def _refresh(self):
        if not (self.active and self.cursor_pos is not None and self.canvas.current_image is not None):
            self.hide()
            return
        self.place()
        if not self.isVisible():
            self.show()
            self.raise_()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.canvas.backgroundBrush())
        source = self.source_rect()
        if not source.isEmpty():
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.canvas.scene.render(painter, QRectF(self.rect()), source)
        self._draw_cursor_mark(painter)
        painter.setPen(QPen(QColor(20, 24, 30), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))

    def _draw_cursor_mark(self, painter):
        """Four ticks around the centre, open in the middle so the target stays visible."""
        centre = self.width() / 2
        gap, arm = 4, max(8, self.width() // 10)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for colour, width in ((QColor(0, 0, 0, 200), 3), (QColor(255, 255, 255), 1)):
            painter.setPen(QPen(colour, width))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                painter.drawLine(QPointF(centre + dx * gap, centre + dy * gap),
                                 QPointF(centre + dx * (gap + arm), centre + dy * (gap + arm)))
