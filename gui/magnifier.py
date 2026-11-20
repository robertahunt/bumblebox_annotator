"""A close-up and a zoomed-out view of the image around the cursor, in a corner of the canvas."""

import math

from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QTransform
from PyQt6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

# Side of a square frame, in screen pixels. HALF_HEIGHT is as tall as half the
# canvas: two such frames, one above the other, fill its left side.
HALF_HEIGHT = 0
SIZES = (100, 150, 200, 300, 600, HALF_HEIGHT)
MARGIN = 8
MIN_SIDE = 50

# The close-up: enlargement relative to the main view.
ZOOMS = (2, 3, 4, 6, 8)
DEFAULT_SIZE = 100
DEFAULT_ZOOM = 3

# The zoomed-out view: how many times as far across as the main view it reaches.
WHOLE_FRAME = 0
REACHES = (1.05, 1.1, 1.15, 1.2, 1.25, 1.5, 1.75, 2, WHOLE_FRAME)
DEFAULT_OVERVIEW_SIZE = 200
DEFAULT_REACH = 2


class CursorViews:
    """The close-up and the zoomed-out view, kept together in a corner of the canvas.

    With the main view between them they give three zoom levels at once. Their
    home is the top-left corner of the viewport, the close-up above the
    zoomed-out view, or side by side where the viewport is too short for that.
    While the cursor is under them they move to another corner, so they never
    cover what is being worked on.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self.cursor_pos = None  # viewport position being followed, None off the canvas
        self.magnifier = CursorMagnifier(self)
        self.overview = CursorOverview(self)
        self.frames = (self.magnifier, self.overview)

    def track(self, viewport_pos):
        """Follow the cursor to a position in the canvas viewport."""
        self.cursor_pos = QPoint(viewport_pos)
        self.refresh()

    def clear(self):
        """The cursor left the canvas, or there is no image."""
        self.cursor_pos = None
        for frame in self.frames:
            frame.hide()

    def view_painted(self, region):
        """The canvas redrew (an edit, a zoom, a scroll, a new frame): match it."""
        for frame in self.frames:
            frame.view_painted(region)

    def refresh(self):
        following = self.cursor_pos is not None and self.canvas.current_image is not None
        shown = [frame for frame in self.frames if following and frame.active]
        for frame in self.frames:
            if frame not in shown:
                frame.hide()
        self.place()
        for frame in shown:
            if not frame.isVisible():
                frame.show()
                frame.raise_()
            frame.update()

    def half_height_side(self):
        """The frame side that lets two frames, one above the other, fill the canvas height."""
        return max(MIN_SIDE, (self.canvas.viewport().height() - 3 * MARGIN) // 2)

    def place(self):
        """Size the frames, and put them in the first corner where the cursor is not under them."""
        for frame in self.frames:
            side = self.half_height_side() if frame.chosen_size == HALF_HEIGHT else frame.chosen_size
            if frame.width() != side:
                frame.setFixedSize(side, side)
        frames = [frame for frame in self.frames if frame.active]
        if not frames:
            return
        viewport = self.canvas.viewport().rect()
        offsets = self._offsets(frames, viewport)
        width = max(offset.x() + frame.width() for frame, offset in zip(frames, offsets))
        height = max(offset.y() + frame.height() for frame, offset in zip(frames, offsets))
        left, top = MARGIN, MARGIN
        right = max(MARGIN, viewport.width() - width - MARGIN)
        bottom = max(MARGIN, viewport.height() - height - MARGIN)
        corners = [QPoint(left, top), QPoint(left, bottom), QPoint(right, top), QPoint(right, bottom)]
        # Wide enough that what the close-up shows is never under a frame.
        reach = max(frame.clearance() for frame in frames)

        def covers_cursor(corner):
            return any(QRect(corner + offset, frame.size()).adjusted(-reach, -reach, reach, reach)
                       .contains(self.cursor_pos) for frame, offset in zip(frames, offsets))

        def distance(corner):
            centre = corner + QPoint(width // 2, height // 2) - self.cursor_pos
            return centre.x() ** 2 + centre.y() ** 2

        corner = corners[0]
        if self.cursor_pos is not None and covers_cursor(corner):
            free = [candidate for candidate in corners if not covers_cursor(candidate)]
            # Frames too big to get out of the way anywhere go as far off as they can.
            corner = free[0] if free else max(corners, key=distance)
        for frame, offset in zip(frames, offsets):
            if frame.pos() != corner + offset:
                frame.move(corner + offset)

    @staticmethod
    def _offsets(frames, viewport):
        """Where each frame sits within the group: a column, or a row where a column is too tall."""
        tall = sum(frame.height() for frame in frames) + MARGIN * (len(frames) + 1)
        wide = sum(frame.width() for frame in frames) + MARGIN * (len(frames) + 1)
        in_a_row = tall > viewport.height() and wide <= viewport.width()
        offsets, along = [], 0
        for frame in frames:
            offsets.append(QPoint(along, 0) if in_a_row else QPoint(0, along))
            along += frame.width() + MARGIN
        return offsets


class CursorFrame(QWidget):
    """A square frame over the canvas showing the scene around the cursor.

    It lets clicks through to the canvas, and is opaque so redrawing it never
    asks the canvas underneath to redraw.
    """

    def __init__(self, views, side):
        super().__init__(views.canvas.viewport())
        self.views = views
        self.canvas = views.canvas
        self.active = True
        self.chosen_size = side  # pixels, or HALF_HEIGHT
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setFixedSize(side, side)
        self.hide()

    @property
    def cursor_pos(self):
        return self.views.cursor_pos

    def set_active(self, active):
        self.active = bool(active)
        self.views.refresh()

    def set_frame_size(self, side):
        """A side in pixels, or HALF_HEIGHT to follow the canvas height."""
        self.chosen_size = int(side)
        self.views.refresh()

    def clearance(self):
        """How far the cursor is kept from the frames, in screen pixels."""
        return MARGIN

    def view_painted(self, region):
        if self.isVisible() and not self.geometry().contains(region.boundingRect()):
            self.update()

    def source_rect(self):
        """The scene area shown in the frame."""
        raise NotImplementedError

    def _draw_cursor_mark(self, painter, centre, gap, arm):
        """Four ticks around the cursor position, open in the middle so the target stays visible."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for colour, width in ((QColor(0, 0, 0, 200), 3), (QColor(255, 255, 255), 1)):
            painter.setPen(QPen(colour, width))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                painter.drawLine(QPointF(centre.x() + dx * gap, centre.y() + dy * gap),
                                 QPointF(centre.x() + dx * (gap + arm), centre.y() + dy * (gap + arm)))

    def _draw_border(self, painter):
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(20, 24, 30), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))


class CursorMagnifier(CursorFrame):
    """The scene around the cursor, enlarged, with a mark at the exact cursor position.

    It shows what the canvas shows (image, masks, boxes, keypoints) and follows
    the main view's zoom.
    """

    def __init__(self, views):
        super().__init__(views, DEFAULT_SIZE)
        self.zoom = DEFAULT_ZOOM

    def set_zoom(self, zoom):
        self.zoom = float(zoom)
        self.views.refresh()

    def clearance(self):
        return math.ceil(self.width() / (2 * self.zoom)) + MARGIN

    def source_rect(self):
        scale = self.canvas.transform().m11() * self.zoom
        if self.cursor_pos is None or scale <= 0:
            return QRectF()
        centre = self.canvas.mapToScene(self.cursor_pos)
        span = self.width() / scale
        return QRectF(centre.x() - span / 2, centre.y() - span / 2, span, span)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.canvas.backgroundBrush())
        source = self.source_rect()
        if not source.isEmpty():
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.canvas.scene.render(painter, QRectF(self.rect()), source)
        centre = QPointF(self.width() / 2, self.height() / 2)
        self._draw_cursor_mark(painter, centre, gap=4, arm=max(8, self.width() // 10))
        self._draw_border(painter)


class CursorOverview(CursorFrame):
    """The surroundings of the cursor, zoomed out, with the main view outlined.

    It reaches ``reach`` times as far across as the main view shows, or takes in
    the whole frame, and never looks past the edge of the frame while there is
    more of the frame to show. A mark shows where the cursor is.
    """

    def __init__(self, views):
        super().__init__(views, DEFAULT_OVERVIEW_SIZE)
        self.reach = DEFAULT_REACH

    def set_reach(self, reach):
        self.reach = float(reach)
        self.views.refresh()

    def main_view_rect(self):
        """The scene area the main view shows."""
        return self.canvas.mapToScene(self.canvas.viewport().rect()).boundingRect()

    def source_rect(self):
        image = self.canvas.image_item
        if self.cursor_pos is None or image is None or self.canvas.current_image is None:
            return QRectF()
        frame = image.sceneBoundingRect()
        side = max(frame.width(), frame.height())
        if self.reach != WHOLE_FRAME:
            seen = self.main_view_rect()
            side = min(side, self.reach * max(seen.width(), seen.height()))
        if side <= 0:
            return QRectF()
        centre = self.canvas.mapToScene(self.cursor_pos)

        def start(position, low, length):
            if side >= length:
                return low + (length - side) / 2  # all of it, centred
            return min(max(position - side / 2, low), low + length - side)

        return QRectF(start(centre.x(), frame.left(), frame.width()),
                      start(centre.y(), frame.top(), frame.height()), side, side)

    def to_frame(self, source):
        """Scene coordinates to positions in this frame."""
        scale = self.width() / source.width()
        return QTransform.fromTranslate(-source.x(), -source.y()) * QTransform.fromScale(scale, scale)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.canvas.backgroundBrush())
        source = self.source_rect()
        if not source.isEmpty():
            to_frame = self.to_frame(source)
            draw_scaling_items(painter, self.canvas.scene, source, to_frame)
            self._draw_main_view(painter, to_frame.mapRect(self.main_view_rect()))
            self._draw_cursor_mark(painter, to_frame.map(self.canvas.mapToScene(self.cursor_pos)), gap=3, arm=6)
        self._draw_border(painter)

    def _draw_main_view(self, painter, rect):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for colour, width in ((QColor(0, 0, 0, 200), 3), (QColor(255, 255, 255), 1)):
            painter.setPen(QPen(colour, width))
            painter.drawRect(rect)


def draw_scaling_items(painter, scene, source, to_device):
    """Draw the scene items in ``source`` that shrink and grow with the view.

    Number labels and keypoint markers keep their screen size at any zoom, so
    in a zoomed-out picture they would cover it. They are left out, which
    ``scene.render`` cannot do.
    """
    option = QStyleOptionGraphicsItem()
    for item in scene.items(source, Qt.ItemSelectionMode.IntersectsItemBoundingRect, Qt.SortOrder.AscendingOrder):
        if not item.isVisible() or _keeps_screen_size(item):
            continue
        painter.save()
        painter.setTransform(item.sceneTransform() * to_device, True)
        painter.setOpacity(item.effectiveOpacity())
        option.exposedRect = item.boundingRect()
        item.paint(painter, option, None)
        painter.restore()


def _keeps_screen_size(item):
    while item is not None:
        if item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations:
            return True
        item = item.parentItem()
    return False
