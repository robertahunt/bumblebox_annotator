"""Four-corner measurement of ArUco tag size on original video frames."""

from pathlib import Path

import cv2
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QMessageBox,
    QRadioButton, QSpinBox, QStyle, QToolButton, QVBoxLayout,
)

from core.aruco_measurement import MeasurementVideoReader, measure_tag, tag_perimeter_bounds


class TagMeasurementView(QGraphicsView):
    point_pressed = pyqtSignal(QPointF)
    point_dragged = pyqtSignal(QPointF)
    point_released = pyqtSignal()
    COLORS = {"smallest": QColor("#ffdf32"), "largest": QColor("#28c9ed")}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.image_item = None
        self.overlays = {}
        self._pan_position = None
        self.setBackgroundBrush(QColor("#202020"))
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(360, 240)

    def set_frame(self, frame):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixmap = QPixmap.fromImage(QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888
        ).copy())
        if self.image_item is None:
            self.image_item = self.scene().addPixmap(pixmap)
        else:
            self.image_item.setPixmap(pixmap)
        self.scene().setSceneRect(0, 0, width, height)

    def fit_frame(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        current = self.transform().m11()
        target = min(32.0, max(0.02, current * factor))
        self.scale(target / current, target / current)

    def wheelEvent(self, event):
        if event.angleDelta().y():
            self.zoom(1.25 if event.angleDelta().y() > 0 else 0.8)
        event.accept()

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan_position = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self.point_pressed.emit(point)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._pan_position is not None:
            delta = event.position().toPoint() - self._pan_position
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_position = event.position().toPoint()
        elif event.buttons() & Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self.sceneRect().contains(point):
                self.point_dragged.emit(point)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._pan_position = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.point_released.emit()
        event.accept()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        radius = 5 / self.transform().m11()
        for role, points in self.overlays.items():
            pen = QPen(self.COLORS[role], 1.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            polygon = QPolygonF([QPointF(*point) for point in points])
            if len(points) == 4:
                painter.drawPolygon(polygon)
            elif len(points) > 1:
                painter.drawPolyline(polygon)
            painter.setBrush(self.COLORS[role])
            for point in polygon:
                painter.drawEllipse(point, radius, radius)


class ArucoMeasurementDialog(QDialog):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.reader = MeasurementVideoReader(self.video_path)
        self.measurements = {"smallest": None, "largest": None}
        self.active_role = "smallest"
        self.frame_index = 0
        self.frame_size = None
        self.drag_index = None
        self.bounds = None
        self.setWindowTitle(f"Measure Tag Bounds - {self.video_path.name}")
        self.resize(1050, 760)

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.previous_button = self._tool("Previous frame", QStyle.StandardPixmap.SP_ArrowBack,
                                          lambda: self.load_frame(self.frame_index - 1))
        self.next_button = self._tool("Next frame", QStyle.StandardPixmap.SP_ArrowForward,
                                      lambda: self.load_frame(self.frame_index + 1))
        controls.addWidget(self.previous_button)
        controls.addWidget(QLabel("Frame:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, self.reader.frame_count or 1)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(lambda value: self.load_frame(value - 1))
        controls.addWidget(self.frame_spin)
        controls.addWidget(self.next_button)
        self.frame_label = QLabel()
        controls.addWidget(self.frame_label)
        controls.addStretch()
        for label, theme, fallback, callback in (
            ("Zoom out", "zoom-out", "-", lambda: self.view.zoom(0.8)),
            ("Zoom in", "zoom-in", "+", lambda: self.view.zoom(1.25)),
            ("Fit frame", "zoom-fit-best", "Fit", lambda: self.view.fit_frame()),
        ):
            button = QToolButton()
            button.setIcon(QIcon.fromTheme(theme))
            if button.icon().isNull():
                button.setText(fallback)
            button.setToolTip(label)
            button.setAccessibleName(label)
            button.clicked.connect(callback)
            controls.addWidget(button)
        layout.addLayout(controls)

        roles = QHBoxLayout()
        self.role_group = QButtonGroup(self)
        for role in self.measurements:
            radio = QRadioButton(f"{role.title()} tag")
            radio.setToolTip("Mark the four outer black-border corners in order around the tag.")
            radio.setChecked(role == self.active_role)
            radio.toggled.connect(lambda checked, key=role: self.set_role(key) if checked else None)
            self.role_group.addButton(radio)
            roles.addWidget(radio)
        roles.addStretch()
        self.undo_button = self._tool("Remove last corner", QStyle.StandardPixmap.SP_ArrowBack,
                                      self.remove_last_corner)
        roles.addWidget(self.undo_button)
        roles.addWidget(self._tool("Clear selected measurement", QStyle.StandardPixmap.SP_TrashIcon,
                                   self.clear_measurement))
        layout.addLayout(roles)

        self.view = TagMeasurementView()
        self.view.setToolTip(
            "Click four outer tag corners in clockwise or counterclockwise order. "
            "Drag corners to adjust. Scroll to zoom; right-drag or middle-drag to pan."
        )
        self.view.point_pressed.connect(self.place_corner)
        self.view.point_dragged.connect(self.move_corner)
        self.view.point_released.connect(self.end_drag)
        layout.addWidget(self.view, 1)

        form = QFormLayout()
        self.summary_labels = {}
        for role in self.measurements:
            label = QLabel()
            label.setWordWrap(True)
            self.summary_labels[role] = label
            form.addRow(f"{role.title()} tag:", label)
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0, 50)
        self.margin_spin.setValue(10)
        self.margin_spin.setSuffix(" %")
        self.margin_spin.setToolTip("Lower bound = smallest rate minus this margin; upper = largest plus it.")
        self.margin_spin.valueChanged.connect(self.refresh)
        form.addRow("Margin:", self.margin_spin)
        self.bounds_label = QLabel()
        self.bounds_label.setWordWrap(True)
        form.addRow("Perimeter bounds:", self.bounds_label)
        layout.addLayout(form)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.addButton("Apply to batch", QDialogButtonBox.ButtonRole.AcceptRole)
        self.apply_button.setToolTip("Replace the minimum and maximum perimeter sweeps for every video in this batch.")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.finished.connect(lambda _result: self.reader.close())
        layout.addWidget(self.buttons)
        try:
            first = self.reader.read(0)
            self._display_frame(0, first)
        except Exception:
            self.reader.close()
            raise
        QTimer.singleShot(0, self.view.fit_frame)

    def _tool(self, label, icon, callback):
        button = QToolButton()
        button.setIcon(self.style().standardIcon(icon))
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.clicked.connect(callback)
        return button

    def load_frame(self, index):
        self.end_drag()
        try:
            frame = self.reader.read(index)
        except ValueError as exc:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(self.frame_index + 1)
            self.frame_spin.blockSignals(False)
            QMessageBox.warning(self, "Frame unavailable", str(exc))
            return
        self._display_frame(index, frame)

    def _display_frame(self, index, frame):
        self.frame_index = index
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.view.set_frame(frame)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(max(self.reader.frame_count or 1, index + 1))
        self.frame_spin.setValue(index + 1)
        self.frame_spin.blockSignals(False)
        count = str(self.reader.frame_count) if self.reader.frame_count else "unknown"
        self.frame_label.setText(f"/ {count}   {self.frame_size[0]} x {self.frame_size[1]} px")
        self.previous_button.setEnabled(index > 0)
        self.next_button.setEnabled(self.reader.frame_count is None or index + 1 < self.reader.frame_count)
        self.refresh()

    def set_role(self, role):
        self.end_drag()
        self.active_role = role
        self.refresh()

    def place_corner(self, point):
        measurement = self.measurements[self.active_role]
        if measurement is None or measurement["frame_index"] != self.frame_index:
            measurement = {"frame_index": self.frame_index, "frame_size": self.frame_size, "points": []}
            self.measurements[self.active_role] = measurement
        points = measurement["points"]
        self.drag_index = None
        for index, corner in enumerate(points):
            if (QPointF(*corner) - point).manhattanLength() * self.view.transform().m11() <= 10:
                self.drag_index = index
                return
        if len(points) < 4:
            points.append((point.x(), point.y()))
            self.drag_index = len(points) - 1
            self.refresh()

    def move_corner(self, point):
        if self.drag_index is not None:
            self.measurements[self.active_role]["points"][self.drag_index] = (point.x(), point.y())
            self.refresh()

    def end_drag(self):
        self.drag_index = None

    def clear_measurement(self):
        self.end_drag()
        self.measurements[self.active_role] = None
        self.refresh()

    def remove_last_corner(self):
        self.end_drag()
        measurement = self.measurements[self.active_role]
        if measurement and measurement["points"]:
            measurement["points"].pop()
        self.refresh()

    def refresh(self):
        rates = {}
        overlays = {}
        for role, measurement in self.measurements.items():
            text = "Not measured"
            if measurement:
                points = measurement["points"]
                if measurement["frame_index"] == self.frame_index:
                    overlays[role] = points
                text = f"{len(points)}/4 corners - frame {measurement['frame_index'] + 1}"
                if len(points) == 4:
                    try:
                        result = measure_tag(points, *measurement["frame_size"])
                        rates[role] = result["perimeter_rate"]
                        text = (f"{result['perimeter_px']:.2f} px / rate {result['perimeter_rate']:.6f}"
                                f" - frame {measurement['frame_index'] + 1}")
                    except ValueError as exc:
                        text = str(exc)
            self.summary_labels[role].setText(text)
        self.view.overlays = overlays
        self.view.viewport().update()
        self.bounds = None
        text = "Pending measurements"
        if len(rates) == 2:
            try:
                self.bounds = tag_perimeter_bounds(rates["smallest"], rates["largest"], self.margin_spin.value())
                text = f"{self.bounds[0]:.6f} to {self.bounds[1]:.6f}"
            except ValueError as exc:
                text = str(exc)
        self.bounds_label.setText(text)
        self.apply_button.setEnabled(self.bounds is not None)
        active = self.measurements[self.active_role]
        self.undo_button.setEnabled(bool(active and active["points"]))

    def accept(self):
        if self.bounds is not None:
            super().accept()
