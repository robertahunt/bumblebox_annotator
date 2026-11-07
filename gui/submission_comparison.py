"""Choose a complete local or submitted annotation set before editing it."""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QPlainTextEdit,
    QRadioButton, QSlider, QStyle, QToolButton, QVBoxLayout, QWidget)

from core.annotation_work import task_categories
from core.categories import CATEGORIES, category_label
from core.submission_comparison import comparison_preview
from gui.project_sync_dialog import SyncProgressDialog


def _pixmap(array):
    height, width, channels = array.shape
    fmt = QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
    return QPixmap.fromImage(QImage(array.data, width, height, array.strides[0], fmt).copy())


class ComparisonView(QGraphicsView):
    moved = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumSize(240, 240)
        self.overlay = None
        self.horizontalScrollBar().valueChanged.connect(lambda: self.moved.emit(self))
        self.verticalScrollBar().valueChanged.connect(lambda: self.moved.emit(self))

    def display(self, image, overlay=None):
        self.overlay = None
        self.scene().clear()
        item = self.scene().addPixmap(image)
        self.setSceneRect(item.boundingRect())
        if overlay is not None:
            self.overlay = self.scene().addPixmap(_pixmap(overlay))

    def wheelEvent(self, event):
        if not event.angleDelta().y():
            event.ignore()
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.02 <= self.transform().m11() * factor <= 40:
            self.scale(factor, factor)
            self.moved.emit(self)
        event.accept()


class SubmissionComparisonDialog(QDialog):
    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.plan = plan
        self.choice = None
        self.ready = False
        self.syncing = False
        self.loading = False
        self.setWindowTitle('Resolve Annotation Conflict')
        self.resize(1200, 850)
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        title = QLabel(plan.video)
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        scope = QLabel(f'Applies to all annotation types on {len(plan.entry["frames"])} assigned frames, '
                       'plus shared-video annotations')
        scope.setWordWrap(True)
        layout.addWidget(scope)
        controls = QHBoxLayout()
        controls.addWidget(QLabel('Frame:'))
        self.frames = QComboBox()
        for frame in plan.entry['frames']:
            self.frames.addItem(str(frame), frame)
        controls.addWidget(self.frames)
        controls.addWidget(QLabel('Show:'))
        self.categories = QComboBox()
        self.categories.addItem('Assigned task', tuple(task_categories(plan.entry.get('task'))))
        self.categories.addItem('All mask types', None)
        for category in CATEGORIES:
            self.categories.addItem(category_label(category), (category,))
        controls.addWidget(self.categories, 1)
        self.masks = QCheckBox('Masks / boxes')
        self.masks.setChecked(True)
        controls.addWidget(self.masks)
        controls.addWidget(QLabel('Opacity:'))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.setMaximumWidth(120)
        controls.addWidget(self.opacity)
        fit = QToolButton()
        fit.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarMaxButton))
        fit.setToolTip('Fit both images')
        fit.setAccessibleName('Fit both images')
        fit.clicked.connect(self.fit_views)
        controls.addWidget(fit)
        layout.addLayout(controls)
        panes = QHBoxLayout()
        self.views, self.summaries, self.options = {}, {}, {}
        for version, heading in (('local', 'Local annotations'),
                                 ('submitted', 'Submitted by ' + plan.entry['contributor']['name'])):
            panel = QWidget()
            column = QVBoxLayout(panel)
            column.setContentsMargins(0, 0, 0, 0)
            label = QLabel(heading)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            column.addWidget(label)
            view = ComparisonView()
            view.moved.connect(self.sync_views)
            column.addWidget(view, 1)
            summary = QLabel('Loading...')
            summary.setTextFormat(Qt.TextFormat.PlainText)
            summary.setWordWrap(True)
            column.addWidget(summary)
            option = QRadioButton('Keep local annotations' if version == 'local' else 'Use submitted annotations')
            # Each pane has its own parent; exclusivity is handled explicitly.
            option.setAutoExclusive(False)
            option.toggled.connect(lambda checked, key=version: self.choose(key, checked))
            column.addWidget(option)
            self.views[version], self.summaries[version], self.options[version] = view, summary, option
            panes.addWidget(panel, 1)
        layout.addLayout(panes, 1)
        details_button = QToolButton()
        details_button.setText(f'Conflicting files ({len(plan.conflicts)})')
        details_button.setCheckable(True)
        details_button.setArrowType(Qt.ArrowType.RightArrow)
        details_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout.addWidget(details_button)
        details = QPlainTextEdit('\n'.join(plan.conflicts))
        details.setReadOnly(True)
        details.setMaximumHeight(120)
        details.hide()
        details_button.toggled.connect(details.setVisible)
        details_button.toggled.connect(lambda checked: details_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow))
        layout.addWidget(details)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.continue_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.continue_button.setText('Use Selection and Review')
        self.continue_button.setEnabled(False)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.frames.currentIndexChanged.connect(self.load_preview)
        self.categories.currentIndexChanged.connect(self.load_preview)
        self.masks.toggled.connect(self.update_opacity)
        self.opacity.valueChanged.connect(self.update_opacity)
        QTimer.singleShot(0, lambda: self.load_preview() if self.isVisible() else None)

    def choose(self, version, checked):
        if checked:
            self.choice = version
            for key, option in self.options.items():
                if key != version:
                    option.setChecked(False)
        elif self.choice == version:
            self.choice = None
        self.continue_button.setEnabled(self.ready and self.choice is not None)

    def load_preview(self):
        if self.loading:
            return
        self.loading, self.ready = True, False
        self.continue_button.setEnabled(False)
        for view in self.views.values():
            view.overlay = None
            view.scene().clear()
        try:
            frame, categories = self.frames.currentData(), self.categories.currentData()
            result = SyncProgressDialog('Load Annotation Comparison', lambda progress, cancelled:
                comparison_preview(self.plan, frame, categories, progress, cancelled), self).run()
            pixmap = _pixmap(result['image'])
            self.syncing = True
            for version, view in self.views.items():
                data = result[version]
                view.display(pixmap, data.get('overlay'))
                self.options[version].setEnabled('error' not in data)
                if 'error' in data:
                    self.options[version].setChecked(False)
                    self.summaries[version].setText('Preview unavailable: ' + data['error'])
                else:
                    counts = data['counts']
                    self.summaries[version].setText(', '.join(f'{category_label(k)}: {v}' for k, v in counts.items())
                                                     or 'No masks or boxes in this view')
            self.ready = True
            self.fit_views()
            self.update_opacity()
        except Exception as exc:
            for version in self.views:
                self.options[version].setEnabled(False)
                self.summaries[version].setText('Preview unavailable: ' + str(exc))
        finally:
            self.syncing, self.loading = False, False
            self.continue_button.setEnabled(self.ready and self.choice is not None)

    def sync_views(self, source):
        if self.syncing or not self.ready:
            return
        self.syncing = True
        try:
            center = source.mapToScene(source.viewport().rect().center())
            for view in self.views.values():
                if view is not source:
                    view.setTransform(source.transform())
                    view.centerOn(center)
        finally:
            self.syncing = False

    def fit_views(self):
        self.syncing = True
        for view in self.views.values():
            view.fitInView(view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.syncing = False

    def update_opacity(self):
        for view in self.views.values():
            if view.overlay is not None:
                view.overlay.setOpacity(self.opacity.value() / 100 if self.masks.isChecked() else 0)

    def accept(self):
        if self.ready and self.choice and self.options[self.choice].isEnabled():
            super().accept()
