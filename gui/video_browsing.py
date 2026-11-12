"""Experiment filters and asynchronous local modification-time sorting."""

from datetime import datetime

from PyQt6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit,
                            QListWidgetItem, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
                            QTableWidgetItem, QVBoxLayout, QWidget)

from core.video_browsing import local_modification_times
from gui.project_sync_dialog import OperationWorker


SORT_ROLE = Qt.ItemDataRole.UserRole + 41
SPLIT_ROLE = Qt.ItemDataRole.UserRole + 43   # 'TRAIN', 'VAL', 'TEST', 'INFERENCE' or None
STATUS_ROLE = Qt.ItemDataRole.UserRole + 44  # {'state': summary state, 'count': '2/5'}

# Distinct shapes as well as colours, so the marks also read without colour.
STATE_COLORS = {'finished': '#2e9d4a', 'not_checked': '#c98200', 'changed': '#c98200',
                'needs_review': '#c62828'}
STATE_NAMES = {'finished': 'Finished', 'not_checked': 'Not checked', 'changed': 'Changed; recheck',
               'needs_review': 'Import review needed', 'not_started': 'Not started',
               'unknown': 'Unknown', 'not_scanned': 'Not scanned', 'loading': 'Loading'}
SPLIT_TAGS = {'TRAIN': ('Train', '#2e7d32'), 'VAL': ('Val', '#1565c0'),
              'TEST': ('Test', '#7b1fa2'), 'INFERENCE': ('Infer', '#616161')}
_STATUS_ICONS = {}


def status_icon(state):
    """Small status mark: check (finished), half circle (not checked), ring (not started),
    ! (changed or needs import review), ? (unknown), dotted ring (loading)."""
    if state in _STATUS_ICONS:
        return _STATUS_ICONS[state]
    size, scale = 14, 3
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(scale, scale)
    color = QColor(STATE_COLORS.get(state, '#8a8a8a'))
    ring = QRectF(1.5, 1.5, size - 3, size - 3)
    if state == 'finished':
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(ring)
        pen = QPen(QColor('white'), 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath(QPointF(4.2, 7.2))
        path.lineTo(6.2, 9.3)
        path.lineTo(10, 4.9)
        painter.drawPath(path)
    elif state == 'not_checked':
        painter.setPen(QPen(color, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(ring)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPie(ring, 90 * 16, 180 * 16)
    elif state in ('not_started', 'loading'):
        pen = QPen(color, 1.4)
        if state == 'loading':
            pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(ring)
    else:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(ring)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QColor('white'))
        painter.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter,
                         '!' if state in ('changed', 'needs_review') else '?')
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    _STATUS_ICONS[state] = QIcon(pixmap)
    return _STATUS_ICONS[state]


def status_count(summary):
    """Finished frames out of the frames in scope, e.g. '2/5'; blank when not meaningful."""
    if summary.get('state') in ('loading', 'unknown', 'not_scanned') or summary.get('shared'):
        return ''
    return f"{summary.get('finished', 0)}/{summary.get('total', 0)}"


class OneLineDelegate(QStyledItemDelegate):
    """Default painting, but rows never ask for more width than the view, so long text
    is shortened with an ellipsis instead of scrolling sideways."""

    def sizeHint(self, option, index):
        return QSize(0, super().sizeHint(option, index).height())


class VideoRowDelegate(QStyledItemDelegate):
    """One line per video: split tag, name shortened in the middle, status mark and count.

    The item's display text keeps the full description for screen readers and tooltips;
    only the painting is compact.
    """

    def sizeHint(self, option, index):
        # Zero width: rows never ask for horizontal scrolling.
        return QSize(0, max(QFontMetrics(option.font).height() + 8, 22))

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ''
        opt.icon = QIcon()
        widget = opt.widget
        style = widget.style() if widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        group = (QPalette.ColorGroup.Disabled if not opt.state & QStyle.StateFlag.State_Enabled else
                 QPalette.ColorGroup.Active if opt.state & QStyle.StateFlag.State_Active else
                 QPalette.ColorGroup.Inactive)
        text_color = opt.palette.color(group, QPalette.ColorRole.HighlightedText if selected
                                       else QPalette.ColorRole.Text)
        rect = opt.rect.adjusted(4, 0, -6, 0)
        middle = rect.top() + rect.height() // 2
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left = rect.left()
        tag = SPLIT_TAGS.get(index.data(SPLIT_ROLE))
        if tag:
            small = QFont(opt.font)
            small.setBold(True)
            small.setPointSizeF(max(6.5, opt.font.pointSizeF() * 0.78))
            metrics = QFontMetrics(small)
            width = max(metrics.horizontalAdvance(text) for text, _ in SPLIT_TAGS.values()) + 8
            height = metrics.height() + 2
            chip = QRect(left, middle - height // 2, width, height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(tag[1]))
            painter.drawRoundedRect(QRectF(chip), 3, 3)
            painter.setPen(QColor('white'))
            painter.setFont(small)
            painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, tag[0])
            left = chip.right() + 7
        painter.setFont(opt.font)
        metrics = QFontMetrics(opt.font)
        status = index.data(STATUS_ROLE) or {}
        right = rect.right()
        count = status.get('count', '')
        if count:
            width = metrics.horizontalAdvance(count)
            painter.setPen(text_color)
            painter.drawText(QRect(right - width + 1, rect.top(), width, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, count)
            right -= width + 4
        if status.get('state'):
            status_icon(status['state']).paint(painter, QRect(right - 13, middle - 7, 14, 14))
            right -= 14 + 6
        name_rect = QRect(left, rect.top(), max(0, right - left), rect.height())
        name = str(index.data(Qt.ItemDataRole.UserRole) or index.data(Qt.ItemDataRole.DisplayRole) or '')
        painter.setPen(text_color)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, name_rect.width()))
        painter.restore()


class VideoRowColors(QObject):
    """Stronger theme-aware stripes without changing selection or text colors."""

    def __init__(self, view):
        super().__init__(view)
        view.setAlternatingRowColors(True)
        view.installEventFilter(self)
        self.update_palette(view)

    def eventFilter(self, view, event):
        if event.type() == QEvent.Type.PaletteChange:
            self.update_palette(view)
        return False

    @staticmethod
    def update_palette(view):
        palette = view.palette()
        changed = False
        for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
            base = palette.color(group, QPalette.ColorRole.Base)
            shift = 32 if base.lightness() < 128 else -24
            alternate = QColor(*(max(0, min(255, value + shift))
                                 for value in (base.red(), base.green(), base.blue())))
            if palette.color(group, QPalette.ColorRole.AlternateBase) != alternate:
                palette.setColor(group, QPalette.ColorRole.AlternateBase, alternate)
                changed = True
        if changed:
            view.setPalette(palette)


class VideoMetadataWorker(OperationWorker):
    @pyqtSlot()
    def cancel(self):
        self.requestInterruption()


class VideoListItem(QListWidgetItem):
    def __lt__(self, other):
        return self.data(SORT_ROLE) < other.data(SORT_ROLE)


class ModifiedTableItem(QTableWidgetItem):
    def __lt__(self, other):
        return self.data(SORT_ROLE) < other.data(SORT_ROLE)


class VideoBrowserControls(QWidget):
    """Experiment filter, sort order and name search for a video list.

    With compact=True the widget lays out only its status line; the caller places
    the experiment, sort, search and "only mine" controls (the annotation sidebar
    puts them in its Filters panel).
    """
    changed = pyqtSignal()

    def __init__(self, parent=None, *, compact=False):
        super().__init__(parent)
        self.project = ''
        self.videos = ()
        self.assignments = {}
        self.my_work_project = ''
        self.my_work = {}
        self.modified = {}
        self.errors = {}
        self.checked_times = False
        self.worker = None
        self.pending = False
        self.stopping = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.only_mine = QCheckBox('Only my assigned videos')
        self.only_mine.setToolTip('Assignments from the last My Work check. Uncheck to show other local videos.')
        self.only_mine.hide()
        self.only_mine.toggled.connect(self.changed)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText('Search videos')
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName('Search videos')
        self.search.setToolTip('Show videos whose name or imaging setup contains all of these words')
        self.search.textChanged.connect(self.changed)
        self.search.setVisible(compact)
        form = QFormLayout()
        self.experiment = QComboBox()
        self.experiment.addItem('All experiments', None)
        self.experiment.setToolTip('Filter by the existing imaging setup. This does not change setup assignments or calibration.')
        self.sort = QComboBox()
        for label, value in (('Default order', 'default'), ('Name (A-Z)', 'name'),
                             ('Recently modified', 'recent')):
            self.sort.addItem(label, value)
        self.sort.setToolTip('Newest local annotation-file or annotation-folder modification first. '
                             'Imports and downloads can change these dates; this is not a shared edit history.')
        for combo in (self.experiment, self.sort):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
        if not compact:
            layout.addWidget(self.only_mine)
            form.addRow('Experiment / setup:', self.experiment)
            form.addRow('Sort:', self.sort)
            layout.addLayout(form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.hide()
        layout.addWidget(self.status)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(400)
        self.timer.timeout.connect(self.refresh)
        self.experiment.currentIndexChanged.connect(self.changed)
        self.sort.currentIndexChanged.connect(self.sort_changed)

    def set_context(self, project, videos, assignments):
        project, videos = str(project or ''), tuple(sorted(videos))
        if project != self.my_work_project:
            self.set_my_work(None, {})
        reset = project != self.project
        changed = reset or videos != self.videos
        selected = None if reset else self.experiment.currentData()
        self.project, self.videos = project, videos
        self.assignments = dict(assignments)
        names = sorted({assignments[v] for v in videos if assignments.get(v)}, key=str.casefold)
        self.experiment.blockSignals(True)
        self.experiment.clear()
        self.experiment.addItem('All experiments', None)
        self.experiment.addItem('No experiment assigned', '')
        for name in names:
            self.experiment.addItem(name, name)
        self.experiment.setCurrentIndex(max(0, self.experiment.findData(selected)))
        self.experiment.blockSignals(False)
        if reset:
            self.modified, self.errors, self.checked_times = {}, {}, False
        if changed:
            self.schedule()

    def accepts(self, video):
        selected = self.experiment.currentData()
        return ((not self.only_mine.isChecked() or video in self.my_work) and
                (selected is None or self.assignments.get(video, '') == selected))

    def matches_search(self, video):
        words = self.search.text().casefold().split()
        text = (video + ' ' + (self.assignments.get(video) or '')).casefold()
        return all(word in text for word in words)

    def active_filters(self):
        """Short descriptions of the filters that currently hide videos."""
        active = []
        if self.only_mine.isChecked():
            active.append('Only my assigned videos')
        if self.experiment.currentData() is not None:
            active.append('Experiment / setup: ' + self.experiment.currentText())
        return active

    def set_my_work(self, project, assignments, *, enabled=False):
        self.my_work_project = str(project or '')
        self.my_work = dict(assignments)
        self.only_mine.blockSignals(True)
        self.only_mine.setChecked(enabled)
        self.only_mine.setVisible(bool(project))
        self.only_mine.blockSignals(False)

    def key(self, video, default_order=0):
        mode = self.sort.currentData()
        name = (video.casefold(), video)
        if mode == 'recent':
            timestamp = self.modified.get(video)
            return (timestamp is None, -(timestamp or 0), *name)
        return name if mode == 'name' else (default_order, *name)

    def date_text(self, video):
        timestamp = self.modified.get(video)
        if timestamp is None:
            return 'Unknown' if self.checked_times else 'Not checked'
        return datetime.fromtimestamp(timestamp / 1e9).strftime('%Y-%m-%d %H:%M:%S')

    def tooltip(self, video):
        return ('Local annotation files modified: ' + self.date_text(video) +
                ('\n' + self.errors[video] if video in self.errors else ''))

    def sort_changed(self):
        self.schedule()
        self.changed.emit()

    def schedule(self):
        if not self.stopping and self.project and self.sort.currentData() == 'recent':
            self.timer.start()

    def refresh(self):
        if self.stopping or not self.project:
            return
        if self.worker:
            self.pending = True
            return
        self.status.setText('Reading local modification times...')
        self.status.show()
        project, videos = self.project, self.videos
        # Application ownership lets a closed/replaced panel cancel without
        # destroying a thread while a filesystem call is still in progress.
        worker = VideoMetadataWorker(lambda progress, cancelled:
            local_modification_times(project, videos, cancelled), QApplication.instance())
        worker.context = (project, videos)
        self.worker = worker
        self.destroyed.connect(worker.cancel)
        worker.finished.connect(self.loaded)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def loaded(self):
        worker, self.worker = self.worker, None
        if self.stopping:
            return
        if worker.context == (self.project, self.videos):
            if worker.error:
                self.modified, self.errors = {}, {}
                self.status.setText('Modification times unavailable: ' + str(worker.error))
            else:
                self.modified = worker.result['modified']
                self.errors = worker.result['errors']
                self.status.setText('Some local modification times are unavailable' if self.errors else '')
                self.status.setVisible(bool(self.errors))
            self.checked_times = True
            self.changed.emit()
        if self.pending or worker.context != (self.project, self.videos):
            self.pending = False
            self.schedule()

    def cancel(self):
        self.timer.stop()
        self.pending = False
        if self.worker:
            self.worker.requestInterruption()

    def stop(self):
        self.stopping = True
        self.cancel()
        return self.worker.wait(2000) if self.worker and self.worker.isRunning() else True
