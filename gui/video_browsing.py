"""Experiment filters and asynchronous local modification-time sorting."""

from datetime import datetime

from PyQt6.QtCore import QEvent, QObject, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (QApplication, QComboBox, QFormLayout, QLabel,
                            QListWidgetItem, QTableWidgetItem, QVBoxLayout, QWidget)

from core.video_browsing import local_modification_times
from gui.project_sync_dialog import OperationWorker


SORT_ROLE = Qt.ItemDataRole.UserRole + 41


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
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = ''
        self.videos = ()
        self.assignments = {}
        self.modified = {}
        self.errors = {}
        self.checked_times = False
        self.worker = None
        self.pending = False
        self.stopping = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
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
        return selected is None or self.assignments.get(video, '') == selected

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
