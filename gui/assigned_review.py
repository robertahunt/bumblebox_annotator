"""Published-review views for locally available assigned videos."""

from copy import deepcopy
from datetime import datetime

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QStyle,
                            QToolButton, QVBoxLayout, QWidget)

from core.assigned_work import assigned_review_statuses
from gui.project_sync_dialog import setting_json
from gui.video_browsing import VideoMetadataWorker


class AssignedReviewViews(QWidget):
    changed = pyqtSignal()
    VIEWS = (('Active Work', 'active'), ('Awaiting Review', 'review'),
             ('Accepted', 'accepted'), ('All Assigned', 'all'))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.context = None
        self.assignments = {}
        self.statuses = {}
        self.checked_at = None
        self.error = None
        self.worker = None
        self.pending = False
        self.stopping = False
        self.visible_count = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel('Review:'))
        self.view = QComboBox()
        self.view.setAccessibleName('Assigned work review view')
        for label, value in self.VIEWS:
            self.view.addItem(label, value)
        self.view.setMinimumWidth(0)
        self.view.setMinimumContentsLength(12)
        self.view.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.view.setToolTip('Published review status, separate from Mark Finished. '
                             'Local edits do not change this status until uploaded. '
                             'Assignments and local files are kept in every view.')
        row.addWidget(self.view, 1)
        self.check = QToolButton()
        self.check.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.check.setAccessibleName('Refresh review status')
        self.check.setToolTip('Refresh published review status')
        row.addWidget(self.check)
        layout.addLayout(row)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.view.currentIndexChanged.connect(self.changed)
        self.check.clicked.connect(self.schedule)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.refresh)
        self.poll = QTimer(self)
        self.poll.setInterval(60000)
        self.poll.timeout.connect(lambda: self.schedule() if self.isVisible() else None)
        self.poll.start()
        self.hide()

    def set_context(self, project, contributor, assignments):
        configuration = setting_json('project_sync/destination', {})
        context = None
        if project and contributor and assignments:
            scope = tuple(sorted((v, a.get('id'), tuple(a['frames']), a.get('task'))
                                 for v, a in assignments.items()))
            context = (str(project), contributor.get('id'), contributor.get('session_id'),
                       configuration.get('root'), configuration.get('root_id'), scope)
        self.setVisible(context is not None)
        if context == self.context:
            return
        self.context = context
        self.project, self.contributor = str(project or ''), dict(contributor or {})
        self.configuration, self.assignments = configuration, deepcopy(assignments)
        self.statuses, self.checked_at, self.error = {}, None, None
        self.visible_count = None
        self.view.blockSignals(True)
        self.view.setCurrentIndex(0)
        self.view.blockSignals(False)
        self.timer.stop()
        if self.worker:
            self.worker.requestInterruption()
        self.pending = bool(context)
        if context:
            self.schedule()
        self.update_labels()

    @staticmethod
    def group(status):
        return {'Ready for Review': 'review', 'Accepted': 'accepted'}.get(status, 'active')

    def accepts(self, video):
        if not self.context:
            return True
        return video in self.assignments and (self.view.currentData() == 'all' or
            self.group(self.statuses.get(video)) == self.view.currentData())

    def status_text(self, video):
        if not self.context or video not in self.assignments:
            return ''
        return 'Published review: ' + self.statuses.get(video, 'Not checked')

    def update_labels(self, visible_count=None):
        if visible_count is not None:
            self.visible_count = visible_count
        counts = {value: sum(self.group(self.statuses.get(v)) == value for v in self.assignments)
                  for _, value in self.VIEWS}
        counts['all'] = len(self.assignments)
        self.view.blockSignals(True)
        for i, (label, value) in enumerate(self.VIEWS):
            count = str(counts[value]) if self.checked_at or value == 'all' else '?'
            self.view.setItemText(i, f'{label} ({count})')
        self.view.blockSignals(False)
        if self.error:
            text = 'Review status unavailable. Unchecked work stays in Active Work.'
        elif self.worker or self.timer.isActive():
            text = 'Checking published review status...'
        elif self.checked_at:
            text = 'Published review checked at ' + self.checked_at
        else:
            text = 'Published review not checked'
        if self.visible_count == 0 and self.context:
            label = dict((value, label) for label, value in self.VIEWS)[self.view.currentData()]
            text += f'\nNo videos in {label} match the current filters.'
        self.note.setText(text)
        self.note.setToolTip(self.error or 'Local edits are not included until uploaded. '
                            'Mark Finished records mask completion, not review approval.')

    def schedule(self):
        if not self.context or self.stopping:
            return
        if self.worker:
            self.pending = True
        else:
            self.timer.start(0)

    def refresh(self):
        if not self.context or self.stopping:
            return
        if self.worker:
            self.pending = True
            return
        project, actor = self.project, dict(self.contributor)
        configuration, assignments = dict(self.configuration), deepcopy(self.assignments)
        self.pending, self.error = False, None
        worker = VideoMetadataWorker(lambda progress, cancelled:
            assigned_review_statuses(project, configuration, actor, assignments, cancelled), QApplication.instance())
        worker.context = self.context
        self.worker = worker
        self.destroyed.connect(worker.cancel)
        worker.finished.connect(self.loaded)
        worker.finished.connect(worker.deleteLater)
        self.check.setEnabled(False)
        self.update_labels()
        worker.start()

    def loaded(self):
        worker, self.worker = self.worker, None
        if self.stopping:
            return
        self.check.setEnabled(True)
        if worker.context == self.context:
            self.statuses = {} if worker.error else worker.result
            self.error = str(worker.error) if worker.error else None
            self.checked_at = None if worker.error else datetime.now().strftime('%H:%M')
            self.changed.emit()
        if self.pending:
            self.pending = False
            self.schedule()
        self.update_labels()

    def stop(self):
        self.stopping = True
        self.timer.stop()
        self.poll.stop()
        if self.worker:
            self.worker.requestInterruption()
            return self.worker.wait(2000)
        return True
