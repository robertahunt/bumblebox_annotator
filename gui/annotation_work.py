"""Shared work filters and a nonblocking annotation-sidebar inventory."""

from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                            QPushButton, QStyle, QToolButton, QVBoxLayout, QWidget)

from core.annotation_work import (TASKS, local_index, mark_finished, matches, summarize,
                                  task_label)
from core.assigned_work import workspace
from gui.assigned_review import AssignedReviewViews
from gui.project_sync_dialog import OperationWorker, SyncProgressDialog
from gui.video_browsing import SORT_ROLE


def task_combo(parent=None):
    combo = QComboBox(parent)
    combo.addItem('All mask types', None)
    for task in ('hive', 'bee', 'pollen', 'nectar', 'chamber', 'brood', 'queen_brood'):
        combo.addItem(task_label(task), task)
    for task in TASKS:
        if task.startswith(('brood_', 'queen_brood_')):
            combo.addItem(task_label(task), task)
    combo.setMinimumWidth(0)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    return combo


class WorkFilters(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.task = task_combo(self)
        self.show = QComboBox()
        for title, value in (('All Work', 'all'), ('Not Started', 'not_started'),
                             ('Not Checked', 'not_checked'), ('Finished', 'finished')):
            self.show.addItem(title, value)
        form.addRow('Work on:', self.task)
        form.addRow('Show:', self.show)
        layout.addLayout(form)
        self.more = QToolButton()
        self.more.setText('More filters')
        self.more.setCheckable(True)
        self.more.setArrowType(Qt.ArrowType.RightArrow)
        self.more.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout.addWidget(self.more)
        self.extra = QWidget()
        extra = QFormLayout(self.extra)
        extra.setContentsMargins(0, 0, 0, 0)
        self.presence = QComboBox()
        for title, value in (('Any mask availability', 'any'), ('Has masks', 'has'), ('No masks', 'none'),
                             ('Some frames lack masks', 'some'), ('Unknown availability', 'unknown')):
            self.presence.addItem(title, value)
        extra.addRow('Masks:', self.presence)
        self.split = QComboBox()
        self.split.addItem('All splits', None)
        for split in ('train', 'val', 'test', 'inference'):
            self.split.addItem(split.title(), split)
        extra.addRow('Split:', self.split)
        layout.addWidget(self.extra)
        self.extra.hide()
        def toggle(visible):
            self.extra.setVisible(visible)
            self.more.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self.more.toggled.connect(toggle)
        for combo in (self.task, self.show, self.presence, self.split):
            combo.currentIndexChanged.connect(self.changed)
        self.changed.connect(self.update_filter_label)

    def update_filter_label(self):
        active = int(self.presence.currentData() != 'any') + int(self.split.currentData() is not None)
        self.more.setText(f'More filters ({active})' if active else 'More filters')

    def accepts(self, summary, split=None):
        return (self.split.currentData() in (None, split) and
                (summary['state'] == 'loading' or
                 matches(summary, self.show.currentData(), self.presence.currentData())))


class AnnotationWorkBrowser(QObject):
    def __init__(self, window, layout):
        super().__init__(window)
        self.window = window
        self.source = QLabel()
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.source.setWordWrap(True)
        layout.insertWidget(0, self.source)
        self.review = AssignedReviewViews()
        layout.insertWidget(layout.indexOf(window.video_list), self.review)
        self.review.changed.connect(self.apply)
        self.filters = WorkFilters()
        self.filters.split.hide()  # The existing sidebar already has a split selector.
        self.filters.extra.layout().labelForField(self.filters.split).hide()
        layout.insertWidget(layout.indexOf(window.video_list), self.filters)
        self.note = QLabel('Local saved annotations')
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        row = QHBoxLayout()
        self.finish = QPushButton('Mark Finished')
        self.finish.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.finish.clicked.connect(self.complete)
        row.addWidget(self.finish, 1)
        self.next = QToolButton()
        self.next.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self.next.setToolTip('Next unfinished frame in this video')
        self.next.clicked.connect(self.next_frame)
        row.addWidget(self.next)
        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip('Refresh saved mask summaries')
        self.refresh_button.clicked.connect(self.schedule)
        self.refresh_button.clicked.connect(window.video_browser.schedule)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        self.filters.changed.connect(self.apply)
        self.worker = None
        self.project = None
        self.index = {'version': 1, 'videos': {}}
        self.checked = False
        self.error = None
        self.pending = False
        self.stopping = False
        self.video = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh)
        self.apply()

    def schedule(self):
        if not self.stopping:
            self.timer.start()

    def refresh(self):
        project = str(self.window.project_path or '')
        if not project or self.stopping:
            return
        if self.worker:
            self.pending = True
            return
        previous = self.index if project == self.project else None
        if project != self.project:
            self.project, self.index = project, {'version': 1, 'videos': {}}
            self.checked = False
        self.error = None
        self.worker = OperationWorker(lambda progress, cancelled:
            local_index(project, previous=previous, cancelled=cancelled), self)
        self.worker.finished.connect(lambda: self.loaded(project))
        self.apply()
        self.worker.start()

    def loaded(self, project):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        if self.stopping:
            return
        if project == str(self.window.project_path or ''):
            self.index = worker.result if not worker.error else {'version': 1, 'videos': {}}
            self.checked = True
            self.error = str(worker.error) if worker.error else None
            self.apply()
        if self.pending or project != str(self.window.project_path or ''):
            self.pending = False
            self.schedule()

    def current(self):
        video = self.window.current_video_id
        frame = self.window._get_frame_idx_in_video(self.window.current_frame_idx) if video else None
        entry = self.index['videos'].get(video)
        return video, frame, entry

    def context_changed(self):
        video = self.window.current_video_id
        if str(self.window.project_path or '') != self.project:
            self.index = {'version': 1, 'videos': {}}
            self.checked = False
            self.error = None
        if self.video != (str(self.window.project_path), video):
            self.video = (str(self.window.project_path), video)
            state = workspace(self.window.project_path)
            task = (state or {}).get('assignments', {}).get(video, {}).get('task')
            browser = self.window.video_browser
            if not state and browser.only_mine.isChecked():
                task = browser.my_work.get(video, {}).get('task')
            if task:
                self.filters.task.setCurrentIndex(self.filters.task.findData(task))
        self.apply()

    def summary(self, entry, task, frames=None):
        if entry is None and self.window.project_path and (
                not self.checked or self.worker or self.timer.isActive()):
            return {'presence': 'loading', 'state': 'loading', 'mask_text': 'Loading...',
                    'work_text': 'Loading...', 'detail': 'Checking local saved masks...'}
        result = summarize(entry, task, frames)
        if entry is None and self.error:
            result['detail'] = self.error
        return result

    @staticmethod
    def summary_text(summary):
        return ('Loading...' if summary['state'] == 'loading' else
                summary['mask_text'] + ' | ' + summary['work_text'])

    def apply(self):
        w = self.window
        self.source.setText('Local working copy\n' + Path(w.project_path).name if w.project_path else 'No local project open')
        self.source.setToolTip(str(w.project_path or ''))
        task = self.filters.task.currentData()
        state = workspace(w.project_path) if w.project_path else None
        assignments = state['assignments'] if state else (
            w.video_browser.my_work if w.video_browser.only_mine.isChecked() else {})
        self.review.set_context(w.project_path, getattr(w, 'contributor_session', None), assignments)
        visible_count = 0
        for row in range(w.video_list.count()):
            item = w.video_list.item(row)
            video = item.data(Qt.ItemDataRole.UserRole)
            if not video:
                continue
            entry = self.index['videos'].get(video)
            assigned_frames = (state or {}).get('assignments', {}).get(video, {}).get('frames')
            summary = self.summary(entry, task, assigned_frames)
            base = item.data(Qt.ItemDataRole.UserRole + 40)
            if base is None:
                base = item.text()
                item.setData(Qt.ItemDataRole.UserRole + 40, base)
            item.setText(base + '\n' + task_label(task) + ': ' + self.summary_text(summary))
            review_text = self.review.status_text(video)
            if review_text:
                item.setText(item.text() + '\n' + review_text)
            item.setToolTip(base + '\nLocal saved annotations\n' + summary.get('detail', '') +
                            '\n' + w.video_browser.tooltip(video) + ('\n' + review_text if review_text else ''))
            # Mask filters retain the active item; experiment/review filters are exact.
            # Neither changes the video currently displayed on the canvas.
            visible = w.video_browser.accepts(video) and self.review.accepts(video) and (
                summary['state'] == 'loading' or video == w.current_video_id or self.filters.accepts(summary))
            item.setHidden(not visible)
            visible_count += int(visible)
            if not visible:
                item.setSelected(False)
            item.setData(SORT_ROLE, w.video_browser.key(video, item.data(Qt.ItemDataRole.UserRole + 42) or 0))
        w.video_list.sortItems(Qt.SortOrder.AscendingOrder)
        self.review.update_labels(visible_count)
        video, frame, entry = self.current()
        summary = self.summary(entry, task, [frame])
        dirty = bool(getattr(w, 'current_frame_modified', False))
        self.note.setText('Unsaved edits; showing saved status' if dirty else
                          'Local saved annotations' if not video else self.summary_text(summary))
        if self.worker or summary['state'] == 'loading':
            self.note.setText(self.note.text() + '\nChecking local saved masks...')
        elif self.error:
            self.note.setText('Mask summary unavailable: ' + self.error)
        if video and not (w.video_browser.accepts(video) and self.review.accepts(video)):
            self.note.setText(self.note.text() + '\nCurrent video is outside the video filters')
        elif video and summary['state'] != 'loading' and not self.filters.accepts(summarize(entry, task)):
            self.note.setText(self.note.text() + '\nCurrent video kept visible outside these filters')
        self.finish.setEnabled(bool(video and entry and task and summary['state'] not in ('unknown', 'loading')))
        self.finish.setText('Reopen Work' if not dirty and summary.get('finished', 0) == summary.get('total', -1)
                            else 'Mark Finished')
        self.finish.setToolTip('Records completion for the selected mask type. Shared masks are recorded once per video. '
                               'Does not approve imported frames or submit work to the director.')
        self.next.setEnabled(bool(video and entry and task))
        if hasattr(w, 'frame_list'):
            for row, index in enumerate(w.frame_list_to_frames_map):
                item = w.frame_list.item(row)
                if item is None:
                    continue
                base = item.data(Qt.ItemDataRole.UserRole + 40)
                if base is None:
                    base = item.text()
                    item.setData(Qt.ItemDataRole.UserRole + 40, base)
                number = w._get_frame_idx_in_video(index)
                details = self.summary(entry, task, [number])
                item.setText(base + ' | ' + details['work_text'])
                item.setToolTip(details['mask_text'] + '\n' + details.get('detail', ''))

    def complete(self):
        w = self.window
        task = self.filters.task.currentData()
        video, frame, entry = self.current()
        try:
            if not task or not video:
                return
            reopen = self.finish.text() == 'Reopen Work'
            if not w._save_current_frame_annotations_blocking():
                return
            # Re-read after the save: the last brush stroke may have changed presence.
            entry = SyncProgressDialog('Check Saved Annotation Work', lambda progress, cancelled:
                local_index(w.project_path, video=video, cancelled=cancelled), w).run()['videos'].get(video)
            summary = summarize(entry, task, [frame])
            text = (f'Mark {task_label(task)} work unfinished?' if reopen else
                    f'Mark {task_label(task)} work finished?' if summary['presence'] != 'none' else
                    f'Confirm that you checked and no {task_label(task).lower()} is present?')
            if summary.get('shared'):
                text += '\n\nThis records the shared mask for the whole video, not just this frame.'
            if summary['state'] == 'needs_review':
                text += '\n\nImported-frame reviews remain required separately.'
            if QMessageBox.question(w, 'Annotation Work', text,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
                return
            SyncProgressDialog('Save Annotation Completion', lambda progress, cancelled:
                mark_finished(w.project_path, video, frame, task, w.contributor_session,
                              complete=not reopen, cancelled=cancelled), w).run()
            if hasattr(w, 'workflow'):
                w.workflow.saved()
            self.schedule()
        except Exception as exc:
            QMessageBox.warning(w, 'Completion Not Saved', str(exc))

    def next_frame(self):
        w = self.window
        _, _, entry = self.current()
        task = self.filters.task.currentData()
        indices = w.frame_list_to_frames_map
        ordered = [i for i in indices if i > w.current_frame_idx] + [i for i in indices if i < w.current_frame_idx]
        for index in ordered:
            summary = summarize(entry, task, [w._get_frame_idx_in_video(index)])
            if summary['state'] != 'finished':
                w.frame_list.setCurrentRow(indices.index(index))
                return
        w.status_label.setText('No other unfinished frames in the current frame list')

    def stop(self):
        self.stopping = True
        self.timer.stop()
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait()
