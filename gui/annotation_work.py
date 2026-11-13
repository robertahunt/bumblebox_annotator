"""Shared work filters and a nonblocking annotation-sidebar inventory."""

from collections import OrderedDict

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                            QPushButton, QStyle, QToolButton, QVBoxLayout, QWidget)

from core.annotation_work import (TASKS, local_index, mark_finished, matches, summarize,
                                  task_categories, task_label)
from core.assigned_work import workspace
from gui.assigned_review import AssignedReviewViews
from gui.project_sync_dialog import OperationWorker, SyncProgressDialog
from gui.video_browsing import SORT_ROLE, STATE_NAMES, STATUS_ROLE, status_count, status_icon


def task_combo(parent=None):
    combo = QComboBox(parent)
    combo.addItem('All mask types', None)
    for task in ('hive', 'bee', 'bee_hive', 'pollen', 'nectar', 'chamber', 'brood', 'queen_brood'):
        combo.addItem(task_label(task), task)
    for task in TASKS:
        if task.startswith(('brood_', 'queen_brood_')):
            combo.addItem(task_label(task), task)
    combo.setMinimumWidth(0)
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(12)
    return combo


class WorkFilters(QWidget):
    """Work on, status and mask filters.

    sidebar=True: one "Work on" row, then a row for the caller's search box and a
    Filters button that opens every other filter (status, masks and any rows the
    caller adds to extra_form). The button shows how many filters are active.
    """
    changed = pyqtSignal()

    def __init__(self, parent=None, *, sidebar=False):
        super().__init__(parent)
        self.sidebar = sidebar
        self.external_filters = []  # callables returning descriptions of active filters
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.task = task_combo(self)
        if sidebar:
            self.task.setToolTip('The mask type you are working on. Status marks, counts and '
                                 'Mark Finished follow it.')
        self.show = QComboBox()
        for title, value in (('All Work', 'all'), ('Not Started', 'not_started'),
                             ('Not Checked', 'not_checked'), ('Finished', 'finished')):
            self.show.addItem(title, value)
        form.addRow('Work on:', self.task)
        if not sidebar:
            form.addRow('Show:', self.show)
        layout.addLayout(form)
        self.more = QToolButton()
        self.more.setText('Filters' if sidebar else 'More filters')
        self.more.setCheckable(True)
        self.more.setArrowType(Qt.ArrowType.RightArrow)
        self.more.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        if sidebar:
            self.more.setAccessibleName('Filters')
            self.search_row = QHBoxLayout()
            self.search_row.addWidget(self.more)
            layout.addLayout(self.search_row)
        else:
            layout.addWidget(self.more)
        self.extra = QWidget()
        extra = QFormLayout(self.extra)
        extra.setContentsMargins(12 if sidebar else 0, 0, 0, 0)
        self.extra_form = extra
        if sidebar:
            extra.addRow('Status:', self.show)
        self.presence = QComboBox()
        for title, value in (('Any' if sidebar else 'Any mask availability', 'any'), ('Has masks', 'has'),
                             ('No masks', 'none'), ('Some frames lack masks', 'some'),
                             ('Unknown' if sidebar else 'Unknown availability', 'unknown')):
            self.presence.addItem(title, value)
        extra.addRow('Masks:', self.presence)
        self.split = QComboBox()
        self.split.addItem('All splits', None)
        for split in ('train', 'val', 'test', 'inference'):
            self.split.addItem(split.title(), split)
        extra.addRow('Split:', self.split)
        for combo in (self.show, self.presence, self.split) if sidebar else ():
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
            combo.setMinimumWidth(0)
        layout.addWidget(self.extra)
        self.extra.hide()
        def toggle(visible):
            self.extra.setVisible(visible)
            self.more.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self.more.toggled.connect(toggle)
        for combo in (self.task, self.show, self.presence, self.split):
            combo.currentIndexChanged.connect(self.changed)
        self.changed.connect(self.update_filter_label)

    def active_filters(self):
        """Short descriptions of the filters in the Filters panel that hide work."""
        active = []
        if self.sidebar and self.show.currentData() != 'all':
            active.append('Status: ' + self.show.currentText())
        if self.presence.currentData() != 'any':
            active.append('Masks: ' + self.presence.currentText())
        if self.split.currentData() is not None:
            active.append('Split: ' + self.split.currentText())
        for describe in self.external_filters:
            active.extend(describe())
        return active

    def update_filter_label(self):
        active = self.active_filters()
        if not self.sidebar:
            self.more.setText(f'More filters ({len(active)})' if active else 'More filters')
            return
        self.more.setText(f'Filters ({len(active)})' if active else 'Filters')
        self.more.setToolTip('Active filters:\n' + '\n'.join(active) if active else
                             'Status, masks, split, experiment and sort order')

    def accepts(self, summary, split=None):
        return (self.split.currentData() in (None, split) and
                (summary['state'] == 'loading' or
                 matches(summary, self.show.currentData(), self.presence.currentData())))


class SummaryWorker(OperationWorker):
    video_ready = pyqtSignal(str, object)

    def __init__(self, project, videos, task, previous, parent):
        super().__init__(lambda progress, cancelled: local_index(
            project, videos=videos, task=task, previous=previous, cancelled=cancelled,
            video_ready=self.video_ready.emit), parent)


class AnnotationWorkBrowser(QObject):
    def __init__(self, window, layout):
        super().__init__(window)
        self.window = window
        browser = window.video_browser
        self.review = AssignedReviewViews()
        self.review.busy = lambda: bool(getattr(window.canvas, 'is_drawing', False))
        layout.insertWidget(layout.indexOf(window.video_list), self.review)
        self.review.changed.connect(self.apply)
        # Work on stays visible; every other filter lives behind one Filters button.
        self.filters = filters = WorkFilters(sidebar=True)
        form = filters.extra_form
        form.setRowVisible(filters.split, False)  # The window's per-video split filter replaces it.
        filters.search_row.insertWidget(0, browser.search, 1)
        form.insertRow(1, window.video_filter_label, window.video_split_filter_combo)
        form.addRow('Experiment / setup:', browser.experiment)
        form.addRow('Sort:', browser.sort)
        form.addRow(browser.only_mine)
        form.addRow(window.tracking_filter_checkbox)
        self.clear_filters_button = QPushButton('Clear Filters')
        self.clear_filters_button.setToolTip('Show all videos again (search text is kept)')
        self.clear_filters_button.clicked.connect(self.clear_filters)
        clear_row = QHBoxLayout()
        clear_row.addWidget(self.clear_filters_button)
        clear_row.addStretch()
        form.addRow(clear_row)
        filters.external_filters.append(self.sidebar_filters)
        layout.insertWidget(layout.indexOf(window.video_list), filters)
        # List-level notes only (checking progress, filtered-out current video); hidden when empty.
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.hide()
        layout.insertWidget(layout.indexOf(window.video_list) + 1, self.note)
        # The current frame's status and Mark Finished; the window puts this in the Frames panel.
        self.frame_controls = QWidget()
        frame_layout = QVBoxLayout(self.frame_controls)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_note = QLabel('Local saved annotations')
        self.frame_note.setWordWrap(True)
        frame_layout.addWidget(self.frame_note)
        row = QHBoxLayout()
        self.finish = QPushButton('Mark Finished')
        self.finish.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.finish.clicked.connect(self.complete)
        row.addWidget(self.finish, 1)
        self.next = QToolButton()
        self.next.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))
        self.next.setToolTip('Next unfinished frame in this video')
        self.next.setAccessibleName('Next unfinished frame')
        self.next.clicked.connect(self.next_frame)
        row.addWidget(self.next)
        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(window.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip('Refresh saved mask summaries')
        self.refresh_button.setAccessibleName('Refresh saved mask summaries')
        self.refresh_button.clicked.connect(self.schedule)
        self.refresh_button.clicked.connect(window.video_browser.schedule)
        self.refresh_button.clicked.connect(
            lambda: window._schedule_done_counter_update(saved_counts_dirty=True))
        row.addWidget(self.refresh_button)
        frame_layout.addLayout(row)
        self.filters.changed.connect(self.apply)
        self.worker = None
        self.project = None
        self.index = {'version': 1, 'videos': {}}
        self.checked = False
        self.error = None
        self.pending = False
        self.stopping = False
        self.video = None
        self.request = None
        self.scanned_videos = set()
        self._summaries = OrderedDict()
        self.apply_timer = QTimer(self)
        self.apply_timer.setSingleShot(True)
        self.apply_timer.setInterval(100)
        self.apply_timer.timeout.connect(self.apply)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.refresh)
        self.apply()

    def sidebar_filters(self):
        w = self.window
        active = []
        if getattr(w, 'video_split_filter', 'all') != 'all':
            active.append('Split: ' + w.video_split_filter_combo.currentText())
        active.extend(w.video_browser.active_filters())
        if w.tracking_filter_checkbox.isChecked():
            active.append('Only videos with tracking sequences')
        return active

    def clear_filters(self):
        w, filters = self.window, self.filters
        for combo in (filters.show, filters.presence, filters.split,
                      w.video_browser.experiment, w.video_split_filter_combo):
            combo.setCurrentIndex(0)
        w.tracking_filter_checkbox.setChecked(False)
        w.video_browser.only_mine.setChecked(False)
        self.apply()

    def schedule(self):
        if not self.stopping:
            self.timer.start()

    def scan_context(self):
        w = self.window
        project = str(w.project_path or '')
        review = getattr(w, 'submission_review_dock', None)
        if review and review.is_focused():
            videos = (review.active['video'],) if review.active else ()
            return project, self.filters.task.currentData(), videos
        state = workspace(project) if project else None
        videos = []
        for row in range(w.video_list.count()):
            video = w.video_list.item(row).data(Qt.ItemDataRole.UserRole)
            if video and w.video_browser.accepts(video) and (
                    not state or video in state['assignments']):
                videos.append(video)
        return project, self.filters.task.currentData(), tuple(sorted(set(videos)))

    def check_context(self):
        request = self.scan_context()
        if request == self.request:
            return
        self.request = request
        self.checked, self.error = False, None
        self.scanned_videos = set()
        if request[0] != self.project:
            self._summaries.clear()
            self.project = request[0]
            self.index = {'version': 1, 'videos': {}}
        if self.worker:
            self.worker.requestInterruption()
        self.schedule()

    def refresh(self):
        self.check_context()
        self.timer.stop()
        project, task, videos = self.request
        if not project or self.stopping:
            return
        if self.worker:
            self.pending = True
            return
        previous = self.index
        request = self.request
        current = self.window.current_video_id
        videos = sorted(videos, key=lambda v: (v != current, v))
        self.error = None
        self.scanned_videos = set()
        self.worker = SummaryWorker(project, videos, task, previous, self)
        self.worker.video_ready.connect(lambda video, entry: self.video_loaded(request, video, entry))
        self.worker.finished.connect(lambda: self.loaded(request))
        self.apply()
        self.worker.start()

    def video_loaded(self, request, video, entry):
        if (self.stopping or request != self.request or
                request[0] != str(self.window.project_path or '')):
            return
        self.index = dict(self.index, videos=dict(self.index['videos'], **{video: entry}))
        self.scanned_videos.add(video)
        # Batch visual updates without withholding completed videos until the end.
        if not self.apply_timer.isActive():
            self.apply_timer.start()

    def loaded(self, request):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        if self.stopping:
            return
        if request == self.scan_context():
            if not worker.error:
                self.index = dict(self.index, videos=dict(self.index['videos'], **worker.result['videos']))
            self.checked = True
            self.error = str(worker.error) if worker.error else None
            self.apply()
        if self.pending or request != self.scan_context():
            self.pending = False
            self.schedule()

    def current(self):
        video = self.window.current_video_id
        frame = self.window._get_frame_idx_in_video(self.window.current_frame_idx) if video else None
        entry = self.index['videos'].get(video)
        return video, frame, entry

    def context_changed(self):
        submission_review = getattr(self.window, 'submission_review_dock', None)
        if submission_review:
            submission_review.update_buttons()
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
        missing = entry is None or not set(task_categories(task)) <= set(entry.get('categories', {}))
        if missing and self.window.project_path and (
                not self.checked or self.worker or self.timer.isActive()):
            return {'presence': 'loading', 'state': 'loading', 'mask_text': 'Loading...',
                    'work_text': 'Loading...', 'detail': 'Checking local saved masks...'}
        # local_index reuses immutable entries when their source files are unchanged.
        # Retain the entry with each cache item so Python cannot recycle its id.
        key = (id(entry), task, None if frames is None else tuple(frames))
        cached = self._summaries.get(key) if entry is not None else None
        if cached is not None and cached[0] is entry:
            self._summaries.move_to_end(key)
            result = dict(cached[1])
        else:
            result = summarize(entry, task, frames)
            if entry is not None:
                self._summaries[key] = (entry, dict(result))
                if len(self._summaries) > 4096:
                    self._summaries.popitem(last=False)
        if entry is None and self.error:
            result['detail'] = self.error
        return result

    @staticmethod
    def summary_text(summary):
        return ('Loading...' if summary['state'] == 'loading' else
                summary['mask_text'] + ' | ' + summary['work_text'])

    @staticmethod
    def frame_text(summary):
        """The current frame's status in plain words, e.g. 'This frame: Has masks · Not checked'."""
        if summary['state'] == 'loading':
            return 'This frame: Loading...'
        masks = summary['mask_text']
        if summary.get('total') == 1 and masks.startswith('Masks on'):
            masks = 'Has masks'
        return 'This frame: ' + masks + ' · ' + summary['work_text']

    def apply(self):
        if self.stopping:
            return
        w = self.window
        if getattr(w.canvas, 'is_drawing', False):
            self.apply_timer.start()
            return
        self.apply_timer.stop()
        self.check_context()
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
            assigned_frames = assignments.get(video, {}).get('frames')
            summary = self.summary(entry, task, assigned_frames)
            base = item.data(Qt.ItemDataRole.UserRole + 40)
            if base is None:
                base = item.text()
                item.setData(Qt.ItemDataRole.UserRole + 40, base)
            # The full text stays the item's text (screen readers, tooltips); the row
            # delegate paints one line from SPLIT_ROLE and STATUS_ROLE.
            status = task_label(task) + ': ' + self.summary_text(summary)
            item.setText(base + '\n' + status)
            review_text = self.review.status_text(video)
            if review_text:
                item.setText(item.text() + '\n' + review_text)
            count = status_count(summary)
            item.setData(STATUS_ROLE, {'state': summary['state'], 'count': count})
            item.setToolTip(base + '\n' + status +
                            (f'\nFinished frames: {count}' if count else '') +
                            '\nLocal saved annotations: ' + summary.get('detail', '') +
                            '\n' + w.video_browser.tooltip(video) + ('\n' + review_text if review_text else ''))
            # Mask filters retain the active item; experiment/review/search filters are exact.
            # Neither changes the video currently displayed on the canvas.
            visible = (w.video_browser.accepts(video) and w.video_browser.matches_search(video) and
                       self.review.accepts(video) and (
                summary['state'] == 'loading' or video == w.current_video_id or self.filters.accepts(summary)))
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
        self.frame_note.setText('This frame: unsaved edits; showing saved status' if dirty else
                                'Choose a video to see its saved work' if not video else
                                self.frame_text(summary))
        notes = []
        if self.worker or summary['state'] == 'loading':
            progress = f' {len(self.scanned_videos)}/{len(self.request[2])} videos' if self.worker else ''
            notes.append('Checking local saved masks...' + progress)
        elif self.error:
            notes.append('Mask summary unavailable: ' + self.error)
        if video and not (w.video_browser.accepts(video) and w.video_browser.matches_search(video)
                          and self.review.accepts(video)):
            notes.append('Current video is outside the video filters')
        elif video and summary['state'] != 'loading' and not self.filters.accepts(self.summary(entry, task)):
            notes.append('Current video kept visible outside these filters')
        self.note.setText('\n'.join(notes))
        self.note.setVisible(bool(notes))
        self.filters.update_filter_label()
        available = summary['state'] not in ('unknown', 'loading', 'not_scanned')
        self.finish.setEnabled(bool(video and entry and task and available))
        self.finish.setText('Reopen Work' if not dirty and summary.get('finished', 0) == summary.get('total', -1)
                            else 'Mark Finished')
        self._finish_tip = self.completion_tip(summary, state, task, video)
        self.finish.setToolTip(self.completion_tip(summary, state, task, video, reopening=True)
                               if self.finish.text() == 'Reopen Work' else self._finish_tip)
        self.next.setEnabled(bool(video and entry and task and available))
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
                # Before a video is chosen the list holds frames from every video.
                frame_video = w.frame_video_ids[index] if index < len(w.frame_video_ids) else video
                frame_entry = entry if frame_video == video else self.index['videos'].get(frame_video)
                details = self.summary(frame_entry, task, [number])
                video_name = item.data(Qt.ItemDataRole.UserRole + 44)  # only when several videos are listed
                item.setText(base + ' · ' + details['work_text'] + (' · ' + video_name if video_name else ''))
                item.setIcon(status_icon(details['state']))
                note = item.data(Qt.ItemDataRole.UserRole + 43)
                item.setToolTip(STATE_NAMES.get(details['state'], details['work_text']) + ': ' +
                                details['mask_text'] + '\n' + details.get('detail', '') +
                                ('\n' + note if note else ''))

    def mark_dirty(self):
        self.frame_note.setText('This frame: unsaved edits; showing saved status')
        self.finish.setText('Mark Finished')
        self.finish.setToolTip(self._finish_tip)

    def completion_tip(self, summary, state, task, video, *, reopening=False):
        if task is None:
            return 'Choose a mask type in Work on before marking it finished.'
        if not video:
            return 'Choose a video and frame before marking work finished.'
        if summary['state'] == 'loading':
            return f'Checking saved {task_label(task)} status. You can keep annotating while it loads.'
        if summary['state'] in ('unknown', 'not_scanned'):
            return ('Saved mask status must be available before marking work finished.\n'
                    'Refresh mask summaries and resolve any missing or unsupported annotation data.')
        scope = 'the whole video (shared mask)' if summary.get('shared') else 'the current frame'
        text = (f'Mark {task_label(task)} unfinished for {scope}.' if reopening else
                f'Record that you checked {task_label(task)} for {scope}, including if none is present.')
        if summary.get('mixed_scope'):
            text += '\n' + summary['detail'] + '. Shared-mask completion applies to the whole video.'
        if state:
            text += '\nSaved changes are queued for automatic upload; offline work waits for a connection.'
            if state.get('assignments', {}).get(video, {}).get('task'):
                text += ('\nWhen all required work for the assigned task is finished,\n'
                         'a verified upload makes this video ready for director review.')
            else:
                text += '\nA director must set an assigned mask task before automatic review submission is available.'
        else:
            text += '\nThis local project is not an assigned working copy; completion does not automatically upload or submit it.'
        text += '\nImported-frame approval and director acceptance are separate.'
        return text

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
                local_index(w.project_path, video=video, task=task, cancelled=cancelled), w).run()['videos'].get(video)
            summary = summarize(entry, task, [frame])
            text = (f'Mark {task_label(task)} work unfinished?' if reopen else
                    f'Mark {task_label(task)} work finished?' if summary['presence'] != 'none' else
                    f'Confirm that you checked and no {task_label(task).lower()} is present?')
            if summary.get('shared'):
                text += '\n\nThis records the shared mask for the whole video, not just this frame.'
            elif summary.get('mixed_scope'):
                text += '\n\n' + summary['detail'] + '. Shared-mask completion applies to the whole video.'
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
                w.workflow.automatic_work.saved(completed=True)
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
        self.apply_timer.stop()
        self._summaries.clear()
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait()
