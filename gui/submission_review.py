"""A focused director queue alongside the normal annotation editor."""

from contextlib import contextmanager
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QDockWidget, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QStyle, QToolButton,
    QVBoxLayout, QWidget)

from core.annotation_work import task_categories, task_label
from core.project_sync import require_no_update, update_journal_path
from core.submission_review import PreparedSubmissionReview, approve_review, review_queue
from gui.project_sync_dialog import SyncProgressDialog, setting_json
from gui.video_browsing import VideoRowColors


class SubmissionReviewDock(QDockWidget):
    def __init__(self, window):
        super().__init__('Review Submissions', window)
        self.setObjectName('submissionReviewDock')
        self.window = window
        self.context = None
        self.active = None
        self.rows = []
        self.busy = False
        self.queue_valid = False
        self.view_before = None
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        body = QWidget()
        layout = QVBoxLayout(body)
        header = QHBoxLayout()
        self.count = QLabel()
        header.addWidget(self.count, 1)
        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip('Refresh submitted work')
        self.refresh_button.setAccessibleName('Refresh submitted work')
        self.refresh_button.clicked.connect(self.refresh)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(0)
        layout.addWidget(self.status)
        self.queue = QListWidget()
        self.queue.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.queue.setWordWrap(True)
        self.queue.setMinimumWidth(240)
        VideoRowColors(self.queue)
        self.queue.itemSelectionChanged.connect(self.update_buttons)
        self.queue.itemClicked.connect(lambda item: self.open_selected())
        self.queue.itemActivated.connect(lambda item: self.open_selected())
        layout.addWidget(self.queue, 1)
        self.approve_button = QPushButton('Approve && Next')
        self.approve_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.approve_button.setToolTip('Save corrections, publish this reviewed submission, and open the next video')
        self.approve_button.clicked.connect(self.approve)
        layout.addWidget(self.approve_button)
        self.exit_button = QPushButton('Exit Review')
        self.exit_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.exit_button.clicked.connect(self.close)
        layout.addWidget(self.exit_button)
        self.setWidget(body)
        self.hide()
        self.update_buttons()

    def is_focused(self):
        return self.view_before is not None and self.context == self.current_context()

    def start(self):
        if not self.refresh():
            QMessageBox.warning(self.window, 'Review Queue Unavailable', self.status.text())
            return
        if not self.rows:
            self.close()
            QMessageBox.information(self.window, 'Submitted Work', 'No submissions are awaiting review.')
            return
        self.open_selected()

    def _enter_focus(self):
        w = self.window
        if self.view_before is None:
            editor = w.workflow.current in ('Annotate', 'Review')
            self.view_before = {'workspace': w.workflow.current, 'video': w.current_video_id,
                'frame': w._get_frame_idx_in_video(w.current_frame_idx) if w.frames else None,
                'frame_filter': w.split_filter_combo.currentIndex(), 'split': w.split_filter,
                'task': w.annotation_work.filters.task.currentIndex(),
                'docks': {dock: (not dock.isHidden() if editor else w.workflow.dock_visibility.get(dock, False))
                          for dock in (w.video_dock, w.frame_dock)}}
        w.workflow.activate('Annotate')
        w.workflow.dock_visibility.update({w.video_dock: False, w.frame_dock: True, self: True})
        w.video_dock.hide()
        w.toggle_video_sidebar_action.setEnabled(False)
        w.frame_dock.show()
        self.show()
        self.raise_()

    def _restore_browsing(self, *, reload=True):
        w = self.window
        before, self.view_before = self.view_before, None
        self.active = None
        self.hide()
        w.workflow.dock_visibility[self] = False
        w.toggle_video_sidebar_action.setEnabled(True)
        if before is None:
            return
        same_context = self.context == self.current_context()
        if same_context:
            w.split_filter_combo.blockSignals(True)
            w.split_filter_combo.setCurrentIndex(before['frame_filter'])
            w.split_filter_combo.blockSignals(False)
            w.split_filter = before['split']
            w.annotation_work.filters.task.setCurrentIndex(before['task'])
            if reload and w.project_path and before['video']:
                w.load_video_frames(before['video'])
                for index in range(len(w.frames)):
                    if w._get_frame_idx_in_video(index) == before['frame']:
                        w.load_frame(index)
                        break
            w.workflow.activate(before['workspace'])
        # Leaving the editor records actual dock visibility; restore preferences
        # afterward so Home does not remember the temporary review-only layout.
        w.workflow.dock_visibility.update(before['docks'])
        w.workflow.dock_visibility[self] = False
        for dock, visible in before['docks'].items():
            dock.setVisible(visible and w.workflow.current in ('Annotate', 'Review'))
        w.annotation_work.check_context()

    def _select_active(self):
        self.queue.setCurrentRow(next((i for i, row in enumerate(self.rows) if row == self.active), -1))

    def _describe_active(self):
        if self.active:
            row = self.active
            state = 'Reviewing' if row in self.rows else 'Submission no longer awaiting review'
            self.status.setText(f'{state}: {row["video"]}\n{row["contributor"]["name"]} | '
                                f'{task_label(row.get("task"))} | {len(row["frames"])} frames')
        else:
            self.status.setText('No submission open')

    def current_context(self):
        w = self.window
        configuration = setting_json('project_sync/destination', {})
        return (str(w.project_path or ''), (w.contributor_session or {}).get('id'), configuration.get('root_id'))

    def context_changed(self):
        if self.context != self.current_context():
            self._restore_browsing(reload=False)
            self.rows = []
            self.queue.clear()
            self.context = None
            self.queue_valid = False
            self.hide()
            self.window.workflow.dock_visibility[self] = False
        self.update_buttons()

    def review_assignment(self, video):
        if (self.active and self.context == self.current_context() and self.active['video'] == video):
            return self.active
        return None

    def selected(self):
        row = self.queue.currentRow()
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def update_buttons(self):
        row = self.selected()
        current = self.context == self.current_context()
        enabled = current and self.queue_valid and not self.busy and not self.window._sync_in_progress
        self.queue.setEnabled(enabled)
        self.approve_button.setEnabled(bool(enabled and row and row == self.active and
                                           self.window.current_video_id == row['video'] and self.window.frames and
                                           self.window.canvas.image_item is not None))
        self.refresh_button.setEnabled(not self.busy and not self.window._sync_in_progress)
        self.exit_button.setEnabled(not self.busy and not self.window._sync_in_progress)

    def refresh(self, video=None):
        if self.busy or self.window._sync_in_progress:
            return False
        self.queue_valid = False
        self.update_buttons()
        context = self.current_context()
        if context != self.context:
            self.context_changed()
        old = video if isinstance(video, str) else (self.active or self.selected() or {}).get('video')
        try:
            configuration = dict(setting_json('project_sync/destination', {}))
            rows = SyncProgressDialog('Check Submitted Work', lambda progress, cancelled:
                review_queue(self.window.project_path, configuration, self.window.contributor_session,
                             progress, cancelled), self).run()
            if context != self.current_context():
                return False
            self.context, self.rows = context, rows
            self.queue_valid = True
            self.queue.clear()
            for entry in rows:
                text = (f'{entry["video"]}\n{entry["contributor"]["name"]} | '
                        f'{task_label(entry.get("task"))} | {len(entry["frames"])} frames')
                item = QListWidgetItem(text)
                item.setToolTip(text + '\nFrames: ' + ', '.join(str(i) for i in entry['frames']))
                self.queue.addItem(item)
            if rows:
                self.queue.setCurrentRow(next((i for i, r in enumerate(rows) if r['video'] == old), 0))
            self.count.setText(f'{len(rows)} awaiting review')
            self._describe_active()
            self.update_buttons()
            return True
        except Exception as exc:
            self.status.setText('Review queue unavailable: ' + str(exc))
            self.queue.setEnabled(False)
            self.approve_button.setEnabled(False)
            return False

    @contextmanager
    def saved_operation(self, *, replacing=False):
        w = self.window
        if w._sync_in_progress or self.busy:
            raise ValueError('Wait for the current transfer to finish.')
        if self.context != self.current_context():
            raise ValueError('The project or contributor changed. Reopen the review queue.')
        source = Path(w.project_path).resolve()
        self.busy = w._sync_in_progress = True
        self.update_buttons()
        stopped = False
        paused_timers = []
        try:
            if w.is_playing:
                w.toggle_play()
            w._commit_canvas_edit_if_needed()
            w.save_worker.wait_until_idle(raise_errors=False)
            w._save_annotation_sources()
            w.save_worker.wait_until_idle()
            if replacing:
                for name in ('_instance_list_update_timer', 'done_counter_update_timer'):
                    timer = getattr(w, name, None)
                    if timer is not None and timer.isActive():
                        paused_timers.append(timer)
                        timer.stop()
                w.preload_worker.stop()
                w.preload_worker.join()
                stopped = True
            yield source, dict(setting_json('project_sync/destination', {})), dict(w.contributor_session)
        finally:
            if update_journal_path(source).exists() or update_journal_path(source).is_symlink():
                w._discard_annotation_view()
                w.project_path = None
                w.centralWidget().setEnabled(False)
                w.status_label.setText('Review update needs recovery; reopen the local project before editing')
            if stopped:
                from gui.main_window import PreloadWorker
                w.preload_worker = PreloadWorker(w.frame_cache, w.annotation_manager)
                w.preload_worker.start()
            for timer in paused_timers:
                if w.project_path:
                    timer.start()
            self.busy = w._sync_in_progress = False
            self.update_buttons()

    def open_selected(self):
        row = self.selected()
        if not row or self.busy or not self.queue_valid:
            return False
        if row == self.active and self.window.current_video_id == row['video'] and self.window.frames:
            self._enter_focus()
            self._describe_active()
            return True
        previous = self.active
        replaced_view = False
        try:
            with self.saved_operation(replacing=True) as (source, configuration, actor):
                plan = SyncProgressDialog('Prepare Submission Review', lambda progress, cancelled:
                    PreparedSubmissionReview(source, configuration, actor, row['video'], row['revision'], row['id'],
                                             progress, cancelled), self).run()
                choice = None
                if plan.conflicts:
                    from gui.submission_comparison import SubmissionComparisonDialog
                    comparison = SubmissionComparisonDialog(plan, self.window)
                    try:
                        accepted = comparison.exec()
                        choice = comparison.choice
                    finally:
                        comparison.deleteLater()
                    if not accepted:
                        self._select_active()
                        return False
                result = SyncProgressDialog('Open Submission for Review', lambda progress, cancelled:
                    plan.apply(progress, cancelled, conflict_choice=choice), self).run()
                require_no_update(source)
                w = self.window
                self._enter_focus()
                replaced_view = True
                w._discard_annotation_view()
                w.frames = []
                self.active = dict(row)
                w.annotation_work.check_context()
                filters = w.annotation_work.filters
                w.split_filter_combo.blockSignals(True)
                w.split_filter_combo.setCurrentIndex(0)
                w.split_filter_combo.blockSignals(False)
                w.split_filter = 'all'
                filters.task.setCurrentIndex(max(0, filters.task.findData(row.get('task'))))
                w.load_video_frames(row['video'])
                if w.current_video_id != row['video'] or not w.frames or w.canvas.image_item is None:
                    raise ValueError('The submitted frames could not be opened. Approval remains unavailable.')
                w.on_menu_show_segmentations_changed(True)
                checks = {'bee': w.toolbar.show_bees_checkbox, 'hive': w.toolbar.show_hives_checkbox,
                          'chamber': w.toolbar.show_chambers_checkbox, 'pollen': w.toolbar.show_pollen_checkbox,
                          'nectar': w.toolbar.show_nectar_checkbox, **w.toolbar.brood_visibility_actions}
                for category in task_categories(row.get('task')):
                    checks[category].setChecked(True)
                w.coco_export_dirty = True
                w.invalidate_backup_estimate()
                self._describe_active()
                if result.get('backup'):
                    self.status.setToolTip('Previous affected files: ' + result['backup'])
        except Exception as exc:
            if replaced_view and self.window.project_path:
                self.window._discard_annotation_view()
                if previous:
                    self.active = previous
                    self.window.load_video_frames(previous['video'])
                else:
                    self._restore_browsing()
            self._select_active()
            self._describe_active()
            if not isinstance(exc, InterruptedError) and not isinstance(exc.__cause__, InterruptedError):
                QMessageBox.warning(self.window, 'Submission Not Opened', str(exc))
            self.update_buttons()
            return False
        self.update_buttons()
        return True

    def open_video(self, video):
        if self.busy or not self.queue_valid:
            return
        index = next((i for i, row in enumerate(self.rows) if row['video'] == video), None)
        if index is None:
            QMessageBox.information(self.window, 'Review in Progress',
                'Use Exit Review before opening other project videos.')
            return
        self.queue.setCurrentRow(index)
        self.open_selected()

    def approve(self):
        row = self.selected()
        if (not self.queue_valid or not row or row != self.active or
                self.window.current_video_id != row['video'] or not self.window.frames or
                self.window.canvas.image_item is None):
            return
        if QMessageBox.question(self, 'Approve Reviewed Submission?',
                f'Approve {row["video"]}?\n\nConfirm that you reviewed all {len(row["frames"])} assigned frames '
                'and any video-wide masks. Saved corrections in those frames and shared masks will be published. '
                'Other local work will not be uploaded.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            with self.saved_operation() as (source, configuration, actor):
                result = SyncProgressDialog('Approve Submission', lambda progress, cancelled:
                    approve_review(source, configuration, actor, row['video'], row['revision'], row['id'],
                                   progress, cancelled), self).run()
                self.active = None
                self.window.invalidate_backup_estimate()
                self.window.status_label.setText('Reviewed submission approved: ' + row['video'])
                if result.get('warning'):
                    QMessageBox.warning(self, 'Approval Published', result['warning'])
            self.window.workflow.home.inbox.request_progress(force=True)
            self.window.workflow.home.inbox.refresh()
            self.window.workflow.assignments_page.clear()
            if self.refresh() and self.rows:
                if not self.open_selected():
                    self.close()
            else:
                self.close()
                if self.queue_valid:
                    QMessageBox.information(self.window, 'Review Complete', 'No submissions are awaiting review.')
        except Exception as exc:
            QMessageBox.warning(self, 'Submission Not Approved', str(exc))

    def closeEvent(self, event):
        if self.busy or self.window._sync_in_progress:
            event.ignore()
            return
        try:
            if self.is_focused():
                with self.saved_operation():
                    self._restore_browsing()
            else:
                self._restore_browsing(reload=False)
        except Exception as exc:
            QMessageBox.warning(self, 'Review Still Open', str(exc))
            event.ignore()
            return
        super().closeEvent(event)
