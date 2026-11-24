"""Take in contributors' uploaded work in the background on the director's computer.

The checks and the plan are read-only and run in a worker thread. Writing happens
only while nothing is being drawn, saved or reviewed, behind a short progress
dialog, through the same journalled update a submission review uses.
"""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import monotonic

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QHBoxLayout, QHeaderView, QLabel,
                             QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from core.assigned_work import workspace
from core.categories import category_label
from core.intake import PreparedIntake, accept_work, fingerprint, read_state, release
from core.project_sync import PROJECT_MARKER, require_no_update, update_journal_path
from gui.lab_startup import working_locally
from gui.project_sync_dialog import OperationWorker, SyncProgressDialog, save_setting, setting_json

SETTING = 'project_sync/automatic_intake'
CHECK_SECONDS = 120  # how often to look for new uploads
RETRY_SECONDS = 15   # when the app was busy at the moment there was something to take in


def intake_enabled():
    return bool(setting_json(SETTING, True))


class AutomaticIntake(QObject):
    updated = pyqtSignal()

    def __init__(self, workflow):
        super().__init__(workflow)
        self.workflow, self.window = workflow, workflow.window
        self.worker = None
        self.context = None
        self.seen = None         # the published state last dealt with
        self.unavailable = None  # why intake does not apply to this project and person
        self.note = ''           # the last outcome, in words
        self.due = 0.0
        self.timer = QTimer(self)
        self.timer.setInterval(15000)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            return self.worker.wait(2000)
        return True

    def context_key(self):
        w = self.window
        return str(w.project_path), (w.contributor_session or {}).get('id')

    def possible(self):
        """Whether this is a director's full project connected to a shared folder (as far as is known locally)."""
        w = self.window
        return bool(w.project_path and w.contributor_session and not workspace(w.project_path) and
                    (Path(w.project_path) / 'annotations' / PROJECT_MARKER).is_file())

    def quiet(self):
        """Nothing is being drawn, saved, transferred or reviewed, so local files can be replaced."""
        w = self.window
        review = getattr(w, 'submission_review_dock', None)
        return not (w._sync_in_progress or w.current_frame_modified or w.save_worker.save_queue.unfinished_tasks or
                    w.save_worker.failed_tasks or getattr(w.canvas, 'is_drawing', False) or
                    (review is not None and review.is_focused()))

    def tick(self):
        if intake_enabled() and not QApplication.activeModalWidget() and monotonic() >= self.due:
            self.check()

    def check(self, *, force=False, replace=(), keep=()):
        """Look for new uploads and take them in. `force` looks even if nothing was published since last time."""
        key = self.context_key()
        if key != self.context:
            self.context, self.seen, self.unavailable, self.note = key, None, None, ''
        if self.worker or not self.possible() or working_locally() or (self.unavailable and not force):
            return False
        configuration = deepcopy(setting_json('project_sync/destination', {}))
        if not configuration.get('root_id'):
            return False
        self.due = monotonic() + CHECK_SECONDS  # also after a failure: never hammer an unavailable drive
        project, actor, seen = Path(self.window.project_path), dict(self.window.contributor_session), self.seen
        replace, keep = set(replace), set(keep)

        def operation(progress, cancelled):
            mark = fingerprint(project, configuration)
            if mark == seen and not force:
                return mark, None
            return mark, PreparedIntake(project, configuration, actor, replace=replace, keep=keep,
                                        progress=progress, cancelled=cancelled)

        worker = self.worker = OperationWorker(operation, self)
        worker.finished.connect(lambda: self.checked(worker, key))
        worker.start()
        return True

    def checked(self, worker, key):
        if worker is not self.worker:
            return
        self.worker = None
        if key != self.context_key():
            return
        if worker.error:
            message = str(worker.error)
            if 'director' in message or 'full project' in message.lower():
                self.unavailable = message  # not this person's job here; do not keep asking the drive
            else:
                self.note = 'Shared work could not be checked: ' + message
            return self.updated.emit()
        mark, plan = worker.result
        if plan is None:
            return
        if not plan.taken:
            try:
                plan.apply()  # nothing to copy; only the record of what was seen and skipped
            except (OSError, ValueError) as exc:
                self.note = 'The intake record could not be saved: ' + str(exc)
                return self.updated.emit()
            self.seen = mark
            return self.finish(plan.summary())
        if not self.quiet() or QApplication.activeModalWidget():
            self.due = monotonic() + RETRY_SECONDS  # wait for a pause in the work, then plan afresh
            return
        self.apply(plan, mark)

    def apply(self, plan, mark):
        w = self.window
        source = Path(w.project_path).resolve()
        current = (w.current_video_id, w.current_frame_idx)
        w._sync_in_progress = True
        stopped, paused = False, []
        try:
            if w.is_playing:
                w.toggle_play()
            w._commit_canvas_edit_if_needed()
            w.save_worker.wait_until_idle(raise_errors=False)
            w._save_annotation_sources()
            w.save_worker.wait_until_idle()
            for name in ('_instance_list_update_timer', 'done_counter_update_timer'):
                timer = getattr(w, name, None)
                if timer is not None and timer.isActive():
                    paused.append(timer)
                    timer.stop()
            w.preload_worker.stop()
            w.preload_worker.join()
            stopped = True
            # Replacing files needs a still project: the dialog keeps strokes and menus away meanwhile.
            dialog = SyncProgressDialog('Take In Shared Work', plan.apply, w)
            dialog.setWindowModality(Qt.WindowModality.WindowModal)
            result = dialog.run()
            require_no_update(source)
            self.seen = mark
            self.refresh_view(set(plan.taken), current)
            w.coco_export_dirty = True
            w.invalidate_backup_estimate()
            self.finish(result)
        except Exception as exc:  # automatic: report in the status line, never in a popup
            self.due = monotonic() + CHECK_SECONDS
            self.note = 'Shared work was not taken in: ' + str(exc)
            w.status_label.setText(self.note)
            self.updated.emit()
        finally:
            if update_journal_path(source).exists() or update_journal_path(source).is_symlink():
                # Never let a stale canvas write into a partly replaced project.
                w._discard_annotation_view()
                w.project_path = None
                w.centralWidget().setEnabled(False)
                w.status_label.setText('Shared work intake needs recovery; reopen the local project before editing')
            if stopped:
                from gui.main_window import PreloadWorker
                w.preload_worker = PreloadWorker(w.frame_cache, w.annotation_manager)
                w.preload_worker.start()
            for timer in paused:
                if w.project_path:
                    timer.start()
            w._sync_in_progress = False

    def refresh_view(self, taken, current):
        """Show what is now on disk: reload the open video if it was replaced, else just forget cached copies."""
        w = self.window
        video, frame = current
        if video in taken:
            w._discard_annotation_view()
            w.frames = []
            w.load_video_frames(video)
            if w.frames and 0 <= frame < len(w.frames):
                w.load_frame(frame)
            return
        manager = w.annotation_manager
        for key in [key for key in manager.frame_annotations if isinstance(key, tuple) and key[0] in taken]:
            del manager.frame_annotations[key]
        baselines = manager.attributor.baselines
        for key in [key for key in baselines if isinstance(key, tuple) and len(key) > 1 and key[1] in taken]:
            del baselines[key]
        for name in taken:
            w.video_next_mask_id.pop(name, None)

    def finish(self, summary):
        taken, skipped = summary.get('taken', []), summary.get('skipped', {})
        parts = []
        if taken:
            people = sorted({row['contributor'] for row in taken})
            parts.append(f"Took in new work on {len(taken)} video{'s' if len(taken) != 1 else ''} from "
                         f"{', '.join(people)}; held until accepted")
        if skipped:
            parts.append(f"{len(skipped)} video{'s' if len(skipped) != 1 else ''} not taken in "
                         "(Project > Shared Work Intake...)")
        if parts:
            self.note = '; '.join(parts)
            self.window.status_label.setText(self.note)
        elif not self.note.startswith('Took in'):
            self.note = 'Shared work is up to date'
        if taken:
            try:
                self.workflow.home.inbox.request_progress(force=True)
            except (AttributeError, RuntimeError):
                pass
        self.updated.emit()


class IntakeDialog(QDialog):
    """What automatic intake has brought in, what it holds back and what it left alone."""

    def __init__(self, window):
        super().__init__(window)
        self.window, self.intake = window, window.workflow.automatic_intake
        self.setWindowTitle('Shared Work Intake')
        self.resize(900, 460)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Work that contributors upload on their assigned videos is copied into this project "
            "automatically, finished or not. It is never copied over annotations changed here. "
            "Until you accept a video, the frames and video-wide masks they changed are held: "
            "left out of COCO export and training.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.enabled = QCheckBox('Take in assigned work automatically')
        self.enabled.setChecked(intake_enabled())
        self.enabled.toggled.connect(self.set_enabled)
        layout.addWidget(self.enabled)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Video', 'From', 'State', 'Last taken in'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.update_buttons)
        layout.addWidget(self.table, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.check_button = QPushButton('Check Now')
        self.check_button.clicked.connect(lambda: self.run())
        self.accept_button = QPushButton('Accept Selected')
        self.accept_button.setToolTip('Accept submitted videos whose annotations here are exactly what was submitted.')
        self.accept_button.clicked.connect(self.accept_selected)
        self.theirs_button = QPushButton('Use Their Version')
        self.theirs_button.setToolTip('Replace the annotations changed here. They are kept in Local Rollback History.')
        self.theirs_button.clicked.connect(lambda: self.run(replace=self.selected('skipped')))
        self.mine_button = QPushButton('Keep Mine')
        self.mine_button.setToolTip('Leave this video as it is here until they upload again.')
        self.mine_button.clicked.connect(lambda: self.run(keep=self.selected('skipped')))
        self.release_button = QPushButton('Stop Holding')
        self.release_button.setToolTip('Let these videos into export and training without accepting them.')
        self.release_button.clicked.connect(self.release_selected)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        for button in (self.check_button, self.accept_button, self.theirs_button, self.mine_button,
                       self.release_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.intake.updated.connect(self.refresh)
        self.refresh()

    def set_enabled(self, enabled):
        save_setting(SETTING, bool(enabled))
        if enabled:
            self.intake.due = 0.0

    def records(self):
        try:
            return read_state(self.window.project_path)['videos'] if self.window.project_path else {}
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return {}

    @staticmethod
    def describe(record):
        if record.get('skipped'):
            return 'skipped', 'Not taken in: ' + record['skipped']['reason']
        if record.get('held'):
            frames, shared = record['held'].get('frames', []), record['held'].get('shared', [])
            parts = ([f"{len(frames)} frame{'s' if len(frames) != 1 else ''}"] if frames else []) + [
                category_label(category) + ' mask' for category in shared]
            return 'held', 'Held, ' + ' and '.join(parts) + ': not accepted yet'
        if record.get('kept_local'):
            return 'kept', 'Kept as it is here until they upload again'
        return 'done', 'Accepted' if record.get('accepted_at') else 'Up to date'

    def refresh(self):
        chosen = set(self.selected())
        records = self.records()
        self.table.setRowCount(0)
        order = {'skipped': 0, 'held': 1, 'kept': 2, 'done': 3}
        rows = sorted(((self.describe(record), video, record) for video, record in records.items()),
                      key=lambda row: (order[row[0][0]], row[1]))
        for (kind, text), video, record in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            taken = record.get('taken_at') or ''
            try:  # recorded in UTC; shown in this computer's time
                taken = datetime.fromisoformat(taken).astimezone().strftime('%Y-%m-%d %H:%M') if taken else ''
            except ValueError:
                taken = taken[:16].replace('T', ' ')
            for column, value in enumerate((video, record.get('contributor', {}).get('name', ''), text, taken)):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, (video, kind))
                self.table.setItem(row, column, item)
            if video in chosen:
                self.table.selectRow(row)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        intake = self.intake
        self.status.setText(intake.unavailable or intake.note or
                            ('No contributor work has been taken in yet.' if not rows else ''))
        self.update_buttons()

    def selected(self, kind=None):
        videos = []
        for index in self.table.selectionModel().selectedRows() if self.table.selectionModel() else []:
            video, row_kind = self.table.item(index.row(), 0).data(Qt.ItemDataRole.UserRole)
            if kind is None or row_kind == kind:
                videos.append(video)
        return videos

    def update_buttons(self):
        usable = self.intake.possible() and not self.intake.unavailable
        self.check_button.setEnabled(usable)
        self.accept_button.setEnabled(usable and bool(self.selected('held')))
        skipped = usable and bool(self.selected('skipped'))
        self.theirs_button.setEnabled(skipped)
        self.mine_button.setEnabled(skipped)
        self.release_button.setEnabled(bool(self.selected('held')))

    def run(self, **choices):
        if not self.intake.check(force=True, **choices):
            self.status.setText('A check is already running, or this project is not connected to a shared folder.')

    def accept_selected(self):
        videos = self.selected('held')
        w = self.window
        if not videos or w._sync_in_progress:
            return
        configuration = deepcopy(setting_json('project_sync/destination', {}))
        project, actor = Path(w.project_path), dict(w.contributor_session)
        w._sync_in_progress = True
        try:
            w.save_worker.wait_until_idle(raise_errors=False)
            w._save_annotation_sources()
            w.save_worker.wait_until_idle()
            accepted, refused = SyncProgressDialog('Accept Submitted Work', lambda progress, cancelled:
                accept_work(project, configuration, actor, videos), self).run()
        except Exception as exc:
            QMessageBox.warning(self, 'Work Not Accepted', str(exc))
            return
        finally:
            w._sync_in_progress = False
        if accepted:
            w.coco_export_dirty = True
            try:
                w.workflow.home.inbox.request_progress(force=True)
            except (AttributeError, RuntimeError):
                pass
        self.refresh()
        lines = [f"Accepted {len(accepted)} video{'s' if len(accepted) != 1 else ''}."] if accepted else []
        lines += [f'{video}: {reason}' for video, reason in sorted(refused.items())]
        self.status.setText('\n'.join(lines))

    def release_selected(self):
        videos = self.selected('held')
        if not videos:
            return
        answer = QMessageBox.question(
            self, 'Stop Holding',
            f"Let {len(videos)} video{'s' if len(videos) != 1 else ''} into export and training "
            "without accepting the work?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            release(self.window.project_path, videos)
            self.window.coco_export_dirty = True
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Still Held', str(exc))
        self.refresh()
