"""Coalesced assigned uploads and half-hour activity reports for an open project."""

from copy import deepcopy
from time import monotonic

from PyQt6.QtCore import QObject, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.assigned_work import publish_work, require_contributor
from core.project_sync import PROJECT_MARKER, SyncConflictError
from core.work_session import WorkSession, publish_report
from gui.lab_startup import working_locally
from gui.project_sync_dialog import OperationWorker, SyncProgressDialog, setting_json


HEARTBEAT_SECONDS = 30 * 60


class AutomaticWork(QObject):
    def __init__(self, workflow):
        super().__init__(workflow)
        self.workflow, self.window = workflow, workflow.window
        self.session = None
        self.configuration = None
        self.worker = None
        self.dialog = None
        self.last_report = self.last_attempt = 0.0
        self.due = 0.0
        self.blocked = False
        self.timer = QTimer(self)
        self.timer.setInterval(10000)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

    def context_changed(self):
        w = self.window
        actor = w.contributor_session or {}
        key = (str(w.project_path), actor.get('session_id'))
        if self.session and key == (str(self.session.project), self.session.actor['session_id']):
            return
        if self.session:
            # A failed close report remains in the local journal, rather than
            # pretending that the shared drive received it.
            try:
                self.session.end()
            except OSError as exc:
                self.window.status_label.setText('Previous session report could not be saved: ' + str(exc))
        self.session = None
        if (not w.project_path or not actor.get('device', {}).get('id') or
                not (w.project_path / 'annotations' / PROJECT_MARKER).is_file()):
            return
        try:
            require_contributor(w.project_path, actor)
            self.session = WorkSession(w.project_path, actor)
            self.session.start()
            self.configuration = deepcopy(setting_json('project_sync/destination', {}))
            self.blocked = False
            self.last_report = self.last_attempt = 0.0
            self.due = monotonic() + 5
            if self.workflow.assigned() and self.workflow.pending():
                self.session.data['pending'] = True
                self.session.save()
        except (OSError, ValueError, KeyError) as exc:
            self.window.status_label.setText('Automatic work status unavailable: ' + str(exc))
            self.session = None

    def saved(self, *, completed=False):
        if not self.session:
            self.context_changed()
        if not self.session:
            return
        try:
            self.session.changed(self.window.current_video_id, queue=self.workflow.assigned(), completed=completed)
            self.due = monotonic() + (1 if completed else 30)
        except (OSError, ValueError) as exc:
            self.window.status_label.setText('Automatic upload queue could not be saved: ' + str(exc))

    def pending(self):
        return bool(self.session and self.workflow.assigned() and self.session.data.get('pending'))

    def queue_existing(self):
        """Recover older/external saved edits without inventing their edit timestamp."""
        if self.session and self.workflow.assigned() and not self.pending():
            try:
                self.session.data['pending'] = True
                self.session.save()
                self.due = monotonic() + 5
            except OSError as exc:
                self.workflow.shared_status.setToolTip('Upload queue could not be saved: ' + str(exc))

    def description(self):
        if not self.pending():
            return ''
        if self.blocked:
            return 'Upload needs attention; work is saved locally'
        if working_locally():
            return 'Saved locally; upload waiting for connection'
        if self.worker:
            return 'Uploading saved annotations...'
        return 'Saved locally; upload queued'

    def retry(self):
        self.blocked = False
        self.last_attempt = 0.0
        self.due = 0.0
        self.tick()

    def tick(self):
        try:
            self._tick()
        except (OSError, ValueError, KeyError) as exc:
            self.workflow.shared_status.setText('Automatic work status needs attention')
            self.workflow.shared_status.setToolTip(str(exc))

    def _tick(self):
        w = self.window
        if not self.session or self.worker or working_locally() or w._sync_in_progress:
            return
        if QApplication.activeModalWidget():
            return
        now = monotonic()
        upload = self.pending() and not self.blocked and now >= self.due and (
            not self.last_attempt or now - self.last_attempt >= 60)
        if (w.current_frame_modified or w.save_worker.save_queue.unfinished_tasks or
                w.save_worker.failed_tasks or getattr(w.canvas, 'is_drawing', False)):
            upload = False
        heartbeat = not self.last_report or now - self.last_report >= HEARTBEAT_SECONDS
        if not upload and not heartbeat:
            return
        configuration = deepcopy(setting_json('project_sync/destination', {}))
        if not configuration.get('root_id'):
            return
        self.configuration = configuration
        session = self.session
        report = session.report()
        report['local_save_state'] = ('failed' if w.save_worker.failed_tasks else
            'saving' if w.save_worker.save_queue.unfinished_tasks else
            'unsaved' if w.current_frame_modified else 'saved')
        self.last_report = now  # Also back off on failure; never hammer an unavailable mount.
        if upload:
            self.last_attempt = now
            w._sync_in_progress = True
        operation = lambda progress, cancelled: self.transfer(
            session, configuration, report, upload, progress, cancelled, self.window._assigned_upload_cache)
        if upload:
            # Publication needs a stable snapshot. A cancellable, non-blocking
            # modal prevents strokes and dock/menu actions from racing that snapshot.
            self.dialog = SyncProgressDialog('Upload Saved Annotations', operation, w)
            self.dialog.setWindowModality(Qt.WindowModality.WindowModal)
            worker = self.dialog.worker
            self.dialog.show()
        else:
            worker = OperationWorker(operation, self)
        worker.upload = upload
        self.worker = worker
        worker.finished.connect(lambda: self.finished(worker, session))
        worker.start()
        self.workflow.update_save_status()

    @staticmethod
    def transfer(session, configuration, report, upload, progress, cancelled, checksum_cache=None):
        result = {}
        if upload:
            result['publication'] = publish_work(session.project, configuration, session.actor,
                auto_ready=True, progress=progress, cancelled=cancelled, checksum_cache=checksum_cache)
            report = dict(report, pending=False, last_uploaded_at=report['report_created_at'],
                          ready=result['publication']['ready'], local_save_state='saved')
        try:
            result['activity'] = publish_report(configuration, report, cancelled)
        except (OSError, ValueError, RuntimeError) as exc:
            # A failed advisory report must not turn a verified upload into a failure.
            result['activity_error'] = str(exc)
        return result

    def finished(self, worker, session):
        if worker is not self.worker:
            return
        self.worker = None
        if worker.upload:
            self.window._sync_in_progress = False
        try:
            if worker.error:
                session.failed(worker.error)
                self.blocked = isinstance(worker.error, (SyncConflictError, ValueError, InterruptedError))
                self.workflow.shared_status.setText('Upload needs attention' if self.blocked else
                                                   'Drive unavailable; upload will retry')
                self.workflow.shared_status.setToolTip(str(worker.error))
            elif worker.result.get('publication'):
                result = worker.result['publication']
                session.published(result)
                if session is self.session:
                    self.workflow.uploaded()
                    self.window.assigned_status.setText('Upload verified')
                    self.workflow.home.inbox.request_progress(force=True)
            if not worker.error and worker.result.get('activity_error'):
                self.workflow.shared_status.setToolTip(
                    'Computer activity report was not sent: ' + worker.result['activity_error'])
        except (OSError, ValueError) as exc:
            self.workflow.shared_status.setText('Local upload receipt needs attention')
            self.workflow.shared_status.setToolTip(str(exc))
        finally:
            if self.dialog and worker.upload:
                self.dialog.deleteLater()
                self.dialog = None
            worker.deleteLater()
            self.workflow.update_save_status()

    def finish_session(self):
        """Close needs no separate submit decision; cancelled/offline work stays queued."""
        if not self.session:
            return True
        session = self.session
        if self.worker:
            self.worker.requestInterruption()
            if not self.worker.wait(2000):
                self.window.status_label.setText('Finishing shared-drive operation; please close again shortly')
                return False
            self.finished(self.worker, session)
        try:
            session.end()
        except OSError as exc:
            QMessageBox.warning(self.window, 'Session Status Not Saved', str(exc))
            return False
        if working_locally():
            return True
        configuration = deepcopy(setting_json('project_sync/destination', {}))
        if not configuration.get('root_id'):
            return True
        try:
            result = SyncProgressDialog('Finish Session', lambda progress, cancelled: self.transfer(
                session, configuration, session.report(), self.pending() and not self.blocked,
                progress, cancelled, self.window._assigned_upload_cache), self.window).run()
            if result.get('publication'):
                session.published(result['publication'])
                self.workflow.uploaded()
        except (OSError, ValueError, RuntimeError) as exc:
            try:
                session.failed(exc)
            except OSError:
                pass  # The pending queue was already persisted before the transfer.
            QMessageBox.information(self.window, 'Work Saved Locally',
                'The upload is still queued on this computer. It will retry when this working copy '
                'is reopened and connected.\n\n' + str(exc))
        return True

    def stop(self):
        self.timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            return self.worker.wait(2000)
        return True
