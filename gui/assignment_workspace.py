"""Assignment discovery and explicit project-director setup."""

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                            QScrollArea, QStyle, QToolButton, QVBoxLayout, QWidget)

from core.assigned_work import assignment_management, assignment_presence, establish_director
from gui.assigned_work_dialog import AssignmentManagerDialog
from gui.project_sync_dialog import (OperationWorker, SyncProgressDialog,
                                     setting_json)


def text_label(text=''):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class AssignmentInbox(QWidget):
    """A failed or stale background check must never become a 'no assignments' notice."""
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.worker = None
        self.notice = None
        self.shown_empty = set()
        self.stopping = False
        self.result = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        row = QHBoxLayout()
        self.summary = text_label('Assignments not checked')
        row.addWidget(self.summary, 1)
        self.check = QToolButton()
        self.check.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.check.setAccessibleName('Check assignments again')
        self.check.setToolTip('Check assignments again')
        self.check.clicked.connect(self.refresh)
        row.addWidget(self.check)
        layout.addLayout(row)
        actions = QHBoxLayout()
        self.share = QPushButton('Profile Sharing...')
        self.share.setToolTip('Make contributor profiles available on other computers through the shared folder')
        self.share.clicked.connect(self.configure_profiles)
        self.download = QPushButton('Download Assigned Work...')
        self.download.clicked.connect(lambda: window.workflow.join_shared())
        self.configure = QPushButton('Connect Shared Folder...')
        self.configure.clicked.connect(self.connect_folder)
        self.manage = QPushButton('Open Assignments')
        self.manage.clicked.connect(window.manage_assignments)
        for button in (self.share, self.download, self.configure, self.manage):
            actions.addWidget(button)
            button.hide()
        actions.addStretch()
        layout.addLayout(actions)

    def context(self):
        actor = self.window.contributor_session or {}
        configuration = setting_json('project_sync/destination', {})
        return (actor.get('id'), actor.get('session_id'), str(self.window.project_path or ''),
                configuration.get('root_id'), configuration.get('root'))

    def reset(self):
        self.result = None
        if self.notice:
            self.notice.close()
            self.notice = None
        QTimer.singleShot(0, self.refresh)

    def refresh(self):
        if self.stopping:
            return
        if self.worker:
            # Completion checks the captured context and restarts if it became stale.
            return
        actor = dict(self.window.contributor_session or {})
        if not actor:
            self.summary.setText('Select a contributor profile to check assignments')
            return
        self.result = None
        for button in (self.share, self.download, self.configure, self.manage):
            button.hide()
        self.summary.setText('Checking assignments...')
        self.check.setEnabled(False)
        context = self.context()
        project = self.window.project_path
        configuration = dict(setting_json('project_sync/destination', {}))
        worker = OperationWorker(lambda progress, cancelled:
            assignment_presence(configuration, actor, project, cancelled), self)
        self.worker = worker
        worker.finished.connect(lambda: self.finished(worker, context, actor))
        worker.start()

    def finished(self, worker, context, actor):
        if worker is not self.worker:
            return
        self.worker = None
        worker.deleteLater()
        if self.stopping:
            return
        self.check.setEnabled(True)
        if context != self.context():
            self.refresh()
            return
        if worker.error:
            self.summary.setText('Unable to check assignments. Local work is still available.')
            self.summary.setToolTip(str(worker.error))
            self.configure.show()
            return
        self.result = result = worker.result
        status, count = result['status'], result['count']
        scope = result.get('scope', 'this project')
        messages = {
            'unconfigured': 'Shared folder not configured',
            'unpublished': 'This local project has not been published for assignments',
            'setup': 'Project director setup is required',
            'director': f'Project director | {count} videos assigned to you',
            'assigned': f'{count} videos assigned to {actor["name"]} in {scope}',
            'empty': f'No work assigned to {actor["name"]} in {scope}',
        }
        self.summary.setText(messages[status])
        self.summary.setToolTip('')
        self.configure.setVisible(status == 'unconfigured')
        self.share.setVisible(status in ('empty', 'assigned'))
        self.download.setVisible(bool(count))
        self.manage.setVisible(bool(self.window.project_path) and status in ('director', 'setup', 'unpublished'))
        if self.window.project_path:
            self.window.workflow.assignment_role_changed(result)
        notice_key = (context[0], context[1], context[3])
        if status == 'empty' and notice_key not in self.shown_empty:
            self.shown_empty.add(notice_key)
            self.show_empty(actor, result)

    def show_empty(self, actor, result):
        if self.notice:
            self.notice.close()
        notice = QDialog(self.window)
        notice.setWindowTitle('No Work Assigned Yet')
        notice.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        notice.setModal(False)
        notice.resize(560, 230)
        layout = QVBoxLayout(notice)
        director = result.get('director')
        manager = director['name'] if director else 'your project director'
        layout.addWidget(text_label(f'No work is assigned to {actor["name"]} in {result["scope"]}.\n\n'
            f'Ask {manager} to assign videos to this profile. Enable Profile Sharing on both computers '
            'to make the same contributor identities available. '
            'Local work remains available.'))
        row = QHBoxLayout()
        share = QPushButton('Profile Sharing...')
        share.clicked.connect(self.configure_profiles)
        retry = QPushButton('Check Again')
        retry.clicked.connect(lambda: (notice.close(), self.refresh()))
        close = QPushButton('Continue Locally')
        close.setDefault(True)
        close.clicked.connect(notice.close)
        for button in (share, retry, close):
            row.addWidget(button)
        layout.addLayout(row)
        notice.finished.connect(lambda: self.notice_finished(notice))
        self.notice = notice
        notice.show()

    def notice_finished(self, notice):
        if self.notice is notice:
            self.notice = None

    def configure_profiles(self):
        from gui.shared_profiles import SharedProfileSettingsDialog
        SharedProfileSettingsDialog(self.window).exec()

    def connect_folder(self):
        self.window.configure_project_sync()
        self.refresh()

    def stop(self):
        self.stopping = True
        if self.notice:
            self.notice.close()
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            return self.worker.wait(2000)
        return True


class AssignmentWorkspace(QScrollArea):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self.panel = None
        self.context_key = None
        self.setWidgetResizable(True)
        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.layout.setContentsMargins(24, 24, 24, 24)
        self.title = text_label('Assignments')
        font = self.title.font()
        font.setPointSize(18)
        font.setBold(True)
        self.title.setFont(font)
        self.layout.addWidget(self.title)
        self.status = text_label()
        self.layout.addWidget(self.status)
        self.setup = QWidget()
        self.setup_layout = QVBoxLayout(self.setup)
        self.setup_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.setup)
        self.retry = QPushButton('Check Setup Again')
        self.retry.clicked.connect(self.refresh)
        self.layout.addWidget(self.retry)
        self.layout.addStretch()
        self.setWidget(self.content)

    def clear(self):
        self.context_key = None
        self.status.show()
        self.setup.show()
        self.retry.show()
        if self.panel:
            self.panel.hide()
            self.layout.removeWidget(self.panel)
            self.panel.deleteLater()
            self.panel = None
        while self.setup_layout.count():
            widget = self.setup_layout.takeAt(0).widget()
            if widget:
                widget.deleteLater()

    def activate(self):
        key = self.window.workflow.home.inbox.context()
        if self.context_key != key:
            self.refresh()

    def confirm_discard(self):
        if self.panel:
            try:
                old = {v: {k: e[k] for k in ('contributor', 'frames')} for v, e in self.panel.data['videos'].items()}
                dirty = old != self.panel.choices()
            except ValueError:
                dirty = True
            if dirty and QMessageBox.question(self, 'Discard Unsaved Assignment Choices?',
                    'This will discard unsaved assignment table edits. Continue?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return False
        return True

    def refresh(self):
        if not self.confirm_discard():
            return
        self.clear()
        window = self.window
        if not window.project_path or not window.contributor_session:
            self.status.setText('Open a local project and choose your profile first.')
            return
        configuration = setting_json('project_sync/destination', {})
        try:
            state = SyncProgressDialog('Check Assignment Setup', lambda progress, cancelled:
                assignment_management(window.project_path, configuration, window.contributor_session), self).run()
            self.setup_revision = state.get('data', {}).get('revision')
            self.context_key = window.workflow.home.inbox.context()
            if state.get('restricted'):
                self.status.setText('Assignments are managed by ' + state['director']['name'] + '.')
                return
            if state['needs']:
                self.status.setText('Assignment setup')
                actions = {
                    'connection': ('Shared folder not connected', 'Connect Shared Folder...', window.configure_project_sync),
                    'backup': ('Publish the project before assigning videos', 'Prepare Project Backup...', self.prepare_backup),
                    'compact_backup': ('A deduplicated backup is required', 'Backup Settings...', window.configure_project_sync),
                    'policy': ('The published annotation policy is incomplete', 'Review Annotation Policy...', window.show_annotation_policy),
                    'frames': ('No frame images are published', 'Add Video...', window.show_add_video_dialog),
                    'director': ('No project director has been set', 'Set Project Director...', self.set_director),
                }
                for need in state['needs']:
                    message, title, callback = actions[need]
                    self.setup_layout.addWidget(text_label(message))
                    button = QPushButton(title)
                    button.clicked.connect(callback)
                    self.setup_layout.addWidget(button)
                if any(need in state['needs'] for need in ('policy', 'frames', 'compact_backup')):
                    button = QPushButton('Prepare Updated Backup...')
                    button.clicked.connect(self.prepare_backup)
                    self.setup_layout.addWidget(button)
                return
            self.status.setText('')
            self.status.hide()
            self.setup.hide()
            self.retry.hide()
            self.panel = AssignmentManagerDialog(window.project_path, configuration, window.contributor_session,
                self.content, initial=(state['data'], state['available'], state['statuses']), embedded=True)
            self.layout.insertWidget(self.layout.count() - 1, self.panel, 1)
            self.panel.finished.connect(self.refresh)
            self.panel.assignments_changed.connect(window.workflow.home.inbox.reset)
        except Exception as exc:
            self.status.setText('Unable to load assignments: ' + str(exc))

    def prepare_backup(self):
        self.window.workflow.activate('Manage')

    def set_director(self):
        actor = dict(self.window.contributor_session)
        answer = QMessageBox.question(self, 'Set Project Director?',
            f'Make {actor["name"]} the director of this project?\n\n'
            'This profile will manage assignments and accept submissions. '
            'The role applies only to this project and can be transferred explicitly later.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            configuration = setting_json('project_sync/destination', {})
            # The backend rejects a director created by another computer in the meantime.
            SyncProgressDialog('Set Project Director', lambda progress, cancelled:
                establish_director(self.window.project_path, configuration, actor, self.setup_revision), self).run()
            self.window.workflow.home.inbox.reset()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Director Not Set', str(exc))
