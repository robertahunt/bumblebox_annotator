"""One-time computer setup and a cancellable connection check after choosing identity."""

import json
from pathlib import Path
import sys

from PyQt6.QtCore import QObject, QProcess, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
                            QProgressBar, QPushButton, QStyle, QToolButton, QVBoxLayout)

from gui import project_sync_dialog as preferences
from gui.drive_login import DriveLoginButton


def working_locally():
    app = QApplication.instance()
    return bool(app and app.property('beewhereLocalSession'))


def set_working_locally(local):
    QApplication.instance().setProperty('beewhereLocalSession', bool(local))


def saved_lab_connection():
    destination = preferences.setting_json('project_sync/destination', {})
    if destination.get('root'):
        return destination
    sharing = preferences.setting_json('contributors/shared', {})
    return sharing if sharing.get('enabled') and sharing.get('root') else {}


class LabAccessCheck(QObject):
    completed = pyqtSignal(object)

    def __init__(self, configuration, *, timeout_ms=15000):
        super().__init__(QApplication.instance())
        self.active = True
        self.configuration = configuration
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(Path(__file__).resolve().parent.parent))
        self.process.started.connect(self.send_configuration)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.failed)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(timeout_ms)
        self.timer.timeout.connect(self.timed_out)

    def start(self):
        self.timer.start()
        self.process.start(sys.executable, ['-m', 'core.lab_access'])

    def send_configuration(self):
        self.process.write(json.dumps(self.configuration).encode('utf-8'))
        self.process.closeWriteChannel()

    def report(self, result):
        self.timer.stop()
        if self.active:
            self.active = False
            self.completed.emit(result)

    def finished(self, code, *_):
        if self.active:
            try:
                result = json.loads(bytes(self.process.readAllStandardOutput()))
                if code or not isinstance(result, dict) or not isinstance(result.get('connected'), bool):
                    raise ValueError('Unexpected connection check result')
            except (ValueError, TypeError):
                result = {'connected': False, 'error': 'The drive check could not complete.'}
            self.report(result)
        self.deleteLater()

    def failed(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.report({'connected': False, 'error': 'The connection helper could not start.'})
            self.deleteLater()

    def timed_out(self):
        self.report({'connected': False, 'error': 'The shared drive did not respond within 15 seconds.'})
        self.process.kill()

    def cancel(self):
        self.active = False
        self.timer.stop()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
        # Keep the helper alive until its process exits; never block the GUI waiting.


class StartupConnectionDialog(QDialog):
    def __init__(self, configuration, name='', parent=None):
        super().__init__(parent)
        self.configuration = dict(configuration)
        self.probe = None
        self.closed = False
        self.setWindowTitle('Research Drive')
        self.resize(600, 280)
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        if name:
            identity = QLabel('Working as ' + name)
            identity.setTextFormat(Qt.TextFormat.PlainText)
            identity.setWordWrap(True)
            layout.addWidget(identity)
        self.heading = QLabel('Connecting to the research drive...')
        font = self.heading.font()
        font.setPointSize(15)
        font.setBold(True)
        self.heading.setFont(font)
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.details_button = QToolButton()
        self.details_button.setText('Connection details')
        self.details_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_button.setArrowType(Qt.ArrowType.RightArrow)
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self.show_details)
        layout.addWidget(self.details_button)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(120)
        layout.addWidget(self.details)
        self.details.hide()
        row = QHBoxLayout()
        self.login = DriveLoginButton(self)
        self.login.completed.connect(self.check)
        row.addWidget(self.login)
        self.retry = QToolButton()
        self.retry.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.retry.setToolTip('Check connection again')
        self.retry.setAccessibleName('Check connection again')
        self.retry.clicked.connect(self.check)
        row.addWidget(self.retry)
        row.addStretch()
        self.local = QPushButton('Continue Locally')
        self.local.setAutoDefault(False)
        self.local.clicked.connect(self.reject)
        row.addWidget(self.local)
        layout.addLayout(row)
        self.finished.connect(self.finish_dialog)
        QTimer.singleShot(0, self.check)

    def show_details(self, visible):
        self.details.setVisible(visible)
        self.details_button.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def check(self):
        if self.closed:
            return
        self.close_check()
        self.heading.setText('Connecting to the research drive...')
        self.note.setText('You can continue with downloaded work while the drive is unavailable.')
        self.progress.show()
        self.login.hide()
        self.retry.setEnabled(False)
        self.details_button.hide()
        self.details.hide()
        probe = self.probe = LabAccessCheck(self.configuration)
        probe.completed.connect(lambda result: self.checked(probe, result))
        probe.start()

    def checked(self, probe, result):
        if probe is not self.probe:
            return
        self.probe = None
        if result.get('connected'):
            self.accept()
            return
        self.progress.hide()
        self.heading.setText('The research drive is unavailable')
        self.note.setText('Sign in to try connecting, or continue with work already downloaded. '
                          'Without a connection, you cannot download assignments or upload annotations. '
                          'Your edits will still save on this computer.')
        self.details.setPlainText(str(self.configuration.get('root', '')) + '\n\n' +
                                  str(result.get('error', 'Connection could not be verified.')))
        self.details_button.show()
        self.details.setVisible(self.details_button.isChecked())
        self.login.show()
        self.login.setDefault(True)
        self.retry.setEnabled(True)
        self.login.setFocus()

    def close_check(self, *_):
        if self.probe is not None:
            self.probe.cancel()
            self.probe = None

    def finish_dialog(self, *_):
        self.closed = True
        self.close_check()


def check_session_connection(name='', parent=None):
    configuration = saved_lab_connection()
    if not configuration:
        return True  # Local-only use needs no drive prompt.
    connected = StartupConnectionDialog(configuration, name, parent).exec() == QDialog.DialogCode.Accepted
    set_working_locally(not connected)
    from gui.shared_profiles import profile_sync_service
    profile_sync_service().refresh()
    return connected


def ensure_computer_setup(parent=None):
    settings = preferences.QSettings()
    if settings.value('lab/computer_setup_complete', False, type=bool):
        return True
    if saved_lab_connection() or preferences.load_contributor_profiles():
        # Existing installations already chose a lab connection or local identities.
        settings.setValue('lab/computer_setup_complete', True)
        return True
    dialog = QDialog(parent)
    dialog.setWindowTitle('Set Up This Computer')
    dialog.resize(520, 240)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(24, 24, 24, 24)
    title = QLabel('Welcome to Bumblebox Annotator')
    font = title.font()
    font.setPointSize(16)
    font.setBold(True)
    title.setFont(font)
    title.setWordWrap(True)
    layout.addWidget(title)
    note = QLabel('Connect this computer to your lab to find shared contributor profiles and assigned work, '
                  'or start with local projects. You can change the connection later in Settings.')
    note.setWordWrap(True)
    layout.addWidget(note)
    layout.addStretch()
    connect = QPushButton('Connect to a Lab...')
    connect.setIcon(dialog.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
    connect.setDefault(True)
    def configure():
        from gui.shared_profiles import LabConnectionDialog
        if LabConnectionDialog(dialog).exec() == QDialog.DialogCode.Accepted:
            dialog.accept()
    connect.clicked.connect(configure)
    layout.addWidget(connect)
    local = QPushButton('Use Locally')
    local.clicked.connect(dialog.accept)
    layout.addWidget(local)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False
    settings.setValue('lab/computer_setup_complete', True)
    return True
