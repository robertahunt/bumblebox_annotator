"""Local contributor preferences and optional mounted-folder publishing UI."""

import json
from pathlib import Path

from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                            QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QMessageBox, QProgressBar, QPushButton,
                            QStyle, QVBoxLayout)

from core.contributors import new_session
from core.project_sync import test_destination


def setting_json(key, default):
    value = QSettings().value(key)
    if not value:
        return default
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return default


def save_setting(key, value):
    settings = QSettings()
    settings.setValue(key, json.dumps(value))
    settings.sync()


def choose_contributor(parent=None):
    profiles = setting_json('contributors/profiles', {})
    dialog = QDialog(parent)
    dialog.setWindowTitle('Session Contributor')
    layout = QVBoxLayout(dialog)
    form = QFormLayout()
    name = QComboBox()
    name.setEditable(True)
    name.lineEdit().setMaxLength(80)
    name.addItems(sorted(profiles))
    name.setCurrentText(QSettings().value('contributors/last_name', '', type=str))
    name.setMinimumWidth(280)
    form.addRow('Contributor name:', name)
    layout.addLayout(form)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Start Session')
    buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(name.currentText().strip()))
    name.currentTextChanged.connect(lambda text: buttons.button(
        QDialogButtonBox.StandardButton.Ok).setEnabled(bool(text.strip())))
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    display_name = name.currentText().strip()
    existing = next((key for key in profiles if key.casefold() == display_name.casefold()), None)
    session = new_session(display_name, profiles.get(existing))
    if existing and existing != display_name:
        del profiles[existing]
    profiles[display_name] = session['id']
    save_setting('contributors/profiles', profiles)
    QSettings().setValue('contributors/last_name', display_name)
    return session


class OperationWorker(QThread):
    progress = pyqtSignal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.operation(self.progress.emit, self.isInterruptionRequested)
        except Exception as exc:
            self.error = str(exc)


class SyncProgressDialog(QDialog):
    def __init__(self, title, operation, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        self.label = QLabel('Preparing...')
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        progress = QProgressBar()
        progress.setRange(0, 0)
        layout.addWidget(progress)
        self.cancel = QPushButton('Cancel')
        self.cancel.clicked.connect(self.reject)
        layout.addWidget(self.cancel)
        self.worker = OperationWorker(operation, self)
        self.worker.progress.connect(self.label.setText)
        self.worker.finished.connect(self.accept)

    def reject(self):
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.label.setText('Cancelling after the current file operation...')
        else:
            super().reject()

    def run(self):
        self.worker.start()
        self.exec()
        self.worker.wait()
        if self.worker.error:
            raise RuntimeError(self.worker.error)
        return self.worker.result


class ProjectSyncDialog(QDialog):
    def __init__(self, project=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Project Sync')
        self.setMinimumWidth(580)
        self.project = str(Path(project).resolve()) if project else None
        self.configuration = setting_json('project_sync/destination', {})
        self.projects = setting_json('project_sync/projects', {})
        self.tested = None
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.enabled = QCheckBox('Enable sync for this project')
        self.enabled.setEnabled(self.project is not None)
        self.enabled.setChecked(self.projects.get(self.project, {}).get('enabled', False))
        form.addRow('Project:', QLabel(Path(self.project).name if self.project else 'No project open'))
        form.addRow('', self.enabled)
        self.folder = QLineEdit(self.configuration.get('root', ''))
        self.folder.setToolTip('An existing mounted or local folder. Authentication is handled outside the app.')
        self.folder.textChanged.connect(self._path_changed)
        browse = QPushButton()
        browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        browse.setToolTip('Choose sync folder')
        browse.clicked.connect(self.browse)
        row = QHBoxLayout()
        row.addWidget(self.folder)
        row.addWidget(browse)
        form.addRow('Destination:', row)
        self.videos = QCheckBox('Include original videos')
        self.videos.setChecked(self.configuration.get('include_videos', True))
        self.videos.setToolTip('Annotations and extracted frames are always included. Models, training runs, '
                              'debug images, and COCO exports are excluded. Without videos, the copy is '
                              'not a complete project backup.')
        form.addRow('', self.videos)
        layout.addLayout(form)
        self.test = QPushButton('Test Connection')
        self.test.clicked.connect(self.test_connection)
        layout.addWidget(self.test)
        self.status = QLabel('Not tested')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        last = self.projects.get(self.project, {})
        self.last = QLabel('Last successful sync: ' + last.get('last_success', 'Never'))
        self.last.setWordWrap(True)
        layout.addWidget(self.last)
        if last.get('last_error'):
            error = QLabel('Last error: ' + last['last_error'])
            error.setWordWrap(True)
            layout.addWidget(error)
        self.sync_button = QPushButton('Sync Now')
        self.sync_button.setEnabled(self.project is not None)
        self.sync_button.clicked.connect(self.sync_now)
        layout.addWidget(self.sync_button)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                  QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.sync_requested = False

    def _path_changed(self):
        self.tested = None
        if hasattr(self, 'status'):
            self.status.setText('Not tested')

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose Sync Folder', self.folder.text())
        if folder:
            self.folder.setText(folder)

    def test_connection(self):
        folder = self.folder.text().strip()
        expected = (self.configuration.get('root_id')
                    if folder == self.configuration.get('root') else None)
        try:
            self.tested = SyncProgressDialog(
                'Test Sync Destination',
                lambda progress, cancelled: test_destination(folder, expected), self).run()
            self.status.setText('Connection verified: write, read, and cleanup succeeded.')
            return True
        except Exception as exc:
            self.tested = None
            self.status.setText(str(exc))
            return False

    def sync_now(self):
        self.enabled.setChecked(True)
        self.sync_requested = True
        self.accept()
        if self.result() != QDialog.DialogCode.Accepted:
            self.sync_requested = False

    def accept(self):
        wants_destination = bool(self.folder.text().strip())
        needs_test = wants_destination and (
            self.folder.text().strip() != self.configuration.get('root') or
            not self.configuration.get('root_id') or self.enabled.isChecked())
        if self.enabled.isChecked() and not wants_destination:
            self.status.setText('Choose and test a destination folder first.')
            return
        if needs_test and self.tested is None and not self.test_connection():
            return
        configuration = dict(self.tested or self.configuration)
        configuration['include_videos'] = self.videos.isChecked()
        save_setting('project_sync/destination', configuration)
        if self.project:
            self.projects.setdefault(self.project, {})['enabled'] = self.enabled.isChecked()
            save_setting('project_sync/projects', self.projects)
        super().accept()
