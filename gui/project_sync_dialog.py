"""Local contributor preferences and optional mounted-folder publishing UI."""

import json
from pathlib import Path
import platform
from uuid import uuid4

from PyQt6.QtCore import QSettings, QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                            QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QMessageBox, QProgressBar, QPushButton,
                            QStyle, QVBoxLayout)

from core.contributors import contributor_profile, new_session
from core.project_delta import default_rollback_root
from core.project_sync import (SyncConflictError, backup_location, restore_project,
                               test_destination, update_journal_path, _read_json, _write_json)


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
    transfer = QHBoxLayout()
    import_button = QPushButton('Import Profile...')
    export_button = QPushButton('Export Profile...')
    transfer.addWidget(import_button)
    transfer.addWidget(export_button)
    layout.addLayout(transfer)

    def import_profile():
        filename, _ = QFileDialog.getOpenFileName(dialog, 'Import Your Contributor Profile', '', 'JSON (*.json)')
        if not filename:
            return
        try:
            profile = contributor_profile(_read_json(filename))
            existing = next((key for key in profiles if key.casefold() == profile['name'].casefold()), None)
            detail = ('\nThis replaces a different locally saved ID for that name; existing annotations '
                      'will not be reattributed.' if existing and profiles[existing] != profile['id'] else '')
            if QMessageBox.question(
                    dialog, 'Use This Contributor Identity?',
                    f"Use {profile['name']} ({profile['id']}) as your identity?\n"
                    'Import only your own profile. This is attribution, not an authenticated account.' + detail,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            if existing:
                del profiles[existing]
            profiles[profile['name']] = profile['id']
            name.clear()
            name.addItems(sorted(profiles))
            name.setCurrentText(profile['name'])
        except Exception as exc:
            QMessageBox.warning(dialog, 'Profile Not Imported', str(exc))

    def export_profile():
        existing = next((key for key in profiles if key.casefold() == name.currentText().strip().casefold()), None)
        if existing is None:
            QMessageBox.information(dialog, 'Start a Session First',
                                    'Start a session with this name first, then export it via Settings > Change Contributor.')
            return
        filename, _ = QFileDialog.getSaveFileName(dialog, 'Export Your Contributor Profile',
                                                 'contributor-profile.json', 'JSON (*.json)')
        if filename:
            try:
                _write_json(filename, contributor_profile({'format': 1, 'kind': 'bumblebox-contributor',
                                                          'name': existing, 'id': profiles[existing]}))
            except Exception as exc:
                QMessageBox.warning(dialog, 'Profile Not Exported', str(exc))

    import_button.clicked.connect(import_profile)
    export_button.clicked.connect(export_profile)
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
    device_id = QSettings().value('contributors/device_id', '', type=str)
    if not device_id:
        device_id = str(uuid4())
        QSettings().setValue('contributors/device_id', device_id)
    session = new_session(display_name, profiles.get(existing),
                          device={'id': device_id, 'name': platform.node()})
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
            self.error = exc


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
            if isinstance(self.worker.error, SyncConflictError):
                raise self.worker.error
            raise RuntimeError(str(self.worker.error)) from self.worker.error
        return self.worker.result


def working_project_path(path, parent=None, default_dir=None):
    """Gate every GUI entry path before loading managers or writing debug files."""
    journal = update_journal_path(path)
    if journal.exists() or journal.is_symlink():
        if QMessageBox.question(
                parent, 'Recover Interrupted Local Update?',
                'An update marker exists for this project. Do not recover while another '
                'annotator is updating it.\n\nClose any other annotator using this local copy first. '
                'Then recover the interrupted update before opening the project?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return None
        from core.project_update import recover_update
        result = SyncProgressDialog('Recover Local Project',
            lambda progress, cancelled: recover_update(path, progress, cancelled), parent).run()
        QMessageBox.information(parent, 'Recovery Complete', result['message'])
    location = backup_location(path)
    if location is None:
        return Path(path)
    if location['kind'] != 'published':
        message = ('This contains only previous affected files, not a complete project. '
                   'Open the local working project and use File > Local Rollback History.'
                   if location['kind'] == 'rollback' else
                   'This backup is incomplete. Wait for publishing to finish.'
                   if location['kind'] == 'pending' else
                   'This is a backup container, not an editable project. Choose a completed '
                   'revisions/<version> folder (or its legacy project folder) to create a local working copy.')
        QMessageBox.warning(parent, 'Backup Not Ready to Open', message)
        return None
    manifest = location['manifest']
    question = QMessageBox(parent)
    question.setWindowTitle('Open a Local Working Copy')
    question.setText('This project is a published backup and will not be edited in place.')
    question.setInformativeText(
        f"Project: {manifest.get('project_name', location['project'].name)}\n"
        f"Contributor: {manifest['contributor']['name']}\n"
        f"Published: {manifest.get('created_at', 'Unknown')}\n\n"
        'Create a verified local copy to work on.' +
        ('\nOriginal videos were excluded; some workflows will need those videos restored separately.'
         if not manifest.get('include_videos', True) else ''))
    create = question.addButton('Create Local Working Copy', QMessageBox.ButtonRole.AcceptRole)
    cancel = question.addButton(QMessageBox.StandardButton.Cancel)
    question.setDefaultButton(cancel)
    question.exec()
    if question.clickedButton() is not create:
        return None
    destination, _ = QFileDialog.getSaveFileName(
        parent, 'Choose a New Local Project Folder', str(default_dir or Path.home()),
        options=QFileDialog.Option.DontConfirmOverwrite)
    if not destination:
        return None
    result = SyncProgressDialog(
        'Create Local Working Copy',
        lambda progress, cancelled: restore_project(path, destination, progress, cancelled), parent).run()
    return Path(result['path'])


def confirm_separate_backup(parent, error):
    question = QMessageBox(parent)
    question.setIcon(QMessageBox.Icon.Warning)
    question.setWindowTitle('Published Work Has Changed')
    question.setText(str(error))
    save = question.addButton('Save Separate Backup', QMessageBox.ButtonRole.ActionRole)
    cancel = question.addButton(QMessageBox.StandardButton.Cancel)
    question.setDefaultButton(cancel)
    question.exec()
    return question.clickedButton() is save


def confirm_publication(parent, estimate, configuration):
    box = QMessageBox(parent)
    box.setWindowTitle('Publish Backup: Storage Estimate')
    box.setTextFormat(Qt.TextFormat.PlainText)
    box.setText(f"{estimate['files']:,} files | {estimate['logical_bytes'] / 1024**3:.2f} GiB logical project size\n"
                f"Estimated additional storage: {estimate['new_bytes'] / 1024**2:.1f} MiB\n"
                f"Destination free space: {estimate['free_bytes'] / 1024**3:.2f} GiB")
    rename_notice = (
        f"\n\nAfter a successful normal publish, the existing archive '{estimate['rename_from']}' will be renamed to match "
        f"the local folder '{estimate['project_name']}', keeping its ID and history. "
        'Saved links to the old path will need updating. Use the same project folder name on other computers.'
        if estimate.get('rename_from') else '')
    box.setInformativeText(
        ('Identical files are stored once across this project\'s contributors and versions. '
         'Compact backups require this updated app to restore; keep the shared objects folder.'
         if configuration.get('storage_mode', 'deduplicated') == 'deduplicated' else
         'Each changed publication stores a complete project folder.') +
        '\nServer quotas may be lower than the reported free space. Existing archives will not be deleted.' +
        rename_notice)
    publish = box.addButton('Publish', QMessageBox.ButtonRole.AcceptRole)
    cancel = box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(cancel)
    box.exec()
    return box.clickedButton() is publish


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
        self.storage = QComboBox()
        self.storage.addItem('Deduplicated versions', 'deduplicated')
        self.storage.addItem('Full project folders (legacy)', 'full')
        self.storage.setCurrentIndex(max(0, self.storage.findData(self.configuration.get('storage_mode', 'deduplicated'))))
        form.addRow('Backup storage:', self.storage)
        self.rollback = QLineEdit(self.configuration.get('rollback_root', ''))
        self.rollback.setPlaceholderText(str(default_rollback_root()))
        self.rollback.setToolTip('Blank uses local application storage, outside the project. Choose an existing folder to override it.')
        rollback_browse = QPushButton()
        rollback_browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        rollback_browse.setToolTip('Choose local rollback folder')
        def choose_rollback():
            folder = QFileDialog.getExistingDirectory(self, 'Choose Local Rollback Folder', self.rollback.text())
            if folder:
                self.rollback.setText(folder)
        rollback_browse.clicked.connect(choose_rollback)
        rollback_row = QHBoxLayout()
        rollback_row.addWidget(self.rollback)
        rollback_row.addWidget(rollback_browse)
        form.addRow('Local rollbacks:', rollback_row)
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
        self.sync_button = QPushButton('Publish Project Backup')
        self.sync_button.setEnabled(self.project is not None)
        self.sync_button.clicked.connect(self.sync_now)
        layout.addWidget(self.sync_button)
        self.update_button = QPushButton('Update Local Copy...')
        self.update_button.setEnabled(self.project is not None)
        self.update_button.clicked.connect(self.update_now)
        layout.addWidget(self.update_button)
        if last.get('last_update'):
            layout.addWidget(QLabel('Last local update: ' + last['last_update']))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                  QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.sync_requested = False
        self.update_requested = False
        from core.assigned_work import workspace
        if workspace(project):
            self.sync_button.setText('Publish Work')
            self.update_button.setEnabled(False)
            self.enabled.setChecked(True)
            for control in (self.enabled, self.videos, self.storage):
                control.setEnabled(False)

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

    def update_now(self):
        self.update_requested = True
        self.accept()
        if self.result() != QDialog.DialogCode.Accepted:
            self.update_requested = False

    def accept(self):
        wants_destination = bool(self.folder.text().strip())
        needs_test = wants_destination and (
            self.folder.text().strip() != self.configuration.get('root') or
            not self.configuration.get('root_id') or self.enabled.isChecked() or self.update_requested)
        if (self.enabled.isChecked() or self.update_requested) and not wants_destination:
            self.status.setText('Choose and test a destination folder first.')
            return
        if needs_test and self.tested is None and not self.test_connection():
            return
        configuration = dict(self.tested or self.configuration)
        configuration['include_videos'] = self.videos.isChecked()
        configuration['storage_mode'] = self.storage.currentData()
        configuration['rollback_root'] = self.rollback.text().strip()
        if configuration['rollback_root'] and self.project:
            from core.project_delta import rollback_root
            try:
                rollback_root(configuration, self.project)
            except ValueError as exc:
                self.status.setText(str(exc))
                return
        save_setting('project_sync/destination', configuration)
        if self.project:
            self.projects.setdefault(self.project, {})['enabled'] = self.enabled.isChecked()
            save_setting('project_sync/projects', self.projects)
        super().accept()
