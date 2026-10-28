"""Local contributor preferences and optional mounted-folder publishing UI."""

import hashlib
import json
from pathlib import Path
import platform
from uuid import uuid4

from PyQt6.QtCore import QSettings, QSize, QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap, QTextOption
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                            QDialogButtonBox, QFileDialog, QFormLayout, QFrame,
                            QHBoxLayout, QLabel, QLineEdit, QListWidget,
                            QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar,
                            QPushButton, QScrollArea, QStyle, QVBoxLayout, QWidget)

from core.contributors import contributor_profile, new_session
from core.project_delta import default_rollback_root
from core.project_sync import (SyncConflictError, backup_location, restore_project,
                               test_destination, update_journal_path, _read_json, _write_json,
                               publication_estimate_matches)
from gui.workflow_help import ASSIGNED_LABEL, BACKUP_LABEL, UPDATE_LABEL, add_help_row


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


def load_contributor_profiles():
    """Migrate name-keyed preferences without losing duplicate display names."""
    profiles = {}
    stored = setting_json('contributors/identities', {})
    legacy = setting_json('contributors/profiles', {})
    candidates = list(stored.values()) if isinstance(stored, dict) else []
    if isinstance(legacy, dict):
        candidates += [{'id': identity, 'name': name} for name, identity in legacy.items()]
    imported = setting_json('assignments/profiles', {})
    if isinstance(imported, dict):
        candidates += list(imported.values())
    for value in candidates:
        try:
            profile = contributor_profile(dict(value, format=1, kind='bumblebox-contributor'))
            profiles.setdefault(profile['id'], profile)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
    return profiles


def save_contributor_profiles(profiles):
    save_setting('contributors/identities', profiles)
    # Retain the old preference for older app versions; the ID-keyed store is authoritative.
    legacy = {}
    for profile in profiles.values():
        legacy.setdefault(profile['name'], profile['id'])
    save_setting('contributors/profiles', legacy)


def remember_contributor_profile(profile, *, newly_created=False):
    profile = contributor_profile(dict(profile, format=1, kind='bumblebox-contributor'))
    profiles = load_contributor_profiles()
    if profile['id'] in profiles and profiles[profile['id']] != profile:
        raise ValueError('This contributor ID already has a different local name. '
                         'The existing identity was not overwritten.')
    profiles.setdefault(profile['id'], profile)
    save_contributor_profiles(profiles)
    sharing = setting_json('contributors/shared', {})
    if newly_created and sharing.get('enabled') and sharing.get('share_new', True):
        sharing['publish_ids'] = sorted(set(sharing.get('publish_ids', [])) | {profile['id']})
        save_setting('contributors/shared', sharing)


def export_contributor_profile(profile, parent=None):
    """Export identity only; session/device details and credentials are never included."""
    filename, _ = QFileDialog.getSaveFileName(parent, 'Export My Contributor Profile',
                                             'contributor-profile.json', 'JSON (*.json)')
    if not filename:
        return
    try:
        _write_json(filename, contributor_profile(dict(profile, format=1, kind='bumblebox-contributor')))
        QMessageBox.information(parent, 'Profile Exported',
            'This file contains only your contributor name and ID, plus its format information. '
            'Send it to your project director so they can assign work to this profile. '
            'Nothing was sent automatically.')
    except Exception as exc:
        QMessageBox.warning(parent, 'Profile Not Exported', str(exc))


def _contributor_avatar(name, identity):
    colors = ('#26776d', '#a44b51', '#426eaa', '#716099', '#94702e', '#386f43')
    color = colors[hashlib.sha256(identity.encode()).digest()[0] % len(colors)]
    parts = name.split()
    initials = (parts[0][0] + parts[-1][0] if len(parts) > 1 else parts[0][:2]).upper()[:2]
    pixmap = QPixmap(128, 128)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(2, 2, 60, 60)
    font = painter.font()
    font.setPixelSize(22)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(Qt.GlobalColor.white)
    painter.drawText(2, 2, 60, 60, Qt.AlignmentFlag.AlignCenter, initials)
    painter.end()
    icon = QIcon(pixmap)
    icon.addPixmap(pixmap, QIcon.Mode.Selected)
    return icon


def choose_contributor(parent=None):
    """Confirm a saved or new identity before starting an attribution session."""
    profiles = load_contributor_profiles()
    dialog = QDialog(parent)
    dialog.setWindowTitle('Choose Contributor Profile')
    dialog.resize(560, 500)
    dialog.setMinimumSize(460, 400)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(20, 20, 20, 16)
    layout.setSpacing(12)
    heading = QLabel("Who's annotating?")
    heading_font = heading.font()
    heading_font.setPointSize(16)
    heading_font.setBold(True)
    heading.setFont(heading_font)
    layout.addWidget(heading)
    empty = QLabel('No saved profiles')
    layout.addWidget(empty)
    tiles = QListWidget()
    tiles.setObjectName('contributorProfiles')
    tiles.setAccessibleName('Contributor profiles')
    tiles.setCursor(Qt.CursorShape.PointingHandCursor)
    tiles.setViewMode(QListWidget.ViewMode.IconMode)
    tiles.setResizeMode(QListWidget.ResizeMode.Adjust)
    tiles.setMovement(QListWidget.Movement.Static)
    tiles.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    tiles.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    tiles.setDragEnabled(False)
    tiles.setIconSize(QSize(64, 64))
    tiles.setGridSize(QSize(144, 144))
    tiles.setWordWrap(True)
    tiles.setTextElideMode(Qt.TextElideMode.ElideRight)
    tiles.setFrameShape(QFrame.Shape.NoFrame)
    tiles.setStyleSheet('''
        QListWidget#contributorProfiles { background: transparent; }
        QListWidget#contributorProfiles::item {
            border: 1px solid palette(mid); border-radius: 6px;
            padding: 8px; margin: 4px;
        }
        QListWidget#contributorProfiles::item:hover { background: palette(alternate-base); }
        QListWidget#contributorProfiles::item:selected {
            background: palette(highlight); color: palette(highlighted-text);
        }
    ''')
    layout.addWidget(tiles, 1)
    transfer = QHBoxLayout()
    new_button = QPushButton('New Profile...')
    import_button = QPushButton('Import Profile...')
    export_button = QPushButton('Export Profile...')
    for button, icon in ((new_button, QStyle.StandardPixmap.SP_FileIcon),
                         (import_button, QStyle.StandardPixmap.SP_DialogOpenButton),
                         (export_button, QStyle.StandardPixmap.SP_DialogSaveButton)):
        button.setIcon(dialog.style().standardIcon(icon))
        button.setAutoDefault(False)
        transfer.addWidget(button)
    export_button.setToolTip('Export the selected contributor profile')
    layout.addLayout(transfer)

    from gui.shared_profiles import profile_sync_service, LabConnectionDialog
    service = profile_sync_service()
    sharing_row = QHBoxLayout()
    sharing_button = QPushButton('Connect to Lab...')
    sharing_button.setAutoDefault(False)
    sharing_button.setIcon(dialog.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
    refresh_button = QPushButton()
    refresh_button.setIcon(dialog.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
    refresh_button.setToolTip('Refresh shared profiles')
    refresh_button.setAutoDefault(False)
    hide_button = QPushButton('Hide Profile')
    hide_button.setAutoDefault(False)
    hide_button.setToolTip('Hide only on this computer. Shared profiles and annotation attribution are unchanged.')
    for control in (sharing_button, refresh_button, hide_button):
        sharing_row.addWidget(control)
    layout.addLayout(sharing_row)
    show_hidden = QCheckBox('Show hidden profiles')
    layout.addWidget(show_hidden)
    sharing_status = QLabel()
    sharing_status.setWordWrap(True)
    sharing_status.setTextFormat(Qt.TextFormat.PlainText)
    layout.addWidget(sharing_status)

    def selected_name():
        item = tiles.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh_profiles(selected=None):
        tiles.clear()
        hidden = setting_json('contributors/hidden', [])
        names = [profile['name'].casefold() for profile in profiles.values()]
        for identity, profile in sorted(profiles.items(), key=lambda entry: (entry[1]['name'].casefold(), entry[0])):
            if identity in hidden and not show_hidden.isChecked():
                continue
            display_name = profile['name']
            label = (f'{display_name}\n{identity[:8]}' if names.count(display_name.casefold()) > 1
                     else display_name)
            if identity in hidden:
                label += '\nHidden locally'
            item = QListWidgetItem(_contributor_avatar(display_name, identity), label)
            item.setData(Qt.ItemDataRole.UserRole, identity)
            item.setToolTip(label)
            item.setSizeHint(QSize(136, 136))
            tiles.addItem(item)
            if identity == selected:
                tiles.setCurrentItem(item)
        empty.setVisible(not tiles.count())
        if tiles.currentItem():
            tiles.scrollToItem(tiles.currentItem())

    def create_profile():
        editor = QDialog(dialog)
        editor.setWindowTitle('New Contributor Profile')
        form = QFormLayout(editor)
        name = QLineEdit()
        name.setMaxLength(80)
        name.setMinimumWidth(280)
        form.addRow('Contributor name:', name)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        create = actions.button(QDialogButtonBox.StandardButton.Ok)
        create.setText('Create and Continue')
        create.setEnabled(False)
        name.textChanged.connect(lambda text: create.setEnabled(bool(text.strip())))
        actions.accepted.connect(editor.accept)
        actions.rejected.connect(editor.reject)
        form.addRow(actions)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        display_name = name.text().strip()
        existing = [profile for profile in profiles.values() if profile['name'].casefold() == display_name.casefold()]
        if existing:
            if QMessageBox.question(editor, 'Create a Separate Identity?',
                    f'A profile named {display_name} already exists. Select its tile if it is yours.\n\n'
                    'Create a different person with the same name? Their attribution will remain separate.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                refresh_profiles(existing[0]['id'])
                return
        identity = str(uuid4())
        profile = {'id': identity, 'name': display_name}
        remember_contributor_profile(profile, newly_created=True)
        profiles.update(load_contributor_profiles())
        refresh_profiles(identity)
        service.refresh()
        dialog.accept()

    def import_profile():
        filename, _ = QFileDialog.getOpenFileName(dialog, 'Import Your Contributor Profile', '', 'JSON (*.json)')
        if not filename:
            return
        try:
            profile = contributor_profile(_read_json(filename))
            detail = '\nOther profiles with this name are kept separate; no existing annotations are reattributed.'
            if QMessageBox.question(
                    dialog, 'Use This Contributor Identity?',
                    f"Use {profile['name']} ({profile['id']}) as your identity?\n"
                    'Import only your own profile. This is attribution, not an authenticated account.' + detail,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
            remember_contributor_profile(profile)
            profiles.update(load_contributor_profiles())
            refresh_profiles(profile['id'])
        except Exception as exc:
            QMessageBox.warning(dialog, 'Profile Not Imported', str(exc))

    def export_profile():
        existing = selected_name()
        if existing is None:
            return
        export_contributor_profile(profiles[existing], dialog)

    def update_shared():
        selected = selected_name()
        profiles.update(load_contributor_profiles())
        refresh_profiles(selected)
        sharing_status.setText(service.status)
        refresh_button.setEnabled(service.worker is None)

    def configure_sharing():
        LabConnectionDialog(dialog).exec()
        update_shared()

    def toggle_hidden():
        identity = selected_name()
        if not identity:
            return
        hidden = set(setting_json('contributors/hidden', []))
        if identity in hidden:
            hidden.remove(identity)
        else:
            hidden.add(identity)
        save_setting('contributors/hidden', sorted(hidden))
        refresh_profiles(identity)

    sharing_button.clicked.connect(configure_sharing)
    refresh_button.clicked.connect(service.refresh)
    hide_button.clicked.connect(toggle_hidden)
    show_hidden.toggled.connect(lambda: refresh_profiles(selected_name()))
    service.changed.connect(update_shared)

    new_button.clicked.connect(create_profile)
    import_button.clicked.connect(import_profile)
    export_button.clicked.connect(export_profile)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
    proceed = buttons.button(QDialogButtonBox.StandardButton.Ok)
    proceed.setText('Continue')
    proceed.setEnabled(False)
    export_button.setEnabled(False)
    tiles.currentItemChanged.connect(lambda current, previous: proceed.setEnabled(current is not None))
    tiles.currentItemChanged.connect(lambda current, previous: export_button.setEnabled(current is not None))
    def update_hide(current, previous):
        hide_button.setEnabled(current is not None)
        hide_button.setText('Show Profile' if current and current.data(Qt.ItemDataRole.UserRole) in
                            setting_json('contributors/hidden', []) else 'Hide Profile')
    tiles.currentItemChanged.connect(update_hide)
    tiles.itemClicked.connect(lambda item: dialog.accept())
    tiles.itemActivated.connect(lambda item: dialog.accept())
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    last_id = QSettings().value('contributors/last_id', '', type=str)
    if not last_id:
        last_name = QSettings().value('contributors/last_name', '', type=str)
        matches = [p['id'] for p in profiles.values() if p['name'] == last_name]
        last_id = matches[0] if len(matches) == 1 else None
    refresh_profiles(last_id)
    sharing_status.setText(service.status)
    QTimer.singleShot(0, service.refresh)
    tiles.setFocus()
    result = dialog.exec()
    service.changed.disconnect(update_shared)
    if result != QDialog.DialogCode.Accepted:
        return None
    identity = selected_name()
    if identity is None:
        return None
    profile = profiles[identity]
    device_id = QSettings().value('contributors/device_id', '', type=str)
    if not device_id:
        device_id = str(uuid4())
        QSettings().setValue('contributors/device_id', device_id)
    session = new_session(profile['name'], identity,
                          device={'id': device_id, 'name': platform.node()})
    remember_contributor_profile(profile)
    QSettings().setValue('contributors/last_name', profile['name'])
    QSettings().setValue('contributors/last_id', identity)
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


def _size_transfer_dialog(dialog, width, height, minimum_height=260):
    available = dialog.screen().availableGeometry()
    width = min(width, max(1, available.width() - 64))
    height = min(height, max(1, available.height() - 64))
    dialog.setMinimumSize(min(640, width), min(minimum_height, height))
    dialog.resize(width, height)
    dialog.setSizeGripEnabled(True)


def _transfer_text(text):
    message = QPlainTextEdit()
    message.setReadOnly(True)
    message.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
    message.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    message.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    font = message.font()
    font.setPointSizeF(max(11.0, font.pointSizeF()))
    message.setFont(font)
    message.setPlainText(text)
    return message


class TransferConfirmationDialog(QDialog):
    def __init__(self, title, text, accept_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.message = _transfer_text(text)
        self.message.setAccessibleName('Transfer details')
        layout.addWidget(self.message, 1)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                        QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(accept_text)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setAutoDefault(False)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setDefault(True)
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setFocus()
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        _size_transfer_dialog(self, 860, 440)


class SyncProgressDialog(QDialog):
    def __init__(self, title, operation, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.message = _transfer_text('Preparing...')
        self.message.setAccessibleName('Transfer status')
        layout.addWidget(self.message, 1)
        progress = QProgressBar()
        progress.setRange(0, 0)
        layout.addWidget(progress)
        self.cancel = QPushButton('Cancel')
        self.cancel.clicked.connect(self.reject)
        layout.addWidget(self.cancel, 0, Qt.AlignmentFlag.AlignRight)
        self.worker = OperationWorker(operation, self)
        self.worker.progress.connect(self.message.setPlainText)
        self.worker.finished.connect(self.accept)
        _size_transfer_dialog(self, 860, 340)

    def reject(self):
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.cancel.setEnabled(False)
            self.message.setPlainText('Cancelling after the current file operation...')
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


def connection_test_result(parent, status, folder, *, error=None, report=False):
    """Explicit tests retain a result; implicit setup checks only update the form."""
    title = 'Connection Test Failed' if error is not None else 'Connection Test Passed'
    if error is not None:
        details = str(error)
    else:
        details = 'Connection verified: write, read, and cleanup succeeded. No project files were transferred.'
    message = f'{details}\n\nFolder: {folder or "(not selected)"}'
    status.setText(message)
    status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    if report:
        box = QMessageBox(parent)
        box.setWindowTitle(title)
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setIcon(QMessageBox.Icon.Warning if error is not None else QMessageBox.Icon.Information)
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()


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
        'Create a verified copy in a new local folder. This restores only the selected version; '
        'it does not merge contributors\' work or upload local changes. Confirm this is the '
        'version your project director intends you to use.' +
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
    question = TransferConfirmationDialog('Published Work Has Changed', str(error),
                                          'Save Separate Backup', parent)
    return question.exec() == QDialog.DialogCode.Accepted


def publication_summary(estimate):
    return (f"Project inventory: {estimate['files']:,} files | "
            f"{estimate['logical_bytes'] / 1024**3:.2f} GiB total (not upload size)\n"
            f"Estimated additional storage: {estimate['new_bytes'] / 1024**2:.1f} MiB\n"
            f"Destination free space: {estimate['free_bytes'] / 1024**3:.2f} GiB")


def show_publication_review(estimate, parent=None):
    from gui.publication_review_dialog import PublicationReviewDialog
    PublicationReviewDialog(estimate, parent).exec()


def confirm_publication(parent, estimate, configuration):
    rename_notice = (
        f"\n\nAfter a successful normal publish, the existing archive '{estimate['rename_from']}' will be renamed to match "
        f"the local folder '{estimate['project_name']}', keeping its ID and history. "
        'Saved links to the old path will need updating. Use the same project folder name on other computers.'
        if estimate.get('rename_from') else '')
    details = (
        ('Identical files are stored once across this project\'s contributors and versions. '
         'Compact backups require this updated app to restore; keep the shared objects folder.'
         if configuration.get('storage_mode', 'deduplicated') == 'deduplicated' else
         'Each changed publication stores a complete project folder.') +
        '\nServer quotas may be lower than the reported free space. Existing archives will not be deleted.' +
        '\nNew contents are verified after upload. Unchanged stored files get presence and size checks, '
        'not a full content check. Use Check Backup Integrity for a full check.' +
        rename_notice)
    box = TransferConfirmationDialog('Back Up Local Project: Storage Estimate',
                                      publication_summary(estimate) + '\n\n' + details,
                                      BACKUP_LABEL, parent)
    review = box.buttons.addButton('Review Changes...', QDialogButtonBox.ButtonRole.ActionRole)
    review.setIcon(box.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
    review.setAutoDefault(False)
    review.setEnabled('review' in estimate)
    review.clicked.connect(lambda: show_publication_review(estimate, box))
    return box.exec() == QDialog.DialogCode.Accepted


class ProjectSyncDialog(QDialog):
    def __init__(self, project=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Project Sync')
        self.project = str(Path(project).resolve()) if project else None
        self.configuration = setting_json('project_sync/destination', {})
        self.projects = setting_json('project_sync/projects', {})
        self.tested = None
        self.transfer_estimate = None
        self.estimator = getattr(parent, 'estimate_project_backup', None)
        self.contributor = getattr(parent, 'contributor_session', None)
        outer = QVBoxLayout(self)
        contents = QWidget()
        layout = QVBoxLayout(contents)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(contents)
        outer.addWidget(scroll, 1)
        form = QFormLayout()
        self.enabled = QCheckBox('Enable sync for this project')
        self.enabled.setEnabled(self.project is not None)
        self.enabled.setChecked(self.projects.get(self.project, {}).get('enabled', False))
        form.addRow('Project:', QLabel(Path(self.project).name if self.project else 'No project open'))
        form.addRow('', self.enabled)
        self.folder = QLineEdit(self.configuration.get('root', ''))
        self.folder.setToolTip('An existing mounted or local folder. Kerberos sign-in opens a separate terminal.')
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
        self.test.clicked.connect(lambda: self.test_connection(report=True))
        layout.addWidget(self.test)
        from gui.drive_login import DriveLoginButton
        self.login = DriveLoginButton(self)
        self.login.completed.connect(lambda: self.test_connection(report=True))
        layout.addWidget(self.login)
        self.status = QLabel('Not tested')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        profiles_button = QPushButton('Connect Profiles and Projects to Lab...')
        profiles_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DriveNetIcon))
        profiles_button.setToolTip('Use one shared folder for contributor profiles and project discovery')
        def configure_profiles():
            from gui.shared_profiles import LabConnectionDialog
            dialog = LabConnectionDialog(self, suggested=self.pending_configuration())
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.configuration = setting_json('project_sync/destination', {})
                self.folder.setText(self.configuration.get('root', ''))
                self.tested = None
                self.invalidate_estimate()
        profiles_button.clicked.connect(configure_profiles)
        layout.addWidget(profiles_button)
        last = self.projects.get(self.project, {})
        self.last = QLabel('Last successful publication: ' + last.get('last_success', 'Never'))
        self.last.setWordWrap(True)
        layout.addWidget(self.last)
        integrity_button = QPushButton('Check Backup Integrity...')
        integrity_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        integrity_button.setEnabled(self.project is not None)
        integrity_button.setToolTip('View version-specific integrity results and run a full content check')
        def integrity():
            from gui.backup_integrity_dialog import open_integrity_dialog
            open_integrity_dialog(self.project, self.pending_configuration(), self)
        integrity_button.clicked.connect(integrity)
        layout.addWidget(integrity_button)
        integrity_note = QLabel('Publication verifies uploaded content, not all unchanged backup content. '
                               'Review full-check status separately in Check Backup Integrity.')
        integrity_note.setWordWrap(True)
        layout.addWidget(integrity_note)
        if last.get('last_error'):
            error = QLabel('Last error: ' + last['last_error'])
            error.setTextFormat(Qt.TextFormat.PlainText)
            error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            error.setWordWrap(True)
            layout.addWidget(error)
        self.estimate_button = QPushButton('Estimate Transfer Size')
        self.estimate_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.estimate_button.setEnabled(self.project is not None and callable(self.estimator))
        self.estimate_button.clicked.connect(self.estimate_transfer)
        self.estimate_help = add_help_row(layout, self.estimate_button, 'estimate')
        self.estimate_status = QLabel('Transfer size not estimated')
        self.estimate_status.setTextFormat(Qt.TextFormat.PlainText)
        self.estimate_status.setWordWrap(True)
        layout.addWidget(self.estimate_status)
        self.review_button = QPushButton('Review Changes...')
        self.review_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.review_button.setEnabled(False)
        self.review_button.setToolTip('Estimate transfer size before reviewing outgoing changes')
        self.review_button.clicked.connect(self.review_transfer)
        self.review_help = add_help_row(layout, self.review_button, 'review')
        self.sync_button = QPushButton(BACKUP_LABEL)
        self.sync_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp))
        self.sync_button.setEnabled(False)
        self.sync_button.setToolTip('Estimate transfer size before publishing')
        self.sync_button.clicked.connect(self.sync_now)
        from core.assigned_work import workspace
        self.assigned = bool(workspace(project))
        add_help_row(layout, self.sync_button, lambda: 'assigned' if self.assigned else 'backup')
        self.update_button = QPushButton(UPDATE_LABEL)
        self.update_button.setEnabled(self.project is not None)
        self.update_button.clicked.connect(self.update_now)
        add_help_row(layout, self.update_button, 'update')
        if last.get('last_update'):
            layout.addWidget(QLabel('Last local update: ' + last['last_update']))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                  QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addStretch()
        outer.addWidget(buttons)
        self.sync_requested = False
        self.update_requested = False
        if self.assigned:
            self.sync_button.setText(ASSIGNED_LABEL)
            self.sync_button.setEnabled(True)
            self.sync_button.setToolTip('Publish assigned annotation work')
            self.estimate_button.hide()
            self.estimate_help.hide()
            self.estimate_status.hide()
            self.review_button.hide()
            self.review_help.hide()
            self.update_button.setEnabled(False)
            self.enabled.setChecked(True)
            for control in (self.enabled, self.videos, self.storage):
                control.setEnabled(False)
        self.videos.toggled.connect(self.invalidate_estimate)
        self.storage.currentIndexChanged.connect(self.invalidate_estimate)
        self.enabled.toggled.connect(self.invalidate_estimate)
        _size_transfer_dialog(self, 860, 660, minimum_height=420)

    def invalidate_estimate(self):
        self.transfer_estimate = None
        if hasattr(self, 'estimate_status'):
            self.estimate_status.setText('Transfer size not estimated')
            self.review_button.setEnabled(False)
            self.sync_button.setEnabled(getattr(self, 'assigned', False))
            if not getattr(self, 'assigned', False):
                self.sync_button.setToolTip('Estimate transfer size before publishing')
        parent = self.parent()
        if parent and hasattr(parent, 'invalidate_backup_estimate'):
            parent.invalidate_backup_estimate()

    def pending_configuration(self):
        configuration = dict(self.tested or self.configuration)
        if self.tested is None:
            configuration['root'] = self.folder.text().strip()
        configuration.update(include_videos=self.videos.isChecked(),
                             storage_mode=self.storage.currentData(), rollback_root=self.rollback.text().strip())
        return configuration

    def estimate_transfer(self):
        self.invalidate_estimate()
        if not self.project or not callable(self.estimator):
            return
        self.enabled.setChecked(True)
        if not self.test_connection():
            return
        self.transfer_estimate = self.estimator(self.pending_configuration(), parent=self)
        if self.transfer_estimate:
            self.estimate_status.setText(publication_summary(self.transfer_estimate))
            self.review_button.setEnabled('review' in self.transfer_estimate)
            self.sync_button.setEnabled(True)
            self.sync_button.setToolTip('Publish the estimated backup')

    def review_transfer(self):
        if not self.transfer_estimate or 'review' not in self.transfer_estimate:
            return
        if self._estimate_is_current():
            show_publication_review(self.transfer_estimate, self)

    def _estimate_is_current(self):
        try:
            current = SyncProgressDialog('Check Transfer Estimate',
                lambda progress, cancelled: publication_estimate_matches(
                    self.transfer_estimate, self.project, self.pending_configuration(), self.contributor), self).run()
            if not current:
                self.invalidate_estimate()
                self.estimate_status.setText('Project files changed. Estimate transfer size again.')
            return current
        except Exception as exc:
            self.invalidate_estimate()
            self.estimate_status.setText(str(exc))
            return False

    def _path_changed(self):
        self.tested = None
        self.invalidate_estimate()
        if hasattr(self, 'status'):
            self.status.setText('Not tested')

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose Sync Folder', self.folder.text())
        if folder:
            self.folder.setText(folder)

    def test_connection(self, *, report=False):
        folder = self.folder.text().strip()
        expected = (self.configuration.get('root_id')
                    if folder == self.configuration.get('root') else None)
        try:
            self.tested = SyncProgressDialog(
                'Test Sync Destination',
                lambda progress, cancelled: test_destination(folder, expected), self).run()
            connection_test_result(self, self.status, self.tested['root'], report=report)
            return True
        except Exception as exc:
            self.tested = None
            self.invalidate_estimate()
            connection_test_result(self, self.status, folder, error=exc, report=report)
            return False

    def sync_now(self):
        if not self.assigned:
            if not self.transfer_estimate or not self._estimate_is_current():
                return
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
