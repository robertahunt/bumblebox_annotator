"""Opt-in shared profile discovery, with a local cache and no session switching."""

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox,
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QStyle, QVBoxLayout)

from core.shared_profiles import sync_profiles
from core.project_sync import test_destination
from gui import project_sync_dialog as preferences


class ProfileSyncService(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.status = 'Profile sharing is off'
        self.stopping = False
        self.again = False
        self.timer = QTimer(self)
        self.timer.setInterval(60000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    def refresh(self):
        if self.stopping:
            return
        if self.worker:
            self.again = True
            if self.context != preferences.setting_json('contributors/shared', {}):
                self.worker.requestInterruption()
            return
        configuration = preferences.setting_json('contributors/shared', {})
        if not configuration.get('enabled'):
            self.status = 'Profile sharing is off'
            self.changed.emit()
            return
        profiles = preferences.load_contributor_profiles()
        outgoing = [profiles[identity] for identity in configuration.get('publish_ids', [])
                    if identity in profiles]
        self.context = configuration
        self.worker = preferences.OperationWorker(
            lambda progress, cancelled: sync_profiles(configuration, outgoing, cancelled), self)
        self.worker.finished.connect(self.finished)
        self.status = 'Checking shared profiles... Local profiles are available.'
        self.changed.emit()
        self.worker.start()

    def finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        if self.stopping:
            return
        if self.context != preferences.setting_json('contributors/shared', {}):
            self.again = False
            self.refresh()
            return
        if worker.error:
            self.status = 'Shared profiles unavailable; using saved profiles. ' + str(worker.error)
        else:
            profiles = preferences.load_contributor_profiles()
            warnings = list(worker.result['warnings'])
            for identity, profile in worker.result['profiles'].items():
                if identity in profiles and profiles[identity] != profile:
                    warnings.append(f'{identity[:8]} has a different shared name. The local profile was kept.')
                else:
                    profiles[identity] = profile
            preferences.save_contributor_profiles(profiles)
            pending = set(self.context.get('publish_ids', [])) - set(worker.result['published'])
            self.status = f"Shared profiles up to date | {len(worker.result['profiles'])} available"
            if pending or warnings:
                self.status = f'Profile sharing needs attention | {len(pending)} pending'
                if warnings:
                    self.status += '\n' + '\n'.join(warnings[:3])
            # This updates identities only, never last_id, devices, sessions, or project roles.
        self.changed.emit()
        if self.again:
            self.again = False
            QTimer.singleShot(0, self.refresh)

    def stop(self):
        self.stopping = True
        self.timer.stop()
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait()


def profile_sync_service():
    app = QApplication.instance()
    if not hasattr(app, '_profile_sync_service') or app._profile_sync_service.stopping:
        app._profile_sync_service = ProfileSyncService(app)
        app.aboutToQuit.connect(app._profile_sync_service.stop)
    return app._profile_sync_service


def stop_profile_sync():
    service = getattr(QApplication.instance(), '_profile_sync_service', None)
    if service is not None:
        service.stop()


class SharedProfileSettingsDialog(QDialog):
    def __init__(self, parent=None, *, suggested=None):
        super().__init__(parent)
        self.setWindowTitle('Shared Contributor Profiles')
        self.resize(660, 520)
        self.setMinimumSize(460, 400)
        self.configuration = preferences.setting_json('contributors/shared', {})
        destination = (self.configuration if self.configuration.get('root') else
                       suggested or preferences.setting_json('project_sync/destination', {}))
        self.tested = None
        layout = QVBoxLayout(self)
        self.enabled = QCheckBox('Share contributor profiles')
        self.enabled.setChecked(self.configuration.get('enabled', False))
        layout.addWidget(self.enabled)
        self.folder = QLineEdit(destination.get('root', ''))
        self.folder.setPlaceholderText('Shared folder')
        self.folder.setToolTip('The shared archive root, not a project folder. Mount the drive first.')
        self.folder.textChanged.connect(lambda: setattr(self, 'tested', None))
        row = QHBoxLayout()
        row.addWidget(self.folder, 1)
        browse = QPushButton()
        browse.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        browse.setToolTip('Choose shared folder')
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        layout.addLayout(row)
        test = QPushButton('Test Connection')
        test.clicked.connect(self.test_connection)
        layout.addWidget(test)
        title = QLabel('Existing profiles to share')
        layout.addWidget(title)
        self.profiles = QListWidget()
        published = self.configuration.get('publish_ids', [])
        for identity, profile in sorted(preferences.load_contributor_profiles().items(),
                                        key=lambda entry: (entry[1]['name'].casefold(), entry[0])):
            item = QListWidgetItem(f"{profile['name']}  ({identity[:8]})")
            item.setData(Qt.ItemDataRole.UserRole, identity)
            item.setToolTip(identity)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if identity in published else Qt.CheckState.Unchecked)
            self.profiles.addItem(item)
        layout.addWidget(self.profiles, 1)
        self.share_new = QCheckBox('Automatically share new profiles created on this computer')
        self.share_new.setChecked(self.configuration.get('share_new', True))
        layout.addWidget(self.share_new)
        note = QLabel('Shares names and contributor IDs only. Everyone with access to this folder can '
                      'see and select these identities; this is not a secure login. Unchecking a profile '
                      'stops publishing it but does not remove its shared copy.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse(self):
        folder = QFileDialog.getExistingDirectory(self, 'Choose Shared Folder', self.folder.text())
        if folder:
            self.folder.setText(folder)

    def test_connection(self):
        folder = self.folder.text().strip()
        expected = self.configuration.get('root_id') if folder == self.configuration.get('root') else None
        try:
            self.tested = preferences.SyncProgressDialog('Test Profile Sharing',
                lambda progress, cancelled: test_destination(folder, expected), self).run()
            self.status.setText('Connection verified')
            return True
        except Exception as exc:
            self.tested = None
            self.status.setText(str(exc))
            return False

    def accept(self):
        configuration = dict(self.configuration)
        if self.enabled.isChecked():
            same = (configuration.get('root_id') and self.folder.text().strip() == configuration.get('root'))
            if not same and self.tested is None and not self.test_connection():
                return
            configuration.update(self.tested or {})
        configuration.update(enabled=self.enabled.isChecked(), share_new=self.share_new.isChecked(),
            publish_ids=[self.profiles.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.profiles.count())
                         if self.profiles.item(i).checkState() == Qt.CheckState.Checked])
        preferences.save_setting('contributors/shared', configuration)
        profile_sync_service().refresh()
        super().accept()
