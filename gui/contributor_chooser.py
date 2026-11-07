"""Identity selection, with infrequent profile administration in its own dialog."""

import platform
from uuid import uuid4

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox,
                            QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
                            QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                            QPushButton, QStyle, QToolButton, QVBoxLayout, QWidget)

from core.contributors import contributor_profile, new_session
from gui import project_sync_dialog as preferences
from gui.shared_profiles import profile_sync_service


class ManageProfilesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Manage Profiles')
        self.resize(580, 460)
        self.setMinimumSize(460, 360)
        self.service = profile_sync_service()
        layout = QVBoxLayout(self)
        self.people = QListWidget()
        self.people.setAccessibleName('Profiles on this computer')
        layout.addWidget(self.people, 1)
        self.removed = QCheckBox('Show removed profiles')
        self.removed.toggled.connect(self.refresh)
        layout.addWidget(self.removed)
        row = QHBoxLayout()
        self.new = QPushButton('New Profile...')
        self.new.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.new.clicked.connect(self.create_profile)
        row.addWidget(self.new)
        self.remove = QPushButton('Remove from This List...')
        self.remove.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon))
        self.remove.clicked.connect(self.toggle_removed)
        row.addWidget(self.remove)
        row.addStretch()
        layout.addLayout(row)
        self.advanced = QToolButton()
        self.advanced.setText('Advanced')
        self.advanced.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced.setCheckable(True)
        layout.addWidget(self.advanced)
        self.extra = QWidget()
        extra = QVBoxLayout(self.extra)
        extra.setContentsMargins(0, 0, 0, 0)
        transfers = QHBoxLayout()
        self.import_button = QPushButton('Import Profile...')
        self.import_button.clicked.connect(self.import_profile)
        self.export_button = QPushButton('Export Profile...')
        self.export_button.clicked.connect(self.export_profile)
        self.refresh_button = QToolButton()
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        self.refresh_button.setToolTip('Refresh shared profiles')
        self.refresh_button.setAccessibleName('Refresh shared profiles')
        self.refresh_button.clicked.connect(self.service.refresh)
        for control in (self.import_button, self.export_button, self.refresh_button):
            transfers.addWidget(control)
        extra.addLayout(transfers)
        self.status = QLabel()
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        extra.addWidget(self.status)
        layout.addWidget(self.extra)
        self.extra.hide()
        self.advanced.toggled.connect(self.show_advanced)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self.people.currentItemChanged.connect(self.update_actions)
        self.service.changed.connect(self.refresh)
        self.finished.connect(lambda: self.service.changed.disconnect(self.refresh))
        self.refresh()

    def selected_id(self):
        item = self.people.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh(self, *_, selected=None):
        identity = selected or self.selected_id()
        self.people.clear()
        self.profiles = preferences.load_contributor_profiles()
        hidden = preferences.setting_json('contributors/hidden', [])
        names = [p['name'].casefold() for p in self.profiles.values()]
        for key, profile in sorted(self.profiles.items(), key=lambda pair: (pair[1]['name'].casefold(), pair[0])):
            if key in hidden and not self.removed.isChecked():
                continue
            name = profile['name']
            if names.count(name.casefold()) > 1:
                name += ' (' + key[:8] + ')'
            if key in hidden:
                name += ' - removed locally'
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.people.addItem(item)
            if key == identity:
                self.people.setCurrentItem(item)
        self.status.setText(self.service.status)
        self.update_actions()

    def update_actions(self, *_):
        identity = self.selected_id()
        hidden = identity in preferences.setting_json('contributors/hidden', [])
        self.remove.setEnabled(bool(identity))
        self.export_button.setEnabled(bool(identity))
        self.remove.setText('Restore Profile' if hidden else 'Remove from This List...')

    def show_advanced(self, visible):
        self.extra.setVisible(visible)
        self.advanced.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)

    def toggle_removed(self):
        identity = self.selected_id()
        if not identity:
            return
        hidden = set(preferences.setting_json('contributors/hidden', []))
        if identity in hidden:
            hidden.remove(identity)
        else:
            if QMessageBox.question(self, 'Remove Profile from This List?',
                    'Remove this profile from the chooser on this computer?\n\n'
                    'Annotations, assignments, downloaded work, and profiles on other computers '
                    'are unchanged. You can restore it using Show removed profiles.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
                return
            hidden.add(identity)
        preferences.save_setting('contributors/hidden', sorted(hidden))
        self.refresh(selected=identity)

    def create_profile(self):
        editor = QDialog(self)
        editor.setWindowTitle('New Contributor Profile')
        form = QFormLayout(editor)
        name = QLineEdit()
        name.setMaxLength(80)
        name.setMinimumWidth(280)
        form.addRow('Contributor name:', name)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        create = actions.button(QDialogButtonBox.StandardButton.Ok)
        create.setText('Create Profile')
        create.setEnabled(False)
        name.textChanged.connect(lambda text: create.setEnabled(bool(text.strip())))
        actions.accepted.connect(editor.accept)
        actions.rejected.connect(editor.reject)
        form.addRow(actions)
        if editor.exec() != QDialog.DialogCode.Accepted:
            return
        display_name = name.text().strip()
        existing = [p for p in self.profiles.values() if p['name'].casefold() == display_name.casefold()]
        if existing and QMessageBox.question(self, 'Create a Separate Identity?',
                'A profile with this name already exists. Use the existing profile for the same person.\n\n'
                'Create a different person with the same name?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            self.removed.setChecked(existing[0]['id'] in preferences.setting_json('contributors/hidden', []))
            self.refresh(selected=existing[0]['id'])
            return
        profile = {'id': str(uuid4()), 'name': display_name}
        preferences.remember_contributor_profile(profile, newly_created=True)
        self.refresh(selected=profile['id'])
        self.service.refresh()

    def import_profile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Import Your Contributor Profile', '', 'JSON (*.json)')
        if not filename:
            return
        try:
            profile = contributor_profile(preferences._read_json(filename))
            if QMessageBox.question(self, 'Import This Contributor Identity?',
                    'Import this profile with its existing contributor ID?\n\n'
                    'Import only your own identity. Existing annotations are not reattributed.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel) != QMessageBox.StandardButton.Yes:
                return
            preferences.remember_contributor_profile(profile)
            self.removed.setChecked(profile['id'] in preferences.setting_json('contributors/hidden', []))
            self.refresh(selected=profile['id'])
        except Exception as exc:
            QMessageBox.warning(self, 'Profile Not Imported', str(exc))

    def export_profile(self):
        if self.selected_id():
            preferences.export_contributor_profile(self.profiles[self.selected_id()], self)


class ContributorChooser(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Choose Contributor Profile')
        self.resize(600, 440)
        self.setMinimumSize(460, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(16)
        heading = QLabel("Who's annotating?")
        font = heading.font()
        font.setPointSize(18)
        font.setBold(True)
        heading.setFont(font)
        layout.addWidget(heading)
        self.empty = QLabel()
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)
        self.tiles = tiles = QListWidget()
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
        tiles.setGridSize(QSize(152, 144))
        tiles.setWordWrap(True)
        tiles.setTextElideMode(Qt.TextElideMode.ElideRight)
        tiles.setFrameShape(QFrame.Shape.NoFrame)
        tiles.setStyleSheet('''
            QListWidget#contributorProfiles { background: transparent; }
            QListWidget#contributorProfiles::item {
                border: 1px solid palette(mid); border-radius: 6px; padding: 8px; margin: 4px;
            }
            QListWidget#contributorProfiles::item:hover { background: palette(alternate-base); }
            QListWidget#contributorProfiles::item:selected {
                background: palette(highlight); color: palette(highlighted-text);
            }
        ''')
        tiles.itemClicked.connect(self.accept)
        tiles.itemActivated.connect(self.accept)
        layout.addWidget(tiles, 1)
        self.manage = QToolButton()
        self.manage.setText('Manage Profiles...')
        self.manage.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.manage.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.manage.clicked.connect(self.manage_profiles)
        layout.addWidget(self.manage, 0, Qt.AlignmentFlag.AlignLeft)
        self.service = profile_sync_service()
        self.profile_check = None
        self.closed = False
        self.resume_sync_timer = self.service.timer.isActive()
        self.service.timer.stop()
        self.service.changed.connect(self.refresh)
        self.finished.connect(self.detach)
        last = preferences.QSettings().value('contributors/last_id', '', type=str)
        if not last:
            name = preferences.QSettings().value('contributors/last_name', '', type=str)
            matches = [p['id'] for p in preferences.load_contributor_profiles().values() if p['name'] == name]
            last = matches[0] if len(matches) == 1 else None
        self.refresh(selected=last)
        QTimer.singleShot(0, self.check_shared_profiles)
        tiles.setFocus()

    def check_shared_profiles(self):
        if self.closed:
            return
        from gui.lab_startup import LabAccessCheck, working_locally
        configuration = preferences.setting_json('contributors/shared', {})
        if working_locally() or not configuration.get('enabled'):
            self.service.refresh()
            return
        # Avoid starting a filesystem worker against an unresponsive mount while
        # the user is still choosing a cached identity. Authentication comes later.
        self.service.status = 'Checking shared profiles...'
        self.refresh()
        check = self.profile_check = LabAccessCheck(configuration)
        def completed(result):
            if check is not self.profile_check:
                return
            self.profile_check = None
            if result.get('connected'):
                self.service.refresh()
            else:
                self.service.status = 'Shared profiles unavailable; using saved profiles.'
                self.refresh()
                if not preferences.load_contributor_profiles():
                    # With no cached identity, authentication must precede selection.
                    from gui.lab_startup import check_session_connection
                    check_session_connection(parent=self)
        check.completed.connect(completed)
        check.start()

    def detach(self):
        self.closed = True
        self.service.changed.disconnect(self.refresh)
        if self.profile_check:
            self.profile_check.cancel()
            self.profile_check = None
        if self.resume_sync_timer:
            self.service.timer.start()

    def selected_id(self):
        item = self.tiles.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def refresh(self, *, selected=None):
        identity = selected or self.selected_id()
        profiles = preferences.load_contributor_profiles()
        hidden = preferences.setting_json('contributors/hidden', [])
        names = [p['name'].casefold() for p in profiles.values()]
        self.tiles.clear()
        for key, profile in sorted(profiles.items(), key=lambda pair: (pair[1]['name'].casefold(), pair[0])):
            if key in hidden:
                continue
            name = profile['name']
            label = name + ('\n' + key[:8] if names.count(name.casefold()) > 1 else '')
            item = QListWidgetItem(preferences._contributor_avatar(name, key), label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(label)
            item.setSizeHint(QSize(144, 136))
            self.tiles.addItem(item)
            if key == identity:
                self.tiles.setCurrentItem(item)
        message = 'No profiles in this list. Add or restore one in Manage Profiles.'
        if 'Checking' in self.service.status:
            message = 'Checking shared profiles...'
        elif 'unavailable' in self.service.status:
            message = 'Shared profiles are unavailable. No saved profiles are visible on this computer.'
        self.empty.setText(message)
        self.empty.setVisible(self.tiles.count() == 0)

    def manage_profiles(self):
        dialog = ManageProfilesDialog(self)
        dialog.refresh(selected=self.selected_id())
        dialog.exec()
        self.refresh(selected=dialog.selected_id())

    def accept(self, *_):
        if self.selected_id():
            super().accept()


def choose_contributor(parent=None):
    dialog = ContributorChooser(parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    profile = preferences.load_contributor_profiles()[dialog.selected_id()]
    settings = preferences.QSettings()
    device_id = settings.value('contributors/device_id', '', type=str) or str(uuid4())
    settings.setValue('contributors/device_id', device_id)
    session = new_session(profile['name'], profile['id'], device={'id': device_id, 'name': platform.node()})
    preferences.remember_contributor_profile(profile)
    settings.setValue('contributors/last_name', profile['name'])
    settings.setValue('contributors/last_id', profile['id'])
    return session
