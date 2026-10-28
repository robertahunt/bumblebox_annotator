"""Choose a reusable identity without starting a session as that person."""

from uuid import uuid4

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QLineEdit,
                            QListWidget, QListWidgetItem, QMessageBox, QTabWidget,
                            QVBoxLayout, QWidget)

from gui.project_sync_dialog import load_contributor_profiles


class AddContributorDialog(QDialog):
    def __init__(self, members, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Project Contributor')
        self.resize(560, 430)
        self.members = members
        self.profile = None
        self.created = False
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        existing = QWidget()
        rows = QVBoxLayout(existing)
        self.search = QLineEdit()
        self.search.setPlaceholderText('Find a contributor')
        rows.addWidget(self.search)
        self.people = QListWidget()
        rows.addWidget(self.people)
        self.empty = QLabel('No other profiles available')
        rows.addWidget(self.empty)
        self.tabs.addTab(existing, 'Existing Profile')
        new = QWidget()
        form = QVBoxLayout(new)
        form.addWidget(QLabel('Contributor name'))
        self.name = QLineEdit()
        self.name.setMaxLength(80)
        form.addWidget(self.name)
        form.addStretch()
        self.tabs.addTab(new, 'New Profile')
        note = QLabel('Adds this person to the project without switching your active profile. '
                      'Their name and ID will be shared with the project. A connected lab also '
                      'makes their profile available on its other connected computers.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Add Contributor')
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.search.textChanged.connect(self.filter)
        self.people.currentItemChanged.connect(self.validate)
        self.name.textChanged.connect(self.validate)
        self.tabs.currentChanged.connect(self.validate)
        from gui.shared_profiles import profile_sync_service
        self.service = profile_sync_service()
        self.service.changed.connect(self.refresh)
        self.finished.connect(self.detach)
        self.refresh()
        self.service.refresh()

    def detach(self):
        self.service.changed.disconnect(self.refresh)

    def refresh(self):
        selected = self.people.currentItem()
        identity = selected.data(Qt.ItemDataRole.UserRole)['id'] if selected else None
        self.people.clear()
        for profile in sorted(load_contributor_profiles().values(), key=lambda p: (p['name'].casefold(), p['id'])):
            if profile['id'] in self.members:
                continue
            item = QListWidgetItem(f"{profile['name']}  ({profile['id'][:8]})")
            item.setData(Qt.ItemDataRole.UserRole, profile)
            self.people.addItem(item)
            if profile['id'] == identity:
                self.people.setCurrentItem(item)
        self.empty.setVisible(not self.people.count())
        self.filter()

    def filter(self):
        for index in range(self.people.count()):
            item = self.people.item(index)
            item.setHidden(self.search.text().casefold() not in item.text().casefold())
        self.validate()

    def validate(self):
        item = self.people.currentItem()
        valid = (bool(item and not item.isHidden()) if self.tabs.currentIndex() == 0
                 else bool(self.name.text().strip()))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)

    def accept(self):
        if self.tabs.currentIndex() == 0:
            item = self.people.currentItem()
            if not item or item.isHidden():
                return
            self.profile = item.data(Qt.ItemDataRole.UserRole)
        else:
            name = self.name.text().strip()
            if not name:
                return
            candidates = list(load_contributor_profiles().values()) + list(self.members.values())
            if any(person['name'].casefold() == name.casefold() for person in candidates):
                if QMessageBox.question(self, 'Create a Separate Identity?',
                        'A profile with this name already exists. Use the existing profile for the same '
                        'person. Create a different person with this name?',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                    return
            self.profile = {'id': str(uuid4()), 'name': name}
            self.created = True
        super().accept()
