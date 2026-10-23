"""Select a published version, inspect differences, and explicitly apply a safe update."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                            QDialogButtonBox, QHBoxLayout, QHeaderView, QLabel,
                            QMessageBox, QPushButton, QStyle, QTableWidget,
                            QTableWidgetItem, QTextEdit, QVBoxLayout)

from core.project_update import PreparedUpdate, published_versions
from gui.project_sync_dialog import SyncProgressDialog


class ProjectUpdateDialog(QDialog):
    def __init__(self, project, configuration, versions, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Review Shared Updates')
        self.resize(980, 650)
        self.project = project
        self.configuration = configuration
        self.plan = None
        self.update_result = None
        layout = QVBoxLayout(self)
        path = QLabel(str(project))
        path.setTextFormat(Qt.TextFormat.PlainText)
        path.setWordWrap(True)
        layout.addWidget(path)
        row = QHBoxLayout()
        self.versions = QComboBox()
        self.versions.setMinimumContentsLength(20)
        self.versions.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.versions.setPlaceholderText('Choose a contributor publication')
        self.versions.currentIndexChanged.connect(self.clear_preview)
        row.addWidget(self.versions, 1)
        refresh = QPushButton()
        refresh.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        refresh.setToolTip('Refresh published versions')
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        self.preview_button = QPushButton('Preview Changes')
        self.preview_button.clicked.connect(self.preview)
        row.addWidget(self.preview_button)
        layout.addLayout(row)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['File', 'Local since base', 'Published since base', 'Update action'])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        layout.addWidget(self.table, 1)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(135)
        layout.addWidget(self.details)
        self.confirm = QCheckBox('I reviewed these changes, including deletions and policy/split changes.')
        layout.addWidget(self.confirm)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText('Back Up and Update')
        self.apply_button.clicked.connect(self.apply_update)
        self.confirm.toggled.connect(self.update_apply_state)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.set_versions(versions)

    def set_versions(self, versions):
        self.versions.clear()
        for snapshot in sorted(versions, key=lambda s: s['manifest'].get('created_at', ''), reverse=True):
            manifest = snapshot['manifest']
            text = (f"{manifest['contributor']['name']} | {manifest.get('created_at', 'Unknown date')} | "
                    f"{snapshot['reference'][1]}" +
                    (' | original videos excluded' if not manifest.get('include_videos', True) else ''))
            self.versions.addItem(text, snapshot['reference'])
            self.versions.setItemData(self.versions.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
        self.versions.setCurrentIndex(0 if len(versions) == 1 else -1)
        self.clear_preview()

    def clear_preview(self, *_):
        self.plan = None
        self.confirm.setChecked(False)
        self.table.setRowCount(0)
        self.details.clear()
        self.summary.setText('Select a published version and preview its changes.' if self.versions.count()
                             else 'No completed contributor publications were found.')
        self.preview_button.setEnabled(self.versions.currentIndex() >= 0)
        self.update_apply_state()

    def update_apply_state(self, *_):
        allowed = self.plan is not None and not self.plan.blocked and not self.plan.noop
        self.confirm.setEnabled(allowed)
        self.apply_button.setEnabled(allowed and self.confirm.isChecked())

    def refresh(self):
        try:
            versions = SyncProgressDialog('Read Published Versions',
                lambda progress, cancelled: published_versions(self.project, self.configuration), self).run()
            self.set_versions(versions)
        except Exception as exc:
            self.clear_preview()
            QMessageBox.warning(self, 'Versions Unavailable', str(exc))

    def preview(self):
        reference = self.versions.currentData()
        self.clear_preview()
        if reference is None:
            return
        try:
            self.plan = SyncProgressDialog('Compare Project Versions',
                lambda progress, cancelled: PreparedUpdate(self.project, self.configuration, reference,
                                                            progress, cancelled), self).run()
            plan = self.plan
            if plan.blocked:
                self.summary.setText('Update blocked: ' + plan.blocked)
            elif plan.local_only:
                self.summary.setText('Only the local copy has edits. Nothing to download; your edits will be kept.')
            elif plan.noop:
                self.summary.setText('This local copy already matches the selected version.')
            else:
                counts = {kind: sum(value == kind for value in plan.changes.values())
                          for kind in ('Add', 'Replace', 'Delete')}
                self.summary.setText(f"{counts['Add']} additions | {counts['Replace']} replacements | "
                    f"{counts['Delete']} deletions. Only affected files are staged and backed up.\n"
                    f"Replacements: {plan.stage_bytes / 1024**2:.1f} MiB | "
                    f"Rollback: {plan.rollback_bytes / 1024**2:.1f} MiB (plus recovery overhead).")
            names = sorted(plan.local_changes.keys() | plan.remote_changes.keys())
            self.table.setRowCount(len(names))
            for row, name in enumerate(names):
                action = ('Review required' if plan.blocked else 'Keep local' if plan.local_only
                          else plan.changes.get(name, 'Unchanged'))
                values = (name, plan.local_changes.get(name, ''), plan.remote_changes.get(name, ''), action)
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    self.table.setItem(row, col, item)
            details = list(plan.policy_changes)
            if plan.split_changes:
                details.append(f'{len(plan.split_changes)} video files or frame-selection/split metadata files change; '
                               'review their entries above.')
            if not plan.include_videos:
                details.append('Original videos are excluded from this publication; local originals will be kept.')
            details.append('COCO exports will be invalidated after an update. Models and other local-only outputs are kept.')
            details.append(f'Rollback folder: {plan.rollback_root}')
            self.details.setPlainText('\n'.join(details))
        except Exception as exc:
            self.summary.setText(str(exc))
        self.update_apply_state()

    def apply_update(self):
        if self.plan is None or self.plan.blocked or self.plan.noop or not self.confirm.isChecked():
            return
        try:
            self.update_result = SyncProgressDialog('Back Up and Update Local Copy', self.plan.apply, self).run()
            self.accept()
        except Exception as exc:
            self.clear_preview()
            self.summary.setText('Update not completed: ' + str(exc))
            QMessageBox.warning(self, 'Update Not Completed', str(exc))
