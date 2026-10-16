"""Explicit local rollback and reviewed retention; never prune published history."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout,
    QHeaderView, QLabel, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

from core.project_delta import delete_rollback_bundle, rollback_bundles, rollback_update
from gui.project_sync_dialog import SyncProgressDialog


class ProjectRollbackDialog(QDialog):
    def __init__(self, project, configuration, parent=None):
        super().__init__(parent)
        self.project, self.configuration = project, configuration
        self.update_result = None
        self.setWindowTitle('Local Rollback History')
        self.resize(850, 460)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Delete', 'Created (UTC)', 'Stored files (MiB)', 'Status'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        row = QHBoxLayout()
        row.addWidget(QLabel('Keep newest:'))
        self.keep = QSpinBox()
        self.keep.setRange(1, 1000)
        self.keep.setValue(3)
        row.addWidget(self.keep)
        select = QPushButton('Select Older Bundles')
        select.clicked.connect(self.select_older)
        row.addWidget(select)
        delete = QPushButton('Delete Checked...')
        delete.clicked.connect(self.delete_checked)
        row.addWidget(delete)
        row.addStretch()
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.undo = buttons.addButton('Restore Before Selected Update...', QDialogButtonBox.ButtonRole.ActionRole)
        self.undo.clicked.connect(self.restore_selected)
        self.table.itemSelectionChanged.connect(self.selection_changed)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self):
        self.items = rollback_bundles(self.project, self.configuration)
        self.table.setRowCount(len(self.items))
        for row, item in enumerate(self.items):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, check)
            for col, text in enumerate((item['manifest']['created_at'], f"{item['bytes'] / 1024**2:.2f}",
                                       item['manifest']['status']), 1):
                cell = QTableWidgetItem(text)
                cell.setToolTip(str(item['path']))
                self.table.setItem(row, col, cell)
        total = sum(item['bytes'] for item in self.items)
        self.summary.setText(f'{len(self.items)} rollback bundles | {total / 1024**2:.1f} MiB. '
                             'Published versions are not removed here.')
        self.selection_changed()

    def selection_changed(self):
        row = self.table.currentRow()
        self.undo.setEnabled(0 <= row < len(self.items) and self.items[row]['manifest']['status'] == 'complete')

    def select_older(self):
        for row, item in enumerate(self.items):
            eligible = row >= self.keep.value() and item['manifest']['status'] in ('complete', 'rolled-back')
            self.table.item(row, 0).setCheckState(Qt.CheckState.Checked if eligible else Qt.CheckState.Unchecked)

    def delete_checked(self):
        selected = [item for row, item in enumerate(self.items)
                    if self.table.item(row, 0).checkState() == Qt.CheckState.Checked]
        if not selected:
            return
        total = sum(item['bytes'] for item in selected)
        if QMessageBox.question(self, 'Delete Local Rollback Bundles?',
                f'Permanently remove {len(selected)} bundles ({total / 1024**2:.1f} MiB)?\n'
                'Their undo history will be lost. Current annotations and published versions are not deleted.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            def remove(progress, cancelled):
                for item in selected:
                    if cancelled():
                        break
                    progress('Removing ' + str(item['path']))
                    delete_rollback_bundle(self.project, item['path'], self.configuration)
            SyncProgressDialog('Remove Local Rollbacks', remove, self).run()
        except Exception as exc:
            QMessageBox.warning(self, 'Cleanup Not Completed', str(exc))
        self.refresh()

    def restore_selected(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.items):
            return
        if QMessageBox.question(self, 'Restore Previous Local Files?',
                'Undo the selected update? This is allowed only if the managed project has not changed since. '
                'A new rollback bundle will preserve the current affected files.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.update_result = SyncProgressDialog('Restore Previous Local Files',
                lambda progress, cancelled: rollback_update(self.project, self.items[row]['path'],
                    self.configuration, progress, cancelled), self).run()
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, 'Rollback Not Applied', str(exc))
