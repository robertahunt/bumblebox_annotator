"""Explicit version-scoped checks, separate from publication and repair."""

from datetime import datetime, timezone
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                            QFormLayout, QHBoxLayout, QLabel, QMessageBox,
                            QPlainTextEdit, QPushButton, QSpinBox, QStyle, QVBoxLayout)

from core.backup_integrity import check_backup_integrity, integrity_versions, read_integrity_report
from core.project_sync import _write_json
from gui.project_sync_dialog import SyncProgressDialog, _size_transfer_dialog
from gui.shared_version import display_time


STATUS = {'passed': 'Passed', 'problems': 'Problems found',
          'cancelled': 'Cancelled', 'unable': 'Unable to complete'}


def check_status(entry, manifest_hash, days):
    latest, passed = entry.get('last_check'), entry.get('last_pass')
    lines = []
    if latest:
        lines.append('Last attempt: ' + STATUS.get(latest['status'], 'Unknown') +
                     ' | ' + display_time(latest['finished_at']))
    if not passed or passed.get('manifest_hash') != manifest_hash:
        lines.append('No successful full integrity check recorded for this exact version. Check recommended.')
    else:
        lines.append('Last full check passed: ' + display_time(passed['finished_at']))
        try:
            age = datetime.now(timezone.utc) - datetime.fromisoformat(passed['finished_at'])
            due = age.days >= days
        except (ValueError, TypeError):
            due = True
        if due:
            lines.append(f'Another check is recommended (reminder interval: {days} days).')
    if latest and latest['status'] != 'passed':
        lines.append('An earlier pass does not certify the outcome of the latest attempt.')
    return '\n'.join(lines)


def report_text(report):
    lines = [STATUS[report['status']],
             'Version: ' + ' / '.join(report['reference']),
             'Manifest SHA-256: ' + report['manifest_hash'],
             'Started: ' + display_time(report['started_at']),
             'Finished: ' + display_time(report['finished_at']),
             f"Checked {report['checked_contents']:,}/{report['unique_contents']:,} unique stored contents "
             f"({report['files']:,} file references).",
             f"Read {report['bytes_read'] / 1024**2:.1f}/{report['total_bytes'] / 1024**2:.1f} MiB."]
    for key in ('error', 'report_warning', 'report_path'):
        if report.get(key):
            lines.append(str(report[key]))
    for issue in report['issues']:
        lines.extend(['', issue['reason'] + ':', *issue['paths'],
                      'Expected: ' + str(issue['expected']), 'Actual: ' + str(issue['actual'])])
    lines.append('\nNo files were repaired, deleted, or merged. This result applies only to the selected version.')
    return '\n'.join(lines)


class BackupIntegrityDialog(QDialog):
    def __init__(self, project, configuration, parent=None):
        super().__init__(parent)
        self.project, self.configuration = project, dict(configuration)
        self.state, self.report = {}, None
        self.setWindowTitle('Check Backup Integrity')
        layout = QVBoxLayout(self)
        title = QLabel(Path(project).name)
        title.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(title)
        self.versions = QComboBox()
        self.versions.setMinimumContentsLength(24)
        self.versions.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.versions.setPlaceholderText('Select a backup version')
        row = QHBoxLayout()
        row.addWidget(self.versions, 1)
        refresh = QPushButton()
        refresh.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        refresh.setToolTip('Refresh backup versions and integrity reports')
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        form = QFormLayout()
        self.days = QSpinBox()
        self.days.setMaximumWidth(150)
        self.days.setRange(1, 365)
        self.days.setSuffix(' days')
        self.days.setValue(QSettings().value('project_sync/integrity_reminder_days', 30, type=int))
        self.days.valueChanged.connect(self.interval_changed)
        form.addRow('Integrity-check reminder:', self.days)
        layout.addLayout(form)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 1)
        self.check = QPushButton('Run Full Check')
        self.check.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        self.check.setEnabled(False)
        self.check.clicked.connect(self.run_check)
        self.export = QPushButton('Save Report...')
        self.export.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.export.setEnabled(False)
        self.export.clicked.connect(self.save_report)
        self.previous = QPushButton('View Last Report')
        self.previous.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        self.previous.setEnabled(False)
        self.previous.clicked.connect(self.view_report)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.addButton(self.check, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.export, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.previous, QDialogButtonBox.ButtonRole.ActionRole)
        layout.addWidget(buttons)
        self.versions.currentIndexChanged.connect(self.selection_changed)
        _size_transfer_dialog(self, 1000, 650, minimum_height=440)
        QTimer.singleShot(0, self.refresh)

    def refresh(self):
        try:
            result = SyncProgressDialog('Read Backup Versions and Integrity Reports',
                lambda progress, cancelled: integrity_versions(self.project, self.configuration, cancelled), self).run()
            selected = self.versions.currentData()
            reference = selected['reference'] if selected else None
            self.versions.blockSignals(True)
            self.versions.clear()
            self.state = result['state']
            for snapshot in result['versions']:
                manifest = snapshot['manifest']
                text = (manifest['contributor']['name'] + ' | ' + display_time(manifest.get('created_at')) +
                        ' | ' + snapshot['reference'][1])
                self.versions.addItem(text, snapshot)
                self.versions.setItemData(self.versions.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
            index = next((i for i, s in enumerate(result['versions']) if s['reference'] == reference),
                         0 if len(result['versions']) == 1 else -1)
            self.versions.setCurrentIndex(index)
            self.versions.blockSignals(False)
            self.selection_changed()
        except Exception as exc:
            self.check.setEnabled(False)
            self.status.setText('Unable to read backup versions: ' + str(exc))

    def interval_changed(self, days):
        QSettings().setValue('project_sync/integrity_reminder_days', days)
        self.update_status()

    def update_status(self):
        snapshot = self.versions.currentData()
        if snapshot:
            entry = self.state.get('versions', {}).get('/'.join(snapshot['reference']), {})
            self.status.setText(check_status(entry, snapshot['manifest_hash'], self.days.value()))
        else:
            self.status.setText('Select the exact backup version to check. No automatic check is running.')

    def selection_changed(self, *_):
        self.report = None
        self.export.setEnabled(False)
        self.check.setEnabled(self.versions.currentData() is not None)
        snapshot = self.versions.currentData()
        entry = self.state.get('versions', {}).get('/'.join(snapshot['reference']), {}) if snapshot else {}
        self.previous.setEnabled(bool(entry.get('last_check')))
        self.details.setPlainText('Reads all stored contents referenced by the selected version, including extracted frames.\n\n'
            'Other computers can continue annotating locally, but publishing this project is locked during the check. '
            'Cancellation is available; interrupted checks are not recorded as passed.\n\n'
            'Checks do not upload annotations or repair files. Newly published content is verified during upload; '
            'unchanged content needs this separate full check. Reminders do not schedule automatic checks.')
        self.update_status()

    def run_check(self):
        snapshot = self.versions.currentData()
        if not snapshot:
            return
        answer = QMessageBox.question(self, 'Run Full Integrity Check?',
            'This reads all contents of the selected backup and may take a long time. Publishing this project '
            'from any computer will be blocked until the check finishes or is cancelled. Continue?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.report = None
        self.export.setEnabled(False)
        try:
            report = SyncProgressDialog('Checking Backup Integrity',
                lambda progress, cancelled: check_backup_integrity(self.project, self.configuration,
                    snapshot['reference'], snapshot['manifest_hash'], progress, cancelled), self).run()
            self.refresh()
            self.report = report
            self.details.setPlainText(report_text(report))
            self.export.setEnabled(True)
        except Exception as exc:
            self.details.setPlainText('Unable to complete integrity check: ' + str(exc))

    def view_report(self):
        snapshot = self.versions.currentData()
        if not snapshot:
            return
        entry = self.state.get('versions', {}).get('/'.join(snapshot['reference']), {})
        if not entry.get('last_check'):
            return
        self.report = None
        self.export.setEnabled(False)
        try:
            self.report = SyncProgressDialog('Read Integrity Report',
                lambda progress, cancelled: read_integrity_report(self.project, self.configuration,
                    snapshot['reference'], entry['last_check']['report_id']), self).run()
            self.details.setPlainText(report_text(self.report))
            self.export.setEnabled(True)
        except Exception as exc:
            self.details.setPlainText('Unable to read integrity report: ' + str(exc))

    def save_report(self):
        if not self.report:
            return
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Integrity Report',
            'backup-integrity-' + self.report['report_id'] + '.json', 'JSON (*.json)')
        if filename:
            try:
                _write_json(Path(filename), self.report)
            except OSError as exc:
                QMessageBox.warning(self, 'Report Not Saved', str(exc))


def open_integrity_dialog(project, configuration, parent=None):
    if not project or not configuration.get('root_id'):
        QMessageBox.information(parent, 'Backup Not Configured',
                                'Open a project and configure its shared destination first.')
        return
    BackupIntegrityDialog(project, configuration, parent).exec()
