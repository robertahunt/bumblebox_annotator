"""Integrity UI on temporary backups, with isolated settings and no real drive access."""

from contextlib import ExitStack
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.backup_integrity import check_backup_integrity, integrity_versions
from gui.backup_integrity_dialog import BackupIntegrityDialog, check_status
from scripts import test_compact_sync as fixtures


class BackupIntegrityGuiTests(unittest.TestCase):
    sync = fixtures.CompactSyncTests.sync

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        fixtures.CompactSyncTests.setUp(self)
        self.addCleanup(self.temp.cleanup)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        settings = QSettings(str(self.base / 'settings.ini'), QSettings.Format.IniFormat)
        self.stack.enter_context(patch('gui.backup_integrity_dialog.QSettings', lambda: settings))
        self.stack.enter_context(patch('gui.backup_integrity_dialog.QTimer.singleShot'))
        self.stack.enter_context(patch('gui.backup_integrity_dialog.SyncProgressDialog.run',
            lambda dialog: dialog.worker.operation(lambda text: None, lambda: False)))
        self.sync()
        self.dialog = BackupIntegrityDialog(self.source, self.configuration)
        self.addCleanup(self.dialog.close)
        self.dialog.refresh()

    def test_first_check_requires_confirmation_and_is_not_automatic(self):
        self.assertTrue(self.dialog.check.isEnabled())
        self.assertIn('No successful full', self.dialog.status.text())
        self.assertFalse(self.dialog.export.isEnabled())
        self.assertFalse(list(self.remote.rglob('verification')))
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            self.dialog.run_check()
        self.assertFalse(list(self.remote.rglob('verification')))

    def test_full_check_report_is_viewable_and_selection_clears_export(self):
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            self.dialog.run_check()
        self.assertIn('Passed', self.dialog.details.toPlainText())
        self.assertIn('Last full check passed', self.dialog.status.text())
        self.assertTrue(self.dialog.previous.isEnabled())
        self.assertTrue(self.dialog.export.isEnabled())
        self.dialog.selection_changed()
        self.assertFalse(self.dialog.export.isEnabled())
        self.dialog.view_report()
        self.assertIn('Passed', self.dialog.details.toPlainText())
        self.assertTrue(self.dialog.export.isEnabled())

    def test_cancelled_attempt_is_distinct_from_last_pass_and_overdue_reminder(self):
        snapshot = self.dialog.versions.currentData()
        args = (self.source, self.configuration, snapshot['reference'], snapshot['manifest_hash'])
        check_backup_integrity(*args)
        check_backup_integrity(*args, cancelled=lambda: True)
        self.dialog.refresh()
        self.assertIn('Last attempt: Cancelled', self.dialog.status.text())
        self.assertIn('Last full check passed', self.dialog.status.text())
        entry = integrity_versions(self.source, self.configuration)['state']['versions']['/'.join(snapshot['reference'])]
        entry['last_pass']['finished_at'] = '2020-01-01T00:00:00+00:00'
        self.assertIn('Another check is recommended', check_status(entry, snapshot['manifest_hash'], 30))
        self.assertIn('No successful full', check_status(entry, 'changed manifest', 30))

    def test_refresh_failure_does_not_claim_current_or_enable_check(self):
        with patch('gui.backup_integrity_dialog.integrity_versions', side_effect=OSError('offline')):
            self.dialog.refresh()
        self.assertFalse(self.dialog.check.isEnabled())
        self.assertIn('Unable to read', self.dialog.status.text())


if __name__ == '__main__':
    unittest.main()
