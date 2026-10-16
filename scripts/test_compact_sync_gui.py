"""Storage estimates, settings, and explicit rollback/retention UI."""

import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.project_delta import rollback_update
from gui.project_rollback_dialog import ProjectRollbackDialog
from gui.project_sync_dialog import ProjectSyncDialog, confirm_publication, setting_json
from scripts import test_compact_sync as fixtures
from scripts import test_contributor_sync_gui as gui_fixtures


class RollbackDialogTests(unittest.TestCase):
    setUp = fixtures.CompactSyncTests.setUp
    tearDown = fixtures.CompactSyncTests.tearDown
    sync = fixtures.CompactSyncTests.sync
    publish_change = fixtures.CompactSyncTests.publish_change
    local_and_plan = fixtures.CompactSyncTests.local_and_plan

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_history_opens_without_deleting_and_undo_requires_confirmation(self):
        local, plan = self.local_and_plan()
        result = plan.apply()
        dialog = ProjectRollbackDialog(local, self.configuration)
        dialog.table.selectRow(0)
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-rollback-history.png')
        self.assertTrue(dialog.undo.isEnabled())
        self.assertTrue(Path(result['backup']).exists())
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            dialog.restore_selected()
        self.assertIsNone(dialog.update_result)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            dialog.restore_selected()
        self.assertTrue(dialog.update_result['updated'])
        self.assertEqual((local / 'annotations/json/video/frame_000000.json').read_text(), '[]')

    def test_retention_selects_older_and_deletes_only_after_confirmation(self):
        local, plan = self.local_and_plan()
        first = plan.apply()
        second = rollback_update(local, first['backup'], self.configuration)
        rollback_update(local, second['backup'], self.configuration)
        dialog = ProjectRollbackDialog(local, self.configuration)
        self.assertEqual(len(dialog.items), 3)
        dialog.keep.setValue(2)
        dialog.select_older()
        self.assertEqual([dialog.table.item(i, 0).checkState() for i in range(3)],
                         [Qt.CheckState.Unchecked, Qt.CheckState.Unchecked, Qt.CheckState.Checked])
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            dialog.delete_checked()
        self.assertTrue(Path(first['backup']).exists())
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            dialog.delete_checked()
        self.assertEqual(len(dialog.items), 2)
        self.assertFalse(Path(first['backup']).exists())


class StorageSettingsTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixtures.ContributorSyncGuiTests.setUpClass.__func__)
    setUp = gui_fixtures.ContributorSyncGuiTests.setUp
    tearDown = gui_fixtures.ContributorSyncGuiTests.tearDown

    def test_compact_is_default_and_custom_rollback_location_is_saved(self):
        rollback = self.base / 'outside-dropbox'
        rollback.mkdir()
        dialog = ProjectSyncDialog(self.source)
        self.assertEqual(dialog.sync_button.text(), 'Publish Project Backup')
        self.assertEqual(dialog.storage.currentData(), 'deduplicated')
        self.assertEqual(dialog.rollback.text(), '')
        dialog.rollback.setText(str(rollback))
        dialog.folder.setText(str(self.remote))
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-storage-settings.png')
        dialog.accept()
        settings = setting_json('project_sync/destination', {})
        self.assertEqual(settings['rollback_root'], str(rollback))
        self.assertEqual(settings['storage_mode'], 'deduplicated')

    def test_publication_preview_is_explicit_and_can_be_cancelled(self):
        def cancel():
            dialog = self.app.activeModalWidget()
            dialog.grab().save('/tmp/bumblebox-publication-estimate.png')
            dialog.button(QMessageBox.StandardButton.Cancel).click()
        QTimer.singleShot(0, cancel)
        self.assertFalse(confirm_publication(None, {
            'files': 28058, 'logical_bytes': int(45.1 * 1024**3),
            'new_bytes': 2 * 1024**2, 'free_bytes': 100 * 1024**3}, {}))

    def test_publication_preview_reports_pending_archive_rename(self):
        messages = []
        def cancel():
            dialog = self.app.activeModalWidget()
            messages.append(dialog.informativeText())
            dialog.button(QMessageBox.StandardButton.Cancel).click()
        QTimer.singleShot(0, cancel)
        self.assertFalse(confirm_publication(None, {
            'files': 1, 'logical_bytes': 1, 'new_bytes': 1, 'free_bytes': 100,
            'project_name': '2026_CV4E', 'rename_from': 'test--12345678'}, {}))
        self.assertIn('test--12345678', messages[0])
        self.assertIn('2026_CV4E', messages[0])
        self.assertIn('keeping its ID and history', messages[0])


if __name__ == '__main__':
    unittest.main()
