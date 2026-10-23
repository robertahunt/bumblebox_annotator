"""Storage estimates, settings, and explicit rollback/retention UI."""

import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QTextOption
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMessageBox, QToolButton

from core.project_delta import rollback_update
from gui.project_rollback_dialog import ProjectRollbackDialog
from gui.project_sync_dialog import (ProjectSyncDialog, SyncProgressDialog, TransferConfirmationDialog,
                                     confirm_publication, confirm_separate_backup, save_setting, setting_json)
from core.contributors import new_session
from core.project_sync import _digest, sync_project
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


class TransferDialogLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_progress_window_wraps_complete_paths_and_scrolls_overflow(self):
        path = '/mnt/researchdrive/Crall_Lab/' + 'long_project_name_' * 35 + '/frame_000001.png'
        status = 'Verifying transferred files\n' + '\n'.join([path] * 8)
        dialog = SyncProgressDialog('Publish Project Backup', lambda progress, cancelled: None)
        dialog.message.setPlainText(status)
        dialog.show()
        self.app.processEvents()
        available = dialog.screen().availableGeometry()
        self.assertEqual(dialog.width(), min(860, available.width() - 64))
        self.assertEqual(dialog.height(), min(340, available.height() - 64))
        self.assertTrue(dialog.isSizeGripEnabled())
        self.assertTrue(dialog.message.isReadOnly())
        self.assertEqual(dialog.message.wordWrapMode(), QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.assertGreaterEqual(dialog.message.font().pointSizeF(), 11)
        self.assertEqual(dialog.message.toPlainText(), status)
        self.assertEqual(dialog.message.horizontalScrollBar().maximum(), 0)
        self.assertGreater(dialog.message.verticalScrollBar().maximum(), 0)
        dialog.grab().save('/tmp/bumblebox-transfer-progress-large.png')
        dialog.resize(640, 260)
        self.app.processEvents()
        self.assertEqual(dialog.message.horizontalScrollBar().maximum(), 0)
        self.assertFalse(dialog.message.geometry().intersects(dialog.cancel.geometry()))
        dialog.grab().save('/tmp/bumblebox-transfer-progress-small.png')
        dialog.reject()

    def test_worker_progress_is_plain_text_and_completion_still_returns_result(self):
        status = 'Copying /project/<trial>&data/frame.png'
        def transfer(progress, cancelled):
            progress(status)
            return {'copied': 1}
        dialog = SyncProgressDialog('Copy', transfer)
        self.assertEqual(dialog.run(), {'copied': 1})
        self.assertEqual(dialog.message.toPlainText(), status)
        self.assertFalse(dialog.worker.isRunning())

    def test_cancellation_waits_for_worker_and_remains_visible(self):
        pause = Event()
        observed = {}
        def transfer(progress, cancelled):
            for _ in range(300):
                if cancelled():
                    return 'cancelled'
                pause.wait(.01)
            raise RuntimeError('Cancellation was not delivered')
        dialog = SyncProgressDialog('Copy', transfer)
        def cancel():
            dialog.cancel.click()
            observed['message'] = dialog.message.toPlainText()
            observed['button_enabled'] = dialog.cancel.isEnabled()
        QTimer.singleShot(50, cancel)
        self.assertEqual(dialog.run(), 'cancelled')
        self.assertIn('Cancelling', observed['message'])
        self.assertFalse(observed['button_enabled'])
        self.assertFalse(dialog.worker.isRunning())

    def test_confirmation_is_resizable_and_cancel_is_default(self):
        dialog = TransferConfirmationDialog('Transfer', 'A long path: /' + 'x' * 1500, 'Publish')
        dialog.show()
        self.app.processEvents()
        self.assertTrue(dialog.isSizeGripEnabled())
        self.assertGreaterEqual(dialog.width(), 640)
        self.assertTrue(dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).isDefault())
        self.assertEqual(dialog.message.horizontalScrollBar().maximum(), 0)
        QTest.keyClick(dialog, Qt.Key.Key_Return)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)

    def test_separate_backup_confirmation_still_requires_explicit_acceptance(self):
        for button, expected in ((QDialogButtonBox.StandardButton.Cancel, False),
                                 (QDialogButtonBox.StandardButton.Ok, True)):
            QTimer.singleShot(0, lambda button=button: self.app.activeModalWidget().buttons.button(button).click())
            self.assertEqual(confirm_separate_backup(None, ValueError('Published work changed')), expected)


class StorageSettingsTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixtures.ContributorSyncGuiTests.setUpClass.__func__)
    setUp = gui_fixtures.ContributorSyncGuiTests.setUp
    tearDown = gui_fixtures.ContributorSyncGuiTests.tearDown
    make_window = gui_fixtures.ContributorSyncGuiTests.make_window

    def linked_window(self):
        window = self.make_window()
        save_setting('project_sync/destination', gui_fixtures.probe_destination(self.remote))
        save_setting('project_sync/projects', {str(self.source): {'enabled': True}})
        return window

    def test_publish_is_disabled_until_separate_estimate_without_uploading(self):
        window = self.linked_window()
        self.assertFalse(window.sync_project_action.isEnabled())
        with patch('gui.main_window.sync_project') as publish:
            window.sync_project_action.trigger()
            publish.assert_not_called()
            estimate = window.estimate_project_backup()
            publish.assert_not_called()
        self.assertIsNotNone(estimate)
        self.assertTrue(window.sync_project_action.isEnabled())
        self.assertIn('Estimated additional storage', window.workflow.backup_summary.text())
        self.assertFalse(list(self.remote.rglob('manifest.json')))
        window.workflow.activate('Manage')
        window.show()
        self.app.processEvents()
        publish_button = next(button for button in window.workflow.manage_page.findChildren(QToolButton)
                              if button.defaultAction() is window.sync_project_action)
        self.assertTrue(publish_button.isEnabled())
        window.workflow.manage_page.grab().save('/tmp/bumblebox-manage-transfer-estimate.png')
        window._mark_current_annotations_dirty()
        self.assertFalse(window.sync_project_action.isEnabled())
        self.assertFalse(publish_button.isEnabled())
        self.assertIsNone(window._backup_estimate)

    def test_settings_require_estimate_and_invalidate_changed_options(self):
        window = self.linked_window()
        dialog = ProjectSyncDialog(self.source, window)
        self.assertFalse(dialog.sync_button.isEnabled())
        dialog.sync_now()
        self.assertFalse(dialog.sync_requested)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        dialog.show()
        for change in (lambda: dialog.videos.toggle(),
                       lambda: dialog.storage.setCurrentIndex(0),
                       lambda: dialog.enabled.setChecked(False),
                       lambda: dialog.folder.setText(str(self.base / 'different'))):
            dialog.estimate_transfer()
            self.assertTrue(dialog.sync_button.isEnabled())
            self.assertIsNotNone(dialog.transfer_estimate)
            self.app.processEvents()
            dialog.grab().save('/tmp/bumblebox-sync-transfer-estimate.png')
            change()
            self.assertFalse(dialog.sync_button.isEnabled())
            self.assertIsNone(dialog.transfer_estimate)
            self.assertFalse(window.sync_project_action.isEnabled())
        dialog.reject()

    def test_settings_can_estimate_then_publish_and_reset_gate(self):
        window = self.linked_window()
        dialog = ProjectSyncDialog(self.source, window)
        dialog.folder.setText(str(self.remote) + '/')
        dialog.estimate_transfer()
        dialog.sync_now()
        self.assertTrue(dialog.sync_requested)
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        with patch('gui.main_window.ProjectSyncDialog', return_value=dialog), \
                patch.object(dialog, 'exec', return_value=QDialog.DialogCode.Accepted), \
                patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'warning') as warning:
            self.assertTrue(window.configure_project_sync())
            warning.assert_not_called()
        self.assertTrue(list(self.remote.rglob('manifest.json')))
        self.assertFalse(window.sync_project_action.isEnabled())
        self.assertIsNone(window._backup_estimate)

    def test_publish_passes_saved_estimate_to_backend(self):
        window = self.linked_window()
        configuration = setting_json('project_sync/destination', {})
        save_setting('project_sync/destination', dict(configuration, storage_mode='deduplicated'))
        for repeat in (False, True):
            estimate = window.estimate_project_backup()
            if repeat:
                self.assertTrue(estimate['source_checksums'])
            with patch('gui.main_window.sync_project', wraps=sync_project) as publish, \
                    patch('core.project_sync._digest', wraps=_digest) as digest, \
                    patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'warning') as warning:
                self.assertTrue(window.sync_current_project())
            warning.assert_not_called()
            self.assertIs(publish.call_args.kwargs['estimate'], estimate)
            if repeat:
                self.assertFalse(any(self.source in call.args[0].parents for call in digest.call_args_list))

    def test_external_file_changes_block_both_publication_entry_points(self):
        window = self.linked_window()
        for from_settings in (False, True):
            with self.subTest(from_settings=from_settings):
                dialog = ProjectSyncDialog(self.source, window) if from_settings else None
                if dialog:
                    dialog.estimate_transfer()
                else:
                    self.assertIsNotNone(window.estimate_project_backup())
                path = self.source / 'frames/video/frame_000000.jpg'
                path.write_bytes(path.read_bytes() + b'changed')
                with patch('gui.main_window.sync_project') as publish, patch.object(QMessageBox, 'information'):
                    if dialog:
                        dialog.sync_now()
                        self.assertFalse(dialog.sync_requested)
                        self.assertFalse(dialog.sync_button.isEnabled())
                    else:
                        self.assertFalse(window.sync_current_project())
                    publish.assert_not_called()
                self.assertFalse(window.sync_project_action.isEnabled())

    def test_failed_cancelled_estimate_or_failed_save_never_enables_publish(self):
        window = self.linked_window()
        for error in (OSError('Drive offline'), InterruptedError('Cancelled')):
            with self.subTest(error=error), \
                    patch('gui.main_window.publication_estimate', side_effect=error), \
                    patch.object(QMessageBox, 'warning'), patch('gui.main_window.sync_project') as publish:
                self.assertIsNone(window.estimate_project_backup())
                self.assertFalse(window.sync_project_action.isEnabled())
                self.assertFalse(window._sync_in_progress)
                publish.assert_not_called()
        with patch.object(window, '_save_annotation_sources', side_effect=OSError('Disk full')), \
                patch.object(QMessageBox, 'warning'), patch('gui.main_window.publication_estimate') as estimate:
            self.assertIsNone(window.estimate_project_backup())
            estimate.assert_not_called()
        self.assertFalse(window.sync_project_action.isEnabled())

    def test_profile_change_invalidates_estimate_and_direct_call_cannot_bypass_gate(self):
        window = self.linked_window()
        self.assertIsNotNone(window.estimate_project_backup())
        window.set_contributor(new_session('Alex'))
        self.assertFalse(window.sync_project_action.isEnabled())
        with patch.object(window, 'configure_project_sync', return_value=False) as configure, \
                patch('gui.main_window.sync_project') as publish:
            self.assertFalse(window.sync_current_project())
            configure.assert_called_once()
            publish.assert_not_called()

    def test_failed_connection_clears_previous_settings_estimate(self):
        window = self.linked_window()
        dialog = ProjectSyncDialog(self.source, window)
        dialog.estimate_transfer()
        self.assertTrue(dialog.sync_button.isEnabled())
        self.remote.rename(self.base / 'offline')
        self.assertFalse(dialog.test_connection())
        self.assertFalse(dialog.sync_button.isEnabled())
        self.assertIsNone(dialog.transfer_estimate)
        self.assertFalse(window.sync_project_action.isEnabled())

    def test_file_changes_during_final_confirmation_do_not_upload(self):
        window = self.linked_window()
        self.assertIsNotNone(window.estimate_project_backup())

        def change_after_preview(*args):
            (self.source / 'frames/new-frame.png').write_bytes(b'new data')
            return True

        with patch('gui.main_window.confirm_publication', side_effect=change_after_preview), \
                patch('core.project_sync.shutil.copy2') as copy, patch.object(QMessageBox, 'warning') as warning:
            self.assertFalse(window.sync_current_project())
            copy.assert_not_called()
            self.assertIn('Estimate transfer size again', warning.call_args.args[2])
        self.assertFalse(list(self.remote.rglob('manifest.json')))
        self.assertFalse(window.sync_project_action.isEnabled())

    def test_compact_is_default_and_custom_rollback_location_is_saved(self):
        rollback = self.base / 'outside-dropbox'
        rollback.mkdir()
        dialog = ProjectSyncDialog(self.source)
        self.assertEqual(dialog.sync_button.text(), 'Back Up Local Project')
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
            dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
        QTimer.singleShot(0, cancel)
        self.assertFalse(confirm_publication(None, {
            'files': 28058, 'logical_bytes': int(45.1 * 1024**3),
            'new_bytes': 2 * 1024**2, 'free_bytes': 100 * 1024**3}, {}))

    def test_publication_preview_reports_pending_archive_rename(self):
        messages = []
        def cancel():
            dialog = self.app.activeModalWidget()
            messages.append(dialog.message.toPlainText())
            dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
        QTimer.singleShot(0, cancel)
        self.assertFalse(confirm_publication(None, {
            'files': 1, 'logical_bytes': 1, 'new_bytes': 1, 'free_bytes': 100,
            'project_name': '2026_CV4E', 'rename_from': 'test--12345678'}, {}))
        self.assertIn('test--12345678', messages[0])
        self.assertIn('2026_CV4E', messages[0])
        self.assertIn('keeping its ID and history', messages[0])


if __name__ == '__main__':
    unittest.main()
