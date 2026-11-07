"""Startup journeys with disposable lab folders and no actual credentials."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from PyQt6.QtCore import QObject, QSettings, QTimer, Qt, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton, QToolButton

from core.contributors import new_session
from core.lab_access import check_lab_access
from core.project_sync import test_destination as probe
from gui import project_sync_dialog as preferences
from gui.contributor_chooser import ContributorChooser, ManageProfilesDialog
from gui.lab_startup import (LabAccessCheck, StartupConnectionDialog, check_session_connection,
                             ensure_computer_setup, set_working_locally, working_locally)
from gui.shared_profiles import ProfileSyncService


class FakeCheck(QObject):
    completed = pyqtSignal(object)

    def __init__(self, result=None):
        super().__init__()
        self.result = result
        self.cancelled = False
        self.started = False

    def start(self):
        self.started = True
        if self.result is not None:
            QTimer.singleShot(0, lambda: self.completed.emit(self.result))

    def cancel(self):
        self.cancelled = True


class LabStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'shared'
        self.root.mkdir()
        self.configuration = probe(self.root)
        self.settings = QSettings(str(self.base / 'settings.ini'), QSettings.Format.IniFormat)
        self.preferences = patch.object(preferences, 'QSettings', lambda: self.settings)
        self.preferences.start()
        self.service = ProfileSyncService()
        self.service.timer.stop()
        self.sharing = patch('gui.shared_profiles.profile_sync_service', return_value=self.service)
        self.sharing.start()
        self.chooser_service = patch('gui.contributor_chooser.profile_sync_service', return_value=self.service)
        self.chooser_service.start()
        set_working_locally(False)

    def tearDown(self):
        self.service.stop()
        set_working_locally(False)
        self.app.processEvents()
        self.chooser_service.stop()
        self.sharing.stop()
        self.preferences.stop()
        self.temp.cleanup()

    def save_connection(self):
        preferences.save_setting('project_sync/destination', self.configuration)

    def person(self, name='Alex'):
        person = new_session(name)
        preferences.remember_contributor_profile(person)
        return person

    def test_probe_is_read_only_and_checks_saved_identity(self):
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        self.assertTrue(check_lab_access(self.configuration)['connected'])
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
        self.assertEqual(before, after)
        with self.assertRaisesRegex(ValueError, 'not the configured'):
            check_lab_access(dict(self.configuration, root_id='wrong-root'))
        self.root.rename(self.base / 'unmounted')
        with self.assertRaises(ValueError):
            check_lab_access(self.configuration)
        self.assertFalse(self.root.exists())

    def test_real_helper_success_and_missing_drive(self):
        for configuration, connected in ((self.configuration, True),
                                         (dict(self.configuration, root=str(self.base / 'missing')), False)):
            results = []
            check = LabAccessCheck(configuration)
            check.completed.connect(results.append)
            check.start()
            for _ in range(300):
                if results:
                    break
                QTest.qWait(10)
            self.assertTrue(results)
            self.assertEqual(results[0]['connected'], connected)
            self.app.processEvents()

    def test_timeout_reaps_slow_process_without_blocking_ui(self):
        results, destroyed = [], []
        check = LabAccessCheck(self.configuration, timeout_ms=50)
        check.completed.connect(results.append)
        check.destroyed.connect(lambda: destroyed.append(True))
        original_start = check.process.start
        with patch.object(check.process, 'start', side_effect=lambda *_:
                original_start(sys.executable, ['-c', 'import sys,time; sys.stdin.read(); time.sleep(5)'])):
            check.start()
            for _ in range(150):
                if destroyed:
                    break
                QTest.qWait(10)
        self.assertTrue(destroyed)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]['connected'])
        self.assertIn('did not respond', results[0]['error'])

    def test_connection_success_continues_without_starting_login(self):
        self.save_connection()
        check = FakeCheck({'connected': True})
        with patch('gui.lab_startup.LabAccessCheck', return_value=check), \
                patch('gui.drive_login.start_login') as login:
            self.assertTrue(check_session_connection('Alex'))
        self.assertFalse(working_locally())
        login.assert_not_called()

    def test_unavailable_connection_offers_local_work_and_pauses_background_sync(self):
        self.save_connection()
        check = FakeCheck({'connected': False, 'error': 'Permission denied'})
        observed = {}
        def respond():
            dialog = self.app.activeModalWidget()
            observed['note'] = dialog.note.text()
            observed['sign_in'] = dialog.login.isVisible()
            observed['details'] = dialog.details.isHidden()
            dialog.grab().save('/tmp/bumblebox-startup-drive-offline.png')
            dialog.resize(480, 280)
            self.app.processEvents()
            dialog.grab().save('/tmp/bumblebox-startup-drive-offline-narrow.png')
            dialog.local.click()
        with patch('gui.lab_startup.LabAccessCheck', return_value=check):
            QTimer.singleShot(30, respond)
            self.assertFalse(check_session_connection('Alex'))
        self.assertTrue(working_locally())
        self.assertTrue(observed['sign_in'])
        self.assertTrue(observed['details'])
        self.assertIn('cannot download assignments or upload', observed['note'])
        self.assertEqual(preferences.setting_json('project_sync/destination', {}), self.configuration)
        with patch('gui.shared_profiles.sync_profiles') as sync:
            self.service.refresh()
            sync.assert_not_called()

    def test_continue_locally_during_slow_check_ignores_late_success(self):
        self.save_connection()
        check = FakeCheck()
        with patch('gui.lab_startup.LabAccessCheck', return_value=check):
            QTimer.singleShot(10, lambda: self.app.activeModalWidget().local.click())
            self.assertFalse(check_session_connection())
        self.assertTrue(check.cancelled)
        check.completed.emit({'connected': True})
        self.assertTrue(working_locally())

    def test_closed_dialog_does_not_start_a_queued_check_or_late_login_retry(self):
        with patch('gui.lab_startup.LabAccessCheck') as check:
            dialog = StartupConnectionDialog(self.configuration)
            dialog.reject()
            self.app.processEvents()
            dialog.login.completed.emit()
            check.assert_not_called()

    def test_login_completion_rechecks_access_not_just_process_exit(self):
        first = FakeCheck({'connected': False, 'error': 'Expired ticket'})
        second = FakeCheck({'connected': False, 'error': 'Wrong mounted folder'})
        third = FakeCheck({'connected': True})
        with patch('gui.lab_startup.LabAccessCheck', side_effect=[first, second, third]):
            dialog = StartupConnectionDialog(self.configuration)
            dialog.show()
            QTest.qWait(20)
            dialog.login.completed.emit()
            QTest.qWait(20)
            self.assertTrue(dialog.isVisible())
            self.assertIn('Wrong mounted folder', dialog.details.toPlainText())
            dialog.retry.click()
            QTest.qWait(20)
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_no_lab_configuration_never_prompts_or_starts_process(self):
        with patch('gui.lab_startup.StartupConnectionDialog') as dialog:
            self.assertTrue(check_session_connection())
            dialog.assert_not_called()

    def test_reconnecting_clears_local_mode(self):
        self.save_connection()
        set_working_locally(True)
        with patch('gui.lab_startup.LabAccessCheck', return_value=FakeCheck({'connected': True})):
            self.assertTrue(check_session_connection())
        self.assertFalse(working_locally())

    def test_first_computer_setup_is_remembered_and_cancel_is_not(self):
        QTimer.singleShot(0, lambda: self.app.activeModalWidget().reject())
        self.assertFalse(ensure_computer_setup())
        self.assertFalse(self.settings.value('lab/computer_setup_complete', False, type=bool))
        def local():
            dialog = self.app.activeModalWidget()
            dialog.grab().save('/tmp/bumblebox-computer-setup.png')
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Use Locally').click()
        QTimer.singleShot(0, local)
        self.assertTrue(ensure_computer_setup())
        with patch('gui.lab_startup.QDialog') as dialog:
            self.assertTrue(ensure_computer_setup())
            dialog.assert_not_called()

    def test_existing_local_users_do_not_repeat_computer_setup(self):
        self.person()
        with patch('gui.lab_startup.QDialog') as dialog:
            self.assertTrue(ensure_computer_setup())
            dialog.assert_not_called()

    def test_chooser_contains_only_tiles_and_management(self):
        self.person('Alex')
        self.person('August')
        dialog = ContributorChooser()
        dialog.show()
        self.app.processEvents()
        buttons = [button.text() for button in dialog.findChildren(QPushButton)]
        self.assertEqual(buttons, [])
        self.assertEqual([b.text() for b in dialog.findChildren(QToolButton)], ['Manage Profiles...'])
        self.assertEqual(dialog.tiles.count(), 2)
        dialog.grab().save('/tmp/bumblebox-focused-profile-chooser.png')
        dialog.resize(460, 360)
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-focused-profile-chooser-narrow.png')
        dialog.reject()

    def test_missing_shared_profiles_offer_login_only_without_cached_identities(self):
        preferences.save_setting('contributors/shared', dict(self.configuration, enabled=True))
        for cached in (False, True):
            if cached:
                self.person()
            with patch('gui.lab_startup.LabAccessCheck',
                       return_value=FakeCheck({'connected': False, 'error': 'Expired ticket'})), \
                    patch('gui.lab_startup.check_session_connection') as connect:
                dialog = ContributorChooser()
                dialog.show()
                QTest.qWait(20)
                if cached:
                    connect.assert_not_called()
                else:
                    connect.assert_called_once_with(parent=dialog)
                dialog.reject()

    def test_remove_cancel_restore_and_refresh_keep_identity_and_assignments(self):
        person = self.person()
        preferences.save_setting('assignments/profiles', {person['id']: person})
        dialog = ManageProfilesDialog()
        dialog.refresh(selected=person['id'])
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Cancel):
            dialog.remove.click()
        self.assertEqual(dialog.people.count(), 1)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            dialog.remove.click()
        self.assertEqual(dialog.people.count(), 0)
        self.assertIn(person['id'], preferences.load_contributor_profiles())
        self.assertEqual(preferences.setting_json('assignments/profiles', {}), {person['id']: person})
        preferences.remember_contributor_profile(person)
        chooser = ContributorChooser()
        self.assertEqual(chooser.tiles.count(), 0)
        chooser.reject()
        dialog.removed.setChecked(True)
        dialog.refresh(selected=person['id'])
        self.assertEqual(dialog.remove.text(), 'Restore Profile')
        dialog.remove.click()
        chooser = ContributorChooser()
        self.assertEqual(chooser.tiles.count(), 1)
        chooser.reject()
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-profile-management.png')
        dialog.advanced.setChecked(True)
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-profile-management-advanced.png')
        dialog.reject()


if __name__ == '__main__':
    unittest.main()
