"""Shared identities use temporary drives/settings, never real lab data."""

import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch
from uuid import uuid4

from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QLineEdit, QListWidget, QMessageBox, QPushButton

from core.project_sync import test_destination as probe, _write_json
from core.shared_profiles import DIRECTORY, portable_profile, sync_profiles
from gui import project_sync_dialog as preferences
from gui.shared_profiles import ProfileSyncService, SharedProfileSettingsDialog


def profile(name='Alex'):
    return portable_profile({'id': str(uuid4()), 'name': name})


class SharedProfilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.configuration = probe(self.root)
        self.alex = profile()

    def tearDown(self):
        self.temp.cleanup()

    def test_discovery_is_read_only_until_explicit_publication(self):
        before = sorted(self.root.iterdir())
        self.assertEqual(sync_profiles(self.configuration, [])['profiles'], {})
        self.assertEqual(before, sorted(self.root.iterdir()))

    def test_roundtrip_only_portable_fields_and_no_rewriting(self):
        source = dict(self.alex, device={'id': 'private'}, session_id='private', director=True)
        result = sync_profiles(self.configuration, [source])
        target = self.root / DIRECTORY / (self.alex['id'] + '.json')
        self.assertEqual(json.loads(target.read_text()), self.alex)
        stamp = target.stat().st_mtime_ns
        self.assertEqual(result['published'], [self.alex['id']])
        sync_profiles(self.configuration, [source])
        self.assertEqual(stamp, target.stat().st_mtime_ns)
        self.assertEqual(sync_profiles(self.configuration, [])['profiles'], {self.alex['id']: self.alex})

    def test_duplicate_names_are_different_identities(self):
        other = profile()
        result = sync_profiles(self.configuration, [self.alex, other])
        self.assertEqual(set(result['profiles']), {self.alex['id'], other['id']})

    def test_concurrent_publications_retry_without_lost_identities(self):
        other = profile()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(sync_profiles, self.configuration, [p]) for p in (self.alex, other)]
            for future in futures:
                future.result()
        sync_profiles(self.configuration, [self.alex, other])
        self.assertEqual(set(sync_profiles(self.configuration, [])['profiles']), {self.alex['id'], other['id']})

    def test_shared_permissions_do_not_become_private_under_restrictive_umask(self):
        self.root.chmod(0o770)
        original = os.umask(0o077)
        try:
            sync_profiles(self.configuration, [self.alex])
        finally:
            os.umask(original)
        directory = self.root / DIRECTORY
        target = directory / (self.alex['id'] + '.json')
        self.assertEqual(directory.stat().st_mode & 0o777, 0o770)
        self.assertEqual(target.stat().st_mode & 0o777, 0o640)
        self.assertEqual(target.stat().st_gid, self.root.stat().st_gid)
        directory.chmod(0o700)
        sync_profiles(self.configuration, [profile('New')])
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)

    def test_same_id_name_conflict_is_not_overwritten(self):
        sync_profiles(self.configuration, [self.alex])
        result = sync_profiles(self.configuration, [dict(self.alex, name='Someone else')])
        self.assertEqual(result['profiles'][self.alex['id']], self.alex)
        self.assertFalse(result['published'])
        self.assertIn('different shared name', result['warnings'][0])

    def test_unavailable_or_wrong_root_never_creates_a_replacement(self):
        with self.assertRaises(ValueError):
            sync_profiles(dict(self.configuration, root_id=str(uuid4())), [self.alex])
        missing = self.root / 'offline'
        with self.assertRaises(ValueError):
            sync_profiles(dict(self.configuration, root=str(missing)), [self.alex])
        self.assertFalse(missing.exists())
        self.assertFalse((self.root / DIRECTORY).exists())

    def test_invalid_records_do_not_hide_valid_profiles(self):
        sync_profiles(self.configuration, [self.alex])
        directory = self.root / DIRECTORY
        (directory / (str(uuid4()) + '.json')).write_text('{')
        _write_json(directory / (str(uuid4()) + '.json'), self.alex)
        (directory / (str(uuid4()) + '.json')).write_text('x' * 4097)
        (directory / (str(uuid4()) + '.json')).symlink_to(directory / (self.alex['id'] + '.json'))
        result = sync_profiles(self.configuration, [])
        self.assertEqual(result['profiles'], {self.alex['id']: self.alex})
        self.assertEqual(len(result['warnings']), 4)

    def test_symlink_registry_is_rejected(self):
        elsewhere = self.root / 'elsewhere'
        elsewhere.mkdir()
        (self.root / DIRECTORY).symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(ValueError):
            sync_profiles(self.configuration, [self.alex])
        self.assertFalse(list(elsewhere.iterdir()))

    def test_busy_registry_still_allows_discovery_and_preserves_lock(self):
        sync_profiles(self.configuration, [self.alex])
        lock = self.root / '.contributor-profiles.lock'
        lock.write_text('another computer')
        result = sync_profiles(self.configuration, [profile('New')])
        self.assertEqual(result['profiles'], {self.alex['id']: self.alex})
        self.assertFalse(result['published'])
        self.assertTrue(result['warnings'])
        self.assertEqual(lock.read_text(), 'another computer')

    def test_cancel_and_failed_write_leave_no_partial_profile_or_lock(self):
        with self.assertRaises(InterruptedError):
            sync_profiles(self.configuration, [self.alex], lambda: True)
        self.assertFalse((self.root / '.contributor-profiles.lock').exists())
        with patch('core.shared_profiles._write_json', side_effect=OSError('Read only')):
            result = sync_profiles(self.configuration, [self.alex])
        self.assertTrue(result['warnings'])
        self.assertFalse(result['published'])
        self.assertFalse(list((self.root / DIRECTORY).glob('*.json')))
        self.assertFalse((self.root / '.contributor-profiles.lock').exists())


class SharedProfileGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'remote'
        self.root.mkdir()
        self.configuration = dict(probe(self.root), enabled=True, share_new=True, publish_ids=[])
        self.settings = QSettings(str(self.base / 'computer-a.ini'), QSettings.Format.IniFormat)
        self.settings_patch = patch.object(preferences, 'QSettings', lambda: self.settings)
        self.settings_patch.start()
        self.service = ProfileSyncService()
        self.service_patch = patch('gui.shared_profiles.profile_sync_service', return_value=self.service)
        self.service_patch.start()
        self.alex = profile()

    def tearDown(self):
        self.service.stop()
        self.app.processEvents()
        self.service_patch.stop()
        self.settings_patch.stop()
        self.temp.cleanup()

    def sync(self):
        self.service.refresh()
        for _ in range(500):
            QTest.qWait(10)
            if self.service.worker is None and not self.service.again:
                break
        self.assertIsNone(self.service.worker)

    def enable(self, **updates):
        preferences.save_setting('contributors/shared', dict(self.configuration, **updates))

    def test_legacy_migration_and_same_name_ids_survive_roundtrip(self):
        preferences.save_setting('contributors/profiles', {'Alex': self.alex['id']})
        self.assertEqual(preferences.load_contributor_profiles(), {self.alex['id']: self.alex})
        other = profile()
        preferences.remember_contributor_profile(other)
        self.assertEqual(set(preferences.load_contributor_profiles()), {self.alex['id'], other['id']})

    def test_disabled_sharing_never_touches_drive(self):
        preferences.remember_contributor_profile(self.alex, newly_created=True)
        with patch('gui.shared_profiles.sync_profiles') as operation:
            self.sync()
        operation.assert_not_called()
        self.assertFalse((self.root / DIRECTORY).exists())

    def test_two_computers_discover_without_changing_active_identity(self):
        self.enable()
        preferences.remember_contributor_profile(self.alex, newly_created=True)
        self.sync()
        self.settings = QSettings(str(self.base / 'computer-b.ini'), QSettings.Format.IniFormat)
        local = profile('August')
        preferences.remember_contributor_profile(local)
        self.settings.setValue('contributors/last_id', local['id'])
        self.settings.setValue('contributors/device_id', 'computer-b')
        self.enable()
        self.sync()
        self.assertEqual(set(preferences.load_contributor_profiles()), {self.alex['id'], local['id']})
        self.assertEqual(self.settings.value('contributors/last_id'), local['id'])
        self.assertEqual(self.settings.value('contributors/device_id'), 'computer-b')
        self.assertNotIn(local['id'], sync_profiles(self.configuration, [])['profiles'])
        self.assertEqual(preferences.setting_json('assignments/profiles', {}), {})

    def test_offline_creation_retries_without_losing_cached_profiles(self):
        self.enable()
        disconnected = self.base / 'offline'
        self.root.rename(disconnected)
        preferences.remember_contributor_profile(self.alex, newly_created=True)
        self.sync()
        self.assertIn('unavailable', self.service.status)
        self.assertFalse(self.root.exists())
        self.assertIn(self.alex['id'], preferences.load_contributor_profiles())
        disconnected.rename(self.root)
        self.sync()
        self.assertIn(self.alex['id'], sync_profiles(self.configuration, [])['profiles'])

    def test_only_checked_existing_profiles_and_opted_in_new_profiles_upload(self):
        other = profile('Private')
        preferences.remember_contributor_profile(self.alex)
        preferences.remember_contributor_profile(other)
        self.enable(publish_ids=[self.alex['id']], share_new=False)
        preferences.remember_contributor_profile(profile('Also private'), newly_created=True)
        self.sync()
        self.assertEqual(set(sync_profiles(self.configuration, [])['profiles']), {self.alex['id']})

    def test_setup_previews_profiles_and_cancellation_writes_nothing(self):
        preferences.remember_contributor_profile(self.alex)
        dialog = SharedProfileSettingsDialog(suggested=self.configuration)
        self.assertFalse(dialog.enabled.isChecked())
        self.assertEqual(dialog.profiles.count(), 1)
        self.assertEqual(dialog.profiles.item(0).checkState(), Qt.CheckState.Unchecked)
        dialog.enabled.setChecked(True)
        dialog.profiles.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.reject()
        self.assertEqual(preferences.setting_json('contributors/shared', {}), {})
        self.assertFalse((self.root / DIRECTORY).exists())

    def test_setup_enables_and_publishes_previewed_profile(self):
        preferences.remember_contributor_profile(self.alex)
        dialog = SharedProfileSettingsDialog(suggested=self.configuration)
        dialog.enabled.setChecked(True)
        dialog.profiles.item(0).setCheckState(Qt.CheckState.Checked)
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-profile-sharing-settings.png')
        dialog.resize(460, 400)
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-profile-sharing-settings-narrow.png')
        self.assertTrue(dialog.test_connection())
        dialog.accept()
        self.sync()
        self.assertIn(self.alex['id'], sync_profiles(self.configuration, [])['profiles'])
        self.assertEqual(preferences.setting_json('project_sync/destination', {}), {})

    def test_disabling_sharing_does_not_delete_shared_or_cached_profiles(self):
        self.enable(publish_ids=[self.alex['id']])
        preferences.remember_contributor_profile(self.alex)
        self.sync()
        dialog = SharedProfileSettingsDialog()
        dialog.enabled.setChecked(False)
        dialog.accept()
        self.sync()
        self.assertIn(self.alex['id'], preferences.load_contributor_profiles())
        self.assertIn(self.alex['id'], sync_profiles(self.configuration, [])['profiles'])

    def test_stale_result_after_destination_change_is_discarded(self):
        self.enable()
        entered, release = Event(), Event()
        def delayed(*args):
            entered.set()
            release.wait(3)
            return {'profiles': {self.alex['id']: self.alex}, 'warnings': [], 'published': []}
        with patch('gui.shared_profiles.sync_profiles', side_effect=delayed):
            self.service.refresh()
            self.assertTrue(entered.wait(2))
            self.enable(enabled=False)
            release.set()
            self.sync()
        self.assertNotIn(self.alex['id'], preferences.load_contributor_profiles())
        self.assertEqual(self.service.status, 'Profile sharing is off')

    def test_new_profile_shared_then_selected_on_second_computer(self):
        self.enable()
        def create():
            dialog = self.app.activeModalWidget()
            def name():
                editor = self.app.activeModalWidget()
                editor.findChild(QLineEdit).setText('Alex')
                editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
            QTimer.singleShot(0, name)
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'New Profile...').click()
        QTimer.singleShot(0, create)
        first = preferences.choose_contributor()
        self.sync()
        self.settings = QSettings(str(self.base / 'computer-b.ini'), QSettings.Format.IniFormat)
        self.enable()
        self.sync()
        def choose():
            dialog = self.app.activeModalWidget()
            tiles = dialog.findChild(QListWidget, 'contributorProfiles')
            tiles.setCurrentRow(0)
            dialog.accept()
        QTimer.singleShot(0, choose)
        second = preferences.choose_contributor()
        self.assertEqual(first['id'], second['id'])
        self.assertNotEqual(first['session_id'], second['session_id'])
        self.assertNotEqual(first['device']['id'], second['device']['id'])

    def test_startup_discovers_profile_without_selecting_it_automatically(self):
        sync_profiles(self.configuration, [self.alex])
        self.enable()
        observed = {}
        def inspect(attempt=0):
            dialog = self.app.activeModalWidget()
            tiles = dialog.findChild(QListWidget, 'contributorProfiles')
            if not tiles.count() and attempt < 50:
                QTimer.singleShot(10, lambda: inspect(attempt + 1))
                return
            observed['count'] = tiles.count()
            observed['selected'] = tiles.currentItem()
            observed['visible'] = dialog.isVisible()
            tiles.setCurrentRow(0)
            dialog.accept()
        QTimer.singleShot(10, inspect)
        chosen = preferences.choose_contributor()
        self.assertEqual(observed, {'count': 1, 'selected': None, 'visible': True})
        self.assertEqual(chosen['id'], self.alex['id'])

    def test_shutdown_interrupts_worker_without_late_settings_writes(self):
        self.enable()
        entered = Event()
        def delayed(configuration, outgoing, cancelled):
            entered.set()
            for _ in range(100):
                if cancelled():
                    raise InterruptedError('Cancelled')
                Event().wait(0.01)
            raise AssertionError('Worker should have been interrupted')
        with patch('gui.shared_profiles.sync_profiles', side_effect=delayed):
            self.service.refresh()
            self.assertTrue(entered.wait(2))
            self.service.stop()
            self.app.processEvents()
        self.assertIsNone(self.service.worker)
        self.assertFalse(preferences.load_contributor_profiles())

    def test_duplicate_tiles_and_local_hide_are_not_shared_deletion(self):
        other = profile()
        sync_profiles(self.configuration, [self.alex, other])
        self.enable()
        self.sync()
        observed = {}
        def inspect():
            dialog = self.app.activeModalWidget()
            tiles = dialog.findChild(QListWidget, 'contributorProfiles')
            observed['count'] = tiles.count()
            observed['labels'] = [tiles.item(i).text() for i in range(tiles.count())]
            tiles.setCurrentRow(0)
            observed['hidden'] = tiles.currentItem().data(Qt.ItemDataRole.UserRole)
            dialog.grab().save('/tmp/bumblebox-shared-profiles.png')
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Hide Profile').click()
            observed['after'] = tiles.count()
            dialog.reject()
        QTimer.singleShot(0, inspect)
        self.assertIsNone(preferences.choose_contributor())
        self.assertEqual(observed['count'], 2)
        self.assertEqual(observed['after'], 1)
        self.assertNotEqual(*observed['labels'])
        self.assertIn(observed['hidden'], preferences.setting_json('contributors/hidden', []))
        self.assertEqual(len(sync_profiles(self.configuration, [])['profiles']), 2)
        self.assertEqual(len(preferences.load_contributor_profiles()), 2)

    def test_background_refresh_after_chooser_closed_does_not_switch_session(self):
        preferences.remember_contributor_profile(self.alex)
        self.enable()
        entered, release = Event(), Event()
        other = profile('August')
        def delayed(*args):
            entered.set()
            release.wait(3)
            return {'profiles': {other['id']: other}, 'warnings': [], 'published': []}
        def choose():
            dialog = self.app.activeModalWidget()
            tiles = dialog.findChild(QListWidget, 'contributorProfiles')
            tiles.setCurrentRow(0)
            dialog.accept()
        with patch('gui.shared_profiles.sync_profiles', side_effect=delayed):
            self.service.refresh()
            self.assertTrue(entered.wait(2))
            QTimer.singleShot(0, choose)
            chosen = preferences.choose_contributor()
            release.set()
            self.sync()
        self.assertEqual(chosen['id'], self.alex['id'])
        self.assertEqual(self.settings.value('contributors/last_id'), self.alex['id'])
        self.assertIn(other['id'], preferences.load_contributor_profiles())

    def test_confirmed_duplicate_creation_keeps_both_profiles(self):
        preferences.remember_contributor_profile(self.alex)
        def create():
            dialog = self.app.activeModalWidget()
            def name():
                editor = self.app.activeModalWidget()
                editor.findChild(QLineEdit).setText('Alex')
                editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
            QTimer.singleShot(0, name)
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'New Profile...').click()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            QTimer.singleShot(0, create)
            created = preferences.choose_contributor()
        self.assertNotEqual(created['id'], self.alex['id'])
        self.assertEqual(len(preferences.load_contributor_profiles()), 2)

    def test_same_id_local_name_conflict_requires_reconciliation(self):
        preferences.remember_contributor_profile(self.alex)
        with self.assertRaisesRegex(ValueError, 'different local name'):
            preferences.remember_contributor_profile(dict(self.alex, name='Other'))
        self.assertEqual(preferences.load_contributor_profiles()[self.alex['id']], self.alex)


if __name__ == '__main__':
    unittest.main()
