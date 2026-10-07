"""Headless GUI coverage for contributor setup, save coordination, and sync."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QMessageBox

from core.annotation import AnnotationManager
from core.contributors import new_session
from core.project_manager import ProjectManager
from core.project_sync import test_destination as probe_destination
from gui.main_window import MainWindow, SaveWorker
from gui.project_sync_dialog import (ProjectSyncDialog, SyncProgressDialog, choose_contributor,
                                     save_setting, setting_json)
from scripts.test_project_sync import annotation


class ContributorSyncGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / 'project'
        self.remote = self.base / 'remote'
        self.remote.mkdir()
        self.settings = QSettings(str(self.base / 'settings.ini'), QSettings.Format.IniFormat)
        self.stack = ExitStack()
        self.stack.enter_context(patch('gui.project_sync_dialog.QSettings', lambda: self.settings))
        self.stack.enter_context(patch('gui.main_window.QSettings', lambda: self.settings))
        self.window = None

    def tearDown(self):
        if self.window:
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                self.window.close()
            self.app.processEvents()
        self.stack.close()
        self.temp.cleanup()

    def choose(self, name, screenshot=None):
        def respond():
            dialog = self.app.activeModalWidget()
            combo = dialog.findChild(QComboBox)
            combo.setCurrentText(name)
            if screenshot:
                dialog.grab().save(screenshot)
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
        QTimer.singleShot(0, respond)
        return choose_contributor()

    def make_window(self):
        manager = ProjectManager()
        manager.create_project(self.source, 'Sync test')
        (self.source / 'input_data/train/video.mp4').write_bytes(b'test placeholder')
        frames = self.source / 'frames/video'
        frames.mkdir()
        cv2.imwrite(str(frames / 'frame_000000.jpg'), np.zeros((32, 40, 3), np.uint8))
        (frames / 'video_metadata.json').write_text('{"selected_frames": [0], "total_frames": 1}')
        with patch.object(MainWindow, 'load_settings'):
            self.window = MainWindow()
        self.window.set_contributor(new_session('August'))
        self.window.load_project(self.source)
        self.window.canvas.set_annotations([annotation()])
        self.window._mark_current_annotations_dirty()
        return self.window

    def test_startup_remembers_profiles_but_creates_new_session(self):
        first = self.choose('August', '/tmp/bumblebox-contributor-preview.png')
        second = self.choose('august')
        third = self.choose('Alex')
        self.assertEqual(first['id'], second['id'])
        self.assertNotEqual(first['session_id'], second['session_id'])
        self.assertNotEqual(second['id'], third['id'])
        self.assertEqual(len(setting_json('contributors/profiles', {})), 2)

    def test_cancel_contributor_does_not_create_identity(self):
        QTimer.singleShot(0, lambda: self.app.activeModalWidget().reject())
        self.assertIsNone(choose_contributor())
        self.assertEqual(setting_json('contributors/profiles', {}), {})

    def test_destination_setup_checks_connection_and_is_locally_persisted(self):
        dialog = ProjectSyncDialog(self.source)
        self.assertFalse(dialog.enabled.isChecked())
        dialog.folder.setText(str(self.remote))
        self.assertTrue(dialog.test_connection())
        dialog.enabled.setChecked(True)
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-sync-settings-preview.png')
        dialog.accept()
        self.assertTrue(setting_json('project_sync/projects', {})[str(self.source)]['enabled'])
        self.assertEqual(setting_json('project_sync/destination', {})['root'], str(self.remote))
        self.assertFalse(self.source.exists())
        later = ProjectSyncDialog(self.source)
        self.assertTrue(later.enabled.isChecked())
        later.enabled.setChecked(False)
        later.accept()
        self.assertFalse(setting_json('project_sync/projects', {})[str(self.source)]['enabled'])

    def test_empty_destination_is_never_tested_as_working_directory(self):
        dialog = ProjectSyncDialog(self.source)
        self.assertFalse(dialog.test_connection())
        self.assertIn('Choose a destination', dialog.status.text())
        dialog.enabled.setChecked(True)
        dialog.accept()
        self.assertEqual(setting_json('project_sync/destination', {}), {})

    def test_sync_setup_can_be_saved_before_any_project_is_open(self):
        dialog = ProjectSyncDialog()
        self.assertFalse(dialog.enabled.isEnabled())
        self.assertFalse(dialog.sync_button.isEnabled())
        dialog.folder.setText(str(self.remote))
        dialog.accept()
        self.assertEqual(setting_json('project_sync/destination', {})['root'], str(self.remote))
        self.assertEqual(setting_json('project_sync/projects', {}), {})

    def test_progress_reports_errors_without_publishing_success(self):
        def fail(progress, cancelled):
            raise OSError('Disconnected')
        with self.assertRaisesRegex(RuntimeError, 'Disconnected'):
            SyncProgressDialog('Sync test', fail).run()

    def test_first_run_can_skip_and_settings_remain_available(self):
        window = self.make_window()
        with patch('gui.main_window.QMessageBox') as message, \
                patch.object(window, 'configure_project_sync') as configure:
            box = message.return_value
            setup, skip = object(), object()
            box.addButton.side_effect = [setup, skip]
            box.clickedButton.return_value = skip
            window.offer_sync_setup()
            configure.assert_not_called()
            window.offer_sync_setup()
            box.exec.assert_called_once()
        self.assertTrue(self.settings.value('project_sync/setup_offered', False, type=bool))
        self.assertEqual(setting_json('project_sync/destination', {}), {})
        with patch('gui.main_window.ProjectSyncDialog') as dialog:
            dialog.return_value.exec.return_value = 0
            window.configure_project_sync()
            dialog.assert_called_once_with(window.project_path, window)

    def test_sync_now_while_disabled_does_not_publish_without_opt_in(self):
        window = self.make_window()
        with patch('gui.main_window.ProjectSyncDialog') as dialog, \
                patch('gui.main_window.sync_project') as publish:
            dialog.return_value.exec.return_value = 0
            window.sync_current_project()
            publish.assert_not_called()
        self.assertEqual(list(self.remote.iterdir()), [])

    def test_change_contributor_saves_pending_pixels_under_previous_identity(self):
        window = self.make_window()
        old = dict(window.contributor_session)
        other = new_session('Alex')
        with patch('gui.main_window.choose_contributor', return_value=other):
            window.change_contributor()
        saved = window.annotation_manager.load_frame_annotations(self.source, 'video', 0)
        self.assertEqual(saved[0]['provenance']['created_by']['id'], old['id'])
        self.assertEqual(window.annotation_manager.contributor_session['id'], other['id'])
        self.assertIn('Alex', window.contributor_label.text())
        self.assertEqual(window.canvas.annotation_metadata[1]['provenance']['created_by']['id'], old['id'])

    def test_queued_save_captures_original_session(self):
        manager = AnnotationManager()
        original = new_session('August')
        manager.contributor_session = original
        worker = SaveWorker(manager)
        worker.add_save_task(self.source, 'video', 0, [annotation()])
        manager.contributor_session = new_session('Alex')
        worker.start()
        try:
            worker.wait_until_idle()
        finally:
            worker.stop()
            worker.wait(5000)
        saved = manager.load_frame_annotations(self.source, 'video', 0)[0]
        self.assertEqual(saved['provenance']['created_by']['id'], original['id'])

    def test_sync_action_saves_and_publishes_reopenable_project(self):
        window = self.make_window()
        save_setting('project_sync/destination', probe_destination(self.remote))
        save_setting('project_sync/projects', {str(self.source): {'enabled': True}})
        with patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'warning') as warning:
            window.sync_current_project()
            warning.assert_not_called()
        state = setting_json('project_sync/projects', {})[str(self.source)]
        self.assertTrue(state['last_success'])
        output = Path(state['last_path'])
        reader = ProjectManager()
        reader.load_project(output)
        self.assertEqual(reader.scan_videos()['train'], ['video'])
        saved = AnnotationManager().load_frame_annotations(output, 'video', 0)[0]
        self.assertEqual(saved['provenance']['created_by']['name'], 'August')
        self.assertFalse(window._sync_in_progress)

    def test_failed_sync_does_not_disable_local_saving(self):
        window = self.make_window()
        configuration = probe_destination(self.remote)
        save_setting('project_sync/destination', configuration)
        save_setting('project_sync/projects', {str(self.source): {'enabled': True}})
        self.remote.rename(self.base / 'offline')
        with patch.object(QMessageBox, 'warning') as warning:
            window.sync_current_project()
            warning.assert_called_once()
        state = setting_json('project_sync/projects', {})[str(self.source)]
        self.assertTrue(state['last_error'])
        self.assertNotIn('last_success', state)
        self.assertTrue(window.annotation_manager.load_frame_annotations(self.source, 'video', 0))
        self.assertFalse(self.remote.exists())
        self.assertFalse(window._sync_in_progress)


if __name__ == '__main__':
    unittest.main()
