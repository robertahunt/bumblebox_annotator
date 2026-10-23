"""Headless GUI coverage for contributor setup, save coordination, and sync."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (QApplication, QComboBox, QDialogButtonBox, QLineEdit,
                            QListWidget, QMessageBox, QPushButton)

from core.annotation import AnnotationManager
from core.contributors import new_session
from core.project_manager import ProjectManager
from core.project_sync import (SyncConflictError, _read_json, _write_json, restore_project,
                               sync_project, test_destination as _probe_destination)
from gui.main_window import MainWindow, SaveWorker
from gui.project_sync_dialog import (ProjectSyncDialog, SyncProgressDialog, choose_contributor,
                                     save_setting, setting_json, working_project_path)
from scripts.test_project_sync import annotation


def probe_destination(path):
    configuration = _probe_destination(path)
    configuration['storage_mode'] = 'full'
    rollback = Path(path).parent / 'rollbacks'
    rollback.mkdir(exist_ok=True)
    configuration['rollback_root'] = str(rollback)
    return configuration


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
        self.stack.enter_context(patch('gui.workspaces.WorkspaceController.confirm_upload_on_close', return_value=True))
        self.stack.enter_context(patch('gui.project_sync_dialog.QSettings', lambda: self.settings))
        self.stack.enter_context(patch('gui.main_window.QSettings', lambda: self.settings))
        self.stack.enter_context(patch('gui.main_window.confirm_publication', return_value=True))
        self.stack.enter_context(patch('core.project_delta.default_rollback_root', lambda: self.base / 'rollbacks'))
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
            tiles = dialog.findChild(QListWidget)
            if screenshot:
                dialog.grab().save(screenshot)
            existing = next((tiles.item(i) for i in range(tiles.count())
                             if tiles.item(i).text().casefold() == name.casefold()), None)
            if existing:
                QTest.mouseClick(tiles.viewport(), Qt.MouseButton.LeftButton,
                                 pos=tiles.visualItemRect(existing).center())
            else:
                def enter_name():
                    editor = self.app.activeModalWidget()
                    editor.findChild(QLineEdit).setText(name)
                    editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
                QTimer.singleShot(0, enter_name)
                next(b for b in dialog.findChildren(QPushButton) if b.text() == 'New Profile...').click()
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
        self.assertEqual(first['device'], second['device'])
        self.assertEqual(len(setting_json('contributors/profiles', {})), 2)

    def test_cancel_contributor_does_not_create_identity(self):
        QTimer.singleShot(0, lambda: self.app.activeModalWidget().reject())
        self.assertIsNone(choose_contributor())
        self.assertEqual(setting_json('contributors/profiles', {}), {})

    def test_startup_confirms_profile_before_home_resume_or_command_line_load(self):
        import main as startup

        profiles = {name: new_session(name)['id'] for name in ('August', 'Alex')}
        save_setting('contributors/profiles', profiles)
        for destination in ('home', 'resume', 'project', 'video'):
            with self.subTest(destination=destination):
                self.settings.setValue('contributors/last_name', 'Alex')
                self.settings.setValue('contributors/last_id', profiles['Alex'])
                args = SimpleNamespace(
                    project=str(self.source) if destination == 'project' else None,
                    video='video.mp4' if destination == 'video' else None,
                    sam2_checkpoint=None, coarse_yolo_checkpoint=None,
                    bbox_checkpoint=None, instance_focused_checkpoint=None)
                observed = {}
                with patch.object(startup, 'QApplication') as application, \
                        patch.object(startup, 'parse_args', return_value=args), \
                        patch.object(startup, 'MainWindow') as create_window:
                    application.return_value.exec.return_value = 0
                    window = create_window.return_value
                    window.workflow.home.auto_resume.isChecked.return_value = destination != 'home'

                    def respond():
                        dialog = self.app.activeModalWidget()
                        tiles = dialog.findChild(QListWidget)
                        observed.update(
                            preselected=tiles.currentItem().text(),
                            profiles=[tiles.item(i).text() for i in range(tiles.count())],
                            window_created=create_window.called)
                        QTest.mouseClick(tiles.viewport(), Qt.MouseButton.LeftButton,
                                         pos=tiles.visualItemRect(tiles.item(1)).center())

                    QTimer.singleShot(0, respond)
                    with self.assertRaises(SystemExit) as exit_result:
                        startup.main()
                    self.assertEqual(exit_result.exception.code, 0)
                    self.assertEqual(observed, {
                        'preselected': 'Alex', 'profiles': ['Alex', 'August'], 'window_created': False})
                    window.set_contributor.assert_called_once()
                    self.assertEqual(window.set_contributor.call_args.args[0]['id'], profiles['August'])
                    self.assertEqual(self.settings.value('contributors/last_name'), 'August')
                    self.assertEqual(setting_json('contributors/profiles', {}), profiles)
                    if destination == 'resume':
                        window.workflow.resume.assert_called_once_with()
                    else:
                        window.workflow.resume.assert_not_called()
                    if args.project:
                        window.load_project.assert_called_once_with(args.project)
                    else:
                        window.load_project.assert_not_called()
                    if args.video:
                        window.load_video.assert_called_once_with(args.video)
                    else:
                        window.load_video.assert_not_called()
                    window.show.assert_called_once_with()

    def test_cancelling_startup_keeps_previous_profile_and_never_opens_window(self):
        import main as startup

        profiles = {name: new_session(name)['id'] for name in ('August', 'Alex')}
        save_setting('contributors/profiles', profiles)
        self.settings.setValue('contributors/last_name', 'Alex')

        def cancel():
            dialog = self.app.activeModalWidget()
            dialog.findChild(QListWidget).setCurrentRow(1)
            dialog.reject()

        with patch.object(startup, 'QApplication') as application, \
                patch.object(startup, 'parse_args'), \
                patch.object(startup, 'MainWindow') as create_window:
            QTimer.singleShot(0, cancel)
            startup.main()
            create_window.assert_not_called()
            application.return_value.exec.assert_not_called()
        self.assertEqual(self.settings.value('contributors/last_name'), 'Alex')
        self.assertEqual(setting_json('contributors/profiles', {}), profiles)

    def test_profile_tiles_are_side_by_side_and_wrap_with_many_profiles(self):
        names = ['Alex', 'August', 'Sam', 'A Very Long Contributor Name ' * 2, 'Z' * 80]
        profiles = {name: new_session(name)['id'] for name in names}
        profiles.update({f'Contributor {i}': new_session(str(i))['id'] for i in range(20)})
        save_setting('contributors/profiles', profiles)
        self.settings.setValue('contributors/last_name', 'August')
        observed = {}

        def inspect():
            dialog = self.app.activeModalWidget()
            tiles = dialog.findChild(QListWidget)
            observed['dropdowns'] = len(dialog.findChildren(QComboBox))
            observed['count'] = tiles.count()
            observed['icons'] = all(not tiles.item(i).icon().isNull() for i in range(tiles.count()))
            observed['tooltips'] = all(tiles.item(i).toolTip() == tiles.item(i).text()
                                       for i in range(tiles.count()))
            rects = [tiles.visualItemRect(tiles.item(i)) for i in range(tiles.count())]
            observed['side_by_side'] = rects[0].top() == rects[1].top() and rects[0].right() < rects[1].left()
            observed['wraps'] = rects[3].top() > rects[0].top()
            observed['scrolls'] = tiles.verticalScrollBar().maximum() > 0
            observed['horizontal_scroll'] = tiles.horizontalScrollBar().maximum()
            dialog.grab().save('/tmp/bumblebox-profile-tiles-many.png')
            dialog.resize(460, 310)
            self.app.processEvents()
            observed['narrow_horizontal_scroll'] = tiles.horizontalScrollBar().maximum()
            tiles.scrollToBottom()
            self.app.processEvents()
            dialog.grab().save('/tmp/bumblebox-profile-tiles-narrow.png')
            dialog.reject()

        QTimer.singleShot(0, inspect)
        self.assertIsNone(choose_contributor())
        self.assertEqual(observed, {'dropdowns': 0, 'count': 25, 'icons': True, 'tooltips': True,
                                   'side_by_side': True, 'wraps': True, 'scrolls': True,
                                   'horizontal_scroll': 0, 'narrow_horizontal_scroll': 0})

    def test_profile_tiles_support_keyboard_selection(self):
        profiles = {name: new_session(name)['id'] for name in ('Alex', 'August', 'Sam')}
        save_setting('contributors/profiles', profiles)
        self.settings.setValue('contributors/last_name', 'Alex')

        def select():
            dialog = self.app.activeModalWidget()
            tiles = dialog.findChild(QListWidget)
            QTest.keyClick(tiles, Qt.Key.Key_Right)
            dialog.grab().save('/tmp/bumblebox-profile-tiles-preview.png')
            QTest.keyClick(tiles, Qt.Key.Key_Return)

        QTimer.singleShot(0, select)
        session = choose_contributor()
        self.assertEqual(session['id'], profiles['August'])

    def test_duplicate_name_requires_explicit_separate_identity_confirmation(self):
        original = self.choose('August')

        def respond():
            dialog = self.app.activeModalWidget()

            def enter_name():
                editor = self.app.activeModalWidget()
                editor.findChild(QLineEdit).setText('  august  ')
                editor.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()

            QTimer.singleShot(0, enter_name)
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'New Profile...').click()
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()

        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            QTimer.singleShot(0, respond)
            selected = choose_contributor()
        self.assertEqual(selected['id'], original['id'])
        self.assertNotEqual(selected['session_id'], original['session_id'])
        self.assertEqual(setting_json('contributors/profiles', {}), {'August': original['id']})

    def test_right_click_does_not_start_session(self):
        original = self.choose('August')
        observed = {}

        def respond():
            dialog = self.app.activeModalWidget()
            tiles = dialog.findChild(QListWidget)
            QTest.mouseClick(tiles.viewport(), Qt.MouseButton.RightButton,
                             pos=tiles.visualItemRect(tiles.item(0)).center())
            observed['still_open'] = dialog.isVisible()
            dialog.reject()

        QTimer.singleShot(0, respond)
        self.assertIsNone(choose_contributor())
        self.assertTrue(observed['still_open'])
        self.assertEqual(setting_json('contributors/profiles', {}), {'August': original['id']})

    def test_new_profile_form_disallows_blank_names_and_can_be_cancelled(self):
        observed = {}

        def respond():
            dialog = self.app.activeModalWidget()
            observed['continue_enabled'] = dialog.findChild(QDialogButtonBox).button(
                QDialogButtonBox.StandardButton.Ok).isEnabled()

            def cancel_creation():
                editor = self.app.activeModalWidget()
                name = editor.findChild(QLineEdit)
                name.setText('   ')
                observed['blank_enabled'] = editor.findChild(QDialogButtonBox).button(
                    QDialogButtonBox.StandardButton.Ok).isEnabled()
                observed['max_length'] = name.maxLength()
                name.setText('August')
                editor.reject()

            QTimer.singleShot(0, cancel_creation)
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'New Profile...').click()
            observed['count'] = dialog.findChild(QListWidget).count()
            dialog.reject()

        QTimer.singleShot(0, respond)
        self.assertIsNone(choose_contributor())
        self.assertEqual(observed, {'continue_enabled': False, 'blank_enabled': False,
                                   'max_length': 80, 'count': 0})
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
            self.assertIsNotNone(window.estimate_project_backup())
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

    def test_restore_prompt_cancellation_does_not_load_or_modify_backup(self):
        window = self.make_window()
        window._save_annotation_sources()
        configuration = probe_destination(self.remote)
        published = Path(sync_project(self.source, configuration, window.contributor_session)['path'])
        before = {p: p.read_bytes() for p in published.rglob('*') if p.is_file()}
        with patch('gui.project_sync_dialog.QMessageBox') as message:
            box = message.return_value
            create, cancel = object(), object()
            box.addButton.side_effect = [create, cancel]
            box.clickedButton.return_value = cancel
            window.load_project(published)
        self.assertEqual(window.project_path, self.source)
        self.assertFalse((published / 'marker_debug').exists())
        self.assertEqual(before, {p: p.read_bytes() for p in published.rglob('*') if p.is_file()})

    def test_restore_prompt_creates_and_opens_verified_local_copy(self):
        window = self.make_window()
        window._save_annotation_sources()
        configuration = probe_destination(self.remote)
        published = Path(sync_project(self.source, configuration, window.contributor_session)['path'])
        destination = self.base / 'restored'
        with patch('gui.project_sync_dialog.QMessageBox') as message, \
                patch('gui.project_sync_dialog.QFileDialog.getSaveFileName', return_value=(str(destination), '')):
            box = message.return_value
            create, cancel = object(), object()
            box.addButton.side_effect = [create, cancel]
            box.clickedButton.return_value = create
            window.load_project(published)
        self.assertEqual(window.project_path, destination)
        self.assertTrue((destination / 'annotations/project.json').is_file())
        self.assertFalse((published / 'marker_debug').exists())

    def test_pending_archive_and_container_are_blocked_before_loading(self):
        window = self.make_window()
        probe_destination(self.remote)
        pending = self.remote / '.pending-test' / 'project'
        pending.mkdir(parents=True)
        with patch.object(QMessageBox, 'warning') as warning:
            window.load_project(pending)
            window.load_project(self.remote)
        self.assertEqual(warning.call_count, 2)
        self.assertEqual(window.project_path, self.source)
        self.assertEqual(list(pending.iterdir()), [])

    def test_switching_copies_saves_pending_work_and_discards_old_frame_cache(self):
        window = self.make_window()
        configuration = probe_destination(self.remote)
        published = Path(sync_project(self.source, configuration, window.contributor_session)['path'])
        destination = self.base / 'restored'
        restore_project(published, destination)
        frame = destination / 'frames/video/frame_000000.jpg'
        cv2.imwrite(str(frame), np.full((32, 40, 3), 200, np.uint8))
        window.load_project(destination)
        self.assertEqual(window.project_path, destination)
        self.assertTrue(np.all(window.canvas.current_image == 200))
        self.assertEqual(len(AnnotationManager().load_frame_annotations(self.source, 'video', 0)), 1)
        self.assertEqual(AnnotationManager().load_frame_annotations(destination, 'video', 0), [])

    def test_failed_save_keeps_current_project_open(self):
        window = self.make_window()
        other = self.base / 'other'
        ProjectManager().create_project(other, 'Other')
        with patch.object(window, '_save_annotation_sources', side_effect=OSError('disk full')), \
                patch.object(QMessageBox, 'warning') as warning:
            window.load_project(other)
        self.assertEqual(window.project_path, self.source)
        self.assertIn('disk full', warning.call_args.args[2])

    def test_conflict_exception_survives_worker_thread(self):
        def fail(progress, cancelled):
            raise SyncConflictError('outdated copy')
        with self.assertRaises(SyncConflictError):
            SyncProgressDialog('Conflict', fail).run()

    def test_stale_gui_publish_can_cancel_or_save_separate_backup(self):
        window = self.make_window()
        configuration = probe_destination(self.remote)
        save_setting('project_sync/destination', configuration)
        save_setting('project_sync/projects', {str(self.source): {'enabled': True}})
        with patch.object(QMessageBox, 'information'):
            self.assertIsNotNone(window.estimate_project_backup())
            window.sync_current_project()
        output = setting_json('project_sync/projects', {})[str(self.source)]['last_path']
        other = self.base / 'other'
        restore_project(output, other)
        (other / 'annotations/new.json').write_text('new')
        sync_project(other, configuration, window.contributor_session)
        latest = next(self.remote.rglob('latest.json'))
        before = latest.read_bytes()
        with patch('gui.main_window.confirm_separate_backup', return_value=False) as confirm:
            self.assertIsNotNone(window.estimate_project_backup())
            window.sync_current_project()
            confirm.assert_called_once()
        self.assertEqual(latest.read_bytes(), before)
        with patch('gui.main_window.confirm_separate_backup', return_value=True), \
                patch.object(QMessageBox, 'information') as notice, patch.object(QMessageBox, 'warning') as warning:
            self.assertIsNotNone(window.estimate_project_backup())
            window.sync_current_project()
            warning.assert_not_called()
        self.assertEqual(notice.call_args.args[1], 'Separate Backup Saved')
        self.assertEqual(latest.read_bytes(), before)
        self.assertFalse(window._sync_in_progress)

    def test_profile_export_import_retains_contributor_but_not_device_identity(self):
        first = self.choose('August')
        profile_file = self.base / 'profile.json'

        def export():
            dialog = self.app.activeModalWidget()
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Export Profile...').click()
            dialog.reject()

        with patch('gui.project_sync_dialog.QFileDialog.getSaveFileName', return_value=(str(profile_file), '')), \
                patch.object(QMessageBox, 'information') as exported:
            QTimer.singleShot(0, export)
            self.assertIsNone(choose_contributor())
            self.assertEqual(exported.call_args.args[1], 'Profile Exported')
        profile = _read_json(profile_file)
        self.assertEqual(profile['id'], first['id'])
        self.assertNotIn('device', profile)
        self.settings.clear()

        def import_and_start():
            dialog = self.app.activeModalWidget()
            next(b for b in dialog.findChildren(QPushButton) if b.text() == 'Import Profile...').click()
            dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()

        with patch('gui.project_sync_dialog.QFileDialog.getOpenFileName', return_value=(str(profile_file), '')), \
                patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            QTimer.singleShot(0, import_and_start)
            second = choose_contributor()
        self.assertEqual(first['id'], second['id'])
        self.assertNotEqual(first['device']['id'], second['device']['id'])
        self.assertNotEqual(first['session_id'], second['session_id'])

    def test_failed_sync_does_not_disable_local_saving(self):
        window = self.make_window()
        configuration = probe_destination(self.remote)
        save_setting('project_sync/destination', configuration)
        save_setting('project_sync/projects', {str(self.source): {'enabled': True}})
        self.assertIsNotNone(window.estimate_project_backup())
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
