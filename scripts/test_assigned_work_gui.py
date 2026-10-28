"""Assigned-work dialogs and the normal annotation save/publish path."""

from contextlib import ExitStack
from pathlib import Path
import os
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog

from core.assigned_work import read_assignments, save_assignments, workspace
from core.contributors import new_session
from gui.assigned_work_dialog import (AssignmentManagerDialog, AssignedDownloadDialog,
                                      assigned_local_path, managed_copy_key)
from gui.annotation_policy_dialog import AnnotationPolicyDialog
from gui.main_window import MainWindow
from gui.project_sync_dialog import ProjectSyncDialog, save_setting
from scripts import test_assigned_work as fixtures


class AssignedWorkGuiTests(unittest.TestCase):
    view = fixtures.AssignedWorkTests.view
    download = fixtures.AssignedWorkTests.download
    status = fixtures.AssignedWorkTests.status

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        fixtures.AssignedWorkTests.setUp(self)
        self.settings = QSettings(str(self.base / 'settings.ini'), QSettings.Format.IniFormat)
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, {'XDG_DATA_HOME': str(self.base / 'appdata')}))
        self.stack.enter_context(patch('gui.workspaces.WorkspaceController.confirm_upload_on_close', return_value=True))
        self.stack.enter_context(patch('gui.project_sync_dialog.QSettings', lambda: self.settings))
        self.stack.enter_context(patch('gui.main_window.QSettings', lambda: self.settings))
        self.stack.enter_context(patch.object(QMessageBox, 'critical', side_effect=AssertionError('Unexpected critical dialog')))
        self.stack.enter_context(patch.object(QMessageBox, 'information'))
        self.warnings = self.stack.enter_context(patch.object(QMessageBox, 'warning'))
        save_setting('project_sync/destination', self.configuration)
        self.window = None

    def tearDown(self):
        if self.window:
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                self.window.close()
            self.app.processEvents()
        self.stack.close()

    def make_window(self, local):
        with patch.object(MainWindow, 'load_settings'):
            self.window = MainWindow()
        self.window.set_contributor(self.alex)
        self.window.load_project(local)
        policy = getattr(self.window, '_policy_dialog', None)
        if policy:
            policy.close()
        return self.window

    def test_coordinator_edits_ranges_and_can_preview_status(self):
        dialog = AssignmentManagerDialog(self.source, self.configuration, self.coordinator)
        actor, frames = dialog.controls['alpha']
        self.assertEqual(actor.currentData(), self.alex['id'])
        frames.setText('0-1')
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            dialog.save()
        self.warnings.assert_not_called()
        self.assertEqual(read_assignments(self.view())['videos']['alpha']['frames'], [0, 1])
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-assignments.png')
        dialog.close()

    def test_student_download_dialog_only_lists_their_work(self):
        dialog = AssignedDownloadDialog(self.configuration, self.alex)
        self.assertEqual(dialog.videos, ['alpha'])
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-my-assignments.png')
        with patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted), \
                patch('gui.assigned_work_dialog.QFileDialog.getExistingDirectory') as chooser:
            dialog.download_selected()
        chooser.assert_not_called()
        self.warnings.assert_not_called()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertTrue(workspace(dialog.download_result['path']))

    def test_repeat_download_uses_same_path_without_parent_chooser(self):
        with patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted), \
                patch('gui.assigned_work_dialog.QFileDialog.getExistingDirectory') as chooser:
            first = AssignedDownloadDialog(self.configuration, self.alex)
            first.download_selected()
            second = AssignedDownloadDialog(self.configuration, self.alex)
            second.download_selected()
        self.warnings.assert_not_called()
        chooser.assert_not_called()
        self.assertEqual(first.download_result['path'], second.download_result['path'])

    def test_unavailable_remembered_copy_does_not_create_another(self):
        save_setting('assigned_work/locations', {
            managed_copy_key(self.configuration, self.project_id, self.alex): str(self.base / 'missing')})
        with self.assertRaisesRegex(ValueError, 'No replacement copy'):
            assigned_local_path(self.configuration, self.project_id, self.alex)

    def test_existing_copy_can_be_adopted_without_copying_or_merging(self):
        local = self.download()
        window = self.make_window(local)
        with patch('gui.assigned_work_dialog.QInputDialog.getItem', return_value=(str(local), True)), \
                patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted):
            dialog = AssignedDownloadDialog(self.configuration, self.alex, window)
            dialog.download_selected()
        self.warnings.assert_not_called()
        self.assertEqual(dialog.download_result['path'], str(local))
        self.assertEqual(dialog.download_result['bytes'], 0)

    def test_main_window_refresh_saves_edits_and_reloads_same_path(self):
        local = self.download()
        window = self.make_window(local)
        annotations = window.canvas.get_annotations()
        annotations[0]['mask'][0, 0] = 255
        window.canvas.set_annotations(annotations)
        window._mark_current_annotations_dirty()

        def download(dialog):
            dialog.download_selected()
            return dialog.result()

        with patch.object(AssignedDownloadDialog, 'exec', download), \
                patch('gui.assigned_work_dialog.QInputDialog.getItem', return_value=(str(local), True)), \
                patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted):
            self.assertTrue(window.open_assigned_work(project_id=self.project_id))
        self.warnings.assert_not_called()
        self.assertFalse(window._sync_in_progress)
        self.assertTrue(window.preload_worker.is_alive())
        loaded = window.annotation_manager.load_frame_annotations(local, 'alpha', 0)
        self.assertEqual(loaded[0]['mask'][0, 0], 255)

    def test_assigned_copy_can_open_local_rollback_history(self):
        window = self.make_window(self.download())
        self.assertTrue(window.rollback_history_action.isEnabled())
        with patch('gui.main_window.ProjectRollbackDialog') as dialog:
            dialog.return_value.update_result = None
            window.update_current_project(history=True)
            dialog.assert_called_once()
        self.warnings.assert_not_called()

    def test_upload_reuses_session_cache_and_does_not_persist_it(self):
        window = self.make_window(self.download())
        result = {'path': 'test-publication', 'copied_files': 0}
        with patch('gui.main_window.publish_work', return_value=result) as publish:
            self.assertTrue(window.publish_assigned_work())
            self.assertTrue(window.publish_assigned_work())
        self.warnings.assert_not_called()
        self.assertEqual(publish.call_count, 2)
        for call in publish.call_args_list:
            self.assertIs(call.kwargs['checksum_cache'], window._assigned_upload_cache)
        self.assertFalse(any('checksum' in key for key in self.settings.allKeys()))

    def test_upload_confirmation_counts_files_and_associated_videos(self):
        window = self.make_window(self.download())
        for count, uploaded, changed, expected in (
            (5, ['alpha'], ['alpha'], '5 changed files (associated with 1 video) uploaded.'),
            (1, ['alpha'], ['alpha'], '1 changed file (associated with 1 video) uploaded.'),
            (1, ['alpha', 'beta'], ['alpha', 'beta'],
             '1 changed file (associated with 2 videos) uploaded.'),
            (0, [], [], 'No new file uploads were needed.'),
            (0, [], ['alpha'], 'Changes for 1 video published. No new file uploads were needed.'),
        ):
            with self.subTest(count=count, uploaded=uploaded, changed=changed):
                result = {'path': 'test-publication', 'copied_files': count,
                          'uploaded_videos': uploaded, 'changed_videos': changed}
                with patch('gui.main_window.publish_work', return_value=result), \
                        patch.object(QMessageBox, 'information') as notice:
                    self.assertTrue(window.publish_assigned_work())
                self.assertEqual(notice.call_args.args[2], expected + '\n\nLocal work is saved.')
        self.warnings.assert_not_called()

    def test_no_assignments_and_same_name_different_identity(self):
        dialog = AssignedDownloadDialog(self.configuration, new_session('Alex'))
        self.assertEqual(dialog.videos, [])
        self.assertFalse(dialog.download.isEnabled())
        dialog.close()

    def test_actual_gui_edit_save_publish_is_annotation_only(self):
        local = self.download()
        window = self.make_window(local)
        self.assertEqual(window.video_list.count(), 1)
        self.assertEqual(len(window.frames), 1)
        self.assertEqual(window.current_video_id, 'alpha')
        self.assertTrue(all(not action.isEnabled() for action in window.full_project_actions))
        self.assertEqual(window.sync_project_action.text(), 'Upload Assigned Annotations')
        self.assertFalse(window.toolbar.imaging_setup_combo.isEnabled())
        window.show()
        self.app.processEvents()
        window.grab().save('/tmp/bumblebox-assigned-work.png')
        annotations = window.canvas.get_annotations()
        annotations[0]['mask'][0, 0] = 255
        window.canvas.set_annotations(annotations)
        window._mark_current_annotations_dirty()
        window.sync_current_project()
        self.warnings.assert_not_called()
        self.assertEqual(self.status()[0], 'In Progress')
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            window.publish_assigned_work(ready=True)
        self.warnings.assert_not_called()
        self.assertEqual(self.status()[0], 'Ready for Review')
        self.assertFalse(window._sync_in_progress)

    def test_policy_readonly_and_full_update_disabled(self):
        local = self.download()
        dialog = AnnotationPolicyDialog(local, self.alex)
        self.assertFalse(dialog.editing)
        dialog.begin_editing()
        self.assertFalse(dialog.editing)
        self.assertTrue(dialog.edit_button.isHidden())
        self.assertTrue(dialog.save_button.isHidden())
        self.assertTrue(all(not widget.isEnabled() for widget in dialog.policy.meanings.values()))
        settings = ProjectSyncDialog(local)
        self.assertEqual(settings.sync_button.text(), 'Upload Assigned Annotations')
        self.assertFalse(settings.update_button.isEnabled())
        dialog.close()
        settings.close()

    def test_sampled_assignments_keep_physical_frame_indices(self):
        data = read_assignments(self.view())
        self.choices['alpha']['frames'] = [0, 2]
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        local = self.download()
        window = self.make_window(local)
        self.assertEqual([p.stem for p in window.frames], ['frame_000000', 'frame_000002'])
        untouched = (local / 'annotations/json/alpha/frame_000001.json').read_bytes()
        window.load_frame(1)
        self.assertEqual(window._get_frame_idx_in_video(window.current_frame_idx), 2)
        window.canvas.set_annotations([])
        window._mark_current_annotations_dirty()
        window.sync_current_project()
        self.warnings.assert_not_called()
        self.assertEqual((local / 'annotations/json/alpha/frame_000001.json').read_bytes(), untouched)
        self.assertEqual(window.annotation_manager.load_frame_annotations(local, 'alpha', 2), [])

    def test_wrong_contributor_is_blocked_and_normal_project_restores_controls(self):
        local = self.download()
        window = self.make_window(local)
        with patch('gui.main_window.choose_contributor', return_value=self.sam):
            window.change_contributor()
        self.assertEqual(window.contributor_session['id'], self.alex['id'])
        self.assertIn('belongs to Alex', self.warnings.call_args.args[2])
        window.load_video_frames('beta')
        self.assertEqual(window.current_video_id, 'alpha')
        window.load_project(self.source)
        self.assertEqual(window.video_list.count(), 2)
        self.assertTrue(window.manage_assignments_action.isEnabled())
        self.assertEqual(window.sync_project_action.text(), 'Back Up Local Project')


if __name__ == '__main__':
    unittest.main()
