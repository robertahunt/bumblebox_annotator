"""Assigned-work dialogs and the normal annotation save/publish path."""

from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog

from core.assigned_work import read_assignments, save_assignments, workspace
from core.contributors import new_session
from gui.assigned_work_dialog import AssignmentManagerDialog, AssignedDownloadDialog
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
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                patch('gui.assigned_work_dialog.QFileDialog.getExistingDirectory', return_value=str(self.base)):
            dialog.download_selected()
        self.warnings.assert_not_called()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertTrue(workspace(dialog.download_result['path']))

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
        self.assertEqual(window.sync_project_action.text(), 'Publish Work...')
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
        self.assertEqual(settings.sync_button.text(), 'Publish Work')
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
        self.assertEqual(window.sync_project_action.text(), 'Publish Project Backup...')


if __name__ == '__main__':
    unittest.main()
