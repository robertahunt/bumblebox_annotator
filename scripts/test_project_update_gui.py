"""Local update dialogs and live canvas reloads, using disposable projects only."""

import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from core.annotation import AnnotationManager
from core.contributors import new_session
from core.project_sync import (restore_project, sync_project, update_journal_path, _write_json,
                               test_destination)
from core.project_update import PreparedUpdate, published_versions
from gui.main_window import MainWindow
from gui.project_sync_dialog import ProjectSyncDialog, save_setting, setting_json, working_project_path
from gui.project_update_dialog import ProjectUpdateDialog
from gui.project_rollback_dialog import ProjectRollbackDialog
from scripts import test_contributor_sync_gui as gui_fixtures
from scripts import test_project_update as core_fixtures
from scripts.test_project_sync import annotation

probe_destination = gui_fixtures.probe_destination


class UpdateDialogTests(unittest.TestCase):
    setUp = core_fixtures.ProjectUpdateTests.setUp
    tearDown = core_fixtures.ProjectUpdateTests.tearDown
    sync = core_fixtures.ProjectUpdateTests.sync
    publish_edit = core_fixtures.ProjectUpdateTests.publish_edit

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self):
        return ProjectUpdateDialog(self.local, self.configuration,
                                   published_versions(self.local, self.configuration))

    def test_preview_confirmation_and_apply(self):
        self.publish_edit()
        dialog = self.dialog()
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.preview()
        self.assertIsNotNone(dialog.plan)
        self.assertIn('1 additions', dialog.summary.text())
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.confirm.setChecked(True)
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-update-preview.png')
        dialog.apply_update()
        self.assertTrue(dialog.update_result['updated'])
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertTrue(Path(dialog.update_result['backup']).is_dir())

    def test_conflicts_are_displayed_without_apply(self):
        self.publish_edit()
        (self.local / 'annotations/local.json').write_text('[]')
        dialog = self.dialog()
        dialog.preview()
        self.assertIn('Update blocked', dialog.summary.text())
        self.assertFalse(dialog.confirm.isEnabled())
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.show()
        self.app.processEvents()
        dialog.grab().save('/tmp/bumblebox-update-conflict-preview.png')
        dialog.reject()

    def test_multiple_contributors_require_explicit_choice(self):
        self.publish_edit(contributor=new_session('Alex'))
        dialog = self.dialog()
        self.assertEqual(dialog.versions.currentIndex(), -1)
        self.assertFalse(dialog.preview_button.isEnabled())
        dialog.versions.setCurrentIndex(0)
        dialog.preview()
        self.assertIsNotNone(dialog.plan)
        dialog.versions.setCurrentIndex(1)
        self.assertIsNone(dialog.plan)
        self.assertFalse(dialog.apply_button.isEnabled())

    def test_apply_failure_requires_fresh_preview(self):
        self.publish_edit()
        dialog = self.dialog()
        dialog.preview()
        dialog.confirm.setChecked(True)
        (self.local / 'annotations/unexpected.json').write_text('[]')
        with patch.object(QMessageBox, 'warning') as warning:
            dialog.apply_update()
            warning.assert_called_once()
        self.assertIsNone(dialog.plan)
        self.assertIsNone(dialog.update_result)
        self.assertFalse(dialog.apply_button.isEnabled())


class UpdateWindowTests(unittest.TestCase):
    setUpClass = classmethod(gui_fixtures.ContributorSyncGuiTests.setUpClass.__func__)
    setUp = gui_fixtures.ContributorSyncGuiTests.setUp
    tearDown = gui_fixtures.ContributorSyncGuiTests.tearDown
    make_window = gui_fixtures.ContributorSyncGuiTests.make_window

    def publish_from_other_computer(self):
        window = self.make_window()
        window._save_annotation_sources()
        self.configuration = probe_destination(self.remote)
        save_setting('project_sync/destination', self.configuration)
        first = sync_project(self.source, self.configuration, window.contributor_session)
        other = self.base / 'other-computer'
        restore_project(first['path'], other)
        manager = AnnotationManager()
        manager.contributor_session = new_session('Alex')
        manager.save_frame_annotations(other, 'video', 0, [annotation(2)])
        sync_project(other, self.configuration, manager.contributor_session)
        return window

    def test_update_reloads_canvas_without_saving_stale_masks(self):
        window = self.publish_from_other_computer()
        before = core_fixtures.contents(self.source)
        remote_before = core_fixtures.contents(self.remote)
        def apply(dialog):
            dialog.versions.setCurrentIndex(next(
                i for i in range(dialog.versions.count()) if 'Alex' in dialog.versions.itemText(i)))
            dialog.preview()
            self.assertIsNotNone(dialog.plan, dialog.summary.text())
            self.assertFalse(dialog.plan.blocked, dialog.plan.local_changes)
            dialog.confirm.setChecked(True)
            dialog.apply_update()
            return dialog.result()
        with patch.object(ProjectUpdateDialog, 'exec', apply), \
                patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'warning') as warning:
            window.update_current_project()
            warning.assert_not_called()
        self.assertEqual([a['mask_id'] for a in window.canvas.get_annotations()], [2])
        self.assertFalse(window.current_frame_modified)
        self.assertFalse(window._sync_in_progress)
        self.assertTrue(window.preload_worker.is_alive())
        self.assertTrue(window.coco_export_dirty)
        state = setting_json('project_sync/projects', {})[str(self.source)]
        bundle = Path(state['last_update_backup'])
        previous = core_fixtures.contents(bundle / 'before')
        self.assertTrue(previous)
        self.assertEqual(previous, {name: before[name] for name in previous})
        self.assertFalse((bundle / 'before/frames').exists())
        self.assertEqual(core_fixtures.contents(self.remote), remote_before)
        versions = published_versions(self.source, self.configuration)
        selected = next(v for v in versions if v['manifest']['contributor']['name'] == 'Alex')
        plan = PreparedUpdate(self.source, self.configuration, selected['reference'])
        self.assertFalse(plan.local_changes, plan.local_changes)
        self.assertTrue(plan.noop)
        window._save_annotation_sources(force=True)
        saved = AnnotationManager().load_frame_annotations(self.source, 'video', 0)
        self.assertEqual([a['mask_id'] for a in saved], [2])
        self.assertEqual(saved[0]['provenance']['created_by']['name'], 'Alex')

    def test_unsaved_local_work_is_saved_and_conflict_blocks_update(self):
        window = self.publish_from_other_computer()
        window.canvas.set_annotations([annotation(3)])
        window._mark_current_annotations_dirty()
        def preview(dialog):
            dialog.versions.setCurrentIndex(next(
                i for i in range(dialog.versions.count()) if 'Alex' in dialog.versions.itemText(i)))
            dialog.preview()
            self.assertIn('Both local and published', dialog.plan.blocked)
            dialog.reject()
            return dialog.result()
        with patch.object(ProjectUpdateDialog, 'exec', preview), \
                patch.object(QMessageBox, 'warning') as warning:
            window.update_current_project()
            warning.assert_not_called()
        saved = AnnotationManager().load_frame_annotations(self.source, 'video', 0)
        self.assertEqual([a['mask_id'] for a in saved], [3])
        self.assertEqual([a['mask_id'] for a in window.canvas.get_annotations()], [3])
        self.assertTrue(window.preload_worker.is_alive())

    def test_history_undo_reloads_original_canvas_without_connected_drive(self):
        window = self.publish_from_other_computer()
        def apply(dialog):
            dialog.versions.setCurrentIndex(next(
                i for i in range(dialog.versions.count()) if 'Alex' in dialog.versions.itemText(i)))
            dialog.preview()
            dialog.confirm.setChecked(True)
            dialog.apply_update()
            return dialog.result()
        with patch.object(ProjectUpdateDialog, 'exec', apply), \
                patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'warning') as warning:
            window.update_current_project()
            warning.assert_not_called()
        self.assertEqual([a['mask_id'] for a in window.canvas.get_annotations()], [2])
        self.remote.rename(self.base / 'disconnected')
        def undo(dialog):
            self.assertEqual(len(dialog.items), 1)
            dialog.table.selectRow(0)
            dialog.restore_selected()
            return dialog.result()
        with patch.object(ProjectRollbackDialog, 'exec', undo), \
                patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'warning') as warning:
            window.update_current_project(history=True)
            warning.assert_not_called()
        self.assertEqual([a['mask_id'] for a in window.canvas.get_annotations()], [1])
        self.assertTrue(window.preload_worker.is_alive())
        self.assertFalse(window.current_frame_modified)
        window._save_annotation_sources(force=True)
        saved = AnnotationManager().load_frame_annotations(self.source, 'video', 0)
        self.assertEqual([a['mask_id'] for a in saved], [1])

    def test_settings_update_does_not_enable_publishing(self):
        dialog = ProjectSyncDialog(self.source)
        dialog.update_now()
        self.assertFalse(dialog.update_requested)
        dialog.folder.setText(str(self.remote))
        dialog.update_now()
        self.assertTrue(dialog.update_requested)
        self.assertFalse(dialog.sync_requested)
        self.assertFalse(setting_json('project_sync/projects', {})[str(self.source)]['enabled'])
        window = self.make_window()
        with patch('gui.main_window.ProjectSyncDialog', return_value=dialog), \
                patch.object(dialog, 'exec', return_value=QDialog.DialogCode.Accepted), \
                patch.object(window, 'update_current_project') as update:
            window.configure_project_sync()
            update.assert_called_once()

    def test_pending_marker_blocks_saves_and_open_requires_recovery_confirmation(self):
        window = self.make_window()
        journal = update_journal_path(self.source)
        journal.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            window._save_annotation_sources()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            self.assertIsNone(working_project_path(self.source))
        with patch.object(QMessageBox, 'warning'):
            save_setting('project_sync/destination', probe_destination(self.remote))
            window.update_current_project()
        self.assertIsNone(window.project_path)
        self.assertFalse(window.centralWidget().isEnabled())
        self.assertFalse(window._sync_in_progress)
        journal.unlink()

    def test_cannot_close_main_window_during_update(self):
        window = self.make_window()
        window._sync_in_progress = True
        event = QCloseEvent()
        window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        window._sync_in_progress = False

    def test_missing_working_folder_stays_in_recents_and_recovers_before_open(self):
        window = self.make_window()
        window._save_annotation_sources()
        window.preload_worker.stop()
        window.preload_worker.join()
        window._discard_annotation_view()
        window.project_path = None
        stage = self.base / '.bumblebox-update-stage-recovery-test'
        backup = self.base / 'bumblebox-before-update-recovery-test'
        stage.mkdir()
        backup.mkdir()
        _write_json(update_journal_path(self.source), {
            'format': 1, 'project': str(self.source), 'stage': stage.name,
            'backup': backup.name, 'phase': 'swapping'})
        self.source.rename(backup / 'project')
        window.update_recent_projects_menu()
        self.assertIn(str(self.source), window.recent_projects)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
                patch.object(QMessageBox, 'information'), patch.object(QMessageBox, 'warning') as warning:
            window.open_recent_project(self.source)
            warning.assert_not_called()
        self.assertEqual(window.project_path, self.source)
        self.assertTrue(window.centralWidget().isEnabled())
        self.assertEqual([a['mask_id'] for a in window.canvas.get_annotations()], [1])
        self.assertFalse(update_journal_path(self.source).exists())


if __name__ == '__main__':
    unittest.main()
