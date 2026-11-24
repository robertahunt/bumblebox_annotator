"""The background intake trigger and the Shared Work Intake window, on disposable fixtures."""

import time
import unittest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication, QMessageBox

from core.intake import CHANGED_HERE, held_work
from gui.automatic_intake import SETTING, IntakeDialog
from gui.project_sync_dialog import save_setting
from scripts import test_assigned_work as backend
from scripts import test_intake as core_tests
from scripts import test_workspaces as fixture


class IntakeGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        fixture.WorkspaceGuiTests.setUp(self)
        self.intake = self.window.workflow.automatic_intake
        self.intake.timer.stop()  # each test decides when a check happens

    tearDown = fixture.WorkspaceGuiTests.tearDown
    load = fixture.WorkspaceGuiTests.load
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    status = backend.AssignedWorkTests.status
    edit = core_tests.IntakeTests.edit
    edit_hive = core_tests.IntakeTests.edit_hive
    publish = core_tests.IntakeTests.publish
    files = core_tests.IntakeTests.files

    def wait(self):
        deadline = time.monotonic() + 60
        while self.intake.worker and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()
        self.assertIsNone(self.intake.worker)

    def check(self, **options):
        self.assertTrue(self.intake.check(force=True, **options))
        self.wait()

    def test_uploaded_work_reaches_the_director_project_without_a_click(self):
        self.load()
        theirs = self.publish()
        self.window.coco_export_dirty = False
        self.check()
        self.assertEqual(self.files(self.source), self.files(theirs))
        self.assertEqual(held_work(self.source), {'alpha': {'frames': [0], 'shared': []}})
        self.assertIn('Took in new work on 1 video from Alex', self.window.status_label.text())
        self.assertTrue(self.window.coco_export_dirty)
        self.assertFalse(self.window._sync_in_progress)
        self.assertTrue(self.window.centralWidget().isEnabled())

    def test_a_timer_tick_takes_work_in_when_the_setting_is_on_and_not_when_it_is_off(self):
        self.load()
        theirs = self.publish()
        before = self.files(self.source)
        save_setting(SETTING, False)
        self.intake.tick()
        self.assertIsNone(self.intake.worker)
        self.assertEqual(self.files(self.source), before)
        save_setting(SETTING, True)
        self.intake.tick()
        self.wait()
        self.assertEqual(self.files(self.source), self.files(theirs))

    def test_the_open_video_shows_their_work_afterwards(self):
        self.load()
        self.window.load_video_frames('alpha')
        self.assertEqual(self.window.current_video_id, 'alpha')
        frame = self.window.current_frame_idx
        self.publish()
        self.check()
        self.assertEqual((self.window.current_video_id, self.window.current_frame_idx), ('alpha', frame))
        shown = self.window.annotation_manager.load_frame_annotations(self.source, 'alpha', 0)
        self.assertEqual(shown[0]['mask'][0, 0], 255)

    def test_another_open_video_is_left_on_screen(self):
        self.load()
        self.window.load_video_frames('beta')
        with patch.object(self.window, 'load_video_frames') as reload:
            self.publish()
            self.check()
            reload.assert_not_called()
        self.assertEqual(self.window.current_video_id, 'beta')
        self.assertEqual(held_work(self.source), {'alpha': {'frames': [0], 'shared': []}})

    def test_nothing_is_replaced_while_an_edit_is_unsaved(self):
        self.load()
        theirs = self.publish()
        before = self.files(self.source)
        self.window.current_frame_modified = True
        try:
            self.check()
            self.assertEqual(self.files(self.source), before)
            self.assertEqual(held_work(self.source), {})
        finally:
            self.window.current_frame_modified = False
        self.check()
        self.assertEqual(self.files(self.source), self.files(theirs))

    def test_someone_who_is_not_the_director_is_asked_once_and_never_again(self):
        self.window.set_contributor(self.alex)
        self.load()
        self.intake.timer.stop()
        self.publish(name='alex-copy')
        before = self.files(self.source)
        self.check()
        self.assertIn('director', self.intake.unavailable)
        self.assertFalse(self.intake.check())
        self.assertEqual(self.files(self.source), before)

    def test_the_window_lists_held_and_skipped_work_and_resolves_both(self):
        self.load()
        self.publish(ready=True)
        self.publish('beta', self.sam, 'sam-copy')
        self.edit(self.source, 'beta', 1, self.coordinator, (2, 2))  # changed here: beta is not taken in
        self.check()
        self.window.show_intake()
        dialog = self.window.intake_dialog
        self.assertIsInstance(dialog, IntakeDialog)
        rows = {dialog.table.item(row, 0).text(): dialog.table.item(row, 2).text()
                for row in range(dialog.table.rowCount())}
        self.assertEqual(rows, {'beta': 'Not taken in: ' + CHANGED_HERE, 'alpha': 'Held, 1 frame: not accepted yet'})
        self.assertEqual(dialog.table.item(0, 0).text(), 'beta')  # what needs a decision comes first
        dialog.table.selectRow(0)
        self.assertTrue(dialog.theirs_button.isEnabled() and dialog.mine_button.isEnabled())
        self.assertFalse(dialog.accept_button.isEnabled())
        dialog.theirs_button.click()
        self.wait()
        self.assertEqual(sorted(held_work(self.source)), ['alpha', 'beta'])
        dialog.table.selectAll()
        dialog.accept_button.click()
        self.assertEqual(self.status('alpha')[0], 'Accepted')
        self.assertEqual(sorted(held_work(self.source)), ['beta'])
        self.assertIn('Accepted 1 video.', dialog.status.text())
        self.assertIn('beta: It is not waiting for review.', dialog.status.text())
        dialog.table.selectAll()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            dialog.release_button.click()
        self.assertEqual(held_work(self.source), {})
        dialog.close()

    def test_the_setting_is_changed_from_the_window(self):
        self.load()
        self.window.show_intake()
        dialog = self.window.intake_dialog
        self.assertTrue(dialog.enabled.isChecked())
        dialog.enabled.setChecked(False)
        self.intake.tick()
        self.assertIsNone(self.intake.worker)
        dialog.enabled.setChecked(True)
        dialog.close()


if __name__ == '__main__':
    unittest.main()
