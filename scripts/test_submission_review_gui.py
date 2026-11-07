"""Review notification, editor navigation and approval using disposable fixtures."""

import unittest
from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from core.assigned_work import publish_work
from core.submission_review import PreparedSubmissionReview, review_queue
from gui.submission_comparison import SubmissionComparisonDialog
from gui.project_sync_dialog import save_setting
from scripts import test_assigned_work as backend
from scripts import test_workspaces as fixture


class SubmissionReviewGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        fixture.WorkspaceGuiTests.setUp(self)
        self.critical = self.stack.enter_context(patch.object(QMessageBox, 'critical'))
    tearDown = fixture.WorkspaceGuiTests.tearDown
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    change = backend.AssignedWorkTests.change
    status = backend.AssignedWorkTests.status
    load = fixture.WorkspaceGuiTests.load

    def submit(self, video='alpha', actor=None, name='working'):
        actor = actor or self.alex
        local = self.download(actor, [video], name)
        self.change(local, video, 0 if video == 'alpha' else 1, actor)
        publish_work(local, self.configuration, actor, ready=[video])

    def open_queue(self):
        self.window.review_submissions()
        return self.window.submission_review_dock

    def test_notification_buttons_open_review_not_assignments(self):
        self.load()
        inbox = self.window.workflow.home.inbox
        inbox.copies.clear()
        inbox.copies.addItem('Director copy', str(self.source))
        with patch.object(self.window, 'review_submissions') as review, \
                patch.object(self.window, 'manage_assignments') as assignments:
            self.window.workflow.review_notification.click()
            inbox.review_notice.click()
            self.assertEqual(review.call_count, 2)
            assignments.assert_not_called()

    def test_start_opens_first_submission_and_replaces_project_browser(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        dock = self.open_queue()
        self.assertEqual(dock.queue.count(), 1)
        self.assertIn('beta', dock.queue.item(0).text())
        self.assertIn('Sam', dock.queue.item(0).text())
        self.assertTrue(dock.is_focused())
        self.assertTrue(self.window.video_dock.isHidden())
        self.assertFalse(self.window.frame_dock.isHidden())
        self.assertEqual(self.window.dockWidgetArea(dock), Qt.DockWidgetArea.LeftDockWidgetArea)
        self.assertEqual(self.window.workflow.current, 'Annotate')
        self.assertEqual(self.window.current_video_id, 'beta')
        self.assertEqual([p.name for p in self.window.frames], ['frame_000001.jpg', 'frame_000002.jpg'])
        self.assertIn('Frame 0001', self.window.frame_list.item(0).text())
        self.assertIn('Frame 0002', self.window.frame_list.item(1).text())
        self.assertTrue(self.window.canvas.annotation_type_visibility['hive'])
        self.assertTrue(dock.approve_button.isEnabled())
        self.critical.assert_not_called()
        self.window.load_video_frames('alpha')
        self.assertEqual(self.window.current_video_id, 'beta')
        self.assertTrue(dock.approve_button.isEnabled())
        self.window.toggle_video_sidebar()
        self.assertTrue(self.window.video_dock.isHidden())

    def test_approval_removes_video_and_opens_next(self):
        self.load()
        self.submit()
        self.submit('beta', self.sam, 'sam-work')
        dock = self.open_queue()
        self.assertEqual(dock.queue.count(), 2)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            self.assertEqual(self.window.current_video_id, 'alpha')
            dock.approve()
        self.assertEqual(self.status()[0], 'Accepted')
        self.assertEqual(dock.queue.count(), 1)
        self.assertEqual(self.window.current_video_id, 'beta')
        self.assertTrue(dock.approve_button.isEnabled())
        self.window.resize(1360, 1000)
        self.window.show()
        self.app.processEvents()
        self.window.grab().save('/tmp/bumblebox-submission-review.png')

    def test_cancel_open_leaves_original_view_without_review_ui(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        self.window.load_frame(2)
        before = (self.window.workflow.current, self.window.current_video_id, self.window.current_frame_idx)
        with patch('gui.submission_review.PreparedSubmissionReview.apply', side_effect=InterruptedError('Cancelled')):
            dock = self.open_queue()
        self.assertEqual(before, (self.window.workflow.current, self.window.current_video_id, self.window.current_frame_idx))
        self.assertIsNone(dock.active)
        self.assertFalse(dock.is_focused())
        self.assertTrue(dock.isHidden())
        self.assertFalse(dock.approve_button.isEnabled())
        self.assertFalse(self.window._sync_in_progress)

    def test_closing_review_restores_full_frame_list(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        self.window.load_frame(2)
        dock = self.open_queue()
        self.assertEqual(len(self.window.frames), 2)
        dock.exit_button.click()
        self.assertEqual(len(self.window.frames), 3)
        self.assertEqual(self.window.current_video_id, 'alpha')
        self.assertEqual(self.window.current_frame_idx, 2)
        self.assertFalse(self.window.video_dock.isHidden())
        self.assertTrue(self.window.toggle_video_sidebar_action.isEnabled())
        self.assertIsNone(dock.active)
        self.assertFalse(dock.is_focused())

    def test_profile_change_invalidates_queue(self):
        self.load()
        self.submit()
        dock = self.open_queue()
        self.window.set_contributor(self.sam)
        self.assertIsNone(dock.active)
        self.assertEqual(dock.queue.count(), 0)
        self.assertTrue(dock.isHidden())
        self.assertFalse(self.window.video_dock.isHidden())

    def test_no_local_copy_is_explicit_and_cancelled_project_open_does_not_review_old_project(self):
        self.load()
        inbox = self.window.workflow.home.inbox
        inbox.copies.clear()
        with patch.object(self.window, 'review_submissions') as review, \
                patch.object(QMessageBox, 'information') as information:
            inbox.review_selected()
            review.assert_not_called()
            information.assert_called_once()
        inbox.copies.addItem('Other', str(self.base / 'other'))
        with patch.object(self.window, 'open_recent_project') as open_project, \
                patch.object(self.window, 'review_submissions') as review:
            inbox.review_selected()
            open_project.assert_called_once()
            review.assert_not_called()

    def test_missing_drive_is_not_presented_as_an_empty_queue(self):
        self.load()
        save_setting('project_sync/destination', dict(self.configuration, root=str(self.base / 'missing')))
        dock = self.open_queue()
        self.assertIn('unavailable', dock.status.text())
        self.assertFalse(dock.queue.isEnabled())
        self.assertTrue(dock.isHidden())

    def test_failed_refresh_stays_disabled_when_selection_changes(self):
        self.load()
        self.submit()
        dock = self.open_queue()
        self.assertTrue(dock.queue.isEnabled())
        with patch('gui.submission_review.review_queue', side_effect=OSError('Drive disconnected')):
            self.assertFalse(dock.refresh())
        dock.update_buttons()
        self.assertFalse(dock.queue.isEnabled())
        self.assertFalse(dock.approve_button.isEnabled())
        self.assertTrue(dock.refresh())
        self.assertTrue(dock.queue.isEnabled())

    def test_selecting_another_submission_loads_it_without_open_button(self):
        self.load()
        self.submit()
        self.submit('beta', self.sam, 'sam-work')
        dock = self.open_queue()
        dock.queue.setCurrentRow(1)
        self.assertFalse(dock.approve_button.isEnabled())
        dock.queue.itemClicked.emit(dock.queue.item(1))
        self.assertEqual(self.window.current_video_id, 'beta')
        self.assertEqual(dock.active['video'], 'beta')
        self.assertTrue(dock.approve_button.isEnabled())

    def test_cancel_switch_keeps_current_review_and_selection(self):
        self.load()
        self.submit()
        self.submit('beta', self.sam, 'sam-work')
        dock = self.open_queue()
        dock.queue.setCurrentRow(1)
        with patch('gui.submission_review.PreparedSubmissionReview.apply', side_effect=InterruptedError('Cancelled')):
            dock.queue.itemClicked.emit(dock.queue.item(1))
        self.assertEqual(self.window.current_video_id, 'alpha')
        self.assertEqual(dock.selected(), dock.active)
        self.assertTrue(dock.approve_button.isEnabled())

    def test_empty_queue_does_not_leave_an_unrelated_video_in_review_mode(self):
        self.load()
        self.window.workflow.activate('Home')
        dock = self.open_queue()
        self.assertFalse(dock.is_focused())
        self.assertTrue(dock.isHidden())
        self.assertEqual(self.window.workflow.current, 'Home')

    def test_last_approval_exits_review_and_restores_previous_browsing(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        dock = self.open_queue()
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            dock.approve()
        self.assertEqual(self.status('beta')[0], 'Accepted')
        self.assertFalse(dock.is_focused())
        self.assertTrue(dock.isHidden())
        self.assertEqual(self.window.current_video_id, 'alpha')
        self.assertFalse(self.window.video_dock.isHidden())

    def test_navigation_out_and_back_preserves_focus_without_reopening_project_list(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        dock = self.open_queue()
        self.window.workflow.activate('Home')
        self.window.workflow.activate('Annotate')
        self.assertTrue(self.window.video_dock.isHidden())
        self.assertFalse(dock.isHidden())
        self.assertEqual(self.window.current_video_id, 'beta')

    def test_review_from_home_restores_normal_sidebar_preferences_on_exit(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        self.window.workflow.activate('Home')
        dock = self.open_queue()
        dock.close()
        self.assertEqual(self.window.workflow.current, 'Home')
        self.window.workflow.activate('Annotate')
        self.assertFalse(self.window.video_dock.isHidden())
        self.assertTrue(dock.isHidden())

    def test_review_restores_task_and_frame_filters(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        task = self.window.annotation_work.filters.task
        task.setCurrentIndex(task.findData('hive'))
        self.window.split_filter_combo.setCurrentIndex(1)
        before = (self.window.split_filter_combo.currentIndex(), self.window.split_filter)
        dock = self.open_queue()
        self.assertEqual(self.window.split_filter_combo.currentIndex(), 0)
        dock.close()
        self.assertEqual(task.currentData(), 'hive')
        self.assertEqual(before, (self.window.split_filter_combo.currentIndex(), self.window.split_filter))

    def test_long_names_fit_in_compact_review_layout(self):
        self.load()
        self.submit()
        self.submit('beta', self.sam, 'sam-work')
        dock = self.open_queue()
        dock.queue.item(0).setText('col_18-2021-06-19_16-06-01\nAugust | Hive | 12 frames')
        dock.queue.item(1).setText('bumblebox-07_2026-02-23_06_10_03\nA longer contributor name | Queen brood | 20 frames')
        dock.status.setText('Reviewing: col_18-2021-06-19_16-06-01\nAugust | Hive | 12 frames')
        self.window.resize(1100, 800)
        self.window.show()
        self.app.processEvents()
        for widget in (dock.status, dock.queue, dock.approve_button, dock.exit_button):
            self.assertLessEqual(widget.geometry().right(), dock.widget().width())
            self.assertLessEqual(widget.geometry().bottom(), dock.widget().height())
        self.window.grab().save('/tmp/bumblebox-submission-review-compact.png')

    def test_frame_load_failure_restores_original_video(self):
        self.load()
        self.submit('beta', self.sam, 'sam-work')
        original = self.window.load_video_frames
        def fail_beta(video):
            if video != 'beta':
                original(video)
        with patch.object(self.window, 'load_video_frames', side_effect=fail_beta):
            dock = self.open_queue()
        self.assertEqual(self.window.current_video_id, 'alpha')
        self.assertFalse(dock.is_focused())
        self.assertTrue(dock.isHidden())

    def conflict_plan(self):
        self.submit('beta', self.sam, 'sam-work')
        self.change(self.source, 'beta', 1, self.coordinator)
        from core.annotation import AnnotationManager
        from scripts.test_project_sync import annotation
        manager = AnnotationManager()
        manager.contributor_session = self.coordinator
        hive = annotation(5, 'hive')
        hive['mask'][20:24, 20:24] = 255
        manager.save_video_annotations(self.source, 'beta', [hive])
        row = review_queue(self.source, self.configuration, self.coordinator)[0]
        return PreparedSubmissionReview(self.source, self.configuration, self.coordinator,
                                        row['video'], row['revision'], row['id'])

    def test_conflict_cancel_preserves_local_files_and_previous_view(self):
        self.load()
        plan = self.conflict_plan()
        before = {name: (self.source / name).read_bytes() for name in plan.local}
        with patch.object(SubmissionComparisonDialog, 'exec', return_value=QDialog.DialogCode.Rejected):
            dock = self.open_queue()
        self.assertEqual(self.window.current_video_id, 'alpha')
        self.assertFalse(dock.is_focused())
        self.assertEqual(before, {name: (self.source / name).read_bytes() for name in plan.local})

    def test_conflict_selection_enters_normal_editable_review(self):
        self.load()
        plan = self.conflict_plan()
        target = self.source / 'annotations/json/beta/frame_000001.json'
        before = target.read_bytes()
        def keep_local(dialog):
            dialog.choice = 'local'
            return QDialog.DialogCode.Accepted
        with patch.object(SubmissionComparisonDialog, 'exec', keep_local):
            dock = self.open_queue()
        self.assertTrue(dock.is_focused())
        self.assertEqual(self.window.current_video_id, 'beta')
        self.assertTrue(dock.approve_button.isEnabled())
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(self.window.annotation_work.scan_context()[2], ('beta',))
        dock.close()
        self.assertEqual(set(self.window.annotation_work.scan_context()[2]), {'alpha', 'beta'})

    def test_comparison_requires_choice_and_supports_visibility_and_linked_views(self):
        self.load()
        dialog = SubmissionComparisonDialog(self.conflict_plan(), self.window)
        try:
            dialog.load_preview()
            self.assertTrue(dialog.ready)
            self.assertIsNone(dialog.choice)
            self.assertFalse(dialog.continue_button.isEnabled())
            dialog.options['local'].setChecked(True)
            self.assertEqual(dialog.choice, 'local')
            self.assertTrue(dialog.continue_button.isEnabled())
            dialog.options['submitted'].setChecked(True)
            self.assertFalse(dialog.options['local'].isChecked())
            self.assertEqual(dialog.choice, 'submitted')
            dialog.masks.setChecked(False)
            self.assertEqual(dialog.views['local'].overlay.opacity(), 0)
            dialog.masks.setChecked(True)
            dialog.opacity.setValue(70)
            self.assertEqual(dialog.views['submitted'].overlay.opacity(), .7)
            dialog.views['local'].scale(2, 2)
            dialog.sync_views(dialog.views['local'])
            self.assertEqual(dialog.views['local'].transform(), dialog.views['submitted'].transform())
            dialog.show()
            self.app.processEvents()
            dialog.fit_views()
            dialog.grab().save('/tmp/bumblebox-conflict-comparison.png')
            dialog.accept()
            self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_failed_comparison_cannot_accept_a_stale_selection(self):
        self.load()
        dialog = SubmissionComparisonDialog(self.conflict_plan(), self.window)
        try:
            dialog.load_preview()
            dialog.options['local'].setChecked(True)
            with patch('gui.submission_comparison.comparison_preview', side_effect=OSError('Disconnected')):
                dialog.load_preview()
            self.assertFalse(dialog.continue_button.isEnabled())
            dialog.accept()
            self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
        finally:
            dialog.close()
            dialog.deleteLater()


if __name__ == '__main__':
    unittest.main()
