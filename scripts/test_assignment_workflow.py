"""Assignment onboarding, explicit ownership, and bulk UI workflows on disposable data."""

from pathlib import Path
import time
from threading import Event
import unittest
from unittest.mock import Mock, patch

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton

from core.assigned_work import (assignment_management, assignment_presence, establish_director, latest_snapshot,
                               read_assignments, save_assignments)
from core.contributors import new_session
from core.project_sync import _read_json, _write_json, sync_project
from gui.assigned_work_dialog import BulkAssignmentDialog
from gui.project_sync_dialog import export_contributor_profile, save_setting, remember_contributor_profile
from scripts import test_assigned_work as backend
from scripts import test_workspaces as gui


class AssignmentDiscoveryTests(unittest.TestCase):
    setUp = backend.AssignedWorkTests.setUp
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download

    def test_existing_director_and_assignment_discovery_are_read_only(self):
        paths = [p for p in self.remote.rglob('*') if p.is_file()]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        self.assertEqual(assignment_presence(self.configuration, self.coordinator)['status'], 'director')
        mine = assignment_presence(self.configuration, self.alex)
        self.assertEqual((mine['status'], mine['count']), ('assigned', 1))
        self.assertEqual(set(mine['projects'][0]['videos']), {'alpha'})
        self.assertEqual(mine['projects'][0]['videos']['alpha']['frames'], [0])
        other = assignment_presence(self.configuration, new_session('Alex'), self.source)
        self.assertEqual(other['status'], 'empty')
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_offline_wrong_drive_and_corrupt_assignments_are_not_empty(self):
        with self.assertRaises((ValueError, OSError)):
            assignment_presence(dict(self.configuration, root=str(self.base / 'missing')), self.alex)
        with self.assertRaises(ValueError):
            assignment_presence(dict(self.configuration, root_id=self.alex['id']), self.alex)
        _write_json(self.view().home / 'assignments.json', {'format': 99})
        with self.assertRaises(ValueError):
            assignment_presence(self.configuration, self.alex)

    def test_unconfigured_and_unpublished_are_not_empty(self):
        self.assertEqual(assignment_presence({}, self.alex)['status'], 'unconfigured')
        self.assertEqual(assignment_presence(self.configuration, self.alex, self.base / 'new')['status'], 'unpublished')

    def test_explicit_director_setup_and_races(self):
        assignments = self.view().home / 'assignments.json'
        assignments.unlink()
        with self.assertRaisesRegex(ValueError, 'explicitly'):
            save_assignments(self.source, self.configuration, self.coordinator, self.choices, None)
        self.assertFalse(assignments.exists())
        self.assertIn('director', assignment_management(self.source, self.configuration, self.coordinator)['needs'])
        with self.assertRaisesRegex(ValueError, 'Refresh'):
            establish_director(self.source, self.configuration, self.coordinator, 'stale')
        established = establish_director(self.source, self.configuration, self.coordinator, None)
        self.assertEqual(established['videos'], {})
        with self.assertRaisesRegex(ValueError, 'already set'):
            establish_director(self.source, self.configuration, self.sam, None)
        self.assertEqual(read_assignments(self.view()), established)

    def test_assigned_copy_cannot_claim_director_and_non_director_is_restricted(self):
        with self.assertRaisesRegex(ValueError, 'full project'):
            establish_director(self.download(), self.configuration, self.alex, None)
        state = assignment_management(self.source, self.configuration, self.alex)
        self.assertTrue(state['restricted'])
        self.assertEqual(state['director']['id'], self.coordinator['id'])

    def test_policy_prerequisite_uses_published_version_and_blocks_assignments(self):
        path = self.source / 'annotations/project.json'
        data = _read_json(path)
        data.pop('annotation_policy')
        _write_json(path, data)
        sync_project(self.source, self.configuration, self.coordinator)
        state = assignment_management(self.source, self.configuration, self.coordinator)
        self.assertIn('policy', state['needs'])
        with self.assertRaisesRegex(ValueError, 'policy'):
            save_assignments(self.source, self.configuration, self.coordinator, self.choices, state['data']['revision'])

    def test_setup_checks_old_backups_without_reading_masks(self):
        snapshot, heads = latest_snapshot(self.view())
        snapshot['manifest'].pop('annotation_work')
        messages = []
        with patch('core.assigned_work.latest_snapshot', return_value=(snapshot, heads)), \
             patch('core.annotation_work._unit', side_effect=AssertionError('Setup must not scan masks')):
            state = assignment_management(self.source, self.configuration, self.coordinator,
                                          progress=messages.append)
        self.assertFalse(state['needs'])
        self.assertEqual(set(state['available']), {'alpha', 'beta'})
        self.assertTrue(messages)
        with patch('core.annotation_work.published_index', side_effect=AssertionError('Saving assignments must not scan masks')):
            save_assignments(self.source, self.configuration, self.coordinator, self.choices, state['data']['revision'])

    def test_setup_honors_cancellation(self):
        with self.assertRaises(InterruptedError):
            assignment_management(self.source, self.configuration, self.coordinator, cancelled=lambda: True)


class AssignmentOnboardingGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    setUp = gui.WorkspaceGuiTests.setUp
    tearDown = gui.WorkspaceGuiTests.tearDown
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    load = gui.WorkspaceGuiTests.load

    def settle(self):
        inbox = self.window.workflow.home.inbox
        deadline = time.monotonic() + 5
        self.app.processEvents()
        while inbox.worker and time.monotonic() < deadline:
            QTest.qWait(10)
        self.app.processEvents()
        self.assertIsNone(inbox.worker)
        return inbox

    def settle_assignments(self):
        page = self.window.workflow.assignments_page
        deadline = time.monotonic() + 5
        while page.worker and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(page.worker)
        return page

    def test_empty_notice_once_per_session_and_persistent_home_status(self):
        self.window.set_contributor(new_session('New Student'))
        inbox = self.settle()
        self.assertEqual(inbox.result['status'], 'empty')
        self.assertIsNotNone(inbox.notice)
        self.assertFalse(inbox.notice.isModal())
        inbox.notice.close()
        inbox.refresh()
        self.settle()
        self.assertIsNone(inbox.notice)
        self.assertIn('No work assigned', inbox.summary.text())
        self.window.show()
        self.app.processEvents()
        self.window.workflow.home.grab().save('/tmp/bumblebox-assignment-empty-home.png')

    def test_failed_check_does_not_claim_no_work_and_old_result_is_discarded(self):
        inbox = self.settle()
        with patch('gui.assignment_workspace.assignment_presence', side_effect=OSError('Drive offline')):
            inbox.refresh()
            self.settle()
        self.assertIsNone(inbox.result)
        self.assertIsNone(inbox.notice)
        self.assertIn('Unable to check', inbox.summary.text())
        old_context = inbox.context()
        self.window.contributor_session = new_session('Different profile')
        worker = Mock(error=None, result={'status': 'empty', 'count': 0, 'scope': 'old project'})
        inbox.worker = worker
        with patch.object(inbox, 'refresh') as retry:
            inbox.finished(worker, old_context, self.coordinator)
            retry.assert_called_once()
        self.assertIsNone(inbox.notice)

    def test_export_contains_only_portable_identity_and_cancellation_writes_nothing(self):
        destination = self.base / 'export.json'
        with patch('gui.project_sync_dialog.QFileDialog.getSaveFileName', return_value=(str(destination), '')):
            export_contributor_profile(dict(self.alex, device={'hostname': 'secret'}, password='secret'), self.window)
        saved = _read_json(destination)
        self.assertEqual(set(saved), {'id', 'name', 'format', 'kind'})
        self.assertEqual(saved['id'], self.alex['id'])
        before = destination.read_bytes()
        with patch('gui.project_sync_dialog.QFileDialog.getSaveFileName', return_value=('', '')):
            export_contributor_profile(self.sam, self.window)
        self.assertEqual(before, destination.read_bytes())

    def test_director_workspace_bulk_assignment_and_filter_selection(self):
        flow = self.load()
        self.settle()
        flow.activate('Assignments')
        self.settle_assignments()
        page = flow.assignments_page
        panel = page.panel
        self.assertIsNotNone(panel)
        panel.check_visible()
        self.assertEqual(panel.checked_videos(), ['alpha', 'beta'])
        panel.search.setText('alpha')
        self.assertEqual(panel.checked_videos(), ['alpha'])
        panel.search.clear()
        panel.check_visible()
        choices = {v: {'contributor': self.coordinator, 'frames': info['frames']}
                   for v, info in panel.available.items()}
        with patch('gui.assigned_work_dialog.BulkAssignmentDialog') as dialog:
            dialog.return_value.exec.return_value = QDialog.DialogCode.Accepted
            dialog.return_value.choices = choices
            panel.bulk_assign()
        saved = read_assignments(self.view())
        self.assertTrue(all(e['contributor']['id'] == self.coordinator['id'] for e in saved['videos'].values()))
        self.assertTrue(all(e['frames'] == [0, 1, 2] for e in saved['videos'].values()))
        self.window.show()
        self.app.processEvents()
        page.grab().save('/tmp/bumblebox-assignment-workspace.png')

    def test_bulk_review_validates_ranges_and_never_assigns_on_cancel(self):
        dialog = BulkAssignmentDialog(['alpha', 'beta'], {'alpha': {'frames': [0, 1]},
            'beta': {'frames': [0]}}, {self.alex['id']: self.alex})
        dialog.contributor.setCurrentIndex(0)
        dialog.scope.setCurrentIndex(1)
        dialog.frames.setText('0-1')
        with patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec') as review:
            dialog.review()
            review.assert_not_called()
        self.assertIsNone(dialog.choices)
        dialog.scope.setCurrentIndex(0)
        with patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Rejected):
            dialog.review()
        self.assertIsNone(dialog.choices)
        with patch('gui.assigned_work_dialog.TransferConfirmationDialog.exec', return_value=QDialog.DialogCode.Accepted):
            dialog.review()
        self.assertEqual(dialog.choices['alpha']['frames'], [0, 1])
        self.assertEqual(dialog.choices['beta']['frames'], [0])
        dialog.close()

    def test_setup_explicitly_claims_director_only_after_confirmation(self):
        (self.view().home / 'assignments.json').unlink()
        flow = self.load()
        self.settle()
        flow.activate('Assignments')
        self.settle_assignments()
        page = flow.assignments_page
        self.assertIsNone(page.panel)
        self.assertTrue(any(b.text() == 'Set Project Director...' for b in page.findChildren(QPushButton)))
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Cancel):
            page.set_director()
        self.assertIsNone(read_assignments(self.view())['coordinator'])
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            page.set_director()
        self.settle_assignments()
        self.assertEqual(read_assignments(self.view())['coordinator']['id'], self.coordinator['id'])
        self.assertIsNotNone(page.panel)

    def test_contributor_cannot_enter_director_workspace(self):
        flow = self.load()
        self.settle()
        self.window.set_contributor(self.alex)
        self.settle()
        self.assertFalse(flow.tabs.isTabVisible(flow.names.index('Assignments')))
        flow.activate('Assignments')
        self.assertNotEqual(flow.current, 'Assignments')

    def test_setup_without_connection_has_action_and_no_empty_notice(self):
        save_setting('project_sync/destination', {})
        flow = self.load()
        inbox = self.settle()
        self.assertEqual(inbox.result['status'], 'unconfigured')
        self.assertIsNone(inbox.notice)
        flow.activate('Assignments')
        self.settle_assignments()
        self.assertTrue(any(b.text() == 'Connect Shared Folder...' for b in flow.assignments_page.findChildren(QPushButton)))

    def test_unsaved_table_choices_block_switch_close_and_reload(self):
        flow = self.load()
        self.settle()
        flow.activate('Assignments')
        self.settle_assignments()
        page = flow.assignments_page
        frames = page.panel.controls['alpha'][1]
        before = frames.text()
        frames.setText('0-1')
        try:
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
                self.assertFalse(flow.prepare_close())
                with self.assertRaisesRegex(ValueError, 'Contributor unchanged'):
                    self.window.set_contributor(self.alex)
                self.assertEqual(self.window.contributor_session['id'], self.coordinator['id'])
                with patch('gui.main_window.working_project_path') as load:
                    self.window.load_project(self.source)
                    load.assert_not_called()
                panel = page.panel
                page.refresh()
                self.assertIs(page.panel, panel)
        finally:
            frames.setText(before)

    def test_discovery_requires_enrollment_and_preserves_unsaved_assignment_choices(self):
        flow = self.load()
        self.settle()
        flow.activate('Assignments')
        self.settle_assignments()
        panel = flow.assignments_page.panel
        actor, frames = panel.controls['alpha']
        original, identity = frames.text(), actor.currentData()
        frames.setText('0-1')
        newcomer = new_session('New contributor')
        try:
            remember_contributor_profile(newcomer)
            from gui.shared_profiles import profile_sync_service
            profile_sync_service().changed.emit()
            self.assertEqual(actor.findData(newcomer['id']), -1)
            panel.enroll(newcomer)
            self.assertGreaterEqual(actor.findData(newcomer['id']), 0)
            self.assertEqual(actor.currentData(), identity)
            self.assertEqual(frames.text(), '0-1')
            self.assertTrue(panel.save_table.isEnabled())
            self.assertEqual(self.window.contributor_session['id'], self.coordinator['id'])
        finally:
            frames.setText(original)

    def test_slow_setup_is_nonmodal_and_cancelled_result_cannot_open_table(self):
        flow = self.load()
        self.settle()
        release = Event()
        started = Event()
        def slow_check(*args, progress, cancelled):
            started.set()
            progress('Reading the published project version...')
            release.wait(5)
            return {'needs': ['director'], 'director': None}
        with patch('gui.assignment_workspace.assignment_management', side_effect=slow_check), \
             patch('gui.assignment_workspace.SyncProgressDialog.run', side_effect=AssertionError('Startup must not open a modal')):
            try:
                flow.activate('Assignments')
                self.assertTrue(started.wait(2))
                QTest.qWait(30)
                page = flow.assignments_page
                self.assertIsNotNone(page.worker)
                self.assertIn('published project', page.status.text())
                self.assertIsNone(QApplication.activeModalWidget())
                page.cancel()
                flow.activate('Annotate')
                self.assertEqual(flow.current, 'Annotate')
            finally:
                release.set()
                self.settle_assignments()
        self.assertIsNone(page.panel)
        self.assertIn('cancelled', page.status.text())

    def test_project_change_discards_outdated_setup_result(self):
        flow = self.load()
        self.settle()
        release = Event()
        def slow_check(*args, **kwargs):
            release.wait(5)
            return {'needs': ['director'], 'director': None}
        with patch('gui.assignment_workspace.assignment_management', side_effect=slow_check):
            try:
                flow.activate('Assignments')
                page = flow.assignments_page
                page.clear()
                flow.activate('Annotate')
            finally:
                release.set()
                self.settle_assignments()
        self.assertIsNone(page.context_key)
        self.assertIsNone(page.panel)
        self.assertFalse(any(b.text() == 'Set Project Director...' for b in page.findChildren(QPushButton)))


if __name__ == '__main__':
    unittest.main()
