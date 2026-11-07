"""Automatic completion/publication and advisory device presence in temporary projects."""

from unittest.mock import patch
from uuid import uuid4
import time
import unittest

from core.annotation_work import mark_finished
from core.assigned_work import (accept_submission, publish_work, read_assignments, save_assignments,
                               shared_assignment_progress)
from core.project_sync import SyncConflictError, _inventory, _read_json, _write_json
from core.project_update import PreparedUpdate
from core.work_session import WorkSession, publish_report, read_reports
from scripts import test_assigned_work as fixtures
from scripts import test_shared_progress as progress_tests


class AutomaticWorkTests(unittest.TestCase):
    setUp = fixtures.AssignedWorkTests.setUp
    view = fixtures.AssignedWorkTests.view
    download = fixtures.AssignedWorkTests.download
    change = fixtures.AssignedWorkTests.change
    status = fixtures.AssignedWorkTests.status
    assign = progress_tests.SharedProgressTests.assign

    def device(self, name='Computer A'):
        return dict(self.alex, device={'id': str(uuid4()), 'name': name})

    def test_partial_completion_then_ready_and_unchanged_retry(self):
        self.assign()
        local = self.download()
        mark_finished(local, 'alpha', 0, 'bee', self.alex)
        self.assertEqual(publish_work(local, self.configuration, self.alex, auto_ready=True)['ready'], [])
        self.assertEqual(self.status()[0], 'In Progress')
        mark_finished(local, 'alpha', 1, 'bee', self.alex)
        first = publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertEqual(first['ready'], ['alpha'])
        self.assertEqual(self.status()[0], 'Ready for Review')
        again = publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertTrue(again['unchanged'])
        self.assertEqual(again['path'], first['path'])

    def test_edit_invalidates_completion_and_withdraws_ready(self):
        self.assign('hive')
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        publish_work(local, self.configuration, self.alex, auto_ready=True)
        mark_finished(local, 'alpha', 0, 'hive', self.alex, complete=False)
        self.assertEqual(publish_work(local, self.configuration, self.alex, auto_ready=True)['ready'], [])
        self.assertEqual(self.status()[0], 'In Progress')

    def test_changed_geometry_requires_new_completion(self):
        self.assign()
        local = self.download()
        for frame in (0, 1):
            mark_finished(local, 'alpha', frame, 'bee', self.alex)
        publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.change(local)
        self.assertEqual(publish_work(local, self.configuration, self.alex, auto_ready=True)['ready'], [])

    def test_shared_hive_needs_one_finish_not_bee_completion(self):
        self.assign('hive')
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        self.assertEqual(publish_work(local, self.configuration, self.alex, auto_ready=True)['ready'], ['alpha'])

    def test_unspecified_task_is_not_guessed_from_last_button(self):
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        self.assertEqual(publish_work(local, self.configuration, self.alex, auto_ready=True)['ready'], [])

    def test_unchanged_legacy_manual_submission_survives_adopting_automation(self):
        local = self.download()
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        prior = self.status()
        result = publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertTrue(result['unchanged'])
        self.assertEqual(self.status(), prior)

    def test_pending_import_review_blocks_submission(self):
        self.assign('hive')
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        from core.annotation_work import summarize as real_summary
        def pending(*args, **kwargs):
            return dict(real_summary(*args, **kwargs), state='needs_review')
        with patch('core.annotation_work.summarize', side_effect=pending):
            self.assertEqual(publish_work(local, self.configuration, self.alex, auto_ready=True)['ready'], [])

    def test_same_profile_two_computers_never_overwrite(self):
        self.assign('hive')
        first, second = self.download(), self.download(name='other-computer')
        mark_finished(first, 'alpha', 0, 'hive', self.alex)
        publish_work(first, self.configuration, self.alex, auto_ready=True)
        self.change(second)
        before = _inventory(second, True, with_identity=True)
        with self.assertRaises(SyncConflictError):
            publish_work(second, self.configuration, self.alex, auto_ready=True)
        self.assertEqual(before, _inventory(second, True, with_identity=True))

    def test_review_badge_persists_and_accepted_video_is_not_resubmitted_with_other_edits(self):
        self.choices = {v: {'contributor': self.alex, 'frames': [0], 'task': 'hive'} for v in ('alpha', 'beta')}
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        local = self.download(videos=['alpha', 'beta'])
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        publish_work(local, self.configuration, self.alex, auto_ready=True)
        result = shared_assignment_progress(self.configuration, self.project_id, self.coordinator, {})
        self.assertEqual([r['video'] for r in result['review_ready']], ['alpha'])
        _, revision = self.status()
        PreparedUpdate(self.source, self.configuration, (self.alex['id'], revision)).apply()
        accept_submission(self.source, self.configuration, self.coordinator, 'alpha', revision)
        self.change(local, 'beta', 0)
        publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertEqual(self.status(), ('Accepted', revision))
        self.assertEqual(shared_assignment_progress(self.configuration, self.project_id,
                         self.coordinator, {})['review_ready'], [])

    def test_metadata_only_change_does_not_reuse_a_mismatched_review_receipt(self):
        self.assign('hive')
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        publish_work(local, self.configuration, self.alex, auto_ready=True)
        previous = self.status()[1]
        path = local / 'frames/alpha/video_metadata.json'
        value = _read_json(path)
        value['max_mask_id'] += 1
        _write_json(path, value)
        publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertEqual(self.status()[0], 'Ready for Review')
        self.assertNotEqual(self.status()[1], previous)

    def test_outbox_survives_restart_and_never_enters_annotation_backups(self):
        local = self.download()
        before = _inventory(local, True)
        actor = self.device()
        session = WorkSession(local, actor)
        session.start()
        session.changed('alpha', queue=True, completed=True)
        session.failed('Offline')
        session.end()
        restarted = WorkSession(local, actor)
        self.assertTrue(restarted.data['pending'])
        self.assertEqual(restarted.data['state'], 'closed')
        self.assertEqual(before, _inventory(local, True))
        restarted.published({'ready': ['alpha']})
        self.assertFalse(WorkSession(local, actor).data['pending'])

    def test_heartbeat_bounded_separate_devices_and_not_a_false_online_claim(self):
        local = self.download()
        a = WorkSession(local, self.device())
        b = WorkSession(local, self.device('Computer B'))
        for session in (a, b):
            session.start()
            session.changed('alpha', queue=True)
            publish_report(self.configuration, session.report())
        a.end()
        for _ in range(3):
            publish_report(self.configuration, a.report())
        records = read_reports(self.view(), self.alex['id'])
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r['pending'] for r in records))
        self.assertEqual({r['state'] for r in records}, {'open', 'closed'})
        self.assertTrue(all('reported_at' in r and 'online' not in r for r in records))
        self.assertTrue(all('upload_error' not in r for r in records))
        offline = dict(self.configuration, root=str(self.base / 'absent'))
        with self.assertRaises((ValueError, OSError)):
            publish_report(offline, a.report())
        self.assertEqual(read_reports(self.view(), self.alex['id']), records)
        self.assertTrue(WorkSession(local, a.actor).data['pending'])


class AutomaticWorkGuiTests(unittest.TestCase):
    from scripts import test_project_home as gui_fixtures
    setUpClass = gui_fixtures.ProjectHomeTests.__dict__['setUpClass']
    setUp = gui_fixtures.ProjectHomeTests.setUp
    tearDown = gui_fixtures.ProjectHomeTests.tearDown
    view = gui_fixtures.ProjectHomeTests.view
    download = gui_fixtures.ProjectHomeTests.download
    load = gui_fixtures.ProjectHomeTests.load

    def prepare(self):
        from gui.lab_startup import set_working_locally
        set_working_locally(False)
        self.addCleanup(set_working_locally, False)
        self.alex['device'] = {'id': str(uuid4()), 'name': 'Test computer'}
        self.choices['alpha']['task'] = 'hive'
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        flow = self.load(assigned=True)
        auto = flow.automatic_work
        auto.timer.stop()
        return flow, auto

    def wait_worker(self, auto):
        from PyQt6.QtTest import QTest
        deadline = time.monotonic() + 10
        while auto.worker:
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)

    def test_offline_finish_stays_queued_then_automatic_verified_submission(self):
        from gui.lab_startup import set_working_locally
        flow, auto = self.prepare()
        mark_finished(self.window.project_path, 'alpha', 0, 'hive', self.alex)
        flow.saved()
        auto.saved(completed=True)
        set_working_locally(True)
        auto.retry()
        self.assertIsNone(auto.worker)
        self.assertTrue(auto.pending())
        self.assertIn('waiting for connection', auto.description())
        set_working_locally(False)
        auto.retry()
        self.wait_worker(auto)
        self.assertFalse(auto.pending())
        self.assertFalse(self.window._sync_in_progress)
        self.assertEqual(auto.session.data['ready'], ['alpha'])
        self.assertTrue(read_reports(self.view()))

    def test_heartbeat_runs_during_unsaved_stroke_without_publishing_masks(self):
        flow, auto = self.prepare()
        auto.saved()
        self.window.current_frame_modified = True
        self.window.canvas.is_drawing = True
        with patch('gui.automatic_work.publish_work', side_effect=AssertionError('Do not publish during a stroke')):
            auto.retry()
            self.wait_worker(auto)
        self.window.canvas.is_drawing = False
        self.window.current_frame_modified = False
        self.assertTrue(auto.pending())
        self.assertEqual(len(read_reports(self.view())), 1)
        with patch('gui.automatic_work.publish_report') as write:
            auto.tick()
            write.assert_not_called()

    def test_heartbeat_interval_is_thirty_minutes_and_updates_one_record(self):
        flow, auto = self.prepare()
        with patch('gui.automatic_work.publish_report', wraps=publish_report) as report:
            auto.tick()
            self.wait_worker(auto)
            initial = auto.last_report
            self.assertEqual(report.call_count, 1)
            with patch('gui.automatic_work.monotonic', return_value=initial + 1799):
                auto.tick()
                self.assertIsNone(auto.worker)
            with patch('gui.automatic_work.monotonic', return_value=initial + 1801):
                auto.tick()
                self.wait_worker(auto)
            self.assertEqual(report.call_count, 2)
        self.assertEqual(len(read_reports(self.view())), 1)

    def test_conflict_retains_queue_and_requires_explicit_retry(self):
        flow, auto = self.prepare()
        auto.saved()
        with patch('gui.automatic_work.publish_work', side_effect=SyncConflictError('Other computer changed alpha')):
            auto.retry()
            self.wait_worker(auto)
        self.assertTrue(auto.pending())
        self.assertTrue(auto.blocked)
        self.assertIn('needs attention', auto.description())
        self.assertFalse(self.window._sync_in_progress)
        self.assertIsNone(auto.dialog)

    def test_close_offline_retains_queue_and_end_report_without_network(self):
        from gui.lab_startup import set_working_locally
        flow, auto = self.prepare()
        auto.saved(completed=True)
        set_working_locally(True)
        with patch('gui.automatic_work.publish_work') as upload, patch('gui.automatic_work.publish_report') as report:
            self.assertTrue(auto.finish_session())
            upload.assert_not_called()
            report.assert_not_called()
        journal = WorkSession(auto.session.project, auto.session.actor)
        self.assertTrue(journal.data['pending'])
        self.assertEqual(journal.data['state'], 'closed')
        journal.start()
        self.assertTrue(journal.data['previous_session_ended_at'])

    def test_activity_failure_cannot_erase_verified_upload(self):
        flow, auto = self.prepare()
        mark_finished(self.window.project_path, 'alpha', 0, 'hive', self.alex)
        auto.saved(completed=True)
        with patch('gui.automatic_work.publish_report', side_effect=OSError('activity folder read-only')):
            auto.retry()
            self.wait_worker(auto)
        self.assertFalse(auto.pending())
        self.assertEqual(auto.session.data['ready'], ['alpha'])

    def test_ready_badge_and_device_reports_render_without_claiming_online(self):
        from PyQt6.QtTest import QTest
        from scripts.test_shared_progress import SharedProgressGuiTests
        flow, auto = self.prepare()
        mark_finished(self.window.project_path, 'alpha', 0, 'hive', self.alex)
        auto.saved(completed=True)
        auto.retry()
        self.wait_worker(auto)
        inbox = flow.home.inbox
        inbox.request_progress(force=True)
        SharedProgressGuiTests.wait_progress(self, inbox)
        # Use the director view, which does not require downloading their own assignments.
        inbox.progress_result = shared_assignment_progress(
            self.configuration, self.project_id, self.coordinator, {})
        inbox.update_progress_rows()
        self.assertFalse(flow.review_notification.isHidden())
        self.assertIn('1 ready', flow.review_notification.text())
        self.assertIn('Test computer', inbox.activity_note.text())
        self.assertIn('last heard from', inbox.activity_note.text())
        # Render an actual director session after the contributor's verified upload.
        self.window._discard_annotation_view()
        self.window.project_path = None
        self.window.set_contributor(self.coordinator)
        self.window.load_project(self.source)
        flow.show_home()
        self.window.show()
        self.window.resize(1200, 850)
        deadline = time.monotonic() + 10
        while not (inbox.progress_result or {}).get('review_ready'):
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline,
                (inbox.summary.text(), inbox.progress_error, inbox.progress_context))
        SharedProgressGuiTests.wait_progress(self, inbox)
        self.assertFalse(flow.review_notification.isHidden())
        self.assertFalse(inbox.activity_note.isHidden())
        self.window.grab().save('/tmp/bumblebox-automatic-work.png')


if __name__ == '__main__':
    unittest.main()
