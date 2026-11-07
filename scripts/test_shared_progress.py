"""Cross-computer published completion without opening or scanning local work."""

from copy import deepcopy
from threading import Event
import time
import unittest
from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from core import assigned_work
from core.annotation_work import mark_finished
from core.assigned_work import (accept_submission, publish_work, read_assignments, save_assignments,
                                shared_assignment_progress, workspace)
from core.project_update import PreparedUpdate, PublishedProject
from gui.lab_startup import set_working_locally
from scripts import test_assigned_work as backend
from scripts import test_project_home as gui


class SharedProgressTests(unittest.TestCase):
    setUp = backend.AssignedWorkTests.setUp
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    status = backend.AssignedWorkTests.status

    def assign(self, task='bee'):
        self.choices['alpha'] = {'contributor': self.alex, 'frames': [0, 1], 'task': task}
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        return {'alpha': read_assignments(self.view())['videos']['alpha']}

    def check(self, assignments):
        return shared_assignment_progress(self.configuration, self.project_id, self.alex, assignments)['videos']['alpha']

    def test_local_finish_upload_submit_and_accept_are_distinct(self):
        assignments = self.assign()
        self.assertEqual(self.check(assignments)['status'], 'Assigned')
        local = self.download()
        mark_finished(local, 'alpha', 0, 'bee', self.alex)
        result = self.check(assignments)
        self.assertEqual(result['status'], 'In Progress')
        self.assertEqual(result['completion']['finished'], 0)
        publish_work(local, self.configuration, self.alex)
        result = self.check(assignments)
        self.assertEqual(result['status'], 'In Progress')
        self.assertEqual((result['completion']['finished'], result['completion']['total']), (1, 2))
        mark_finished(local, 'alpha', 1, 'bee', self.alex)
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        self.assertEqual(self.check(assignments)['status'], 'Ready for Review')
        _, revision = self.status()
        PreparedUpdate(self.source, self.configuration, (self.alex['id'], revision)).apply()
        accept_submission(self.source, self.configuration, self.coordinator, 'alpha', revision)
        self.assertEqual(self.check(assignments)['status'], 'Accepted')

    def test_shared_video_task_is_one_completion_unit_not_one_finished_frame(self):
        assignments = self.assign('hive')
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        publish_work(local, self.configuration, self.alex)
        summary = self.check(assignments)['completion']
        self.assertTrue(summary['shared'])
        self.assertEqual((summary['finished'], summary['total']), (1, 1))

    def test_readonly_without_local_copy_and_without_mask_payload_scans(self):
        assignments = self.assign()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.remote.rglob('*') if p.is_file()}
        with patch('core.assigned_work.require_contributor', side_effect=AssertionError('No local copy required')), \
                patch('core.assigned_work._inventory', side_effect=AssertionError('No inventory scan')), \
                patch('core.annotation_work.Sources.mask', side_effect=AssertionError('No mask decoding')):
            self.assertEqual(self.check(assignments)['completion']['total'], 2)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})

    def test_old_summary_is_unknown_without_fallback_scanning(self):
        assignments = self.assign()
        original = assigned_work.assignment_review_snapshot
        def old_summary(*args, **kwargs):
            result = deepcopy(original(*args, **kwargs))
            result['snapshot']['manifest'].pop('annotation_work', None)
            return result
        with patch('core.assigned_work.assignment_review_snapshot', side_effect=old_summary), \
                patch('core.annotation_work._unit', side_effect=AssertionError('No fallback scanning')):
            result = self.check(assignments)
        self.assertEqual(result['status'], 'Assigned')
        self.assertEqual(result['completion']['state'], 'not_scanned')
        self.assertNotIn('finished', result['completion'])

    def test_changed_assignment_does_not_inherit_completion(self):
        assignments = self.assign()
        self.assign('hive')
        result = self.check(assignments)
        self.assertEqual(result['status'], 'Assignment changed')
        self.assertIsNone(result['completion'])

    def test_offline_cancel_and_concurrent_update_are_not_reported_as_zero(self):
        assignments = self.assign()
        with self.assertRaises((ValueError, OSError)):
            shared_assignment_progress(dict(self.configuration, root=str(self.base / 'offline')),
                                       self.project_id, self.alex, assignments)
        with self.assertRaises(InterruptedError):
            shared_assignment_progress(self.configuration, self.project_id, self.alex, assignments, lambda: True)
        original = PublishedProject.heads
        checks = []
        def heads(view, **kwargs):
            result = original(view, **kwargs)
            if kwargs.get('validate_snapshots') is False:
                checks.append(True)
                if len(checks) == 2:
                    return {}
            return result
        with patch.object(PublishedProject, 'heads', heads), self.assertRaisesRegex(ValueError, 'changed during'):
            self.check(assignments)


class SharedProgressGuiTests(unittest.TestCase):
    setUpClass = gui.ProjectHomeTests.__dict__['setUpClass']
    setUp = gui.ProjectHomeTests.setUp
    tearDown = gui.ProjectHomeTests.tearDown
    view = gui.ProjectHomeTests.view
    download = gui.ProjectHomeTests.download
    load = gui.ProjectHomeTests.load
    settle = gui.ProjectHomeTests.settle
    shared = gui.ProjectHomeTests.shared

    def wait_progress(self, inbox):
        deadline = time.monotonic() + 8
        while inbox.progress_worker or inbox.progress_requested or inbox.progress_timer.isActive():
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)

    def test_status_and_shared_completion_are_automatic_and_do_not_download(self):
        self.choices['alpha']['task'] = 'hive'
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        self.window.set_contributor(self.alex)
        with patch.object(self.window, 'open_assigned_work') as download:
            inbox = self.settle()
            self.wait_progress(inbox)
            row = inbox.videos.topLevelItem(0)
            self.assertEqual(inbox.videos.headerItem().text(3), 'Shared Status')
            self.assertEqual(row.text(3), 'Ready for Review\nVideo task finished')
            self.assertIn('until uploaded', row.toolTip(3))
            self.assertIn('Shared progress checked', inbox.progress_note.text())
            download.assert_not_called()
        self.window.show()
        self.app.processEvents()
        self.window.workflow.home.grab().save('/tmp/bumblebox-my-work-shared-progress.png')

    def test_stale_result_cannot_fill_the_next_project(self):
        inbox = self.settle()
        self.wait_progress(inbox)
        first, second = self.shared(video='alpha'), self.shared(video='beta')
        release, entered = Event(), Event()
        def read(configuration, project_id, actor, assignments, cancelled):
            if project_id == first['id']:
                entered.set()
                release.wait(3)
            return {'checked_at': '2026-11-06T12:00:00Z', 'videos': {
                v: {'status': 'Accepted' if project_id == first['id'] else 'Assigned', 'completion': None}
                for v in assignments}}
        with patch('gui.assignment_workspace.shared_assignment_progress', side_effect=read):
            inbox.show_projects([first, second])
            deadline = time.monotonic() + 3
            while not entered.is_set():
                QTest.qWait(10)
                self.assertLess(time.monotonic(), deadline)
            inbox.projects.setCurrentRow(1)
            release.set()
            self.wait_progress(inbox)
        row = inbox.videos.topLevelItem(0)
        self.assertEqual(row.text(0), 'beta')
        self.assertEqual(row.text(3), 'Assigned')

    def test_failure_and_offline_are_explicit_not_assumed_unfinished(self):
        self.load(assigned=True)
        inbox = self.settle()
        self.wait_progress(inbox)
        with patch('gui.assignment_workspace.shared_assignment_progress', side_effect=OSError('Drive unavailable')):
            inbox.request_progress(force=True)
            self.wait_progress(inbox)
        self.assertEqual(inbox.videos.topLevelItem(0).text(3), 'Unavailable')
        set_working_locally(True)
        try:
            with patch('gui.assignment_workspace.shared_assignment_progress') as read:
                inbox.refresh()
                self.app.processEvents()
                read.assert_not_called()
            self.assertEqual(inbox.videos.topLevelItem(0).text(3), 'Not checked')
        finally:
            set_working_locally(False)


if __name__ == '__main__':
    unittest.main()
