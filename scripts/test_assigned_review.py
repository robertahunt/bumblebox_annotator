"""Published-review views, using disposable projects and offscreen Qt windows."""

from threading import Event
import time
import unittest
from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from core.assigned_work import (accept_submission, assigned_review_statuses, publish_work,
                                read_assignments, save_assignments, workspace)
from core.contributors import new_session
from core.project_sync import sync_project
from core.project_update import PreparedUpdate, PublishedProject
from gui.assigned_review import AssignedReviewViews
from scripts import test_assigned_work as backend
from scripts import test_assigned_work_gui as gui


class AssignedReviewTests(unittest.TestCase):
    setUp = backend.AssignedWorkTests.setUp
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    change = backend.AssignedWorkTests.change
    status = backend.AssignedWorkTests.status

    def check(self, local):
        return assigned_review_statuses(local, self.configuration, self.alex, workspace(local)['assignments'])

    def test_published_lifecycle_and_local_edits_remain_separate(self):
        local = self.download()
        self.assertEqual(self.check(local), {'alpha': 'In Progress'})
        self.change(local)
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        self.assertEqual(self.check(local), {'alpha': 'Ready for Review'})
        _, revision = self.status()
        PreparedUpdate(self.source, self.configuration, (self.alex['id'], revision)).apply()
        accept_submission(self.source, self.configuration, self.coordinator, 'alpha', revision)
        self.assertEqual(self.check(local), {'alpha': 'Accepted'})
        path = local / 'annotations/bbox/alpha/frame_000000.json'
        path.unlink()
        self.assertEqual(self.check(local), {'alpha': 'Accepted'})
        publish_work(local, self.configuration, self.alex)
        self.assertEqual(self.check(local), {'alpha': 'In Progress'})
        self.assertIn('alpha', workspace(local)['assignments'])
        self.assertIn('alpha', read_assignments(self.view())['videos'])

    def test_changed_published_video_needs_attention(self):
        local = self.download()
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        _, revision = self.status()
        PreparedUpdate(self.source, self.configuration, (self.alex['id'], revision)).apply()
        self.change(self.source, index=2, actor=self.coordinator)
        sync_project(self.source, self.configuration, self.coordinator)
        self.assertEqual(self.check(local), {'alpha': 'Changed Since Submission'})

    def test_reassignment_is_not_reported_as_accepted_or_ready(self):
        local = self.download()
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        self.choices['alpha']['contributor'] = self.sam
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        self.assertEqual(self.check(local), {'alpha': 'Assignment changed'})

    def test_checks_are_readonly_and_share_one_snapshot_without_mask_reads(self):
        self.choices['beta']['contributor'] = self.alex
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        local = self.download(videos=['alpha', 'beta'])
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        paths = [p for root in (local, self.remote) for p in root.rglob('*') if p.is_file()]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        calls = []
        original = PublishedProject.heads
        def heads(view, **kwargs):
            calls.append(kwargs)
            return original(view, **kwargs)
        with patch.object(PublishedProject, 'heads', heads), \
                patch('core.assigned_work.snapshot_file', side_effect=AssertionError('No payload reads')), \
                patch('core.assigned_work._inventory', side_effect=AssertionError('No file scan')):
            self.assertEqual(self.check(local), {'alpha': 'Ready for Review', 'beta': 'In Progress'})
        self.assertEqual(len(calls), 2)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths})

    def test_offline_wrong_root_wrong_profile_and_cancellation(self):
        local = self.download()
        assignments = workspace(local)['assignments']
        for configuration, actor in (({}, self.alex),
                (dict(self.configuration, root=str(self.base / 'missing')), self.alex),
                (dict(self.configuration, root_id=new_session('Other')['id']), self.alex),
                (self.configuration, self.sam)):
            with self.assertRaises((ValueError, OSError)):
                assigned_review_statuses(local, configuration, actor, assignments)
        with self.assertRaises(InterruptedError):
            assigned_review_statuses(local, self.configuration, self.alex, assignments, lambda: True)

    def test_concurrent_publication_discards_the_mixed_result(self):
        local = self.download()
        original = PublishedProject.heads
        def changed_heads(view, **kwargs):
            heads = original(view, **kwargs)
            return {} if kwargs.get('validate_snapshots') is False else heads
        with patch.object(PublishedProject, 'heads', changed_heads), self.assertRaisesRegex(ValueError, 'changed during'):
            self.check(local)


class AssignedReviewGuiTests(unittest.TestCase):
    setUpClass = gui.AssignedWorkGuiTests.__dict__['setUpClass']
    setUp = gui.AssignedWorkGuiTests.setUp
    tearDown = gui.AssignedWorkGuiTests.tearDown
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    status = backend.AssignedWorkTests.status
    make_window = gui.AssignedWorkGuiTests.make_window

    def settle(self, controls):
        deadline = time.monotonic() + 8
        self.app.processEvents()
        while controls.timer.isActive() or controls.worker or controls.pending:
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)

    def visible(self):
        return [self.window.video_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.window.video_list.count()) if not self.window.video_list.item(i).isHidden()]

    def test_submit_moves_video_to_awaiting_without_changing_canvas_or_assignment(self):
        local = self.download()
        w = self.make_window(local)
        controls = w.annotation_work.review
        self.settle(controls)
        self.assertEqual(controls.view.currentData(), 'active')
        self.assertEqual(self.visible(), ['alpha'])
        transform, video, frame = w.canvas.transform(), w.current_video_id, w.current_frame_idx
        with patch('gui.main_window.QMessageBox.question', return_value=gui.QMessageBox.StandardButton.Yes):
            self.assertTrue(w.publish_assigned_work(ready=True))
        self.settle(controls)
        self.assertEqual(self.visible(), [])
        self.assertIn('No videos in Active Work', controls.note.text())
        self.assertIn('outside the video filters', w.annotation_work.note.text())
        controls.view.setCurrentIndex(controls.view.findData('review'))
        self.assertEqual(self.visible(), ['alpha'])
        self.assertIn('Ready for Review', w.video_list.item(0).text())
        controls.view.setCurrentIndex(controls.view.findData('accepted'))
        self.assertEqual(self.visible(), [])
        controls.view.setCurrentIndex(controls.view.findData('all'))
        self.assertEqual(self.visible(), ['alpha'])
        self.assertEqual((w.current_video_id, w.current_frame_idx), (video, frame))
        self.assertEqual(w.canvas.transform(), transform)
        self.assertIn('alpha', workspace(local)['assignments'])

    def test_acceptance_refresh_and_completion_filter_remain_independent(self):
        local = self.download()
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        _, revision = self.status()
        w = self.make_window(local)
        controls = w.annotation_work.review
        self.settle(controls)
        filters = w.annotation_work.filters
        filters.task.setCurrentIndex(filters.task.findData('hive'))
        filters.show.setCurrentIndex(filters.show.findData('finished'))
        self.assertEqual(self.visible(), [])  # Active video exception cannot bypass review status.
        self.assertEqual(controls.statuses, {'alpha': 'Ready for Review'})
        PreparedUpdate(self.source, self.configuration, (self.alex['id'], revision)).apply()
        accept_submission(self.source, self.configuration, self.coordinator, 'alpha', revision)
        controls.check.click()
        self.settle(controls)
        controls.view.setCurrentIndex(controls.view.findData('accepted'))
        self.assertEqual(self.visible(), ['alpha'])
        self.assertIn('Accepted (1)', controls.view.currentText())
        self.assertIn('Published review: Accepted', w.video_list.item(0).text())
        self.assertEqual(filters.show.currentData(), 'finished')
        w.resize(1500, 1050)
        w.show()
        self.settle(w.annotation_work)
        self.app.processEvents()
        w.video_dock.grab().save('/tmp/bumblebox-assigned-review-sidebar.png')

    def test_mixed_work_has_exact_views_counts_and_preserves_unsaved_edits(self):
        self.choices['beta']['contributor'] = self.alex
        save_assignments(self.source, self.configuration, self.coordinator, self.choices,
                         read_assignments(self.view())['revision'])
        local = self.download(videos=['alpha', 'beta'])
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        w = self.make_window(local)
        controls = w.annotation_work.review
        self.settle(controls)
        self.assertEqual(self.visible(), ['beta'])
        self.assertEqual([controls.view.itemText(i) for i in range(4)],
                         ['Active Work (1)', 'Awaiting Review (1)', 'Accepted (0)', 'All Assigned (2)'])
        w.current_frame_modified = True
        for value, expected in (('review', ['alpha']), ('accepted', []), ('all', ['alpha', 'beta'])):
            controls.view.setCurrentIndex(controls.view.findData(value))
            self.assertEqual(self.visible(), expected)
            self.assertTrue(w.current_frame_modified)
        w.current_frame_modified = False

    def test_offline_status_is_explicit_and_keeps_work_accessible(self):
        local = self.download()
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        controls = self.make_window(local).annotation_work.review
        self.settle(controls)
        self.assertEqual(self.visible(), [])
        with patch('gui.assigned_review.assigned_review_statuses', side_effect=OSError('Drive offline')):
            controls.check.click()
            self.settle(controls)
        self.assertEqual(self.visible(), ['alpha'])
        self.assertIn('unavailable', controls.note.text())
        self.assertEqual(controls.note.toolTip(), 'Drive offline')
        self.assertIn('Not checked', self.window.video_list.item(0).text())
        controls.view.setCurrentIndex(controls.view.findData('review'))
        self.assertEqual(self.visible(), [])
        controls.check.click()
        self.settle(controls)
        self.assertEqual(self.visible(), ['alpha'])
        self.assertIsNone(controls.error)

    def test_full_project_only_my_work_can_be_disabled_without_affecting_other_videos(self):
        w = self.make_window(self.source)
        controls = w.annotation_work.review
        self.assertIsNone(controls.context)
        self.assertTrue(controls.isHidden())
        assignment = read_assignments(self.view())['videos']['alpha']
        w.video_browser.set_my_work(self.source, {'alpha': assignment}, enabled=True)
        w.annotation_work.apply()
        self.settle(controls)
        self.assertFalse(controls.isHidden())
        self.assertEqual(self.visible(), ['alpha'])
        w.video_browser.only_mine.setChecked(False)
        self.assertTrue(controls.isHidden())
        self.assertEqual(self.visible(), ['alpha', 'beta'])

    def test_stale_profile_or_project_results_are_discarded(self):
        controls = AssignedReviewViews()
        started, release = Event(), Event()
        def slow(*args):
            started.set()
            release.wait(5)
            return {'alpha': 'Accepted'}
        assignment = read_assignments(self.view())['videos']['alpha']
        with patch('gui.assigned_review.assigned_review_statuses', side_effect=slow):
            try:
                controls.set_context(self.source, self.alex, {'alpha': assignment})
                controls.timer.stop()
                controls.refresh()
                self.assertTrue(started.wait(2))
                controls.set_context(None, None, {})
                release.set()
                self.settle(controls)
                self.assertIsNone(controls.context)
                self.assertEqual(controls.statuses, {})
            finally:
                release.set()
                controls.stop()
                controls.close()

    def test_view_groups_include_attention_states_and_unknowns_in_active_work(self):
        for status in ('Assigned', 'In Progress', 'Changed Since Submission', 'Assignment changed', None):
            self.assertEqual(AssignedReviewViews.group(status), 'active')
        self.assertEqual(AssignedReviewViews.group('Ready for Review'), 'review')
        self.assertEqual(AssignedReviewViews.group('Accepted'), 'accepted')


if __name__ == '__main__':
    unittest.main()
