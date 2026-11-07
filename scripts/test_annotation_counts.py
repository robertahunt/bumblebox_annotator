"""Background metadata counts must not block annotation or replace newer saves."""

from pathlib import Path
from threading import Event
import time
import unittest
from unittest.mock import patch

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtTest import QTest

from core.annotation_counts import count_records, scan_counts
from core.project_sync import _write_json
from scripts import test_annotation_work as core_fixtures
from scripts import test_assigned_work_gui as gui_fixtures


class AnnotationCountTests(unittest.TestCase):
    setUp = core_fixtures.AnnotationWorkTests.setUp

    def test_counts_reuse_unchanged_metadata_and_remove_deleted_files(self):
        first, cache = scan_counts(self.project)
        self.assertEqual(first['bee'], 3)
        with patch.object(Path, 'open', side_effect=AssertionError('Do not parse unchanged files')):
            second, _ = scan_counts(self.project, cache)
        self.assertEqual(first, second)
        path = self.project / 'annotations/json/video/frame_000000.json'
        path.unlink()
        third, new_cache = scan_counts(self.project, cache)
        self.assertEqual(third['bee'], 2)
        self.assertNotIn((str(path), False), new_cache)
        self.assertIn((str(path), False), cache)  # Worker does not mutate UI-owned state.

    def test_counts_limit_assigned_videos_and_never_decode_masks(self):
        unrelated = self.project / 'annotations/json/unassigned/frame_000000.json'
        unrelated.parent.mkdir(parents=True)
        _write_json(unrelated, {'annotations': 'broken'})
        with patch('cv2.imread', side_effect=AssertionError('Counts need metadata only')):
            counts, _ = scan_counts(self.project, videos=['video'])
        self.assertEqual(counts['bee'], 3)
        with self.assertRaisesRegex(ValueError, 'Invalid annotation metadata'):
            scan_counts(self.project)
        with self.assertRaises(InterruptedError):
            scan_counts(self.project, cancelled=lambda: True)

    def test_bbox_only_counts_do_not_duplicate_mask_instances(self):
        records = [{'mask_id': 1, 'category': 'bee', 'bbox_only': True},
                   {'mask_id': 1, 'category': 'bee', 'bbox_only': True},
                   {'mask_id': 2, 'category': 'bee', 'from_mask': True, 'bbox_only': True},
                   {'mask_id': 3, 'category': 'hive'}]
        self.assertEqual(count_records(records, bbox_only=True)['bee'], 1)
        self.assertEqual(count_records(records, bbox_only=True)['hive'], 0)


class AnnotationCountGuiTests(unittest.TestCase):
    setUpClass = gui_fixtures.AssignedWorkGuiTests.__dict__['setUpClass']
    setUp = gui_fixtures.AssignedWorkGuiTests.setUp
    tearDown = gui_fixtures.AssignedWorkGuiTests.tearDown
    make_window = gui_fixtures.AssignedWorkGuiTests.make_window
    download = gui_fixtures.AssignedWorkGuiTests.download
    view = gui_fixtures.AssignedWorkGuiTests.view

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            QTest.qWait(10)
            self.assertLess(time.monotonic(), deadline)

    def settle(self):
        w = self.window
        self.wait_until(lambda: not w._count_scan_worker and not w.done_counter_saved_counts_dirty
                        and not w.done_counter_update_timer.isActive())

    def test_slow_count_does_not_block_launch_or_video_navigation(self):
        started, release = Event(), Event()
        def slow(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release instance count')
            return scan_counts(*args, **kwargs)
        with patch('core.annotation_counts.scan_counts', slow):
            try:
                window = self.make_window(self.source)
                self.wait_until(started.is_set)
                self.assertIn('Counting', window.done_counter_summary_label.text())
                ticks = []
                QTimer.singleShot(0, lambda: ticks.append(True))
                self.wait_until(lambda: ticks)
                item = next(window.video_list.item(i) for i in range(window.video_list.count())
                            if window.video_list.item(i).data(Qt.ItemDataRole.UserRole) == 'beta')
                window.video_list.itemClicked.emit(item)
                self.assertEqual(window.current_video_id, 'beta')
            finally:
                release.set()
            self.settle()
        expected, _ = scan_counts(self.source)
        self.assertEqual(window.done_counter_saved_counts, expected)

    def test_save_during_count_discards_stale_result_and_recounts(self):
        started, release = Event(), Event()
        calls = []
        def slow(*args, **kwargs):
            result = scan_counts(*args, **kwargs)
            calls.append(True)
            if len(calls) == 1:
                started.set()
                if not release.wait(5):
                    raise TimeoutError('Test did not release instance count')
            return result
        with patch('core.annotation_counts.scan_counts', slow):
            try:
                window = self.make_window(self.source)
                self.wait_until(started.is_set)
                path = self.source / 'annotations/json/alpha/frame_000000.json'
                window._capture_count_baseline('alpha', 0, True, False)
                _write_json(path, {'annotations': [{'mask_id': 1, 'category': 'bee'},
                                                  {'mask_id': 2, 'category': 'bee'}]})
                window._refresh_saved_annotation_files(str(self.source), 'alpha', 0)
            finally:
                release.set()
            self.settle()
        expected, _ = scan_counts(self.source)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(window.done_counter_saved_counts, expected)
        self.assertEqual(expected['bee'] - window.done_counter_session_baseline_counts['bee'], 1)

    def test_switch_project_does_not_adopt_previous_projects_counts(self):
        local = self.download()
        started, release = Event(), Event()
        def slow(project, *args, **kwargs):
            if str(project) == str(self.source):
                started.set()
                if not release.wait(5):
                    raise TimeoutError('Test did not release instance count')
            return scan_counts(project, *args, **kwargs)
        with patch('core.annotation_counts.scan_counts', slow):
            try:
                window = self.make_window(self.source)
                self.wait_until(started.is_set)
                window.load_project(local)
            finally:
                release.set()
            self.settle()
        expected, _ = scan_counts(local, videos=['alpha'])
        self.assertEqual(window.done_counter_saved_counts, expected)
        self.assertEqual(window._count_cache_project, str(local))


if __name__ == '__main__':
    unittest.main()
