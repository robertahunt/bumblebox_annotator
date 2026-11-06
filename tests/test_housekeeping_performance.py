"""Saving stays exact while summaries and autosave preparation leave the brush free."""
from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Event, Thread, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.annotation import AnnotationManager
from core.annotation_snapshot import AnnotationSnapshot
from core.annotation_work import summarize, unit_for, _hash
from core.categories import CATEGORIES
from core.contributors import new_session
from gui.annotation_work import AnnotationWorkBrowser
from gui.main_window import MainWindow


def entry():
    return {'frames': list(range(100)), 'categories': {
        c: {'scope': 'frame', 'policy': {'category': c}, 'units': {}}
        for c in CATEGORIES}}


class SummaryTests(unittest.TestCase):
    def test_display_does_not_compute_completion_fingerprints(self):
        data = entry()
        data['categories']['bee']['units']['0'] = {'count': 1, 'status': 'finished', 'signature': 'existing'}
        with patch('core.annotation_work._hash', side_effect=AssertionError('display hashed policy')):
            actual = summarize(data, None)
            self.assertEqual(actual['presence'], 'some')
            self.assertEqual(unit_for(data, 'bee', 0)['signature'], 'existing')
        self.assertEqual(unit_for(data, 'bee', 1)['signature'], _hash([{'category': 'bee'}, []]))

    def test_summary_cache_reuses_unchanged_entry_but_rechecks_new_entry_and_task(self):
        browser = SimpleNamespace(window=SimpleNamespace(project_path='project'), _summaries=OrderedDict())
        data = entry()
        with patch('gui.annotation_work.summarize', wraps=summarize) as compute:
            first = AnnotationWorkBrowser.summary(browser, data, 'bee')
            first['state'] = 'corrupted by caller'
            again = AnnotationWorkBrowser.summary(browser, data, 'bee')
            self.assertEqual(again['state'], 'not_started')
            self.assertEqual(compute.call_count, 1)
            changed = deepcopy(data)
            changed['categories']['bee']['units']['0'] = {'count': 1, 'status': 'changed'}
            self.assertEqual(AnnotationWorkBrowser.summary(browser, changed, 'bee')['state'], 'changed')
            AnnotationWorkBrowser.summary(browser, changed, 'hive')
            AnnotationWorkBrowser.summary(browser, changed, 'bee', [0])
            self.assertEqual(compute.call_count, 4)


class HousekeepingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        with patch.object(MainWindow, 'load_settings'):
            self.window = MainWindow()
        self.addCleanup(self.cleanup_window)
        w = self.window
        w.canvas.load_image(np.zeros((64, 96), np.uint8))
        records = []
        for iid, category in enumerate(('bee', 'bee', 'hive', 'nectar', 'brood_early'), 1):
            mask = np.zeros((64, 96), np.uint8)
            mask[iid: iid + 5, iid: iid + 5] = 255
            records.append(dict(mask_id=iid, category=category, mask=mask, label=category,
                                metadata={'nested': [iid]}))
        records.append(dict(mask_id=10, category='bee', bbox_only=True, bbox=[20, 20, 5, 6]))
        w.canvas.set_annotations(records)
        w.canvas.start_editing_instance(1, 'bee')
        w.project_path = self.root
        w.current_video_id = 'video'
        w.frames = [np.zeros((64, 96), np.uint8)]
        w.frame_video_ids = ['video']
        w.annotation_manager.project_info = {}
        w.contributor_session = new_session('Test annotator')
        w.annotation_manager.contributor_session = dict(w.contributor_session)
        w._mark_current_annotations_dirty()
        # These tests exercise source saving, not project inventory workers.
        self.work_schedule = patch.object(w.annotation_work, 'schedule').start()
        self.video_schedule = patch.object(w.video_browser, 'schedule').start()
        self.addCleanup(patch.stopall)

    def cleanup_window(self):
        w = self.window
        w.save_worker.wait_until_idle(raise_errors=False)
        self.app.processEvents()
        w.project_path = None
        w.current_frame_modified = False
        w.save_worker.failed_tasks.clear()
        with patch('gui.main_window.QSettings'), patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
            w.close()
        self.app.processEvents()

    def assert_annotations_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual(set(a), set(b))
            for key in a:
                if isinstance(a[key], np.ndarray):
                    np.testing.assert_array_equal(a[key], b[key])
                else:
                    self.assertEqual(a[key], b[key], key)

    def test_snapshot_matches_export_and_owns_pixels_and_metadata(self):
        c = self.window.canvas
        before = deepcopy(c.get_annotations())
        snapshot = AnnotationSnapshot.capture(c)
        for _, raster, _ in snapshot.sources:
            self.assertFalse(raster.flags.writeable)
        c.editing_mask[:] = 0
        c.bee_mask[:] = 0
        c.annotation_metadata[1]['metadata']['nested'].append(99)
        self.assert_annotations_equal(snapshot.annotations(), before)

    def test_empty_active_edit_matches_existing_export(self):
        c = self.window.canvas
        c.editing_mask[:] = 0
        self.assert_annotations_equal(AnnotationSnapshot.capture(c).annotations(), c.get_annotations())

    def test_expansion_runs_in_worker_and_old_completion_cannot_clear_new_edits(self):
        w = self.window
        entered, release = Event(), Event()
        self.addCleanup(release.set)
        original = AnnotationSnapshot.annotations
        threads = []
        def gated(snapshot):
            threads.append(get_ident())
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test did not release save')
            return original(snapshot)
        with patch.object(AnnotationSnapshot, 'annotations', gated), \
             patch.object(w.canvas, 'get_annotations', side_effect=AssertionError('UI expanded masks')), \
             patch.object(w.annotation_manager, 'set_frame_annotations', side_effect=AssertionError('UI copied masks')):
            w.workflow.autosave()
            self.assertTrue(entered.wait(3))
            self.assertTrue(w.current_frame_modified)
            w.canvas.editing_mask[40, 40] = 255
            w._mark_current_annotations_dirty()
            w.workflow.autosave()
            self.assertEqual(w.save_worker.save_queue.unfinished_tasks, 1)
            release.set()
            w.save_worker.wait_until_idle()
        self.app.processEvents()
        self.assertNotEqual(threads[0], get_ident())
        self.assertTrue(w.current_frame_modified)
        reader = AnnotationManager()
        saved = reader.load_frame_annotations(self.root, 'video', 0)
        bee = next(a for a in saved if a['mask_id'] == 1)
        self.assertEqual(bee['mask'][40, 40], 0)
        w.workflow.autosave()
        w.save_worker.wait_until_idle()
        self.app.processEvents()
        self.assertFalse(w.current_frame_modified)
        saved = reader.load_frame_annotations(self.root, 'video', 0)
        self.assertEqual(next(a for a in saved if a['mask_id'] == 1)['mask'][40, 40], 255)
        self.assertIn('nectar', {a['category'] for a in saved})
        shared, _ = reader.load_video_annotations(self.root, 'video')
        self.assertEqual({a['category'] for a in shared}, {'hive'})
        self.assertEqual(w.canvas.editing_instance_id, 1)
        self.assertFalse(w.save_worker.pending_snapshots)

    def test_failed_shared_save_stays_dirty_and_retry_preserves_sources(self):
        w = self.window
        with patch.object(w.annotation_manager, 'save_video_annotations', side_effect=OSError('disk full')):
            w.workflow.autosave()
            with self.assertRaisesRegex(OSError, 'disk full'):
                w.save_worker.wait_until_idle()
            self.app.processEvents()
        self.assertTrue(w.current_frame_modified)
        self.assertTrue(w.save_worker.failed_tasks)
        w.workflow.autosave()
        w.save_worker.wait_until_idle()
        self.app.processEvents()
        self.assertFalse(w.current_frame_modified)
        self.assertFalse(w.save_worker.failed_tasks)
        self.assertTrue((self.root / 'annotations/json/video/video_annotations.json').exists())

    def test_worker_uses_captured_scope_and_session(self):
        w = self.window
        w.annotation_manager.project_info = {'hive_annotation_scope': 'frame'}
        old_session = dict(w.annotation_manager.contributor_session)
        original = AnnotationSnapshot.annotations
        entered, release = Event(), Event()
        def gated(snapshot):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test release')
            return original(snapshot)
        with patch.object(AnnotationSnapshot, 'annotations', gated):
            w.workflow.autosave()
            try:
                self.assertTrue(entered.wait(3))
                w.annotation_manager.project_info = {}
                w.annotation_manager.contributor_session = None
            finally:
                release.set()
                w.save_worker.wait_until_idle()
        self.app.processEvents()
        saved = AnnotationManager().load_frame_annotations(self.root, 'video', 0)
        self.assertIn('hive', {a['category'] for a in saved})
        self.assertEqual(saved[0]['provenance']['created_by']['id'], old_session['id'])

    def test_explicit_save_waits_for_older_snapshot_and_newer_pixels_win(self):
        w = self.window
        entered, release = Event(), Event()
        original = AnnotationSnapshot.annotations
        def gated(snapshot):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test release')
            return original(snapshot)
        with patch.object(AnnotationSnapshot, 'annotations', gated):
            w.workflow.autosave()
            try:
                self.assertTrue(entered.wait(3))
                w.canvas.editing_mask[45, 45] = 255
                w._mark_current_annotations_dirty()
                # The explicit save waits for the old snapshot before writing.
                release.set()
                w._save_annotation_sources()
            finally:
                release.set()
                w.save_worker.wait_until_idle(raise_errors=False)
        self.app.processEvents()
        saved = AnnotationManager().load_frame_annotations(self.root, 'video', 0)
        self.assertEqual(next(a for a in saved if a['mask_id'] == 1)['mask'][45, 45], 255)
        self.assertFalse(w.current_frame_modified)

    def test_navigation_waits_for_snapshot_and_failure_keeps_current_frame(self):
        w = self.window
        w.frames.append(np.zeros((64, 96), np.uint8))
        w.frame_video_ids.append('video')
        with patch.object(w.save_worker, 'pending_snapshots', {('pending',)}), \
             patch.object(w.save_worker, 'wait_until_idle', side_effect=OSError('disk full')) as wait, \
             patch.object(QMessageBox, 'warning') as warning:
            w.load_frame(1)
            wait.assert_called_once()
            warning.assert_called_once()
        self.assertEqual(w.current_frame_idx, 0)
        self.assertTrue(w.current_frame_modified)

    def test_delete_bees_waits_for_pending_save_and_preserves_nest_labels(self):
        w = self.window
        w.workflow.autosave()
        with patch.object(w.save_worker, 'wait_until_idle', wraps=w.save_worker.wait_until_idle) as wait, \
             patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
             patch.object(QMessageBox, 'warning') as warning, \
             patch.object(w, '_refresh_video_next_mask_id', return_value=11):
            w.delete_all_instances()
            wait.assert_called_once()
            warning.assert_not_called()
        self.app.processEvents()
        saved = AnnotationManager().load_frame_annotations(self.root, 'video', 0)
        self.assertNotIn('bee', {a['category'] for a in saved})
        self.assertEqual({a['category'] for a in saved}, {'nectar', 'brood_early'})
        shared, _ = AnnotationManager().load_video_annotations(self.root, 'video')
        self.assertEqual({a['category'] for a in shared}, {'hive'})

    def test_stale_save_result_cannot_clean_another_frame_or_project(self):
        w = self.window
        result = dict(project=str(self.root), video='video', frame=0,
                      revision=w._annotation_revision, records={0: [], -1: []})
        w.current_video_id = 'different'
        w._receive_snapshot_saved(result)
        self.assertTrue(w.current_frame_modified)
        w.current_video_id = 'video'
        result['project'] = str(self.root / 'other-project')
        w._receive_snapshot_saved(result)
        self.assertTrue(w.current_frame_modified)

    def test_tag_map_write_cannot_race_shared_annotation_write(self):
        manager = self.window.annotation_manager
        done, started = Event(), Event()
        def tag_write():
            started.set()
            manager.update_aruco_tracking(self.root, 'video', 23, 1)
            done.set()
        with manager._save_lock:
            worker = Thread(target=tag_write)
            worker.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(done.wait(.05))
        worker.join(3)
        self.assertTrue(done.is_set())
        self.assertEqual(manager.load_aruco_tracking(self.root, 'video'), {'23': 1})

    def write_json(self, relative, records):
        path = self.root / 'annotations' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records))
        return path

    def test_incremental_counts_handle_new_deleted_bbox_and_shared_files(self):
        w = self.window
        frame = self.write_json('json/video/frame_000000.json', [{'mask_id': 1, 'category': 'bee'}])
        self.write_json('bbox/video/frame_000000.json', [
            {'mask_id': 1, 'category': 'bee', 'from_mask': True},
            {'mask_id': 9, 'category': 'bee', 'bbox_only': True}])
        self.write_json('json/video/video_annotations.json', {'annotations': [{'mask_id': 3, 'category': 'hive'}]})
        w.done_counter_saved_counts = w._scan_saved_annotation_counts()
        w.done_counter_saved_counts_dirty = False
        self.assertEqual(w.done_counter_saved_counts['bee'], 2)
        with patch.object(w, '_read_annotation_records', wraps=w._read_annotation_records) as read:
            self.assertEqual(w._scan_saved_annotation_counts()['bee'], 2)
            read.assert_not_called()
        frame.write_text(json.dumps([{'mask_id': 2, 'category': 'bee'}, {'mask_id': 4, 'category': 'nectar'}]))
        with patch.object(w, '_scan_saved_annotation_counts', side_effect=AssertionError('full recount')):
            w._refresh_saved_annotation_files(str(self.root), 'video', 0)
            self.assertEqual(w.done_counter_saved_counts['nectar'], 1)
            frame.unlink()
            w._refresh_saved_annotation_files(str(self.root), 'video', 0)
            self.assertEqual(w.done_counter_saved_counts['bee'], 1)
            self.assertEqual(w.done_counter_saved_counts['nectar'], 0)
            self.assertEqual(w.done_counter_saved_counts['hive'], 1)
        self.assertEqual(w.done_counter_saved_counts, w._scan_saved_annotation_counts())

    def test_count_rescan_detects_external_replacement_removal_and_project_change(self):
        w = self.window
        path = self.write_json('json/video/frame_000000.json', [{'mask_id': 1, 'category': 'bee'}])
        self.assertEqual(w._scan_saved_annotation_counts()['bee'], 1)
        replacement = path.with_suffix('.replacement')
        replacement.write_text(json.dumps([{'mask_id': 2, 'category': 'hive'}]))
        replacement.replace(path)
        counts = w._scan_saved_annotation_counts()
        self.assertEqual((counts['bee'], counts['hive']), (0, 1))
        path.unlink()
        self.assertEqual(sum(w._scan_saved_annotation_counts().values()), 0)
        w.project_path = self.root / 'new-project'
        self.assertEqual(sum(w._scan_saved_annotation_counts().values()), 0)
        self.assertEqual(w._annotation_count_cache(), {})

    def test_summary_refresh_is_deferred_during_stroke(self):
        w = self.window
        w.canvas.is_drawing = True
        with patch.object(w.annotation_work.source, 'setText') as update:
            w.annotation_work.apply()
            w.annotation_work.apply()
            update.assert_not_called()
            self.assertTrue(w.annotation_work.apply_timer.isActive())
            w.canvas.is_drawing = False
            w.annotation_work.apply()
            update.assert_called_once()
            self.assertFalse(w.annotation_work.apply_timer.isActive())


if __name__ == '__main__':
    unittest.main()
