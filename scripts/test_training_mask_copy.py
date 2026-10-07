"""Training-only copy routing, exact masks, collision protection, and persistence."""

from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QListWidget, QListWidgetItem, QWidget

from core.annotation import AnnotationManager
from core.annotation_copy import (
    annotation_key, merge_copied_masks, selected_copy_masks, training_copy_targets,
)
from core.project_manager import ProjectManager
from gui.canvas import ImageCanvas
from gui.dialogs import TrainingMaskCopyDialog
from gui.main_window import MainWindow, SaveWorker


def nest_mask():
    mask = np.zeros((32, 32), np.uint8)
    mask[2:20, 2:20] = 255
    mask[6:12, 6:12] = 0
    mask[25:30, 25:30] = 255
    return mask


def annotation(mask=None, category='hive', mask_id=22):
    return dict(mask=nest_mask() if mask is None else mask,
                category=category, mask_id=mask_id)


class CopyWindow(QWidget):
    """Exercise real copy methods without starting models or the full GUI."""

    _get_training_copy_targets = MainWindow._get_training_copy_targets
    _get_frame_idx_in_video = MainWindow._get_frame_idx_in_video
    _get_selected_instance_keys = MainWindow._get_selected_instance_keys
    _split_source_annotations = MainWindow._split_source_annotations
    _training_copy_existing_annotations = MainWindow._training_copy_existing_annotations
    _copy_masks_to_training_frame = MainWindow._copy_masks_to_training_frame
    _copy_selected_training_masks = MainWindow._copy_selected_training_masks


class TrainingMaskCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        ProjectManager().create_project(self.root, 'copy test')
        self.manager = AnnotationManager(max_cache_size=2)
        self.manager.load_project(self.root)
        self.window = CopyWindow()
        window = self.window
        window.annotation_manager = self.manager
        window.project_path = self.root
        window.current_video_id = 'video'
        window.current_frame_idx = 0
        window.frames = []
        window.frame_video_ids = ['video'] * 6 + ['other']
        window.frame_splits = ['train', 'val', 'train', 'train', 'train', 'train', 'train']
        window.frame_selected = [True, True, False, True, True, True, True]
        window.frame_list_to_frames_map = [0, 1, 2, 6]
        for index, number in enumerate((0, 10, 20, 30, 40, 50, 60)):
            frame = self.root / window.frame_video_ids[index] / f'frame_{number:06d}.jpg'
            frame.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(frame), np.zeros((32, 32), np.uint8))
            window.frames.append(frame)
        window.dirty_frame_annotation_keys = set()
        window.coco_export_dirty = False
        window._schedule_done_counter_update = Mock()
        window._commit_canvas_edit_if_needed = Mock()
        window._save_annotation_sources = Mock()
        window.save_worker = Mock()
        window.canvas = Mock()
        window.canvas.get_annotations.return_value = [annotation(mask_id=4, category='bee'), annotation()]
        window.instance_list = QListWidget()
        item = QListWidgetItem('Hive 22')
        item.setData(Qt.ItemDataRole.UserRole, {'id': 22, 'type': 'hive'})
        window.instance_list.addItem(item)
        window.instance_list.setCurrentItem(item)
        window.status_label = Mock()
        window.toolbar = Mock()
        window.update_instance_list_from_canvas = Mock()
        window.load_frame = Mock(side_effect=lambda index: setattr(window, 'current_frame_idx', index))

    def tearDown(self):
        self.window.close()
        self.temp.cleanup()

    def sources(self):
        return selected_copy_masks([annotation()], [(22, 'hive')], self.manager.project_info)

    def run_copy(self, *, next_only=False, replace=False, canceled=None):
        with ExitStack() as stack:
            dialog = stack.enter_context(patch('gui.main_window.TrainingMaskCopyDialog')).return_value
            dialog.exec.return_value = QDialog.DialogCode.Accepted
            dialog.endpoint.currentIndex.return_value = 0 if next_only else 2
            dialog.replace_check.isChecked.return_value = replace
            progress = stack.enter_context(patch('gui.main_window.QProgressDialog')).return_value
            progress.wasCanceled.side_effect = canceled
            progress.wasCanceled.return_value = False
            message = stack.enter_context(patch('gui.main_window.QMessageBox'))
            self.window._copy_selected_training_masks(next_only=next_only)
            return message, dialog, progress

    def test_strict_targets_ignore_display_filter(self):
        self.assertEqual(self.window._get_training_copy_targets(), [3, 4, 5])
        self.window.current_frame_idx = 5
        self.assertEqual(self.window._get_training_copy_targets(), [])
        self.window.current_frame_idx = 1
        self.assertEqual(self.window._get_training_copy_targets(), [])

    def test_incomplete_metadata_fails_closed(self):
        self.assertEqual(training_copy_targets(0, 'a', ['a'] * 5, ['train'] * 2, [True]), [])
        self.assertEqual(training_copy_targets(0, None, ['a'], ['train'], [True]), [])
        self.assertEqual(training_copy_targets(-1, 'a', ['a'], ['train'], [True]), [])
        self.assertEqual(training_copy_targets(0, 'b', ['a'], ['train'], [True]), [])

    def test_source_selection_is_category_and_id_not_row(self):
        keys = self.window._get_selected_instance_keys()
        sources = selected_copy_masks(self.window.canvas.get_annotations(), keys, self.manager.project_info)
        self.assertEqual([annotation_key(ann) for ann in sources], [(22, 'hive')])

    def test_canvas_snapshot_includes_active_edits_and_category_overlap(self):
        canvas = ImageCanvas()
        try:
            canvas.load_image(np.zeros((32, 32), np.uint8))
            canvas.set_annotations([annotation(), annotation(category='bee', mask_id=4)])
            canvas.start_editing_instance(22, category='hive')
            canvas.editing_mask[31, 31] = 255
            sources = selected_copy_masks(canvas.get_annotations(), [(22, 'hive')], self.manager.project_info)
            expected = nest_mask()
            expected[31, 31] = 255
            np.testing.assert_array_equal(sources[0]['mask'], expected)
            self.assertEqual(sources[0]['bbox'], [2, 2, 30, 30])
        finally:
            canvas.close()

    def test_empty_and_shared_sources_are_rejected(self):
        for category in ('hive', 'pollen', 'chamber'):
            with self.subTest(category=category), self.assertRaisesRegex(ValueError, 'shared'):
                selected_copy_masks([annotation(category=category)], [(22, category)], {})
        with self.assertRaisesRegex(ValueError, 'no segmentation'):
            selected_copy_masks([annotation(np.zeros((32, 32), np.uint8))], [(22, 'hive')], self.manager.project_info)
        with self.assertRaisesRegex(ValueError, 'Select'):
            selected_copy_masks([], [], self.manager.project_info)

    def test_queen_brood_can_copy_as_frame_specific_mask_in_legacy_projects(self):
        source = annotation(category='queen_brood_middle')
        sources = selected_copy_masks([source], [(22, 'queen_brood_middle')], {})
        merged, copied, skipped = merge_copied_masks([], sources, (32, 32))
        self.assertEqual(copied, [(22, 'queen_brood_middle')])
        self.assertFalse(skipped)
        np.testing.assert_array_equal(merged[0]['mask'], source['mask'])

    def test_matching_mask_skipped_or_explicitly_replaced(self):
        original = annotation(np.fliplr(nest_mask()).copy())
        bee = annotation(category='bee', mask_id=4)
        sources = self.sources()
        merged, copied, skipped = merge_copied_masks([original, bee], sources, (32, 32))
        self.assertFalse(copied)
        self.assertEqual(skipped, [(22, 'hive')])
        np.testing.assert_array_equal(merged[0]['mask'], original['mask'])
        merged, copied, skipped = merge_copied_masks([original, bee], sources, (32, 32), replace=True)
        self.assertEqual(copied, [(22, 'hive')])
        self.assertEqual(annotation_key(merged[0]), (4, 'bee'))
        np.testing.assert_array_equal(merged[1]['mask'], nest_mask())
        merged[1]['mask'][2, 2] = 0
        self.assertEqual(sources[0]['mask'][2, 2], 255)
        np.testing.assert_array_equal(original['mask'], np.fliplr(nest_mask()))

    def test_rejects_id_conflicts_overlap_and_dimension_changes(self):
        for existing, shape, shared in (
            ([annotation(category='bee')], (32, 32), []),
            ([], (32, 32), [annotation(category='pollen')]),
            ([annotation(mask_id=23)], (32, 32), []),
            ([], (64, 64), []),
        ):
            with self.subTest(shape=shape, shared=bool(shared)), self.assertRaises(ValueError):
                merge_copied_masks(existing, self.sources(), shape, shared=shared, replace=True)

    def test_direct_write_cannot_bypass_training_target_guard(self):
        for index in (0, 1, 2, 6):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'selected training'):
                self.window._copy_masks_to_training_frame(index, self.sources(), False, [])
        self.assertFalse(self.window.coco_export_dirty)

    def test_batch_saves_only_selected_train_frames_past_cache_eviction(self):
        bee = annotation(category='bee', mask_id=4)
        for index in range(7):
            self.manager.save_frame_annotations(self.root, self.window.frame_video_ids[index], index * 10, [bee])
        self.manager.save_video_annotations(self.root, 'video', [annotation(category='pollen', mask_id=99)])
        protected = [path for path in (self.root / 'annotations').rglob('*')
                     if path.is_file() and ('frame_000010' in path.name or 'frame_000020' in path.name
                                          or 'frame_000060' in path.name or 'video_annotations' in path.name)]
        before = {path: path.read_bytes() for path in protected}
        self.run_copy()
        self.window.save_worker.wait_until_idle.assert_called_once()
        self.window.load_frame.assert_not_called()
        self.assertTrue(self.window.coco_export_dirty)
        for index in (3, 4, 5):
            loaded = self.manager.load_frame_annotations(self.root, 'video', index * 10)
            self.assertEqual({annotation_key(ann) for ann in loaded}, {(4, 'bee'), (22, 'hive')})
            np.testing.assert_array_equal(next(ann['mask'] for ann in loaded if ann['category'] == 'hive'), nest_mask())
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(self.window.current_frame_idx, 0)

    def test_next_skips_validation_and_unselected_frames_and_selects_brush(self):
        self.run_copy(next_only=True)
        self.window.load_frame.assert_called_once_with(3)
        self.window.toolbar.set_tool.assert_called_once_with('brush')
        self.window.canvas.set_selected_instance.assert_called_once_with(22, category='hive', zoom=False)
        loaded = self.manager.load_frame_annotations(self.root, 'video', 30)
        self.assertEqual([annotation_key(ann) for ann in loaded], [(22, 'hive')])
        self.assertFalse(self.manager.load_frame_annotations(self.root, 'video', 10))

    def test_cancellation_keeps_completed_copies_only(self):
        self.run_copy(canceled=[False, True])
        self.assertTrue(self.manager.load_frame_annotations(self.root, 'video', 30))
        self.assertFalse(self.manager.load_frame_annotations(self.root, 'video', 40))
        self.window.load_frame.assert_not_called()
        self.assertIn('Canceled', self.window.status_label.setText.call_args.args[0])

    def test_cache_empty_masks_are_not_resurrected(self):
        self.manager.save_frame_annotations(self.root, 'video', 30, [annotation(np.fliplr(nest_mask()).copy())])
        self.manager.set_frame_annotations(3, [], video_id='video')
        copied, _ = self.window._copy_masks_to_training_frame(3, self.sources(), False, [])
        self.assertEqual(copied, [(22, 'hive')])

    def test_batch_default_skip_preserves_matching_file_bytes(self):
        self.manager.save_frame_annotations(self.root, 'video', 30, [annotation(np.fliplr(nest_mask()).copy())])
        protected = list((self.root / 'annotations').rglob('frame_000030.*'))
        before = {path: path.read_bytes() for path in protected}
        self.run_copy()
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertIn('Skipped 1', self.window.status_label.setText.call_args.args[0])

    def test_partial_write_failure_stops_without_caching_failed_frame(self):
        original = self.manager.save_frame_annotations

        def save(project, video, number, anns):
            if number == 40:
                raise OSError('disk full')
            return original(project, video, number, anns)

        with patch.object(self.manager, 'save_frame_annotations', side_effect=save):
            message, _, _ = self.run_copy()
        self.assertTrue(self.manager.load_frame_annotations(self.root, 'video', 30))
        self.assertNotIn(('video', 4), self.manager.frame_annotations)
        self.assertFalse(self.manager.load_frame_annotations(self.root, 'video', 50))
        self.assertIn('disk full', message.return_value.setDetailedText.call_args.args[0])

    def test_dialog_only_lists_eligible_endpoints_and_defaults_to_no_overwrite(self):
        dialog = TrainingMaskCopyDialog(self.window, [(3, 30), (4, 40), (5, 50)], 1)
        try:
            self.assertFalse(dialog.replace_check.isChecked())
            self.assertEqual(dialog.endpoint.currentData(), 3)
            self.assertEqual(dialog.endpoint.itemText(0), 'Frame 000030')
            dialog.endpoint.setCurrentIndex(2)
            self.assertIn('3 training frame', dialog.count_label.text())
        finally:
            dialog.close()


class SaveQueueCopyTests(unittest.TestCase):
    def test_wait_includes_inflight_saves_even_when_queue_is_empty(self):
        entered, release, drained = threading.Event(), threading.Event(), threading.Event()
        manager = Mock(contributor_session=None)

        def save(*args, **kwargs):
            entered.set()
            release.wait(5)

        manager.save_frame_annotations.side_effect = save
        worker = SaveWorker(manager)
        worker.start()
        thread = None
        try:
            worker.add_save_task('/tmp', 'video', 0, [])
            self.assertTrue(entered.wait(5))
            self.assertTrue(worker.save_queue.empty())
            thread = threading.Thread(target=lambda: (worker.wait_until_idle(), drained.set()))
            thread.start()
            self.assertFalse(drained.wait(0.05))
            release.set()
            self.assertTrue(drained.wait(5))
        finally:
            release.set()
            if thread:
                thread.join(5)
            worker.stop()
            worker.wait(5000)

    def test_failed_save_releases_queue_and_stopped_worker_is_rejected(self):
        manager = Mock(contributor_session=None)
        manager.save_frame_annotations.side_effect = OSError('test failure')
        worker = SaveWorker(manager)
        worker.start()
        try:
            worker.add_save_task('/tmp', 'video', 0, [])
            with self.assertRaisesRegex(OSError, 'test failure'):
                worker.wait_until_idle()
            self.assertEqual(worker.save_queue.unfinished_tasks, 0)
        finally:
            worker.stop()
            worker.wait(5000)
        worker.add_save_task('/tmp', 'video', 1, [])
        with self.assertRaises(RuntimeError):
            worker.wait_until_idle()


if __name__ == '__main__':
    unittest.main()
