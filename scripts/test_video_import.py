#!/usr/bin/env python3
"""Synthetic video import and annotation-to-training checks, without model loading."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QLabel, QMessageBox

from core.annotation import AnnotationManager
from core.project_manager import ProjectManager
from gui.dialogs import ProjectVideoImportDialog
from training.coco_video_export import export_coco_per_video
from training.yolo_trainer import YOLOTrainingWorker


class VideoImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        from gui.main_window import MainWindow
        cls.MainWindow = MainWindow
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, cls.settings_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.settings_dir.cleanup()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.project = self.root / 'project'
        self.manager = ProjectManager()
        self.manager.create_project(self.project, 'Synthetic import')
        self.source = self.root / 'sample.avi'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 32))
        self.assertTrue(writer.isOpened(), 'MJPEG writer is required for this synthetic video test')
        for index in range(12):
            writer.write(np.full((32, 32, 3), index * 15, dtype=np.uint8))
        writer.release()
        self.window = SimpleNamespace(
            project_path=self.project, project_manager=self.manager,
            status_label=QLabel(), update_video_list=Mock(), load_video_frames=Mock(),
        )
        warning_patch = patch.object(QMessageBox, 'warning')
        critical_patch = patch.object(QMessageBox, 'critical')
        self.warning = warning_patch.start()
        self.critical = critical_patch.start()
        self.addCleanup(warning_patch.stop)
        self.addCleanup(critical_patch.stop)

    def import_video(self, **settings):
        self.MainWindow.add_video_to_project(self.window, self.source, **settings)
        self.critical.assert_not_called()
        frames_dir = self.manager.get_frames_dir('sample')
        metadata = json.loads((frames_dir / 'video_metadata.json').read_text())
        indices = [int(path.stem.split('_')[1]) for path in sorted(frames_dir.glob('frame_*.jpg'))]
        return metadata, indices

    def test_dialog_defaults_to_all_frames_and_sampling_is_optional(self):
        dialog = ProjectVideoImportDialog()
        self.assertEqual(dialog.get_settings(), {'split': 'train', 'n_selected': 15, 'extract_all': True})
        self.assertEqual(dialog.frames_label.text(), 'Frames to select:')
        dialog.extract_all_check.setChecked(False)
        dialog.frames_spin.setValue(3)
        dialog.split_combo.setCurrentText('val')
        self.assertEqual(dialog.get_settings(), {'split': 'val', 'n_selected': 3, 'extract_all': False})
        self.assertEqual(dialog.frames_label.text(), 'Frames to extract:')

    def test_sampling_warning_only_shows_when_all_frames_is_unchecked(self):
        dialog = ProjectVideoImportDialog()
        dialog.show()
        try:
            self.app.processEvents()
            self.assertFalse(dialog.sampled_frames_warning.isVisible())
            dialog.extract_all_check.setChecked(False)
            self.app.processEvents()
            self.assertTrue(dialog.sampled_frames_warning.isVisible())
            for tool in ('GUI tracking', 'SAM2/YOLO propagation', 'tracking validation'):
                self.assertIn(tool, dialog.sampled_frames_warning.text())
            self.assertFalse(dialog.get_settings()['extract_all'])
            dialog.extract_all_check.setChecked(True)
            self.app.processEvents()
            self.assertFalse(dialog.sampled_frames_warning.isVisible())
        finally:
            dialog.close()

    def test_import_menu_passes_extraction_settings_to_importer(self):
        self.window.add_video_to_project = Mock()
        with patch('gui.main_window.QFileDialog.getOpenFileName', return_value=(str(self.source), '')), \
                patch('gui.main_window.ProjectVideoImportDialog') as dialog_type:
            dialog = dialog_type.return_value
            dialog.exec.return_value = 1
            dialog.get_settings.return_value = {'split': 'val', 'n_selected': 3, 'extract_all': False}
            self.MainWindow.show_add_video_dialog(self.window)
        self.window.add_video_to_project.assert_called_once_with(
            str(self.source), split='val', n_selected=3, extract_all=False
        )

    def test_cancelled_import_does_not_copy_video(self):
        self.window.add_video_to_project = Mock()
        with patch('gui.main_window.QFileDialog.getOpenFileName', return_value=(str(self.source), '')), \
                patch('gui.main_window.ProjectVideoImportDialog') as dialog_type:
            dialog_type.return_value.exec.return_value = 0
            self.MainWindow.show_add_video_dialog(self.window)
        self.window.add_video_to_project.assert_not_called()
        self.assertIsNone(self.manager.get_video_path('sample'))

    def test_default_extracts_whole_video_but_selects_subset(self):
        metadata, indices = self.import_video(n_selected=3)
        self.assertEqual(indices, list(range(12)))
        self.assertEqual(metadata['selected_frames'], [0, 4, 8])
        self.assertEqual(metadata['extraction_mode'], 'all')
        self.assertEqual(metadata['extracted_frame_count'], 12)

    def test_sampled_import_keeps_source_indices_and_original_video(self):
        original_bytes = self.source.read_bytes()
        metadata, indices = self.import_video(n_selected=3, extract_all=False)
        self.assertEqual(indices, [0, 4, 8])
        self.assertEqual(metadata['selected_frames'], indices)
        self.assertEqual(metadata['total_frames'], 12)
        self.assertEqual(metadata['n_selected'], 3)
        self.assertEqual(metadata['extraction_mode'], 'selected')
        self.assertEqual(metadata['extracted_frame_count'], 3)
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(self.manager.get_video_path('sample').read_bytes(), original_bytes)
        for index in indices:
            image = cv2.imread(str(self.manager.get_frame_path('sample', index)))
            self.assertAlmostEqual(float(image.mean()), index * 15, delta=2)

    def test_short_video_clamps_requested_count(self):
        metadata, indices = self.import_video(n_selected=50, extract_all=False)
        self.assertEqual(indices, list(range(12)))
        self.assertEqual(metadata['n_selected'], 12)
        self.assertIn('12 marked for train', self.window.status_label.text())

    def test_one_frame_and_empty_video_selection(self):
        self.assertEqual(self.manager.select_frames_uniform(12, 1), [0])
        self.assertEqual(self.manager.select_frames_uniform(0, 15), [])
        with self.assertRaises(ValueError):
            self.manager.select_frames_uniform(12, 0)

    def test_inference_import_creates_missing_split_folder(self):
        metadata, indices = self.import_video(split='inference', n_selected=3, extract_all=False)
        self.assertEqual(indices, [0, 4, 8])
        self.assertEqual(metadata['split'], 'inference')
        self.assertTrue((self.project / 'input_data/inference/sample.avi').exists())

    def test_failed_image_write_is_not_marked_selected(self):
        original_imwrite = cv2.imwrite

        def save_except_middle(path, image):
            if Path(path).stem == 'frame_000004':
                return False
            return original_imwrite(path, image)

        with patch('core.project_manager.cv2.imwrite', side_effect=save_except_middle):
            metadata, indices = self.import_video(n_selected=3, extract_all=False)
        self.assertEqual(indices, [0, 8])
        self.assertEqual(metadata['selected_frames'], indices)
        self.assertEqual(metadata['n_selected'], 2)
        self.warning.assert_called_once()

    def test_no_successful_extraction_does_not_open_empty_frames(self):
        with patch('core.project_manager.cv2.imwrite', return_value=False):
            metadata, indices = self.import_video(n_selected=3, extract_all=False)
        self.assertEqual(indices, [])
        self.assertEqual(metadata['selected_frames'], [])
        self.window.load_video_frames.assert_not_called()
        self.warning.assert_called_once()

    def test_sequential_unknown_count_extraction_keeps_requested_indices(self):
        self.manager.add_videos([self.source])
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 0
        frames = [np.full((32, 32, 3), index * 15, dtype=np.uint8) for index in range(12)]
        capture.read.side_effect = [(True, frames[0])] + [(True, frame) for frame in frames]
        with patch('core.project_manager.cv2.VideoCapture', return_value=capture):
            result = self.manager.extract_video_frames('sample', [0, 4, 8])
        self.assertEqual(result['frame_indices'], [0, 4, 8])
        self.assertEqual(result['failed'], 0)
        capture.release.assert_called_once()
        for index in result['frame_indices']:
            image = cv2.imread(str(self.manager.get_frame_path('sample', index)))
            self.assertAlmostEqual(float(image.mean()), index * 15, delta=2)

    def test_sparse_selection_survives_both_project_and_video_loading(self):
        metadata, _ = self.import_video(n_selected=3, extract_all=False)
        metadata['selected_frames'] = [0, 8]
        (self.manager.get_frames_dir('sample') / 'video_metadata.json').write_text(json.dumps(metadata))
        with patch.object(self.MainWindow, 'load_settings'):
            window = self.MainWindow()
        try:
            window.project_path = self.project
            window.project_manager = ProjectManager(self.project)
            with patch.object(window, 'load_frame'):
                window.load_frames_from_project()
                self.assertEqual(window.frame_selected, [True, False, True])
                self.assertEqual(window._get_frame_idx_in_video(2), 8)
                window.load_video_frames('sample')
                self.assertEqual(window.frame_selected, [True, False, True])
                self.assertEqual(window._get_frame_idx_in_video(2), 8)
        finally:
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
                window.close()
            self.app.processEvents()
        self.assertFalse(window.save_worker.isRunning())

    def test_export_excludes_unannotated_frames_even_with_video_level_masks(self):
        self.import_video(n_selected=3, extract_all=False)
        manager = AnnotationManager()
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:20, 8:20] = 255
        manager.save_frame_annotations_png(self.project, 'sample', 0, [])
        manager.save_frame_annotations_png(self.project, 'sample', 4,
                                           [{'mask_id': 1, 'category': 'bee', 'mask': mask}])
        manager.save_video_annotations(self.project, 'sample',
                                       [{'mask_id': 2, 'category': 'hive', 'mask': mask}])
        exported = export_coco_per_video(self.project, ['sample'], 'train')
        self.assertEqual(len(exported), 1)
        coco = json.loads(exported[0].read_text())
        self.assertEqual([image['frame_index'] for image in coco['images']], [4])
        worker = YOLOTrainingWorker(self.project, {})
        self.assertEqual(worker._coco_to_yolo(exported[0], self.root / 'yolo'), 1)
        self.assertEqual([path.name for path in (self.root / 'yolo/images/train').iterdir()],
                         ['sample_frame_000004.jpg'])

    def test_segmentation_conversion_requires_masks_of_target_class(self):
        self.import_video(n_selected=4, extract_all=False)
        polygon = [[8, 8, 20, 8, 20, 20, 8, 20]]
        coco = {
            'categories': [{'id': 1, 'name': 'bee'}, {'id': 4, 'name': 'pollen'}],
            'images': [{'id': index, 'file_name': f'frames/sample/frame_{index:06d}.jpg',
                        'width': 32, 'height': 32} for index in [0, 3, 6, 9]],
            'annotations': [
                {'image_id': 3, 'category_id': 1, 'segmentation': polygon},
                {'image_id': 6, 'category_id': 4, 'segmentation': polygon},
                {'image_id': 9, 'category_id': 1, 'bbox': [8, 8, 12, 12]},
            ],
        }
        source = self.root / 'coco.json'
        source.write_text(json.dumps(coco))
        worker = YOLOTrainingWorker(self.project, {})
        for category, expected_index in [('bee', 3), ('pollen', 6)]:
            with self.subTest(category=category):
                output = self.root / category
                self.assertEqual(worker._coco_to_yolo(source, output, model_type=category), 1)
                self.assertEqual([path.name for path in (output / 'images/train').iterdir()],
                                 [f'sample_frame_{expected_index:06d}.jpg'])
                self.assertEqual(len(list((output / 'labels/train').iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
