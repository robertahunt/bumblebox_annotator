#!/usr/bin/env python3
"""Synthetic integration checks; no research data, model downloads, or GPU needed."""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.marker_detector import MarkerDetector
from core.temporal_hive_prior import TemporalHivePrior
from gui.batch_video_inference_worker import BatchVideoInferenceWorker
from gui.frame_level_validation_worker import FrameLevelValidationWorker
from merge_batch_output_folders import merge_csv


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.settings_dir = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, cls.settings_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.settings_dir.cleanup()

    def marker_image(self, two=False):
        image = np.full((200, 350), 255, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image[50:130, 50:130] = cv2.aruco.generateImageMarker(dictionary, 7, 80)
        if two:
            image[50:130, 220:300] = cv2.aruco.generateImageMarker(dictionary, 8, 80)
        return image

    def test_aruco_defaults_and_explicit_overrides(self):
        detector = MarkerDetector(aruco_dicts=['4x4_50'], enable_qr=False)
        params = detector._create_detector_params({'adaptiveThreshConstant': 11})
        self.assertEqual(params.adaptiveThreshConstant, 11)
        self.assertEqual(params.adaptiveThreshWinSizeMin, 5)
        self.assertEqual(params.polygonalApproxAccuracyRate, 0.06)

    def test_repeated_aruco_bank_detection_is_not_multiple_tags(self):
        detector = MarkerDetector(
            aruco_dicts=['4x4_50', '4x4_100'], aruco_params=[{}, {}], enable_qr=False
        )
        image = self.marker_image()
        result = detector._detect_aruco(image, np.ones_like(image), 0, 0, reject_multiple=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.marker_id, 7)
        image = self.marker_image(two=True)
        self.assertIsNone(detector._detect_aruco(
            image, np.ones_like(image), 0, 0, reject_multiple=True
        ))

    def test_aruco_exclusions_override_allowlist(self):
        detector = MarkerDetector(
            aruco_dicts=['4x4_50'], enable_qr=False,
            allowed_tag_ids=[7, 8], excluded_tag_ids=[7],
        )
        image = self.marker_image(two=True)
        result = detector._detect_aruco(image, np.ones_like(image), 0, 0, reject_multiple=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.marker_id, 8)

    def make_prior(self):
        prior = TemporalHivePrior(resolution=(12, 18), cleanup_kernel_size=0,
                                  min_component_pixels=0, min_prior_weight=1)
        mask = np.ones((18, 12), dtype=np.uint8) * 255
        prior.update('arena', 1, {'bbox': (0, 0, 12, 18), 'mask': mask},
                     mask, [], mask.shape, 100)
        return prior

    def test_temporal_checkpoint_roundtrip_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = [root / 'first.mp4', root / 'second.mp4']
            config = {'output_folder': directory, 'temporal_hive_checkpointing': True,
                      'temporal_hive_resolution': (12, 18),
                      'selected_file_order_signature': 'test-order'}
            worker = BatchVideoInferenceWorker(config)
            worker.temporal_hive_prior = self.make_prior()
            expected = worker.temporal_hive_prior.summaries()
            worker._save_temporal_hive_prior_checkpoint(videos[1], 2, 2)
            loaded, _ = TemporalHivePrior.load_checkpoint(worker._temporal_prior_checkpoint_path())
            self.assertEqual(loaded.resolution, (12, 18))
            self.assertEqual(loaded.summaries(), expected)
            for key in worker.temporal_hive_prior._states:
                before = worker.temporal_hive_prior._states[key]
                after = loaded._states[key]
                np.testing.assert_array_equal(before.hive_sum, after.hive_sum)
                np.testing.assert_array_equal(before.weight_sum, after.weight_sum)
                self.assertEqual(before.last_time_seconds, after.last_time_seconds)

            completed = {worker._video_manifest_path(path) for path in videos}
            cases = [({}, videos, completed, True),
                     ({'confidence_threshold': 0.9}, videos, completed, False),
                     ({'selected_file_order_signature': 'changed'}, videos, completed, False),
                     ({}, list(reversed(videos)), completed, False),
                     ({}, videos, set(), False)]
            for overrides, ordered, done, accepted in cases:
                with self.subTest(overrides=overrides, done=done, accepted=accepted):
                    reader = BatchVideoInferenceWorker({**config, **overrides})
                    reader.temporal_hive_prior = TemporalHivePrior(resolution=(12, 18))
                    reader._load_temporal_hive_prior_checkpoint(ordered, done)
                    self.assertEqual(reader.temporal_hive_prior_checkpoint_index, 2 if accepted else 0)

    def test_cli_rectangular_resolution_reaches_worker_config(self):
        import batch_video_inference_cli as cli
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, model = root / 'sample.mp4', root / 'sample.pt'
            video.touch()
            model.touch()
            argv = ['cli', '--video', str(video), '--bee-model', str(model),
                    '--output-folder', str(root / 'output'),
                    '--temporal-resolution', '800x1500']
            with patch.object(sys, 'argv', argv):
                config = cli.build_config(cli.parse_args())
            self.assertEqual(config['temporal_hive_resolution'], [800, 1500])

    def test_refresh_cli_accepts_explicit_models_without_pollen(self):
        import batch_hive_refresh_cli as cli
        argv = ['cli', '--source-output-folder', 'source', '--file-list', 'videos.txt',
                '--output-folder', 'out', '--hive-model', 'hive.pt',
                '--chamber-model', 'chamber.pt', '--no-pollen-model']
        with patch.object(sys, 'argv', argv):
            args = cli.parse_args()
        self.assertIsNone(args.pollen_model)
        self.assertEqual(args.hive_model, Path('hive.pt'))

    def test_merge_csv_keeps_primary_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ('primary.csv', 'secondary.csv', 'merged.csv')]
            for path, rows in zip(paths, ([('a', 'original')], [('a', 'duplicate'), ('b', 'new')])):
                with path.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(['video_id', 'value'])
                    writer.writerows(rows)
            result = merge_csv(*paths, {'a'})
            self.assertEqual(result['dropped_secondary_overlap_rows'], 1)
            with paths[2].open(newline='') as handle:
                self.assertEqual(list(csv.DictReader(handle)),
                                 [{'video_id': 'a', 'value': 'original'},
                                  {'video_id': 'b', 'value': 'new'}])

    def test_merge_refuses_to_overwrite_its_input(self):
        import merge_batch_output_folders as cli
        with tempfile.TemporaryDirectory() as directory:
            argv = ['cli', '--primary-folder', directory, '--secondary-folder', directory,
                    '--output-folder', directory, '--overwrite']
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(SystemExit, 'separate folder'):
                cli.main()

    def test_disposable_project_annotations_roundtrip(self):
        from core.annotation import AnnotationManager
        from core.project_manager import ProjectManager
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            ProjectManager().create_project(project, 'Synthetic review')
            manager = AnnotationManager()
            annotations = []
            for index, category in enumerate(('bee', 'pollen'), start=1):
                mask = np.zeros((30, 30), dtype=np.uint8)
                mask[index * 5:index * 5 + 4, 5:10] = 255
                annotations.append({'mask_id': index, 'category': category, 'mask': mask})
            manager.save_frame_annotations_png(project, 'sample', 0, annotations)
            reopened = ProjectManager().load_project(project)
            self.assertEqual(reopened['name'], 'Synthetic review')
            loaded = AnnotationManager().load_frame_annotations_png(project, 'sample', 0)
            self.assertEqual([item['category'] for item in loaded], ['bee', 'pollen'])
            for expected, actual in zip(annotations, loaded):
                np.testing.assert_array_equal(expected['mask'], actual['mask'])

    def test_validation_keeps_bee_and_pollen_categories_separate(self):
        worker = FrameLevelValidationWorker(None, {})
        polygon = [[1, 1, 5, 1, 5, 5, 1, 5]]
        coco = {'images': [{'id': 1, 'file_name': 'frame_000000.png'}],
                'categories': [{'id': 1, 'name': 'bee'}, {'id': 4, 'name': 'pollen'}],
                'annotations': [{'id': index, 'image_id': 1, 'category_id': category,
                                 'segmentation': polygon, 'bbox': [1, 1, 4, 4]}
                                for index, category in ((10, 1), (20, 4))]}
        pollen = worker._extract_gt_pollen_from_coco(coco, 0, (10, 10))
        self.assertEqual([item['pollen_id'] for item in pollen], [20])
        self.assertEqual(len(worker._extract_gt_bees_from_coco(coco, 0, (10, 10))), 1)

    def test_main_window_constructs_with_both_workflows(self):
        from gui.main_window import MainWindow
        with patch.object(MainWindow, 'load_settings'):
            window = MainWindow()
        try:
            self.assertTrue(hasattr(window, 'canvas'))
            self.assertTrue(hasattr(window, 'dirty_frame_annotation_keys'))
            self.assertTrue(hasattr(window, 'imaging_setup_profiles'))
            window.show()
            self.app.processEvents()
        finally:
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
                window.close()
            self.app.processEvents()
            self.assertFalse(window.save_worker.isRunning())


if __name__ == '__main__':
    unittest.main()
