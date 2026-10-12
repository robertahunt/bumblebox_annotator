#!/usr/bin/env python3
"""Ordered detector-bank concurrency and batch configuration regressions."""

import argparse
import os
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from core.marker_detector import MarkerDetector
from core.batch_video_processor import BatchVideoProcessor
from gui.batch_video_inference_worker import BatchVideoInferenceWorker
from gui.batch_video_inference_dialog import BatchVideoInferenceConfigDialog
import batch_video_inference_cli as cli


def snapshot(detections):
    return [(key, value.marker_id, value.dict_type, value.confidence,
             value.corners.tolist(), list(value.center))
            for key, value in detections.items()]


def square(x, y, size=20):
    return np.array([[[x, y], [x + size, y], [x + size, y + size], [x, y + size]]],
                    dtype=np.float32)


class DetectionWorkerTests(unittest.TestCase):
    def setUp(self):
        for target, value in (('core.marker_detector.os.sched_getaffinity', set(range(16))),
                              ('core.marker_detector.cv2.getNumThreads', 1)):
            mocked = patch(target, return_value=value, create=True)
            mocked.start()
            self.addCleanup(mocked.stop)

    def tearDown(self):
        self.assertFalse([t for t in threading.enumerate() if t.name.startswith('aruco-detect')])

    def detector(self, workers=4, **kwargs):
        return MarkerDetector(aruco_dicts=['4x4_50', '4x4_100'], aruco_params=[{}, {}],
                              enable_qr=False, detection_workers=workers, **kwargs)

    def test_ordered_merge_filtering_bbox_fallback_and_rejection(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        annotations = [
            {'category': 'bee', 'mask_id': 1, 'bbox': [0, 0, 45, 45],
             'mask': np.zeros(image.shape[:2], dtype=np.uint8)},
            {'category': 'bee', 'mask_id': 2, 'bbox': [60, 0, 60, 45]},
            {'category': 'bee', 'mask_id': 3, 'bbox': [0, 0, 200, 100]},
            {'category': 'hive', 'mask_id': 4, 'bbox': [0, 0, 200, 100]},
        ]
        corners = [square(10, 10), square(75, 10), square(140, 10)]
        ids = np.array([[7], [7], [8]])
        detector = self.detector(allowed_tag_ids=[7, 8], excluded_tag_ids=[8])
        completed = []
        other_finished = threading.Event()
        for index, (_, entry) in enumerate(detector._aruco_jobs()):
            def detect(frame, index=index):
                if index == 0:
                    self.assertTrue(other_finished.wait(timeout=2))
                completed.append(index)
                if index != 0:
                    other_finished.set()
                # Equal-size shifted corners test confidence-tie precedence.
                return [corner + index / 4 for corner in corners], ids.copy(), []
            entry['detector'] = Mock(detectMarkers=detect)
        result = detector.detect_aruco_in_bee_instances(image, annotations)
        self.assertNotEqual(completed[0], 0)
        self.assertEqual(list(result), [1, 2])
        np.testing.assert_array_equal(result[1].corners, corners[0][0])
        self.assertEqual(result[1].dict_type, '4x4_50')
        detector.detection_workers = 1
        self.assertEqual(snapshot(result), snapshot(detector.detect_aruco_in_bee_instances(image, annotations)))
        serial = detector.detect_aruco_in_bee_instances(image, annotations, reject_multiple=False)
        detector.detection_workers = 4
        self.assertEqual(snapshot(serial), snapshot(detector.detect_aruco_in_bee_instances(
            image, annotations, reject_multiple=False)))
        self.assertIn(3, serial)

    def test_native_markers_identical_and_global_thread_setting_untouched(self):
        image = np.full((200, 350), 255, dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image[50:130, 50:130] = cv2.aruco.generateImageMarker(dictionary, 7, 80)
        image[50:130, 220:300] = cv2.aruco.generateImageMarker(dictionary, 8, 80)
        annotations = [{'category': 'bee', 'mask_id': i + 1, 'bbox': [x, 0, 170, 200]}
                       for i, x in enumerate((0, 175))]
        serial = self.detector(workers=1)
        parallel = self.detector()
        with patch('core.marker_detector.cv2.setNumThreads') as setter:
            a = serial.detect_aruco_in_bee_instances(image, annotations)
            b = parallel.detect_aruco_in_bee_instances(image, annotations)
            self.assertEqual(snapshot(a), snapshot(b))
            self.assertEqual({v.marker_id for v in b.values()}, {7, 8})
            setter.assert_not_called()

    def test_worker_caps_and_duplicate_native_objects(self):
        detector = self.detector(workers=8)
        self.assertEqual(detector.effective_detection_workers, 4)
        with patch('core.marker_detector.os.sched_getaffinity', return_value={0, 1, 2}):
            self.assertEqual(detector.effective_detection_workers, 3)
        with patch('core.marker_detector.cv2.getNumThreads', return_value=8):
            self.assertEqual(detector.effective_detection_workers, 2)
        with patch('core.marker_detector.cv2.getNumThreads', return_value=32):
            self.assertEqual(detector.effective_detection_workers, 1)
        with patch('core.marker_detector.os.sched_getaffinity', side_effect=OSError), \
                patch('core.marker_detector.os.cpu_count', return_value=2):
            self.assertEqual(detector.effective_detection_workers, 2)
        detector.aruco_dicts_to_try.append('4x4_50')
        self.assertEqual(detector.effective_detection_workers, 1)
        detector.aruco_dicts_to_try.clear()
        self.assertEqual(detector.effective_detection_workers, 1)
        with patch('core.marker_detector.ThreadPoolExecutor') as executor:
            self.assertEqual(detector._detect_aruco_bank(np.zeros((5, 5))), [])
            executor.assert_not_called()

    def test_parallelism_is_bounded_and_frames_do_not_share_native_detectors(self):
        detector = self.detector(workers=2)
        lock = threading.Lock()
        active = set()
        peak = [0]
        for index, (_, entry) in enumerate(detector._aruco_jobs()):
            def detect(image, index=index):
                with lock:
                    self.assertNotIn(index, active)
                    active.add(index)
                    peak[0] = max(peak[0], len(active))
                time.sleep(0.02)
                with lock:
                    active.remove(index)
                return [], None, []
            entry['detector'] = Mock(detectMarkers=detect)
        with ThreadPoolExecutor(max_workers=2) as caller:
            futures = [caller.submit(detector._detect_aruco_bank, np.zeros((5, 5))) for _ in range(2)]
            for future in futures:
                self.assertEqual(len(future.result()), 4)
        self.assertEqual(peak[0], 2)
        self.assertFalse(active)

    def test_exception_waits_for_active_calls_and_releases_lock(self):
        detector = self.detector(workers=2)
        active = threading.Event()
        finished = threading.Event()
        def slow(image):
            active.set()
            time.sleep(0.05)
            finished.set()
            return [], None, []
        def fail(image):
            self.assertTrue(active.wait(timeout=2))
            raise RuntimeError('native failure')
        for _, entry in detector._aruco_jobs():
            entry['detector'] = Mock(detectMarkers=Mock(return_value=([], None, [])))
        jobs = detector._aruco_jobs()
        jobs[0][1]['detector'].detectMarkers = fail
        jobs[1][1]['detector'].detectMarkers = slow
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            detector._detect_aruco_bank(np.zeros((5, 5)))
        self.assertTrue(finished.is_set())
        jobs[0][1]['detector'].detectMarkers = lambda image: ([], None, [])
        self.assertEqual(len(detector._detect_aruco_bank(np.zeros((5, 5)))), 4)

    def test_batch_stop_finishes_current_frame(self):
        processor = BatchVideoProcessor(
            Path('synthetic.mp4'), 'synthetic', None, None, None, None, 0.5, 0.5,
            aruco_dicts=['4x4_50'], aruco_params_bank=[{}, {}], aruco_detection_workers=2,
            compute_spatial_metrics=False, log_callback=lambda message: None,
        )
        stop = threading.Event()
        calls = []
        for index, (_, entry) in enumerate(processor.marker_detector._aruco_jobs()):
            def detect(image, index=index):
                stop.set()
                time.sleep(0.02)
                calls.append(index)
                return [], None, []
            entry['detector'] = Mock(detectMarkers=detect)
        processor.stop_callback = stop.is_set
        processor._process_frame = lambda frame, number: processor.marker_detector._detect_aruco_bank(frame)
        capture = Mock()
        capture.get.return_value = 3
        capture.read.return_value = (True, np.zeros((20, 20, 3), dtype=np.uint8))
        with patch('core.batch_video_processor.cv2.VideoCapture', return_value=capture):
            self.assertFalse(processor.process())
        self.assertEqual(processor.frame_count, 1)
        self.assertEqual(sorted(calls), [0, 1])
        capture.release.assert_called_once()


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_cli_validation_config_and_resume_compatibility(self):
        for value in ('0', '-1', '1.5', 'invalid'):
            with self.assertRaises(argparse.ArgumentTypeError):
                cli.positive_worker_count(value)
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                MarkerDetector(detection_workers=value)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, model = root / 'sample.mp4', root / 'bee.pt'
            video.touch()
            model.touch()
            defaults = cli.load_saved_defaults()
            defaults.update({key: None for key in (
                'bee_model', 'hive_model', 'pollen_model', 'chamber_model', 'tag_map', 'exclude_tags')})
            arguments = ['cli', '--video', str(video), '--bee-model', str(model),
                         '--output-folder', str(root / 'output'), '--aruco-detection-workers', '3']
            with patch.object(cli, 'load_saved_defaults', return_value=defaults), \
                    patch.object(sys, 'argv', arguments):
                config = cli.build_config(cli.parse_args())
            self.assertEqual(config['aruco_detection_workers'], 3)
        baseline = BatchVideoInferenceWorker({})
        for count in (1, 8):
            other = BatchVideoInferenceWorker({'aruco_detection_workers': count})
            self.assertEqual(baseline.config_signature, other.config_signature)
            self.assertEqual(baseline.analysis_config_signature, other.analysis_config_signature)

    def test_dialog_control_and_saved_settings_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = QSettings(str(root / 'settings.ini'), QSettings.Format.IniFormat)
            with patch('gui.batch_video_inference_dialog.QSettings', return_value=settings), \
                    patch('gui.batch_video_inference_dialog.os.cpu_count', return_value=8):
                dialog = BatchVideoInferenceConfigDialog()
                dialog.enable_aruco_check.setChecked(True)
                dialog.aruco_optimize_check.setChecked(False)
                self.assertTrue(dialog.aruco_detection_workers_spin.isEnabled())
                self.assertFalse(dialog.aruco_workers_spin.isEnabled())
                dialog.enable_aruco_check.setChecked(False)
                self.assertFalse(dialog.aruco_detection_workers_spin.isEnabled())
                dialog.enable_aruco_check.setChecked(True)
                dialog.aruco_detection_workers_spin.setValue(2)
                dialog.input_folder_edit.setText(folder)
                model = root / 'bee.pt'
                model.touch()
                dialog.bee_model_edit.setText(str(model))
                for edit in (dialog.hive_model_edit, dialog.pollen_model_edit, dialog.chamber_model_edit,
                             dialog.brood_model_edit, dialog.aruco_tag_list_edit, dialog.aruco_exclude_tag_list_edit):
                    edit.clear()
                dialog.output_folder_edit.setText(str(root / 'output'))
                with patch('gui.batch_video_inference_dialog.QMessageBox.warning') as warning:
                    dialog.accept()
                    warning.assert_not_called()
                self.assertEqual(dialog.config['aruco_detection_workers'], 2)
                second = BatchVideoInferenceConfigDialog()
                self.assertEqual(second.aruco_detection_workers_spin.value(), 2)
                self.assertEqual(cli.positive_worker_count(settings.value('aruco/detection_workers')), 2)
                dialog.close()
                second.close()


if __name__ == '__main__':
    unittest.main()
