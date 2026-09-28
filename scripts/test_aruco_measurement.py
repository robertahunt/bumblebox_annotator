#!/usr/bin/env python3
"""Synthetic ArUco measurement checks; no models, GPU, or research data needed."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QSettings, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog

from core.aruco_measurement import MeasurementVideoReader, measure_tag, tag_perimeter_bounds
from core.aruco_parameter_optimizer import _evaluate_candidate, build_parameter_grid
from gui.aruco_measurement_dialog import ArucoMeasurementDialog
from gui.batch_video_inference_dialog import BatchVideoInferenceConfigDialog


class GeometryTests(unittest.TestCase):
    def test_perimeter_uses_longest_original_dimension(self):
        for width, height in ((200, 100), (100, 200)):
            points = [(10, 10), (30, 10), (30, 40), (10, 40)]
            for ordered in (points, list(reversed(points))):
                result = measure_tag(ordered, width, height)
                self.assertAlmostEqual(result["perimeter_px"], 100)
                self.assertAlmostEqual(result["perimeter_rate"], 0.5)

    def test_perspective_quadrilateral_uses_all_four_sides(self):
        result = measure_tag([(10, 10), (30, 10), (40, 30), (0, 30)], 200, 100)
        self.assertAlmostEqual(result["perimeter_px"], 60 + 2 * np.sqrt(500))

    def test_invalid_corners_are_rejected(self):
        invalid = [
            [], [(0, 0)] * 4,
            [(10, 10), (30, 30), (30, 10), (10, 30)],
            [(10, 10), (20, 10), (30, 10), (40, 10)],
            [(-1, 10), (30, 10), (30, 30), (10, 30)],
            [(10, 10), (100, 10), (100, 30), (10, 30)],
            [(10, 10), (30, 10), (30, float("nan")), (10, 30)],
        ]
        for corners in invalid:
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                measure_tag(corners, 100, 100)

    def test_bounds_and_margin(self):
        self.assertEqual(tag_perimeter_bounds(0.02, 0.04), (0.018, 0.044))
        self.assertEqual(tag_perimeter_bounds(0.02, 0.04, 0), (0.02, 0.04))
        self.assertEqual(tag_perimeter_bounds(0.02, 0.02), (0.018, 0.022))
        self.assertEqual(tag_perimeter_bounds(0.2, 0.4), (0.18, 0.44))

    def test_invalid_bounds_cannot_be_applied(self):
        for args in ((0, 0.04), (0.04, 0.02), (0.02, float("inf")),
                     (0.02, 0.04, 100), (0.02, 0.04, -1), (0.02, 0.02, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                tag_perimeter_bounds(*args)


class FakeCapture:
    def __init__(self, count=0, seek_works=True):
        self.count = count
        self.seek_works = seek_works
        self.position = 0
        self.closed = False

    def isOpened(self):
        return not self.closed

    def get(self, prop):
        return self.count if prop == cv2.CAP_PROP_FRAME_COUNT else self.position

    def set(self, prop, index):
        if self.seek_works:
            self.position = index
        return True

    def read(self):
        if self.position >= 4:
            return False, None
        frame = np.full((20, 30, 3), self.position, np.uint8)
        self.position += 1
        return True, frame

    def grab(self):
        return self.read()[0]

    def release(self):
        self.closed = True


class ReaderTests(unittest.TestCase):
    def test_unknown_mjpeg_count_and_backwards_navigation(self):
        for reported_count in (0, -192153584101141, float("nan")):
            captures = []

            def create(_path):
                capture = FakeCapture(reported_count, seek_works=False)
                captures.append(capture)
                return capture

            with patch("core.aruco_measurement.cv2.VideoCapture", side_effect=create):
                reader = MeasurementVideoReader("synthetic.mjpeg")
                self.assertIsNone(reader.frame_count)
                for index in (0, 1, 2, 0, 3):
                    self.assertEqual(int(reader.read(index)[0, 0, 0]), index)
                with self.assertRaises(ValueError):
                    reader.read(4)
                self.assertEqual(int(reader.read(0)[0, 0, 0]), 0)
                reader.close()
                self.assertTrue(all(cap.closed for cap in captures))

    def test_unreliable_seek_falls_back_to_sequential(self):
        with patch("core.aruco_measurement.cv2.VideoCapture",
                   side_effect=lambda _path: FakeCapture(4, seek_works=False)):
            reader = MeasurementVideoReader("synthetic.avi")
            self.assertEqual(int(reader.read(3)[0, 0, 0]), 3)
            self.assertEqual(int(reader.read(1)[0, 0, 0]), 1)
            reader.close()

    def test_invalid_video_releases_capture(self):
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch("core.aruco_measurement.cv2.VideoCapture", return_value=capture):
            with self.assertRaises(ValueError):
                MeasurementVideoReader("missing.avi")
        capture.release.assert_called_once()


class DialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.temp = tempfile.TemporaryDirectory()
        QSettings.setDefaultFormat(QSettings.Format.IniFormat)
        for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
            QSettings.setPath(QSettings.Format.IniFormat, scope, cls.temp.name)
        cls.video = Path(cls.temp.name) / "tags.avi"
        writer = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 360))
        if not writer.isOpened():
            raise RuntimeError("MJPG test video writer unavailable")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for _ in range(3):
            frame = np.full((360, 640, 3), 225, np.uint8)
            for tag_id, x, size in ((7, 80, 40), (12, 400, 80)):
                marker = cv2.aruco.generateImageMarker(dictionary, tag_id, size)
                frame[140:140 + size, x:x + size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        QSettings("BumbleBoxAnnotator", "BatchVideoInference").clear()
        self.dialogs = []

    def tearDown(self):
        for dialog in self.dialogs:
            dialog.reject()
            dialog.deleteLater()
        self.app.processEvents()

    def measurement_dialog(self):
        dialog = ArucoMeasurementDialog(self.video)
        self.dialogs.append(dialog)
        dialog.show()
        self.app.processEvents()
        return dialog

    def batch_dialog(self):
        dialog = BatchVideoInferenceConfigDialog()
        self.dialogs.append(dialog)
        dialog.files_radio.setChecked(True)
        dialog.selected_files = [str(self.video)]
        dialog.aruco_dictionary_combo.setCurrentText("4x4_50")
        return dialog

    def add_corners(self, dialog, points):
        for point in points:
            dialog.place_corner(QPointF(*point))
            dialog.end_drag()

    def complete_measurements(self, dialog):
        self.add_corners(dialog, [(80, 140), (120, 140), (120, 180), (80, 180)])
        dialog.role_group.buttons()[1].setChecked(True)
        self.add_corners(dialog, [(400, 140), (480, 140), (480, 220), (400, 220)])

    def test_measured_bounds_keep_both_tags_in_optimizer_and_runtime(self):
        dialog = self.measurement_dialog()
        self.complete_measurements(dialog)
        lower, upper = dialog.bounds
        params = {"minMarkerPerimeterRate": lower, "maxMarkerPerimeterRate": upper}
        result = _evaluate_candidate(params, [dialog.reader.current_frame], "4x4_50", None, None, None)
        self.assertEqual(result.mean_decoded, 2)
        self.assertEqual(result.mean_detected, 2)
        self.assertEqual(result.mean_filtered, 0)
        from core.marker_detector import MarkerDetector
        detector = MarkerDetector(aruco_dicts=["4x4_50"], aruco_params=[params], enable_qr=False)
        configured = detector.aruco_detectors["4x4_50"][0]["detector"]
        _, ids, _ = configured.detectMarkers(dialog.reader.current_frame)
        self.assertEqual(set(ids.flatten()), {7, 12})

    def test_measurements_across_frames_and_zoom(self):
        dialog = self.measurement_dialog()
        self.assertFalse(dialog.apply_button.isEnabled())
        self.add_corners(dialog, [(80, 140), (120, 140), (120, 180), (80, 180)])
        dialog.load_frame(1)
        self.assertEqual(dialog.view.overlays, {})
        dialog.view.zoom(2)
        transform = dialog.view.transform()
        dialog.role_group.buttons()[1].setChecked(True)
        self.add_corners(dialog, [(400, 140), (480, 140), (480, 220), (400, 220)])
        self.assertEqual(dialog.bounds, (0.225, 0.55))
        dialog.load_frame(0)
        self.assertEqual(dialog.view.transform(), transform)
        self.assertEqual(set(dialog.view.overlays), {"smallest"})
        self.assertEqual(dialog.bounds, (0.225, 0.55))
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertIsNone(dialog.reader.cap)

    def test_mouse_placement_and_drag_use_source_coordinates(self):
        dialog = self.measurement_dialog()
        dialog.view.zoom(2)
        dialog.view.centerOn(100, 160)
        self.app.processEvents()
        position = dialog.view.mapFromScene(QPointF(80, 140))
        QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        actual = dialog.measurements["smallest"]["points"][0]
        self.assertAlmostEqual(actual[0], 80, delta=1)
        self.assertAlmostEqual(actual[1], 140, delta=1)
        QTest.mousePress(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertEqual(dialog.drag_index, 0)
        target = dialog.view.mapFromScene(QPointF(85, 145))
        QTest.mouseMove(dialog.view.viewport(), target)
        QTest.mouseRelease(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=target)
        self.assertIsNone(dialog.drag_index)
        moved = dialog.measurements["smallest"]["points"][0]
        self.assertAlmostEqual(moved[0], 85, delta=1)
        self.assertAlmostEqual(moved[1], 145, delta=1)
        QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.RightButton, pos=target)
        self.assertEqual(len(dialog.measurements["smallest"]["points"]), 1)

    def test_undo_clear_margin_and_screenshot(self):
        dialog = self.measurement_dialog()
        self.complete_measurements(dialog)
        self.assertTrue(dialog.apply_button.isEnabled())
        dialog.margin_spin.setValue(20)
        self.assertEqual(dialog.bounds, (0.2, 0.6))
        self.app.processEvents()
        self.assertTrue(dialog.grab().save("/tmp/annotation-tag-bounds.png"))
        dialog.remove_last_corner()
        self.assertIsNone(dialog.bounds)
        self.assertFalse(dialog.apply_button.isEnabled())
        dialog.clear_measurement()
        self.assertIsNone(dialog.measurements["largest"])
        self.assertIsNotNone(dialog.measurements["smallest"])

    def test_outside_click_and_incomplete_accept_do_nothing(self):
        dialog = self.measurement_dialog()
        dialog.accept()
        self.assertTrue(dialog.isVisible())
        dialog.view.zoom(0.2)
        QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(1, 1))
        self.assertIsNone(dialog.measurements["smallest"])
        dialog.reject()
        self.assertIsNone(dialog.reader.cap)

    def test_batch_apply_changes_only_bounds_and_enables_optimization(self):
        batch = self.batch_dialog()
        before = batch._collect_aruco_sweep_overrides()
        with patch("gui.aruco_measurement_dialog.ArucoMeasurementDialog") as factory:
            measurement = factory.return_value
            measurement.exec.return_value = QDialog.DialogCode.Accepted
            measurement.bounds = (0.018, 0.044)
            batch.measure_aruco_bounds()
            factory.assert_called_once_with(str(self.video), batch)
            measurement.reader.close.assert_called_once()
        after = batch._collect_aruco_sweep_overrides()
        self.assertEqual(after, {**before, "minMarkerPerimeterRate": [0.018], "maxMarkerPerimeterRate": [0.044]})
        self.assertTrue(batch.aruco_optimize_check.isChecked())
        for params in build_parameter_grid("daily", after, max_combinations=10):
            self.assertEqual(params["minMarkerPerimeterRate"], 0.018)
            self.assertEqual(params["maxMarkerPerimeterRate"], 0.044)
        batch._save_last_settings()
        restored = self.batch_dialog()
        self.assertEqual(restored.aruco_sweep_min_perimeter_edit.text(), "0.018000")
        self.assertEqual(restored.aruco_sweep_max_perimeter_edit.text(), "0.044000")

    def test_batch_cancel_leaves_settings_unchanged(self):
        batch = self.batch_dialog()
        before = batch._collect_aruco_sweep_overrides()
        with patch("gui.aruco_measurement_dialog.ArucoMeasurementDialog") as factory:
            factory.return_value.exec.return_value = QDialog.DialogCode.Rejected
            batch.measure_aruco_bounds()
        self.assertEqual(batch._collect_aruco_sweep_overrides(), before)
        self.assertFalse(batch.aruco_optimize_check.isChecked())

    def test_folder_picker_and_cancel(self):
        batch = self.batch_dialog()
        batch.folder_radio.setChecked(True)
        batch.input_folder_edit.setText(str(self.video.parent))
        with patch("gui.batch_video_inference_dialog.QFileDialog.getOpenFileName", return_value=("", "")) as picker:
            with patch("gui.aruco_measurement_dialog.ArucoMeasurementDialog") as factory:
                batch.measure_aruco_bounds()
                self.assertEqual(picker.call_args.args[2], str(self.video.parent))
                factory.assert_not_called()

    def test_dictionary_required_and_aruco_disabled(self):
        batch = self.batch_dialog()
        batch.aruco_dictionary_combo.setCurrentText("Auto 4x4 dictionaries")
        with patch("gui.batch_video_inference_dialog.QMessageBox.information") as info:
            batch.measure_aruco_bounds()
            info.assert_called_once()
        batch.enable_aruco_check.setChecked(False)
        self.assertFalse(batch.aruco_measure_bounds_btn.isEnabled())
        batch.enable_aruco_check.setChecked(True)
        self.assertTrue(batch.aruco_measure_bounds_btn.isEnabled())

    def test_unreadable_video_keeps_batch_dialog_open(self):
        batch = self.batch_dialog()
        before = batch._collect_aruco_sweep_overrides()
        with patch("gui.aruco_measurement_dialog.ArucoMeasurementDialog", side_effect=ValueError("Unreadable")):
            with patch("gui.batch_video_inference_dialog.QMessageBox.warning") as warning:
                batch.measure_aruco_bounds()
                warning.assert_called_once()
        self.assertEqual(batch._collect_aruco_sweep_overrides(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
