#!/usr/bin/env python3
"""Cleaning a model's predicted mask must not cut a bee in two."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox

from core.predicted_masks import MIN_PIECE_FRACTION, instance_pieces


def largest_piece_only(mask):
    """The clean-up this replaces."""
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(binary)
    if contours:
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], 0, 255, -1)
    return clean


def wire_split_bee(shape=(80, 120)):
    """A bee in two pieces either side of a wire, plus a stray fragment far from both."""
    mask = np.zeros(shape, np.uint8)
    mask[20:60, 20:60] = 255     # front of the bee
    mask[25:55, 66:90] = 255     # the rest of it, beyond a six-pixel wire
    mask[4:8, 108:112] = 255     # fragment
    return mask


class Tensor:
    """Stands in for the tensors a YOLO result holds."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __getitem__(self, index):
        return self.array[index]


class InstancePiecesTests(unittest.TestCase):
    def test_a_bee_cut_in_two_keeps_both_pieces_and_loses_the_fragment(self):
        mask = wire_split_bee()
        kept = instance_pieces(mask)
        self.assertEqual(kept.dtype, np.uint8)
        self.assertTrue(np.all(kept[20:60, 20:60] == 255))
        self.assertTrue(np.all(kept[25:55, 66:90] == 255))
        self.assertFalse(np.any(kept[:, 60:66]))    # the wire stays uncovered
        self.assertFalse(np.any(kept[4:8, 108:112]))
        # Keeping only the largest piece lost the second half.
        self.assertFalse(np.any(largest_piece_only(mask)[25:55, 66:90]))

    def test_a_small_piece_is_kept_only_where_the_existing_mask_covers_it(self):
        mask = np.zeros((80, 120), np.uint8)
        mask[20:60, 20:60] = 255
        mask[30:36, 66:70] = 255     # a leg tip beyond the wire: far under a tenth of the bee
        self.assertFalse(np.any(instance_pieces(mask)[30:36, 66:70]))
        drawn = np.zeros_like(mask)
        drawn[22:58, 22:58] = 255
        self.assertFalse(np.any(instance_pieces(mask, drawn)[30:36, 66:70]))
        drawn[31:35, 66:69] = 255    # the annotator had drawn that leg tip
        kept = instance_pieces(mask, drawn)
        self.assertTrue(np.all(kept[30:36, 66:70] == 255))
        self.assertTrue(np.array_equal(kept, mask))

    def test_the_size_limit_is_a_fraction_of_the_largest_piece(self):
        self.assertEqual(MIN_PIECE_FRACTION, 0.1)
        mask = np.zeros((60, 120), np.uint8)
        mask[10:50, 10:50] = 255     # outline area 39 x 39
        mask[10:22, 60:73] = 255     # outline area 11 x 12, under a tenth
        self.assertFalse(np.any(instance_pieces(mask)[10:22, 60:73]))
        mask[10:24, 60:73] = 255     # 13 x 12, just over a tenth
        self.assertTrue(np.all(instance_pieces(mask)[10:24, 60:73] == 255))
        self.assertFalse(np.any(instance_pieces(mask, min_fraction=0.5)[10:24, 60:73]))

    def test_one_piece_and_noisy_masks_come_out_as_before(self):
        rng = np.random.default_rng(11)
        for trial in range(60):
            mask = np.zeros((70, 90), np.uint8)
            y, x = int(rng.integers(5, 25)), int(rng.integers(5, 35))
            mask[y:y + 35, x:x + 40] = 255
            mask[y + 10:y + 16, x + 12:x + 20] = 0          # a hole, which the clean-up fills
            for _ in range(int(rng.integers(0, 6))):         # specks well under a tenth of the piece
                sy, sx = int(rng.integers(0, 66)), int(rng.integers(0, 86))
                mask[sy:sy + 3, sx:sx + 3] = 255
            with self.subTest(trial):
                self.assertTrue(np.array_equal(instance_pieces(mask), largest_piece_only(mask)))

    def test_empty_thin_and_mismatched_masks(self):
        empty = np.zeros((10, 12), np.uint8)
        self.assertFalse(np.any(instance_pieces(empty)))
        self.assertEqual(instance_pieces(empty).shape, empty.shape)
        dots = empty.copy()
        dots[2, 2] = dots[7, 9] = 255   # outlines with no area: one is kept, as before
        self.assertEqual(int(np.count_nonzero(instance_pieces(dots))), 1)
        self.assertTrue(np.array_equal(instance_pieces(dots), largest_piece_only(dots)))
        self.assertTrue(np.array_equal(instance_pieces(wire_split_bee() > 0), instance_pieces(wire_split_bee())))
        with self.assertRaises(ValueError):
            instance_pieces(empty, np.zeros((5, 5), np.uint8))


class ValidationCleanUpTests(unittest.TestCase):
    def test_two_stage_validation_keeps_both_pieces_of_a_predicted_bee(self):
        from gui.frame_level_validation_worker import FrameLevelValidationWorker
        truth = wire_split_bee()
        frame = np.zeros((*truth.shape, 3), np.uint8)
        predicted = cv2.resize(truth, (640, 640), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
        result = SimpleNamespace(masks=[SimpleNamespace(data=[Tensor(predicted)])],
                                 boxes=SimpleNamespace(conf=Tensor([0.9])))
        worker = SimpleNamespace(
            config={'conf_threshold': 0.5, 'instance_crop_margin': 0},
            _iter_bbox_detections=lambda frame, model: [((0, 0, truth.shape[1] - 1, truth.shape[0] - 1), 0.8)],
            _bee_from_mask=lambda mask, confidence: {'mask': mask, 'confidence': confidence})
        model = SimpleNamespace(predict=lambda **options: [result])
        bees = FrameLevelValidationWorker._predict_bees_bbox_instance_focused(worker, frame, None, model)
        self.assertEqual(len(bees), 1)
        mask = bees[0]['mask']
        self.assertTrue(np.all(mask[21:59, 21:59] == 255))      # front of the bee
        self.assertTrue(np.all(mask[26:54, 67:89] == 255))      # the piece beyond the wire
        self.assertFalse(np.any(mask[:, 61:65]))                # the wire
        self.assertFalse(np.any(mask[0:12, 100:120]))           # the fragment
        # Resizing to the model's size and back may move an edge by a pixel, no more.
        expected = instance_pieces(truth) > 0
        overlap = np.count_nonzero(expected & (mask > 0)) / np.count_nonzero(expected | (mask > 0))
        self.assertGreater(overlap, 0.95)


class RefineCleanUpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from gui.main_window import MainWindow
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        settings = QSettings(str(Path(self.temp.name) / 'settings.ini'), QSettings.Format.IniFormat)
        mock = patch('gui.main_window.QSettings', lambda *a, **k: settings)
        mock.start()
        self.addCleanup(mock.stop)
        with patch.object(MainWindow, 'load_settings'):
            self.window = w = MainWindow()

        def close():
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                w.close()
            self.app.processEvents()
        self.addCleanup(close)

        # The annotator drew the bee in two pieces either side of a wire.
        self.drawn = np.zeros((120, 160), np.uint8)
        self.drawn[40:80, 40:80] = 255
        self.drawn[45:75, 86:110] = 255
        w.canvas.load_image(np.zeros((120, 160, 3), np.uint8))
        w.canvas.set_annotations([{'mask': self.drawn, 'mask_id': 1, 'category': 'bee'}])

        # The model returns the same bee one pixel tighter, plus a fragment in the corner of its crop.
        padding = 10
        crop = np.zeros((40 + 2 * padding, 70 + 2 * padding), np.uint8)   # rows 30..89, columns 30..119
        crop[11:49, 11:49] = 255
        crop[16:44, 57:79] = 255
        crop[2:5, 84:87] = 255
        predicted = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255
        result = SimpleNamespace(masks=[SimpleNamespace(data=[Tensor(predicted)])],
                                 boxes=[SimpleNamespace(conf=[0.9])])
        toolbar = w.yolo_instance_focused_toolbar
        for name, value in (('get_padding', lambda: padding),
                            ('get_model', lambda: SimpleNamespace(predict=lambda **options: [result]))):
            mock = patch.object(toolbar, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def assert_both_pieces(self, mask):
        self.assertTrue(np.all(mask[42:78, 42:78] > 0))      # front of the bee
        self.assertTrue(np.all(mask[47:73, 88:108] > 0))     # the piece beyond the wire
        self.assertFalse(np.any(mask[:, 81:86] > 0))         # the wire
        self.assertFalse(np.any(mask[30:36, 112:120] > 0))   # the fragment

    def test_refining_a_split_bee_keeps_both_pieces(self):
        with patch('builtins.print'):
            success, message, stats = self.window._refine_instance_focused_by_index(1)
        self.assertTrue(success, message)
        self.assert_both_pieces(self.window.canvas.editing_mask)
        self.assertGreater(stats['refined_area'], 0.8 * stats['original_area'])

    def test_prediction_for_overlap_resolution_keeps_both_pieces(self):
        with patch('builtins.print'):
            success, message, data = self.window._get_instance_focused_prediction(1)
        self.assertTrue(success, message)
        self.assert_both_pieces(data['mask_probabilities'])


if __name__ == '__main__':
    unittest.main()
