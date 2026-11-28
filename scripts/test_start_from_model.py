"""Whole-frame training can continue from a trained model of the same annotation type."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PyQt6.QtWidgets import QApplication
from ultralytics.nn.tasks import DetectionModel, SegmentationModel

from gui.training_dialog import STANDARD_WEIGHTS_LABEL, TrainingConfigDialog
from training.yolo_trainer import StartWeights, YOLOTrainingWorker


def save_checkpoint(path, model, names, epoch=340, image_size=3072):
    model.names = names
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model, 'ema': None, 'epoch': epoch, 'train_args': {'imgsz': image_size}}, path)
    return path


class StartFromModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        cls.bee = save_checkpoint(root / 'bee_recovery/weights/best.pt',
                                  SegmentationModel('yolov8n-seg.yaml', nc=1, verbose=False), {0: 'bee'})
        cls.hive = save_checkpoint(root / 'hive_run/weights/best.pt',
                                   SegmentationModel('yolov8n-seg.yaml', nc=1, verbose=False), {0: 'hive'})
        cls.boxes = save_checkpoint(root / 'bee_bbox/weights/best.pt',
                                    DetectionModel('yolov8n.yaml', nc=1, verbose=False), {0: 'bee'})

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def worker(self, **config):
        return YOLOTrainingWorker(Path(self.folder.name), dict({'model_type': 'bee', 'imgsz': 64}, **config))

    def test_the_dialog_starts_from_the_standard_weights_until_a_model_is_chosen(self):
        dialog = TrainingConfigDialog(start_from=True)
        self.assertEqual(dialog.start_combo.currentText(), STANDARD_WEIGHTS_LABEL)
        self.assertIsNone(dialog.get_config()['start_weights'])
        with patch('gui.training_dialog.QFileDialog.getOpenFileName', return_value=(str(self.bee), '')):
            dialog.choose_start_weights()
            dialog.choose_start_weights()  # the same file again is not listed twice
        self.assertEqual(dialog.start_combo.count(), 2)
        self.assertEqual(dialog.start_combo.currentText(), 'bee_recovery / best.pt')
        self.assertEqual(dialog.get_config()['start_weights'], str(self.bee))
        with patch('gui.training_dialog.QFileDialog.getOpenFileName', return_value=('', '')):
            dialog.choose_start_weights()  # cancelled: the choice stays
        self.assertEqual(dialog.get_config()['start_weights'], str(self.bee))

    def test_other_trainings_do_not_offer_it(self):
        for dialog in (TrainingConfigDialog(), TrainingConfigDialog(instance_focused=True)):
            self.assertFalse(hasattr(dialog, 'start_combo'))
            self.assertNotIn('start_weights', dialog.get_config())

    def test_a_trained_model_is_checked_and_described(self):
        self.assertIsNone(self.worker()._check_start_weights())
        self.assertEqual(self.worker(start_weights=str(self.bee))._check_start_weights(),
                         StartWeights(self.bee, 'bee_recovery', 3072, 341))

    def test_a_model_that_cannot_continue_this_training_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'is a detect model'):
            self.worker(start_weights=str(self.boxes))._check_start_weights()
        with self.assertRaisesRegex(ValueError, 'predicts hive, but this training is for bee'):
            self.worker(start_weights=str(self.hive))._check_start_weights()
        self.worker(start_weights=str(self.hive), model_type='hive')._check_start_weights()
        with self.assertRaisesRegex(FileNotFoundError, 'missing'):
            self.worker(start_weights=str(self.bee.with_name('gone.pt')))._check_start_weights()

    def test_a_refused_model_fails_before_any_data_is_prepared(self):
        worker = self.worker(start_weights=str(self.boxes))
        failures = []
        worker.training_failed.connect(failures.append)
        with patch.object(YOLOTrainingWorker, '_prepare_yolo_dataset') as prepare:
            worker.run()
        prepare.assert_not_called()
        self.assertEqual(len(failures), 1)
        self.assertIn('needs a segmentation model', failures[0])

    def train_options(self, start):
        worker = self.worker(lr0=0.0005)
        worker.start_weights = start
        notes = []
        worker.stage_update.connect(notes.append)
        with patch('training.yolo_trainer.require_training_device', return_value=(0, 'test GPU')), \
                patch('training.yolo_trainer.CarbonTrainingRun'), patch('training.yolo_trainer.YOLO') as yolo:
            yolo.return_value.train.return_value = Mock()
            worker._train_model(Path(self.folder.name) / 'dataset.yaml')
        return yolo.call_args.args[0], yolo.return_value.train.call_args.kwargs, notes

    def test_training_continues_from_the_chosen_model_without_the_bias_warm_up_jolt(self):
        weights, options, notes = self.train_options(StartWeights(self.bee, 'bee_recovery', 3072, 341))
        self.assertEqual(weights, str(self.bee))
        self.assertEqual(options['warmup_bias_lr'], 0.0005)
        self.assertEqual(options['lr0'], 0.0005)
        self.assertIn('Starting from bee_recovery / best.pt (epoch 341, trained at 3072 px), at 64 px.', notes[-1])

    def test_the_standard_weights_train_as_before(self):
        weights, options, notes = self.train_options(None)
        self.assertEqual(weights, 'yolov8s-seg.pt')
        self.assertNotIn('warmup_bias_lr', options)
        self.assertFalse(any('Starting from' in note for note in notes))


if __name__ == '__main__':
    unittest.main()
