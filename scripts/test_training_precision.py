"""Numerical safety and actual save/interrupt/resume tests for raster training."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator

from training.precision import (
    CHECKPOINT_FORMAT, PrecisionSegmentationTrainer, PrecisionSegmentationValidator,
    atomic_write, check_checkpoint, cpu_snapshot, load_checked_checkpoint,
)
from training.raster_masks import (
    MASK_FORMAT, RasterSegmentationTrainer, RasterSegmentationValidator,
    mask_record, raster_training_options, write_mask_labels,
)
from scripts.recover_segmentation_training import main as recover_main, recovery_options, sha256, verify_audit


class CheckpointPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2))
        self.model[1].running_var.fill_(1e6)
        optimizer = torch.optim.AdamW(self.model.parameters())
        for parameter in self.model.parameters():
            optimizer.state[parameter] = {
                'step': torch.tensor(1.0),
                'exp_avg': torch.full_like(parameter, 1e-7),
                'exp_avg_sq': torch.full_like(parameter, 1e-12),
            }
        self.trainer = PrecisionSegmentationTrainer.__new__(PrecisionSegmentationTrainer)
        self.trainer.model = self.model
        self.trainer.ema = SimpleNamespace(ema=deepcopy(self.model), updates=3)
        self.trainer.optimizer = optimizer
        self.trainer.scaler = SimpleNamespace(state_dict=lambda: {'scale': 65536.0})
        self.trainer.epoch = 0
        self.trainer.tloss = torch.ones(5)
        self.trainer.fitness = self.trainer.best_fitness = 0.7
        self.trainer.metrics = {'val/seg_loss': 1.0}
        self.trainer.args = SimpleNamespace(epochs=2)
        self.trainer.wdir = self.root / 'weights'
        self.trainer.last = self.trainer.wdir / 'last.pt'
        self.trainer.best = self.trainer.wdir / 'best.pt'
        self.trainer.save_period = 1
        self.trainer.read_results_csv = lambda: {}

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_preserves_tiny_moments_and_large_normalization_stats(self):
        before = deepcopy(self.trainer.optimizer.state_dict())
        self.trainer.save_model()
        for path in (self.trainer.last, self.trainer.best, self.trainer.wdir / 'epoch0.pt'):
            checkpoint = load_checked_checkpoint(path, resume=True)
            self.assertEqual(checkpoint['bumblebox_checkpoint_format'], CHECKPOINT_FORMAT)
            for key, value in self.trainer.ema.ema.state_dict().items():
                torch.testing.assert_close(checkpoint['ema'].state_dict()[key], value, rtol=0, atol=0)
            for key, state in before['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(checkpoint['optimizer']['state'][key][name], value, rtol=0, atol=0)
            checkpoint['ema'].eval()
            self.trainer.ema.ema.eval()
            image = torch.ones(1, 3, 8, 8)
            torch.testing.assert_close(checkpoint['ema'](image), self.trainer.ema.ema(image), rtol=0, atol=0)
        self.assertEqual(self.model[1].running_var.dtype, torch.float32)

    def test_cpu_snapshot_does_not_alias_live_state(self):
        original = {'items': [torch.tensor([1e-12])], 'tuple': (torch.tensor(3.),)}
        snapshot = cpu_snapshot(original)
        snapshot['items'][0].zero_()
        self.assertNotEqual(original['items'][0].item(), 0)
        self.assertIsInstance(snapshot['tuple'], tuple)

    def test_legacy_optimizer_rejected_but_weights_only_allowed(self):
        self.trainer.save_model()
        checkpoint = load_checked_checkpoint(self.trainer.last)
        for state in checkpoint['optimizer']['state'].values():
            state['exp_avg_sq'] = state['exp_avg_sq'].half()
        check_checkpoint(checkpoint)
        with self.assertRaisesRegex(ValueError, 'fresh optimizer'):
            check_checkpoint(checkpoint, resume=True)

    def test_corrupt_state_never_overwrites_previous_checkpoints(self):
        self.trainer.save_model()
        before = {path: path.read_bytes() for path in (self.trainer.last, self.trainer.best)}
        corruptions = [
            lambda: self.trainer.ema.ema[1].running_var.fill_(float('inf')),
            lambda: self.trainer.metrics.update({'val/box_loss': float('nan')}),
            lambda: next(iter(self.trainer.optimizer.state.values()))['exp_avg_sq'].fill_(float('nan')),
            lambda: self.trainer.tloss.fill_(float('nan')),
        ]
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                corrupt()
                with self.assertRaises(FloatingPointError):
                    self.trainer.save_model()
                for path, contents in before.items():
                    self.assertEqual(path.read_bytes(), contents)
                self.trainer.ema.ema[1].running_var.fill_(1e6)
                self.trainer.metrics = {'val/seg_loss': 1.0}
                for state in self.trainer.optimizer.state.values():
                    state['exp_avg_sq'].fill_(1e-12)
                self.trainer.tloss.fill_(1)

    def test_atomic_write_failure_preserves_file_and_removes_temporary(self):
        path = self.root / 'last.pt'
        path.write_bytes(b'healthy')
        with patch('training.precision.os.replace', side_effect=OSError('disk problem')):
            with self.assertRaises(OSError):
                atomic_write(path, b'new')
        self.assertEqual(path.read_bytes(), b'healthy')
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_nonfinite_model_rejected_even_for_weights_only(self):
        self.trainer.save_model()
        checkpoint = load_checked_checkpoint(self.trainer.last)
        checkpoint['ema'][1].running_mean.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):
            check_checkpoint(checkpoint)

    def test_nonfinite_best_fitness_rejected_for_resume(self):
        self.trainer.save_model()
        checkpoint = load_checked_checkpoint(self.trainer.last)
        checkpoint['best_fitness'] = float('nan')
        with self.assertRaises(FloatingPointError):
            check_checkpoint(checkpoint, resume=True)

    def test_batch_nan_stops_before_user_callback(self):
        callback = Mock()
        self.trainer.callbacks = {'on_train_batch_end': [callback]}
        self.trainer.loss_items = torch.tensor([float('nan')])
        self.trainer.loss = torch.tensor(1.)
        with self.assertRaises(FloatingPointError):
            self.trainer.run_callbacks('on_train_batch_end')
        callback.assert_not_called()

    def test_validation_nan_cannot_fall_through_upstream_recovery(self):
        self.trainer.metrics['val/box_loss'] = float('nan')
        with self.assertRaises(FloatingPointError):
            self.trainer._handle_nan_recovery(0)

    def test_validation_batch_is_one_training_batch_unchanged(self):
        with patch.object(SegmentationTrainer, 'get_dataloader', return_value='loader') as loader:
            self.trainer.get_dataloader('val', 8, -1, 'val')
            loader.assert_called_once_with('val', 1, -1, 'val')
            loader.reset_mock()
            self.trainer.get_dataloader('train', 8, -1, 'train')
            loader.assert_called_once_with('train', 8, -1, 'train')

    def test_validator_uses_fp32_and_restores_amp_on_success_and_failure(self):
        validator = PrecisionSegmentationValidator(
            save_dir=self.root / 'validation', args=get_cfg(overrides={'task': 'segment', 'half': True}))
        trainer = SimpleNamespace(amp=True, ema=self.trainer.ema)

        def validate(**kwargs):
            self.assertFalse(kwargs['trainer'].amp)
            self.assertFalse(validator.args.half)
            self.assertEqual(validator.args.batch, 1)
            return {'val/seg_loss': 1.}

        with patch.object(SegmentationValidator, '__call__', side_effect=validate):
            validator(trainer=trainer)
        self.assertTrue(trainer.amp)
        with patch.object(SegmentationValidator, '__call__', side_effect=RuntimeError('failed')):
            with self.assertRaises(RuntimeError):
                validator(trainer=trainer)
        self.assertTrue(trainer.amp)
        with patch.object(SegmentationValidator, '__call__', return_value={'val/box_loss': float('nan')}):
            with self.assertRaises(FloatingPointError):
                validator(trainer=trainer)
        self.assertTrue(trainer.amp)

    def test_validator_rejects_bad_predictions_before_nms(self):
        validator = PrecisionSegmentationValidator(
            save_dir=self.root / 'validation', args=get_cfg(overrides={'task': 'segment'}))
        with patch.object(SegmentationValidator, 'postprocess') as postprocess:
            with self.assertRaises(FloatingPointError):
                validator.postprocess({'boxes': torch.tensor([float('nan')])})
            postprocess.assert_not_called()


class RecoveryCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data' / 'dataset.yaml'
        self.data.parent.mkdir()
        self.data.write_text(yaml.safe_dump({'mask_format': MASK_FORMAT, 'names': {0: 'bee'}}))
        self.source = self.root / 'epoch19_backup' / 'weights' / 'last.pt'
        self.source.parent.mkdir(parents=True)
        self.saved = {'task': 'segment', 'imgsz': 3072, 'optimizer': 'AdamW', 'amp': True,
                      'degrees': 20.0, 'lr0': 0.001, 'warmup_bias_lr': 0.1,
                      'save_dir': str(self.root / 'old_run'), 'data': str(self.data)}
        torch.save({'ema': torch.nn.Linear(2, 2), 'epoch': 18, 'train_args': self.saved,
                    'optimizer': {'state': {'bad_for_resume': torch.ones(1).half()}}}, self.source)
        self.output = self.root / 'new_run'
        self.argv = ['--checkpoint', str(self.source), '--data', str(self.data), '--output', str(self.output)]

    def tearDown(self):
        self.temp.cleanup()

    def test_preflight_is_read_only_and_does_not_require_cuda(self):
        before = sha256(self.source)
        with patch('scripts.recover_segmentation_training.YOLO') as yolo, \
                patch('torch.cuda.is_available') as cuda:
            recover_main(self.argv)
        yolo.assert_not_called()
        cuda.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertEqual(sha256(self.source), before)

    def test_run_loads_weights_only_and_records_explicit_changes(self):
        before = sha256(self.source)
        with patch('scripts.recover_segmentation_training.YOLO') as yolo, \
                patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=1), \
                patch('torch.cuda.get_device_name', return_value='test GPU'):
            recover_main(self.argv + ['--run'])
        options = yolo.return_value.train.call_args.kwargs
        self.assertIs(options['trainer'], RasterSegmentationTrainer)
        self.assertFalse(options['resume'])
        self.assertFalse(options['pretrained'])
        self.assertEqual(options['lr0'], 0.0001)
        self.assertEqual(options['warmup_bias_lr'], 0.0001)
        self.assertEqual(options['degrees'], 20)
        self.assertEqual(options['imgsz'], 3072)
        self.assertEqual(options['batch'], 1)
        self.assertEqual(options['epochs'], 100)
        self.assertEqual(sha256(self.source), before)
        record = json.loads((self.output / 'recovery.json').read_text())
        self.assertEqual(record['source_sha256'], before)
        self.assertEqual(record['source_completed_epoch'], 19)

    def test_unavailable_cuda_does_not_create_run(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'No CPU fallback'):
                recover_main(self.argv + ['--run'])
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_reused(self):
        self.output.mkdir()
        marker = self.output / 'results.csv'
        marker.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            recover_main(self.argv)
        self.assertEqual(marker.read_text(), 'keep')

    def test_nonfinite_weights_are_rejected(self):
        checkpoint = torch.load(self.source, weights_only=False)
        with torch.no_grad():
            checkpoint['ema'].weight.fill_(float('nan'))
        torch.save(checkpoint, self.source)
        with self.assertRaises(FloatingPointError):
            recover_main(self.argv)
        self.assertFalse(self.output.exists())

    def test_audit_detects_changed_labels(self):
        labels = self.data.parent / 'labels'
        labels.mkdir()
        label = labels / '0.json'
        label.write_text('{}')
        manifest = {
            'backup_sha256': {'weights/last.pt': sha256(self.source)},
            'dataset_files': {p.relative_to(self.data.parent).as_posix():
                              {'size': p.stat().st_size, 'sha256': sha256(p)} for p in (self.data, label)},
        }
        path = self.root / 'backup_manifest.json'
        path.write_text(json.dumps(manifest))
        verify_audit(path, self.source, self.data)
        label.write_text('[]')
        with self.assertRaisesRegex(ValueError, 'labels/settings changed'):
            verify_audit(path, self.source, self.data)

    def test_invalid_learning_rate_and_batch_rejected(self):
        for batch, lr in ((0, 0.0001), (1, float('nan')), (1, 0.0)):
            with self.assertRaises(ValueError):
                recovery_options(self.saved, checkpoint=self.source, data=self.data, output=self.output,
                                 epochs=100, batch=batch, imgsz=3072, lr0=lr, device=0)


class InterruptedRasterTrainingTests(unittest.TestCase):
    def test_real_adamw_interrupt_resume_and_final_checkpoints(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                (root / 'images' / split).mkdir(parents=True)
                (root / 'labels' / split).mkdir(parents=True)
                for index in range(2):
                    image = np.full((64, 64, 3), 120, np.uint8)
                    mask = np.zeros((64, 64), np.uint8)
                    mask[10:50, 10:50] = 1
                    mask[20:30, 20:30] = 0
                    mask[55:60, 55:60] = 1
                    cv2.imwrite(str(root / 'images' / split / f'{index}.png'), image)
                    write_mask_labels(root / 'labels' / split / f'{index}.json', mask.shape, [mask_record(mask)])
            data = root / 'dataset.yaml'
            data.write_text(yaml.safe_dump({'path': str(root), 'train': 'images/train', 'val': 'images/val',
                                            'names': {0: 'bee'}, 'nc': 1, 'mask_format': MASK_FORMAT}))

            class Interrupted(Exception):
                pass

            def interrupt(trainer):
                raise Interrupted('Simulated interrupt after a completed checkpoint')

            with patch('ultralytics.engine.model.checks.check_pip_update_available', return_value=False), \
                    patch('ultralytics.data.utils.check_font'):
                model = YOLO('yolov8n-seg.yaml')
                model.add_callback('on_model_save', interrupt)
                with self.assertRaises(Interrupted):
                    model.train(trainer=RasterSegmentationTrainer, data=str(data), epochs=2,
                                imgsz=64, batch=2, workers=0, device='cpu', amp=False, pretrained=False,
                                optimizer='AdamW', lr0=0.001, warmup_epochs=0, nbs=2, plots=False,
                                project=str(root / 'runs'), name='resume', **raster_training_options())
                path = model.trainer.last
                first = load_checked_checkpoint(path, resume=True)
                self.assertEqual(first['epoch'], 0)
                self.assertTrue(first['optimizer']['state'])
                self.assertEqual(model.trainer.test_loader.batch_size, 1)
                saved_values = first['optimizer']['state']
                resumed = YOLO(str(path))
                observed = []

                def check_restored(trainer):
                    self.assertEqual(trainer.start_epoch, 1)
                    restored = trainer.optimizer.state_dict()['state']
                    for key, state in saved_values.items():
                        for name, value in state.items():
                            torch.testing.assert_close(restored[key][name], value, rtol=0, atol=0)
                    observed.append(True)

                resumed.add_callback('on_pretrain_routine_end', check_restored)
                resumed.train(trainer=RasterSegmentationTrainer, resume=True, device='cpu', workers=0, plots=False)
                self.assertEqual(observed, [True])
                last = load_checked_checkpoint(path, resume=True)
                self.assertEqual(last['epoch'], 1)
                self.assertIsNotNone(last['optimizer'])
                self.assertIsNone(last['model'])
                self.assertFalse(resumed.trainer.validator.args.augment)
                best = load_checked_checkpoint(resumed.trainer.best, resume=True)
                self.assertEqual(next(best['ema'].parameters()).dtype, torch.float32)
                reloaded = YOLO(str(resumed.trainer.best))
                metrics = reloaded.val(validator=RasterSegmentationValidator, data=str(data), device='cpu',
                                       imgsz=64, batch=2, half=True, overlap_mask=False, plots=False,
                                       project=str(root / 'runs'), name='validation')
                self.assertTrue(np.isfinite(metrics.seg.map))


if __name__ == '__main__':
    unittest.main()
