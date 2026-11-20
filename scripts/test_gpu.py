#!/usr/bin/env python3
"""Models must run on an NVIDIA card, an Apple GPU or the CPU, whichever the computer has.

No Apple GPU is needed to run these: the Apple cases replace PyTorch's answers
about the hardware, and the real-model case forces the CPU on whatever computer
runs it, which exercises the same "use the device the model was placed on" route.
"""

import os
import sys
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from core import gpu
from training.gpu import require_training_device


class HardwareCase(unittest.TestCase):
    """Pretend to be a computer with the given hardware."""

    def hardware(self, *, cuda=False, apple=False, force_cpu=False):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('torch.cuda.is_available', return_value=cuda))
        stack.enter_context(patch('torch.backends.mps.is_available', return_value=apple))
        environment = {gpu.FORCE_CPU_VARIABLE: '1'} if force_cpu else {}
        stack.enter_context(patch.dict(os.environ, environment))
        if not force_cpu:
            os.environ.pop(gpu.FORCE_CPU_VARIABLE, None)
        return stack


class DeviceChoiceTests(HardwareCase):
    def test_an_nvidia_card_comes_first_then_an_apple_gpu_then_the_cpu(self):
        for cuda, apple, expected in ((True, True, 'cuda'), (True, False, 'cuda'),
                                      (False, True, 'mps'), (False, False, 'cpu')):
            with self.subTest(cuda=cuda, apple=apple), self.hardware(cuda=cuda, apple=apple):
                self.assertEqual(gpu.device_name(), expected)

    def test_the_cpu_can_be_forced_on_any_computer(self):
        for cuda, apple in ((True, False), (False, True)):
            with self.subTest(cuda=cuda, apple=apple), self.hardware(cuda=cuda, apple=apple, force_cpu=True):
                self.assertEqual(gpu.device_name(), 'cpu')

    def test_only_a_clear_yes_forces_the_cpu(self):
        for value, forced in (('1', True), ('true', True), (' Yes ', True), ('0', False), ('', False), ('no', False)):
            with self.subTest(value=value), patch.dict(os.environ, {gpu.FORCE_CPU_VARIABLE: value}):
                self.assertEqual(gpu.cpu_forced(), forced)


def loaded_model(network=None):
    """Stand in for what YOLO(path) returns: a wrapper whose .model is the network, or a file name."""
    model = MagicMock()
    model.model = torch.nn.Linear(1, 1) if network is None else network
    return model


class ModelPlacementTests(HardwareCase):
    def test_a_model_on_an_nvidia_computer_is_left_as_loaded(self):
        model = loaded_model()
        with self.hardware(cuda=True):
            self.assertIs(gpu.place_model(model), model)
        model.to.assert_not_called()

    def test_a_model_is_moved_to_the_apple_gpu(self):
        model = loaded_model()
        with self.hardware(apple=True):
            self.assertIs(gpu.place_model(model), model)
        model.to.assert_called_once_with('mps')

    def test_a_forced_cpu_is_stated_even_when_an_nvidia_card_is_there(self):
        model = loaded_model()
        with self.hardware(cuda=True, force_cpu=True):
            gpu.place_model(model)
        model.to.assert_called_once_with('cpu')

    def test_an_exported_model_is_returned_as_loaded(self):
        # Ultralytics refuses to move ONNX and similar files; they keep its own device choice.
        model = loaded_model('weights.onnx')
        with self.hardware(apple=True):
            self.assertIs(gpu.place_model(model), model)
        model.to.assert_not_called()

    def test_a_stand_in_that_is_not_a_model_is_returned_untouched(self):
        stand_in = object()
        with self.hardware(apple=True):
            self.assertIs(gpu.place_model(stand_in), stand_in)

    def test_a_placed_model_predicts_and_tracks_on_its_device(self):
        # Ultralytics would pick an NVIDIA card here if one is present; the
        # forced CPU must win, by the same route an Apple GPU is chosen.
        from ultralytics import YOLO
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch.dict(os.environ, {gpu.FORCE_CPU_VARIABLE: '1'}):
            model = gpu.place_model(YOLO('yolov8n-seg.yaml'))
            model.predict(frame, verbose=False, imgsz=64)
            self.assertEqual(model.predictor.device.type, 'cpu')
            model.track(frame, persist=True, verbose=False, imgsz=64)
            self.assertEqual(model.predictor.device.type, 'cpu')


class MemoryTests(HardwareCase):
    def test_clean_up_uses_the_gpu_the_computer_has(self):
        for cuda, apple, cuda_calls, apple_calls in ((True, False, 1, 0), (False, True, 0, 1), (False, False, 0, 0)):
            with self.subTest(cuda=cuda, apple=apple), self.hardware(cuda=cuda, apple=apple), \
                    patch('torch.cuda.empty_cache') as cuda_cache, patch('torch.mps.empty_cache') as apple_cache, \
                    patch('torch.cuda.synchronize') as cuda_sync, patch('torch.mps.synchronize') as apple_sync:
                gpu.release_gpu_memory()
                gpu.synchronize_gpu()
                self.assertEqual((cuda_cache.call_count, cuda_sync.call_count), (cuda_calls, cuda_calls))
                self.assertEqual((apple_cache.call_count, apple_sync.call_count), (apple_calls, apple_calls))

    def test_memory_in_use_is_reported_for_either_gpu_and_absent_without_one(self):
        with self.hardware():
            self.assertIsNone(gpu.gpu_memory_in_use())
        with self.hardware(cuda=True), patch('torch.cuda.memory_allocated', return_value=3), \
                patch('torch.cuda.memory_reserved', return_value=5):
            self.assertEqual(gpu.gpu_memory_in_use(), (3, 5))
        with self.hardware(apple=True), patch('torch.mps.current_allocated_memory', return_value=7), \
                patch('torch.mps.driver_allocated_memory', return_value=11):
            self.assertEqual(gpu.gpu_memory_in_use(), (7, 11))

    def test_batch_video_log_text_leaves_out_the_peak_an_apple_gpu_does_not_keep(self):
        from core.batch_video_processor import BatchVideoProcessor
        processor = BatchVideoProcessor.__new__(BatchVideoProcessor)
        with self.hardware(apple=True), patch('torch.mps.current_allocated_memory', return_value=2e9), \
                patch('torch.mps.driver_allocated_memory', return_value=3e9):
            self.assertEqual(processor._gpu_memory_text(), ", GPU 2.00 GB allocated/3.00 GB reserved")
        with self.hardware():
            self.assertEqual(processor._gpu_memory_text(), "")


class TrainingDeviceTests(HardwareCase):
    def test_an_nvidia_card_gives_its_index(self):
        with self.hardware(cuda=True), patch('torch.cuda.device_count', return_value=1), \
                patch('torch.cuda.set_device'), patch('torch.cuda.get_device_name', return_value='Test card'), \
                patch('torch.cuda.get_device_properties', return_value=MagicMock(total_memory=24 * 1024 ** 3)):
            self.assertEqual(require_training_device(), (0, 'Test card (24.0 GB)'))

    def test_an_apple_gpu_gives_the_name_ultralytics_expects(self):
        with self.hardware(apple=True):
            device, label = require_training_device()
        self.assertEqual(device, 'mps')
        self.assertIn('Apple GPU', label)

    def test_training_still_refuses_to_run_on_the_cpu(self):
        with self.hardware(), patch('training.gpu.sys.platform', 'linux'):
            with self.assertRaisesRegex(RuntimeError, 'cannot access CUDA'):
                require_training_device()
        with self.hardware(), patch('training.gpu.sys.platform', 'darwin'):
            with self.assertRaisesRegex(RuntimeError, 'cannot access the Apple GPU'):
                require_training_device()

    def test_training_refuses_while_the_cpu_is_forced(self):
        for cuda, apple in ((True, False), (False, True)):
            with self.subTest(cuda=cuda, apple=apple), self.hardware(cuda=cuda, apple=apple, force_cpu=True):
                with self.assertRaisesRegex(RuntimeError, gpu.FORCE_CPU_VARIABLE):
                    require_training_device()

    def test_yolo_trainers_hand_the_apple_gpu_to_ultralytics(self):
        import tempfile
        from training.yolo_trainer import YOLOTrainingWorker
        from training.yolo_trainer_bbox import YOLOTrainingWorkerBBox
        from training.yolo_trainer_instance_focused import YOLOTrainingWorkerInstanceFocused
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'run'
            (run / 'weights').mkdir(parents=True)
            (run / 'weights' / 'best.pt').write_bytes(b'')
            for module, worker_class in (('training.yolo_trainer', YOLOTrainingWorker),
                                         ('training.yolo_trainer_bbox', YOLOTrainingWorkerBBox),
                                         ('training.yolo_trainer_instance_focused', YOLOTrainingWorkerInstanceFocused)):
                with self.subTest(module=module), \
                        patch(f'{module}.require_training_device', return_value=('mps', 'Apple GPU')), \
                        patch(f'{module}.YOLO') as yolo:
                    yolo.return_value.train.return_value.save_dir = run
                    worker_class(root, {})._train_model(root / 'dataset.yaml')
                    self.assertEqual(yolo.return_value.train.call_args.kwargs['device'], 'mps')


class Sam2DeviceTests(HardwareCase):
    def test_the_app_builds_both_sam2_models_on_the_chosen_device(self):
        from core.sam2_integrator import SAM2Integrator
        for cuda, apple, expected in ((True, False, 'cuda'), (False, True, 'mps'), (False, False, 'cpu')):
            with self.subTest(expected=expected), self.hardware(cuda=cuda, apple=apple), \
                    patch('sam2.build_sam.build_sam2') as image_model, \
                    patch('sam2.build_sam.build_sam2_video_predictor') as video_model, \
                    patch('sam2.sam2_image_predictor.SAM2ImagePredictor'):
                integrator = SAM2Integrator('checkpoint.pt')
                self.assertEqual(integrator.device, expected)
                self.assertEqual(image_model.call_args.kwargs['device'], expected)
                self.assertEqual(video_model.call_args.kwargs['device'], expected)

    def test_a_device_given_by_the_caller_is_passed_on(self):
        from core.sam2_integrator import SAM2Integrator
        with self.hardware(cuda=True), patch('sam2.build_sam.build_sam2') as image_model, \
                patch('sam2.build_sam.build_sam2_video_predictor') as video_model, \
                patch('sam2.sam2_image_predictor.SAM2ImagePredictor'):
            SAM2Integrator('checkpoint.pt', device='cpu')
        self.assertEqual(image_model.call_args.kwargs['device'], 'cpu')
        self.assertEqual(video_model.call_args.kwargs['device'], 'cpu')

    def test_sam2_training_builds_the_model_on_the_training_device(self):
        from training import sam2_trainer
        for device, expected in ((0, 'cuda'), ('mps', 'mps')):
            worker = sam2_trainer.SAM2TrainingWorker.__new__(sam2_trainer.SAM2TrainingWorker)
            worker.config = {'checkpoint_path': 'checkpoint.pt'}
            worker.stage_update = MagicMock()
            with self.subTest(device=device), \
                    patch('training.sam2_trainer.require_training_device', return_value=(device, 'test GPU')), \
                    patch('training.sam2_trainer.build_sam2') as build, \
                    patch('training.sam2_trainer.SAM2ImagePredictor'):
                worker._load_sam2_model()
                self.assertEqual(build.call_args.kwargs['device'], expected)

    def test_sam2_training_uses_mixed_precision_only_on_an_nvidia_card(self):
        from training.sam2_trainer import mixed_precision
        self.assertIsInstance(mixed_precision(torch.device('cuda')), torch.autocast)
        self.assertIsInstance(mixed_precision(torch.device('mps')), nullcontext)
        # With the scaler off, as on an Apple GPU, a step is an ordinary full-precision step.
        weight = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([weight], lr=0.5)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        scaler.scale(weight * 2).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        self.assertEqual(weight.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
