"""Training on an Apple GPU: what the app does differently there.

No Apple GPU is needed to run these. Where the Apple case cannot be made from
real tensors, stand-ins report an Apple GPU as their device.
"""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from PyQt6.QtWidgets import QApplication
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors

from core.gpu import APPLE_GPU_MEMORY_LIMITS, limit_apple_gpu_memory
from training.apple_gpu import CPU_ASSIGNMENT_DEVICES, assign_targets_on_cpu
from training.carbon_tracking import CarbonTrainingRun
from training.sam2_trainer import SAM2TrainingWorker
from training.yolo_trainer import YOLOTrainingWorker
from training.yolo_trainer_bbox import YOLOTrainingWorkerBBox
from training.yolo_trainer_instance_focused import YOLOTrainingWorkerInstanceFocused

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def matching_inputs(bees=3, size=256):
    """Scores, predicted boxes, positions, labels, boxes and validity, as the loss hands them to the match."""
    torch.manual_seed(0)
    strides = [8, 16, 32]
    features = [torch.zeros(1, 1, size // stride, size // stride) for stride in strides]
    points, stride_tensor = make_anchors(features, torch.tensor(strides, dtype=torch.float32), 0.5)
    positions = points * stride_tensor
    centre = torch.rand(bees, 2) * (size - 80) + 40
    extent = torch.rand(bees, 2) * 40 + 30
    boxes = torch.cat((centre - extent / 2, centre + extent / 2), 1)[None]
    predicted = torch.cat((positions - 20, positions + 20), 1)[None]
    scores = torch.rand(1, len(positions), 1)
    return scores, predicted, positions, torch.zeros(1, bees, 1), boxes, torch.ones(1, bees, 1)


def route_active():
    return TaskAlignedAssigner.forward.__module__ == 'training.apple_gpu'


def test_only_an_apple_gpu_is_rerouted_unless_told_otherwise():
    assert CPU_ASSIGNMENT_DEVICES == ('mps',)


def test_matching_on_an_apple_gpu_is_computed_on_the_cpu_and_returned_to_the_gpu(capsys):
    apple = Mock()
    apple.type = 'mps'
    inputs = [MagicMock(device=apple) for _ in range(6)]
    results = [MagicMock() for _ in range(5)]
    with patch.object(TaskAlignedAssigner, 'forward', return_value=tuple(results)) as original:
        with assign_targets_on_cpu():
            assigner = TaskAlignedAssigner()
            returned = assigner.forward(*inputs)
            assigner.forward(*inputs)
    assert original.call_args_list[0].args[1:] == tuple(tensor.cpu.return_value for tensor in inputs)
    for result in results:
        assert result.to.call_args_list == [((apple,),), ((apple,),)]
    assert returned == tuple(result.to.return_value for result in results)
    # Said once, however many steps follow, so the training log shows the route is in use.
    assert capsys.readouterr().out.count('Matching targets to predictions on the CPU instead of mps.') == 1


def test_nothing_is_announced_when_no_step_is_rerouted(capsys):
    assigner = TaskAlignedAssigner(topk=10, num_classes=1, alpha=0.5, beta=6.0)
    with assign_targets_on_cpu():
        assigner(*matching_inputs())
    assert 'Matching targets' not in capsys.readouterr().out


def test_matching_on_any_other_device_is_passed_through_untouched():
    assigner = TaskAlignedAssigner(topk=10, num_classes=1, alpha=0.5, beta=6.0)
    inputs = matching_inputs()
    expected = assigner(*inputs)
    with patch.object(TaskAlignedAssigner, 'forward', autospec=True,
                      side_effect=TaskAlignedAssigner.forward) as original:
        with assign_targets_on_cpu():
            result = assigner(*inputs)
    assert all(passed is given for passed, given in zip(original.call_args.args[1:], inputs))
    assert all(torch.equal(a, b) for a, b in zip(result, expected))


def test_the_cpu_route_gives_the_same_match_as_the_direct_one():
    assigner = TaskAlignedAssigner(topk=10, num_classes=1, alpha=0.5, beta=6.0)
    inputs = matching_inputs(bees=5)
    expected = assigner(*inputs)
    assert expected[3].any(), 'the example must match some positions'
    with assign_targets_on_cpu(device_types=('cpu',)):
        assert route_active()
        result = assigner(*inputs)
    assert len(result) == len(expected) == 5
    assert all(torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(result, expected))


def test_the_route_is_removed_afterwards_even_when_training_fails():
    before = TaskAlignedAssigner.forward
    with pytest.raises(RuntimeError, match='training failed'):
        with assign_targets_on_cpu():
            assert route_active()
            raise RuntimeError('training failed')
    assert TaskAlignedAssigner.forward is before and not route_active()


@pytest.mark.parametrize('module, make', [
    ('training.yolo_trainer', lambda path: YOLOTrainingWorker(path, {'imgsz': 64})),
    ('training.yolo_trainer_bbox', lambda path: YOLOTrainingWorkerBBox(path, {'imgsz': 64})),
    ('training.yolo_trainer_instance_focused', lambda path: YOLOTrainingWorkerInstanceFocused(path, {'imgsz': 64})),
])
def test_every_yolo_worker_trains_with_the_route_in_place(app, tmp_path, module, make):
    weights = tmp_path / 'run' / 'weights'
    weights.mkdir(parents=True)
    (weights / 'best.pt').touch()
    seen = []

    def train(**options):
        seen.append(route_active())
        return Mock(save_dir=str(tmp_path / 'run'))

    with patch(f'{module}.require_training_device', return_value=(0, 'test GPU')), \
            patch(f'{module}.CarbonTrainingRun'), patch(f'{module}.YOLO') as yolo:
        yolo.return_value.train.side_effect = train
        make(tmp_path)._train_model(tmp_path / 'dataset.yaml')
    assert seen == [True] and not route_active()


def test_apple_gpu_memory_stops_at_what_macos_recommends():
    environment = {}
    limit_apple_gpu_memory(environment)
    assert environment == {'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '1.0', 'PYTORCH_MPS_LOW_WATERMARK_RATIO': '0.8'}
    # PyTorch refuses to start its Apple GPU support if the lower value is above the upper one.
    assert float(environment['PYTORCH_MPS_LOW_WATERMARK_RATIO']) <= float(environment['PYTORCH_MPS_HIGH_WATERMARK_RATIO'])


@pytest.mark.parametrize('name', sorted(APPLE_GPU_MEMORY_LIMITS))
def test_a_memory_limit_someone_set_leaves_both_alone(name):
    environment = {name: '1.5'}
    limit_apple_gpu_memory(environment)
    assert environment == {name: '1.5'}


def test_the_memory_limits_are_in_place_before_the_app_loads_pytorch():
    script = (
        "import importlib.abc, os, sys\n"
        "seen = []\n"
        "class Watch(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'torch' and not seen:\n"
        "            seen.append(sorted((key, value) for key, value in os.environ.items() if key.startswith('PYTORCH_MPS')))\n"
        "sys.meta_path.insert(0, Watch())\n"
        "import main\n"
        "print('WHEN TORCH LOADED', seen[0])\n"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith('PYTORCH_MPS')}
    environment.update(QT_QPA_PLATFORM='offscreen', NO_ALBUMENTATIONS_UPDATE='1')
    done = subprocess.run([sys.executable, '-c', script], cwd=ROOT, env=environment,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    assert ("WHEN TORCH LOADED [('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '1.0'), "
            "('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.8')]") in done.stdout


def carbon_run_on(platform, sudo_exit_code, project):
    """Start and finish a tracked run as if on this platform; return (summary, sudo call, tracker class)."""
    sudo = Mock(return_value=Mock(returncode=sudo_exit_code))
    tracker = MagicMock()
    modules = {'carbontracker': MagicMock(), 'carbontracker.tracker': MagicMock(CarbonTracker=tracker)}
    with patch('training.carbon_tracking.sys', Mock(platform=platform)), \
            patch('training.carbon_tracking.subprocess.run', sudo), patch.dict(sys.modules, modules):
        run = CarbonTrainingRun(project, 'yolo', 'test')
        run.start()
        summary = run.finish()
    return summary, sudo, tracker


def test_energy_tracking_on_a_mac_never_leaves_a_password_prompt(tmp_path, capsys):
    summary, sudo, tracker = carbon_run_on('darwin', 1, tmp_path)
    command = sudo.call_args.args[0]
    assert command[:3] == ['sudo', '-n', 'powermetrics']  # -n: fail instead of asking
    assert sudo.call_args.kwargs['stdin'] == subprocess.DEVNULL
    tracker.assert_not_called()
    assert 'sudo powermetrics' in summary['carbon']['error'] and summary['carbon']['available'] is False
    assert 'Energy use not tracked' in capsys.readouterr().out


def test_energy_tracking_on_a_mac_runs_when_no_password_is_needed(tmp_path):
    summary, sudo, tracker = carbon_run_on('darwin', 0, tmp_path)
    sudo.assert_called_once()
    tracker.assert_called_once()
    tracker.return_value.epoch_start.assert_called_once()
    assert summary['carbon']['available'] is True


def test_energy_tracking_elsewhere_does_not_touch_sudo(tmp_path):
    summary, sudo, tracker = carbon_run_on('linux', 1, tmp_path)
    sudo.assert_not_called()
    tracker.assert_called_once()


@pytest.mark.parametrize('make, failing_step', [
    (lambda path: YOLOTrainingWorker(path, {}), 'training.yolo_trainer.YOLOTrainingWorker._prepare_yolo_dataset'),
    (lambda path: YOLOTrainingWorkerBBox(path, {}),
     'training.yolo_trainer_bbox.YOLOTrainingWorkerBBox._prepare_yolo_dataset'),
    (lambda path: YOLOTrainingWorkerInstanceFocused(path, {}), 'core.import_review.validate_coco_reviews'),
    (lambda path: SAM2TrainingWorker(path, {}), 'training.sam2_trainer.SAM2TrainingWorker._load_training_data'),
])
def test_a_training_failure_is_printed_in_the_terminal_as_well_as_sent_to_the_window(
        app, tmp_path, capsys, make, failing_step):
    worker = make(tmp_path)
    sent = []
    worker.training_failed.connect(sent.append)
    with patch(failing_step, side_effect=RuntimeError('no frames to train on')), \
            patch('training.sam2_trainer.SAM2_AVAILABLE', True):
        worker.run()
    printed = capsys.readouterr().err
    assert 'Training failed: no frames to train on' in printed and 'Traceback' in printed
    assert len(sent) == 1 and sent[0].startswith('Training failed: no frames to train on')
