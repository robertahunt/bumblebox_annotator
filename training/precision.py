"""Precision-safe checkpoints and validation for the pinned Ultralytics trainer.

Training still uses AMP. Validation and saved models use FP32, and optimizer
state is never compressed to FP16. No installed Ultralytics files are patched.
"""

from copy import copy, deepcopy
from datetime import datetime, timezone
import io
import math
import os
from pathlib import Path
import tempfile

import torch
import ultralytics
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils import GIT, LOGGER
from ultralytics.utils.torch_utils import unwrap_model


CHECKPOINT_FORMAT = 'bumblebox-fp32-v1'


def tensor_items(value, prefix='state'):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_items(item, f'{prefix}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_items(item, f'{prefix}[{index}]')
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f'Non-finite value in {prefix}: {value}')


def require_finite(value, label):
    for name, tensor in tensor_items(value, label):
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise FloatingPointError(
                f'Non-finite tensor in {name}. Stop and inspect the run; '
                'do not resume or replace a healthy checkpoint with this state.'
            )


def cpu_snapshot(value):
    """Copy tensor trees without quantizing or sharing storage with live state."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return deepcopy(value)


def checkpoint_model(checkpoint):
    model = checkpoint.get('ema')
    if model is None:
        model = checkpoint.get('model')
    if not isinstance(model, torch.nn.Module):
        raise ValueError('Checkpoint has no saved model or EMA model.')
    return model


def check_checkpoint(checkpoint, *, resume=False):
    require_finite(checkpoint_model(checkpoint).state_dict(), 'checkpoint model')
    if not resume:
        return
    optimizer = checkpoint.get('optimizer')
    if optimizer is None or checkpoint.get('epoch', -1) < 0:
        raise ValueError('This checkpoint has no resumable optimizer/epoch state. Use weights-only training.')
    require_finite(optimizer, 'checkpoint optimizer')
    require_finite(checkpoint.get('scaler'), 'checkpoint scaler')
    require_finite(checkpoint.get('train_metrics'), 'checkpoint metrics')
    require_finite(checkpoint.get('best_fitness'), 'checkpoint best fitness')
    for name, tensor in tensor_items(optimizer):
        if tensor.dtype in (torch.float16, torch.bfloat16):
            raise ValueError(
                f'Unsafe low-precision optimizer state: {name} ({tensor.dtype}). '
                'Use verified model weights in a NEW run with resume=False and a fresh optimizer. '
                'Casting this checkpoint to FP32 cannot recover lost precision.'
            )


def load_checked_checkpoint(path, *, resume=False):
    # Full model checkpoints use pickle. Only load files from trusted sources.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    check_checkpoint(checkpoint, resume=resume)
    return checkpoint


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class PrecisionSegmentationValidator(SegmentationValidator):
    def __call__(self, trainer=None, model=None):
        self.args.half = False
        self.args.batch = 1
        if trainer is None:
            if isinstance(model, (str, Path)):
                load_checked_checkpoint(model)
            elif isinstance(model, torch.nn.Module):
                require_finite(model.state_dict(), 'validation model')
            results = super().__call__(trainer=trainer, model=model)
        else:
            validation_model = trainer.ema.ema if trainer.ema is not None else trainer.model
            require_finite(validation_model.state_dict(), 'pre-validation model')
            # 8.4.16 ignores args.half during training and uses trainer.amp.
            # Scope this flag to validation; the training scaler is untouched.
            training_amp = trainer.amp
            trainer.amp = False
            try:
                results = super().__call__(trainer=trainer, model=model)
            finally:
                trainer.amp = training_amp
            require_finite(validation_model.state_dict(), 'post-validation model')
        require_finite(results, 'validation metrics/loss')
        return results

    def postprocess(self, preds):
        # NMS can discard invalid boxes and leave deceptively normal metrics.
        require_finite(preds, 'validation predictions')
        return super().postprocess(preds)


class PrecisionSegmentationTrainer(SegmentationTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        return super().get_dataloader(dataset_path, 1 if mode == 'val' else batch_size, rank, mode)

    def get_validator(self):
        validator = super().get_validator()
        return PrecisionSegmentationValidator(self.test_loader, save_dir=self.save_dir,
                                              args=copy(validator.args), _callbacks=self.callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, torch.nn.Module):
            require_finite(weights.state_dict(), 'initial model')
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        require_finite(model.state_dict(), 'initial model')
        return model

    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            load_checked_checkpoint(self.args.resume, resume=True)

    def _load_checkpoint_state(self, checkpoint):
        check_checkpoint(checkpoint, resume=True)
        super()._load_checkpoint_state(checkpoint)

    def run_callbacks(self, event):
        if event == 'on_train_batch_end':
            require_finite(self.loss_items, 'training batch loss')
            require_finite(self.loss, 'training batch total loss')
        elif event == 'on_train_epoch_end':
            require_finite(self.tloss, 'training epoch loss')
            require_finite(unwrap_model(self.model).state_dict(), 'training model')
        super().run_callbacks(event)

    def _handle_nan_recovery(self, epoch):
        # Fail closed instead of silently reloading an older, quantized state.
        require_finite(self.tloss, 'training epoch loss')
        require_finite(self.metrics, 'validation metrics/loss')
        require_finite(self.fitness, 'validation fitness')
        return False

    def save_model(self):
        require_finite(unwrap_model(self.model).state_dict(), 'training model')
        require_finite(self.tloss, 'training epoch loss')
        require_finite(self.metrics, 'validation metrics/loss')
        require_finite(self.fitness, 'validation fitness')
        require_finite(self.best_fitness, 'best fitness')
        model = deepcopy(unwrap_model(self.ema.ema)).cpu().float()
        if hasattr(model, 'criterion'):
            model.criterion = None
        checkpoint = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': None,
            'ema': model,
            'updates': self.ema.updates,
            'optimizer': cpu_snapshot(self.optimizer.state_dict()),
            'scaler': cpu_snapshot(self.scaler.state_dict()),
            'train_args': cpu_snapshot(vars(self.args)),
            'train_metrics': {**self.metrics, 'fitness': self.fitness},
            'train_results': self.read_results_csv(),
            'date': datetime.now(timezone.utc).isoformat(),
            'version': ultralytics.__version__,
            'git': {'root': str(GIT.root), 'branch': GIT.branch,
                    'commit': GIT.commit, 'origin': GIT.origin},
            'license': 'AGPL-3.0 (https://ultralytics.com/license)',
            'docs': 'https://docs.ultralytics.com',
            'bumblebox_checkpoint_format': CHECKPOINT_FORMAT,
        }
        check_checkpoint(checkpoint, resume=True)
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        data = buffer.getbuffer()
        atomic_write(self.last, data)
        if self.best_fitness == self.fitness:
            atomic_write(self.best, data)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            atomic_write(self.wdir / f'epoch{self.epoch}.pt', data)

    def final_eval(self):
        # Upstream strip_optimizer() halves models and destroys resume state.
        # Keep best.pt/last.pt intact; ordinary YOLO inference can load them.
        if self.best.exists():
            LOGGER.info(f'Validating FP32 checkpoint {self.best} (resume state preserved)...')
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False
            self.validator.args.augment = False
            self.metrics = self.validator(model=str(self.best))
            self.metrics.pop('fitness', None)
            self.run_callbacks('on_fit_epoch_end')
