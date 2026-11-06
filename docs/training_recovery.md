# Precision-Safe Segmentation Training

The raster-mask trainer retains AMP for training, but uses FP32 validation and
FP32 model checkpoints. AdamW optimizer tensors are saved without half-precision
compression. This is a repository-level adapter for the pinned Ultralytics
8.4.16 API, not a modification to the installed package. Label export,
augmentation, architecture, and loss functions are unchanged.

## Why

Ultralytics 8.4.16 compresses AdamW's second-moment statistics to FP16 when saving.
Small values can underflow, producing large updates after resume. See
[upstream fix #24085](https://github.com/ultralytics/ultralytics/pull/24085).
It also halves the EMA model during validation and halves saved/finalized models.
Large normalization statistics can overflow in these conversions. Converting
an already quantized or invalid checkpoint back to FP32 does not repair it.

`training.precision` protects the existing `RasterSegmentationTrainer`:

- Training AMP remains enabled when requested. Validation is FP32 with batch 1,
  including final/standalone validation. Validation AMP settings are restored
  even after exceptions. Validation can require more memory per image.
- Invalid losses, raw validation predictions, model parameters/buffers, or saved
  optimizer state stop the run with an error. Values are not silently replaced
  with zeros. Existing healthy checkpoints are not overwritten by invalid state.
- Checkpoints use CPU copies, full-precision models and uncompressed optimizer
  tensors. Each file is replaced atomically after serialization and flushing.
- `best.pt` and `last.pt` retain their optimizer/epoch state at normal completion;
  final evaluation does not strip or halve them. They still load in standard
  YOLO inference. They are larger than the old compressed/stripped files.
- Resuming low-precision optimizer state is rejected with instructions to start
  a new weights-only run. Nonfinite model state is rejected even for weights-only
  training. Use only trusted checkpoints: model files use Python pickle.

The GPU memory used for AMP training is not doubled by full-precision saving.
Saving uses additional host memory and disk space. FP32 validation has a separate
memory cost; test at the intended image resolution before a long run. The
adapter is tested on single-device CPU/CUDA; distributed recovery is not covered
by these tests.

## Recover With Verified Weights

Use a checkpoint with finite model parameters **and buffers**. A finite-looking
terminal mAP is not sufficient evidence. Preserve the old run and backups.

```bash
python scripts/recover_segmentation_training.py \
  --checkpoint /path/to/verified/weights/last.pt \
  --data /path/to/existing/yolo_format/dataset.yaml \
  --output /path/to/models/yolo_runs/bee_recovery_fp32 \
  --epochs 100 --batch 1 --imgsz 3072 --lr0 0.0001
```

This is a read-only preflight by default. Add `--run` to actually train. CUDA is
required for that command, with no CPU fallback. The output must not exist and
must be separate from the original run and dataset. Original files are never
overwritten. An optional `--audit-manifest /path/to/backup_manifest.json` verifies
the existing epoch19 recovery backup checksums and exported dataset manifest.

Recovery loads the learned model weights, not the old optimizer, scaler or
schedule (`resume=False`). Epoch numbering starts at 1. Saved augmentation/loss
settings and image size are inherited, with raster-safe transforms retained.
The defaults deliberately use batch 1, 100 new epochs, learning rate 0.0001 and
a bias warmup rate no larger than that learning rate. These are conservative
recovery settings, not an exact continuation of the previous schedule. Model
checkpoints are also retained every 10 epochs (Ultralytics filenames are
zero-indexed). `recovery.json` records the source hash and chosen settings.

## Resume A New Protected Run

Once a run has written precision-safe checkpoints, resume it using the same
custom trainer, not the standard polygon-only YOLO CLI:

```python
from ultralytics import YOLO
from training.raster_masks import RasterSegmentationTrainer

YOLO('/path/to/bee_recovery_fp32/weights/last.pt').train(
    trainer=RasterSegmentationTrainer, resume=True, device=0, workers=0,
)
```

Do not use the earlier recovery script that points to the original damaged run.
Updating packages underneath a running process is not supported. Any future
Ultralytics upgrade must be tested with the raster adapter and precision tests.

## Tests

```bash
NO_ALBUMENTATIONS_UPDATE=1 python -m unittest scripts.test_training_precision -v
```

The tests include tiny optimizer values, normalization statistics above FP16's
range, atomic-save failure, invalid loss/state rejection, AMP restoration, and
an actual two-epoch raster/AdamW CPU run interrupted after its first saved epoch
and resumed through final validation. They check optimizer equality on reload
and verify that final checkpoints remain full-precision and resumable. They do
not promise bitwise-identical resumed training trajectories; upstream loading
uses EMA weights and does not restore every data-loader/augmentation RNG state.

### Full-Resolution GPU Check (2026-11-06)

A bounded RTX A5000 test used the verified epoch-19 YOLOv8s-seg weights,
four densely labeled training images and all 37 validation images at 3072,
batch 1. After one short epoch, it saved, interrupted, resumed for one more
short epoch, and completed final validation. Mask mAP50 was 0.94488 before
resume and 0.94371 afterward; validation losses and checkpoint tensors stayed
finite. Peak PyTorch allocated GPU memory was 16.72 GiB (16.24 GiB after
resume), and each resumable checkpoint was 137.44 MiB. Allocated memory is not
total GPU usage, and this short test does not guarantee every future augmented
batch will fit. All outputs were isolated from the original run and dataset.
