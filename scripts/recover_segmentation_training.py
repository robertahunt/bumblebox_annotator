"""Preflight a weights-only raster-mask recovery; add --run to start it.

Only use trusted local checkpoints. The source checkpoint and exported dataset
are never edited. A recovery always requires a new output directory.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT
import yaml

from training.precision import load_checked_checkpoint
from training.raster_masks import MASK_FORMAT, RasterSegmentationTrainer, raster_training_options


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_audit(manifest_path, checkpoint_path, dataset_yaml):
    """Check the original recovery manifest without re-exporting annotations."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    backup_root = manifest_path.parent / 'epoch19_backup'
    relative_checkpoint = checkpoint_path.relative_to(backup_root).as_posix()
    if relative_checkpoint not in manifest['backup_sha256']:
        raise ValueError('The selected checkpoint is not recorded in the backup manifest.')
    for relative, expected in manifest['backup_sha256'].items():
        if sha256(backup_root / relative) != expected:
            raise ValueError(f'Backup checksum mismatch: {relative}')
    root = dataset_yaml.parent
    current = {p.relative_to(root).as_posix() for folder in ('images', 'labels')
               for p in (root / folder).rglob('*') if p.is_file()}
    current.add(dataset_yaml.name)
    if current != set(manifest['dataset_files']):
        raise ValueError('The exported dataset file list has changed; audit it before recovery.')
    for relative, expected in manifest['dataset_files'].items():
        path = root / relative
        stat = path.stat()
        if stat.st_size != expected['size']:
            raise ValueError(f'Exported dataset file size changed: {relative}')
        if 'sha256' in expected:
            if sha256(path) != expected['sha256']:
                raise ValueError(f'Exported labels/settings changed: {relative}')
        elif stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError(f'Exported image timestamp changed: {relative}')


def recovery_options(saved, *, checkpoint, data, output, epochs, batch, imgsz, lr0, device):
    if epochs < 1 or batch < 1 or (imgsz is not None and imgsz < 32) or not 0 < lr0 < 1:
        raise ValueError('Use positive epochs/batch, image size >=32, and 0 < lr0 < 1.')
    if saved.get('task') != 'segment':
        raise ValueError('Recovery requires a segmentation checkpoint.')
    options = {key: value for key, value in saved.items() if key in DEFAULT_CFG_DICT}
    # Inherit the saved augmentation/loss settings, but never its run locations
    # or resume/pretrained-file overrides. Load only the explicitly chosen model.
    options.update(
        model=str(checkpoint), data=str(data), project=str(output.parent), name=output.name,
        resume=False, pretrained=False, cfg=None, epochs=epochs, batch=batch,
        imgsz=imgsz or saved.get('imgsz', 640), device=str(device), workers=0, cache=False,
        lr0=lr0, warmup_bias_lr=min(float(saved.get('warmup_bias_lr', lr0)), lr0),
        save=True, save_period=10, exist_ok=True, time=None, half=False, compile=False,
        val=True, mode='train',
    )
    options.update(raster_training_options())
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path, help='Existing raster dataset.yaml; not re-exported.')
    parser.add_argument('--output', required=True, type=Path, help='New run directory; must not already exist.')
    parser.add_argument('--audit-manifest', type=Path, help='Optional epoch19 backup_manifest.json.')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs in the NEW run (default: 100).')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--imgsz', type=int, help='Default: original checkpoint image size.')
    parser.add_argument('--lr0', type=float, default=0.0001, help='Fresh optimizer learning rate (default: 0.0001).')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index; no CPU training fallback.')
    parser.add_argument('--run', action='store_true', help='Start training after preflight; otherwise checks only.')
    args = parser.parse_args(argv)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    data = args.data.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'Recovery output already exists; choose a new directory: {output}')
    if output == data.parent or data.parent in output.parents:
        raise ValueError('Keep recovery outputs outside the exported dataset.')
    dataset = yaml.safe_load(data.read_text())
    if dataset.get('mask_format') != MASK_FORMAT:
        raise ValueError('Expected the existing BumbleBox raster-mask dataset, not polygon YOLO labels.')
    if args.audit_manifest:
        verify_audit(args.audit_manifest, checkpoint_path, data)
    checkpoint = load_checked_checkpoint(checkpoint_path)
    saved = checkpoint['train_args']
    old_run = Path(saved.get('save_dir') or checkpoint_path.parent.parent).resolve()
    if output == old_run or old_run in output.parents:
        raise ValueError('Use a separate run directory, outside the original run.')
    options = recovery_options(saved, checkpoint=checkpoint_path, data=data, output=output,
                               epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                               lr0=args.lr0, device=args.device)
    provenance = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_checkpoint': str(checkpoint_path), 'source_sha256': sha256(checkpoint_path),
        'source_completed_epoch': checkpoint.get('epoch', -1) + 1,
        'source_ultralytics_version': checkpoint.get('version'),
        'ultralytics_version': ultralytics.__version__,
        'audit_manifest': str(args.audit_manifest.resolve()) if args.audit_manifest else None,
        'recovery': 'model weights only; fresh optimizer/scaler/schedule; no old optimizer state',
        'training_options': options,
    }
    del checkpoint
    print(f'Verified finite model weights: {checkpoint_path}')
    print(f'Original completed epoch: {provenance["source_completed_epoch"]}')
    print(f'Dataset (unchanged): {data}')
    print(f'NEW output directory: {output}')
    print(f'Fresh run: epochs={args.epochs}, batch={args.batch}, imgsz={options["imgsz"]}, lr0={args.lr0}')
    print(f'AMP training={options.get("amp", True)}; FP32 validation batch=1; FP32 checkpoints.')
    print('Masks, augmentation, and loss settings inherited; optimizer/scaler/schedule reset.')
    print(f'Bias warmup capped at {options["warmup_bias_lr"]}; periodic checkpoints every 10 epochs.')
    if not args.run:
        print('Preflight passed. No training started and no output directory created. Add --run to start.')
        return
    if args.device < 0 or not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
        raise RuntimeError('The requested CUDA device is unavailable. No CPU fallback; training not started.')
    print(f'GPU: {torch.cuda.get_device_name(args.device)}')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'recovery.json').write_text(json.dumps(provenance, indent=2) + '\n')
    model = YOLO(str(checkpoint_path))
    model.train(trainer=RasterSegmentationTrainer, **options)


if __name__ == '__main__':
    main()
