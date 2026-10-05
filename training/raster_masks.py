"""Lossless instance labels for Ultralytics segmentation training and validation.

YOLO TXT polygons cannot encode holes or disconnected regions of one instance.
JSON sidecars contain COCO RLE masks; the model and checkpoint format are unchanged.
"""

import json
from copy import copy
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from ultralytics.data.augment import RandomHSV
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils.torch_utils import unwrap_model

from core.coco_masks import decode_segmentation, encode_mask


MASK_FORMAT = 'bumblebox-coco-rle-v1'


def mask_record(mask, class_id=0):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return {
        'cls': int(class_id),
        'bbox': [(x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h],
        'segmentation': encode_mask(mask),
    }


def write_mask_labels(path, shape, instances):
    with Path(path).open('w') as stream:
        json.dump({'format': MASK_FORMAT, 'shape': list(shape), 'instances': instances}, stream)


def raster_training_options():
    # These transforms use polygon-only Ultralytics code. Do not silently apply
    # them to images while leaving the raster labels unchanged.
    return dict(overlap_mask=False, mosaic=0.0, mixup=0.0, cutmix=0.0,
                copy_paste=0.0, perspective=0.0, erasing=0.0, close_mosaic=0,
                multi_scale=False)


class RasterMaskDataset(BaseDataset):
    collate_fn = staticmethod(YOLODataset.collate_fn)

    def get_labels(self):
        labels = []
        for image_path in self.im_files:
            image = Path(image_path)
            sidecar = image.parent.parent.parent / 'labels' / image.parent.name / f'{image.stem}.json'
            with sidecar.open() as stream:
                record = json.load(stream)
            if record.get('format') != MASK_FORMAT:
                raise ValueError(f'Unsupported raster label format: {sidecar}')
            instances = record['instances']
            labels.append(dict(
                im_file=image_path, shape=tuple(record['shape']),
                cls=np.array([r['cls'] for r in instances], dtype=np.float32).reshape(-1, 1),
                bboxes=np.array([r['bbox'] for r in instances], dtype=np.float32).reshape(-1, 4),
                segments=[], normalized=True, bbox_format='xywh',
                mask_sidecar=str(sidecar),
            ))
        return labels

    def build_transforms(self, hyp=None):
        if hyp.overlap_mask:
            raise ValueError('Raster masks require overlap_mask=False to preserve instance overlap')
        if getattr(hyp, 'classes', None) is not None:
            raise ValueError('Raster training datasets are already class-filtered at export')
        self.mask_ratio = max(1, int(hyp.mask_ratio))
        self.geometry = A.Compose([
            A.Affine(scale=(max(0.01, 1 - hyp.scale), 1 + hyp.scale),
                     translate_percent=(-hyp.translate, hyp.translate),
                     rotate=(-hyp.degrees, hyp.degrees), shear=(-hyp.shear, hyp.shear),
                     interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                     cval=114, cval_mask=0, mode=cv2.BORDER_CONSTANT, p=1.0),
            A.HorizontalFlip(p=hyp.fliplr), A.VerticalFlip(p=hyp.flipud),
        ]) if self.augment else None
        self.hsv = RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v)
        self.blur = A.Compose([
            A.Blur(blur_limit=(3, 15), p=0.15), A.MedianBlur(blur_limit=15, p=0.15),
        ]) if self.augment and hyp.augment else None
        return self._format_sample

    def _format_sample(self, label):
        image = label['img']
        h, w = image.shape[:2]
        with open(label['mask_sidecar']) as stream:
            record = json.load(stream)
        if tuple(record['shape']) != tuple(label['ori_shape']):
            raise ValueError(f"Raster label/image shape mismatch: {label['im_file']}")
        masks = []
        classes = []
        for instance in record['instances']:
            mask = decode_segmentation(instance['segmentation'], *record['shape'])
            masks.append(cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST))
            classes.append(0 if self.single_cls else instance['cls'])

        target_h, target_w = label.get('rect_shape', (self.imgsz, self.imgsz))
        top, left = (int(target_h) - h) // 2, (int(target_w) - w) // 2
        bottom, right = int(target_h) - h - top, int(target_w) - w - left
        if min(top, left, bottom, right) < 0:
            raise ValueError('Raster letterbox target is smaller than the resized image')
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        masks = [cv2.copyMakeBorder(m, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) for m in masks]
        if self.geometry is not None:
            # Albumentations 1.4.24's Affine.apply_to_masks delegates to the
            # image path (linear interpolation, image fill). A multi-channel
            # `mask` uses apply_to_mask and preserves nearest/zero-fill semantics.
            stacked = np.stack(masks, axis=-1) if masks else np.zeros(image.shape[:2], np.uint8)
            transformed = self.geometry(image=image, mask=stacked)
            image = transformed['image']
            if masks:
                transformed_masks = transformed['mask']
                if transformed_masks.ndim == 2:
                    transformed_masks = transformed_masks[..., None]
                masks = list(np.moveaxis(transformed_masks, -1, 0))
            image = self.hsv({'img': image})['img']
            if self.blur is not None:
                image = self.blur(image=image)['image']

        # Recompute boxes after paired geometry. Keep one target per original
        # instance, irrespective of how many disconnected regions it contains.
        h, w = image.shape[:2]
        boxes, kept_classes, targets = [], [], []
        mh, mw = h // self.mask_ratio, w // self.mask_ratio
        for class_id, mask in zip(classes, masks):
            if np.any((mask != 0) & (mask != 1)):
                raise ValueError('Raster augmentation produced non-binary instance labels')
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            boxes.append([(x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h])
            kept_classes.append(class_id)
            targets.append(cv2.resize(mask, (mw, mh), interpolation=cv2.INTER_NEAREST))
        mask_array = np.stack(targets) if targets else np.zeros((0, mh, mw), dtype=np.uint8)
        sem_mask = np.zeros((mh, mw), dtype=np.int64)
        for class_id, mask in zip(kept_classes, mask_array):
            sem_mask[mask > 0] = class_id
        return dict(
            img=torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))),
            cls=torch.tensor(kept_classes, dtype=torch.float32).reshape(-1, 1),
            bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            masks=torch.from_numpy(mask_array), sem_masks=torch.from_numpy(sem_mask),
            batch_idx=torch.zeros(len(boxes)), im_file=label['im_file'],
            ori_shape=label['ori_shape'], resized_shape=(h, w),
            ratio_pad=(label['ratio_pad'], (left, top)),
        )

    def close_mosaic(self, hyp):
        pass  # No polygon-only mixing is used in this dataset.


def build_raster_dataset(args, img_path, batch, data, mode, stride=32):
    if data.get('mask_format') != MASK_FORMAT:
        raise ValueError('Re-export training data with BumbleBox before using the raster trainer')
    return RasterMaskDataset(
        img_path=img_path, imgsz=args.imgsz, batch_size=batch,
        augment=mode == 'train', hyp=args, rect=mode == 'val', cache=False,
        single_cls=args.single_cls, stride=stride, pad=0.0 if mode == 'train' else 0.5,
        prefix=f'{mode}: ', fraction=args.fraction if mode == 'train' else 1.0,
    )


class RasterSegmentationValidator(SegmentationValidator):
    def build_dataset(self, img_path, mode='val', batch=None):
        return build_raster_dataset(self.args, img_path, batch, self.data, mode, self.stride)


class RasterSegmentationTrainer(SegmentationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()) if self.model is not None else 0, 32)
        return build_raster_dataset(self.args, img_path, batch, self.data, mode, stride)

    def get_validator(self):
        validator = super().get_validator()
        return RasterSegmentationValidator(self.test_loader, save_dir=self.save_dir,
                                           args=copy(validator.args), _callbacks=self.callbacks)
