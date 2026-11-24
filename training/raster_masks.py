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
from pycocotools import mask as mask_utils
from scipy import ndimage
from ultralytics.data.augment import RandomHSV
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.torch_utils import unwrap_model

from core.coco_masks import encode_mask, segmentation_rle
from training.precision import PrecisionSegmentationTrainer, PrecisionSegmentationValidator


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
        if self.geometry is not None and self.geometry.transforms[0].mask_interpolation != cv2.INTER_NEAREST:
            raise ValueError('Instance label images need nearest-neighbour sampling')
        self.hsv = RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v)
        self.blur = A.Compose([
            A.Blur(blur_limit=(3, 15), p=0.15), A.MedianBlur(blur_limit=15, p=0.15),
        ]) if self.augment and hyp.augment else None
        return self._format_sample

    @staticmethod
    def _instance_layers(record):
        """Draw every instance mask into a few label images at the label file's size.

        One label image holds instances that share no pixel, numbered 1, 2, ...;
        overlapping instances go into different images, so no pixel is lost. Each
        image is then resized, padded and augmented once, rather than every instance
        on a full-size canvas of its own. Nearest-neighbour sampling takes the same
        source pixel for every output pixel whatever it holds, so each instance comes
        out exactly as it would have on its own.

        Returns (label images, how many labels each holds, and per instance its
        (image, label), or None for an empty mask).
        """
        height, width = record['shape']
        layers, counts, places = [], [], []
        for instance in record['instances']:
            rle = segmentation_rle(instance['segmentation'], height, width)
            x, y, box_w, box_h = (int(v) for v in mask_utils.toBbox(rle)) if rle is not None else (0, 0, 0, 0)
            if not box_w or not box_h:
                places.append(None)
                continue
            # The box from the RLE always contains the mask, so only that part is touched.
            mask = mask_utils.decode(rle)[y:y + box_h, x:x + box_w] > 0
            for index, layer in enumerate(layers):
                region = layer[y:y + box_h, x:x + box_w]
                if counts[index] < 255 and not region[mask].any():
                    break
            else:
                layers.append(np.zeros((height, width), np.uint8))
                counts.append(0)
                index, region = len(layers) - 1, layers[-1][y:y + box_h, x:x + box_w]
            counts[index] += 1
            region[mask] = counts[index]
            places.append((index, counts[index]))
        return layers, counts, places

    def _format_sample(self, label):
        image = label['img']
        h, w = image.shape[:2]
        with open(label['mask_sidecar']) as stream:
            record = json.load(stream)
        if tuple(record['shape']) != tuple(label['ori_shape']):
            raise ValueError(f"Raster label/image shape mismatch: {label['im_file']}")
        layers, counts, places = self._instance_layers(record)
        layers = [cv2.resize(layer, (w, h), interpolation=cv2.INTER_NEAREST) for layer in layers]
        classes = [0 if self.single_cls else instance['cls'] for instance in record['instances']]

        target_h, target_w = label.get('rect_shape', (self.imgsz, self.imgsz))
        top, left = (int(target_h) - h) // 2, (int(target_w) - w) // 2
        bottom, right = int(target_h) - h - top, int(target_w) - w - left
        if min(top, left, bottom, right) < 0:
            raise ValueError('Raster letterbox target is smaller than the resized image')
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        layers = [cv2.copyMakeBorder(m, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) for m in layers]
        if self.geometry is not None:
            # Albumentations 1.4.24's Affine.apply_to_masks delegates to the
            # image path (linear interpolation, image fill). A multi-channel
            # `mask` uses apply_to_mask and preserves nearest/zero-fill semantics.
            stacked = np.stack(layers, axis=-1) if layers else np.zeros(image.shape[:2], np.uint8)
            transformed = self.geometry(image=image, mask=stacked)
            image = transformed['image']
            if layers:
                moved = transformed['mask']
                if moved.ndim == 2:
                    moved = moved[..., None]
                layers = list(np.moveaxis(moved, -1, 0))
            image = self.hsv({'img': image})['img']
            if self.blur is not None:
                image = self.blur(image=image)['image']

        # Recompute boxes after paired geometry. Keep one target per original
        # instance, irrespective of how many disconnected regions it contains.
        h, w = image.shape[:2]
        mh, mw = h // self.mask_ratio, w // self.mask_ratio
        extents, reduced = [], []
        for layer, count in zip(layers, counts):
            if layer.max(initial=0) > count:
                raise ValueError('Raster augmentation produced non-binary instance labels')
            extents.append(ndimage.find_objects(layer, max_label=count))  # each label's exact box
            reduced.append(cv2.resize(layer, (mw, mh), interpolation=cv2.INTER_NEAREST))
        boxes, kept_classes, targets = [], [], []
        for class_id, place in zip(classes, places):
            found = extents[place[0]][place[1] - 1] if place else None
            if found is None:
                continue  # an empty mask, or one moved out of the frame
            (y1, y2), (x1, x2) = ((part.start, part.stop) for part in found)
            boxes.append([(x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h])
            kept_classes.append(class_id)
            targets.append((reduced[place[0]] == place[1]).astype(np.uint8))
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


class RasterSegmentationValidator(PrecisionSegmentationValidator):
    def build_dataset(self, img_path, mode='val', batch=None):
        return build_raster_dataset(self.args, img_path, batch, self.data, mode, self.stride)


class RasterSegmentationTrainer(PrecisionSegmentationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        stride = max(int(unwrap_model(self.model).stride.max()) if self.model is not None else 0, 32)
        return build_raster_dataset(self.args, img_path, batch, self.data, mode, stride)

    def get_validator(self):
        validator = super().get_validator()
        return RasterSegmentationValidator(self.test_loader, save_dir=self.save_dir,
                                           args=copy(validator.args), _callbacks=self.callbacks)
