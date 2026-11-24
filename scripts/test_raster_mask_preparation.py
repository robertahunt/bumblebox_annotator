"""Faster training-frame preparation must give exactly the training samples the earlier code gave.

The earlier code prepared every instance mask on a full-size canvas of its own. It is
kept below, unchanged, as the reference. Every case compares the two sample by sample
with the same random augmentation, including overlapping, holed, split, empty and
off-frame instances, more instances than one label image holds, resized frames and
the validation path.
"""

import json
from copy import deepcopy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from ultralytics.cfg import get_cfg

from core.coco_masks import decode_segmentation, encode_mask
from training.raster_masks import RasterMaskDataset, mask_record, raster_training_options, write_mask_labels


def reference_format_sample(dataset, label):
    """training/raster_masks.py _format_sample as of commit 7678525, verbatim apart from `self`."""
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
        classes.append(0 if dataset.single_cls else instance['cls'])

    target_h, target_w = label.get('rect_shape', (dataset.imgsz, dataset.imgsz))
    top, left = (int(target_h) - h) // 2, (int(target_w) - w) // 2
    bottom, right = int(target_h) - h - top, int(target_w) - w - left
    if min(top, left, bottom, right) < 0:
        raise ValueError('Raster letterbox target is smaller than the resized image')
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    masks = [cv2.copyMakeBorder(m, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) for m in masks]
    if dataset.geometry is not None:
        stacked = np.stack(masks, axis=-1) if masks else np.zeros(image.shape[:2], np.uint8)
        transformed = dataset.geometry(image=image, mask=stacked)
        image = transformed['image']
        if masks:
            transformed_masks = transformed['mask']
            if transformed_masks.ndim == 2:
                transformed_masks = transformed_masks[..., None]
            masks = list(np.moveaxis(transformed_masks, -1, 0))
        image = dataset.hsv({'img': image})['img']
        if dataset.blur is not None:
            image = dataset.blur(image=image)['image']

    h, w = image.shape[:2]
    boxes, kept_classes, targets = [], [], []
    mh, mw = h // dataset.mask_ratio, w // dataset.mask_ratio
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


def blob(shape, centre, axes, angle=0, hole=None, extra=None):
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, centre, axes, angle, 0, 360, 1, -1)
    if hole:
        cv2.circle(mask, hole[0], hole[1], 0, -1)
    if extra:  # a second, disconnected piece of the same instance
        cv2.circle(mask, extra[0], extra[1], 1, -1)
    return mask


def crowded_frame(shape=(220, 300), count=40, seed=0):
    """Bee-like instances: many overlapping, some holed or in two pieces, some at the frame's edge."""
    generator = np.random.default_rng(seed)
    h, w = shape
    masks = []
    for index in range(count):
        centre = (int(generator.integers(-5, w + 5)), int(generator.integers(-5, h + 5)))
        axes = (int(generator.integers(4, 22)), int(generator.integers(3, 12)))
        hole = ((centre[0], centre[1]), 2) if index % 7 == 0 else None
        extra = ((min(w - 2, centre[0] + 25), centre[1]), 3) if index % 9 == 0 else None
        masks.append(blob(shape, centre, axes, int(generator.integers(0, 180)), hole, extra))
    return masks


class RasterPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frames = 0

    def args(self, imgsz, **overrides):
        return get_cfg(overrides={
            **raster_training_options(), 'imgsz': imgsz, 'mask_ratio': 4, 'hsv_h': 0.015, 'hsv_s': 0.7,
            'hsv_v': 0.2, 'scale': 0.5, 'translate': 0.3, 'degrees': 30.0, 'shear': 10.0, 'fliplr': 0.5,
            'flipud': 0.5, 'augment': True, **overrides})

    def write(self, shape, instances, name=None):
        """One frame and its label file. `instances` holds masks, or ready-made label records."""
        name = name or f'frame_{self.frames:03d}'
        self.frames += 1
        images, labels = self.root / 'images/train', self.root / 'labels/train'
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(self.frames)
        cv2.imwrite(str(images / f'{name}.png'), generator.integers(0, 255, (*shape, 3), dtype=np.uint8))
        records = []
        for index, item in enumerate(instances):
            record = item if isinstance(item, dict) else mask_record(item, index % 2)
            if record is not None:
                records.append(record)
        write_mask_labels(labels / f'{name}.json', shape, records)

    def dataset(self, imgsz=256, augment=True, rect=False, single_cls=False, **overrides):
        return RasterMaskDataset(img_path=str(self.root / 'images/train'), imgsz=imgsz, batch_size=1,
                                 hyp=self.args(imgsz, **overrides), augment=augment, rect=rect,
                                 single_cls=single_cls, stride=32, pad=0.5 if rect else 0.0)

    def assert_same(self, dataset, seeds=range(12)):
        compared = 0
        for index in range(len(dataset)):
            for seed in seeds:
                label = dataset.get_image_and_label(index)
                outputs = []
                for function in (dataset._format_sample, lambda l: reference_format_sample(dataset, l)):
                    # The same random augmentation for both: Albumentations keeps its own generators.
                    random.seed(seed)
                    np.random.seed(seed)
                    for transform in (dataset.geometry, dataset.blur):
                        if transform is not None:
                            transform.set_random_seed(seed)
                    outputs.append(function(deepcopy(label)))
                new, old = outputs
                self.assertEqual(set(new), set(old))
                for key in old:
                    with self.subTest(frame=index, seed=seed, key=key):
                        if isinstance(old[key], torch.Tensor):
                            self.assertEqual(new[key].dtype, old[key].dtype)
                            self.assertTrue(torch.equal(new[key], old[key]))
                        else:
                            self.assertEqual(new[key], old[key])
                compared += 1
        return compared

    def test_crowded_frames_are_prepared_exactly_as_before(self):
        for seed in range(3):
            self.write((220, 300), crowded_frame(count=40 + 10 * seed, seed=seed))
        self.write((300, 220), crowded_frame((300, 220), count=30, seed=9))  # portrait
        self.assertEqual(self.assert_same(self.dataset()), 48)

    def test_frames_enlarged_or_reduced_to_the_model_size_are_prepared_exactly_as_before(self):
        self.write((97, 131), crowded_frame((97, 131), count=25, seed=4))   # enlarged
        self.write((450, 610), crowded_frame((450, 610), count=60, seed=5))  # reduced
        self.assert_same(self.dataset(imgsz=320))

    def test_validation_frames_are_prepared_exactly_as_before(self):
        self.write((220, 300), crowded_frame(count=50, seed=6))
        self.write((300, 220), crowded_frame((300, 220), count=50, seed=7))
        self.assert_same(self.dataset(augment=False, rect=True), seeds=[0])
        self.assert_same(self.dataset(augment=False), seeds=[0])

    def test_one_class_training_is_prepared_exactly_as_before(self):
        self.write((220, 300), crowded_frame(count=30, seed=8))
        self.assert_same(self.dataset(single_cls=True), seeds=range(4))

    def squares(self, count, shape=(200, 400)):
        masks = []
        for index in range(count):  # separate small squares
            mask = np.zeros(shape, np.uint8)
            y, x = divmod(index, 40)
            mask[y * 13 + 2:y * 13 + 8, x * 10 + 2:x * 10 + 7] = 1
            masks.append(mask)
        return masks

    def test_more_instances_than_one_label_image_holds(self):
        self.write((200, 400), self.squares(400))  # more than 255: two label images
        dataset = self.dataset(imgsz=448, degrees=5.0, shear=0.0, scale=0.1, translate=0.05)
        record = json.loads(Path(dataset.labels[0]['mask_sidecar']).read_text())
        self.assertEqual(RasterMaskDataset._instance_layers(record)[1], [255, 145])
        self.assert_same(dataset, seeds=range(3))

    def test_a_frame_with_more_instances_than_the_earlier_code_could_flip(self):
        # The earlier code stacked one image per instance, and OpenCV cannot flip more than 512 of them.
        self.write((200, 400), self.squares(600))
        dataset = self.dataset(imgsz=448, degrees=5.0, shear=0.0, scale=0.1, translate=0.05, fliplr=1.0)
        label = dataset.get_image_and_label(0)
        dataset.geometry.set_random_seed(0)
        with self.assertRaises(cv2.error):
            reference_format_sample(dataset, deepcopy(label))
        dataset.geometry.set_random_seed(0)
        self.assertGreater(len(dataset._format_sample(deepcopy(label))['cls']), 500)

    def test_overlapping_instances_keep_every_pixel(self):
        shape = (120, 160)
        first = blob(shape, (60, 60), (30, 20))
        second = blob(shape, (75, 60), (30, 20))  # overlaps the first
        third = blob(shape, (70, 60), (8, 8))     # inside both
        self.write(shape, [first, second, third])
        dataset = self.dataset(imgsz=160)
        record = json.loads(Path(dataset.labels[0]['mask_sidecar']).read_text())
        layers, counts, places = RasterMaskDataset._instance_layers(record)
        self.assertEqual(len(layers), 3)
        for mask, (layer, number) in zip((first, second, third), places):
            np.testing.assert_array_equal(layers[layer] == number, mask > 0)
        self.assert_same(dataset, seeds=range(6))

    def test_empty_polygon_and_single_instances(self):
        shape = (90, 120)
        polygon = {'cls': 1, 'bbox': [0.5, 0.5, 0.2, 0.2],
                   'segmentation': [[40, 30, 70, 30, 70, 60, 40, 60], [90, 70, 100, 70, 100, 80]]}
        empty_rle = {'cls': 0, 'bbox': [0, 0, 0, 0], 'segmentation': encode_mask(np.zeros(shape, np.uint8))}
        empty_polygon = {'cls': 0, 'bbox': [0, 0, 0, 0], 'segmentation': [[1, 2]]}
        self.write(shape, [polygon, empty_rle, empty_polygon, blob(shape, (20, 20), (9, 6))], 'mixed')
        self.write(shape, [blob(shape, (60, 45), (25, 12))], 'single')
        self.write(shape, [], 'none')
        self.write(shape, [empty_rle], 'only_empty')
        self.assert_same(self.dataset(imgsz=128))
        self.assert_same(self.dataset(imgsz=128, augment=False, rect=True), seeds=[0])

    def test_instances_moved_out_of_the_frame_are_dropped_as_before(self):
        shape = (100, 100)
        self.write(shape, [blob(shape, (3, 50), (4, 4)), blob(shape, (96, 50), (4, 4)), blob(shape, (50, 50), (10, 10))])
        dataset = self.dataset(imgsz=96, translate=0.45, scale=0.0, degrees=0.0, shear=0.0)
        self.assert_same(dataset, seeds=range(40))
        kept = {len(dataset._format_sample(dataset.get_image_and_label(0))['cls']) for _ in range(40)}
        self.assertTrue(kept & {1, 2}, 'the translation should sometimes push an instance out')

    def test_mask_sampling_must_stay_nearest(self):
        # Linear sampling would blend neighbouring labels into a third, valid-looking label.
        import albumentations as A
        self.write((64, 64), [blob((64, 64), (30, 30), (10, 10))])
        real = A.Affine

        def linear(*args, **kwargs):
            return real(*args, **dict(kwargs, mask_interpolation=cv2.INTER_LINEAR))

        with patch('training.raster_masks.A.Affine', linear):
            with self.assertRaisesRegex(ValueError, 'nearest'):
                self.dataset(imgsz=64)


if __name__ == '__main__':
    unittest.main()
