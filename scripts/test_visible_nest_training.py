"""Regression coverage for frame-local nests and exact-mask YOLO training."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np
import torch
import yaml
from PyQt6.QtWidgets import QApplication
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel

from core.annotation import AnnotationManager
from core.categories import BROOD_CATEGORIES
from core.annotation_scope import frame_categories, merge_annotations, split_annotations
from core.coco_masks import decode_segmentation, encode_mask
from core.project_manager import ProjectManager
from gui.dialogs import ProjectDialog
from training.coco_video_export import export_coco_per_video, export_coco_with_tracking
from training.raster_masks import (
    MASK_FORMAT, RasterMaskDataset, RasterSegmentationTrainer, RasterSegmentationValidator,
    mask_record, raster_training_options, write_mask_labels,
)
from training.yolo_trainer import YOLOTrainingWorker


def nest_mask(h=64, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[4:36, 4:36] = 1
    mask[12:24, 12:24] = 0
    mask[44:60, 44:60] = 1
    return mask


def annotation(mask, category='hive', mask_id=1):
    return {'mask': mask * 255, 'mask_id': mask_id, 'category': category,
            'category_id': {'bee': 1, 'hive': 2, 'chamber': 3, 'pollen': 4}[category]}


class VisibleNestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        ProjectManager().create_project(self.root, 'visible nests')
        self.manager = AnnotationManager()
        self.manager.load_project(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def make_video(self, name='video', scope='frame'):
        frames = self.root / 'frames' / name
        frames.mkdir(parents=True)
        (frames / 'video_metadata.json').write_text(json.dumps({'selected_frames': [0, 1, 2]}))
        for i in range(3):
            cv2.imwrite(str(frames / f'frame_{i:06d}.jpg'), np.zeros((64, 64, 3), np.uint8))
        for i, mask in enumerate((nest_mask(), np.fliplr(nest_mask()))):
            self.manager.save_frame_annotations(self.root, name, i, [annotation(mask)])
        # A stale shared hive must not be added to frame-local labels.
        self.manager.save_video_annotations(self.root, name, [annotation(np.ones((64, 64), np.uint8), mask_id=8)])
        return frames

    def test_new_project_scope_and_dialog(self):
        self.assertEqual(frame_categories(self.manager.project_info), {'bee', 'hive'} | set(BROOD_CATEGORIES))
        dialog = ProjectDialog(default_dir=self.root)
        self.assertEqual(dialog.get_project_info()['hive_annotation_scope'], 'frame')
        dialog.hive_scope_combo.setCurrentIndex(1)
        self.assertEqual(dialog.get_project_info()['hive_annotation_scope'], 'video')
        dialog.close()

    def test_legacy_projects_keep_shared_hives(self):
        for info in ({}, {'project_info': {}}, {'hive_annotation_scope': 'video'}):
            self.assertEqual(frame_categories(info), {'bee'} | set(BROOD_CATEGORIES))
            frame, shared = split_annotations([{'category': 'hive'}, {'category': 'bee'}], info)
            self.assertEqual(frame, [{'category': 'bee'}])
            self.assertEqual(shared, [{'category': 'hive'}])

    def test_legacy_shared_hive_export_is_unchanged(self):
        self.make_video()
        info = self.root / 'annotations/project.json'
        info.write_text(json.dumps({'name': 'old project'}))
        self.manager.load_project(self.root)
        self.assertEqual(frame_categories(self.manager.project_info), {'bee'} | set(BROOD_CATEGORIES))
        before = (self.root / 'annotations/json/video/video_annotations.json').read_bytes()
        paths = export_coco_per_video(self.root, ['video'], 'train')
        coco = json.loads(paths[0].read_text())
        self.assertEqual(len(coco['annotations']), 2)
        for ann in coco['annotations']:
            self.assertEqual(ann['instance_id'], 8)
            self.assertEqual(ann['area'], 64 * 64)
        self.assertEqual(before, (self.root / 'annotations/json/video/video_annotations.json').read_bytes())

    def window_stub(self):
        from gui.main_window import MainWindow
        window = SimpleNamespace(
            annotation_manager=self.manager, project_path=self.root,
            current_video_id='video', current_frame_idx=0,
            _get_frame_idx_in_video=lambda index: index,
            save_worker=Mock(), status_label=Mock(),
            dirty_frame_annotation_keys=set(), canvas=Mock(),
            update_instance_list_from_canvas=Mock(),
            _refresh_video_next_mask_id=lambda video: 2,
        )
        window._split_source_annotations = lambda anns: MainWindow._split_source_annotations(window, anns)
        return window

    def test_tracking_preserves_nest_and_rejects_conflicting_ids(self):
        from gui.main_window import MainWindow
        self.make_video()
        window = self.window_stub()
        MainWindow._queue_tracked_bee_annotations(window, 0, [])
        saved = window.save_worker.add_save_task.call_args.args[3]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['category'], 'hive')
        bee = annotation(nest_mask(), 'bee', 2)
        MainWindow._queue_tracked_bee_annotations(window, 0, [bee])
        saved = window.save_worker.add_save_task.call_args.args[3]
        self.assertEqual([ann['category'] for ann in saved], ['hive', 'bee'])
        window.save_worker.reset_mock()
        with self.assertRaisesRegex(ValueError, 'conflict'):
            MainWindow._queue_tracked_bee_annotations(window, 0, [annotation(nest_mask(), 'bee', 1)])
        window.save_worker.add_save_task.assert_not_called()

    def test_tracking_does_not_resurrect_deleted_cached_nest(self):
        from gui.main_window import MainWindow
        self.make_video()
        window = self.window_stub()
        self.manager.set_frame_annotations(0, [], video_id='video')
        MainWindow._queue_tracked_bee_annotations(window, 0, [])
        self.assertEqual(window.save_worker.add_save_task.call_args.args[3], [])

    def test_delete_all_bees_keeps_frame_local_nest(self):
        from gui.main_window import MainWindow
        from PyQt6.QtWidgets import QMessageBox
        window = self.window_stub()
        anns = [annotation(nest_mask()), annotation(nest_mask(), 'bee', 2)]
        self.manager.save_frame_annotations(self.root, 'video', 0, anns)
        window.canvas.get_annotations.return_value = anns
        with patch('gui.main_window.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
            MainWindow.delete_all_instances(window)
        loaded = self.manager.load_frame_annotations(self.root, 'video', 0)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['category'], 'hive')
        np.testing.assert_array_equal(loaded[0]['mask'] > 0, nest_mask() > 0)

    def test_nested_project_metadata_survives_reload(self):
        self.manager.save_project(self.root)
        self.manager.load_project(self.root)
        self.assertIn('hive', frame_categories(self.manager.project_info))

    def test_merge_frame_specific_hive_wins_without_deleting_disk_source(self):
        self.make_video()
        local = self.manager.load_frame_annotations(self.root, 'video', 0)
        shared, _ = self.manager.load_video_annotations(self.root, 'video')
        merged = merge_annotations(local, shared, self.manager.project_info)
        self.assertEqual(len(merged), 1)
        np.testing.assert_array_equal(merged[0]['mask'] > 0, nest_mask() > 0)
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]['mask_id'], 8)

    def test_distinct_frames_roundtrip_and_blank_frame_stays_blank(self):
        self.make_video()
        for i, expected in enumerate((nest_mask(), np.fliplr(nest_mask()))):
            loaded = self.manager.load_frame_annotations(self.root, 'video', i)
            np.testing.assert_array_equal(loaded[0]['mask'] > 0, expected > 0)
        self.assertFalse(self.manager.load_frame_annotations(self.root, 'video', 2))

    def test_overlapping_instances_and_reused_category_ids_roundtrip(self):
        masks = [nest_mask(), np.ones((64, 64), np.uint8)]
        anns = [annotation(masks[0]), annotation(masks[1], 'bee')]
        self.manager.save_frame_annotations(self.root, 'video', 0, anns)
        loaded = self.manager.load_frame_annotations(self.root, 'video', 0)
        self.assertEqual(len(loaded), 2)
        for ann, expected in zip(loaded, masks):
            self.assertNotIn('mask_coco_rle', ann)
            np.testing.assert_array_equal(ann['mask'] > 0, expected > 0)

    def test_legacy_png_without_rle_loads(self):
        self.manager.save_frame_annotations(self.root, 'video', 0, [annotation(nest_mask())])
        path = self.root / 'annotations/json/video/frame_000000.json'
        records = json.loads(path.read_text())
        records[0].pop('mask_coco_rle')
        path.write_text(json.dumps(records))
        loaded = self.manager.load_frame_annotations(self.root, 'video', 0)
        np.testing.assert_array_equal(loaded[0]['mask'] > 0, nest_mask() > 0)

    def test_both_coco_exports_keep_topology_and_skip_unlabeled_frame(self):
        self.make_video()
        paths = export_coco_per_video(self.root, ['video'], 'train')
        paths.append(export_coco_with_tracking(self.root, ['video'], 'val'))
        for path in paths:
            coco = json.loads(path.read_text())
            self.assertEqual(len(coco['images']), 2)
            self.assertEqual(len(coco['annotations']), 2)
            for ann, expected in zip(coco['annotations'], (nest_mask(), np.fliplr(nest_mask()))):
                self.assertEqual(ann['category_id'], 2)
                self.assertEqual(ann['area'], int(expected.sum()))
                self.assertEqual(ann['iscrowd'], 0)
                np.testing.assert_array_equal(decode_segmentation(ann['segmentation'], 64, 64), expected)

    def test_coco_to_training_labels_preserves_all_parts(self):
        self.make_video()
        export_coco_per_video(self.root, ['video'], 'train')
        export_coco_per_video(self.root, ['video'], 'val')
        worker = YOLOTrainingWorker(self.root, {'model_type': 'hive'})
        output, dataset_yaml = worker._prepare_yolo_dataset()
        self.assertEqual(yaml.safe_load(dataset_yaml.read_text())['mask_format'], MASK_FORMAT)
        for split in ('train', 'val'):
            labels = sorted((output / 'labels' / split).glob('*.json'))
            self.assertEqual(len(labels), 2)
            self.assertFalse(list((output / 'labels' / split).glob('*.txt')))
            for path, expected in zip(labels, (nest_mask(), np.fliplr(nest_mask()))):
                data = json.loads(path.read_text())
                self.assertEqual(len(data['instances']), 1)
                np.testing.assert_array_equal(
                    decode_segmentation(data['instances'][0]['segmentation'], 64, 64), expected
                )

    def test_instance_focused_crops_accept_exact_masks(self):
        from training.yolo_trainer_instance_focused import YOLOTrainingWorkerInstanceFocused
        self.make_video()
        paths = export_coco_per_video(self.root, ['video'], 'train')
        coco = json.loads(paths[0].read_text())
        for ann in coco['annotations']:
            ann['category_id'] = 1
        paths[0].write_text(json.dumps(coco))
        images = self.root / 'crops/images/train'
        labels = self.root / 'crops/labels/train'
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        worker = YOLOTrainingWorkerInstanceFocused(self.root, {})
        count = worker._process_coco_to_instance_crops(paths[0], images, labels,
                                                      'absolute', 4, 0, 64, 1, True)
        self.assertEqual(count, 2)
        for path, expected in zip(sorted(labels.glob('*.json')), (nest_mask(), np.fliplr(nest_mask()))):
            instance = json.loads(path.read_text())['instances'][0]
            np.testing.assert_array_equal(decode_segmentation(instance['segmentation'], 64, 64), expected)

    def test_validation_preview_accepts_rle(self):
        from gui.validation_viewer import ValidationViewer
        actual = ValidationViewer.polygon_to_mask(None, encode_mask(nest_mask()), (64, 64))
        np.testing.assert_array_equal(actual, nest_mask())

    def test_frame_level_validation_uses_each_frames_hive(self):
        from gui.frame_level_validation_worker import FrameLevelValidationWorker
        self.make_video()
        path = export_coco_per_video(self.root, ['video'], 'val')[0]
        coco = json.loads(path.read_text())
        worker = FrameLevelValidationWorker(self.window_stub(), {})
        for index, expected in enumerate((nest_mask(), np.fliplr(nest_mask()))):
            actual = worker._extract_gt_hive_from_coco(coco, index, (64, 64))
            np.testing.assert_array_equal(actual > 0, expected > 0)
        self.assertIsNone(worker._extract_gt_hive_from_coco(coco, 2, (64, 64)))

    def test_gui_training_workers_use_raster_trainer(self):
        from training.yolo_trainer_instance_focused import YOLOTrainingWorkerInstanceFocused
        for module, worker_class in (
            ('training.yolo_trainer', YOLOTrainingWorker),
            ('training.yolo_trainer_instance_focused', YOLOTrainingWorkerInstanceFocused),
        ):
            with patch(f'{module}.require_cuda_device', return_value=(0, 'test GPU')), \
                    patch(f'{module}.YOLO') as yolo:
                worker_class(self.root, {})._train_model(self.root / 'dataset.yaml')
                options = yolo.return_value.train.call_args.kwargs
                self.assertIs(options['trainer'], RasterSegmentationTrainer)
                self.assertTrue(options['augment'])
                for key, value in raster_training_options().items():
                    self.assertEqual(options[key], value)

    def test_no_cpu_fallback_was_added_to_gui_training(self):
        with patch('training.yolo_trainer.require_cuda_device', side_effect=RuntimeError('CUDA unavailable')), \
                patch('training.yolo_trainer.YOLO') as yolo:
            with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
                YOLOTrainingWorker(self.root, {})._train_model(self.root / 'dataset.yaml')
            yolo.assert_not_called()


class RasterDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = get_cfg(overrides={
            **raster_training_options(), 'imgsz': 64, 'mask_ratio': 1,
            'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 'scale': 0.0,
            'translate': 0.0, 'degrees': 0.0, 'shear': 0.0,
            'fliplr': 0.0, 'flipud': 0.0,
        })

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, masks=None, augment=False, rect=False):
        masks = masks if masks is not None else [nest_mask()]
        shape = masks[0].shape
        images = self.root / 'images/train'
        labels = self.root / 'labels/train'
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        image = np.repeat((masks[0] * 255)[..., None], 3, axis=2)
        cv2.imwrite(str(images / 'sample.png'), image)
        write_mask_labels(labels / 'sample.json', shape, [mask_record(m) for m in masks])
        return RasterMaskDataset(img_path=str(images), imgsz=64, batch_size=1,
                                 hyp=self.args, augment=augment, rect=rect, stride=32, pad=0)

    def test_rle_is_coco_column_major_and_json_safe(self):
        mask = nest_mask(64, 80)
        mask[0, 77] = 1
        encoded = json.loads(json.dumps(encode_mask(mask)))
        np.testing.assert_array_equal(decode_segmentation(encoded, 64, 80), mask)
        with self.assertRaises(ValueError):
            decode_segmentation(encoded, 80, 64)

    def test_polygon_components_are_unioned_without_bridges(self):
        mask = decode_segmentation([[1, 1, 9, 1, 9, 9, 1, 9], [40, 40, 48, 40, 48, 48, 40, 48]], 64, 64)
        self.assertEqual(mask[5, 5], 1)
        self.assertEqual(mask[44, 44], 1)
        self.assertEqual(mask[20, 20], 0)
        self.assertEqual(mask_record(mask)['cls'], 0)

    def test_training_and_validation_preserve_holes_and_instance_count(self):
        for augment in (False, True):
            sample = self.dataset(augment=augment)[0]
            self.assertEqual(sample['cls'].shape, (1, 1))
            np.testing.assert_array_equal(sample['masks'][0].numpy(), nest_mask())
            self.assertEqual(tuple(sample['masks'].shape), (1, 64, 64))

    def test_flip_applies_to_image_and_mask_together(self):
        self.args.fliplr = 1.0
        sample = self.dataset(augment=True)[0]
        expected = np.fliplr(nest_mask())
        np.testing.assert_array_equal(sample['masks'][0].numpy(), expected)
        np.testing.assert_array_equal(sample['img'][0].numpy() > 0, expected > 0)

    def test_affine_padding_is_background_and_each_mask_stays_binary(self):
        import albumentations as A
        dataset = self.dataset([nest_mask(), np.ones((64, 64), np.uint8)], augment=True)
        dataset.geometry = A.Compose([A.Affine(
            scale=1.0, rotate=0, shear=0, translate_px={'x': 8, 'y': 0},
            interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
            cval=114, cval_mask=0, mode=cv2.BORDER_CONSTANT, p=1,
        )])
        sample = dataset[0]
        for i, original in enumerate((nest_mask(), np.ones((64, 64), np.uint8))):
            expected = np.zeros((64, 64), np.uint8)
            expected[:, 8:] = original[:, :-8]
            np.testing.assert_array_equal(sample['masks'][i].numpy(), expected)
        self.assertTrue((sample['img'][:, :, :8] == 114).all())

    def test_random_affine_masks_remain_binary(self):
        self.args.scale, self.args.translate = 0.5, 0.2
        self.args.degrees, self.args.shear = 20, 5
        dataset = self.dataset(augment=True)
        for _ in range(20):
            self.assertTrue(set(dataset[0]['masks'].unique().tolist()).issubset({0, 1}))

    def test_overlapping_masks_remain_separate_and_collate(self):
        dataset = self.dataset([nest_mask(), np.ones((64, 64), np.uint8)])
        sample = dataset[0]
        batch = dataset.collate_fn([sample, dataset[0]])
        self.assertEqual(tuple(batch['masks'].shape), (4, 64, 64))
        self.assertEqual(batch['batch_idx'].tolist(), [0, 0, 1, 1])
        self.assertEqual(int(sample['masks'][1].sum()), 4096)

    def test_letterbox_coordinates_and_boxes(self):
        mask = np.zeros((32, 64), np.uint8)
        mask[4:24, 4:24] = 1
        sample = self.dataset([mask])[0]
        self.assertEqual(sample['ratio_pad'], ((1.0, 1.0), (0, 16)))
        expected = np.pad(mask, ((16, 16), (0, 0)))
        np.testing.assert_array_equal(sample['masks'][0].numpy(), expected)
        np.testing.assert_allclose(sample['bboxes'][0].numpy(), [14/64, 30/64, 20/64, 20/64])

    def test_real_segmentation_loss_backward(self):
        torch.set_num_threads(2)
        dataset = self.dataset()
        batch = dataset.collate_fn([dataset[0], dataset[0]])
        batch['img'] = batch['img'].float() / 255
        model = SegmentationModel('yolov8n-seg.yaml', nc=1, verbose=False)
        model.args = self.args
        model.train()
        loss, items = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue((items >= 0).all())
        loss.sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_complete_training_and_standalone_validation(self):
        from ultralytics import YOLO
        torch.set_num_threads(2)
        self.dataset()
        data = self.root / 'dataset.yaml'
        data.write_text(yaml.safe_dump({
            'path': str(self.root), 'train': 'images/train', 'val': 'images/train',
            'names': {0: 'hive'}, 'nc': 1, 'mask_format': MASK_FORMAT,
        }))
        model = YOLO('yolov8n-seg.yaml')
        model.add_callback('on_train_batch_end', lambda trainer: self.assertTrue((trainer.loss_items >= 0).all()))
        with patch('ultralytics.engine.model.checks.check_pip_update_available', return_value=False), \
                patch('ultralytics.data.utils.check_font'):
            model.train(
                trainer=RasterSegmentationTrainer, data=str(data), epochs=1,
                imgsz=64, batch=1, workers=0, device='cpu', amp=False, pretrained=False,
                plots=False, project=str(self.root / 'runs'), name='smoke',
                **raster_training_options(),
            )
            self.assertIsInstance(model.trainer.train_loader.dataset, RasterMaskDataset)
            self.assertIsInstance(model.trainer.test_loader.dataset, RasterMaskDataset)
            best = model.trainer.best
            self.assertTrue(best.exists())
            reloaded = YOLO(str(best))
            result = reloaded.val(validator=RasterSegmentationValidator, data=str(data),
                                  imgsz=64, batch=1, workers=0, device='cpu', plots=False,
                                  project=str(self.root / 'runs'), name='reload', overlap_mask=False)
            self.assertTrue(np.isfinite(result.box.map))


if __name__ == '__main__':
    unittest.main()
