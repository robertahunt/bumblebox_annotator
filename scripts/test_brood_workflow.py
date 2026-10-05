"""Synthetic coverage for visible brood labels, training and experimental history."""

import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np
import torch
import yaml
from PyQt6.QtWidgets import QApplication
from ultralytics import YOLO
from ultralytics.engine.results import Results

from core.annotation import AnnotationManager
from core.annotation_scope import split_annotations
from core.brood_inference import BroodVideoWriter, brood_evidence, brood_model_classes, paint_brood_overlay
from core.categories import BROOD_CATEGORIES, CATEGORIES, CATEGORY_COLORS, training_categories
from core.coco_masks import decode_segmentation
from core.instance_tracker import Detection
from core.project_manager import ProjectManager
from core.temporal_brood import TemporalBroodMap
from gui.canvas import ImageCanvas
from gui.toolbar import AnnotationToolbar
from gui.training_dialog import TrainingConfigDialog
from training.coco_video_export import export_coco_per_video
from training.raster_masks import RasterSegmentationTrainer, raster_training_options
from training.yolo_trainer import YOLOTrainingWorker


def stage_masks():
    masks = []
    for i in range(5):
        mask = np.zeros((64, 64), np.uint8)
        mask[8:56, 2 + 12 * i:10 + 12 * i] = 1
        mask[22:26, 4 + 12 * i:8 + 12 * i] = 0
        masks.append(mask)
    return masks


class BroodAnnotationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def annotations(self):
        return [dict(mask_id=i + 2, category=category, mask=mask * 255)
                for i, (category, mask) in enumerate(zip(BROOD_CATEGORIES, stage_masks()))]

    def test_category_ids_and_frame_scope(self):
        self.assertEqual(CATEGORIES[:4], ('bee', 'hive', 'chamber', 'pollen'))
        self.assertEqual(training_categories('brood'), BROOD_CATEGORIES)
        for scope in ('frame', 'video'):
            frame, shared = split_annotations(self.annotations(), {'hive_annotation_scope': scope})
            self.assertEqual(len(frame), 5)
            self.assertFalse(shared)

    def test_canvas_overlap_color_visibility_and_roundtrip(self):
        canvas = ImageCanvas()
        self.addCleanup(canvas.close)
        canvas.load_image(np.zeros((64, 64, 3), np.uint8))
        hive = dict(mask_id=1, category='hive', mask=np.ones((64, 64), np.uint8) * 255)
        canvas.set_annotations([hive] + self.annotations())
        for i, category in enumerate(BROOD_CATEGORIES, 2):
            canvas.set_annotation_type_visibility(category, True)
            self.assertTrue(canvas.is_annotation_instance_visible(i, category))
            self.assertEqual(canvas.color_for_category(category, i), CATEGORY_COLORS[category])
            canvas.start_editing_instance(i, category=category)
            canvas.set_instance_visible(i, category, False)
            self.assertFalse(canvas._is_editing_mask_visible())
            canvas.set_instance_visible(i, category, True)
            self.assertTrue(canvas._is_editing_mask_visible())
            canvas.commit_editing()
        reloaded = {a['category']: a for a in canvas.get_annotations()}
        for category, mask in zip(BROOD_CATEGORIES, stage_masks()):
            np.testing.assert_array_equal(reloaded[category]['mask'] > 0, mask > 0)
        self.assertEqual(int((reloaded['hive']['mask'] > 0).sum()), 4096)
        canvas.change_instance_category(2, 'brood_late', old_category='brood_early')
        self.assertFalse(np.any(canvas.brood_early_mask == 2))
        self.assertTrue(np.any(canvas.brood_late_mask == 2))
        canvas.delete_instance(2, category='brood_late')
        self.assertFalse(np.any(canvas.brood_late_mask == 2))
        self.assertTrue(np.all(canvas.hive_mask == 1))

    def test_toolbar_and_training_selection(self):
        toolbar = AnnotationToolbar()
        self.addCleanup(toolbar.close)
        self.assertEqual(set(toolbar.brood_visibility_actions), set(BROOD_CATEGORIES))
        self.assertTrue(all(not a.isChecked() for a in toolbar.brood_visibility_actions.values()))
        seen = []
        toolbar.annotation_type_visibility_changed.connect(lambda *args: seen.append(args))
        toolbar.brood_visibility_actions['brood_late'].setChecked(True)
        self.assertEqual(seen, [('brood_late', True)])
        dialog = TrainingConfigDialog()
        self.addCleanup(dialog.close)
        dialog.model_type_combo.setCurrentText('Brood (5 appearance classes)')
        self.assertEqual(dialog.get_config()['model_type'], 'brood')
        with patch('gui.training_dialog.QMessageBox.warning') as warning:
            dialog.accept()
            warning.assert_called_once()
        dialog.brood_review_check.setChecked(True)
        self.assertTrue(dialog.get_config()['brood_reviewed'])

    def test_new_instance_from_main_window_and_batch_slot(self):
        from gui.main_window import MainWindow
        from gui.batch_video_inference_dialog import BatchVideoInferenceConfigDialog
        from PyQt6.QtWidgets import QMessageBox
        with patch.object(MainWindow, 'load_settings'):
            window = MainWindow()
        try:
            window.canvas.load_image(np.zeros((64, 64, 3), np.uint8))
            for category in BROOD_CATEGORIES:
                window.new_instance(category)
                self.assertEqual(window.canvas.current_annotation_type, category)
                self.assertTrue(window.toolbar.brood_visibility_actions[category].isChecked())
            with patch.object(BatchVideoInferenceConfigDialog, '_restore_last_settings'):
                dialog = BatchVideoInferenceConfigDialog()
            dialog.brood_model_edit.setText('/tmp/test-brood.pt')
            self.assertTrue(dialog.temporal_hive_window_spin.isEnabled())
            dialog.close()
        finally:
            with patch('gui.main_window.QSettings'), \
                    patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
                window.close()
            self.app.processEvents()

    def prepare_dataset(self):
        ProjectManager().create_project(self.root, 'brood test')
        manager = AnnotationManager()
        manager.load_project(self.root)
        frames = self.root / 'frames' / 'video'
        frames.mkdir(parents=True)
        (frames / 'video_metadata.json').write_text(json.dumps({'selected_frames': [0, 1]}))
        for index in range(2):
            cv2.imwrite(str(frames / f'frame_{index:06d}.jpg'), np.full((64, 64, 3), 100, np.uint8))
            manager.save_frame_annotations(self.root, 'video', index, self.annotations())
        paths = export_coco_per_video(self.root, ['video'], 'train')
        export_coco_per_video(self.root, ['video'], 'val')
        return paths[0]

    def test_source_export_and_multiclass_conversion(self):
        path = self.prepare_dataset()
        data = json.loads(path.read_text())
        self.assertEqual([c['name'] for c in data['categories']][-5:], list(BROOD_CATEGORIES))
        worker = YOLOTrainingWorker(self.root, {'model_type': 'brood', 'brood_reviewed': True})
        _, dataset_path = worker._prepare_yolo_dataset()
        config = yaml.safe_load(dataset_path.read_text())
        self.assertEqual(config['nc'], 5)
        self.assertEqual(config['names'], dict(enumerate(BROOD_CATEGORIES)))
        record = json.loads(next((self.root / 'yolo_format/labels/train').glob('*.json')).read_text())
        self.assertEqual([i['cls'] for i in record['instances']], list(range(5)))
        for item, mask in zip(record['instances'], stage_masks()):
            np.testing.assert_array_equal(decode_segmentation(item['segmentation'], 64, 64), mask)

    def test_conflicting_stages_rejected_but_hive_overlap_allowed(self):
        path = self.prepare_dataset()
        data = json.loads(path.read_text())
        data['annotations'][1]['segmentation'] = data['annotations'][0]['segmentation']
        path.write_text(json.dumps(data))
        worker = YOLOTrainingWorker(self.root, {})
        with self.assertRaisesRegex(ValueError, 'Conflicting brood stages'):
            worker._coco_to_yolo(path, self.root / 'converted', 'train', 'brood')

    def test_training_requires_review_before_touching_dataset(self):
        worker = YOLOTrainingWorker(self.root, {'model_type': 'brood'})
        with self.assertRaisesRegex(ValueError, 'all visible brood'):
            worker._prepare_yolo_dataset()
        self.assertFalse((self.root / 'yolo_format').exists())

    def test_five_class_training_checkpoint(self):
        self.prepare_dataset()
        worker = YOLOTrainingWorker(self.root, {'model_type': 'brood', 'brood_reviewed': True})
        _, path = worker._prepare_yolo_dataset()
        torch.set_num_threads(2)
        model = YOLO('yolov8n-seg.yaml')
        with patch('ultralytics.engine.model.checks.check_pip_update_available', return_value=False), \
                patch('ultralytics.data.utils.check_font'):
            model.train(trainer=RasterSegmentationTrainer, data=str(path), epochs=1, imgsz=64,
                        batch=2, workers=0, device='cpu', amp=False, pretrained=False,
                        plots=False, project=str(self.root / 'runs'), name='brood', **raster_training_options())
        reloaded = YOLO(str(model.trainer.best))
        self.assertEqual(brood_model_classes(reloaded), {i: i + 1 for i in range(5)})


class BroodHistoryTests(unittest.TestCase):
    def setUp(self):
        self.history = TemporalBroodMap(resolution=(16, 16), window_seconds=100)
        self.chamber = {'bbox': [0, 0, 16, 16]}
        self.labels = np.zeros((16, 16), np.uint8)
        self.labels[4:12, 4:12] = 1

    def update(self, time, labels=None, bees=(), **kwargs):
        return self.history.update('test', 0, self.chamber, self.labels if labels is None else labels,
                                   bees, (16, 16), time, **kwargs)

    def test_visible_evidence_and_occlusion(self):
        self.assertFalse(self.update(0)['labels'].any())
        snap = self.update(1)
        self.assertTrue(np.all(snap['labels'][4:12, 4:12] == 2))
        bee = Detection(np.array([4, 4, 12, 12]))
        snap = self.update(2, np.zeros_like(self.labels), [bee])
        self.assertTrue(np.all(snap['labels'][4:12, 4:12] == 2))
        overlap = self.history.bee_overlap(snap, bee, self.chamber, (16, 16))
        self.assertEqual(overlap['brood_fraction'], 1)
        self.assertEqual(overlap['brood_early_fraction'], 1)

    def test_unknown_does_not_mean_uncertain_stage(self):
        unknown = np.full_like(self.labels, 255)
        self.update(0, unknown)
        self.assertFalse(self.update(1, unknown)['labels'].any())
        ambiguous = np.full_like(self.labels, 2)
        self.update(2, ambiguous)
        self.assertTrue(np.all(self.update(3, ambiguous)['labels'] == 3))

    def test_recent_stage_replaces_long_history_and_stale_evidence_expires(self):
        for time in range(100):
            self.update(time)
        late = np.where(self.labels == 1, 5, 0).astype(np.uint8)
        for time in range(100, 115):
            snap = self.update(time, late)
        self.assertTrue(np.all(snap['labels'][4:12, 4:12] == 6))
        self.assertLessEqual(float(snap['weight'].max()), 8.001)
        snap = self.history.update('test', 0, self.chamber, None, [], (16, 16), 220)
        self.assertFalse(snap['labels'].any())

    def test_empty_observation_is_evidence_and_missing_is_not(self):
        self.update(0)
        snap = self.update(1)
        before = snap['probability'].copy()
        missing = self.history.update('test', 0, self.chamber, None, [], (16, 16), 2)
        np.testing.assert_allclose(missing['probability'], before, atol=1e-6)
        for time in range(3, 20):
            snap = self.update(time, np.zeros_like(self.labels))
        self.assertTrue(np.all(snap['labels'] == 1))

    def test_backwards_time_and_bad_shapes_rejected(self):
        self.update(10)
        with self.assertRaisesRegex(ValueError, 'out-of-order'):
            self.update(9)
        with self.assertRaisesRegex(ValueError, 'full-frame'):
            self.update(11, np.zeros((3, 3), np.uint8))

    def test_contexts_are_independent_and_bbox_alignment(self):
        self.update(0)
        self.update(1)
        shifted = np.zeros((16, 32), np.uint8)
        shifted[:, 16:] = self.labels
        snap = self.history.update('test', 0, {'bbox': [16, 0, 32, 16]}, shifted, [], (16, 32), 2)
        self.assertTrue(np.all(snap['labels'][4:12, 4:12] == 2))
        other = self.history.update('other', 0, self.chamber, np.full_like(self.labels, 255), [], (16, 16), 2)
        self.assertFalse(other['labels'].any())

    def test_model_validation_and_prediction_conflicts(self):
        with self.assertRaisesRegex(ValueError, 'five brood'):
            brood_model_classes(SimpleNamespace(task='segment', names={0: 'bee'}))
        masks = torch.ones((2, 16, 16))
        result = SimpleNamespace(masks=SimpleNamespace(data=masks), boxes=SimpleNamespace(cls=torch.tensor([0, 1])))
        labels = brood_evidence(result, (16, 16), {0: 1, 1: 2})
        self.assertTrue(np.all(labels == 255))

    def test_archive_preview_and_updated_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            frame = np.full((16, 16, 3), 80, np.uint8)
            writer = BroodVideoWriter(temp, 'video', 'test', self.history, preview=True, preview_limit=2)
            for number in range(1, 3):
                writer.write(number, frame, self.labels, {0: self.chamber}, [], number, 6)
            bee = Detection(np.array([4, 4, 12, 12]), instance_id=1)
            writer.write(3, frame, np.zeros_like(self.labels), {0: self.chamber}, [bee], 3, 6)
            writer.close(complete=True)
            with zipfile.ZipFile(Path(temp) / 'video_brood_maps.zip') as archive:
                metadata = json.loads(archive.read('metadata.json'))
                self.assertTrue(metadata['complete'])
                self.assertEqual(metadata['frame_count'], 3)
                with np.load(io.BytesIO(archive.read('frame_000003_chamber_0.npz')), allow_pickle=False) as data:
                    self.assertTrue(np.all(data['observed_labels'][4:12, 4:12] == 255))
                    self.assertTrue(np.all(data['labels'][4:12, 4:12] == 2))
                    self.assertFalse(np.array_equal(frame, paint_brood_overlay(frame, [dict(data)])))
            self.assertTrue((Path(temp) / 'video_brood_overlap.csv').exists())
            cap = cv2.VideoCapture(str(Path(temp) / 'video_brood_annotated.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            cap.release()

    def test_processor_optional_hook(self):
        from core.batch_video_processor import BatchVideoProcessor
        frame = np.full((16, 16, 3), 80, np.uint8)
        class Model:
            task = 'segment'
            names = dict(enumerate(BROOD_CATEGORIES))
            def __call__(self, image, **kwargs):
                self.kwargs = kwargs
                return [Results(image, 'test', self.names, boxes=torch.tensor([[4, 4, 12, 12, .9, 0]]),
                                masks=torch.tensor((self_outer.labels == 1)[None].astype(np.float32)))]
        self_outer = self
        model = Model()
        with tempfile.TemporaryDirectory() as temp:
            processor = BatchVideoProcessor(Path(temp) / 'video.mp4', 'video', None, None, None,
                                            None, .25, .5, enable_aruco=False, output_folder=Path(temp),
                                            brood_model=model, temporal_brood_map=self.history)
            processor.video_fps = 6
            processor._process_brood_frame(frame, 1, {0: self.chamber}, [])
            processor._process_brood_frame(frame, 2, {0: self.chamber}, [])
            processor._brood_writer.close(complete=True)
            self.assertTrue(model.kwargs['retina_masks'])
            self.assertTrue((Path(temp) / 'brood/video_brood_maps.zip').exists())

    def test_batch_refuses_resume_or_existing_data(self):
        from gui.batch_video_inference_worker import BatchVideoInferenceWorker
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / 'keep.txt'
            existing.write_text('untouched')
            worker = BatchVideoInferenceWorker({'output_folder': temp, 'brood_model_path': 'unused.pt'})
            errors = []
            worker.inference_failed.connect(errors.append)
            worker.run()
            self.assertTrue(any('empty output folder' in e for e in errors))
            self.assertEqual(existing.read_text(), 'untouched')

    def test_full_video_processing_writes_completed_archive(self):
        from core.batch_video_processor import BatchVideoProcessor
        class EmptyModel:
            task = 'segment'
            names = dict(enumerate(BROOD_CATEGORIES))
            def __call__(self, frame, **kwargs):
                return [Results(frame, 'test', self.names, boxes=torch.zeros((0, 6)))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.avi'
            video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 6, (32, 32))
            for _ in range(3):
                video.write(np.full((32, 32, 3), 80, np.uint8))
            video.release()
            model = EmptyModel()
            processor = BatchVideoProcessor(
                path, 'video', model, None, None, SimpleNamespace(update=lambda detections: detections),
                .25, .5, enable_aruco=False, output_folder=Path(temp), compute_spatial_metrics=False,
                brood_model=model, temporal_brood_map=self.history, brood_preview=True)
            self.assertTrue(processor.process())
            with zipfile.ZipFile(Path(temp) / 'brood/video_brood_maps.zip') as archive:
                metadata = json.loads(archive.read('metadata.json'))
                self.assertTrue(metadata['complete'])
                self.assertEqual(metadata['frame_count'], 3)

    def test_missing_chamber_is_not_a_full_frame_observation(self):
        self.update(0)
        self.update(1)
        snap = self.history.update('test', 0, dict(self.chamber, temporal_unavailable=True),
                                   np.zeros_like(self.labels), [], (16, 16), 2)
        self.assertFalse(snap['labels'].any())
        self.assertTrue(np.all(self.update(3)['labels'][4:12, 4:12] == 2))

    def test_worker_shares_known_context_history_and_disabled_signature_is_unchanged(self):
        from gui.batch_video_inference_worker import BatchVideoInferenceWorker
        class EmptyModel:
            task = 'segment'
            names = dict(enumerate(BROOD_CATEGORIES))
            def __call__(self, frame, **kwargs):
                return [Results(frame, 'test', self.names, boxes=torch.zeros((0, 6)))]
        self.assertEqual(BatchVideoInferenceWorker({}).config_signature,
                         BatchVideoInferenceWorker({'brood_model_path': None}).config_signature)
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for second in (0, 1):
                path = Path(temp) / f'bumblebox-1_2026-10-05_01_00_0{second}.avi'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 6, (32, 32))
                for _ in range(3):
                    writer.write(np.full((32, 32, 3), 80, np.uint8))
                writer.release()
                paths.append(path)
            output = Path(temp) / 'output'
            config = dict(video_source=list(reversed(paths)), folder_mode=False,
                          bee_model_path='bee.pt', brood_model_path='brood.pt',
                          output_folder=str(output),
                          tracking_config={'algorithm': 'centroid', 'max_distance': 100, 'max_frames_missing': 5},
                          enable_aruco=False, confidence_threshold=.25, nms_iou_threshold=.5,
                          compute_spatial_metrics=False, temporal_hive_resolution=16)
            worker = BatchVideoInferenceWorker(config)
            errors, completed = [], []
            worker.inference_failed.connect(errors.append)
            worker.inference_complete.connect(lambda: completed.append(True))
            with patch('gui.batch_video_inference_worker.YOLO', return_value=EmptyModel()):
                worker.run()
            self.assertFalse(errors, errors)
            self.assertTrue(completed)
            with zipfile.ZipFile(output / 'brood' / f'{paths[1].stem}_brood_maps.zip') as archive:
                self.assertTrue(json.loads(archive.read('metadata.json'))['complete'])
                with np.load(io.BytesIO(archive.read('frame_000001_chamber_0.npz'))) as snap:
                    self.assertTrue(np.all(snap['labels'] == 1))


if __name__ == '__main__':
    unittest.main()
