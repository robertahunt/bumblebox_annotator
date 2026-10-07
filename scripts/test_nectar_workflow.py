"""Synthetic nectar annotation, export, training, and current-frame inference tests."""

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
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox
from ultralytics import YOLO
from ultralytics.engine.results import Results

from core.annotation import AnnotationManager
from core.annotation_scope import split_annotations
from core.categories import CATEGORIES, CATEGORY_COLORS, training_categories
from core.coco_masks import decode_segmentation
from core.project_manager import ProjectManager
from gui.canvas import ImageCanvas
from gui.hive_chamber_toolbar import HiveChamberToolbar
from gui.main_window import MainWindow
from gui.toolbar import AnnotationToolbar
from gui.training_dialog import TrainingConfigDialog
from training.coco_video_export import export_coco_per_video
from training.raster_masks import RasterSegmentationTrainer, raster_training_options
from training.yolo_trainer import YOLOTrainingWorker


def nectar_mask():
    mask = np.zeros((64, 64), np.uint8)
    mask[8:32, 8:32] = 255
    mask[14:18, 14:18] = 0
    mask[40:48, 40:48] = 255
    return mask


class NectarWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def canvas(self):
        canvas = ImageCanvas()
        self.addCleanup(canvas.close)
        canvas.load_image(np.zeros((64, 64, 3), np.uint8))
        return canvas

    def window(self):
        with patch.object(MainWindow, 'load_settings'):
            window = MainWindow()
        window.canvas.load_image(np.zeros((64, 64, 3), np.uint8))

        def close():
            with patch('gui.main_window.QSettings'), patch.object(
                    QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
                window.close()
            self.app.processEvents()

        self.addCleanup(close)
        return window

    def test_category_appended_and_old_custom_order_retained(self):
        self.assertEqual(CATEGORIES.index('nectar'), 12)
        self.assertEqual(training_categories('nectar'), ('nectar',))
        ProjectManager().create_project(self.root, 'old project')
        path = self.root / 'annotations/project.json'
        info = json.loads(path.read_text())
        old = list(CATEGORIES[:-1]) + ['custom_category']
        info['classes'] = old
        path.write_text(json.dumps(info))
        manager = AnnotationManager()
        manager.load_project(self.root)
        self.assertEqual(manager.class_names, old + ['nectar'])
        for scope in ('frame', 'video'):
            ann = {'category': 'nectar', 'mask_id': 1, 'mask': nectar_mask()}
            frame, shared = split_annotations([ann], {'hive_annotation_scope': scope})
            self.assertEqual(frame, [ann])
            self.assertFalse(shared)

    def test_canvas_overlap_visibility_hit_testing_and_category_change(self):
        canvas = self.canvas()
        canvas.set_annotations([
            dict(mask_id=1, category='hive', mask=np.full((64, 64), 255, np.uint8)),
            dict(mask_id=2, category='nectar', mask=nectar_mask()),
        ])
        canvas.set_annotation_type_visibility('nectar', True)
        self.assertEqual(canvas.color_for_category('nectar', 2), CATEGORY_COLORS['nectar'])
        self.assertEqual(canvas._find_instance_entry_at_point(10, 10), (2, 'nectar'))
        canvas.start_editing_instance(2, category='nectar')
        canvas.set_instance_visible(2, 'nectar', False)
        self.assertFalse(canvas._is_editing_mask_visible())
        canvas.set_instance_visible(2, 'nectar', True)
        self.assertTrue(canvas._is_editing_mask_visible())
        canvas.set_annotation_type_visibility('nectar', False)
        self.assertFalse(canvas._is_editing_mask_visible())
        canvas.set_annotation_type_visibility('nectar', True)
        self.assertTrue(canvas._is_editing_mask_visible())
        menu = QMenu()
        self.addCleanup(menu.close)
        canvas._add_change_category_menu(menu, 2, 'nectar')
        category_menu = menu.actions()[0].menu()
        pollen = next(a for a in category_menu.actions() if a.data() == 'pollen')
        pollen.trigger()
        annotations = canvas.get_annotations()
        self.assertEqual([a['category'] for a in annotations].count('pollen'), 1)
        self.assertNotIn('nectar', [a['category'] for a in annotations])
        np.testing.assert_array_equal(canvas.hive_mask, np.ones((64, 64), np.int32))
        canvas.change_instance_category(2, 'nectar', old_category='pollen')
        np.testing.assert_array_equal(canvas.get_annotations()[1]['mask'], nectar_mask())
        canvas.delete_instance(2, category='nectar')
        self.assertEqual([a['category'] for a in canvas.get_annotations()], ['hive'])

    def test_new_instance_selects_brush_shows_mask_and_counts(self):
        window = self.window()
        window.toolbar.set_tool('eraser')
        window.toolbar.new_instance_actions['nectar'].trigger()
        canvas = window.canvas
        self.assertEqual(canvas.current_tool, 'brush')
        self.assertEqual(canvas.editing_instance_category, 'nectar')
        self.assertTrue(window.toolbar.show_nectar_checkbox.isChecked())
        self.assertTrue(canvas._start_brush_stroke_at(QPointF(8, 8)))
        canvas.draw_on_mask(QPointF(8, 8), QPointF(20, 20))
        canvas._finish_brush_history_step()
        canvas.is_drawing = False
        edited = canvas.editing_mask.copy()
        canvas.undo()
        self.assertFalse(np.any(canvas.editing_mask))
        canvas.redo()
        np.testing.assert_array_equal(canvas.editing_mask, edited)
        self.assertTrue(canvas._is_editing_mask_visible())
        self.assertEqual(window._live_canvas_annotation_counts()['nectar'], 1)
        window.update_instance_list_from_canvas()
        self.assertEqual(window.instance_list.currentItem().data(Qt.ItemDataRole.UserRole)['type'], 'nectar')
        canvas.commit_editing()
        self.assertEqual(canvas.get_annotations()[0]['category'], 'nectar')

    def test_toolbar_signals_and_training_config(self):
        toolbar = AnnotationToolbar()
        self.addCleanup(toolbar.close)
        created, visibility = [], []
        toolbar.new_instance_requested.connect(created.append)
        toolbar.annotation_type_visibility_changed.connect(lambda *args: visibility.append(args))
        toolbar.new_instance_actions['nectar'].trigger()
        toolbar.show_nectar_checkbox.setChecked(True)
        self.assertEqual(created, ['nectar'])
        self.assertEqual(visibility, [('nectar', True)])
        dialog = TrainingConfigDialog()
        self.addCleanup(dialog.close)
        dialog.model_type_combo.setCurrentText('Nectar source')
        config = dialog.get_config()
        self.assertEqual(config['model_type'], 'nectar')
        self.assertEqual(config['name'], 'nectar_segmentation')
        self.assertTrue(config['export_coco'])
        self.assertFalse(dialog.brood_review_check.isVisible())

    def prepare_dataset(self):
        ProjectManager().create_project(self.root, 'nectar test')
        manager = AnnotationManager()
        manager.load_project(self.root)
        frames = self.root / 'frames' / 'video'
        frames.mkdir(parents=True)
        (frames / 'video_metadata.json').write_text(json.dumps({'selected_frames': [0, 1, 2]}))
        for index in range(3):
            cv2.imwrite(str(frames / f'frame_{index:06d}.jpg'), np.full((64, 64, 3), 100, np.uint8))
            anns = [dict(mask_id=1, category='bee', mask=nectar_mask()),
                    dict(mask_id=2, category='hive', mask=nectar_mask()),
                    dict(mask_id=3, category='pollen', mask=nectar_mask())]
            if index < 2:
                anns.append(dict(mask_id=4, category='nectar', mask=nectar_mask()))
            manager.save_frame_annotations(self.root, 'video', index, anns)
        path = export_coco_per_video(self.root, ['video'], 'train')[0]
        export_coco_per_video(self.root, ['video'], 'val')
        return manager, path

    def test_roundtrip_export_and_nectar_only_training_labels(self):
        manager, path = self.prepare_dataset()
        saved = manager.load_frame_annotations(self.root, 'video', 0)
        source = next(a for a in saved if a['category'] == 'nectar')
        np.testing.assert_array_equal(source['mask'], nectar_mask())
        self.assertNotIn('nectar', [a['category'] for a in manager.load_frame_annotations(self.root, 'video', 2)])
        coco = json.loads(path.read_text())
        self.assertEqual(next(c for c in coco['categories'] if c['name'] == 'nectar'),
                         {'id': 13, 'name': 'nectar', 'supercategory': 'resource'})
        worker = YOLOTrainingWorker(self.root, {'model_type': 'nectar'})
        directory, config_path = worker._prepare_yolo_dataset()
        config = yaml.safe_load(config_path.read_text())
        self.assertEqual(config['names'], {0: 'nectar'})
        self.assertEqual(config['nc'], 1)
        for split in ('train', 'val'):
            labels = list((directory / 'labels' / split).glob('*.json'))
            self.assertEqual(len(labels), 2)
            for label in labels:
                instances = json.loads(label.read_text())['instances']
                self.assertEqual(len(instances), 1)
                self.assertEqual(instances[0]['cls'], 0)
                np.testing.assert_array_equal(
                    decode_segmentation(instances[0]['segmentation'], 64, 64), nectar_mask() > 0)

    def test_missing_category_and_validation_masks_fail_clearly(self):
        _, path = self.prepare_dataset()
        coco = json.loads(path.read_text())
        coco['categories'] = [c for c in coco['categories'] if c['name'] != 'nectar']
        path.write_text(json.dumps(coco))
        worker = YOLOTrainingWorker(self.root, {'model_type': 'nectar'})
        with self.assertRaisesRegex(ValueError, 'does not define category.*nectar'):
            worker._prepare_yolo_dataset()
        export_coco_per_video(self.root, ['video'], 'train')
        val_path = export_coco_per_video(self.root, ['video'], 'val')[0]
        coco = json.loads(val_path.read_text())
        coco['annotations'] = [a for a in coco['annotations'] if a['category_id'] != 13]
        val_path.write_text(json.dumps(coco))
        with self.assertRaisesRegex(ValueError, 'No validation images with nectar'):
            worker._prepare_yolo_dataset()

    def test_model_loader_accepts_nectar_rejects_other_classes_without_replacing_model(self):
        toolbar = HiveChamberToolbar()
        self.addCleanup(toolbar.close)
        good = SimpleNamespace(task='segment', names={0: 'nectar'})
        with patch('ultralytics.YOLO', return_value=good):
            self.assertTrue(toolbar._load_nectar_checkpoint_from_path('nectar.pt'))
        for task, names in [('detect', {0: 'nectar'}), ('segment', {0: 'bee'}),
                            ('segment', {0: 'pollen'}), ('segment', {0: 'nectar', 1: 'bee'})]:
            with patch('ultralytics.YOLO', return_value=SimpleNamespace(task=task, names=names)):
                self.assertFalse(toolbar._load_nectar_checkpoint_from_path('wrong.pt'))
            self.assertIs(toolbar.get_nectar_model(), good)
        self.assertTrue(toolbar.nectar_inference_btn.isEnabled())
        self.assertTrue(toolbar.both_inference_btn.isEnabled())
        requests = []
        toolbar.nectar_inference_requested.connect(lambda: requests.append(True))
        toolbar.nectar_inference_btn.click()
        self.assertEqual(requests, [True])

    def test_current_frame_prediction_is_nectar_and_preserves_hive(self):
        window = self.window()
        window.canvas.set_annotations([dict(mask_id=1, category='hive', mask=nectar_mask())])
        window.frames = [self.root / 'frame.png']
        window.current_frame_idx = 0
        result = Results(orig_img=np.zeros((64, 64, 3), np.uint8), path='frame.png',
                         names={0: 'nectar'}, boxes=torch.tensor([[8, 8, 48, 48, .9, 0]]),
                         masks=torch.tensor((nectar_mask() > 0)[None]))
        model = Mock()
        model.predict.return_value = [result]
        window.hive_chamber_toolbar.nectar_model = model
        with patch('gui.main_window.QMessageBox.information'), \
                patch('gui.main_window.QMessageBox.critical') as error:
            window.run_nectar_inference()
        error.assert_not_called()
        anns = window.canvas.get_annotations()
        self.assertEqual([a['category'] for a in anns], ['hive', 'nectar'])
        self.assertEqual(len({a['mask_id'] for a in anns}), 2)
        for ann in anns:
            np.testing.assert_array_equal(ann['mask'], nectar_mask())
        self.assertTrue(window.toolbar.show_nectar_checkbox.isChecked())

    def test_single_class_training_checkpoint(self):
        self.prepare_dataset()
        worker = YOLOTrainingWorker(self.root, {'model_type': 'nectar'})
        _, path = worker._prepare_yolo_dataset()
        torch.set_num_threads(2)
        model = YOLO('yolov8n-seg.yaml')
        with patch('ultralytics.engine.model.checks.check_pip_update_available', return_value=False), \
                patch('ultralytics.data.utils.check_font'):
            model.train(trainer=RasterSegmentationTrainer, data=str(path), epochs=1, imgsz=64,
                        batch=2, workers=0, device='cpu', amp=False, pretrained=False,
                        plots=False, project=str(self.root / 'runs'), name='nectar', **raster_training_options())
        reloaded = YOLO(str(model.trainer.best))
        self.assertEqual(reloaded.names, {0: 'nectar'})
        toolbar = HiveChamberToolbar()
        self.addCleanup(toolbar.close)
        self.assertTrue(toolbar._load_nectar_checkpoint_from_path(model.trainer.best))


if __name__ == '__main__':
    unittest.main()
