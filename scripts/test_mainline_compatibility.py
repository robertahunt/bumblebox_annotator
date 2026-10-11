"""Mainline merge boundaries: saved labels, CSV generations, and wrapping controls."""

import csv
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

import batch_hive_refresh_cli as refresh
from core.annotation_scope import frame_categories
from core.categories import CATEGORIES
from core.import_review import complete_review, review_status
from core.project_manager import ProjectManager
from core.project_sync import _write_json
from core.video_inference_exporter import VideoInferenceExporter
from gui.batch_video_inference_worker import BatchVideoInferenceWorker
from gui.project_overview_dialog import ProjectOverviewDialog
from gui.toolbar import AnnotationToolbar, FlowLayout
from gui.hive_chamber_toolbar import HiveChamberToolbar
from scripts.test_annotation_policy import PolicyFixture
from scripts import test_annotation_policy_gui as policy_gui
from scripts.test_project_import import mask_annotation


class CsvCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_refresh_retains_disconnected_polygons_and_legacy_single_polygon(self):
        polygon = '2 2 5 2 5 5 2 5'
        expected = np.zeros((24, 24), np.uint8)
        expected[2:6, 2:6] = 255
        np.testing.assert_array_equal(refresh.polygon_to_mask(polygon, expected.shape), expected)
        expected[15:19, 15:19] = 255
        text = polygon + '|15 15 18 15 18 18 15 18'
        row = dict(pred_polygon=text, bbox_x='0', bbox_y='0', bbox_width='24', bbox_height='24')
        np.testing.assert_array_equal(refresh.row_to_bee(row, expected.shape).mask, expected)
        self.assertEqual(int(np.count_nonzero(expected)), 32)

    def test_polygon_invalid_coordinates_return_none_and_clipping_is_safe(self):
        for value in ('', '1 2 3', '1 2 nan 4 5 6', '1 2 inf 4 5 6', 'a b c', '1 2 3 4 5 6|'):
            with self.subTest(value=value):
                self.assertIsNone(refresh.polygon_to_mask(value, (8, 8)))
        self.assertIsNotNone(refresh.polygon_to_mask('-5,-5 12,-5 12,12 -5,12', (8, 8)))

    def test_exporter_polygon_roundtrip_does_not_fill_gap(self):
        expected = np.zeros((32, 32), np.uint8)
        expected[2:7, 2:7] = expected[18:25, 18:25] = 255
        exporter = VideoInferenceExporter(self.root, polygon_epsilon=0)
        text = exporter._mask_polygon_string(expected)
        self.assertIn('|', text)
        np.testing.assert_array_equal(refresh.polygon_to_mask(text, expected.shape), expected)

    def test_incompatible_structure_outputs_are_preserved_before_reset_or_append(self):
        for name in ('pollen', 'hive', 'chamber'):
            with self.subTest(category=name):
                folder = self.root / name
                folder.mkdir()
                old = folder / f'{name}_detections.csv'
                old.write_text('video_id,chamber_id,frame_number,pixels\nvideo,1,0,20\n')
                sentinel = folder / 'bee_detections.csv'
                sentinel.write_text('unchanged')
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                for resume in (False, True):
                    worker = BatchVideoInferenceWorker({'output_folder': str(folder),
                                                       'resume_completed_videos': resume})
                    with self.assertRaisesRegex(ValueError, 'new output folder'):
                        worker._prepare_resume_state([])
                    with self.assertRaisesRegex(ValueError, 'Incompatible batch CSV'):
                        worker._reset_streaming_csvs(include_manifest=True)
                with self.assertRaisesRegex(ValueError, 'Incompatible batch CSV'):
                    VideoInferenceExporter(folder)
                self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_current_schema_append_works_and_pollen_summary_stays_separate(self):
        exporter = VideoInferenceExporter(self.root)
        masks = {('video', 1): {'accumulated_mask': np.ones((12, 12), np.float32),
                               'frame_count': 1}}
        exporter.export_pollen_detections(masks)
        exporter.export_pollen_detections(masks, append=True)
        frame = SimpleNamespace(video_id='video', chamber_id=1, frame_number=2,
                                pollen_count=1, pollen_pixels=144, pollen_area_mm2=None)
        exporter.export_pollen_frame_summary([frame])
        with (self.root / 'pollen_detections.csv').open() as handle:
            reader = csv.DictReader(handle)
            self.assertIn('pollen_instance_id', reader.fieldnames)
            self.assertEqual(len(list(reader)), 2)
        with (self.root / 'pollen_frame_summary.csv').open() as handle:
            reader = csv.DictReader(handle)
            self.assertIn('frame_number', reader.fieldnames)
            self.assertEqual(len(list(reader)), 1)

    def test_schema_changed_after_exporter_construction_is_not_overwritten(self):
        exporter = VideoInferenceExporter(self.root)
        path = self.root / 'pollen_detections.csv'
        path.write_text('legacy\noriginal\n')
        with self.assertRaises(ValueError):
            exporter.export_chamber_detections({})
        self.assertEqual(path.read_text(), 'legacy\noriginal\n')
        self.assertFalse((self.root / 'chamber_detections.csv').exists())


class MainlineGuiCompatibilityTests(PolicyFixture):
    make_window = policy_gui.AnnotationPolicyGuiTests.make_window

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_aruco_blocking_save_retains_all_categories_scope_and_creator(self):
        for scope in ('frame', 'video'):
            with self.subTest(scope=scope):
                info = ProjectManager().create_project(self.dest, 'Destination', hive_annotation_scope=scope)
                _write_json(self.dest / 'frames/video/video_metadata.json',
                            {'selected_frames': [2, 5], 'split': 'train', 'project_import': True})
                window = self.make_window()
                annotations = [mask_annotation(index + 1, category)
                               for index, category in enumerate(CATEGORIES)]
                window.canvas.set_annotations(annotations)
                window._mark_current_annotations_dirty()
                csv_path = self.root / 'tags.csv'
                csv_path.write_text('video_id,frame,tag_id,center_x,center_y\nvideo,2,9,12,12\n')
                report = window._apply_aruco_csv_to_current_frame(csv_path, show_dialog=False)
                self.assertEqual(report['markers_found'], 1)
                with patch.object(window.save_worker, 'wait_until_idle',
                                  wraps=window.save_worker.wait_until_idle) as wait:
                    self.assertTrue(window._save_current_frame_annotations_blocking())
                self.assertGreaterEqual(wait.call_count, 2)
                local = self.manager.load_frame_annotations(self.dest, 'video', 2)
                shared, tracking = self.manager.load_video_annotations(self.dest, 'video')
                self.assertEqual(tracking, {'9': 1})
                self.assertEqual({a['category'] for a in local}, frame_categories(info))
                self.assertEqual({a['category'] for a in shared}, set(CATEGORIES) - frame_categories(info))
                for ann in local + shared:
                    self.assertEqual(ann['provenance']['created_by']['name'], 'Original author')
                    np.testing.assert_array_equal(ann['mask'] > 0, mask_annotation()['mask'] > 0)

    def test_aruco_blocking_save_invalidates_import_review_when_markers_change(self):
        self.import_hives()
        window = self.make_window()
        window._save_current_frame_annotations_blocking()
        complete_review(self.dest, 'video', 2)
        window._update_import_review_controls()
        self.assertTrue(window.frame_review_check.isChecked())
        annotations = window.canvas.get_annotations()
        annotations.append(mask_annotation(25, 'bee'))
        annotations[-1]['marker_id'] = 9
        window.canvas.set_annotations(annotations)
        window._mark_current_annotations_dirty()
        window._save_current_frame_annotations_blocking()
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')
        self.assertFalse(window.frame_review_check.isChecked())

    def test_overview_counts_selected_frame_objects_without_duplicating_derived_boxes(self):
        _write_json(self.dest / 'annotations/project.json', self.source_info)
        self.manager.save_frame_annotations(self.dest, 'video', 2,
                                            [mask_annotation(i + 1, category) for i, category in enumerate(CATEGORIES)],
                                            contributor=None)
        self.manager.save_frame_annotations(self.dest, 'video', 5,
                                            [mask_annotation(30, 'nectar')], contributor=None)
        _write_json(self.dest / 'frames/video/video_metadata.json',
                    {'selected_frames': [2], 'split': 'train', 'project_import': True})
        self.manager.save_video_annotations(self.dest, 'video', [mask_annotation()], contributor=None)
        dialog = ProjectOverviewDialog(self.dest, ProjectManager(self.dest))
        self.addCleanup(dialog.close)
        stats = dialog._collect_stats()
        self.assertEqual(stats['video_level']['train']['hive']['total'], 0)
        for category in frame_categories(self.source_info) - {'bee'}:
            self.assertEqual(stats['frame_objects']['train'][category]['total'], 1)
            self.assertEqual(stats['frame_objects']['train'][category]['masks'], 1)
        self.assertEqual(stats['annotations']['train_segmentations'], 1)

    def test_wrapping_toolbars_keep_new_category_controls_and_visibility_groups(self):
        host = QWidget()
        self.addCleanup(host.close)
        layout = QVBoxLayout(host)
        toolbar, structures = AnnotationToolbar(), HiveChamberToolbar()
        layout.addWidget(toolbar)
        layout.addWidget(structures)
        host.resize(900, 900)
        host.show()
        self.app.processEvents()
        for width in (900, 1280):
            host.resize(width, 900)
            self.app.processEvents()
            for flow in [toolbar.layout().itemAt(i).layout() for i in range(3)] + [structures.layout()]:
                self.assertIsInstance(flow, FlowLayout)
                for index in range(flow.count()):
                    item = flow.itemAt(index)
                    if not item.isEmpty():
                        self.assertLessEqual(item.geometry().right(), flow.geometry().right())
            host.grab().save(f'/tmp/merged-toolbars-{width}.png')
        self.assertTrue(structures.load_nectar_btn.isVisible())
        self.assertTrue(toolbar.show_nectar_checkbox.isVisible())
        for key in ('brood', 'queen_brood'):
            checkbox = toolbar.brood_visibility_checkboxes[key]
            checkbox.setCheckState(Qt.CheckState.Checked)
            category = toolbar.brood_visibility_groups[key][0]
            toolbar.brood_visibility_actions[category].setChecked(False)
            self.assertEqual(checkbox.checkState(), Qt.CheckState.PartiallyChecked)


if __name__ == '__main__':
    unittest.main()
