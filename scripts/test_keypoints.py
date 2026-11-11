#!/usr/bin/env python3
"""Tests for optional per-instance keypoints: data model, canvas layer, round trips."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from core import keypoints as kp

BODY = kp.KeypointConfig(names=('head', 'thorax', 'abdomen_tip'),
                         edges=(('head', 'thorax'), ('thorax', 'abdomen_tip')))


class KeypointModelTests(unittest.TestCase):
    def test_default_config_round_trips_through_project_info(self):
        config = kp.default_config()
        info = {'name': 'p', kp.PROJECT_KEY: config.to_dict()}
        loaded = kp.config_from_project_info(info)
        self.assertEqual(loaded.names, ('head', 'thorax', 'abdomen_tip',
                                        'left_antenna_tip', 'right_antenna_tip'))
        self.assertEqual(loaded.edges, (('head', 'thorax'), ('thorax', 'abdomen_tip'),
                                        ('head', 'left_antenna_tip'), ('head', 'right_antenna_tip')))
        self.assertEqual(loaded.marker_keypoint, 'thorax')
        self.assertEqual(loaded.coco_category_fields()['skeleton'], [[1, 2], [2, 3], [1, 4], [1, 5]])
        self.assertEqual(loaded.symmetries, (('left_antenna_tip', 'right_antenna_tip'),))
        self.assertEqual(config.to_dict()['symmetries'], [['left_antenna_tip', 'right_antenna_tip']])

    def test_unspecified_lines_use_defaults_between_defined_keypoints_only(self):
        self.assertEqual(kp.KeypointConfig(names=('head', 'thorax', 'sting')).edges, (('head', 'thorax'),))
        self.assertEqual(kp.KeypointConfig(names=('head', 'thorax'), edges=()).edges, ())
        loaded = kp.KeypointConfig.from_dict({'names': ['head', 'left_antenna_tip']})
        self.assertEqual(loaded.edges, (('head', 'left_antenna_tip'),))

    def test_symmetric_pairs_need_both_sides(self):
        self.assertEqual(kp.symmetric_pairs(('head', 'left_antenna_tip')), ())
        self.assertEqual(kp.symmetric_pairs(('left_eye', 'right_eye', 'left_antenna_tip')),
                         (('left_eye', 'right_eye'),))
        self.assertEqual(BODY.symmetries, ())

    def test_missing_disabled_or_malformed_settings_mean_off(self):
        self.assertIsNone(kp.config_from_project_info({}))
        self.assertIsNone(kp.config_from_project_info(None))
        disabled = kp.default_config().to_dict()
        disabled['enabled'] = False
        self.assertIsNone(kp.config_from_project_info({kp.PROJECT_KEY: disabled}))
        self.assertIsNone(kp.config_from_project_info({kp.PROJECT_KEY: {'names': ['a', 'a']}}))
        self.assertIsNone(kp.config_from_project_info({kp.PROJECT_KEY: 'yes'}))

    def test_config_validation_and_unknown_fields_kept(self):
        with self.assertRaises(ValueError):
            kp.KeypointConfig(names=('head', 'thorax'), edges=(('head', 'wing'),))
        with self.assertRaises(ValueError):
            kp.KeypointConfig(names=('head', 'bad name!'))
        config = kp.KeypointConfig.from_dict({'names': ['Head', 'Left Antenna'], 'future': 1,
                                              'edges': [['head', 'left_antenna']]})
        self.assertEqual(config.names, ('head', 'left_antenna'))
        self.assertIsNone(config.marker_keypoint)  # no thorax defined
        self.assertEqual(config.to_dict()['future'], 1)

    def test_parse_names_and_edges(self):
        self.assertEqual(kp.parse_names('head, thorax\nabdomen tip'), ('head', 'thorax', 'abdomen_tip'))
        self.assertEqual(kp.parse_edges('head-thorax; thorax->abdomen_tip'),
                         (('head', 'thorax'), ('thorax', 'abdomen_tip')))
        self.assertEqual(kp.format_edges((('a', 'b'),)), 'a-b')
        with self.assertRaises(ValueError):
            kp.parse_edges('head thorax')

    def test_points_are_normalized_and_invalid_ones_dropped(self):
        meta = {'keypoints': {'head': [1, 2, 2], 'thorax': {'x': 3, 'y': 4, 'v': 1},
                              'abdomen_tip': [5, 6, 0], 'junk': 'x', 'nan': [float('nan'), 1, 2]}}
        self.assertEqual(kp.get_keypoints(meta), {'head': [1.0, 2.0, 2], 'thorax': [3.0, 4.0, 1]})
        kp.set_keypoints(meta, {})
        self.assertNotIn('keypoints', meta)

    def test_next_unlabelled_wraps_and_merge_prefers_target(self):
        names = ('head', 'thorax', 'abdomen_tip')
        self.assertEqual(kp.next_unlabelled({}, names), 'head')
        self.assertEqual(kp.next_unlabelled({'head': [0, 0, 2]}, names, after='head'), 'thorax')
        self.assertEqual(kp.next_unlabelled({'thorax': [0, 0, 2]}, names, after='abdomen_tip'), 'head')
        self.assertIsNone(kp.next_unlabelled({n: [0, 0, 2] for n in names}, names))
        merged = kp.merge_keypoints({'head': [1, 1, 2]}, {'head': [9, 9, 2], 'thorax': [2, 2, 1]})
        self.assertEqual(merged, {'head': [1, 1, 2], 'thorax': [2, 2, 1]})

    def test_coco_keypoints_and_marker_center(self):
        flat, n = kp.coco_keypoints({'head': [1.234, 2, 2], 'abdomen_tip': [5, 6, 1]},
                                    ('head', 'thorax', 'abdomen_tip'))
        self.assertEqual(flat, [1.23, 2.0, 2, 0, 0, 0, 5.0, 6.0, 1])
        self.assertEqual(n, 2)
        self.assertEqual(kp.marker_center({'marker': {'center': (10, 20.5)}}), (10.0, 20.5))
        self.assertIsNone(kp.marker_center({'marker': {'center': None}}))
        self.assertIsNone(kp.marker_center({}))

    def test_nearest_keypoint(self):
        found = kp.nearest_keypoint({1: {'head': [10, 10, 2]}, 2: {'head': [13, 10, 2]}}, 12.6, 10, 5)
        self.assertEqual(found[:2], (2, 'head'))
        self.assertIsNone(kp.nearest_keypoint({1: {'head': [10, 10, 2]}}, 30, 30, 5))


class KeypointCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_canvas(self):
        from gui.canvas import ImageCanvas
        canvas = ImageCanvas()
        canvas.resize(400, 300)
        canvas.load_image(np.zeros((200, 300, 3), dtype=np.uint8))
        bee1 = np.zeros((200, 300), dtype=np.uint8)
        bee1[20:80, 20:120] = 1
        bee2 = np.zeros((200, 300), dtype=np.uint8)
        bee2[120:180, 150:280] = 1
        canvas.set_annotations([
            {'mask': bee1, 'mask_id': 1, 'category': 'bee',
             'marker': {'type': 'aruco', 'id': 7, 'center': (60.0, 50.0)}},
            {'mask': bee2, 'mask_id': 2, 'category': 'bee',
             'keypoints': {'head': [160, 130, 2]}},
        ])
        canvas.resetTransform()  # 1 scene px = 1 view px, so clicks map exactly
        canvas.centerOn(150, 100)
        self.app.processEvents()
        return canvas

    def click(self, canvas, x, y, button=Qt.MouseButton.LeftButton):
        pos = canvas.mapFromScene(QPointF(x, y))
        QTest.mouseClick(canvas.viewport(), button, pos=pos, delay=5)
        self.app.processEvents()

    def test_layer_is_inert_without_config(self):
        canvas = self.make_canvas()
        canvas.set_tool('keypoint')
        self.assertFalse(canvas.keypoint_layer.enabled)
        self.assertEqual(canvas.keypoint_layer.items, [])
        self.click(canvas, 50, 50)
        self.assertNotIn('keypoints', canvas.annotation_metadata[1])
        # Existing keypoints still pass through get_annotations unchanged.
        anns = {a['mask_id']: a for a in canvas.get_annotations()}
        self.assertEqual(anns[2]['keypoints'], {'head': [160, 130, 2]})

    def test_click_sequence_places_points_in_order_and_undo(self):
        canvas = self.make_canvas()
        layer = canvas.keypoint_layer
        layer.set_config(BODY)
        canvas.set_tool('keypoint')
        changes = []
        canvas.annotation_changed.connect(lambda: changes.append(1))

        self.click(canvas, 30, 40)    # on bee 1: starts it and places head
        self.assertEqual(layer.target_id, 1)
        self.click(canvas, 150, 150)  # target incomplete: thorax for bee 1 even over bee 2
        self.click(canvas, 110, 60)
        points = layer.keypoints_for(1)
        self.assertEqual(sorted(points), ['abdomen_tip', 'head', 'thorax'])
        self.assertAlmostEqual(points['head'][0], 30, delta=1)
        self.assertAlmostEqual(points['thorax'][1], 150, delta=1)
        self.assertIsNone(layer.active_name)
        self.assertEqual(len(changes), 3)

        self.click(canvas, 200, 150)  # bee 1 complete: a click on bee 2 starts bee 2
        self.assertEqual(layer.target_id, 2)
        self.assertEqual(sorted(layer.keypoints_for(2)), ['head', 'thorax'])  # head kept, thorax placed

        self.assertTrue(layer.undo())
        self.assertEqual(sorted(layer.keypoints_for(2)), ['head'])
        self.assertTrue(layer.redo())
        self.assertEqual(sorted(layer.keypoints_for(2)), ['head', 'thorax'])

        saved = {a['mask_id']: a for a in canvas.get_annotations()}
        self.assertEqual(sorted(saved[1]['keypoints']), ['abdomen_tip', 'head', 'thorax'])
        self.assertGreater(len(layer.items), 0)

    def test_drag_moves_point_and_is_one_undo_step(self):
        canvas = self.make_canvas()
        layer = canvas.keypoint_layer
        layer.set_config(kp.default_config())
        canvas.set_tool('keypoint')
        start = canvas.mapFromScene(QPointF(160, 130))
        end = canvas.mapFromScene(QPointF(170, 140))
        QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(canvas.viewport(), QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2))
        QTest.mouseMove(canvas.viewport(), end)
        QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.app.processEvents()
        head = layer.keypoints_for(2)['head']
        self.assertAlmostEqual(head[0], 170, delta=1)
        self.assertAlmostEqual(head[1], 140, delta=1)
        self.assertEqual(len(layer.undo_stack), 1)
        layer.undo()
        self.assertEqual(layer.keypoints_for(2)['head'][:2], [160.0, 130.0])

    def test_marker_fill_keys_visibility_and_hidden_mode(self):
        canvas = self.make_canvas()
        layer = canvas.keypoint_layer
        layer.set_config(kp.default_config())
        self.assertEqual(layer.count_marker_candidates(), 1)
        self.assertEqual(layer.fill_from_markers(), 1)
        self.assertEqual(layer.keypoints_for(1)['thorax'], [60.0, 50.0, 2])
        self.assertEqual(layer.fill_from_markers(), 0)  # only missing points

        canvas.set_tool('keypoint')
        layer.set_target(1)
        self.assertEqual(layer.active_name, 'head')
        layer.set_place_occluded(True)
        self.click(canvas, 25, 30)
        self.assertEqual(layer.keypoints_for(1)['head'][2], kp.OCCLUDED)
        layer.toggle_visibility(1, 'head')
        self.assertEqual(layer.keypoints_for(1)['head'][2], kp.VISIBLE)

        backspace = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(backspace)  # removes the last placed point (head)
        self.assertNotIn('head', layer.keypoints_for(1))
        escape = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(escape)
        self.assertEqual(layer.target_id, 0)

    def test_hidden_bee_and_space_toggle_hide_keypoints(self):
        canvas = self.make_canvas()
        layer = canvas.keypoint_layer
        layer.set_config(kp.default_config())
        self.assertEqual(len(layer.items), 1)  # bee 2's head
        canvas.set_instance_visible(2, 'bee', False)
        self.assertEqual(len(layer.items), 0)
        canvas.set_instance_visible(2, 'bee', True)
        self.assertEqual(len(layer.items), 1)
        canvas.toggle_annotation_overlays()
        self.assertEqual(len(layer.items), 0)
        canvas.toggle_annotation_overlays()
        self.assertEqual(len(layer.items), 1)
        layer.set_visible(False)
        self.assertEqual(len(layer.items), 0)

    def test_id_merge_keeps_target_points_and_fills_missing(self):
        canvas = self.make_canvas()
        layer = canvas.keypoint_layer
        layer.set_config(kp.default_config())
        layer.place(1, 'head', 30, 30)
        layer.place(1, 'thorax', 60, 50)
        self.assertTrue(canvas.reassign_instance_id(1, 2))
        merged = layer.keypoints_for(2)
        self.assertEqual(merged['head'], [160.0, 130.0, 2])  # bee 2 kept its own head
        self.assertEqual(merged['thorax'], [60.0, 50.0, 2])  # filled from bee 1

    def test_split_does_not_copy_keypoints_to_new_instance(self):
        canvas = self.make_canvas()
        meta = canvas._metadata_for_split_instance(2, 9, 'bee', np.ones((200, 300), bool))
        self.assertNotIn('keypoints', meta)

    def test_frame_change_resets_target_and_history(self):
        canvas = self.make_canvas()
        layer = canvas.keypoint_layer
        layer.set_config(kp.default_config())
        layer.set_target(1)
        layer.place(1, 'head', 30, 30)
        canvas.set_annotations([])
        self.assertEqual(layer.target_id, 0)
        self.assertEqual(layer.undo_stack, [])
        self.assertEqual(layer.items, [])


class KeypointToolbarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_toolbar_follows_layer_state(self):
        from gui.canvas import ImageCanvas
        from gui.keypoint_toolbar import KeypointToolbar
        canvas = ImageCanvas()
        toolbar = KeypointToolbar(canvas.keypoint_layer)
        self.assertFalse(toolbar.place_btn.isEnabled())
        self.assertIn('off', toolbar.status_label.text())
        canvas.keypoint_layer.set_config(kp.KeypointConfig(names=('head', 'thorax', 'abdomen_tip', 'sting')))
        self.assertTrue(toolbar.place_btn.isEnabled())
        self.assertEqual(toolbar.next_combo.count(), 4)
        emitted = []
        toolbar.tool_changed.connect(emitted.append)
        toolbar.place_btn.click()
        self.assertEqual(emitted, ['keypoint'])


class KeypointSettingsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_defaults_restore_and_layout_does_not_overlap(self):
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import QLabel
        from gui.keypoint_settings_dialog import KeypointSettingsDialog
        original = self.app.font()
        self.addCleanup(self.app.setFont, original)
        for size in (9, 14):
            self.app.setFont(QFont(original.family(), size))
            dialog = KeypointSettingsDialog(BODY, enabled=True)
            self.addCleanup(dialog.close)
            dialog.show()
            self.app.processEvents()
            # Word-wrapped labels make Qt size the window too short; keep text on explicit lines.
            self.assertFalse(any(label.wordWrap() for label in dialog.findChildren(QLabel)))
            names = dialog.names_edit.geometry()
            below = [w for w in dialog.findChildren(QLabel) if w.text().startswith('One per line')][0]
            self.assertLessEqual(names.bottom(), below.geometry().top())
            self.assertGreaterEqual(names.height(), dialog.names_edit.minimumHeight())
        dialog.restore_defaults()
        dialog._accept()
        config, enabled = dialog.result_config()
        self.assertEqual(config.names, kp.DEFAULT_KEYPOINT_NAMES)
        self.assertEqual(config.marker_keypoint, 'thorax')
        self.assertTrue(enabled)


class KeypointProjectTests(unittest.TestCase):
    """Settings in project.json, edit tracking, and COCO export on disposable projects."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import tempfile
        import cv2
        from core.annotation import AnnotationManager
        from core.contributors import new_session
        from core.project_manager import ProjectManager
        from core.project_sync import _write_json
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'project'
        ProjectManager().create_project(self.project, 'Keypoints', hive_annotation_scope='frame')
        frames = self.project / 'frames/video'
        frames.mkdir(parents=True, exist_ok=True)
        for index in (2, 5):
            cv2.imwrite(str(frames / f'frame_{index:06d}.jpg'), np.full((64, 96, 3), index, np.uint8))
        _write_json(frames / 'video_metadata.json', {'selected_frames': [2, 5], 'split': 'train',
                                                    'total_frames': 10})
        # The frame list reads a video's split from its input file location.
        (self.project / 'input_data/train').mkdir(parents=True, exist_ok=True)
        (self.project / 'input_data/train/video.mp4').touch()
        self.manager = AnnotationManager()
        self.actor = new_session('Annotator')

    def bee(self, keypoints=None):
        from scripts.test_project_import import mask_annotation
        ann = mask_annotation(8, 'bee')
        if keypoints is not None:
            ann['keypoints'] = keypoints
        return ann

    def test_settings_round_trip_off_keeps_names_and_policy_save_keeps_settings(self):
        from core.annotation_policy import category_policies, save_policy
        from core.annotation_scope import read_project_info
        self.assertIsNone(kp.read_project_config(self.project))
        config = kp.KeypointConfig(names=('head', 'thorax', 'abdomen_tip', 'sting'),
                                   edges=(('head', 'thorax'), ('thorax', 'abdomen_tip')))
        kp.save_project_config(self.project, config, enabled=True)
        self.assertEqual(kp.read_project_config(self.project).names[-1], 'sting')
        # The annotation policy editor rewrites project.json; keypoint settings must survive.
        info = read_project_info(self.project)
        save_policy(self.project, category_policies(info))
        self.assertEqual(kp.read_project_config(self.project), config)
        kp.save_project_config(self.project, config, enabled=False)
        self.assertIsNone(kp.read_project_config(self.project))
        self.assertEqual(kp.stored_project_config(read_project_info(self.project)).names, config.names)

    def test_keypoints_survive_save_and_load(self):
        points = {'head': [20.5, 12.0, 2], 'abdomen_tip': [30.0, 40.0, 1]}
        self.manager.save_frame_annotations(self.project, 'video', 2, [self.bee(points)], contributor=None)
        loaded = self.manager.load_frame_annotations(self.project, 'video', 2)
        bee = next(a for a in loaded if a.get('category', 'bee') == 'bee')
        self.assertEqual(kp.get_keypoints(bee), points)

    def test_finished_frame_changes_when_only_keypoints_change(self):
        from core.annotation_work import local_index, mark_finished, summarize
        self.manager.save_frame_annotations(self.project, 'video', 2, [self.bee()], contributor=None)
        mark_finished(self.project, 'video', 2, 'bee', self.actor)
        entry = lambda: local_index(self.project)['videos']['video']
        self.assertEqual(summarize(entry(), 'bee', [2])['state'], 'finished')
        self.manager.save_frame_annotations(self.project, 'video', 2, [self.bee()], contributor=None)
        self.assertEqual(summarize(entry(), 'bee', [2])['state'], 'finished')  # no false change
        self.manager.save_frame_annotations(self.project, 'video', 2,
                                            [self.bee({'head': [20, 12, 2]})], contributor=None)
        self.assertEqual(summarize(entry(), 'bee', [2])['state'], 'changed')

    def test_provenance_digest_sees_keypoint_edits(self):
        from core.contributors import annotation_digest
        plain = annotation_digest(self.bee())
        self.assertEqual(plain, annotation_digest(self.bee()))
        with_head = annotation_digest(self.bee({'head': [1, 2, 2]}))
        self.assertNotEqual(plain, with_head)
        self.assertNotEqual(with_head, annotation_digest(self.bee({'head': [1, 3, 2]})))

    def test_coco_exports_add_keypoints_only_when_enabled(self):
        from core.project_sync import _read_json
        from training.coco_video_export import export_coco_per_video, export_coco_with_tracking
        points = {'head': [20.5, 12.0, 2], 'abdomen_tip': [30.0, 40.0, 1]}
        for index in (2, 5):
            self.manager.save_frame_annotations(self.project, 'video', index, [self.bee(points)],
                                                contributor=None)
        plain = _read_json(export_coco_per_video(self.project, ['video'], 'train')[0])
        self.assertNotIn('keypoints', plain['info'])
        self.assertTrue(all('keypoints' not in a for a in plain['annotations']))
        self.assertTrue(all('skeleton' not in c for c in plain['categories']))

        kp.save_project_config(self.project, kp.default_config(), enabled=True)
        for data in (_read_json(export_coco_per_video(self.project, ['video'], 'train')[0]),
                     _read_json(export_coco_with_tracking(self.project, ['video'], 'train'))):
            bee_cat = next(c for c in data['categories'] if c['name'] == 'bee')
            names = ['head', 'thorax', 'abdomen_tip', 'left_antenna_tip', 'right_antenna_tip']
            self.assertEqual(bee_cat['keypoints'], names)
            self.assertEqual(bee_cat['skeleton'], [[1, 2], [2, 3], [1, 4], [1, 5]])
            self.assertEqual(data['info']['keypoints']['names'], names)
            self.assertEqual(data['info']['keypoints']['symmetries'],
                             [['left_antenna_tip', 'right_antenna_tip']])
            bees = [a for a in data['annotations'] if a['category_id'] == bee_cat['id']]
            self.assertEqual(len(bees), 2)
            for ann in bees:
                self.assertEqual(ann['keypoints'],
                                 [20.5, 12.0, 2, 0, 0, 0, 30.0, 40.0, 1, 0, 0, 0, 0, 0, 0])
                self.assertEqual(ann['num_keypoints'], 2)

    def test_main_window_turns_keypoints_on_and_off_per_project(self):
        from unittest.mock import patch
        from PyQt6.QtCore import QSettings
        from PyQt6.QtWidgets import QMessageBox
        from gui.main_window import MainWindow
        settings = QSettings(str(self.root / 'settings.ini'), QSettings.Format.IniFormat)
        mock = patch('gui.main_window.QSettings', lambda: settings)
        mock.start()
        self.addCleanup(mock.stop)
        with patch.object(MainWindow, 'load_settings'):
            window = MainWindow()

        def close():
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                window.close()
            self.app.processEvents()
        self.addCleanup(close)
        window.set_contributor(self.actor)

        def load():
            with patch.object(QMessageBox, 'critical') as errors, \
                    patch.object(window, 'show_annotation_policy'):
                window.load_project(self.project)
                errors.assert_not_called()

        load()
        self.assertFalse(window.canvas.keypoint_layer.enabled)
        self.assertFalse(window.keypoint_toolbar_action.isEnabled())
        self.assertFalse(window.keypoint_toolbar.isVisibleTo(window))

        kp.save_project_config(self.project, kp.default_config(), enabled=True)
        load()
        self.assertTrue(window.canvas.keypoint_layer.enabled)
        self.assertTrue(window.keypoint_toolbar_action.isChecked())
        self.assertTrue(window.keypoint_toolbar.isVisibleTo(window))

        # Tool switching between toolbars, and undo routed to keypoints while placing them.
        window.keypoint_toolbar.place_btn.click()
        self.assertEqual(window.canvas.current_tool, 'keypoint')
        self.assertTrue(window.keypoint_toolbar.place_btn.isChecked())
        window.toolbar.pan_btn.click()
        self.assertEqual(window.canvas.current_tool, 'pan')
        self.assertFalse(window.keypoint_toolbar.place_btn.isChecked())

        kp.save_project_config(self.project, kp.default_config(), enabled=False)
        load()
        self.assertFalse(window.canvas.keypoint_layer.enabled)
        self.assertFalse(window.keypoint_toolbar_action.isEnabled())
        self.assertFalse(window.keypoint_toolbar.isVisibleTo(window))


if __name__ == "__main__":
    unittest.main()
