#!/usr/bin/env python3
"""CPU-only temporal overlay, replay, GUI and CLI regressions."""

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QGroupBox

from core.temporal_hive_prior import TemporalHivePrior, TemporalHiveSnapshot, TemporalChamberStabilizer
from core.temporal_hive_visualization import (
    TemporalHiveOverlayReader, TemporalHiveOverlayWriter,
    draw_temporal_hive_overlay, draw_hive_overlay_label, project_snapshot,
    resolve_hive_overlay_mode,
)
from core.visualization_generator import VisualizationGenerator
from gui.batch_video_inference_dialog import BatchVideoInferenceConfigDialog
from gui.batch_video_inference_worker import BatchVideoInferenceWorker
from render_temporal_hive_video import render_video
import batch_video_inference_cli as batch_cli


class TemporalHiveVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = np.full((240, 640, 3), 90, np.uint8)
        self.labels = np.ones((12, 32), np.uint8)
        self.labels[3:9, 4:11] = 2
        self.labels[:, 25:] = 0
        self.snapshot = TemporalHiveSnapshot(2, (0, 0, 640, 240), self.labels)
        self.mask = np.zeros(self.frame.shape[:2], np.uint8)
        self.mask[50:190, 100:260] = 255

    def writer(self, path=None, source=None):
        return TemporalHiveOverlayWriter(path or self.root / 'cache.zip',
                                         source or self.root / 'source.avi', 'context', fps=6)

    def make_source_and_cache(self):
        source = self.root / 'source.avi'
        video = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 6, (640, 240))
        self.assertTrue(video.isOpened())
        try:
            with self.writer(source=source) as cache:
                for frame_number in range(1, 4):
                    video.write(self.frame)
                    cache.write(frame_number, self.frame.shape,
                                [] if frame_number == 1 else [self.snapshot])
        finally:
            video.release()
        return source, self.root / 'cache.zip'

    def test_snapshot_thresholds_context_and_nonmutating_decay(self):
        prior = TemporalHivePrior(resolution=(8, 4), window_seconds=10,
                                  min_prior_weight=1, cleanup_kernel_size=0, min_component_pixels=0)
        chamber = {'bbox': [0, 0, 8, 4], 'mask': np.ones((4, 8), np.uint8)}
        hive = np.zeros((4, 8), np.uint8)
        hive[:, :4] = 255
        before = prior.visualization_snapshot('c', 0, chamber, hive.shape, 0)
        self.assertFalse(before.labels.any())
        prior.update('c', 0, chamber, hive, [], hive.shape, 0)
        state = prior._states['c', 0]
        weights = state.weight_sum.copy()
        snapshot = prior.visualization_snapshot('c', 0, chamber, hive.shape, 0)
        self.assertTrue(np.all(snapshot.labels[:, :4] == 2))
        self.assertTrue(np.all(snapshot.labels[:, 4:] == 1))
        self.assertFalse(prior.visualization_snapshot('other', 0, chamber, hive.shape, 0).labels.any())
        self.assertFalse(prior.visualization_snapshot('c', 0, chamber, hive.shape, 100).labels.any())
        np.testing.assert_array_equal(state.weight_sum, weights)
        self.assertEqual(state.last_time_seconds, 0)
        snapshot.labels[:] = 0
        self.assertTrue(prior.visualization_snapshot('c', 0, chamber, hive.shape, 0).labels.any())

    def test_projection_uses_chamber_bbox_and_categorical_cells(self):
        snapshot = TemporalHiveSnapshot(1, (4, 2, 8, 6), np.array([[0, 1], [2, 2]], np.uint8))
        projected = project_snapshot(snapshot, (8, 12))
        expected = np.zeros((8, 12), np.uint8)
        expected[2:4, 6:8] = 1
        expected[4:6, 4:8] = 2
        np.testing.assert_array_equal(projected, expected)
        with self.assertRaises(ValueError):
            project_snapshot(snapshot, (4, 4))

    def test_unknown_never_falls_back_to_current_hive(self):
        frame = self.frame.copy()
        self.assertFalse(draw_temporal_hive_overlay(frame, [], {2: self.mask}))
        np.testing.assert_array_equal(frame, self.frame)
        self.assertFalse(draw_temporal_hive_overlay(frame, [], {2: self.mask}, compare=True))
        self.assertTrue(np.any(frame != self.frame))

    def test_prior_fill_and_comparison_outline_leave_background_clear(self):
        prior_frame = self.frame.copy()
        self.assertTrue(draw_temporal_hive_overlay(prior_frame, [self.snapshot]))
        self.assertFalse(np.array_equal(prior_frame[100, 150], self.frame[100, 150]))
        np.testing.assert_array_equal(prior_frame[100, 350], self.frame[100, 350])
        np.testing.assert_array_equal(prior_frame[100, 550], self.frame[100, 550])
        compare = self.frame.copy()
        draw_temporal_hive_overlay(compare, [self.snapshot], {2: self.mask}, compare=True)
        self.assertFalse(np.array_equal(compare[50, 180], prior_frame[50, 180]))
        # Current-only interior remains clear; compare mode adds outlines, not another fill.
        np.testing.assert_array_equal(compare[100, 240], self.frame[100, 240])

    def test_legend_scales_for_high_resolution_video(self):
        preview = np.zeros((720, 960, 3), np.uint8)
        full = np.zeros((2880, 3840, 3), np.uint8)
        draw_hive_overlay_label(preview, 'temporal', True)
        draw_hive_overlay_label(full, 'temporal', True)
        small_text_rows = np.where(np.any(preview > 150, axis=(1, 2)))[0]
        large_text_rows = np.where(np.any(full > 150, axis=(1, 2)))[0]
        self.assertGreater(len(large_text_rows), 3 * len(small_text_rows))
        self.assertLess(len(large_text_rows), 5 * len(small_text_rows))

    def test_archive_round_trip_empty_frames_and_interrupted_prefix(self):
        cache = self.root / 'cache.zip'
        with self.writer(cache) as writer:
            writer.write(1, self.frame.shape, [])
            writer.write(2, self.frame.shape, [self.snapshot])
        with TemporalHiveOverlayReader(cache) as reader:
            self.assertTrue(reader.metadata['complete'])
            self.assertEqual(reader.metadata['frame_count'], 2)
            self.assertEqual(reader.read(1), [])
            result = reader.read(2)[0]
            self.assertEqual(result.chamber_id, 2)
            self.assertEqual(result.bbox, self.snapshot.bbox)
            np.testing.assert_array_equal(result.labels, self.labels)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with self.writer(cache) as writer:
                writer.write(1, self.frame.shape, [self.snapshot])
                raise RuntimeError('interrupted')
        with TemporalHiveOverlayReader(cache) as reader:
            self.assertFalse(reader.metadata['complete'])
            self.assertEqual(reader.metadata['frame_count'], 1)
            self.assertEqual(len(reader.read(1)), 1)

    def test_writer_rejects_nonconsecutive_frames_and_size_changes(self):
        with self.writer() as writer:
            with self.assertRaises(ValueError):
                writer.write(2, self.frame.shape, [])
            writer.write(1, self.frame.shape, [])
            with self.assertRaises(ValueError):
                writer.write(2, (100, 100), [])

    def test_real_video_replay_has_correct_fps_frames_and_overlay(self):
        _, cache = self.make_source_and_cache()
        output = self.root / 'temporal.mp4'
        self.assertEqual(render_video(cache, output), 3)
        capture = cv2.VideoCapture(str(output))
        try:
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 6)
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, self.frame.shape)
            self.assertGreater(float(np.std(frames[1])), 1)
            self.assertGreater(float(np.mean(np.abs(frames[1][80:140, 110:190].astype(float)
                                                   - frames[0][80:140, 110:190]))), 5)
        finally:
            capture.release()

    def test_replay_protects_existing_outputs_and_cleans_failed_output(self):
        source, cache = self.make_source_and_cache()
        output = self.root / 'temporal.mp4'
        output.write_bytes(b'preserve me')
        with self.assertRaises(FileExistsError):
            render_video(cache, output)
        self.assertEqual(output.read_bytes(), b'preserve me')
        with self.assertRaisesRegex(ValueError, 'filename'):
            render_video(cache, output, self.root / 'wrong.avi', overwrite=True)
        # A missing cached frame must not leave a success-looking MP4 or replace an old one.
        with patch.object(TemporalHiveOverlayReader, 'read', side_effect=KeyError('missing')):
            with self.assertRaises(KeyError):
                render_video(cache, output, source, overwrite=True)
        self.assertEqual(output.read_bytes(), b'preserve me')
        self.assertEqual(list(self.root.glob('.temporal-*')), [])

    def test_visualizer_live_and_deferred_match_and_clear_snapshot_cache(self):
        options = dict(video_path=self.root / 'source.avi', output_folder=self.root,
                       video_id='source', bee_detections=[], chamber_frame_data=[],
                       chambers_by_frame={}, show_chambers=False, show_chamber_info=False,
                       visualization_mode='pretty', hive_overlay_mode='temporal')
        deferred = VisualizationGenerator(**options, hive_masks_by_frame={1: {2: self.mask}},
                                          temporal_hive_snapshots_by_frame={1: [self.snapshot]})
        live = VisualizationGenerator(**options, hive_masks_by_frame={})
        live_frame = live.annotate_live_frame(self.frame.copy(), 1, [], [], {}, hive_masks={2: self.mask},
                                               temporal_hive_snapshots=[self.snapshot])
        np.testing.assert_array_equal(live_frame, deferred._annotate_frame(self.frame.copy(), 1))
        self.assertEqual(live.temporal_hive_snapshots_by_frame, {})
        np.testing.assert_array_equal(live_frame[100, 240], self.frame[100, 240])

    def test_signatures_keep_legacy_default_and_analysis_compatibility(self):
        legacy = BatchVideoInferenceWorker({'output_folder': str(self.root)})
        current = BatchVideoInferenceWorker({'output_folder': str(self.root), 'hive_overlay_mode': 'current'})
        temporal = BatchVideoInferenceWorker({'output_folder': str(self.root), 'hive_overlay_mode': 'temporal'})
        self.assertEqual(legacy._config_signature(), current._config_signature())
        self.assertNotEqual(current._config_signature(), temporal._config_signature())
        self.assertEqual(current._config_signature(ignore_visualization=True),
                         temporal._config_signature(ignore_visualization=True))

    def test_gui_modes_require_prior_and_preserve_user_choice(self):
        settings = QSettings(str(self.root / 'ui.ini'), QSettings.Format.IniFormat)
        with patch('gui.batch_video_inference_dialog.QSettings', return_value=settings):
            dialog = BatchVideoInferenceConfigDialog()
            self.addCleanup(dialog.close)
            self.assertEqual(dialog.hive_overlay_combo.currentData(), 'scored')
            self.assertEqual(dialog.temporal_hive_scoring_combo.currentData(), 'updated')
            self.assertFalse(dialog.hive_overlay_combo.isEnabled())
            dialog.save_visualizations_check.setChecked(True)
            self.assertFalse(dialog.hive_overlay_combo.model().item(1).isEnabled())
            dialog.hive_model_edit.setText('/synthetic/hive.pt')
            dialog.temporal_hive_prior_check.setChecked(True)
            dialog._update_temporal_hive_controls()
            self.assertTrue(dialog.hive_overlay_combo.model().item(1).isEnabled())
            dialog.hive_overlay_combo.setCurrentIndex(dialog.hive_overlay_combo.findData('updated'))
            dialog.stabilize_temporal_hive_check.setChecked(True)
            dialog.temporal_hive_scoring_combo.setCurrentIndex(dialog.temporal_hive_scoring_combo.findData('prior'))
            dialog._save_last_settings()
            restored = BatchVideoInferenceConfigDialog()
            self.addCleanup(restored.close)
            self.assertEqual(restored.hive_overlay_combo.currentData(), 'updated')
            self.assertTrue(restored.stabilize_temporal_hive_check.isChecked())
            self.assertEqual(restored.temporal_hive_scoring_combo.currentData(), 'prior')
            dialog.temporal_hive_prior_check.setChecked(False)
            self.assertEqual(dialog.hive_overlay_combo.currentData(), 'current')
            self.assertFalse(dialog.stabilize_temporal_hive_check.isEnabled())
            self.assertFalse(dialog.temporal_hive_scoring_combo.isEnabled())
            qa = os.environ.get('BBX_VISUAL_QA_DIR')
            if qa:
                Path(qa).mkdir(parents=True, exist_ok=True)
                restored.show()
                self.app.processEvents()
                group = next(g for g in restored.findChildren(QGroupBox)
                             if g.isAncestorOf(restored.hive_overlay_combo))
                group.grab().save(str(Path(qa) / 'output-controls.png'))
                prior_group = next(g for g in restored.findChildren(QGroupBox)
                                   if g.isAncestorOf(restored.stabilize_temporal_hive_check))
                prior_group.grab().save(str(Path(qa) / 'prior-controls.png'))

    def test_cli_defaults_to_updated_scoring_and_matching_overlay(self):
        settings = QSettings(str(self.root / 'cli.ini'), QSettings.Format.IniFormat)
        source, model = self.root / 'source.avi', self.root / 'hive.pt'
        source.touch()
        model.touch()
        argv = ['batch_video_inference_cli.py', '--video', str(source), '--bee-model', str(model),
                '--output-folder', str(self.root / 'output')]
        with patch('batch_video_inference_cli.QSettings', return_value=settings):
            with patch.object(sys, 'argv', argv):
                config = batch_cli.build_config(batch_cli.parse_args())
                self.assertEqual(config['hive_overlay_mode'], 'scored')
                self.assertEqual(config['temporal_hive_scoring'], 'updated')
            with patch.object(sys, 'argv', argv + ['--hive-overlay', 'temporal']):
                with self.assertRaisesRegex(ValueError, 'hive-model'):
                    batch_cli.build_config(batch_cli.parse_args())
            with patch.object(sys, 'argv', argv + ['--hive-overlay', 'compare', '--hive-model', str(model)]):
                config = batch_cli.build_config(batch_cli.parse_args())
                self.assertEqual(config['hive_overlay_mode'], 'compare')
                self.assertTrue(config['use_temporal_hive_prior'])
            with patch.object(sys, 'argv', argv + ['--hive-overlay', 'updated', '--hive-model', str(model),
                                                   '--stabilize-temporal-hive']):
                config = batch_cli.build_config(batch_cli.parse_args())
                self.assertEqual(config['hive_overlay_mode'], 'updated')
                self.assertTrue(config['stabilize_temporal_hive'])
            with patch.object(sys, 'argv', argv + ['--temporal-hive-scoring', 'prior', '--hive-model', str(model)]):
                config = batch_cli.build_config(batch_cli.parse_args())
                self.assertEqual(config['temporal_hive_scoring'], 'prior')
                self.assertEqual(config['hive_overlay_mode'], 'scored')

    def test_visual_qa_contact_sheet(self):
        qa = os.environ.get('BBX_VISUAL_QA_DIR')
        if not qa:
            return
        panels = []
        for mode, snapshots in (('temporal', []), ('temporal', [self.snapshot]), ('compare', [self.snapshot])):
            frame = self.frame.copy()
            cv2.rectangle(frame, (35, 35), (595, 195), (120, 120, 120), 2)
            supported = draw_temporal_hive_overlay(frame, snapshots, {2: self.mask}, mode == 'compare')
            draw_hive_overlay_label(frame, mode, supported)
            panels.append(frame)
        Path(qa).mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(Path(qa) / 'temporal-overlays.png'), np.vstack(panels)))

    def test_updated_cache_replay_uses_recorded_timing_not_a_relabelled_prior(self):
        source, cache = self.make_source_and_cache()
        updated = self.root / 'updated.zip'
        with TemporalHiveOverlayWriter(updated, source, 'context', fps=6,
                                       timing='after_current_frame_update') as writer:
            for number in range(1, 4):
                writer.write(number, self.frame.shape, [self.snapshot])
        with TemporalHiveOverlayReader(cache) as reader:
            self.assertEqual(reader.overlay_mode, 'temporal')
        with TemporalHiveOverlayReader(updated) as reader:
            self.assertEqual(reader.overlay_mode, 'updated')
        with patch('render_temporal_hive_video.draw_hive_overlay_label') as label:
            render_video(updated, self.root / 'updated.mp4')
            self.assertEqual([call.args[1] for call in label.call_args_list], ['updated'] * 3)

    def test_stabilizer_smooths_small_jitter_preserves_raw_and_resets_large_moves(self):
        stabilizer = TemporalChamberStabilizer()
        first = {0: {'bbox': [100, 100, 1100, 2100]}}
        second = {0: {'bbox': [110, 100, 1110, 2100]}}
        stabilizer.prepare(first, (2300, 1400))
        prepared = stabilizer.prepare(second, (2300, 1400))
        np.testing.assert_allclose(prepared[0]['temporal_bbox'], [102, 100, 1102, 2100])
        self.assertEqual(prepared[0]['bbox'], second[0]['bbox'])
        self.assertNotIn('temporal_bbox', second[0])
        moved = {0: {'bbox': [200, 100, 1200, 2100]}}
        np.testing.assert_allclose(stabilizer.prepare(moved, (2300, 1400))[0]['temporal_bbox'], moved[0]['bbox'])
        # A new chamber set, resolution or processor/video must not reuse old geometry.
        changed_ids = {2: first[0]}
        self.assertEqual(stabilizer.prepare(changed_ids, (2300, 1400))[2]['temporal_bbox'], tuple(first[0]['bbox']))
        self.assertEqual(stabilizer.prepare(first, (2400, 1500))[0]['temporal_bbox'], tuple(first[0]['bbox']))
        stabilizer.reset()
        self.assertEqual(stabilizer.boxes, {})

    def test_stabilized_coordinates_used_for_update_scoring_and_visualization(self):
        prior = TemporalHivePrior(resolution=(16, 16), min_prior_weight=1, cleanup_kernel_size=0,
                                  min_component_pixels=0, stabilize_chambers=True)
        chamber = {'bbox': [8, 0, 24, 16], 'temporal_bbox': (0, 0, 16, 16), 'mask': None}
        hive = np.zeros((32, 32), np.uint8)
        hive[4:8, 4:8] = 1
        prior.update('c', 0, chamber, hive, [], hive.shape, None)
        snapshot = prior.visualization_snapshot('c', 0, chamber, hive.shape, None)
        self.assertEqual(snapshot.bbox, (0, 0, 16, 16))
        np.testing.assert_array_equal(project_snapshot(snapshot, hive.shape) == 2, hive > 0)
        overlap = prior.query_bee_overlap('c', 0, chamber, hive, None, hive.shape, None)
        self.assertEqual(overlap.overlap_fraction, 1)
        self.assertEqual(overlap.known_fraction, 1)
        unavailable = dict(chamber, temporal_unavailable=True)
        self.assertIsNone(prior.query_bee_overlap('c', 0, unavailable, hive, None, hive.shape, None).on_hive)
        self.assertFalse(prior.visualization_snapshot('c', 0, unavailable, hive.shape, None).labels.any())
        count = prior._states['c', 0].observation_count
        prior.update('c', 0, unavailable, hive, [], hive.shape, None)
        self.assertEqual(prior._states['c', 0].observation_count, count)
        checkpoint = self.root / 'prior.npz'
        prior.save_checkpoint(checkpoint)
        loaded, _ = TemporalHivePrior.load_checkpoint(checkpoint)
        self.assertTrue(loaded.stabilize_chambers)
        self.assertEqual(loaded.scoring_mode, 'updated')
        self.assertEqual(loaded.summaries(), prior.summaries())

    def test_stabilization_changes_analysis_signature_but_disabled_keeps_legacy(self):
        base = {'output_folder': str(self.root)}
        legacy = BatchVideoInferenceWorker(base)
        disabled = BatchVideoInferenceWorker(dict(base, stabilize_temporal_hive=False))
        enabled = BatchVideoInferenceWorker(dict(base, stabilize_temporal_hive=True))
        self.assertEqual(legacy._config_signature(), disabled._config_signature())
        self.assertNotEqual(disabled._config_signature(ignore_visualization=True),
                            enabled._config_signature(ignore_visualization=True))

    def test_checkpoint_rejects_different_stabilization_even_with_matching_signature(self):
        worker = BatchVideoInferenceWorker({'output_folder': str(self.root),
                                           'temporal_hive_checkpointing': True})
        prior = TemporalHivePrior(stabilize_chambers=True)
        worker.temporal_hive_prior = prior
        source = self.root / 'source.mp4'
        source_key = worker._video_manifest_path(source)
        metadata = {'analysis_config_signature': worker.analysis_config_signature,
                    'selected_file_order_signature': None, 'video_index': 1, 'video_path': source_key}
        TemporalHivePrior(stabilize_chambers=False).save_checkpoint(
            worker._temporal_prior_checkpoint_path(), metadata)
        worker._load_temporal_hive_prior_checkpoint([source], {source_key})
        self.assertIs(worker.temporal_hive_prior, prior)
        self.assertEqual(worker.temporal_hive_prior_checkpoint_index, 0)

    def test_scoring_mode_controls_linked_overlay_and_analysis_signature(self):
        self.assertEqual(resolve_hive_overlay_mode('scored', None), 'current')
        for mode, expected in (('updated', 'updated'), ('prior', 'temporal')):
            prior = TemporalHivePrior(scoring_mode=mode)
            self.assertEqual(resolve_hive_overlay_mode('scored', prior), expected)
            self.assertEqual(resolve_hive_overlay_mode('compare', prior), 'compare')
        base = {'output_folder': str(self.root), 'hive_model_path': 'hive.pt', 'use_temporal_hive_prior': True}
        default = BatchVideoInferenceWorker(base)
        updated = BatchVideoInferenceWorker(dict(base, temporal_hive_scoring='updated'))
        legacy = BatchVideoInferenceWorker(dict(base, temporal_hive_scoring='prior'))
        self.assertEqual(default.analysis_config_signature, updated.analysis_config_signature)
        self.assertNotEqual(updated.analysis_config_signature, legacy.analysis_config_signature)
        self.assertNotIn(legacy.config_signature, updated._legacy_visualization_signature_candidates())
        disabled = dict(base, use_temporal_hive_prior=False)
        self.assertEqual(
            BatchVideoInferenceWorker(disabled).analysis_config_signature,
            BatchVideoInferenceWorker(dict(disabled, temporal_hive_scoring='updated')).analysis_config_signature,
        )

    def test_checkpoint_rejects_different_scoring_mode(self):
        worker = BatchVideoInferenceWorker({'output_folder': str(self.root),
                                           'temporal_hive_checkpointing': True})
        prior = TemporalHivePrior(scoring_mode='updated')
        worker.temporal_hive_prior = prior
        source = self.root / 'source.mp4'
        source_key = worker._video_manifest_path(source)
        metadata = {'analysis_config_signature': worker.analysis_config_signature,
                    'selected_file_order_signature': None, 'video_index': 1, 'video_path': source_key}
        TemporalHivePrior(scoring_mode='prior').save_checkpoint(worker._temporal_prior_checkpoint_path(), metadata)
        worker._load_temporal_hive_prior_checkpoint([source], {source_key})
        self.assertIs(worker.temporal_hive_prior, prior)
        self.assertEqual(worker.temporal_hive_prior_checkpoint_index, 0)

    def test_old_checkpoint_without_scoring_metadata_remains_past_only(self):
        checkpoint = self.root / 'legacy.npz'
        TemporalHivePrior(scoring_mode='prior').save_checkpoint(checkpoint)
        with np.load(checkpoint, allow_pickle=False) as archive:
            payload = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads(str(payload['metadata'].item()))
        del metadata['prior_settings']['scoring_mode']
        payload['metadata'] = np.array(json.dumps(metadata), dtype=np.str_)
        np.savez_compressed(checkpoint, **payload)
        loaded, _ = TemporalHivePrior.load_checkpoint(checkpoint)
        self.assertEqual(loaded.scoring_mode, 'prior')
        with self.assertRaises(ValueError):
            TemporalHivePrior(scoring_mode='invalid')


if __name__ == '__main__':
    unittest.main()
