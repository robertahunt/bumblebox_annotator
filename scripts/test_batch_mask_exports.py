#!/usr/bin/env python3
"""Synthetic batch mask-export regressions; no model downloads or GPU required."""

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.batch_video_processor import BatchVideoProcessor
from core.instance_tracker import Detection
from core.temporal_hive_prior import TemporalHivePrior
from core.temporal_hive_visualization import TemporalHiveOverlayReader, TemporalHiveOverlayWriter
from core.video_inference_exporter import VideoInferenceExporter
from gui.batch_video_inference_worker import BatchVideoInferenceWorker
import batch_hive_refresh_cli as refresh_cli


class BatchMaskExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.chamber_mask = np.full((8, 8), 255, dtype=np.uint8)
        self.hive_masks = []
        for other_pixel in ((6, 6), (1, 2), (1, 2)):
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[1, 1] = mask[other_pixel] = 255
            self.hive_masks.append(mask)

        for target, options in (
            ('core.batch_video_processor.TORCH_AVAILABLE', {'new': False}),
            ('gui.batch_video_inference_worker.torch.cuda.is_available', {'return_value': False}),
            ('core.batch_video_processor.cv2.VideoCapture', {'side_effect': self.make_capture}),
            ('core.batch_video_processor.cv2.VideoWriter', {'return_value': Mock(isOpened=Mock(return_value=True))}),
        ):
            patcher = patch(target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_capture(self, *args):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 10.0 if prop == cv2.CAP_PROP_FPS else 3
        capture.read.side_effect = [(True, self.frame.copy()) for _ in range(3)] + [(False, None)]
        return capture

    def make_processor(self, **overrides):
        options = dict(
            video_path=self.root / 'sample.mp4', video_id='sample',
            bee_model=Mock(return_value=[object()]), hive_model=Mock(return_value=[object()]),
            chamber_model=None, tracker=None, confidence_threshold=0.5,
            nms_iou_threshold=0.5, enable_aruco=False, compute_spatial_metrics=False,
            log_callback=lambda message: None,
        )
        options.update(overrides)
        processor = BatchVideoProcessor(**options)
        processor._detect_chambers = Mock(side_effect=[
            {0: {'mask': self.chamber_mask, 'centroid': (coordinate, coordinate), 'bbox': [0, 0, 8, 8]}}
            for coordinate in (2, 4, 6)
        ])
        processor._yolo_to_detections = Mock(return_value=[])
        processor._apply_tracking = Mock(return_value=[])
        processor._extract_hive_masks = Mock(side_effect=[{0: mask} for mask in self.hive_masks])
        visualizer = Mock()
        visualizer.annotate_live_frame.side_effect = lambda **kwargs: kwargs['frame'].copy()
        processor._streaming_visualizer = visualizer
        return processor

    def make_worker(self, folder='output', **overrides):
        config = dict(
            output_folder=str(self.root / folder), confidence_threshold=0.5,
            nms_iou_threshold=0.5, enable_aruco=False, compute_spatial_metrics=False,
            hive_model_path='synthetic-hive.pt', save_visualizations=False,
        )
        config.update(overrides)
        return BatchVideoInferenceWorker(config)

    def run_worker_video(self, worker, video_id='sample', stop_after=None):
        processors = []

        def factory(**kwargs):
            processor = self.make_processor(**kwargs)
            if stop_after is not None:
                processor.stop_callback = lambda: processor.frame_count >= stop_after
            processors.append(processor)
            return processor

        with patch('gui.batch_video_inference_worker.BatchVideoProcessor', side_effect=factory):
            result = worker._process_video(
                self.root / f'{video_id}.mp4',
                Mock(return_value=[object()]), Mock(return_value=[object()]),
                None, None, None, 1,
            )
        return result, processors[0]

    def read_rows(self, path):
        with path.open(newline='') as handle:
            return list(csv.DictReader(handle))

    def test_refresh_caches_past_only_maps_without_bee_model_or_tracking(self):
        prior = TemporalHivePrior(resolution=(8, 8), min_prior_weight=1,
                                  cleanup_kernel_size=0, min_component_pixels=0)
        args = SimpleNamespace(confidence=0.5, nms_iou=0.5, verbose=False,
                               pixel_size_mm=None, keep_pollen_in_hive=True)
        source = self.root / 'sample.mp4'
        processor = self.make_processor(video_path=source, temporal_hive_prior=prior,
                                        temporal_hive_context_id=refresh_cli.context_id_for_path(source))
        processor._detect_pollen_balls = Mock(return_value=[])
        cache_path = self.root / 'refresh.zip'
        with patch('batch_hive_refresh_cli.BatchVideoProcessor', return_value=processor), \
                patch('batch_hive_refresh_cli.run_hive_model', return_value=[object()]), \
                TemporalHiveOverlayWriter(cache_path, source, processor.temporal_hive_context_id,
                                          provenance='reconstructed_from_saved_bee_detections') as writer:
            counts = refresh_cli.refresh_video(source, {}, ['video_id'], self.root,
                                               None, None, None, prior, args, writer)
        self.assertEqual(counts, (3, 0))
        processor.bee_model.assert_not_called()
        processor._apply_tracking.assert_not_called()
        with TemporalHiveOverlayReader(cache_path) as reader:
            self.assertEqual(reader.metadata['frame_count'], 3)
            self.assertEqual(reader.metadata['fps'], 10)
            self.assertFalse(reader.read(1)[0].labels.any())
            self.assertEqual(reader.read(2)[0].labels[6, 6], 2)
            self.assertEqual(reader.read(2)[0].labels[1, 2], 1)

    def test_refresh_updated_cache_contains_current_frame_evidence(self):
        prior = TemporalHivePrior(resolution=(8, 8), min_prior_weight=1,
                                  cleanup_kernel_size=0, min_component_pixels=0)
        args = SimpleNamespace(confidence=0.5, nms_iou=0.5, verbose=False,
                               pixel_size_mm=None, keep_pollen_in_hive=True)
        source = self.root / 'sample.mp4'
        processor = self.make_processor(video_path=source, temporal_hive_prior=prior,
                                        temporal_hive_context_id=refresh_cli.context_id_for_path(source))
        processor._detect_pollen_balls = Mock(return_value=[])
        cache_path = self.root / 'refresh_updated.zip'
        with patch('batch_hive_refresh_cli.BatchVideoProcessor', return_value=processor), \
                patch('batch_hive_refresh_cli.run_hive_model', return_value=[object()]), \
                TemporalHiveOverlayWriter(cache_path, source, processor.temporal_hive_context_id,
                                          timing='after_current_frame_update') as writer:
            self.assertEqual(refresh_cli.refresh_video(source, {}, ['video_id'], self.root,
                                                        None, None, None, prior, args, writer), (3, 0))
        with TemporalHiveOverlayReader(cache_path) as reader:
            self.assertEqual(reader.overlay_mode, 'updated')
            self.assertEqual(reader.read(1)[0].labels[6, 6], 2)
            self.assertEqual(reader.read(1)[0].labels[1, 2], 1)

    def assert_summary_rows(self, folder, video_ids=('sample',)):
        hive = self.read_rows(folder / 'hive_detections.csv')
        chamber = self.read_rows(folder / 'chamber_detections.csv')
        self.assertEqual([row['video_id'] for row in hive], list(video_ids))
        self.assertEqual([row['video_id'] for row in chamber], list(video_ids))
        for row in hive:
            self.assertEqual((row['hive_pixels'], row['centroid_x'], row['centroid_y']),
                             ('2', '1.50', '1.00'))
        for row in chamber:
            self.assertEqual((row['chamber_pixels'], row['centroid_x'], row['centroid_y']),
                             ('64', '4.00', '4.00'))

    def test_all_visualization_modes_use_all_analyzed_frames(self):
        modes = [
            {},
            {'store_masks': True},
            {'store_masks': True, 'store_masks_until_frame': 1},
            {'streaming_visualization_format': 'frames'},
            {'streaming_visualization_format': 'frames', 'streaming_visualization_max_frames': 1},
            {'streaming_visualization_format': 'video'},
            {'streaming_visualization_format': 'video', 'streaming_visualization_max_frames': 1},
        ]
        for index, options in enumerate(modes):
            with self.subTest(options=options):
                folder = self.root / str(index)
                if 'streaming_visualization_format' in options:
                    options = {**options, 'streaming_visualization_path': folder / 'annotated'}
                processor = self.make_processor(**options)
                self.assertTrue(processor.process())
                self.assertEqual(processor.accumulated_hive_masks[('sample', 0)]['frame_count'], 3)
                self.assertEqual(processor.accumulated_chamber_masks[('sample', 0)]['frame_count'], 3)
                if not options.get('store_masks'):
                    self.assertEqual(processor.get_hive_masks_by_frame(), {})
                    self.assertEqual(processor.get_chambers_by_frame(), {})
                exporter = VideoInferenceExporter(folder)
                exporter.export_hive_detections(processor.get_accumulated_hive_masks())
                exporter.export_chamber_detections(processor.get_accumulated_chamber_masks())
                self.assert_summary_rows(folder)

    def test_worker_collects_exports_with_visualization_off_or_streaming(self):
        for index, options in enumerate((
            {},
            {'save_visualizations': True, 'visualization_format': 'frames', 'visualization_max_frames': 1},
            {'save_visualizations': True, 'visualization_format': 'video'},
        )):
            with self.subTest(options=options):
                worker = self.make_worker(folder=str(index), **options)
                result, processor = self.run_worker_video(worker)
                self.assertTrue(result['completed'])
                self.assertIs(worker.accumulated_hive_masks[('sample', 0)],
                              processor.accumulated_hive_masks[('sample', 0)])
                self.assertGreater(worker.accumulated_data_size_mb, 0)
                self.assertTrue(worker._export_csvs(append=True, clear_after=True))
                self.assert_summary_rows(Path(worker.config['output_folder']))
                self.assertFalse(worker._has_pending_export_data())
                self.assertEqual(worker.accumulated_data_size_mb, 0)

    def test_stopped_video_keeps_partial_summaries(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                worker = self.make_worker(folder=str(streaming), save_visualizations=streaming,
                                          visualization_format='frames')
                result, processor = self.run_worker_video(worker, stop_after=2)
                self.assertIsNone(result)
                self.assertTrue(processor.was_stopped)
                self.assertEqual(worker.accumulated_hive_masks[('sample', 0)]['frame_count'], 2)
                self.assertTrue(worker._export_csvs(append=True, clear_after=True))
                folder = Path(worker.config['output_folder'])
                # The two other pixels tie at 50%, so only the common pixel survives.
                self.assertEqual(self.read_rows(folder / 'hive_detections.csv')[0]['hive_pixels'], '1')
                self.assertEqual(self.read_rows(folder / 'chamber_detections.csv')[0]['centroid_x'], '3.00')

    def test_successive_videos_append_without_reusing_previous_counts(self):
        worker = self.make_worker()
        for video_id in ('first', 'second'):
            result, _ = self.run_worker_video(worker, video_id=video_id)
            self.assertTrue(result['completed'])
            self.assertEqual(set(worker.accumulated_hive_masks), {(video_id, 0)})
            self.assertTrue(worker._export_csvs(append=True, clear_after=True))
        self.assert_summary_rows(Path(worker.config['output_folder']), ('first', 'second'))

    def test_prior_only_replay_does_not_accumulate_export_masks(self):
        prior = TemporalHivePrior(resolution=(8, 8), cleanup_kernel_size=0, min_component_pixels=0)
        processor = self.make_processor(prior_only=True, temporal_hive_prior=prior)
        self.assertTrue(processor.process())
        self.assertEqual(processor.get_accumulated_hive_masks(), {})
        self.assertEqual(processor.get_accumulated_chamber_masks(), {})
        self.assertEqual(prior.summaries()[0]['observation_count'], 3)

    def test_pollen_exclusion_precedes_export_accumulation(self):
        processor = self.make_processor()
        pollen_mask = np.zeros((8, 8), dtype=np.uint8)
        pollen_mask[1, 2] = 255
        processor._assign_pollen_to_chambers = Mock(return_value={
            0: [{'pollen_id': 1, 'mask': pollen_mask, 'pixels': 1}],
        })
        self.assertTrue(processor.process())
        summary = processor.get_accumulated_hive_masks()[('sample', 0)]
        self.assertEqual(summary['accumulated_mask'][1, 2], 0)
        self.assertEqual(summary['accumulated_mask'][1, 1], 3)
        self.assertEqual(self.hive_masks[1][1, 2], 255)

    def test_missing_masks_are_skipped_but_empty_masks_count(self):
        processor = self.make_processor()
        processor._accumulate_export_masks({0: None}, {0: {'mask': None}})
        self.assertEqual(processor.get_accumulated_hive_masks(), {})
        processor._accumulate_export_masks({0: np.zeros((8, 8), dtype=np.uint8)}, {})
        data = processor.get_accumulated_hive_masks()[('sample', 0)]
        self.assertEqual(data['frame_count'], 1)
        self.assertEqual(int(data['accumulated_mask'].sum()), 0)

    def test_accumulators_do_not_overflow_after_65535_frames(self):
        processor = self.make_processor()
        chambers = {0: {'mask': self.chamber_mask, 'centroid': (3, 3)}}
        processor._accumulate_export_masks({0: self.chamber_mask}, chambers)
        for summaries in (processor.accumulated_hive_masks, processor.accumulated_chamber_masks):
            data = summaries[('sample', 0)]
            self.assertEqual(data['accumulated_mask'].dtype, np.uint32)
            data['accumulated_mask'].fill(65535)
            data['frame_count'] = 65535
        processor._accumulate_export_masks({0: self.chamber_mask}, chambers)
        for summaries in (processor.accumulated_hive_masks, processor.accumulated_chamber_masks):
            self.assertEqual(summaries[('sample', 0)]['frame_count'], 65536)
            self.assertTrue(np.all(summaries[('sample', 0)]['accumulated_mask'] == 65536))

    def test_chambers_have_independent_counts_and_do_not_keep_source_masks(self):
        processor = self.make_processor()
        source = self.chamber_mask.copy()
        processor._accumulate_export_masks({0: source, 1: np.zeros_like(source)}, {})
        source.fill(0)
        self.assertTrue(np.all(processor.accumulated_hive_masks[('sample', 0)]['accumulated_mask'] == 1))
        self.assertFalse(np.any(processor.accumulated_hive_masks[('sample', 1)]['accumulated_mask']))

    def test_visualization_failure_does_not_discard_export_counts(self):
        processor = self.make_processor(streaming_visualization_path=self.root / 'annotated',
                                        streaming_visualization_format='frames')
        processor._streaming_visualizer.annotate_live_frame.side_effect = RuntimeError('synthetic render failure')
        self.assertTrue(processor.process())
        self.assertTrue(processor._streaming_video_failed)
        self.assertEqual(processor.accumulated_hive_masks[('sample', 0)]['frame_count'], 3)
        self.assertEqual(processor.get_hive_masks_by_frame(), {})

    def test_temporal_overlay_captures_history_before_current_frame_update(self):
        prior = TemporalHivePrior(resolution=(8, 8), min_prior_weight=1,
                                  cleanup_kernel_size=0, min_component_pixels=0)
        cache = self.root / 'history.zip'
        processor = self.make_processor(
            temporal_hive_prior=prior, hive_overlay_mode='temporal',
            streaming_visualization_path=self.root / 'frames', streaming_visualization_format='frames',
            temporal_overlay_path=cache,
        )
        self.assertTrue(processor.process())
        with TemporalHiveOverlayReader(cache) as reader:
            self.assertEqual(reader.metadata['frame_count'], 3)
            self.assertTrue(reader.metadata['complete'])
            self.assertFalse(np.any(reader.read(1)[0].labels))
            previous = reader.read(2)[0].labels
            self.assertEqual(previous[6, 6], 2)  # First frame's hive, absent from current frame.
            self.assertEqual(previous[1, 2], 1)  # Current frame's new hive is not in history yet.
        calls = processor._streaming_visualizer.annotate_live_frame.call_args_list
        np.testing.assert_array_equal(calls[1].kwargs['temporal_hive_snapshots'][0].labels, previous)
        self.assertFalse(processor.temporal_hive_snapshots_by_frame)

    def test_worker_caches_only_preview_frames_and_defaults_to_scoring_map(self):
        worker = self.make_worker(save_visualizations=True, visualization_format='frames',
                                  visualization_max_frames=1)
        worker.temporal_hive_prior = TemporalHivePrior(resolution=(8, 8))
        result, processor = self.run_worker_video(worker)
        self.assertTrue(result['completed'])
        self.assertEqual(processor.hive_overlay_mode, 'updated')
        cache = Path(worker.config['output_folder']) / 'temporal_hive_overlays' / 'sample_updated.zip'
        with TemporalHiveOverlayReader(cache) as reader:
            self.assertEqual(reader.metadata['frame_count'], 1)
        self.assertEqual(processor.accumulated_hive_masks[('sample', 0)]['frame_count'], 3)

    def test_updated_scoring_and_overlay_both_include_current_evidence(self):
        prior = TemporalHivePrior(resolution=(8, 8), min_prior_weight=1,
                                  cleanup_kernel_size=0, min_component_pixels=0)
        cache = self.root / 'updated.zip'
        processor = self.make_processor(temporal_hive_prior=prior, hive_overlay_mode='updated',
                                        store_masks=True, temporal_overlay_path=cache)
        scoring_counts = []
        original = processor._process_bee_detections
        def score(*args, **kwargs):
            scoring_counts.append(sum(state.observation_count for state in prior._states.values()))
            return original(*args, **kwargs)
        processor._process_bee_detections = score
        self.assertTrue(processor.process())
        self.assertEqual(scoring_counts, [1, 2, 3])
        with TemporalHiveOverlayReader(cache) as reader:
            self.assertEqual(reader.overlay_mode, 'updated')
            self.assertEqual(reader.read(1)[0].labels[6, 6], 2)
            self.assertEqual(reader.read(2)[0].labels[1, 2], 2)

    def test_stabilized_processor_marks_missing_chamber_geometry_unavailable(self):
        processor = self.make_processor(chamber_model=Mock(),
                                        temporal_hive_prior=TemporalHivePrior(stabilize_chambers=True))
        fallback = {0: {'bbox': [0, 0, 8, 8], 'mask': None}}
        prepared = processor._prepare_temporal_chambers(fallback, (8, 8))
        self.assertTrue(prepared[0]['temporal_unavailable'])
        self.assertNotIn('temporal_unavailable', fallback[0])

    def test_scoring_order_is_independent_of_overlay_and_updates_once(self):
        for mode in ('updated', 'prior'):
            for overlay in ('scored', 'current', 'temporal', 'compare', 'updated'):
                with self.subTest(mode=mode, overlay=overlay):
                    prior = TemporalHivePrior(resolution=(8, 8), min_prior_weight=1,
                                              cleanup_kernel_size=0, min_component_pixels=0, scoring_mode=mode)
                    processor = self.make_processor(temporal_hive_prior=prior, hive_overlay_mode=overlay,
                                                    store_masks=False)
                    scores, counts = [], []
                    # Probe an observable cell during scoring to distinguish before/after update.
                    probe = np.zeros((8, 8), np.uint8)
                    probe[1, 2] = 1
                    def score(bees, number, chambers, hives, pollen, shape, timestamp, timings):
                        counts.append(sum(s.observation_count for s in prior._states.values()))
                        result = prior.query_bee_overlap('default', 0, chambers[0], probe, None, shape, timestamp)
                        scores.append(result.on_hive)
                    processor._process_bee_detections = score
                    self.assertTrue(processor.process())
                    self.assertEqual(counts, [1, 2, 3] if mode == 'updated' else [0, 1, 2])
                    self.assertEqual(scores, [False, True, True] if mode == 'updated' else [None, False, True])
                    self.assertEqual(prior._states['default', 0].observation_count, 3)

    def test_exported_contacts_match_updated_overlay_and_preserve_occluded_history(self):
        prior = TemporalHivePrior(resolution=(8, 8), min_prior_weight=1,
                                  cleanup_kernel_size=0, min_component_pixels=0)
        chamber = {'bbox': [0, 0, 8, 8], 'mask': self.chamber_mask}
        prior.update('default', 0, chamber, self.hive_masks[0], [], (8, 8), None)
        mask = np.zeros((8, 8), np.uint8)
        mask[6, 6] = 255
        bee = Detection(np.array([6, 6, 7, 7]), mask=mask, confidence=0.9, instance_id=1)
        processor = self.make_processor(temporal_hive_prior=prior, store_masks=True,
                                        temporal_overlay_path=self.root / 'scored.zip')
        processor._apply_tracking.return_value = [bee]
        self.assertTrue(processor.process())
        rows = processor.get_bee_detections()
        self.assertEqual(len(rows), 3)
        self.assertEqual(prior._states['default', 0].observation_count, 4)
        # Current YOLO hive disappears at this cell, but a covering bee cannot erase its history.
        self.assertEqual(prior._states['default', 0].weight_sum[6, 6], 1)
        with TemporalHiveOverlayReader(self.root / 'scored.zip') as reader:
            self.assertEqual(reader.overlay_mode, 'updated')
            self.assertEqual(reader.metadata['prior_settings']['scoring_mode'], 'updated')
            for row in rows:
                labels = reader.read(row.frame_number)[0].labels
                known = (mask > 0) & (labels > 0)
                overlap = np.count_nonzero(known & (labels == 2)) / np.count_nonzero(known)
                self.assertEqual(row.temporal_hive_overlap_fraction, overlap)
                self.assertTrue(row.on_temporal_hive)
        exporter = VideoInferenceExporter(self.root)
        exporter.export_bee_detections(rows)
        exported = self.read_rows(self.root / 'bee_detections.csv')
        self.assertEqual([r['temporal_hive_overlap_fraction'] for r in exported], ['1.0000'] * 3)

    def test_refresh_queries_selected_state_and_does_not_double_update(self):
        for mode in ('updated', 'prior'):
            with self.subTest(mode=mode):
                prior = TemporalHivePrior(resolution=(8, 8), min_prior_weight=1,
                                          cleanup_kernel_size=0, min_component_pixels=0, scoring_mode=mode)
                args = SimpleNamespace(confidence=0.5, nms_iou=0.5, verbose=False,
                                       pixel_size_mm=None, keep_pollen_in_hive=True)
                source = self.root / 'sample.mp4'
                processor = self.make_processor(video_path=source, temporal_hive_prior=prior,
                                                temporal_hive_context_id=refresh_cli.context_id_for_path(source))
                processor._detect_pollen_balls = Mock(return_value=[])
                counts = []
                original_query = processor._query_temporal_hive_overlap
                def query(*args):
                    counts.append(sum(s.observation_count for s in prior._states.values()))
                    return original_query(*args)
                processor._query_temporal_hive_overlap = query
                row = {'bee_id': '1', 'chamber_id': '0', 'bbox_x': '6', 'bbox_y': '6',
                       'bbox_width': '1', 'bbox_height': '1'}
                with patch('batch_hive_refresh_cli.BatchVideoProcessor', return_value=processor), \
                        patch('batch_hive_refresh_cli.run_hive_model', return_value=[object()]):
                    result = refresh_cli.refresh_video(source, {i: [row] for i in (1, 2, 3)},
                                                       ['on_temporal_hive'], self.root,
                                                       None, None, None, prior, args)
                self.assertEqual(result, (3, 3))
                self.assertEqual(counts, [1, 2, 3] if mode == 'updated' else [0, 1, 2])
                state = next(iter(prior._states.values()))
                self.assertEqual(state.observation_count, 3)
                self.assertEqual(state.weight_sum[6, 6], 0)

    def test_refresh_defaults_and_resume_scoring_compatibility(self):
        argv = ['refresh', '--source-output-folder', str(self.root), '--file-list', 'videos.txt',
                '--output-folder', str(self.root), '--hive-model', 'hive.pt',
                '--chamber-model', 'chamber.pt', '--no-pollen-model']
        with patch.object(sys, 'argv', argv):
            parsed = refresh_cli.parse_args()
        self.assertEqual(parsed.temporal_hive_scoring, 'updated')
        self.assertEqual(parsed.temporal_overlay_timing, 'scored')
        args = parsed
        args.resume = True
        video = self.root / 'sample.mp4'
        prior = TemporalHivePrior(resolution=(8, 8), scoring_mode='prior')
        refresh_cli.save_temporal_prior_checkpoint(self.root, prior, video, 1, 1, args, (8, 8))
        with self.assertRaisesRegex(RuntimeError, 'scoring mode'):
            refresh_cli.initialize_temporal_prior(args, [video], (8, 8), 1)
        args.temporal_hive_scoring = 'prior'
        restored = refresh_cli.initialize_temporal_prior(args, [video], (8, 8), 1)
        self.assertEqual(restored.scoring_mode, 'prior')


if __name__ == '__main__':
    unittest.main()
