"""Mask inventories, per-category completion, and selective-work publications."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.annotation import AnnotationManager
from core.annotation_policy import default_policy
from core.annotation_work import (build_index, local_index, mark_finished, matches, published_index,
                                  summarize, unit_for, Sources, task_categories, task_label)
from core.assigned_work import (catalog, latest_snapshot, publish_work, read_assignments,
                               save_assignments, workspace)
from core.contributors import new_session
from core.coco_masks import encode_mask
from core.import_review import add_review, complete_review, require_reviewed
from core.project_sync import _read_json, _write_json, SyncConflictError
from scripts.test_project_sync import annotation
from scripts import test_assigned_work as assigned


class AnnotationWorkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / 'project'
        (self.project / 'annotations').mkdir(parents=True)
        self.actor = new_session('Annotator')
        self.manager = AnnotationManager()
        self.info = {'hive_annotation_scope': 'frame'}
        self.info['annotation_policy'] = default_policy(self.info)
        _write_json(self.project / 'annotations/project.json', self.info)
        for frame in (0, 2, 5):
            path = self.project / f'frames/video/frame_{frame:06d}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.zeros((32, 40, 3), np.uint8))
            self.manager.save_frame_annotations(self.project, 'video', frame, [annotation(1)])
        _write_json(self.project / 'frames/video/video_metadata.json', {'split': 'train'})

    def entry(self):
        return local_index(self.project)['videos']['video']

    def shared(self):
        self.info = {'annotation_policy': default_policy({})}
        _write_json(self.project / 'annotations/project.json', self.info)
        self.manager.save_video_annotations(self.project, 'video', [annotation(7, 'hive')])

    def test_presence_is_category_specific_not_bbox_or_empty_instance(self):
        empty = annotation(4, 'hive')
        empty['mask'][:] = 0
        self.manager.save_frame_annotations(self.project, 'video', 0, [annotation(1), empty])
        self.manager.save_frame_annotations(self.project, 'video', 2, [annotation(1), annotation(3, 'hive')])
        self.manager.save_bbox_annotations(self.project, 'video', 5,
            [dict(mask_id=9, category='hive', bbox=[0, 0, 10, 10], bbox_only=True)])
        hive = summarize(self.entry(), 'hive')
        self.assertEqual(hive['mask_text'], 'Masks on 1/3 frames')
        self.assertEqual(hive['state'], 'not_checked')
        self.assertTrue(matches(hive, 'not_checked', 'some'))
        self.assertFalse(matches(hive, 'not_started', 'any'))
        self.assertFalse(matches(hive, 'finished', 'any'))
        self.assertEqual(summarize(self.entry(), 'bee')['presence'], 'all')
        self.assertEqual(summarize(self.entry(), 'hive', [5])['presence'], 'none')

    def test_show_filters_partition_statuses_without_hiding_unverified_work(self):
        for state, expected in (
            ('not_started', 'not_started'), ('not_checked', 'not_checked'),
            ('changed', 'not_checked'), ('needs_review', 'not_checked'),
            ('unknown', 'not_checked'), ('not_scanned', 'not_checked'), ('finished', 'finished'),
        ):
            summary = {'state': state, 'presence': 'unknown' if state == 'unknown' else 'all'}
            with self.subTest(state=state):
                self.assertTrue(matches(summary, 'all'))
                for show in ('not_started', 'not_checked', 'finished'):
                    self.assertEqual(matches(summary, show), show == expected)
                self.assertFalse(matches(summary, expected, 'none'))
        empty = summarize(self.entry(), 'hive')
        self.assertTrue(matches(empty, 'not_started', 'none'))
        mark_finished(self.project, 'video', 0, 'hive', self.actor)
        self.assertTrue(matches(summarize(self.entry(), 'hive'), 'not_checked'))
        self.assertTrue(matches(summarize(self.entry(), 'hive', [0]), 'finished', 'none'))

    def test_unscanned_summary_is_not_an_error_or_an_absent_mask(self):
        summary = summarize({'scan_state': 'not_scanned', 'error': 'Missing cached summary'}, 'hive')
        self.assertEqual(summary['state'], 'not_scanned')
        self.assertEqual(summary['mask_text'], 'Not scanned')
        self.assertIn('Scan Mask Status', summary['detail'])
        self.assertTrue(matches(summary, 'not_checked', 'unknown'))
        self.assertFalse(matches(summary, 'not_started', 'none'))

    def test_frame_completion_absence_and_reopen(self):
        mark_finished(self.project, 'video', 0, 'hive', self.actor)
        value = summarize(self.entry(), 'hive', [0])
        self.assertEqual(value['work_text'], 'Checked: none present')
        self.assertNotEqual(summarize(self.entry(), 'hive')['state'], 'finished')
        path = self.project / 'annotations/work/video/frame_000000.json'
        self.assertEqual(_read_json(path)['categories']['hive']['by']['id'], self.actor['id'])
        self.manager.save_frame_annotations(self.project, 'video', 0, [annotation(1), annotation(3, 'hive')])
        self.assertEqual(summarize(self.entry(), 'hive', [0])['state'], 'changed')
        mark_finished(self.project, 'video', 0, 'hive', self.actor)
        mark_finished(self.project, 'video', 0, 'hive', self.actor, complete=False)
        self.assertEqual(summarize(self.entry(), 'hive', [0])['state'], 'not_checked')

    def test_shared_counted_once_and_unrelated_bee_edit_preserves_completion(self):
        self.shared()
        self.assertEqual(summarize(self.entry(), 'hive')['total'], 1)
        mark_finished(self.project, 'video', 2, 'hive', self.actor)
        self.assertEqual(summarize(self.entry(), 'hive')['state'], 'finished')
        bee = annotation(1)
        bee['mask'][0, 0] = 255
        self.manager.save_frame_annotations(self.project, 'video', 2, [bee])
        self.assertEqual(summarize(self.entry(), 'hive')['state'], 'finished')
        hive = annotation(7, 'hive')
        hive['mask'][0, 0] = 255
        self.manager.save_video_annotations(self.project, 'video', [hive])
        self.assertEqual(summarize(self.entry(), 'hive', [5])['state'], 'changed')

    def test_shared_inventory_uses_actual_raster_not_stale_rle_metadata(self):
        self.shared()
        path = self.project / 'annotations/json/video/video_annotations.json'
        data = _read_json(path)
        data['annotations'][0]['mask_coco_rle'] = encode_mask(np.zeros((32, 40), np.uint8))
        _write_json(path, data)
        self.assertEqual(summarize(self.entry(), 'hive')['presence'], 'all')
        mark_finished(self.project, 'video', 0, 'hive', self.actor)
        cv2.imwrite(str(self.project / 'annotations/png/video/video_annotations_hive.png'), np.zeros((32, 40), np.uint16))
        self.assertEqual(summarize(self.entry(), 'hive')['state'], 'changed')

    def test_frame_geometry_signature_ignores_other_classes_and_attribution(self):
        self.manager.save_frame_annotations(self.project, 'video', 2, [annotation(1), annotation(3, 'hive')])
        mark_finished(self.project, 'video', 2, 'hive', self.actor)
        bee = annotation(1)
        bee['mask'][0, 0] = 255
        self.manager.save_frame_annotations(self.project, 'video', 2, [bee, annotation(3, 'hive')])
        self.assertEqual(summarize(self.entry(), 'hive', [2])['state'], 'finished')

    def test_brood_group_completion_and_category_move(self):
        self.manager.save_frame_annotations(self.project, 'video', 0, [annotation(8, 'brood_middle')])
        mark_finished(self.project, 'video', 0, 'brood', self.actor)
        self.assertEqual(summarize(self.entry(), 'brood', [0])['state'], 'finished')
        self.assertEqual(summarize(self.entry(), 'queen_brood', [0])['presence'], 'none')
        self.manager.save_frame_annotations(self.project, 'video', 0, [annotation(8, 'brood_late')])
        self.assertEqual(summarize(self.entry(), 'brood', [0])['state'], 'changed')

    def test_bee_hive_task_tracks_only_both_categories_at_their_saved_scopes(self):
        self.shared()
        self.assertEqual(task_categories('bee_hive'), ('bee', 'hive'))
        self.assertEqual(task_label('bee_hive'), 'Bees + Hive (nest)')
        mark_finished(self.project, 'video', 0, 'bee', self.actor)
        self.assertNotEqual(summarize(self.entry(), 'bee_hive', [0])['state'], 'finished')
        mark_finished(self.project, 'video', 0, 'bee_hive', self.actor)
        entry = self.entry()
        summary = summarize(entry, 'bee_hive', [0])
        self.assertEqual(summary['state'], 'finished')
        self.assertTrue(summary['mixed_scope'])
        self.assertEqual(summary['detail'], 'Frame-specific: Bee; shared across video: Hive')
        self.assertNotEqual(summarize(entry, 'bee_hive', [2])['state'], 'finished')
        self.assertEqual(summarize(entry, 'hive', [2])['state'], 'finished')
        self.assertEqual(set(_read_json(self.project / 'annotations/work/video/frame_000000.json')['categories']), {'bee'})
        self.assertEqual(set(_read_json(self.project / 'annotations/work/video/video_work.json')['categories']), {'hive'})
        self.manager.save_frame_annotations(self.project, 'video', 0, [annotation(1), annotation(8, 'brood_middle')])
        self.assertEqual(summarize(self.entry(), 'bee_hive', [0])['state'], 'finished')
        hive = annotation(7, 'hive')
        hive['mask'][0, 0] = 255
        self.manager.save_video_annotations(self.project, 'video', [hive])
        self.assertEqual(summarize(self.entry(), 'bee_hive', [0])['state'], 'changed')
        mark_finished(self.project, 'video', 0, 'bee_hive', self.actor, complete=False)
        self.assertEqual(summarize(self.entry(), 'bee_hive', [0])['state'], 'not_checked')

    def test_bee_hive_task_respects_frame_specific_hive_policy(self):
        mark_finished(self.project, 'video', 0, 'bee_hive', self.actor)
        entry = self.entry()
        self.assertEqual(summarize(entry, 'bee_hive', [0])['state'], 'finished')
        self.assertFalse(summarize(entry, 'bee_hive', [0])['mixed_scope'])
        self.assertNotEqual(summarize(entry, 'bee_hive', [2])['state'], 'finished')
        self.assertEqual(set(_read_json(self.project / 'annotations/work/video/frame_000000.json')['categories']),
                         {'bee', 'hive'})

    def test_corrupt_missing_and_legacy_sources_are_unknown_not_absent(self):
        self.shared()
        (self.project / 'annotations/png/video/video_annotations_hive.png').unlink()
        self.assertEqual(summarize(self.entry(), 'hive')['state'], 'unknown')
        self.assertIn('mask', summarize(self.entry(), 'hive')['detail'].lower())
        self.assertFalse(matches(summarize(self.entry(), 'hive'), 'all', 'none'))
        _write_json(self.project / 'annotations/json/video/video_annotations.json', [])
        self.manager.save_frame_annotations(self.project, 'video', 0, [annotation(3, 'hive')])
        self.assertEqual(summarize(self.entry(), 'hive')['state'], 'unknown')
        _write_json(self.project / 'annotations/json/video/frame_000000.json', {'annotations': 'invalid'})
        self.assertEqual(summarize(self.entry(), 'bee', [0])['state'], 'unknown')

    def test_unchanged_videos_reuse_index_without_loading_rasters(self):
        self.shared()
        first = local_index(self.project)
        with patch('core.annotation_work.Sources.mask', side_effect=AssertionError('No raster reload')):
            second = local_index(self.project, previous=first)
        self.assertIs(first['videos']['video'], second['videos']['video'])

    def test_shared_hive_scan_does_not_read_other_masks_or_frame_metadata(self):
        self.shared()
        reads = []
        original = Sources.json
        def record(source, name, default=None):
            reads.append(name)
            return original(source, name, default)
        with patch.object(Sources, 'json', record):
            index = local_index(self.project, video='video', task='hive')
        entry = index['videos']['video']
        self.assertEqual(set(entry['categories']), {'hive'})
        self.assertEqual(summarize(entry, 'hive')['mask_text'], 'Shared mask')
        self.assertFalse(any('/frame_' in name for name in reads))
        self.assertEqual(summarize(entry, 'bee')['state'], 'not_scanned')
        self.assertEqual(summarize(entry, None)['state'], 'not_scanned')

    def test_partial_cache_can_expand_but_cannot_certify_unchecked_categories(self):
        self.shared()
        hive = local_index(self.project, task='hive')
        bee = local_index(self.project, task='bee', previous=hive)
        self.assertEqual(set(bee['videos']['video']['categories']), {'bee', 'hive'})
        self.assertIs(bee['videos']['video']['categories']['hive'], hive['videos']['video']['categories']['hive'])
        from core.project_sync import _inventory, _safe_child
        files = _inventory(self.project, False, with_identity=True)
        cached = build_index(files, lambda n: _safe_child(self.project, n), previous=bee, cached_only=True)
        self.assertEqual(summarize(cached['videos']['video'], None)['state'], 'not_scanned')
        full = local_index(self.project, previous=bee)
        self.assertNotEqual(summarize(full['videos']['video'], None)['state'], 'not_scanned')
        # The partial input is immutable and is never silently promoted to full.
        self.assertEqual(set(hive['videos']['video']['categories']), {'hive'})

    def test_video_scope_never_walks_unrelated_directories(self):
        self.shared()
        import os
        real_walk, visited = os.walk, []
        def walk(path, *args, **kwargs):
            visited.append(str(path))
            return real_walk(path, *args, **kwargs)
        (self.project / 'frames/unrelated').symlink_to(self.project / 'frames/video', target_is_directory=True)
        with patch('core.annotation_work.os.walk', walk), patch('core.project_sync._inventory',
                side_effect=AssertionError('No project-wide sync inventory')):
            index = local_index(self.project, videos=['video'], task='hive')
        self.assertEqual(set(index['videos']), {'video'})
        self.assertTrue(visited)
        self.assertTrue(all(path.endswith('/video') for path in visited))
        with self.assertRaisesRegex(ValueError, 'symbolic'):
            local_index(self.project, video='unrelated')

    def test_incremental_results_follow_requested_priority_and_cancellation(self):
        import shutil
        shutil.copytree(self.project / 'frames/video', self.project / 'frames/other')
        delivered = []
        result = local_index(self.project, videos=['video', 'other'], task='bee',
                             video_ready=lambda video, entry: delivered.append(video))
        self.assertEqual(delivered, ['video', 'other'])
        self.assertEqual(set(result['videos']), {'video', 'other'})
        delivered.clear()
        with self.assertRaises(InterruptedError):
            local_index(self.project, videos=['video', 'other'], task='bee',
                        video_ready=lambda video, entry: delivered.append(video),
                        cancelled=lambda: bool(delivered))
        self.assertEqual(delivered, ['video'])

    def test_task_scan_keeps_legacy_and_import_review_safety_checks(self):
        self.shared()
        _write_json(self.project / 'annotations/json/video/video_annotations.json', [])
        self.manager.save_frame_annotations(self.project, 'video', 0, [annotation(3, 'hive')])
        entry = local_index(self.project, video='video', task='hive')['videos']['video']
        self.assertEqual(summarize(entry, 'hive')['state'], 'unknown')
        self.shared()
        records = {'version': 1, 'frames': {}, 'shared': {}}
        policy = self.info['annotation_policy']['categories']['hive']
        add_review(records, 'video', 0, 'hive', {'reasons': ['shared import'],
                   'source_policy': policy, 'destination_policy': policy}, shared=True)
        _write_json(self.project / 'annotations/import_review.json', records)
        entry = local_index(self.project, video='video', task='hive')['videos']['video']
        self.assertEqual(summarize(entry, 'hive')['state'], 'needs_review')

    def test_partial_cache_invalidates_after_save_and_policy_change(self):
        self.shared()
        mark_finished(self.project, 'video', 0, 'hive', self.actor)
        first = local_index(self.project, task='hive')
        self.assertEqual(summarize(first['videos']['video'], 'hive')['state'], 'finished')
        mask = annotation(7, 'hive')
        mask['mask'][0, 0] = 255
        self.manager.save_video_annotations(self.project, 'video', [mask])
        second = local_index(self.project, task='hive', previous=first)
        self.assertEqual(summarize(second['videos']['video'], 'hive')['state'], 'changed')
        self.info = {'hive_annotation_scope': 'frame'}
        self.info['annotation_policy'] = default_policy(self.info)
        _write_json(self.project / 'annotations/project.json', self.info)
        third = local_index(self.project, task='hive', previous=second)
        self.assertEqual(third['videos']['video']['categories']['hive']['scope'], 'frame')

    def test_import_reviews_are_not_approved_by_completion(self):
        records = {'version': 1, 'frames': {}, 'shared': {}}
        policy = self.info['annotation_policy']['categories']['hive']
        requirement = {'reasons': ['scope conversion'], 'source_policy': policy, 'destination_policy': policy}
        add_review(records, 'video', 2, 'hive', requirement)
        _write_json(self.project / 'annotations/import_review.json', records)
        mark_finished(self.project, 'video', 2, 'hive', self.actor)
        self.assertEqual(summarize(self.entry(), 'hive', [2])['state'], 'needs_review')
        with self.assertRaises(ValueError):
            require_reviewed(self.project, {'video': [2]})
        complete_review(self.project, 'video', 2, self.actor)
        self.assertEqual(summarize(self.entry(), 'hive', [2])['state'], 'finished')

    def test_invalid_frame_and_unconfirmed_policy_cannot_be_finished(self):
        with self.assertRaisesRegex(ValueError, 'available frame'):
            mark_finished(self.project, 'video', 1, 'hive', self.actor)
        _write_json(self.project / 'annotations/project.json', {'hive_annotation_scope': 'frame'})
        with self.assertRaisesRegex(ValueError, 'policy'):
            mark_finished(self.project, 'video', 0, 'hive', self.actor)
        with self.assertRaises(ValueError):
            mark_finished(self.project, 'video', 0, None, self.actor)

    def test_completion_cancellation_never_creates_a_record(self):
        checks = []
        def cancelled():
            checks.append(True)
            return len(checks) > 4
        with self.assertRaises(InterruptedError):
            mark_finished(self.project, 'video', 0, 'hive', self.actor, cancelled=cancelled)
        self.assertFalse((self.project / 'annotations/work').exists())

    def test_shared_completion_does_not_approve_other_imported_frames(self):
        self.shared()
        records = {'version': 1, 'frames': {}, 'shared': {}}
        policy = self.info['annotation_policy']['categories']['hive']
        requirement = {'reasons': ['shared import'], 'source_policy': policy, 'destination_policy': policy}
        add_review(records, 'video', 0, 'hive', requirement, shared=True)
        _write_json(self.project / 'annotations/import_review.json', records)
        mark_finished(self.project, 'video', 0, 'hive', self.actor)
        complete_review(self.project, 'video', 0, self.actor)
        self.assertEqual(summarize(self.entry(), 'hive')['work_text'], 'Import review: 2 frames')
        self.assertEqual(summarize(self.entry(), 'hive', [0])['state'], 'finished')
        with self.assertRaises(ValueError):
            require_reviewed(self.project, {'video': [2]})


class PublishedWorkTests(unittest.TestCase):
    setUp = assigned.AssignedWorkTests.setUp
    view = assigned.AssignedWorkTests.view
    download = assigned.AssignedWorkTests.download
    status = assigned.AssignedWorkTests.status

    def test_combined_task_requires_both_categories_before_auto_submission(self):
        data = read_assignments(self.view())
        self.choices['alpha'].update(task='bee_hive', frames=[0])
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        local = self.download()
        self.assertEqual(workspace(local)['assignments']['alpha']['task'], 'bee_hive')
        mark_finished(local, 'alpha', 0, 'bee', self.alex)
        result = publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertNotIn('alpha', result['ready'])
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        result = publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertIn('alpha', result['ready'])
        self.assertEqual(self.status()[0], 'Ready for Review')

    def test_publication_embeds_index_and_catalog_does_not_open_images(self):
        snapshot, _ = latest_snapshot(self.view())
        self.assertEqual(snapshot['manifest']['annotation_work']['version'], 1)
        metadata = self.view().snapshot(snapshot['reference'], metadata_only=True)
        self.assertNotIn('annotation_work', metadata['manifest'])
        with patch('core.annotation_work.Sources.mask', side_effect=AssertionError('Do not read rasters')):
            data = catalog(snapshot, include_work=True)
        self.assertEqual(summarize(data['alpha']['annotation_work'], 'hive')['mask_text'], 'Shared mask')
        snapshot['manifest'].pop('annotation_work')
        index = published_index(snapshot)
        self.assertEqual(summarize(index['videos']['alpha'], 'hive')['presence'], 'all')

    def test_task_assignment_round_trip_and_retirement(self):
        data = read_assignments(self.view())
        self.choices['alpha']['task'] = 'hive'
        updated = save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        self.assertNotEqual(data['videos']['alpha']['id'], updated['videos']['alpha']['id'])
        local = self.download()
        self.assertEqual(workspace(local)['assignments']['alpha']['task'], 'hive')
        self.choices['alpha']['task'] = 'bee'
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, read_assignments(self.view())['revision'])
        with self.assertRaisesRegex(SyncConflictError, 'Assignment changed'):
            publish_work(local, self.configuration, self.alex)

    def test_old_clients_carried_forward_index_cannot_report_stale_masks(self):
        snapshot, _ = latest_snapshot(self.view())
        self.assertEqual(summarize(published_index(snapshot)['videos']['alpha'], 'hive')['presence'], 'all')
        # Simulate an older publisher preserving optional metadata after deleting a mask.
        snapshot['manifest']['files'].pop('annotations/json/alpha/video_annotations.json')
        snapshot['manifest']['files'].pop('annotations/png/alpha/video_annotations_hive.png')
        self.assertEqual(summarize(published_index(snapshot)['videos']['alpha'], 'hive')['presence'], 'none')

    def test_completion_publishes_without_removing_undownloaded_frames(self):
        local = self.download()
        mark_finished(local, 'alpha', 0, 'hive', self.alex)
        with self.assertRaises(ValueError):
            mark_finished(local, 'alpha', 1, 'hive', self.alex)
        result = publish_work(local, self.configuration, self.alex)
        self.assertGreater(result['copied_files'], 0)
        snapshot, _ = latest_snapshot(self.view())
        self.assertIn('frames/alpha/frame_000002.jpg', snapshot['manifest']['files'])
        self.assertIn('annotations/work/alpha/video_work.json', snapshot['manifest']['files'])
        self.assertEqual(summarize(catalog(snapshot, include_work=True)['alpha']['annotation_work'], 'hive')['state'], 'finished')
        self.assertNotEqual(self.status()[0], 'Ready for Review')

    def test_catalog_and_cached_summaries_never_scan_missing_or_stale_annotations(self):
        snapshot, _ = latest_snapshot(self.view())
        with patch('core.annotation_work.published_index', side_effect=AssertionError('No work summaries needed')):
            self.assertEqual(set(catalog(snapshot)), {'alpha', 'beta'})
        snapshot['manifest']['files'].pop('annotations/json/alpha/video_annotations.json')
        with patch('core.annotation_work._unit', side_effect=AssertionError('No annotation scan at startup')):
            cached = published_index(snapshot, cached_only=True)
            self.assertEqual(summarize(cached['videos']['alpha'], 'hive')['state'], 'not_scanned')
            self.assertEqual(summarize(cached['videos']['beta'], 'hive')['presence'], 'all')
            snapshot['manifest'].pop('annotation_work')
            cached = catalog(snapshot, include_work=True)
            self.assertTrue(all(summarize(v['annotation_work'], 'hive')['state'] == 'not_scanned'
                                for v in cached.values()))

    def test_explicit_scan_reports_progress_and_cancels_without_partial_index(self):
        snapshot, _ = latest_snapshot(self.view())
        snapshot['manifest'].pop('annotation_work')
        messages = []
        with self.assertRaises(InterruptedError):
            published_index(snapshot, progress=messages.append, cancelled=lambda: bool(messages))
        self.assertTrue(messages)
        self.assertNotIn('annotation_work', snapshot['manifest'])
        index = published_index(snapshot, progress=messages.append)
        self.assertEqual(set(index['videos']), {'alpha', 'beta'})
        self.assertTrue(any('Frame ' in message for message in messages))


if __name__ == '__main__':
    unittest.main()
