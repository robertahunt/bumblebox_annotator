"""Project copies and scope conversion tests, using only disposable projects."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.annotation import AnnotationManager
from core.contributors import new_session
from core.project_import import inspect_project, prepare_import
from core.project_manager import ProjectManager
from core.project_sync import _read_json, _write_json
from training.coco_video_export import export_coco_per_video


def mask_annotation(identity=1, category='hive', offset=0):
    mask = np.zeros((64, 96), np.uint8)
    mask[8:38, 10 + offset:40 + offset] = 255
    mask[16:23, 18 + offset:25 + offset] = 0
    mask[45:50, 10 + offset:15 + offset] = 255
    return {'mask_id': identity, 'category': category, 'mask': mask,
            'source': 'manual', 'provenance': {'created_by': {'id': 'creator', 'name': 'Original author'},
                                             'created_at': '2026-01-02T00:00:00+00:00'}}


def file_bytes(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*')
            if p.is_file() and 'import_backups' not in p.relative_to(root).parts}


class ProjectImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source, self.dest = self.base / 'source', self.base / 'dest'
        ProjectManager().create_project(self.source, 'Source', hive_annotation_scope='frame')
        ProjectManager().create_project(self.dest, 'Destination', hive_annotation_scope='video')
        self.manager = AnnotationManager()
        self.session = new_session('Importer')
        self.frames(self.source)
        self.manager.save_frame_annotations(self.source, 'video', 2, [mask_annotation()], contributor=None)
        self.manager.save_frame_annotations(self.source, 'video', 5, [mask_annotation(offset=20)], contributor=None)
        self.plans = []

    def tearDown(self):
        for plan in self.plans:
            plan.close()
        self.temp.cleanup()

    def frames(self, project, split='train', indices=(2, 5), video=True):
        directory = project / 'frames/video'
        directory.mkdir(parents=True, exist_ok=True)
        for index in indices:
            cv2.imwrite(str(directory / f'frame_{index:06d}.jpg'), np.full((64, 96, 3), index, np.uint8))
        _write_json(directory / 'video_metadata.json', {'split': split, 'selected_frames': list(indices),
                                                      'total_frames': 10, 'extraction_mode': 'selected'})
        if video:
            (project / 'input_data' / split).mkdir(parents=True, exist_ok=True)
            (project / 'input_data' / split / 'video.mp4').write_bytes(b'original video fixture')

    def plan(self, indices=(2, 5), categories=('hive',), **kwargs):
        plan = prepare_import(self.source, self.dest, {'video': list(indices)}, categories,
                              contributor=self.session, **kwargs)
        self.plans.append(plan)
        return plan

    def set_scope(self, project, scope):
        path = project / 'annotations/project.json'
        data = _read_json(path)
        data['hive_annotation_scope'] = scope
        _write_json(path, data)

    def test_preview_changes_neither_project_and_defaults_to_first_nonempty_frame(self):
        source_before, dest_before = file_bytes(self.source), file_bytes(self.dest)
        plan = self.plan()
        row = next(r for r in plan.rows if r['scope'] == 'Whole video')
        self.assertEqual(row['frame'], 2)
        self.assertEqual(source_before, file_bytes(self.source))
        self.assertEqual(dest_before, file_bytes(self.dest))
        plan.apply()
        self.assertEqual(source_before, file_bytes(self.source))
        loaded = self.manager.load_video_annotations(self.dest, 'video')[0]
        self.assertEqual(len(loaded), 1)
        np.testing.assert_array_equal(loaded[0]['mask'], mask_annotation()['mask'])
        self.assertEqual(self.manager.load_frame_annotations(self.dest, 'video', 2), [])

    def test_explicit_later_reference_and_reference_must_be_selected(self):
        self.plan(hive_frames={'video': 5}).apply()
        loaded = self.manager.load_video_annotations(self.dest, 'video')[0]
        np.testing.assert_array_equal(loaded[0]['mask'], mask_annotation(offset=20)['mask'])
        with self.assertRaisesRegex(ValueError, 'chosen hive reference'):
            self.plan(indices=(2,), hive_frames={'video': 5})

    def test_multiple_instances_from_chosen_frame_are_not_collapsed(self):
        annotations = [mask_annotation(), mask_annotation(7, offset=42)]
        self.manager.save_frame_annotations(self.source, 'video', 2, annotations, contributor=None)
        self.plan().apply()
        loaded = self.manager.load_video_annotations(self.dest, 'video')[0]
        self.assertEqual(len(loaded), 2)
        for actual, expected in zip(loaded, annotations):
            np.testing.assert_array_equal(actual['mask'], expected['mask'])

    def test_empty_earlier_mask_does_not_hide_later_annotation(self):
        annotation = mask_annotation()
        annotation['mask'][:] = 0
        self.manager.save_frame_annotations(self.source, 'video', 2, [annotation], contributor=None)
        plan = self.plan()
        self.assertEqual(next(r for r in plan.rows if r['scope'] == 'Whole video')['frame'], 5)

    def test_source_authorship_dates_and_pixels_survive_import(self):
        self.plan().apply()
        annotation = self.manager.load_video_annotations(self.dest, 'video')[0][0]
        provenance = annotation['provenance']
        self.assertEqual(provenance['created_by']['name'], 'Original author')
        self.assertEqual(provenance['created_at'], '2026-01-02T00:00:00+00:00')
        self.assertNotIn('last_modified_by', provenance)
        history = provenance['import_history'][-1]
        self.assertEqual(history['imported_by']['id'], self.session['id'])
        self.assertEqual(history['source_scope'], 'frame')
        self.assertEqual(history['destination_scope'], 'video')
        self.assertEqual(history['frame_index'], 2)
        self.assertTrue(history['scope_review_required'])

    def test_frame_to_frame_preserves_both_different_masks(self):
        self.set_scope(self.dest, 'frame')
        self.plan().apply()
        self.assertEqual(self.manager.load_video_annotations(self.dest, 'video')[0], [])
        for index, offset in ((2, 0), (5, 20)):
            annotation = self.manager.load_frame_annotations(self.dest, 'video', index)[0]
            np.testing.assert_array_equal(annotation['mask'], mask_annotation(offset=offset)['mask'])

    def test_video_to_frame_materializes_selected_images_with_review_warning(self):
        self.set_scope(self.source, 'video')
        self.set_scope(self.dest, 'frame')
        self.manager.save_video_annotations(self.source, 'video', [mask_annotation(offset=10)], contributor=None)
        plan = self.plan()
        self.assertTrue(any('review visible pixels' in w for w in plan.warnings))
        plan.apply()
        for index in (2, 5):
            actual = self.manager.load_frame_annotations(self.dest, 'video', index)[0]
            np.testing.assert_array_equal(actual['mask'], mask_annotation(offset=10)['mask'])

    def test_keep_destination_is_default_and_repeat_does_not_duplicate(self):
        self.frames(self.dest)
        old = mask_annotation(20, offset=4)
        self.manager.save_video_annotations(self.dest, 'video', [old], contributor=None)
        before = file_bytes(self.dest / 'annotations')
        self.plan().apply()
        self.plan().apply()
        self.assertEqual(before, file_bytes(self.dest / 'annotations'))

    def test_replace_shared_backups_and_preserves_other_categories_and_tracking(self):
        self.frames(self.dest)
        self.manager.save_video_annotations(self.dest, 'video',
                                           [mask_annotation(1), mask_annotation(8, 'chamber')], contributor=None)
        path = self.dest / 'annotations/json/video/video_annotations.json'
        data = _read_json(path)
        data['aruco_tracking'] = {'4': 8}
        data['metadata'] = {'keep': True}
        _write_json(path, data)
        before = path.read_bytes()
        result = self.plan(replace=True).apply()
        after = _read_json(path)
        self.assertEqual(after['aruco_tracking'], data['aruco_tracking'])
        self.assertEqual(after['metadata'], data['metadata'])
        self.assertEqual(len(after['annotations']), 2)
        self.assertEqual((Path(result['backup']) / 'originals' / path.relative_to(self.dest)).read_bytes(), before)
        self.assertEqual([a['mask_id'] for a in after['annotations'] if a['category'] == 'chamber'], [8])

    def test_replacement_removes_legacy_frame_shared_copies_even_outside_selection(self):
        self.frames(self.dest, indices=(2, 5, 7))
        for index in (2, 5, 7):
            self.manager.save_frame_annotations(self.dest, 'video', index,
                                                [mask_annotation(), mask_annotation(3, 'bee')], contributor=None)
        self.plan(indices=(2,), replace=True).apply()
        for index in (2, 5, 7):
            anns = self.manager.load_frame_annotations(self.dest, 'video', index)
            self.assertEqual([a['category'] for a in anns], ['bee'])

    def test_frame_conflicts_keep_other_classes_and_remap_ids(self):
        self.set_scope(self.dest, 'frame')
        self.frames(self.dest)
        self.manager.save_frame_annotations(self.dest, 'video', 2,
                                            [mask_annotation(1, 'bee'), mask_annotation(2)], contributor=None)
        self.plan(indices=(2,), replace=True).apply()
        anns = self.manager.load_frame_annotations(self.dest, 'video', 2)
        self.assertEqual([a['category'] for a in anns], ['bee', 'hive'])
        self.assertEqual(anns[0]['mask_id'], 1)
        self.assertGreater(anns[1]['mask_id'], 2)
        self.assertNotIn('max_mask_id', _read_json(self.dest / 'frames/video/video_metadata.json'))

    def test_importing_different_categories_with_reused_source_id_remaps_independently(self):
        self.manager.save_frame_annotations(self.source, 'video', 2,
                                            [mask_annotation(), mask_annotation(1, 'bee')], contributor=None)
        self.plan(categories=('hive', 'bee')).apply()
        local = self.manager.load_frame_annotations(self.dest, 'video', 2)
        shared = self.manager.load_video_annotations(self.dest, 'video')[0]
        self.assertNotEqual(local[0]['mask_id'], shared[0]['mask_id'])

    def test_frames_only_import_is_discovered_reopened_and_can_move_split(self):
        self.plan().apply()
        manager = ProjectManager(self.dest)
        self.assertIsNone(manager.get_video_path('video'))
        self.assertEqual(manager.get_video_split('video'), 'train')
        self.assertEqual(manager.get_videos_by_split('train'), ['video'])
        self.assertEqual(manager.scan_videos()['train'], ['video'])
        self.assertTrue(manager.move_video('video', 'val'))
        self.assertEqual(manager.get_videos_by_split('val'), ['video'])

    def test_original_video_copy_optional_and_project_identity_unchanged(self):
        (self.source / 'annotations/project_sync.json').write_text('{"project_id":"source-id"}')
        (self.dest / 'annotations/project_sync.json').write_text('{"project_id":"destination-id"}')
        before = (self.dest / 'annotations/project.json').read_bytes()
        self.plan(include_videos=True).apply()
        self.assertEqual((self.dest / 'input_data/train/video.mp4').read_bytes(), b'original video fixture')
        self.assertEqual((self.dest / 'annotations/project.json').read_bytes(), before)
        self.assertEqual(_read_json(self.dest / 'annotations/project_sync.json')['project_id'], 'destination-id')

    def test_missing_original_does_not_prevent_frame_only_import(self):
        (self.source / 'input_data/train/video.mp4').unlink()
        self.plan().apply()
        with self.assertRaisesRegex(ValueError, 'original video is unavailable'):
            self.plan(include_videos=True)

    def test_split_collision_rejected_and_explicit_reassignment_warns(self):
        self.frames(self.dest, split='val')
        with self.assertRaisesRegex(ValueError, 'already belongs to val'):
            self.plan()
        plan = self.plan(split='val')
        self.assertTrue(any('train -> val' in w for w in plan.warnings))

    def test_duplicate_video_across_splits_is_rejected(self):
        (self.source / 'input_data/val/video.mp4').write_bytes(b'duplicate')
        with self.assertRaisesRegex(ValueError, 'multiple dataset splits'):
            self.plan()

    def test_different_images_with_same_video_id_are_rejected(self):
        self.frames(self.dest)
        cv2.imwrite(str(self.dest / 'frames/video/frame_000002.jpg'), np.full((64, 96, 3), 50, np.uint8))
        with self.assertRaisesRegex(ValueError, 'different reference image pixels'):
            self.plan()

    def test_changed_selected_image_not_just_reference_is_rejected(self):
        self.frames(self.dest)
        cv2.imwrite(str(self.dest / 'frames/video/frame_000005.jpg'), np.full((64, 96, 3), 50, np.uint8))
        with self.assertRaisesRegex(ValueError, 'different pixels'):
            self.plan()

    def test_mask_dimension_mismatch_and_missing_mask_fail_before_writes(self):
        path = self.source / 'frames/video/frame_000002.jpg'
        cv2.imwrite(str(path), np.zeros((65, 96, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, 'dimensions differ'):
            self.plan()
        cv2.imwrite(str(path), np.zeros((64, 96, 3), np.uint8))
        (self.source / 'annotations/png/video/frame_000002.png').unlink()
        with self.assertRaisesRegex(ValueError, 'missing or unreadable'):
            self.plan()

    def test_same_category_overlap_refused_for_video_storage(self):
        self.manager.save_frame_annotations(self.source, 'video', 2,
                                            [mask_annotation(), mask_annotation(2, offset=5)], contributor=None)
        with self.assertRaisesRegex(ValueError, 'Overlapping instances'):
            self.plan()

    def test_source_or_destination_edits_invalidate_preview(self):
        for project in (self.source, self.dest):
            plan = self.plan()
            path = project / 'annotations/project.json'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'changed after'):
                plan.apply()

    def test_cancel_leaves_destination_unchanged(self):
        before = file_bytes(self.dest)
        plan = self.plan()
        with self.assertRaises(InterruptedError):
            plan.apply(cancelled=lambda: True)
        self.assertEqual(file_bytes(self.dest), before)

    def test_publication_failure_rolls_back_destination_files(self):
        self.frames(self.dest)
        self.manager.save_video_annotations(self.dest, 'video', [mask_annotation(10)], contributor=None)
        plan = self.plan(replace=True)
        before = file_bytes(self.dest)
        replace = os.replace
        writes = []
        def fail(source, destination):
            if str(destination).startswith(str(self.dest / 'annotations')):
                writes.append(str(destination))
                if len(writes) == 2:
                    raise OSError('simulated failure')
            return replace(source, destination)
        with patch('core.project_import.os.replace', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'simulated failure'):
                plan.apply()
        self.assertEqual(file_bytes(self.dest), before)
        self.assertEqual(_read_json(next((self.dest / 'import_backups').glob('*/manifest.json')))['status'], 'rolled_back')

    def test_external_edit_just_before_publication_is_not_overwritten(self):
        self.frames(self.dest)
        self.manager.save_video_annotations(self.dest, 'video', [mask_annotation(10)], contributor=None)
        plan = self.plan(replace=True)
        path = self.dest / 'annotations/json/video/video_annotations.json'
        changed = _read_json(path)
        changed['metadata'] = {'external': True}
        def write_json(target, value):
            _write_json(target, value)
            if value.get('status') == 'publishing':
                _write_json(path, changed)
        with patch('core.project_import._write_json', side_effect=write_json):
            with self.assertRaisesRegex(ValueError, 'changed during publication'):
                plan.apply()
        self.assertEqual(_read_json(path), changed)

    def test_rollback_preserves_subsequent_external_edits(self):
        self.frames(self.dest)
        self.manager.save_video_annotations(self.dest, 'video', [mask_annotation(10)], contributor=None)
        plan = self.plan(replace=True)
        replace = os.replace
        writes = []
        def fail(source, destination):
            if str(destination).startswith(str(self.dest / 'annotations')):
                writes.append(Path(destination))
                if len(writes) == 2:
                    writes[0].write_text('{"annotations": [], "external_edit": true}')
                    raise OSError('simulated failure')
            return replace(source, destination)
        with patch('core.project_import.os.replace', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'Those edits were preserved'):
                plan.apply()
        self.assertTrue(_read_json(writes[0])['external_edit'])
        manifest = _read_json(next((self.dest / 'import_backups').glob('*/manifest.json')))
        self.assertEqual(manifest['status'], 'rollback_incomplete')

    def test_rejects_nested_projects_unknown_frames_and_symlinks(self):
        with self.assertRaisesRegex(ValueError, 'separate, non-nested'):
            prepare_import(self.source, self.source, {'video': [2]}, ['hive'])
        with self.assertRaisesRegex(ValueError, 'Unknown source'):
            self.plan(indices=(999,))
        (self.source / 'frames/link').symlink_to(self.dest, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            self.plan()

    def test_export_after_frame_only_import_includes_expected_training_frames(self):
        self.set_scope(self.dest, 'frame')
        self.plan().apply()
        exported = export_coco_per_video(self.dest, ['video'], 'train')
        self.assertTrue(exported)
        data = _read_json(self.dest / 'annotations/coco/train/video.json')
        self.assertEqual(len(data['images']), 2)
        self.assertEqual(len(data['annotations']), 2)
        self.assertEqual({a['category_id'] for a in data['annotations']}, {2})

    def test_shared_only_import_does_not_change_existing_export_eligibility(self):
        self.plan().apply()
        self.assertEqual(export_coco_per_video(self.dest, ['video'], 'train'), [])


if __name__ == '__main__':
    unittest.main()
