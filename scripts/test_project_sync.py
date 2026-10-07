"""Contributor attribution and folder sync tests using temporary data only."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.annotation import AnnotationManager
from core.contributors import AnnotationAttributor, annotation_digest, new_session
from core.project_sync import (LOCATION_MARKERS, ROOT_MARKER, check_destination,
                              ensure_project_identity, prepare_sync_location,
                              sync_project, test_destination as probe_destination)
from scripts.attribute_project_annotations import attribute_project


def annotation(mask_id=1, category='bee'):
    mask = np.zeros((32, 40), np.uint8)
    mask[4:12, 7:22] = 255
    return {'mask_id': mask_id, 'category': category, 'mask': mask, 'source': 'manual'}


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = AnnotationManager()
        self.august = new_session('August')
        self.alex = new_session('Alex')
        self.manager.contributor_session = self.august

    def tearDown(self):
        self.temp.cleanup()

    def save(self, annotations, frame=0):
        self.manager.save_frame_annotations(self.root, 'video', frame, annotations)
        return self.manager.load_frame_annotations(self.root, 'video', frame)

    def test_new_annotations_record_creator_session_and_origin(self):
        source = annotation()
        saved = self.save([source])[0]['provenance']
        self.assertEqual(saved['created_by']['id'], self.august['id'])
        self.assertEqual(saved['last_modified_by']['name'], 'August')
        self.assertEqual(saved['session_id'], self.august['session_id'])
        self.assertEqual(saved['origin'], 'manual')
        self.assertEqual(source['provenance'], saved)

    def test_only_changed_instance_is_credited_to_new_contributor(self):
        saved = self.save([annotation(1), annotation(2)])
        before = saved[1]['provenance'].copy()
        self.manager.contributor_session = self.alex
        saved[0]['mask'][0, 0] = 255
        result = self.save(saved)
        self.assertEqual(result[0]['provenance']['created_by']['name'], 'August')
        self.assertEqual(result[0]['provenance']['last_modified_by']['name'], 'Alex')
        self.assertEqual(result[1]['provenance'], before)

    def test_unchanged_force_save_and_reload_dont_claim_authorship(self):
        saved = self.save([annotation()])
        before = saved[0]['provenance'].copy()
        self.manager = AnnotationManager()
        self.manager.contributor_session = self.alex
        saved[0]['area'] = 999
        saved[0]['color'] = [1, 2, 3]
        saved[0]['bbox'] = [0, 0, 32, 40]
        self.assertEqual(self.save(saved)[0]['provenance'], before)

    def test_legacy_creator_stays_unknown_even_after_editing(self):
        self.manager.contributor_session = None
        saved = self.save([annotation()])
        self.manager.contributor_session = self.august
        untouched = self.save(saved)[0]
        self.assertIsNone(untouched['provenance']['created_by'])
        self.assertNotIn('last_modified_by', untouched['provenance'])
        untouched['mask'][0, 0] = 255
        edited = self.save([untouched])[0]['provenance']
        self.assertIsNone(edited['created_by'])
        self.assertEqual(edited['last_modified_by']['name'], 'August')

    def test_reclassification_preserves_creator(self):
        saved = self.save([annotation()])
        self.manager.contributor_session = self.alex
        saved[0]['category'] = 'hive'
        result = self.save(saved)[0]['provenance']
        self.assertEqual(result['created_by']['name'], 'August')
        self.assertEqual(result['last_modified_by']['name'], 'Alex')

    def test_video_scope_move_keeps_unknown_legacy_creator(self):
        self.manager.contributor_session = None
        self.manager.save_video_annotations(self.root, 'video', [annotation(1, 'hive')])
        old, _ = self.manager.load_video_annotations(self.root, 'video')
        self.manager.contributor_session = self.alex
        result = self.save(old)[0]['provenance']
        self.assertIsNone(result['created_by'])
        self.assertEqual(result['last_modified_by']['name'], 'Alex')

    def test_bbox_edit_and_conversion_keep_attribution(self):
        box = {'mask_id': 1, 'bbox': [1, 2, 3, 4], 'bbox_only': True, 'category': 'bee'}
        saved = self.save([box])
        self.manager.contributor_session = self.alex
        saved[0]['bbox'][0] += 2
        result = self.save(saved)[0]
        self.assertEqual(result['provenance']['last_modified_by']['name'], 'Alex')
        result.pop('bbox_only')
        result['mask'] = annotation()['mask']
        self.assertEqual(self.save([result])[0]['provenance']['created_by']['name'], 'August')

    def test_reused_ids_in_different_categories_are_independent(self):
        saved = self.save([annotation(1, 'bee'), annotation(1, 'hive')])
        before = saved[1]['provenance'].copy()
        self.manager.contributor_session = self.alex
        saved[0]['mask'][1, 1] = 255
        result = self.save(saved)
        self.assertEqual(result[1]['provenance'], before)

    def test_video_masks_unchanged_edit_and_empty_deletion(self):
        manager = self.manager
        manager.save_video_annotations(self.root, 'video', [annotation(1, 'hive')])
        saved, _ = manager.load_video_annotations(self.root, 'video')
        before = saved[0]['provenance'].copy()
        manager.contributor_session = self.alex
        manager.save_video_annotations(self.root, 'video', saved)
        self.assertEqual(manager.load_video_annotations(self.root, 'video')[0][0]['provenance'], before)
        saved[0]['mask'][1, 1] = 255
        manager.save_video_annotations(self.root, 'video', saved)
        self.assertEqual(manager.load_video_annotations(self.root, 'video')[0][0]
                         ['provenance']['last_modified_by']['name'], 'Alex')
        manager.save_video_annotations(self.root, 'video', [])
        self.assertEqual(manager.load_video_annotations(self.root, 'video')[0], [])

    def test_model_origin_survives_manual_edit(self):
        generated = annotation()
        generated['source'] = 'yolo'
        saved = self.save([generated])
        self.manager.contributor_session = self.alex
        saved[0]['mask'][0, 0] = 255
        self.assertEqual(self.save(saved)[0]['provenance']['origin'], 'yolo')

    def test_failed_save_does_not_advance_baseline(self):
        saved = self.save([annotation()])
        previous = dict(self.manager.attributor.baselines)
        saved[0]['mask'][0, 0] = 255
        with patch.object(self.manager, 'save_frame_annotations_png', side_effect=OSError('full')):
            with self.assertRaises(OSError):
                self.save(saved)
        self.assertEqual(self.manager.attributor.baselines, previous)

    def test_copied_annotation_preserves_creator_and_numpy_metadata_roundtrips(self):
        ann = annotation()
        ann['confidence'] = np.float32(0.7)
        saved = self.save([ann])
        self.manager = AnnotationManager()
        self.manager.contributor_session = self.alex
        unchanged = self.save(saved)[0]['provenance']
        self.assertEqual(unchanged['last_modified_by']['name'], 'August')
        copied = self.save(saved, frame=1)[0]['provenance']
        self.assertEqual(copied['created_by']['name'], 'August')
        self.assertEqual(copied['last_modified_by']['name'], 'Alex')

    def test_deleting_then_recreating_id_does_not_inherit_deleted_authorship(self):
        self.save([annotation()])
        self.save([])
        self.manager.contributor_session = self.alex
        self.assertEqual(self.save([annotation()])[0]['provenance']['created_by']['name'], 'Alex')

    def test_baselines_are_bounded_and_digest_ignores_binary_scale(self):
        cache = AnnotationAttributor(max_sources=2)
        ann = annotation()
        other = dict(ann, mask=ann['mask'] > 0)
        self.assertEqual(annotation_digest(ann), annotation_digest(other))
        for key in range(4):
            cache.committed(key, [ann])
        self.assertEqual(list(cache.baselines), [2, 3])


class LegacyAttributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'annotations').mkdir()
        (self.root / 'annotations/project.json').write_text('{}')
        manager = AnnotationManager()
        manager.save_frame_annotations(self.root, 'video', 0, [annotation(1, 'hive')])
        manager.save_video_annotations(self.root, 'video', [annotation(2, 'chamber')])
        self.contributor = new_session('August')

    def tearDown(self):
        self.temp.cleanup()

    def attribute(self, **kwargs):
        return attribute_project(self.root, self.contributor, **kwargs)

    def test_dry_run_and_closed_app_confirmation_do_not_modify_files(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertGreater(self.attribute()['records'], 0)
        with self.assertRaisesRegex(ValueError, 'close the annotator'):
            self.attribute(apply=True)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_assigns_all_scopes_without_changing_geometry_or_dates_and_is_idempotent(self):
        before = {p.relative_to(self.root): p.read_bytes()
                  for p in (self.root / 'annotations').rglob('*') if p.is_file()}
        result = self.attribute(apply=True, app_closed=True)
        self.assertTrue(result['applied'])
        for relative, original in before.items():
            actual = (self.root / relative).read_bytes()
            if relative.suffix != '.json' or 'project.json' in relative.parts:
                self.assertEqual(actual, original)
                continue
            old, new = json.loads(original), json.loads(actual)
            old_anns = old['annotations'] if isinstance(old, dict) else old
            new_anns = new['annotations'] if isinstance(new, dict) else new
            for old_ann, new_ann in zip(old_anns, new_anns):
                provenance = new_ann.pop('provenance')
                old_ann.pop('provenance', None)
                self.assertEqual(old_ann, new_ann)
                self.assertEqual(provenance['created_by']['id'], self.contributor['id'])
                self.assertIsNone(provenance['created_at'])
                self.assertNotIn('last_modified_at', provenance)
                self.assertEqual(len(provenance['attribution_history']), 1)
            if old_anns:
                self.assertEqual((Path(result['backup']) / relative).read_bytes(), original)
        self.assertEqual(self.attribute(apply=True, app_closed=True)['records'], 0)
        # Normal saves preserve the retrospectively assigned creator.
        manager = AnnotationManager()
        manager.contributor_session = new_session('Alex')
        loaded = manager.load_frame_annotations(self.root, 'video', 0)
        manager.save_frame_annotations(self.root, 'video', 0, loaded)
        self.assertEqual(loaded[0]['provenance']['created_by']['name'], 'August')

    def test_refuses_existing_other_creator_before_any_writes(self):
        file = self.root / 'annotations/json/video/frame_000000.json'
        data = json.loads(file.read_text())
        data[0]['provenance'] = {'created_by': new_session('Alex')}
        file.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'another contributor'):
            self.attribute(apply=True, app_closed=True)
        self.assertFalse((self.root / 'attribution_backups').exists())

    def test_failed_write_restores_previous_files(self):
        import core.project_sync as sync
        before = {p: p.read_bytes() for p in (self.root / 'annotations').rglob('*') if p.is_file()}
        def fail(path, value):
            if Path(path).parent == self.root / 'annotations/json/video':
                raise OSError('disk full')
            sync._write_json(path, value)
        with patch('scripts.attribute_project_annotations._write_json', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.attribute(apply=True, app_closed=True)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertTrue(list((self.root / 'attribution_backups').glob('*/manifest.json')))


class FolderSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.remote = self.base / 'remote'
        self.remote.mkdir()
        for relative, data in {
            'annotations/project.json': '{}',
            'annotations/json/video/frame_000000.json': '[]',
            'annotations/png/video/frame_000000.png': 'mask',
            'frames/video/frame_000000.jpg': 'image',
            'frames/video/video_metadata.json': '{}',
            'input_data/train/video.mp4': 'video',
            'tracking_sequences.json': '{}',
            'models/large.pt': 'excluded',
            'annotations/coco/train/video.json': 'excluded',
            'marker_debug/debug.png': 'excluded',
            'runs/result.txt': 'excluded',
        }.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        self.configuration = probe_destination(self.remote)
        self.contributor = new_session('August')

    def tearDown(self):
        self.temp.cleanup()

    def sync(self, **kwargs):
        return sync_project(self.source, self.configuration, self.contributor, **kwargs)

    def test_probe_checks_identity_and_cleans_up_only_its_files(self):
        existing = self.remote / 'keep.txt'
        existing.write_text('keep')
        self.assertEqual(probe_destination(self.remote), self.configuration)
        self.assertEqual(existing.read_text(), 'keep')
        self.assertFalse(list(self.remote.glob('.bumblebox-test-*')))
        with self.assertRaises(ValueError):
            probe_destination('')
        with self.assertRaises(ValueError):
            check_destination(self.remote, 'wrong')

    def test_first_revision_contains_complete_source_and_excludes_outputs(self):
        result = self.sync()
        project = Path(result['path'])
        self.assertEqual((project / 'input_data/train/video.mp4').read_text(), 'video')
        self.assertTrue((project / 'tracking_sequences.json').exists())
        self.assertTrue((project / 'frames/video/video_metadata.json').exists())
        for excluded in ('models', 'runs', 'marker_debug', 'annotations/coco'):
            self.assertFalse((project / excluded).exists())
        self.assertEqual(result['copied_files'], 8)
        self.assertFalse(list(self.remote.rglob('.pending-*')))
        self.assertFalse(list(self.remote.rglob('.sync-lock')))

    def test_unchanged_sync_reuses_revision_and_changed_sync_keeps_history(self):
        first = self.sync()
        second = self.sync()
        self.assertEqual(first['path'], second['path'])
        self.assertTrue(second['unchanged'])
        relative = 'annotations/json/video/frame_000000.json'
        (self.source / relative).write_text('[{"new": true}]')
        changed = self.sync()
        self.assertEqual(changed['copied_files'], 1)
        self.assertEqual((Path(first['path']) / relative).read_text(), '[]')
        self.assertEqual((Path(changed['path']) / relative).read_text(), '[{"new": true}]')
        # No writable hard links between independently reopenable revisions.
        unchanged = 'frames/video/video_metadata.json'
        (Path(changed['path']) / unchanged).write_text('edited')
        self.assertEqual((Path(first['path']) / unchanged).read_text(), '{}')

    def test_deletions_are_absent_in_new_revision_but_old_copy_survives(self):
        first = Path(self.sync()['path'])
        relative = 'annotations/png/video/frame_000000.png'
        (self.source / relative).unlink()
        second = Path(self.sync()['path'])
        self.assertTrue((first / relative).exists())
        self.assertFalse((second / relative).exists())

    def test_different_contributors_never_write_each_others_copy(self):
        first = self.sync()
        self.contributor = new_session('Alex')
        second = self.sync()
        self.assertNotEqual(first['path'], second['path'])
        self.assertTrue(Path(first['path']).exists())

    def test_unmounted_or_replaced_destination_is_not_recreated(self):
        self.remote.rename(self.base / 'disconnected')
        with self.assertRaises(ValueError):
            self.sync()
        self.assertFalse(self.remote.exists())
        self.remote.mkdir()
        with self.assertRaisesRegex(ValueError, 'marker is missing'):
            self.sync()
        self.assertEqual(list(self.remote.iterdir()), [])

    def test_nested_source_and_destination_and_symlinks_are_rejected(self):
        original = dict(self.configuration)
        self.configuration = probe_destination(self.source)
        with self.assertRaisesRegex(ValueError, 'non-nested'):
            self.sync()
        self.configuration = original
        (self.source / 'frames/link').symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            self.sync()

    def test_cancel_and_copy_failure_never_publish_partial_revision(self):
        first = self.sync()
        latest = next(self.remote.rglob('latest.json'))
        before = latest.read_bytes()
        (self.source / 'annotations/new.json').write_text('new')
        with self.assertRaises(InterruptedError):
            self.sync(cancelled=lambda: True)
        with patch('core.project_sync.shutil.copy2', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.sync()
        self.assertEqual(latest.read_bytes(), before)
        self.assertTrue(Path(first['path']).exists())
        self.assertFalse(list(self.remote.rglob('.pending-*')))

    def test_source_changes_during_sync_abort_publish(self):
        def progress(message):
            if message.startswith('Copying 1/'):
                (self.source / 'annotations/late.json').write_text('new')
        with self.assertRaisesRegex(ValueError, 'changed during sync'):
            self.sync(progress=progress)
        self.assertFalse(list(self.remote.rglob('latest.json')))

    def test_remote_edit_is_not_silently_overwritten(self):
        first = Path(self.sync()['path'])
        (first / 'frames/video/frame_000000.jpg').write_text('collaborator edit')
        with self.assertRaisesRegex(ValueError, 'edited or damaged'):
            self.sync()
        self.assertEqual((first / 'frames/video/frame_000000.jpg').read_text(), 'collaborator edit')

    def test_excluding_videos_is_explicit_and_does_not_delete_old_revision(self):
        first = Path(self.sync()['path'])
        self.configuration['include_videos'] = False
        second = Path(self.sync()['path'])
        self.assertTrue((first / 'input_data').exists())
        self.assertFalse((second / 'input_data').exists())
        self.assertTrue((second / 'frames').exists())

    def test_concurrent_same_contributor_sync_is_rejected(self):
        self.sync()
        home = next(self.remote.rglob('latest.json')).parent
        lock = home / '.sync-lock'
        lock.write_text('owned by another run')
        with self.assertRaisesRegex(ValueError, 'locked'):
            self.sync()
        self.assertEqual(lock.read_text(), 'owned by another run')

    def test_identity_survives_project_copy(self):
        identity = ensure_project_identity(self.source)
        self.assertEqual(identity, ensure_project_identity(self.source))
        with self.assertRaises(ValueError):
            ensure_project_identity(self.base)

    def legacy_location(self):
        first = Path(self.sync()['path'])
        home = first.parents[2]
        project_home = home.parent.parent
        contributor_marker = home / LOCATION_MARKERS['contributor']
        contributor_marker.unlink()
        home.rename(home.parent / self.contributor['id'])
        (project_home / LOCATION_MARKERS['project']).unlink()
        project_id = ensure_project_identity(self.source)
        legacy = project_home.parent / project_id
        project_home.rename(legacy)
        return legacy, first.parent.name

    def test_readable_names_include_short_ids_and_preserve_full_identity(self):
        (self.source / 'annotations/project.json').write_text('{"name": "Hive Test"}')
        project = Path(self.sync()['path'])
        home = project.parents[2]
        self.assertEqual(home.name, 'August--' + self.contributor['id'][:8])
        self.assertEqual(home.parent.parent.name, 'Hive-Test--' + ensure_project_identity(self.source)[:8])
        marker = json.loads((home / LOCATION_MARKERS['contributor']).read_text())
        self.assertEqual(marker['id'], self.contributor['id'])

    def test_legacy_layout_migrates_without_copying_or_modifying_revisions(self):
        legacy, revision = self.legacy_location()
        original = legacy / 'contributors' / self.contributor['id'] / 'revisions'
        before = {p.relative_to(original): p.read_bytes() for p in original.rglob('*') if p.is_file()}
        with patch('core.project_sync.shutil.copy2', side_effect=AssertionError('must not copy')):
            home = prepare_sync_location(self.source, self.configuration, self.contributor)
            result = self.sync()
        self.assertFalse(legacy.exists())
        self.assertTrue(result['unchanged'])
        self.assertEqual(Path(result['path']), home / 'revisions' / revision / 'project')
        for relative, data in before.items():
            self.assertEqual((home / 'revisions' / relative).read_bytes(), data)

    def test_project_migration_retains_other_contributors(self):
        legacy, _ = self.legacy_location()
        other = legacy / 'contributors' / new_session('Alex')['id']
        other.mkdir()
        (other / 'keep.txt').write_text('other contribution')
        home = prepare_sync_location(self.source, self.configuration, self.contributor)
        self.assertEqual((home.parent / other.name / 'keep.txt').read_text(), 'other contribution')

    def test_display_name_changes_reuse_existing_full_identity(self):
        first = Path(self.sync()['path'])
        self.contributor['name'] = 'August renamed'
        self.assertEqual(Path(self.sync()['path']), first)
        (self.source / 'annotations/project.json').write_text('{"name":"Renamed project"}')
        self.assertEqual(Path(self.sync()['path']).parents[2], first.parents[2])

    def test_short_id_collision_uses_full_id_without_touching_other_folder(self):
        first = Path(self.sync()['path'])
        identity = self.contributor['id']
        replacement = '0' if identity[-1] != '0' else '1'
        self.contributor = new_session('August', identity[:-1] + replacement)
        second = Path(self.sync()['path'])
        self.assertEqual(second.parents[2].name, 'August--' + self.contributor['id'])
        self.assertTrue(first.exists())

    def test_unsafe_labels_cannot_escape_destination(self):
        (self.source / 'annotations/project.json').write_text('{"name":"../../a/b\\\\c"}')
        self.contributor['name'] = '../ /'
        project = Path(self.sync()['path'])
        self.assertEqual(project.parents[4].parent, self.remote)
        self.assertTrue(project.parents[2].name.startswith('contributor--'))

    def test_duplicate_identity_is_rejected_without_merging(self):
        first = Path(self.sync()['path'])
        home = first.parents[2]
        shutil.copytree(home, home.parent / 'duplicate')
        with self.assertRaisesRegex(ValueError, 'Multiple sync folders'):
            self.sync()

    def test_active_legacy_sync_prevents_migration(self):
        legacy, _ = self.legacy_location()
        lock = legacy / 'contributors' / self.contributor['id'] / '.sync-lock'
        lock.write_text('other run')
        with self.assertRaisesRegex(ValueError, 'locked'):
            self.sync()
        self.assertEqual(lock.read_text(), 'other run')

    def test_project_lock_prevents_migration_and_remains_owned_by_other_run(self):
        identity = ensure_project_identity(self.source)
        lock = self.remote / f'.project-{identity}.sync-lock'
        lock.write_text('other run')
        with self.assertRaisesRegex(ValueError, 'project is locked'):
            self.sync()
        self.assertEqual(lock.read_text(), 'other run')

    def test_failed_legacy_rename_can_be_retried(self):
        legacy, _ = self.legacy_location()
        with patch.object(Path, 'rename', side_effect=OSError('access denied')):
            with self.assertRaisesRegex(OSError, 'access denied'):
                self.sync()
        self.assertTrue(legacy.exists())
        self.assertTrue(self.sync()['unchanged'])


if __name__ == '__main__':
    unittest.main()
