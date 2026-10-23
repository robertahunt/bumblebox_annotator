"""Selective working copies and publication safety, using disposable projects only."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.annotation import AnnotationManager
from core.annotation_policy import default_policy, save_policy
from core.assigned_work import (accept_submission, assignment_status, download_plan, download_work,
    latest_snapshot, my_projects, parse_frames, publish_work, read_assignments, save_assignments,
    workspace, establish_director)
from core.contributors import new_session
from core.import_review import require_reviewed, validate_coco_reviews
from core.project_sync import (_read_json, _write_json, WORK_STATE, SyncConflictError,
    restore_project, snapshot_file, sync_project, test_destination as probe_destination)
from core.project_update import PreparedUpdate, PublishedProject
from scripts.test_project_sync import annotation


class AssignedWorkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.remote = self.base / 'full-project', self.base / 'shared'
        self.remote.mkdir()
        self.configuration = probe_destination(self.remote)
        self.configuration['include_videos'] = True
        self.configuration['rollback_root'] = str(self.base / 'rollbacks')
        (self.base / 'rollbacks').mkdir()
        self.coordinator, self.alex, self.sam = (new_session(n) for n in ('August', 'Alex', 'Sam'))
        (self.source / 'annotations').mkdir(parents=True)
        _write_json(self.source / 'annotations/project.json', {'name': 'test', 'annotation_policy': default_policy({})})
        manager = AnnotationManager()
        manager.contributor_session = self.coordinator
        for video in ('alpha', 'beta'):
            for index in range(3):
                frame = self.source / f'frames/{video}/frame_{index:06d}.jpg'
                frame.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(frame), np.full((32, 40, 3), 80 + index, np.uint8))
                manager.save_frame_annotations(self.source, video, index, [annotation(index + 1)])
            manager.save_video_annotations(self.source, video, [annotation(5, 'hive')])
            _write_json(self.source / f'frames/{video}/video_metadata.json',
                        {'split': 'train', 'selected_frames': [0, 1, 2], 'max_mask_id': 6})
            original = self.source / f'input_data/train/{video}.mp4'
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_bytes(b'original video')
        self.initial = sync_project(self.source, self.configuration, self.coordinator)
        self.project_id = _read_json(self.source / 'annotations/project_sync.json')['project_id']
        self.choices = {'alpha': {'contributor': self.alex, 'frames': [0]},
                        'beta': {'contributor': self.sam, 'frames': [1, 2]}}
        setup = establish_director(self.source, self.configuration, self.coordinator, None)
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, setup['revision'])

    def view(self):
        return PublishedProject(self.source, self.configuration)

    def download(self, actor=None, videos=None, name='working'):
        actor = actor or self.alex
        destination = self.base / name
        download_work(self.configuration, self.project_id, actor, videos or ['alpha'], destination)
        return destination

    def change(self, project, video='alpha', index=0, actor=None):
        manager = AnnotationManager()
        manager.contributor_session = actor or self.alex
        anns = manager.load_frame_annotations(project, video, index)
        anns[0]['mask'][0, 0] = 255
        manager.save_frame_annotations(project, video, index, anns)

    def status(self, video='alpha'):
        view = self.view()
        return assignment_status(view, read_assignments(view)['videos'][video])

    def test_selective_copy_uses_identity_and_preserves_policy_and_annotations(self):
        self.assertEqual(my_projects(self.configuration, self.alex)[0]['name'], 'full-project')
        self.assertEqual(my_projects(self.configuration, new_session('Alex')), [])
        plan = download_plan(self.configuration, self.project_id, self.alex, ['alpha'])
        self.assertNotIn('frames/alpha/frame_000001.jpg', plan['names'])
        local = self.download()
        self.assertEqual([p.name for p in (local / 'frames/alpha').glob('*.jpg')], ['frame_000000.jpg'])
        self.assertFalse((local / 'input_data').exists())
        self.assertFalse((local / 'frames/beta').exists())
        self.assertTrue((local / 'annotations/json/alpha/frame_000002.json').exists())
        self.assertTrue((local / 'annotations/png/alpha/video_annotations_hive.png').exists())
        self.assertEqual((local / 'annotations/project.json').read_bytes(),
                         (self.source / 'annotations/project.json').read_bytes())
        self.assertEqual(_read_json(local / 'frames/alpha/video_metadata.json')['selected_frames'], [0])
        self.assertEqual(self.status()[0], 'In Progress')
        self.assertEqual(workspace(local)['contributor']['id'], self.alex['id'])

    def test_different_contributors_publish_without_losing_unmaterialized_files(self):
        first, second = self.download(), self.download(self.sam, ['beta'], 'other')
        self.change(first)
        self.change(second, 'beta', 1, self.sam)
        one = publish_work(first, self.configuration, self.alex, ready=['alpha'])
        two = publish_work(second, self.configuration, self.sam, ready=['beta'])
        restored = self.base / 'restored'
        restore_project(two['path'], restored)
        self.assertEqual((restored / 'annotations/json/alpha/frame_000000.json').read_bytes(),
                         (first / 'annotations/json/alpha/frame_000000.json').read_bytes())
        self.assertEqual((restored / 'annotations/json/beta/frame_000001.json').read_bytes(),
                         (second / 'annotations/json/beta/frame_000001.json').read_bytes())
        for path in self.source.glob('frames/*/*.jpg'):
            self.assertEqual(path.read_bytes(), (restored / path.relative_to(self.source)).read_bytes())
        self.assertTrue((restored / 'input_data/train/alpha.mp4').exists())
        self.assertEqual(self.status()[0], 'Ready for Review')
        self.assertEqual(self.status('beta')[0], 'Ready for Review')
        self.assertLess(one['copied_files'], 5)
        self.assertEqual(Path(two['path']).parents[3].name.split('--')[0], 'full-project')

    def test_shared_masks_and_max_id_publish_without_subset_metadata(self):
        local = self.download()
        manager = AnnotationManager()
        manager.contributor_session = self.alex
        manager.save_video_annotations(local, 'alpha', [annotation(10, 'hive')])
        meta_path = local / 'frames/alpha/video_metadata.json'
        meta = _read_json(meta_path)
        meta['max_mask_id'] = 11
        _write_json(meta_path, meta)
        result = publish_work(local, self.configuration, self.alex)
        restored = self.base / 'restored'
        restore_project(result['path'], restored)
        meta = _read_json(restored / 'frames/alpha/video_metadata.json')
        self.assertEqual(meta['selected_frames'], [0, 1, 2])
        self.assertNotIn('project_import', meta)
        self.assertEqual(meta['max_mask_id'], 11)
        anns, _ = manager.load_video_annotations(restored, 'alpha')
        self.assertEqual(anns[0]['mask_id'], 10)

    def test_only_explicit_local_annotation_deletions_propagate(self):
        local = self.download()
        removed = local / 'annotations/json/alpha/frame_000000.json'
        removed.unlink()
        result = publish_work(local, self.configuration, self.alex)
        files = _read_json(Path(result['path']) / 'manifest.json')['files']
        self.assertNotIn('annotations/json/alpha/frame_000000.json', files)
        self.assertIn('annotations/json/alpha/frame_000001.json', files)
        self.assertIn('annotations/json/beta/frame_000000.json', files)

    def test_unassigned_or_global_file_changes_are_rejected(self):
        for index, name in enumerate(('annotations/project.json', 'annotations/json/alpha/frame_000002.json',
                                      'frames/alpha/frame_000000.jpg', 'annotations/json/beta/frame_000000.json')):
            local = self.download(name='copy' + str(index))
            path = local / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
            heads = self.view().heads()
            with self.assertRaisesRegex(ValueError, 'protected file'):
                publish_work(local, self.configuration, self.alex)
            self.assertEqual(self.view().heads(), heads)

    def test_wrong_identity_and_full_project_operations_are_rejected(self):
        local = self.download()
        with self.assertRaisesRegex(ValueError, 'belongs to Alex'):
            publish_work(local, self.configuration, new_session('Alex'))
        with self.assertRaisesRegex(ValueError, 'Upload Assigned Annotations'):
            sync_project(local, self.configuration, self.alex)
        with self.assertRaisesRegex(ValueError, 'Assigned work'):
            PreparedUpdate(local, self.configuration, latest_snapshot(self.view())[0]['reference'])
        for operation in (lambda: save_policy(local, default_policy({})['categories']),
                          lambda: require_reviewed(local, {}), lambda: validate_coco_reviews(local, [])):
            with self.assertRaisesRegex(ValueError, 'full project'):
                operation()
        manager = AnnotationManager()
        manager.contributor_session = self.alex
        with self.assertRaisesRegex(ValueError, 'outside your assigned'):
            manager.save_frame_annotations(local, 'alpha', 1, [])
        with self.assertRaisesRegex(ValueError, 'outside your assigned'):
            manager.save_video_annotations(local, 'beta', [])

    def test_same_video_changes_and_policy_changes_stop_stale_publications(self):
        local = self.download()
        self.change(local)
        self.change(self.source, index=2, actor=self.coordinator)
        sync_project(self.source, self.configuration, self.coordinator)
        with self.assertRaisesRegex(SyncConflictError, 'video alpha'):
            publish_work(local, self.configuration, self.alex)
        newer = self.download(name='newer')
        info = _read_json(self.source / 'annotations/project.json')
        info['description'] = 'Changed intent'
        _write_json(self.source / 'annotations/project.json', info)
        sync_project(self.source, self.configuration, self.coordinator)
        with self.assertRaisesRegex(SyncConflictError, 'metadata changed'):
            publish_work(newer, self.configuration, self.alex)

    def test_reassignment_is_explicit_and_retires_stale_work(self):
        local = self.download()
        old = read_assignments(self.view())
        stable = save_assignments(self.source, self.configuration, self.coordinator, self.choices, old['revision'])
        self.assertEqual(stable['videos']['alpha']['id'], old['videos']['alpha']['id'])
        self.choices['alpha']['contributor'] = self.sam
        with self.assertRaisesRegex(ValueError, 'Refresh'):
            save_assignments(self.source, self.configuration, self.coordinator, self.choices, old['revision'])
        changed = save_assignments(self.source, self.configuration, self.coordinator, self.choices, stable['revision'])
        self.assertNotEqual(changed['videos']['alpha']['id'], old['videos']['alpha']['id'])
        with self.assertRaisesRegex(SyncConflictError, 'Assignment changed'):
            publish_work(local, self.configuration, self.alex)
        with self.assertRaisesRegex(ValueError, 'coordinator'):
            save_assignments(self.source, self.configuration, self.sam, self.choices, changed['revision'])

    def test_acceptance_requires_local_review_and_noop_retains_acceptance(self):
        local = self.download()
        self.change(local)
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        status, revision = self.status()
        self.assertEqual(status, 'Ready for Review')
        with self.assertRaisesRegex(ValueError, 'Review Shared Updates'):
            accept_submission(self.source, self.configuration, self.coordinator, 'alpha', revision)
        PreparedUpdate(self.source, self.configuration, (self.alex['id'], revision)).apply()
        accept_submission(self.source, self.configuration, self.coordinator, 'alpha', revision)
        self.assertEqual(self.status()[0], 'Accepted')
        heads = self.view().heads()
        self.assertTrue(publish_work(local, self.configuration, self.alex)['unchanged'])
        self.assertEqual(self.view().heads(), heads)
        self.assertEqual(self.status()[0], 'Accepted')
        manager = AnnotationManager()
        manager.contributor_session = self.alex
        manager.save_frame_annotations(local, 'alpha', 0, [])
        publish_work(local, self.configuration, self.alex)
        self.assertEqual(self.status()[0], 'In Progress')

    def test_one_contributor_multiple_workspaces_retains_each_submission(self):
        data = read_assignments(self.view())
        self.choices['beta']['contributor'] = self.alex
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        first = self.download()
        second = self.download(self.alex, ['beta'], 'second')
        self.change(first)
        self.change(second, 'beta', 1)
        publish_work(first, self.configuration, self.alex, ready=['alpha'])
        publish_work(second, self.configuration, self.alex, ready=['beta'])
        self.assertEqual(self.status()[0], 'Ready for Review')
        self.assertEqual(self.status('beta')[0], 'Ready for Review')

    def test_cancel_and_disk_full_leave_work_and_shared_heads_intact(self):
        local = self.download()
        self.change(local)
        heads = self.view().heads()
        with self.assertRaises(InterruptedError):
            publish_work(local, self.configuration, self.alex, cancelled=lambda: True)
        with patch('core.assigned_work.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'Insufficient shared'):
                publish_work(local, self.configuration, self.alex)
            with self.assertRaisesRegex(OSError, 'Insufficient local'):
                self.download(name='no-space')
        self.assertEqual(self.view().heads(), heads)
        with self.assertRaises(InterruptedError):
            download_work(self.configuration, self.project_id, self.alex, ['alpha'], self.base / 'cancelled',
                          cancelled=lambda: True)
        self.assertFalse((self.base / 'cancelled').exists())
        self.assertEqual(list(self.base.glob('.assigned-download-*')), [])

    def test_damaged_download_is_not_promoted(self):
        snapshot, _ = latest_snapshot(self.view())
        name = 'frames/alpha/frame_000000.jpg'
        snapshot_file(snapshot['manifest_path'], snapshot['manifest'], name).write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError, 'verification'):
            self.download()
        self.assertFalse((self.base / 'working').exists())

    def test_failed_receipt_is_recovered_even_after_more_local_edits(self):
        local = self.download()
        self.change(local)
        write = _write_json
        def fail_receipt(path, data):
            if Path(path).name == WORK_STATE and 'pending_publication' not in data:
                raise OSError('local receipt temporarily unavailable')
            return write(path, data)
        with patch('core.assigned_work._write_json', side_effect=fail_receipt):
            result = publish_work(local, self.configuration, self.alex, ready=['alpha'])
        self.assertIn('warning', result)
        self.assertIn('pending_publication', workspace(local))
        manager = AnnotationManager()
        manager.contributor_session = self.alex
        manager.save_frame_annotations(local, 'alpha', 0, [])
        publish_work(local, self.configuration, self.alex)
        self.assertNotIn('pending_publication', workspace(local))
        self.assertEqual(self.status()[0], 'In Progress')

    def test_legacy_and_incomplete_policy_downloads_are_rejected(self):
        self.configuration['storage_mode'] = 'full'
        sync_project(self.source, self.configuration, self.coordinator)
        with self.assertRaisesRegex(ValueError, 'deduplicated'):
            self.download()
        self.configuration['storage_mode'] = 'deduplicated'
        _write_json(self.source / 'annotations/project.json', {})
        sync_project(self.source, self.configuration, self.coordinator)
        with self.assertRaisesRegex(ValueError, 'annotation policy'):
            self.download()

    def test_partial_backup_without_original_videos_supports_assignments(self):
        self.configuration['include_videos'] = False
        sync_project(self.source, self.configuration, self.coordinator)
        local = self.download()
        self.change(local)
        result = publish_work(local, self.configuration, self.alex)
        files = _read_json(Path(result['path']) / 'manifest.json')['files']
        self.assertFalse(any(name.startswith('input_data/') for name in files))
        self.assertIn('frames/beta/frame_000002.jpg', files)

    def test_profile_rename_keeps_assignment_identity(self):
        data = read_assignments(self.view())
        self.choices['alpha']['contributor'] = dict(self.alex, name='Alexandra')
        saved = save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        self.assertEqual(saved['videos']['alpha']['id'], data['videos']['alpha']['id'])

    def test_protected_video_metadata_change_does_not_publish(self):
        local = self.download()
        meta_path = local / 'frames/alpha/video_metadata.json'
        meta = _read_json(meta_path)
        meta['split'] = 'val'
        _write_json(meta_path, meta)
        with self.assertRaisesRegex(ValueError, 'cannot change video scope'):
            publish_work(local, self.configuration, self.alex)

    def test_interrupted_latest_pointer_can_retry_without_losing_changes(self):
        local = self.download()
        self.change(local)
        def disconnect(path, data):
            if Path(path).name == 'latest.json':
                raise OSError('network disconnected')
            return _write_json(path, data)
        heads = self.view().heads()
        with patch('core.assigned_work._write_json', side_effect=disconnect):
            with self.assertRaisesRegex(OSError, 'disconnected'):
                publish_work(local, self.configuration, self.alex)
        self.assertEqual(self.view().heads(), heads)
        self.assertIn('pending_publication', workspace(local))
        publish_work(local, self.configuration, self.alex)
        self.assertNotIn('pending_publication', workspace(local))
        self.assertEqual(self.status()[0], 'In Progress')

    def test_assignment_change_mid_download_does_not_promote_stale_work(self):
        changed = []
        def progress(message):
            if not changed:
                changed.append(True)
                data = read_assignments(self.view())
                self.choices['alpha']['contributor'] = self.sam
                save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        with self.assertRaisesRegex(SyncConflictError, 'Assignment changed'):
            download_work(self.configuration, self.project_id, self.alex, ['alpha'], self.base / 'stale', progress)
        self.assertFalse((self.base / 'stale').exists())

    def test_download_never_overwrites_an_existing_working_copy(self):
        local = self.download()
        self.change(local)
        before = (local / 'annotations/json/alpha/frame_000000.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'new local folder'):
            self.download()
        self.assertEqual(before, (local / 'annotations/json/alpha/frame_000000.json').read_bytes())

    def test_frame_range_validation(self):
        self.assertEqual(parse_frames('0,2-4', range(5)), [0, 2, 3, 4])
        self.assertEqual(parse_frames('All', [1, 5]), [1, 5])
        for invalid in ('4-2', 'no', '-1', '1,9'):
            with self.assertRaises(ValueError):
                parse_frames(invalid, range(5))


if __name__ == '__main__':
    unittest.main()
