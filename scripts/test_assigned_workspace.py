"""Persistent assignment downloads, using small disposable projects only."""

from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from core.assigned_work import publish_work, read_assignments, save_assignments, workspace
from core.assigned_workspace import (PreparedAssignedDownload, _require_edits_keep_their_place, existing_copies,
                                     managed_directory)
from core.project_activity import project_activity
from core.project_delta import file_record, rollback_update
from core.project_sync import (WORK_STATE, _inventory, _read_json, _write_json, SyncConflictError,
                               snapshot_file, sync_project, update_journal_path)
from scripts import test_assigned_work as fixtures


class AssignedWorkspaceTests(unittest.TestCase):
    setUp = fixtures.AssignedWorkTests.setUp
    view = fixtures.AssignedWorkTests.view
    download = fixtures.AssignedWorkTests.download
    change = fixtures.AssignedWorkTests.change
    status = fixtures.AssignedWorkTests.status

    def prepare(self, local, videos=('alpha',), **kwargs):
        return PreparedAssignedDownload(self.configuration, self.project_id, self.alex,
                                        videos, local, **kwargs)

    def assign(self, choices):
        data = read_assignments(self.view())
        save_assignments(self.source, self.configuration, self.coordinator, choices, data['revision'])

    def records(self, local):
        return {n: file_record(local / n) for n in _inventory(local, True)}

    def annotation_records(self, local):
        return {n: r for n, r in self.records(local).items() if n.startswith('annotations/')}

    def frame_images(self, local, video='alpha'):
        return sorted(int(p.stem.split('_')[1]) for p in (local / 'frames' / video).glob('frame_*.jpg'))

    def as_older_copy(self, local, video='alpha'):
        """Strip a copy to what earlier versions downloaded: only the assigned frame images."""
        state_path = local / 'annotations' / WORK_STATE
        state = _read_json(state_path)
        for frame in self.frame_images(local, video):
            if frame not in state['assignments'][video]['frames']:
                name = f'frames/{video}/frame_{frame:06d}.jpg'
                (local / name).unlink()
                del state['local_files'][name]
        _write_json(state_path, state)

    def assign_frames(self, frames, video='alpha', **changes):
        choices = deepcopy(self.choices)
        choices[video].update(frames=frames, contributor=self.alex, **changes)
        self.assign(choices)

    def test_managed_location_is_stable_and_isolates_identities(self):
        with patch.dict(os.environ, {'XDG_DATA_HOME': str(self.base / 'appdata')}):
            first = managed_directory(self.configuration, self.project_id, self.alex)
            self.assertIn(self.base / 'appdata', first.parents)
            self.assertEqual(first, managed_directory(self.configuration, self.project_id,
                                                      dict(self.alex, name='Renamed')))
            self.assertNotEqual(first, managed_directory(self.configuration, self.project_id, self.sam))
            self.prepare(first).apply()
            self.assertFalse((first / 'input_data').exists())
            self.assertEqual(self.prepare(first).file_count, 0)

    def test_repeat_download_is_noop_and_preserves_unsent_edits(self):
        local = self.download()
        self.change(local)
        before = self.records(local)
        plan = self.prepare(local)
        self.assertEqual(plan.bytes, 0)
        self.assertEqual(plan.kept_edits, ['alpha'])
        plan.apply()  # Migrate the v1 record without copying any annotation/media files.
        again = self.prepare(local)
        self.assertFalse(again.apply()['updated'])
        self.assertEqual(before, self.records(local))
        self.assertTrue(publish_work(local, self.configuration, self.alex)['copied_files'])

    def test_new_assignments_extend_copy_preserving_old_edits_and_publish(self):
        local = self.download()
        self.change(local)
        before = self.records(local)
        choices = deepcopy(self.choices)
        choices['beta']['contributor'] = self.alex
        self.assign(choices)
        plan = self.prepare(local, ('beta',))
        self.assertEqual(plan.new_videos, ['beta'])
        self.assertEqual(plan.kept_edits, ['alpha'])
        plan.apply()
        after = self.records(local)
        self.assertTrue(all(after[n] == r for n, r in before.items()))
        self.assertEqual(set(workspace(local)['assignments']), {'alpha', 'beta'})
        self.assertEqual(self.frame_images(local, 'beta'), [0, 1, 2])  # the whole video, for viewing
        self.assertFalse((local / 'input_data').exists())
        self.assertEqual(self.status('beta')[0], 'In Progress')
        self.change(local, 'beta', 1)
        self.assertTrue(publish_work(local, self.configuration, self.alex)['copied_files'])

    def test_clean_video_gets_remote_edits_and_can_rollback(self):
        local = self.download()
        original = self.records(local)
        self.change(self.source, actor=self.coordinator)
        sync_project(self.source, self.configuration, self.coordinator)
        plan = self.prepare(local)
        self.assertEqual(plan.updated_videos, ['alpha'])
        self.assertLess(plan.bytes, sum(r['size'] for r in original.values()))
        result = plan.apply()
        name = 'annotations/json/alpha/frame_000000.json'
        self.assertEqual((local / name).read_bytes(), (self.source / name).read_bytes())
        rollback_update(local, result['backup'], self.configuration)
        self.assertEqual(original, self.records(local))
        self.assertEqual(workspace(local)['format'], 1)

    def test_same_video_conflict_blocks_entire_update(self):
        local = self.download()
        self.change(local)
        self.change(self.source, index=1, actor=self.coordinator)
        sync_project(self.source, self.configuration, self.coordinator)
        before = self.records(local)
        with self.assertRaisesRegex(SyncConflictError, 'Both.*alpha'):
            self.prepare(local)
        self.assertEqual(before, self.records(local))

    def test_whole_video_is_downloaded_and_only_assigned_frames_can_be_saved_or_uploaded(self):
        local = self.download()
        self.assertEqual(self.frame_images(local), [0, 1, 2])
        self.assertEqual(_read_json(local / 'frames/alpha/video_metadata.json')['selected_frames'], [0])
        before = self.records(local)
        with self.assertRaisesRegex(ValueError, 'outside your assigned work'):
            self.change(local, index=1)
        self.assertEqual(before, self.records(local))
        self.change(local)
        self.assertEqual(publish_work(local, self.configuration, self.alex)['changed_videos'], ['alpha'])

    def test_older_copy_gets_the_rest_of_the_video_and_keeps_unsent_edits(self):
        local = self.download()
        self.as_older_copy(local)
        self.assertEqual(self.frame_images(local), [0])
        self.change(local)
        edits = self.annotation_records(local)
        plan = self.prepare(local)
        self.assertEqual(plan.kept_edits, ['alpha'])
        plan.apply()
        self.assertEqual(self.frame_images(local), [0, 1, 2])
        self.assertEqual(edits, self.annotation_records(local))
        self.assertFalse(self.prepare(local).apply()['updated'])
        self.assertEqual(publish_work(local, self.configuration, self.alex)['changed_videos'], ['alpha'])

    def test_range_change_on_a_clean_copy_is_taken(self):
        local = self.download()
        self.assign_frames([0, 1])
        self.prepare(local).apply()
        self.assertEqual(workspace(local)['assignments']['alpha']['frames'], [0, 1])
        self.assertEqual(_read_json(local / 'frames/alpha/video_metadata.json')['selected_frames'], [0, 1])

    def test_frames_added_over_unsent_work_are_accepted(self):
        # A student is partway through frame 0 when the director assigns the whole video.
        local = self.download()
        self.as_older_copy(local)
        self.change(local)
        path = local / 'frames/alpha/video_metadata.json'
        _write_json(path, dict(_read_json(path), max_mask_id=50))  # new masks raised the instance counter
        edits = self.annotation_records(local)
        self.assign_frames([0, 1, 2])
        plan = self.prepare(local)
        self.assertEqual(plan.kept_edits, ['alpha'])
        plan.apply()
        state = workspace(local)
        published = read_assignments(self.view())['videos']['alpha']
        self.assertEqual((state['assignments']['alpha']['id'], state['assignments']['alpha']['frames']),
                         (published['id'], [0, 1, 2]))
        self.assertIn('started_at', published)  # the director sees the new assignment was picked up
        self.assertEqual(self.frame_images(local), [0, 1, 2])
        self.assertEqual(edits, self.annotation_records(local))
        self.assertEqual(_read_json(path)['selected_frames'], [0, 1, 2])
        self.assertEqual(_read_json(path)['max_mask_id'], 50)
        self.assertNotEqual(state['local_files']['frames/alpha/video_metadata.json'], file_record(path))
        upload = publish_work(local, self.configuration, self.alex)
        self.assertEqual(upload['changed_videos'], ['alpha'])
        manifest = Path(upload['path']) / 'manifest.json'
        remote = snapshot_file(manifest, _read_json(manifest), 'frames/alpha/video_metadata.json')
        self.assertEqual(_read_json(remote)['max_mask_id'], 50)
        self.change(local, index=2)  # a newly assigned frame can be worked on and uploaded
        self.assertEqual(publish_work(local, self.configuration, self.alex)['changed_videos'], ['alpha'])
        self.assertFalse(self.prepare(local).apply()['updated'])

    def test_a_frame_with_unsent_work_cannot_be_taken_away(self):
        local = self.download()
        self.assign_frames([0, 1, 2])
        self.prepare(local).apply()
        self.change(local, index=1)
        before = self.records(local)
        self.assign_frames([0, 2])
        with self.assertRaisesRegex(SyncConflictError, 'no longer includes frame 1, which has unpublished work'):
            self.prepare(local)
        self.assertEqual(before, self.records(local))
        # Assigning that frame again lets the copy carry on, under a different assignment than it had.
        self.assign_frames([0, 1])
        self.prepare(local).apply()
        self.assertEqual(workspace(local)['assignments']['alpha']['frames'], [0, 1])
        self.assertEqual(publish_work(local, self.configuration, self.alex)['changed_videos'], ['alpha'])

    def test_a_frame_without_unsent_work_can_be_taken_away(self):
        local = self.download()
        self.assign_frames([0, 1, 2])
        self.prepare(local).apply()
        self.change(local)
        edits = self.annotation_records(local)
        self.assign_frames([0])
        self.prepare(local).apply()
        self.assertEqual(workspace(local)['assignments']['alpha']['frames'], [0])
        self.assertEqual(edits, self.annotation_records(local))
        self.assertEqual(self.frame_images(local), [0, 1, 2])  # the video stays viewable
        self.assertEqual(publish_work(local, self.configuration, self.alex)['changed_videos'], ['alpha'])

    def test_task_change_over_unsent_work_still_requires_review(self):
        local = self.download()
        self.change(local)
        before = self.records(local)
        self.assign_frames([0, 1, 2], task='bee')
        with self.assertRaisesRegex(SyncConflictError, 'mask task for alpha changed'):
            self.prepare(local)
        self.assertEqual(before, self.records(local))

    def test_which_unsent_files_block_a_changed_assignment(self):
        old, new = {'task': None, 'frames': [0, 1]}, {'task': None, 'frames': [5]}
        whole_video = ['annotations/json/alpha/video_annotations.json',
                       'annotations/png/alpha/video_annotations_hive.png',
                       'annotations/work/alpha/video_work.json', 'frames/alpha/video_metadata.json']
        _require_edits_keep_their_place('alpha', old, new, whole_video)  # shared masks belong to no frame
        _require_edits_keep_their_place('alpha', old, new, ['annotations/json/alpha/frame_000005.json'])
        with self.assertRaisesRegex(SyncConflictError, 'no longer includes frame 0, 1,'):
            _require_edits_keep_their_place('alpha', old, new, ['annotations/json/alpha/frame_000001.json',
                                                                'annotations/png/alpha/frame_000000_bee.png'])
        with self.assertRaisesRegex(SyncConflictError, 'Both the local work'):
            _require_edits_keep_their_place('alpha', old, new, ['input_data/train/alpha.mp4'])

    def test_revoked_work_is_retained_but_never_published(self):
        local = self.download()
        self.change(local)
        original = self.records(local)
        choices = deepcopy(self.choices)
        choices['alpha']['contributor'] = self.sam
        choices['beta']['contributor'] = self.alex
        self.assign(choices)
        self.prepare(local, ('beta',)).apply()
        state = workspace(local)
        self.assertEqual(set(state['assignments']), {'beta'})
        self.assertEqual(set(state['retained_assignments']), {'alpha'})
        self.assertTrue(all(self.records(local)[n] == r for n, r in original.items()))
        self.change(local, 'beta', 1)
        publication = publish_work(local, self.configuration, self.alex)
        files = _read_json(Path(publication['path']) / 'manifest.json')['files']
        name = 'annotations/json/alpha/frame_000000.json'
        self.assertEqual(files[name], file_record(self.source / name))
        self.assertNotEqual(files[name], file_record(local / name))

    def test_policy_change_requires_review(self):
        local = self.download()
        path = self.source / 'annotations/project.json'
        data = _read_json(path)
        data['annotation_policy']['categories']['bee']['meaning'] = 'full_extent'
        _write_json(path, data)
        sync_project(self.source, self.configuration, self.coordinator)
        with self.assertRaisesRegex(SyncConflictError, 'policy'):
            self.prepare(local)
        self.assertEqual(project_activity(local, self.configuration)['status'], 'attention')

    def test_keypoint_settings_reach_an_existing_copy_without_review(self):
        """Settings that are not annotation policy must not strand outstanding work."""
        from core import keypoints as kp
        name = 'annotations/project.json'
        local = self.download()
        self.change(local)  # Unsent edits made before the settings change.
        self.assertIsNone(kp.read_project_config(local))
        kp.save_project_config(self.source, kp.default_config())
        sync_project(self.source, self.configuration, self.coordinator)

        activity = project_activity(local, self.configuration)
        self.assertEqual(activity['status'], 'current')
        self.assertIn('settings', activity['message'])
        published = publish_work(local, self.configuration, self.alex)  # upload is not blocked
        files = _read_json(Path(published['path']) / 'manifest.json')['files']
        self.assertEqual(files[name], file_record(self.source / name))  # and keeps the new settings

        plan = self.prepare(local)
        self.assertEqual((plan.new_videos, plan.updated_videos), ([], []))
        plan.apply()
        self.assertEqual((local / name).read_bytes(), (self.source / name).read_bytes())
        self.assertEqual(kp.read_project_config(local).names, kp.DEFAULT_KEYPOINT_NAMES)
        self.assertEqual(workspace(local)['local_files'][name], file_record(local / name))
        self.assertFalse(self.prepare(local).apply()['updated'])
        self.assertNotIn('settings', project_activity(local, self.configuration)['message'])

        # Contributors still may not change project settings themselves.
        data = _read_json(local / name)
        data['keypoints']['enabled'] = False
        _write_json(local / name, data)
        with self.assertRaisesRegex(SyncConflictError, 'Local project settings'):
            self.prepare(local)

    def test_new_imaging_setup_entries_do_not_block_upload_or_download(self):
        """Adding videos updates imaging_setups.json; outstanding work must keep flowing."""
        name = 'annotations/imaging_setups.json'
        local = self.download()
        self.change(local)  # Unsent edits, like a contributor partway through a round.
        setups = {'version': 1, 'setups': {'rig': {'pixels_per_cm': 10.0}},
                  'video_assignments': {'alpha': 'rig', 'beta': 'rig'}}
        _write_json(self.source / name, setups)
        sync_project(self.source, self.configuration, self.coordinator)
        choices = deepcopy(self.choices)
        choices['beta']['contributor'] = self.alex
        self.assign(choices)

        self.assertNotEqual(project_activity(local, self.configuration)['status'], 'attention')
        published = publish_work(local, self.configuration, self.alex)
        self.assertTrue(published['copied_files'])
        files = _read_json(Path(published['path']) / 'manifest.json')['files']
        self.assertEqual(files[name], file_record(self.source / name))

        plan = self.prepare(local, ('beta',))
        self.assertEqual(plan.new_videos, ['beta'])
        plan.apply()
        self.assertEqual((local / name).read_bytes(), (self.source / name).read_bytes())
        self.assertEqual(workspace(local)['local_files'][name], file_record(local / name))
        self.assertFalse(self.prepare(local).apply()['updated'])
        self.change(local, 'beta', 1)
        self.assertTrue(publish_work(local, self.configuration, self.alex)['copied_files'])

        # Contributors still may not edit it themselves.
        _write_json(local / name, dict(setups, video_assignments={}))
        with self.assertRaisesRegex(SyncConflictError, 'Local project settings'):
            self.prepare(local)

    def test_local_change_after_preview_rejected(self):
        local = self.download()
        plan = self.prepare(local)
        self.change(local)
        before = self.records(local)
        with self.assertRaisesRegex(ValueError, 'Local files changed'):
            plan.apply()
        self.assertEqual(before, self.records(local))

    def test_assignment_change_after_preview_rejected(self):
        local = self.download()
        plan = self.prepare(local)
        choices = deepcopy(self.choices)
        choices['alpha']['frames'] = [0, 1]
        self.assign(choices)
        with self.assertRaises((ValueError, SyncConflictError)):
            plan.apply()

    def test_damaged_remote_file_does_not_replace_local_work(self):
        local = self.download()
        self.change(self.source, actor=self.coordinator)
        sync_project(self.source, self.configuration, self.coordinator)
        plan = self.prepare(local)
        snap = plan.plan['snapshot']
        source = snapshot_file(snap['manifest_path'], snap['manifest'],
                               'annotations/json/alpha/frame_000000.json')
        source.write_bytes(b'damaged test object')
        before = self.records(local)
        with self.assertRaisesRegex(ValueError, 'verification'):
            plan.apply()
        self.assertEqual(before, self.records(local))
        self.assertFalse(update_journal_path(local).exists())

    def test_local_metadata_counter_stays_unpublished_after_adding_work(self):
        local = self.download()
        path = local / 'frames/alpha/video_metadata.json'
        meta = _read_json(path)
        meta['max_mask_id'] = 50
        _write_json(path, meta)
        choices = deepcopy(self.choices)
        choices['beta']['contributor'] = self.alex
        self.assign(choices)
        self.prepare(local, ('beta',)).apply()
        self.assertEqual(_read_json(path)['max_mask_id'], 50)
        self.assertNotEqual(workspace(local)['local_files']['frames/alpha/video_metadata.json'], file_record(path))
        published = publish_work(local, self.configuration, self.alex)
        manifest = Path(published['path']) / 'manifest.json'
        remote = snapshot_file(manifest, _read_json(manifest), 'frames/alpha/video_metadata.json')
        self.assertEqual(_read_json(remote)['max_mask_id'], 50)
        self.assertEqual(_read_json(remote)['selected_frames'], [0, 1, 2])

    def test_cancel_and_failed_apply_leave_old_copy_intact(self):
        local = self.download()
        self.change(self.source, actor=self.coordinator)
        sync_project(self.source, self.configuration, self.coordinator)
        original = self.records(local)
        plan = self.prepare(local)
        with self.assertRaises(InterruptedError):
            plan.apply(cancelled=lambda: True)
        self.assertEqual(original, self.records(local))
        self.assertFalse(update_journal_path(local).exists())
        replace = os.replace
        failed = False

        def fail_once(src, dst):
            nonlocal failed
            if not failed and '.bumblebox-update-stage-' in str(src) and str(dst).endswith('frame_000000.json'):
                failed = True
                raise OSError('simulated disk failure')
            return replace(src, dst)

        with patch('core.project_delta.os.replace', side_effect=fail_once):
            with self.assertRaisesRegex(OSError, 'simulated'):
                plan.apply()
        self.assertTrue(failed)
        self.assertEqual(original, self.records(local))
        self.assertEqual(workspace(local)['format'], 1)
        self.assertFalse(update_journal_path(local).exists())

    def test_legacy_discovery_does_not_pick_newest_or_mix_people(self):
        first, second = self.download(), self.download(name='working-2')
        other = self.download(self.sam, ['beta'], 'sam')
        self.assertEqual(existing_copies([first, second, first, other, self.source], self.configuration,
                                        self.project_id, self.alex), [first, second])

    def test_wrong_identity_or_shared_destination_rejected(self):
        local = self.download(self.sam, ['beta'])
        with self.assertRaisesRegex(ValueError, 'belongs to'):
            self.prepare(local)
        with self.assertRaises(ValueError):
            self.prepare(self.remote / 'local')


if __name__ == '__main__':
    unittest.main()
