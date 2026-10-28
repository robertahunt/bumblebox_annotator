"""Persistent assignment downloads, using small disposable projects only."""

from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from core.assigned_work import publish_work, read_assignments, save_assignments, workspace
from core.assigned_workspace import PreparedAssignedDownload, existing_copies, managed_directory
from core.project_delta import file_record, rollback_update
from core.project_sync import (_inventory, _read_json, _write_json, SyncConflictError,
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
        self.assertEqual(len(list((local / 'frames/beta').glob('*.jpg'))), 2)
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

    def test_range_change_clean_adds_frames_dirty_requires_review(self):
        local = self.download()
        choices = deepcopy(self.choices)
        choices['alpha']['frames'] = [0, 1]
        self.assign(choices)
        self.prepare(local).apply()
        self.assertTrue((local / 'frames/alpha/frame_000001.jpg').exists())
        self.change(local)
        choices['alpha']['frames'] = [0, 1, 2]
        self.assign(choices)
        with self.assertRaisesRegex(SyncConflictError, 'assignment changed'):
            self.prepare(local)

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
        data['name'] = 'Changed settings'
        _write_json(path, data)
        sync_project(self.source, self.configuration, self.coordinator)
        with self.assertRaisesRegex(SyncConflictError, 'policy'):
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
