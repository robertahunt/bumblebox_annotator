"""Controlled updates on disposable two-computer projects, including rollback failures."""

from pathlib import Path
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.contributors import new_session
from core.project_sync import (_inventory, _read_json, _write_json, restore_project,
                               sync_project, update_journal_path, require_no_update)
from core.project_update import PreparedUpdate, published_versions, recover_update
from core.project_delta import rollback_update
from scripts import test_project_sync as fixtures


def contents(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


class ProjectUpdateTests(unittest.TestCase):
    tearDown = fixtures.FolderSyncTests.tearDown
    sync = fixtures.FolderSyncTests.sync

    def setUp(self):
        fixtures.FolderSyncTests.setUp(self)
        self.first = self.sync()
        self.local = self.base / 'other-computer'
        restore_project(self.first['path'], self.local)

    def publish_edit(self, name='annotations/changed.json', data='[]', contributor=None, configuration=None):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return sync_project(self.source, configuration or self.configuration, contributor or self.contributor)

    def plan(self, contributor=None):
        versions = published_versions(self.local, self.configuration)
        selected = next(v for v in versions if v['reference'][0] == (contributor or self.contributor)['id'])
        return PreparedUpdate(self.local, self.configuration, selected['reference'])

    def test_fast_forward_preserves_local_extras_and_only_backs_up_affected_files(self):
        for name in ('models/model.pt', 'runs/output.txt', 'annotations/coco/old.json', 'dataset_info.json'):
            path = self.local / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('local-only')
        before = contents(self.local)
        (self.source / 'annotations/png/video/frame_000000.png').unlink()
        result = self.publish_edit()
        remote_before = contents(self.remote)
        plan = self.plan()
        self.assertFalse(plan.blocked)
        self.assertEqual(plan.changes['annotations/png/video/frame_000000.png'], 'Delete')
        self.assertEqual(plan.changes['annotations/changed.json'], 'Add')
        applied = plan.apply()
        self.assertTrue(applied['updated'])
        stored = contents(Path(applied['backup']) / 'before')
        self.assertEqual(stored, {name: data for name, data in before.items() if name in plan.before})
        self.assertNotIn('models/model.pt', stored)
        self.assertNotIn('frames/video/frame_000000.jpg', stored)
        self.assertEqual(contents(self.remote), remote_before)
        self.assertEqual((self.local / 'models/model.pt').read_text(), 'local-only')
        self.assertEqual((self.local / 'runs/output.txt').read_text(), 'local-only')
        self.assertFalse(any(p.is_file() for p in (self.local / 'annotations/coco').rglob('*')))
        self.assertFalse((self.local / 'dataset_info.json').exists())
        self.assertEqual(set(_inventory(self.local, True)), set(_inventory(Path(result['path']), True)))
        self.assertFalse(update_journal_path(self.local).exists())
        self.assertTrue(sync_project(self.local, self.configuration, self.contributor)['unchanged'])
        rollback_update(self.local, applied['backup'], self.configuration)
        self.assertEqual(contents(self.local), before)

    def test_both_changed_is_blocked_even_when_files_do_not_overlap(self):
        self.publish_edit()
        (self.local / 'annotations/local-other-frame.json').write_text('["local"]')
        before = contents(self.local)
        plan = self.plan()
        self.assertIn('Both local and published', plan.blocked)
        with self.assertRaisesRegex(RuntimeError, 'Both local and published'):
            plan.apply()
        self.assertEqual(contents(self.local), before)

    def test_no_remote_changes_keeps_local_edits(self):
        (self.local / 'annotations/local.json').write_text('local')
        before = contents(self.local)
        plan = self.plan()
        self.assertTrue(plan.local_only)
        self.assertFalse(plan.apply()['updated'])
        self.assertEqual(contents(self.local), before)

    def test_identical_and_convergent_copies(self):
        self.assertTrue(self.plan().noop)
        self.publish_edit()
        (self.local / 'annotations/changed.json').write_text('[]')
        plan = self.plan()
        self.assertFalse(plan.blocked)
        self.assertFalse(plan.noop)
        self.assertTrue(plan.apply()['updated'])

    def test_can_follow_another_contributor_and_multiple_revisions(self):
        alex = new_session('Alex')
        self.publish_edit(contributor=alex)
        self.publish_edit('annotations/second.json', contributor=alex)
        self.assertEqual(len(published_versions(self.local, self.configuration)), 2)
        plan = self.plan(alex)
        self.assertFalse(plan.blocked)
        plan.apply()
        self.assertTrue((self.local / 'annotations/second.json').exists())
        self.assertFalse(sync_project(self.local, self.configuration, self.contributor)['separate_backup'])

    def test_unknown_base_and_wrong_project_are_rejected_without_modification(self):
        from core.project_sync import LOCAL_STATE
        (self.local / 'annotations' / LOCAL_STATE).unlink()
        before = contents(self.local)
        with self.assertRaisesRegex(ValueError, 'no recorded starting'):
            self.plan()
        self.assertEqual(before, contents(self.local))
        _write_json(self.local / 'annotations/project_sync.json', {'project_id': new_session('Other')['id']})
        with self.assertRaisesRegex(ValueError, 'Expected one published project'):
            published_versions(self.local, self.configuration)

    def test_divergent_or_older_publication_cannot_roll_back(self):
        self.publish_edit()
        self.plan().apply()
        latest = next(self.remote.rglob('latest.json'))
        _write_json(latest, {'revision': Path(self.first['path']).parent.name})
        plan = self.plan()
        self.assertIn('older or comes from a different history', plan.blocked)
        with self.assertRaises(RuntimeError):
            plan.apply()

    def test_partial_publication_keeps_local_original_videos(self):
        self.publish_edit(configuration=dict(self.configuration, include_videos=False))
        plan = self.plan()
        self.assertFalse(plan.include_videos)
        self.assertFalse(any(name.startswith('input_data/') for name in plan.changes))
        plan.apply()
        self.assertEqual((self.local / 'input_data/train/video.mp4').read_text(), 'video')

    def test_policy_and_split_changes_are_explicit_in_preview(self):
        _write_json(self.source / 'annotations/project.json', {'hive_annotation_scope': 'frame'})
        self.publish_edit('frames/video/video_metadata.json', '{"split":"val"}')
        plan = self.plan()
        self.assertTrue(any('Hive' in text for text in plan.policy_changes))
        self.assertIn('frames/video/video_metadata.json', plan.split_changes)

    def test_local_or_remote_changes_after_preview_require_new_preview(self):
        self.publish_edit()
        plan = self.plan()
        (self.local / 'annotations/unexpected.json').write_text('late change')
        with self.assertRaisesRegex(ValueError, 'Local files changed'):
            plan.apply()
        (self.local / 'annotations/unexpected.json').unlink()
        plan = self.plan()
        self.publish_edit('annotations/later.json')
        with self.assertRaisesRegex(ValueError, 'Published versions changed'):
            plan.apply()

    def test_corrupted_or_missing_remote_file_never_replaces_local_copy(self):
        result = self.publish_edit()
        plan = self.plan()
        before = contents(self.local)
        path = Path(result['path']) / 'annotations/changed.json'
        path.write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'failed verification'):
            plan.apply()
        self.assertEqual(contents(self.local), before)
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            plan.apply()
        self.assertEqual(contents(self.local), before)
        self.assertFalse(update_journal_path(self.local).exists())

    def test_cancellation_and_disk_failure_leave_original_unchanged(self):
        self.publish_edit()
        plan = self.plan()
        before = contents(self.local)
        with self.assertRaises(InterruptedError):
            plan.apply(cancelled=lambda: True)
        cancel = [False]
        with self.assertRaises(InterruptedError):
            plan.apply(progress=lambda text: cancel.__setitem__(0, True), cancelled=lambda: cancel[0])
        with patch('core.project_delta.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'Insufficient'):
                plan.apply()
        with patch('core.project_delta._clone_or_copy', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                plan.apply()
        self.assertEqual(contents(self.local), before)
        self.assertFalse(update_journal_path(self.local).exists())
        self.assertFalse(list(self.base.glob('.bumblebox-update-stage-*')))

    def test_promotion_failure_restores_original(self):
        self.publish_edit()
        plan = self.plan()
        before = contents(self.local)
        replace = os.replace
        def fail_stage(path, target):
            if Path(target) == self.local / 'annotations/changed.json':
                raise OSError('promotion failed')
            return replace(path, target)
        with patch('core.project_delta.os.replace', fail_stage):
            with self.assertRaisesRegex(OSError, 'promotion failed'):
                plan.apply()
        self.assertEqual(contents(self.local), before)
        self.assertFalse(update_journal_path(self.local).exists())

    def test_failed_rollback_retains_original_and_is_recoverable(self):
        self.publish_edit()
        plan = self.plan()
        before = contents(self.local)
        replace = os.replace
        failed = [False]
        def fail_after_first(path, target):
            if Path(target) == self.local / 'annotations/changed.json':
                failed[0] = True
            if failed[0] and self.local in Path(target).parents:
                raise OSError('temporary failure')
            return replace(path, target)
        with patch('core.project_delta.os.replace', fail_after_first):
            with self.assertRaisesRegex(RuntimeError, 'old affected files are safe'):
                plan.apply()
        self.assertTrue(self.local.exists())
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            require_no_update(self.local)
        recovered = recover_update(self.local)
        self.assertIn('original local files were restored', recovered['message'])
        self.assertEqual(contents(self.local), before)

    def test_completed_update_with_leftover_journal_is_verified_on_recovery(self):
        self.publish_edit()
        plan = self.plan()
        journal = update_journal_path(self.local)
        unlink = Path.unlink
        def fail_journal(path, *args, **kwargs):
            if path == journal:
                raise OSError('busy')
            return unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', fail_journal):
            applied = plan.apply()
        self.assertIn('warning', applied)
        self.assertTrue(Path(applied['backup']).exists())
        self.assertIn('completed update was verified', recover_update(self.local)['message'])
        self.assertFalse(journal.exists())

    def test_existing_update_journal_blocks_preparation_and_publish(self):
        update_journal_path(self.local).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            self.plan()
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            sync_project(self.local, self.configuration, self.contributor)

    def test_links_in_unaffected_outputs_are_ignored_but_remote_data_links_are_rejected(self):
        result = self.publish_edit()
        link = self.local / 'model-link'
        link.symlink_to(self.source, target_is_directory=True)
        self.assertFalse(self.plan().blocked)
        link.unlink()
        plan = self.plan()
        remote_file = Path(result['path']) / 'annotations/changed.json'
        remote_file.unlink()
        remote_file.symlink_to(self.source / 'annotations/changed.json')
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            plan.apply()
        self.assertFalse(update_journal_path(self.local).exists())

    def test_local_changes_during_staging_abort_before_replacement(self):
        self.publish_edit()
        plan = self.plan()
        before = contents(self.local)
        external = self.local / 'annotations/external-edit.json'
        def edit(message):
            external.write_text('do not lose this')
        with self.assertRaisesRegex(ValueError, 'Local files changed'):
            plan.apply(progress=edit)
        before['annotations/external-edit.json'] = external.read_bytes()
        self.assertEqual(contents(self.local), before)
        self.assertFalse(update_journal_path(self.local).exists())

    def test_published_manifest_changes_require_new_preview(self):
        result = self.publish_edit()
        plan = self.plan()
        before = contents(self.local)
        manifest = Path(result['path']).parent / 'manifest.json'
        data = _read_json(manifest)
        data['project_name'] = 'Unexpected rewrite'
        _write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, 'published manifest changed'):
            plan.apply()
        self.assertEqual(contents(self.local), before)

    def test_recovery_stops_if_installed_copy_was_edited(self):
        self.publish_edit()
        plan = self.plan()
        journal = update_journal_path(self.local)
        unlink = Path.unlink
        def leave_marker(path, *args, **kwargs):
            if path == journal:
                raise OSError('busy')
            return unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', leave_marker):
            applied = plan.apply()
        (self.local / 'annotations/changed.json').write_text('an edit after interruption')
        before = contents(self.local)
        backup_before = contents(Path(applied['backup']))
        with self.assertRaisesRegex(ValueError, 'changed after interruption'):
            recover_update(self.local)
        self.assertEqual(before, contents(self.local))
        self.assertEqual(backup_before, contents(Path(applied['backup'])))
        self.assertTrue(journal.exists())


if __name__ == '__main__':
    unittest.main()
