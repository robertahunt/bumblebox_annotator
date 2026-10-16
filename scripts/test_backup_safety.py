"""Archive opening, verified restores, and stale-copy publication safeguards."""

from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from core.contributors import new_session
from core.project_sync import (LOCAL_STATE, SyncConflictError, _read_json, _write_json,
                              backup_location, restore_project, sync_project)
from scripts import test_project_sync as fixtures


class BackupSafetyTests(unittest.TestCase):
    setUp = fixtures.FolderSyncTests.setUp
    tearDown = fixtures.FolderSyncTests.tearDown
    sync = fixtures.FolderSyncTests.sync

    def other_copy(self, path):
        other = self.base / 'other'
        restore_project(path, other)
        return other

    def test_restore_preserves_policy_and_identity_without_modifying_archive(self):
        policy = {'name': 'Example', 'annotation_policy': {'version': 1, 'categories': {}}}
        _write_json(self.source / 'annotations/project.json', policy)
        published = Path(self.sync()['path'])
        before = {p: p.read_bytes() for p in self.remote.rglob('*') if p.is_file()}
        other = self.other_copy(published)
        self.assertEqual(_read_json(other / 'annotations/project.json'), policy)
        self.assertEqual(_read_json(other / 'annotations/project_sync.json'),
                         _read_json(published / 'annotations/project_sync.json'))
        self.assertEqual(before, {p: p.read_bytes() for p in self.remote.rglob('*') if p.is_file()})
        (other / 'frames/video/frame_000000.jpg').write_text('edited')
        self.assertEqual((published / 'frames/video/frame_000000.jpg').read_text(), 'image')
        self.assertIsNone(backup_location(other))

    def test_stale_same_contributor_cannot_replace_latest(self):
        first = self.sync()
        other = self.other_copy(first['path'])
        relative = 'annotations/json/video/frame_000000.json'
        (self.source / relative).write_text('["desktop"]')
        current = self.sync()
        (other / relative).write_text('["laptop"]')
        latest = next(self.remote.rglob('latest.json'))
        before = latest.read_bytes()
        ancestry = (other / 'annotations' / LOCAL_STATE).read_bytes()
        with self.assertRaises(SyncConflictError):
            sync_project(other, self.configuration, self.contributor)
        self.assertEqual(latest.read_bytes(), before)
        separate = sync_project(other, self.configuration, self.contributor, separate_backup=True)
        self.assertTrue(separate['separate_backup'])
        self.assertEqual(latest.read_bytes(), before)
        self.assertEqual((other / 'annotations' / LOCAL_STATE).read_bytes(), ancestry)
        self.assertEqual((Path(current['path']) / relative).read_text(), '["desktop"]')
        self.assertEqual((Path(separate['path']) / relative).read_text(), '["laptop"]')
        restored = self.base / 'separate-copy'
        restore_project(separate['path'], restored)
        with self.assertRaises(SyncConflictError):
            sync_project(restored, self.configuration, self.contributor)

    def test_behind_but_unedited_copy_cannot_roll_back_latest(self):
        other = self.other_copy(self.sync()['path'])
        (self.source / 'annotations/new.json').write_text('new')
        self.sync()
        with patch('core.project_sync._digest', side_effect=AssertionError('Do not hash large videos before warning')):
            with self.assertRaises(SyncConflictError):
                sync_project(other, self.configuration, self.contributor)

    def test_different_contributor_publication_is_detected(self):
        other = self.other_copy(self.sync()['path'])
        alex = new_session('Alex')
        (other / 'annotations/new.json').write_text('alex edit')
        result = sync_project(other, self.configuration, alex)
        self.assertFalse(result['separate_backup'])
        with self.assertRaises(SyncConflictError):
            self.sync()
        # A verified copy of Alex's snapshot carries the common ancestry forward.
        restored = self.base / 'combined'
        restore_project(result['path'], restored)
        self.assertFalse(sync_project(restored, self.configuration, self.contributor)['separate_backup'])

    def test_legacy_unchanged_copy_bootstraps_but_changed_unknown_copy_requires_review(self):
        self.sync()
        state = self.source / 'annotations' / LOCAL_STATE
        state.unlink()
        self.assertTrue(self.sync()['unchanged'])
        self.assertTrue(state.exists())
        state.unlink()
        (self.source / 'annotations/new.json').write_text('unknown base')
        with self.assertRaises(SyncConflictError):
            self.sync()
        self.assertFalse(state.exists())

    def test_ancestry_is_local_only_and_is_not_an_annotation_change(self):
        first = self.sync()
        project = Path(first['path'])
        manifest = _read_json(project.parent / 'manifest.json')
        self.assertNotIn('annotations/' + LOCAL_STATE, manifest['files'])
        self.assertFalse((project / 'annotations' / LOCAL_STATE).exists())
        self.assertTrue(self.sync()['unchanged'])

    def test_old_manifest_restore_retains_known_parent(self):
        published = Path(self.sync()['path'])
        manifest_path = published.parent / 'manifest.json'
        manifest = _read_json(manifest_path)
        for key in ('base_heads', 'root_id', 'publication'):
            manifest.pop(key)
        manifest['format'] = 1
        _write_json(manifest_path, manifest)
        other = self.other_copy(published)
        self.assertTrue(sync_project(other, self.configuration, self.contributor)['unchanged'])

    def test_cancel_or_damaged_backup_never_creates_working_project(self):
        published = Path(self.sync()['path'])
        target = self.base / 'copy'
        with self.assertRaises(InterruptedError):
            restore_project(published, target, cancelled=lambda: True)
        self.assertFalse(target.exists())
        with patch('core.project_sync._clone_or_copy', side_effect=OSError('full')):
            with self.assertRaises(OSError):
                restore_project(published, target)
        self.assertFalse(target.exists())
        (published / 'annotations/png/video/frame_000000.png').write_text('damaged')
        with self.assertRaisesRegex(ValueError, 'damaged'):
            restore_project(published, target)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.base.glob('.restore-*')))

    def test_existing_destination_and_archive_destinations_are_refused(self):
        published = Path(self.sync()['path'])
        with self.assertRaisesRegex(ValueError, 'existing folders'):
            restore_project(published, self.source)
        with self.assertRaisesRegex(ValueError, 'local working copy'):
            restore_project(published, self.remote / 'accidental-working-copy')
        self.assertFalse((self.remote / 'accidental-working-copy').exists())

    def test_archive_containers_pending_and_symlink_aliases_are_recognized(self):
        published = Path(self.sync()['path'])
        self.assertEqual(backup_location(self.remote)['kind'], 'container')
        self.assertEqual(backup_location(published)['kind'], 'published')
        self.assertEqual(backup_location(published.parent)['kind'], 'published')
        pending = published.parent.parent / '.pending-incomplete' / 'project'
        pending.mkdir(parents=True)
        self.assertEqual(backup_location(pending)['kind'], 'pending')
        with self.assertRaisesRegex(ValueError, 'completed backup'):
            self.other_copy(pending)
        alias = self.base / 'alias'
        alias.symlink_to(published, target_is_directory=True)
        self.assertEqual(backup_location(alias)['kind'], 'published')
        standalone = self.base / 'standalone-version'
        shutil.copytree(published.parent, standalone)
        self.assertEqual(backup_location(standalone / 'project')['kind'], 'published')

    def test_manifest_paths_and_symlinked_files_cannot_escape_restore(self):
        published = Path(self.sync()['path'])
        manifest_path = published.parent / 'manifest.json'
        manifest = _read_json(manifest_path)
        manifest['files']['../escape'] = {'size': 0, 'sha256': 'bad'}
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(ValueError, 'Unsafe path'):
            self.other_copy(published)
        del manifest['files']['../escape']
        _write_json(manifest_path, manifest)
        relative = 'frames/video/frame_000000.jpg'
        (published / relative).unlink()
        (published / relative).symlink_to(self.source / relative)
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            self.other_copy(published)
        self.assertFalse((self.base / 'other').exists())

    def test_failed_local_ancestry_write_reports_completed_backup(self):
        with patch('core.project_sync._save_local_heads', side_effect=OSError('read only')):
            result = self.sync()
        self.assertTrue(Path(result['path']).exists())
        self.assertIn('Backup completed', result['warning'])

    def test_missing_file_and_mid_restore_manifest_change_are_rejected(self):
        published = Path(self.sync()['path'])
        target = self.base / 'copy'
        manifest_path = published.parent / 'manifest.json'
        original = manifest_path.read_bytes()
        def change_manifest(message):
            if message.startswith('Copying and verifying 1/'):
                modified = _read_json(manifest_path)
                modified['project_name'] = 'Changed during copy'
                _write_json(manifest_path, modified)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            restore_project(published, target, progress=change_manifest)
        self.assertFalse(target.exists())
        manifest_path.write_bytes(original)
        (published / 'frames/video/frame_000000.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            restore_project(published, target)
        self.assertFalse(target.exists())

    def test_partial_backup_restore_reports_original_video_omission(self):
        self.configuration['include_videos'] = False
        published = self.sync()['path']
        result = restore_project(published, self.base / 'copy')
        self.assertFalse(result['include_videos'])
        self.assertFalse((Path(result['path']) / 'input_data').exists())
        self.assertTrue((Path(result['path']) / 'frames').is_dir())


if __name__ == '__main__':
    unittest.main()
