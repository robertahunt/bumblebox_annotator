"""Shared-object publications, delta updates, and retained rollback controls."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.contributors import new_session
from core.project_sync import (LOCATION_MARKERS, _digest, _inventory, _read_json, _write_json,
    backup_location, object_path, publication_estimate, restore_project, snapshot_file,
    sync_project, update_journal_path)
from core.project_update import PreparedUpdate, published_versions, recover_update
from core.project_delta import (delete_rollback_bundle, rollback_bundles, rollback_update,
                                rollback_root)
from scripts import test_project_sync as fixtures
from scripts.test_project_update import contents


class CompactSyncTests(unittest.TestCase):
    tearDown = fixtures.FolderSyncTests.tearDown
    sync = fixtures.FolderSyncTests.sync

    def setUp(self):
        fixtures.FolderSyncTests.setUp(self)
        self.configuration.pop('storage_mode')  # Compact publications are the default.

    def publish_change(self):
        (self.source / 'annotations/json/video/frame_000000.json').write_text('["changed"]')
        return self.sync()

    def objects(self):
        return {p.relative_to(self.remote).as_posix(): (p.stat().st_ino, p.read_bytes())
                for p in self.remote.glob('*/objects/sha256/*/*') if p.is_file()}

    def local_and_plan(self):
        first = self.sync()
        local = self.base / 'working'
        restore_project(first['path'], local)
        self.publish_change()
        reference = published_versions(local, self.configuration)[0]['reference']
        return local, PreparedUpdate(local, self.configuration, reference)

    def test_default_compact_publish_restores_exact_managed_data(self):
        result = self.sync()
        version = Path(result['path'])
        manifest = _read_json(version / 'manifest.json')
        self.assertEqual(manifest['format'], 3)
        self.assertFalse((version / 'project').exists())
        self.assertEqual(backup_location(version)['kind'], 'published')
        self.assertEqual(len(self.objects()), len({entry['sha256'] for entry in manifest['files'].values()}))
        local = self.base / 'restored'
        restore_project(version, local)
        for name in _inventory(self.source, True):
            self.assertEqual((local / name).read_bytes(), (self.source / name).read_bytes())
        self.assertFalse((local / 'models').exists())
        self.assertEqual(contents(version).keys(), {'manifest.json'})

    def test_current_folder_name_overrides_stale_internal_name(self):
        renamed = self.base / '2026_CV4E'
        self.source.rename(renamed)
        self.source = renamed
        _write_json(self.source / 'annotations/project.json', {'name': 'test'})
        version = Path(self.sync()['path'])
        home = version.parents[3]
        self.assertTrue(home.name.startswith('2026_CV4E--'))
        self.assertEqual(_read_json(home / LOCATION_MARKERS['project'])['name'], '2026_CV4E')
        self.assertEqual(_read_json(self.source / 'annotations/project.json')['name'], 'test')

    def test_outdated_archive_name_is_renamed_without_copying_history_or_objects(self):
        first = Path(self.sync()['path'])
        sync_project(self.source, self.configuration, new_session('Alex'))
        current = first.parents[3]
        marker = _read_json(current / LOCATION_MARKERS['project'])
        old = self.remote / ('test--' + marker['id'][:8])
        current.rename(old)
        _write_json(old / LOCATION_MARKERS['project'], dict(marker, name='test'))
        history, objects = contents(old / 'contributors'), contents(old / 'objects')
        estimate = publication_estimate(self.source, self.configuration, self.contributor)
        self.assertEqual(estimate['rename_from'], old.name)
        self.assertEqual(estimate['project_name'], self.source.name)
        self.assertTrue(old.exists())
        with patch('core.project_sync.shutil.copy2', side_effect=AssertionError('must not copy')):
            result = self.sync()
        self.assertTrue(result['unchanged'])
        self.assertFalse(old.exists())
        self.assertEqual(Path(result['path']), first)
        self.assertEqual(contents(current / 'contributors'), history)
        self.assertEqual(contents(current / 'objects'), objects)
        self.assertEqual(len(list(self.remote.glob('*/' + LOCATION_MARKERS['project']))), 1)
        self.assertIsNone(publication_estimate(self.source, self.configuration, self.contributor)['rename_from'])
        restore_project(result['path'], self.base / 'restored')

    def test_archive_rename_does_not_overwrite_colliding_project_folder(self):
        first = Path(self.sync()['path'])
        old = first.parents[3]
        identity = _read_json(old / LOCATION_MARKERS['project'])['id']
        renamed = self.base / 'renamed'
        self.source.rename(renamed)
        self.source = renamed
        collision = self.remote / ('renamed--' + identity[:8])
        collision.mkdir()
        (collision / 'keep.txt').write_text('unrelated project')
        result = self.sync()
        self.assertEqual(Path(result['path']).parents[3].name, 'renamed--' + identity)
        self.assertEqual((collision / 'keep.txt').read_text(), 'unrelated project')
        self.assertFalse(old.exists())
        self.assertEqual(self.sync()['path'], result['path'])

    def test_locked_or_failed_project_rename_preserves_history_for_retry(self):
        first = Path(self.sync()['path'])
        old = first.parents[3]
        renamed = self.base / 'renamed'
        self.source.rename(renamed)
        self.source = renamed
        lock = first.parents[1] / '.sync-lock'
        lock.write_text('another process')
        history = contents(old / 'contributors')
        with self.assertRaisesRegex(ValueError, 'locked'):
            self.sync()
        self.assertEqual(contents(old / 'contributors'), history)
        lock.unlink()
        history = contents(old / 'contributors')
        with patch.object(Path, 'rename', side_effect=OSError('access denied')):
            result = self.sync()
        self.assertIn('archive folder could not be renamed: access denied', result['warning'])
        self.assertEqual(Path(result['path']), first)
        self.assertEqual(contents(old / 'contributors'), history)
        result = self.sync()
        self.assertTrue(result['unchanged'])
        self.assertFalse(old.exists())
        self.assertEqual(contents(Path(result['path']).parents[3] / 'contributors'), history)

    def test_small_change_adds_one_object_not_media_copies(self):
        first = self.sync()
        before = self.objects()
        second = self.publish_change()
        after = self.objects()
        self.assertEqual(len(after) - len(before), 1)
        for name, value in before.items():
            self.assertEqual(after[name], value)
        self.assertEqual(second['copied_files'], 1)
        restore_project(first['path'], self.base / 'old')
        self.assertEqual((self.base / 'old/annotations/json/video/frame_000000.json').read_text(), '[]')
        self.assertTrue(self.sync()['unchanged'])
        self.assertEqual(self.objects(), after)

    def test_excluding_original_videos_keeps_frames_and_annotations(self):
        self.configuration['include_videos'] = False
        result = self.sync()
        restored = self.base / 'restored'
        restore_project(result['path'], restored)
        self.assertFalse((restored / 'input_data').exists())
        for name in _inventory(self.source, False):
            self.assertEqual((restored / name).read_bytes(), (self.source / name).read_bytes())
        self.assertTrue((restored / 'frames/video/frame_000000.jpg').exists())
        self.assertTrue((restored / 'annotations/json/video/frame_000000.json').exists())
        self.assertTrue((self.source / 'input_data/train/video.mp4').exists())

    def test_objects_are_shared_across_contributors(self):
        self.sync()
        before = self.objects()
        other = sync_project(self.source, self.configuration, new_session('Alex'))
        self.assertEqual(other['copied_files'], 0)
        self.assertEqual(before, self.objects())
        self.assertEqual(len(list(self.remote.rglob('latest.json'))), 2)

    def test_unchanged_shared_media_is_verified_once_per_publication(self):
        version = Path(self.sync()['path'])
        manifest = _read_json(version / 'manifest.json')
        frame = snapshot_file(version / 'manifest.json', manifest, 'frames/video/frame_000000.jpg')
        with patch('core.project_sync._digest', wraps=_digest) as digest:
            self.publish_change()
        self.assertEqual(sum(call.args[0] == frame for call in digest.call_args_list), 1)

    def test_legacy_then_compact_then_full_remain_restorable(self):
        self.configuration['storage_mode'] = 'full'
        old = self.sync()
        self.configuration['storage_mode'] = 'deduplicated'
        compact = self.publish_change()
        self.configuration['storage_mode'] = 'full'
        (self.source / 'annotations/next.json').write_text('[]')
        full = self.sync()
        for index, version in enumerate((old, compact, full)):
            restored = self.base / str(index)
            restore_project(version['path'], restored)
            self.assertEqual((restored / 'frames/video/frame_000000.jpg').read_text(), 'image')

    def test_storage_mode_change_alone_creates_correct_format(self):
        for index, mode in enumerate(('full', 'deduplicated', 'full')):
            self.configuration['storage_mode'] = mode
            result = self.sync()
            self.assertFalse(result['unchanged'])
            location = backup_location(result['path'])
            self.assertEqual(location['manifest']['format'], 3 if mode == 'deduplicated' else 2)
            restored = self.base / str(index)
            restore_project(result['path'], restored)
            self.assertEqual((restored / 'annotations/json/video/frame_000000.json').read_text(), '[]')

    def test_compact_update_only_stages_changed_files_and_can_undo(self):
        local, plan = self.local_and_plan()
        model = local / 'models/huge.pt'
        model.parent.mkdir()
        model.write_bytes(b'x' * 1000000)
        before = contents(local)
        frame = local / 'frames/video/frame_000000.jpg'
        inode = frame.stat().st_ino
        self.assertLess(plan.additional_bytes, 10000)
        applied = plan.apply()
        bundle = Path(applied['backup'])
        self.assertEqual(bundle.parent, Path(self.configuration['rollback_root']))
        self.assertFalse((bundle / 'before/frames').exists())
        self.assertFalse((bundle / 'before/models').exists())
        self.assertEqual(frame.stat().st_ino, inode)
        self.assertEqual(model.read_bytes(), b'x' * 1000000)
        rollback_update(local, bundle, self.configuration)
        self.assertEqual(contents(local), before)
        self.assertEqual(frame.stat().st_ino, inode)

    def test_rollback_blocks_new_annotations_but_allows_local_only_outputs(self):
        local, plan = self.local_and_plan()
        bundle = plan.apply()['backup']
        (local / 'annotations/newer.json').write_text('[]')
        before = contents(local)
        with self.assertRaisesRegex(ValueError, 'changed since'):
            rollback_update(local, bundle, self.configuration)
        self.assertEqual(before, contents(local))
        (local / 'annotations/newer.json').unlink()
        (local / 'analysis.txt').write_text('new output')
        rollback_update(local, bundle, self.configuration)
        self.assertEqual((local / 'analysis.txt').read_text(), 'new output')

    def test_retention_only_removes_selected_completed_local_bundle(self):
        local, plan = self.local_and_plan()
        bundle = Path(plan.apply()['backup'])
        self.assertEqual(len(rollback_bundles(local, self.configuration)), 1)
        remote = contents(self.remote)
        current = contents(local)
        update_journal_path(local).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'interrupted'):
            delete_rollback_bundle(local, bundle, self.configuration)
        update_journal_path(local).unlink()
        delete_rollback_bundle(local, bundle, self.configuration)
        self.assertFalse(bundle.exists())
        self.assertEqual(remote, contents(self.remote))
        self.assertEqual(current, contents(local))

    def test_rollback_verifies_backed_up_bytes_before_any_replacement(self):
        local, plan = self.local_and_plan()
        bundle = Path(plan.apply()['backup'])
        (bundle / 'before/annotations/json/video/frame_000000.json').write_text('damaged backup')
        before = contents(local)
        with self.assertRaisesRegex(ValueError, 'failed verification'):
            rollback_update(local, bundle, self.configuration)
        self.assertEqual(before, contents(local))
        self.assertFalse(update_journal_path(local).exists())

    def test_interrupted_staging_recovers_without_changing_working_files(self):
        local, plan = self.local_and_plan()
        before = contents(local)
        cancel = [False]
        with patch('core.project_delta.shutil.rmtree', side_effect=OSError('folder busy')):
            with self.assertRaisesRegex(OSError, 'folder busy'):
                plan.apply(progress=lambda text: cancel.__setitem__(0, True), cancelled=lambda: cancel[0])
        journal = update_journal_path(local)
        self.assertEqual(_read_json(journal)['phase'], 'staging')
        recover_update(local)
        self.assertEqual(before, contents(local))
        self.assertFalse(journal.exists())

    def test_unknown_recovery_phase_preserves_all_files_for_review(self):
        local, plan = self.local_and_plan()
        bundle = Path(plan.apply()['backup'])
        manifest = _read_json(bundle / 'manifest.json')
        journal = update_journal_path(local)
        _write_json(journal, dict(manifest, format=2, phase='invalid', backup=str(bundle), stage='invalid'))
        before = contents(self.base)
        with self.assertRaisesRegex(ValueError, 'Unknown update recovery phase'):
            recover_update(local)
        self.assertEqual(before, contents(self.base))

    def test_incomplete_other_project_bundle_does_not_block_history(self):
        local, plan = self.local_and_plan()
        bundle = Path(plan.apply()['backup'])
        incomplete = bundle.parent / 'bumblebox-before-update-interrupted'
        incomplete.mkdir()
        self.assertEqual([item['path'] for item in rollback_bundles(local, self.configuration)], [bundle])
        self.assertTrue(incomplete.exists())

    def test_missing_or_damaged_object_cannot_restore_or_republish(self):
        version = Path(self.sync()['path'])
        manifest = _read_json(version / 'manifest.json')
        path = snapshot_file(version / 'manifest.json', manifest, 'frames/video/frame_000000.jpg')
        path.write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'damaged'):
            restore_project(version, self.base / 'bad-copy')
        with self.assertRaisesRegex(ValueError, 'damaged'):
            self.sync()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            restore_project(version, self.base / 'missing-copy')
        self.assertFalse((self.base / 'bad-copy').exists())

    def test_bad_digest_and_detached_manifest_do_not_escape_store(self):
        version = Path(self.sync()['path'])
        manifest = _read_json(version / 'manifest.json')
        bad = dict(manifest, files={'frames/image.jpg': {'sha256': '../outside', 'size': 1}})
        with self.assertRaisesRegex(ValueError, 'checksum'):
            snapshot_file(version / 'manifest.json', bad, 'frames/image.jpg')
        isolated = self.base / 'isolated'
        isolated.mkdir()
        _write_json(isolated / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'shared file store is missing'):
            restore_project(isolated, self.base / 'bad-copy')

    def test_estimates_and_disk_full_do_not_advance_history(self):
        self.sync()
        estimate = publication_estimate(self.source, self.configuration, self.contributor)
        self.assertEqual(estimate['files'], len(_inventory(self.source, True)))
        before = contents(self.remote)
        (self.source / 'annotations/next.json').write_text('["new"]')
        with patch('core.project_sync.shutil.disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(OSError, 'Insufficient destination'):
                self.sync()
        self.assertEqual(before, contents(self.remote))

    def test_rollback_folder_must_not_be_project_or_archive(self):
        for path in (self.source, self.source / 'backup', self.remote):
            with self.assertRaisesRegex(ValueError, 'outside'):
                rollback_root(dict(self.configuration, rollback_root=str(path)), self.source)


if __name__ == '__main__':
    unittest.main()
