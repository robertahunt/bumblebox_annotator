"""Routine transfer verification versus explicit full checks on disposable backups."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from core import backup_integrity as integrity
from core.project_sync import (_digest, _read_json, _write_json, project_storage_lock,
                               publication_estimate, restore_project, snapshot_file)
from core.project_update import PublishedProject
from scripts import test_compact_sync as compact
from scripts import test_assigned_work as assigned


class BackupIntegrityTests(unittest.TestCase):
    setUp = compact.CompactSyncTests.setUp
    tearDown = compact.CompactSyncTests.tearDown
    sync = compact.CompactSyncTests.sync

    def snapshot(self):
        view = PublishedProject(self.source, self.configuration)
        return view.snapshot(next(iter(view.heads().items())))

    def check(self, snapshot=None, **kwargs):
        snapshot = snapshot or self.snapshot()
        return integrity.check_backup_integrity(self.source, self.configuration,
            snapshot['reference'], snapshot['manifest_hash'], **kwargs)

    def stored(self, name='frames/video/frame_000000.jpg'):
        s = self.snapshot()
        return snapshot_file(s['manifest_path'], s['manifest'], name)

    def test_check_reads_unique_contents_and_preserves_backup_and_local_data(self):
        self.sync()
        before = {p: p.read_bytes() for p in self.remote.rglob('*') if p.is_file()}
        local_before = {p: p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        with patch.object(integrity, '_hash_file', wraps=integrity._hash_file) as hashed:
            result = self.check()
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['bytes_read'], result['total_bytes'])
        self.assertEqual(hashed.call_count, result['unique_contents'])
        self.assertLess(result['unique_contents'], result['files'])  # Repeated '{}' contents.
        self.assertTrue(Path(result['report_path']).is_file())
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(local_before, {p: p.read_bytes() for p in local_before})
        self.assertFalse(list(self.remote.glob('*.sync-lock')))

    def test_detects_same_size_corruption_then_blocks_publish_until_rechecked(self):
        self.sync()
        path = self.stored()
        original, info = path.read_bytes(), path.stat()
        path.write_bytes(b'x' * len(original))
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        self.assertTrue(self.sync()['unchanged'])  # Routine checks explicitly do not promise this detection.
        result = self.check()
        self.assertEqual(result['status'], 'problems')
        self.assertEqual(result['issues'][0]['reason'], 'checksum mismatch')
        with self.assertRaisesRegex(ValueError, 'Known backup damage'):
            self.sync()
        path.write_bytes(original)  # Simulate an explicit administrator repair.
        with self.assertRaisesRegex(ValueError, 'Known backup damage'):
            self.sync()
        self.assertEqual(self.check()['status'], 'passed')
        self.assertTrue(self.sync()['unchanged'])

    def test_missing_and_truncated_files_block_routine_publish(self):
        self.sync()
        path = self.stored()
        data = path.read_bytes()
        for contents in (None, b''):
            if contents is None:
                path.unlink()
            else:
                path.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, 'damaged'):
                self.sync()
            result = self.check()
            self.assertEqual(result['status'], 'problems')
            path.write_bytes(data)
            self.assertEqual(self.check()['status'], 'passed')

    def test_cancelled_check_does_not_replace_last_pass_or_clear_damage(self):
        self.sync()
        self.check()
        path = self.stored()
        original = path.read_bytes()
        path.write_bytes(b'x' * len(original))
        self.check()
        path.write_bytes(original)
        result = self.check(cancelled=lambda: True)
        self.assertEqual(result['status'], 'cancelled')
        state = integrity.integrity_versions(self.source, self.configuration)['state']
        entry = state['versions']['/'.join(result['reference'])]
        self.assertEqual(entry['last_check']['status'], 'cancelled')
        self.assertEqual(entry['last_pass']['status'], 'passed')
        self.assertTrue(state['problems'])

    def test_cancellation_inside_large_file_is_not_recorded_as_pass(self):
        (self.source / 'frames/video/frame_000000.jpg').write_bytes(b'x' * 4 * 1024**2)
        self.sync()
        cancel = [False]
        def progress(message):
            if 'frame_000000.jpg' in message and '1.0/' in message:
                cancel[0] = True
        result = self.check(progress=progress, cancelled=lambda: cancel[0])
        self.assertEqual(result['status'], 'cancelled')
        self.assertLess(result['bytes_read'], result['total_bytes'])

    def test_offline_read_is_unable_not_corruption_and_report_write_can_fail(self):
        self.sync()
        with patch.object(integrity, '_hash_file', side_effect=OSError('drive disconnected')), \
                patch.object(integrity, '_save_report', side_effect=OSError('drive disconnected')):
            result = self.check()
        self.assertEqual(result['status'], 'unable')
        self.assertEqual(result['issues'], [])
        self.assertIn('Could not save', result['report_warning'])

    def test_changed_selected_manifest_is_rejected_before_audit(self):
        self.sync()
        old = self.snapshot()
        _write_json(old['manifest_path'], dict(old['manifest'], project_name='changed'))
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            self.check(old)

    def test_manifest_change_during_check_cannot_certify_version(self):
        self.sync()
        s = self.snapshot()
        def progress(_):
            _write_json(s['manifest_path'], dict(s['manifest'], project_name='changed'))
        result = self.check(s, progress=progress)
        self.assertEqual(result['status'], 'unable')
        self.assertIn('manifest changed', result['error'])

    def test_check_is_pinned_to_old_version_not_new_publication(self):
        self.sync()
        old = self.snapshot()
        (self.source / 'frames/new.jpg').write_bytes(b'new content')
        self.sync()
        new = self.snapshot()
        path = snapshot_file(new['manifest_path'], new['manifest'], 'frames/new.jpg')
        path.write_bytes(b'bad content')
        self.assertEqual(self.check(old)['status'], 'passed')
        self.assertEqual(self.check(new)['status'], 'problems')
        listing = integrity.integrity_versions(self.source, self.configuration)
        self.assertEqual(len(listing['versions']), 2)

    def test_full_check_and_publication_share_lock_without_creating_contributors(self):
        self.sync()
        view = PublishedProject(self.source, self.configuration)
        with project_storage_lock(self.configuration, view.project_id, 'another process'):
            with self.assertRaisesRegex(ValueError, 'locked'):
                self.check()
            with self.assertRaisesRegex(ValueError, 'locked'):
                self.sync()
        with patch.object(integrity, '_hash_file', wraps=integrity._hash_file) as hashed:
            result = self.check()
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(hashed.called)
        self.assertEqual(len(list(view.contributors.iterdir())), 1)

    def test_changed_file_reusing_existing_content_is_hashed(self):
        self.sync()
        path = self.stored()
        (self.source / 'frames/new.jpg').write_bytes((self.source / 'frames/video/frame_000000.jpg').read_bytes())
        # Content already checked structurally for another unchanged reference still
        # needs full verification when newly referenced by this publication.
        with patch('core.project_sync._digest', wraps=_digest) as hashed:
            self.sync()
        self.assertTrue(any(call.args[0] == path for call in hashed.call_args_list))

    def test_legacy_full_backup_check_and_restore(self):
        self.configuration['storage_mode'] = 'full'
        result = self.sync()
        self.assertEqual(self.check()['status'], 'passed')
        restore_project(result['path'], self.base / 'restored')
        path = self.stored()
        path.write_bytes(b'x' * path.stat().st_size)
        self.assertEqual(self.check()['status'], 'problems')
        with self.assertRaisesRegex(ValueError, 'Known backup damage'):
            self.sync()

    def test_legacy_damage_warning_survives_contributor_folder_rename(self):
        self.configuration['storage_mode'] = 'full'
        self.sync()
        path = self.stored()
        path.write_bytes(b'x' * path.stat().st_size)
        self.assertEqual(self.check()['status'], 'problems')
        self.contributor['name'] = 'August renamed'
        with self.assertRaisesRegex(ValueError, 'Known backup damage'):
            self.sync()

    def test_new_reference_to_corrupt_existing_object_does_not_publish(self):
        self.sync()
        path = self.stored()
        source = self.source / 'frames/video/frame_000000.jpg'
        (self.source / 'frames/new.jpg').write_bytes(source.read_bytes())
        path.write_bytes(b'x' * path.stat().st_size)
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'damaged'):
            self.sync()
        self.assertEqual(snapshot['reference'], self.snapshot()['reference'])

    def test_report_lookup_checks_identity_and_rejects_unsafe_ids(self):
        self.sync()
        report = self.check()
        loaded = integrity.read_integrity_report(self.source, self.configuration,
                                                 report['reference'], report['report_id'])
        self.assertEqual(loaded['manifest_hash'], report['manifest_hash'])
        with self.assertRaises(ValueError):
            integrity.read_integrity_report(self.source, self.configuration, report['reference'], '../../bad')
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            integrity.read_integrity_report(self.source, self.configuration,
                                            [report['reference'][0], 'another-version'], report['report_id'])

    def test_wrong_destination_and_invalid_shared_report_fail_closed(self):
        self.sync()
        view = PublishedProject(self.source, self.configuration)
        self.check()
        _write_json(view.home / 'verification/state.json', {'format': 99})
        with self.assertRaisesRegex(ValueError, 'integrity records'):
            self.sync()
        with self.assertRaises(ValueError):
            integrity.integrity_versions(self.source, dict(self.configuration, root_id='wrong'))

    def test_unchanged_estimated_publish_does_not_read_remote_content(self):
        self.sync()
        estimate = publication_estimate(self.source, self.configuration, self.contributor)
        with patch('core.project_sync._digest', side_effect=AssertionError('Must not hash payloads')):
            self.assertTrue(self.sync(estimate=estimate)['unchanged'])


class AssignedIntegrityTests(unittest.TestCase):
    setUp = assigned.AssignedWorkTests.setUp
    view = assigned.AssignedWorkTests.view
    download = assigned.AssignedWorkTests.download
    change = assigned.AssignedWorkTests.change

    def test_partial_publish_preserves_undownloaded_data_without_hashing_it(self):
        local = self.download()
        self.change(local)
        from core.assigned_work import publish_work, latest_snapshot
        from core.project_delta import file_record
        snapshot, _ = latest_snapshot(self.view())
        remote_media = {snapshot_file(snapshot['manifest_path'], snapshot['manifest'], name)
                        for name in snapshot['manifest']['files'] if name.startswith(('frames/', 'input_data/'))
                        and not name.endswith('.json')}
        def record(path):
            self.assertNotIn(path, remote_media)
            return file_record(path)
        with patch('core.assigned_work.file_record', side_effect=record):
            result = publish_work(local, self.configuration, self.alex)
        restore_project(result['path'], self.base / 'restored')
        self.assertTrue((self.base / 'restored/frames/beta/frame_000002.jpg').is_file())

    def test_known_corruption_in_undownloaded_file_blocks_assigned_publish(self):
        from core.assigned_work import latest_snapshot, publish_work
        local = self.download()
        self.change(local)
        snapshot, _ = latest_snapshot(self.view())
        path = snapshot_file(snapshot['manifest_path'], snapshot['manifest'], 'input_data/train/beta.mp4')
        path.write_bytes(b'x' * path.stat().st_size)
        result = integrity.check_backup_integrity(self.source, self.configuration,
            snapshot['reference'], snapshot['manifest_hash'])
        self.assertEqual(result['status'], 'problems')
        with self.assertRaisesRegex(ValueError, 'Known backup damage'):
            publish_work(local, self.configuration, self.alex)


if __name__ == '__main__':
    unittest.main()
