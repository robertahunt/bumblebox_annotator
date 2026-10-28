"""Incremental assigned uploads using disposable local/shared projects only."""

from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from core.annotation import AnnotationManager
from core.assigned_work import (AssignedUploadCache, latest_snapshot, publish_work,
                               read_assignments, save_assignments)
from core.backup_integrity import check_backup_integrity
from core.project_delta import file_record
from core.project_sync import _inventory, _read_json, _write_json, snapshot_file
from scripts import test_assigned_work as fixtures


class AssignedUploadTests(unittest.TestCase):
    view = fixtures.AssignedWorkTests.view
    download = fixtures.AssignedWorkTests.download
    change = fixtures.AssignedWorkTests.change

    def setUp(self):
        fixtures.AssignedWorkTests.setUp(self)
        self.cache = AssignedUploadCache()

    def publish(self, local, **kwargs):
        return publish_work(local, self.configuration, self.alex,
                            checksum_cache=self.cache, **kwargs)

    def local_reads(self, mocked, local):
        return {Path(c.args[0]).relative_to(local).as_posix()
                for c in mocked.call_args_list if local in Path(c.args[0]).parents}

    def change_hive(self, local):
        manager = AnnotationManager()
        manager.contributor_session = self.alex
        annotations, _ = manager.load_video_annotations(local, 'alpha')
        annotations[0]['mask'][0, 0] = 255
        manager.save_video_annotations(local, 'alpha', annotations)

    def test_second_upload_hashes_only_changed_hive_files(self):
        local = self.download()
        with patch('core.assigned_work.file_record', wraps=file_record) as checked:
            self.publish(local)
        inventory = _inventory(local, True, with_identity=True)
        self.assertEqual(self.local_reads(checked, local), set(inventory))
        self.change_hive(local)
        changed = {n for n, value in _inventory(local, True, with_identity=True).items()
                   if inventory.get(n) != value}
        messages = []
        with patch('core.assigned_work.file_record', wraps=file_record) as checked:
            result = self.publish(local, progress=messages.append)
        self.assertEqual(self.local_reads(checked, local), changed)
        self.assertIn('annotations/png/alpha/video_annotations_hive.png', changed)
        self.assertGreater(result['copied_files'], 0)
        self.assertEqual(result['uploaded_videos'], ['alpha'])
        self.assertEqual(result['changed_videos'], ['alpha'])
        self.assertFalse(any('Checking stored file information' in m for m in messages))
        self.assertTrue(any('unchanged checksums reused' in m for m in messages))
        with patch('core.assigned_work.file_record', wraps=file_record) as checked:
            result = self.publish(local)
            self.assertTrue(result['unchanged'])
            self.assertEqual(result['uploaded_videos'], [])
            self.assertEqual(result['changed_videos'], [])
        self.assertEqual(self.local_reads(checked, local), set())

    def test_cache_is_session_only_and_bound_to_one_working_copy(self):
        first, second = self.download(), self.download(name='second')
        self.publish(first)
        for local in (second, first):
            with patch('core.assigned_work.file_record', wraps=file_record) as checked:
                self.publish(local)
            self.assertEqual(self.local_reads(checked, local), set(_inventory(local, True)))
        self.cache = AssignedUploadCache()  # Restarting the app has no trusted disk cache.
        with patch('core.assigned_work.file_record', wraps=file_record) as checked:
            self.publish(first)
        self.assertEqual(self.local_reads(checked, first), set(_inventory(first, True)))

    def test_same_size_edit_with_restored_mtime_cannot_hide_protected_change(self):
        local = self.download()
        self.publish(local)
        path = local / 'frames/alpha/frame_000000.jpg'
        original, info = path.read_bytes(), path.stat()
        path.write_bytes(b'x' * len(original))
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
        self.assertEqual(path.stat().st_mtime_ns, info.st_mtime_ns)
        self.assertNotEqual(path.stat().st_ctime_ns, info.st_ctime_ns)
        heads = self.view().heads()
        with self.assertRaisesRegex(ValueError, 'protected file'):
            self.publish(local)
        self.assertEqual(self.view().heads(), heads)

    def test_atomic_replacement_and_new_protected_file_are_detected(self):
        local = self.download()
        self.publish(local)
        path = local / 'frames/alpha/frame_000000.jpg'
        info = path.stat()
        replacement = self.base / 'replacement.jpg'
        replacement.write_bytes(b'x' * info.st_size)
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        os.replace(replacement, path)
        with self.assertRaisesRegex(ValueError, 'protected file'):
            self.publish(local)
        # Use a separate valid copy to test newly introduced unassigned files.
        second = self.download(name='second')
        self.publish(second)
        path = second / 'frames/alpha/unassigned.jpg'
        path.write_bytes(b'not assigned')
        with self.assertRaisesRegex(ValueError, 'protected file'):
            self.publish(second)

    def test_cached_annotation_deletion_is_still_published(self):
        local = self.download()
        self.publish(local)
        name = 'annotations/json/alpha/frame_000000.json'
        (local / name).unlink()
        result = self.publish(local)
        files = _read_json(Path(result['path']) / 'manifest.json')['files']
        self.assertNotIn(name, files)
        self.assertNotIn(name, self.cache.records)
        self.assertIn('annotations/json/alpha/frame_000001.json', files)
        self.assertEqual(result['copied_files'], 0)
        self.assertEqual(result['uploaded_videos'], [])
        self.assertEqual(result['changed_videos'], ['alpha'])

    def test_upload_counts_changed_videos_not_all_downloaded_assignments(self):
        choices = deepcopy(self.choices)
        choices['beta']['contributor'] = self.alex
        data = read_assignments(self.view())
        save_assignments(self.source, self.configuration, self.coordinator, choices, data['revision'])
        local = self.download(videos=['alpha', 'beta'])
        self.change_hive(local)
        result = self.publish(local)
        self.assertEqual(result['uploaded_videos'], ['alpha'])
        self.assertEqual(result['changed_videos'], ['alpha'])
        # Identical new files are transferred once, but belong to both videos.
        payload = b'[]\n'
        (local / 'annotations/json/alpha/frame_000000.json').write_bytes(payload)
        (local / 'annotations/json/beta/frame_000001.json').write_bytes(payload)
        result = self.publish(local)
        self.assertEqual(result['copied_files'], 1)
        self.assertEqual(result['uploaded_videos'], ['alpha', 'beta'])
        self.assertEqual(result['changed_videos'], ['alpha', 'beta'])

    def test_metadata_only_change_counts_its_video(self):
        local = self.download()
        path = local / 'frames/alpha/video_metadata.json'
        metadata = _read_json(path)
        metadata['max_mask_id'] += 1
        _write_json(path, metadata)
        result = self.publish(local)
        self.assertEqual(result['copied_files'], 1)
        self.assertEqual(result['uploaded_videos'], ['alpha'])
        self.assertEqual(result['changed_videos'], ['alpha'])

    def test_reusing_stored_content_reports_changes_without_uploads(self):
        local = self.download()
        (local / 'annotations/json/alpha/frame_000000.json').write_bytes(
            (local / 'annotations/json/alpha/frame_000001.json').read_bytes())
        result = self.publish(local)
        self.assertEqual(result['copied_files'], 0)
        self.assertEqual(result['uploaded_videos'], [])
        self.assertEqual(result['changed_videos'], ['alpha'])

    def test_edits_during_hashing_do_not_seed_cache_or_publish(self):
        local = self.download()
        name = 'frames/alpha/frame_000000.jpg'
        path = local / name
        heads = self.view().heads()

        def changed_after_read(candidate):
            result = file_record(candidate)
            if candidate == path:
                path.write_bytes(b'x' * path.stat().st_size)
            return result

        with patch('core.assigned_work.file_record', side_effect=changed_after_read):
            with self.assertRaisesRegex(ValueError, 'Local files changed'):
                self.publish(local)
        self.assertEqual(self.cache.records, {})
        self.assertEqual(self.view().heads(), heads)

    def test_edits_after_cached_check_and_cancellation_do_not_publish(self):
        local = self.download()
        self.publish(local)
        heads = self.view().heads()
        with self.assertRaises(InterruptedError):
            self.publish(local, cancelled=lambda: True)
        self.change_hive(local)
        path = local / 'frames/alpha/frame_000000.jpg'

        def edit(message):
            if message.startswith('Keeping '):
                info = path.stat()
                path.write_bytes(b'x' * info.st_size)
                os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))

        with self.assertRaisesRegex(ValueError, 'Local files changed'):
            self.publish(local, progress=edit)
        self.assertEqual(self.view().heads(), heads)

    def test_28000_references_do_not_probe_remote_media_or_virtual_project_paths(self):
        snapshot, _ = latest_snapshot(self.view())
        manifest = deepcopy(snapshot['manifest'])
        record = manifest['files']['input_data/train/beta.mp4']
        manifest['files'].update({f'input_data/train/synthetic_{i}.mp4': dict(record)
                                  for i in range(28000)})
        _write_json(snapshot['manifest_path'], manifest)
        local = self.download()
        self.change_hive(local)
        media = {snapshot_file(snapshot['manifest_path'], manifest, n)
                 for n in snapshot['manifest']['files']
                 if n.startswith(('frames/', 'input_data/')) and not n.endswith('.json')}
        lstat = Path.lstat

        def guarded_lstat(path, *args, **kwargs):
            self.assertNotIn(path, media, 'Unchanged remote media must not be stat-checked')
            if self.remote in path.parents:
                self.assertNotIn('project', path.relative_to(self.remote).parts,
                                 'Compact manifests must not probe a virtual project tree')
            return lstat(path, *args, **kwargs)

        with patch.object(Path, 'lstat', guarded_lstat), \
                patch('core.backup_integrity.check_stored_file',
                      side_effect=AssertionError('Do not scan unchanged remote files')):
            result = self.publish(local)
        files = _read_json(Path(result['path']) / 'manifest.json')['files']
        self.assertEqual(files['input_data/train/synthetic_27999.mp4'], record)
        self.assertGreater(len(files), 28000)

    def test_unknown_remote_damage_is_deferred_to_integrity_check_and_download(self):
        local = self.download()
        self.change_hive(local)
        snapshot, _ = latest_snapshot(self.view())
        name = 'frames/beta/frame_000002.jpg'
        snapshot_file(snapshot['manifest_path'], snapshot['manifest'], name).unlink()
        result = self.publish(local)
        latest, _ = latest_snapshot(self.view())
        checked = check_backup_integrity(self.source, self.configuration,
                                        latest['reference'], latest['manifest_hash'])
        self.assertEqual(checked['status'], 'problems')
        self.assertTrue(any(name in issue['paths'] for issue in checked['issues']))
        with self.assertRaises((OSError, ValueError)):
            self.download(self.sam, ['beta'], 'damaged-download')
        self.assertFalse((self.base / 'damaged-download').exists())
        with self.assertRaisesRegex(ValueError, 'Known backup damage'):
            self.publish(local)
        self.assertEqual(Path(result['path']), latest['manifest_path'].parent)

    def test_changed_content_is_verified_even_if_object_already_exists(self):
        local = self.download()
        self.publish(local)
        name = 'annotations/json/alpha/frame_000001.json'
        (local / 'annotations/json/alpha/frame_000000.json').write_bytes((local / name).read_bytes())
        snapshot, _ = latest_snapshot(self.view())
        existing = snapshot_file(snapshot['manifest_path'], snapshot['manifest'], name)
        existing.write_bytes(b'x' * existing.stat().st_size)
        heads = self.view().heads()
        with self.assertRaisesRegex(ValueError, 'damaged'):
            self.publish(local)
        self.assertEqual(self.view().heads(), heads)

    def test_new_upload_is_verified_before_publishing_latest_pointer(self):
        local = self.download()
        self.change_hive(local)
        heads = self.view().heads()
        from core.assigned_work import shutil
        copy = shutil.copy2

        def corrupt_copy(source, destination, *args, **kwargs):
            result = copy(source, destination, *args, **kwargs)
            if self.remote in Path(destination).parents:
                Path(destination).write_bytes(b'x' * Path(destination).stat().st_size)
            return result

        with patch('core.assigned_work.shutil.copy2', side_effect=corrupt_copy):
            with self.assertRaisesRegex(ValueError, 'failed verification'):
                self.publish(local)
        self.assertEqual(self.view().heads(), heads)

    def test_invalid_compact_manifest_paths_and_records_are_still_rejected(self):
        snapshot, _ = latest_snapshot(self.view())
        for name, record in (
            ('/frames/outside.jpg', {'size': 1, 'sha256': 'a' * 64}),
            ('frames/../outside.jpg', {'size': 1, 'sha256': 'a' * 64}),
            ('frames/bad.jpg', {'size': 1, 'sha256': 'z' * 64}),
            ('frames/bad.jpg', {'size': -1, 'sha256': 'a' * 64}),
        ):
            with self.subTest(name=name, record=record):
                manifest = deepcopy(snapshot['manifest'])
                manifest['files'][name] = record
                _write_json(snapshot['manifest_path'], manifest)
                with self.assertRaisesRegex(ValueError, 'Unsupported file'):
                    self.view().snapshot(snapshot['reference'])


if __name__ == '__main__':
    unittest.main()
