"""Outgoing publication review uses temporary projects and archives only."""

from pathlib import Path
import unittest
from unittest.mock import patch

from core.contributors import new_session
from core.project_sync import _digest, _inventory, _read_json, _write_json, publication_estimate, sync_project
from core.publication_review import file_context
from scripts import test_compact_sync as fixtures
from scripts.test_project_update import contents


class PublicationReviewTests(unittest.TestCase):
    setUp = fixtures.CompactSyncTests.setUp
    tearDown = fixtures.CompactSyncTests.tearDown
    sync = fixtures.CompactSyncTests.sync

    def estimate(self, **options):
        return publication_estimate(self.source, dict(self.configuration, **options), self.contributor)

    def test_first_backup_review_is_read_only_and_includes_only_managed_data(self):
        local_before, remote_before = contents(self.source), contents(self.remote)
        estimate = self.estimate()
        review = estimate['review']
        self.assertIsNone(review['baseline'])
        self.assertEqual(review['counts'], {'Added': len(_inventory(self.source, True))})
        self.assertEqual(review['unchanged'], 0)
        self.assertEqual({row['path'] for row in review['rows']}, set(estimate['source_inventory']))
        self.assertEqual(local_before, contents(self.source))
        self.assertEqual(remote_before, contents(self.remote))

    def test_changes_group_annotations_frames_originals_and_project_files(self):
        shared = self.source / 'annotations/json/video/video_annotations.json'
        shared.write_text('[]')
        self.sync()
        shared.write_text('["shared mask changed"]')
        (self.source / 'annotations/json/video/frame_000000.json').write_text('["frame changed"]')
        (self.source / 'frames/video/frame_000000.jpg').unlink()
        (self.source / 'annotations/project.json').write_text('{"name": "new name"}')
        new = self.source / 'frames/second/frame_000000.png'
        new.parent.mkdir()
        new.write_bytes(b'new frame')
        review = self.estimate()['review']
        rows = {row['path']: row for row in review['rows']}
        self.assertEqual(review['counts'], {'Added': 1, 'Changed': 3, 'Removed': 1})
        self.assertEqual(rows[shared.relative_to(self.source).as_posix()]['scope'], 'Shared across video')
        self.assertEqual(rows['annotations/json/video/frame_000000.json']['scope'], 'Frame annotation')
        self.assertEqual(rows['frames/video/frame_000000.jpg']['action'], 'Removed')
        self.assertIsNone(rows['frames/video/frame_000000.jpg']['after_bytes'])
        self.assertEqual(rows['frames/second/frame_000000.png']['video'], 'second')
        self.assertIsNone(rows['annotations/project.json']['video'])
        self.assertGreater(review['unchanged'], 0)
        self.assertNotIn('files', review['baseline'])

    def test_excluded_originals_are_not_reported_as_deleted(self):
        self.sync()
        estimate = self.estimate(include_videos=False)
        rows = estimate['review']['rows']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['path'], 'input_data/train/video.mp4')
        self.assertEqual(rows[0]['action'], 'Excluded')
        self.assertEqual(rows[0]['video'], 'video')
        self.assertNotIn(rows[0]['path'], estimate['source_inventory'])

    def test_identical_content_at_a_new_path_is_an_addition_even_if_reused_remotely(self):
        self.sync()
        original = self.source / 'frames/video/frame_000000.jpg'
        (original.parent / 'frame_000001.jpg').write_bytes(original.read_bytes())
        review = self.estimate()['review']
        self.assertEqual(review['counts'], {'Added': 1})
        self.assertEqual(review['rows'][0]['path'], 'frames/video/frame_000001.jpg')

    def test_own_contributor_baseline_and_no_change_review(self):
        first = self.sync()
        review = self.estimate()['review']
        self.assertEqual(review['baseline']['revision'], Path(first['path']).name)
        self.assertEqual(review['baseline']['contributor_name'], self.contributor['name'])
        self.assertEqual(review['rows'], [])
        other = new_session('Someone Else')
        other_estimate = publication_estimate(self.source, self.configuration, other)
        self.assertIsNone(other_estimate['review']['baseline'])
        self.assertTrue(all(row['action'] == 'Added' for row in other_estimate['review']['rows']))

    def test_full_mode_comparison_also_reuses_local_hashes(self):
        configuration = dict(self.configuration, storage_mode='full')
        sync_project(self.source, configuration, self.contributor)
        (self.source / 'annotations/json/video/frame_000000.json').write_text('["changed"]')
        estimate = self.estimate(storage_mode='full')
        self.assertEqual(estimate['review']['counts'], {'Changed': 1})
        with patch('core.project_sync._digest', wraps=_digest) as digest:
            sync_project(self.source, configuration, self.contributor, estimate=estimate)
        self.assertFalse(any(self.source in call.args[0].parents for call in digest.call_args_list))

    def test_modified_review_baseline_cannot_be_published_without_new_estimate(self):
        version = Path(self.sync()['path'])
        estimate = self.estimate()
        manifest_path = version / 'manifest.json'
        manifest_path.write_bytes(manifest_path.read_bytes() + b'\n')
        before = contents(self.remote)
        with self.assertRaisesRegex(ValueError, 'review baseline changed'):
            self.sync(estimate=estimate)
        self.assertEqual(contents(self.remote), before)

    def test_unsafe_manifest_path_blocks_review(self):
        version = Path(self.sync()['path'])
        path = version / 'manifest.json'
        manifest = _read_json(path)
        manifest['files']['../private.txt'] = next(iter(manifest['files'].values()))
        _write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, 'Invalid file'):
            self.estimate()

    def test_grouping_supported_annotation_formats_and_split_move(self):
        for storage in ('json', 'png', 'bbox', 'pkl', 'frames'):
            self.assertEqual(file_context(f'annotations/{storage}/video/frame_000000.png')[0], 'video')
        self.assertEqual(file_context('annotations/png/video/video_annotations_hive.png'),
                         ('video', 'Shared across video'))
        self.assertEqual(file_context('input_data/val/video.mp4'), ('video', 'Original video'))
        self.assertEqual(file_context('frames/video/video_metadata.json'), ('video', 'Video metadata'))
        self.assertIsNone(file_context('tracking_sequences.json')[0])


if __name__ == '__main__':
    unittest.main()
