"""Director review safety on disposable local projects and shared archives."""

import unittest
from pathlib import Path
from unittest.mock import patch

from core.annotation import AnnotationManager
from core.assigned_work import (latest_snapshot, publish_work, read_assignments, save_assignments,
                                shared_assignment_progress)
from core.project_delta import file_record, rollback_update
from core.project_sync import SyncConflictError, _read_json, _write_json, restore_project
from core.submission_review import PreparedSubmissionReview, approve_review, review_queue
from core.submission_comparison import comparison_preview
from scripts.test_assigned_work import AssignedWorkTests
from scripts.test_project_sync import annotation


class SubmissionReviewTests(unittest.TestCase):
    setUp = AssignedWorkTests.setUp
    view = AssignedWorkTests.view
    download = AssignedWorkTests.download
    change = AssignedWorkTests.change
    status = AssignedWorkTests.status

    def submit(self, video='alpha', actor=None, name='working'):
        actor = actor or self.alex
        local = self.download(actor, [video], name)
        self.change(local, video, 0 if video == 'alpha' else 1, actor)
        publish_work(local, self.configuration, actor, ready=[video])
        return local

    def queue(self):
        return review_queue(self.source, self.configuration, self.coordinator)

    def prepare(self, row=None):
        row = row or self.queue()[0]
        return PreparedSubmissionReview(self.source, self.configuration, self.coordinator,
            row['video'], row['revision'], row['id'])

    def approve(self, row):
        return approve_review(self.source, self.configuration, self.coordinator,
            row['video'], row['revision'], row['id'])

    def test_queue_only_ready_and_requires_director(self):
        self.submit()
        self.assertEqual([r['video'] for r in self.queue()], ['alpha'])
        self.assertEqual(self.queue()[0]['contributor']['name'], 'Alex')
        with self.assertRaisesRegex(ValueError, 'director'):
            review_queue(self.source, self.configuration, self.sam)

    def test_open_exact_submission_preserves_other_edits_and_supports_scoped_rollback(self):
        local = self.submit()
        self.change(self.source, 'beta', 2, self.coordinator)
        unrelated = self.source / 'annotations/json/beta/frame_000002.json'
        before = unrelated.read_bytes()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        old_target = target.read_bytes()
        result = self.prepare().apply()
        self.assertEqual(target.read_bytes(), (local / target.relative_to(self.source)).read_bytes())
        self.assertEqual(unrelated.read_bytes(), before)
        # A later edit to an unrelated video must not prevent this scoped rollback.
        unrelated.write_bytes(before + b'\n')
        rollback_update(self.source, result['backup'], self.configuration)
        self.assertEqual(target.read_bytes(), old_target)
        self.assertEqual(unrelated.read_bytes(), before + b'\n')

    def test_local_conflict_is_not_overwritten(self):
        self.submit()
        path = self.source / 'annotations/json/alpha/frame_000000.json'
        path.write_bytes(path.read_bytes() + b'\n')
        before = path.read_bytes()
        plan = self.prepare()
        self.assertIn(path.relative_to(self.source).as_posix(), plan.conflicts)
        with self.assertRaisesRegex(SyncConflictError, 'Choose local or submitted'):
            plan.apply()
        self.assertEqual(path.read_bytes(), before)

    def test_ready_discovery_does_not_validate_or_print_unsubmitted_videos(self):
        self.submit()
        messages = []
        from core.submission_review import assignment_status
        with patch('core.submission_review.assignment_status', wraps=assignment_status) as status:
            rows = review_queue(self.source, self.configuration, self.coordinator, messages.append)
        self.assertEqual([r['video'] for r in rows], ['alpha'])
        self.assertEqual([call.args[1]['video'] for call in status.call_args_list], ['alpha'])
        self.assertFalse(any('beta' in message for message in messages))

    def test_keep_local_conflict_retains_backup_and_can_be_approved(self):
        self.submit()
        self.change(self.source, 'alpha', 0, self.coordinator)
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        before = target.read_bytes()
        row = self.queue()[0]
        plan = self.prepare(row)
        self.assertTrue(plan.conflicts)
        result = plan.apply(conflict_choice='local')
        self.assertEqual(target.read_bytes(), before)
        published = self.approve(row)
        manifest = _read_json(published['path'] + '/manifest.json')
        self.assertEqual(manifest['submission_reviews']['alpha']['conflict_resolution']['choice'], 'local')
        self.assertEqual(self.status()[0], 'Accepted')
        self.assertTrue((Path(result['backup']) / 'before' / target.relative_to(self.source)).exists())

    def test_use_submitted_conflict_preserves_complete_local_set_for_rollback(self):
        submitted = self.submit()
        self.change(self.source, 'alpha', 0, self.coordinator)
        manager = AnnotationManager()
        manager.contributor_session = self.coordinator
        manager.save_video_annotations(self.source, 'alpha', [annotation(9, 'hive')])
        names = ['annotations/json/alpha/frame_000000.json', 'annotations/png/alpha/frame_000000.png',
                 'annotations/json/alpha/video_annotations.json', 'annotations/png/alpha/video_annotations_hive.png']
        before = {name: (self.source / name).read_bytes() for name in names}
        plan = self.prepare()
        result = plan.apply(conflict_choice='submitted')
        for name in names:
            self.assertEqual((self.source / name).read_bytes(), (submitted / name).read_bytes())
        rollback_update(self.source, result['backup'], self.configuration)
        self.assertEqual(before, {name: (self.source / name).read_bytes() for name in names})

    def test_local_choice_does_not_mix_nonconflicting_submitted_files(self):
        local = self.download()
        self.change(local, 'alpha', 0, self.alex)
        manager = AnnotationManager()
        manager.contributor_session = self.alex
        manager.save_video_annotations(local, 'alpha', [annotation(9, 'hive')])
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        self.change(self.source, 'alpha', 0, self.coordinator)
        shared = self.source / 'annotations/json/alpha/video_annotations.json'
        before = shared.read_bytes()
        self.prepare().apply(conflict_choice='local')
        self.assertEqual(shared.read_bytes(), before)

    def test_conflict_choice_cannot_overwrite_changes_made_after_preview(self):
        self.submit()
        self.change(self.source, 'alpha', 0, self.coordinator)
        plan = self.prepare()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        target.write_bytes(target.read_bytes() + b'\n')
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Local annotations changed'):
            plan.apply(conflict_choice='submitted')
        self.assertEqual(target.read_bytes(), before)

    def test_superseded_unfinished_review_offers_comparison(self):
        local = self.submit()
        self.prepare().apply()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        target.write_bytes(target.read_bytes() + b'\n')
        other = local / target.relative_to(self.source)
        other.write_bytes(other.read_bytes() + b'\n\n')
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        plan = self.prepare()
        self.assertTrue(plan.conflicts)
        plan.apply(conflict_choice='local')
        self.assertEqual(target.read_bytes(), other.read_bytes()[:-1])

    def test_comparison_is_read_only_shows_distinct_masks_and_shared_layers(self):
        self.submit()
        manager = AnnotationManager()
        manager.contributor_session = self.coordinator
        hive = annotation(5, 'hive')
        hive['mask'][20:24, 20:24] = 255
        manager.save_video_annotations(self.source, 'alpha', [hive])
        plan = self.prepare()
        before = {n: (self.source / n).read_bytes() for n in plan.local}
        preview = comparison_preview(plan, 0, ('hive',))
        self.assertEqual(preview['image'].shape, (32, 40, 3))
        self.assertEqual(preview['local']['overlay'][21, 21, 3], 255)
        self.assertEqual(preview['submitted']['overlay'][21, 21, 3], 0)
        self.assertEqual(preview['local']['counts'], {'hive': 1})
        self.assertEqual(before, {n: (self.source / n).read_bytes() for n in plan.local})
        with self.assertRaisesRegex(ValueError, 'assigned frame'):
            comparison_preview(plan, 2)

    def test_preview_handles_empty_local_version_and_detects_damaged_masks(self):
        self.submit()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        target.unlink()
        (self.source / 'annotations/png/alpha/frame_000000.png').unlink()
        (self.source / 'annotations/bbox/alpha/frame_000000.json').unlink(missing_ok=True)
        plan = self.prepare()
        preview = comparison_preview(plan, 0, ('bee',))
        self.assertEqual(preview['local']['counts'], {})
        self.assertTrue(preview['submitted']['counts']['bee'])
        plan.apply(conflict_choice='submitted')
        target = self.source / 'annotations/png/alpha/frame_000000.png'
        plan = self.prepare()
        target.write_bytes(b'damaged')
        preview = comparison_preview(plan, 0)
        self.assertIn('error', preview['local'])
        self.assertNotIn('error', preview['submitted'])

    def test_preview_preserves_holes_and_overlapping_instance_rle(self):
        self.submit()
        manager = AnnotationManager()
        manager.contributor_session = self.coordinator
        bee, pollen = annotation(1), annotation(2, 'pollen')
        bee['mask'][6:8, 9:11] = 0
        manager.save_frame_annotations(self.source, 'alpha', 0, [bee, pollen])
        preview = comparison_preview(self.prepare(), 0, ('bee',))
        self.assertEqual(preview['local']['overlay'][6, 9, 3], 0)
        # The later pollen instance covers this pixel in the combined ID PNG.
        self.assertEqual(preview['local']['overlay'][5, 8, 3], 255)

    def test_approve_corrected_submission_is_atomic_and_preserves_other_videos(self):
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        target.write_bytes(target.read_bytes() + b'\n')
        self.change(self.source, 'beta', 2, self.coordinator)
        current, _ = latest_snapshot(self.view())
        old_beta = {n: r for n, r in current['manifest']['files'].items() if '/beta/' in n}
        result = self.approve(row)
        manifest = _read_json(result['path'] + '/manifest.json')
        self.assertEqual(manifest['files'][target.relative_to(self.source).as_posix()], file_record(target))
        self.assertEqual({n: r for n, r in manifest['files'].items() if '/beta/' in n}, old_beta)
        self.assertEqual(self.status()[0], 'Accepted')
        self.assertEqual(self.queue(), [])
        self.assertEqual(manifest['submission_reviews']['alpha']['submission'], [self.alex['id'], row['revision']])
        restored = self.base / 'reviewed-copy'
        restore_project(result['path'], restored)
        self.assertEqual((restored / target.relative_to(self.source)).read_bytes(), target.read_bytes())
        self.assertTrue(self.approve(row)['unchanged'])

    def test_reopening_review_keeps_corrections(self):
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        target.write_bytes(target.read_bytes() + b'\n')
        before = target.read_bytes()
        result = self.prepare(row).apply()
        self.assertTrue(result['resuming'])
        self.assertEqual(target.read_bytes(), before)

    def test_no_open_no_approval_and_cancellation_does_not_publish(self):
        self.submit()
        row = self.queue()[0]
        heads = self.view().heads()
        with self.assertRaisesRegex(ValueError, 'Open this exact'):
            self.approve(row)
        with self.assertRaises(InterruptedError):
            PreparedSubmissionReview(self.source, self.configuration, self.coordinator,
                row['video'], row['revision'], row['id'], cancelled=lambda: True)
        with self.assertRaises(InterruptedError):
            self.prepare(row).apply(cancelled=lambda: True)
        self.prepare(row).apply()
        with self.assertRaises(InterruptedError):
            approve_review(self.source, self.configuration, self.coordinator,
                row['video'], row['revision'], row['id'], cancelled=lambda: True)
        self.assertEqual(self.view().heads(), heads)
        self.assertEqual(self.status()[0], 'Ready for Review')

    def test_changed_submission_and_reassignment_block_approval(self):
        local = self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        path = local / 'annotations/json/alpha/frame_000000.json'
        path.write_bytes(path.read_bytes() + b'\n')
        publish_work(local, self.configuration, self.alex, ready=['alpha'])
        heads = self.view().heads()
        with self.assertRaisesRegex(ValueError, 'submission changed'):
            self.approve(row)
        self.choices['alpha']['contributor'] = self.sam
        data = read_assignments(self.view())
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        with self.assertRaisesRegex(ValueError, 'assignment changed'):
            self.approve(row)
        self.assertEqual(self.view().heads(), heads)

    def test_approval_keeps_newly_uploaded_other_video_and_updates_counts(self):
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        self.submit('beta', self.sam, 'sam-work')
        self.approve(row)
        self.assertEqual(self.status('beta')[0], 'Ready for Review')
        result = shared_assignment_progress(self.configuration, self.project_id, self.coordinator, {})
        self.assertEqual([r['video'] for r in result['review_ready']], ['beta'])

    def test_changed_review_frame_and_policy_are_rejected(self):
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        image = self.source / 'frames/alpha/frame_000000.jpg'
        original = image.read_bytes()
        image.write_bytes(original + b'changed')
        with self.assertRaisesRegex(SyncConflictError, 'review frame'):
            self.approve(row)
        image.write_bytes(original)
        policy = self.source / 'annotations/project.json'
        info = _read_json(policy)
        info['annotation_policy']['categories']['bee']['meaning'] = 'full_extent'
        _write_json(policy, info)
        with self.assertRaisesRegex(SyncConflictError, 'policy differs'):
            self.approve(row)

    def test_failed_pointer_update_leaves_submission_ready(self):
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        original = _write_json
        def fail_pointer(path, value):
            if path.name == 'latest.json':
                raise OSError('disconnected')
            original(path, value)
        with patch('core.submission_review._write_json', side_effect=fail_pointer):
            with self.assertRaises(OSError):
                self.approve(row)
        self.assertEqual(self.status()[0], 'Ready for Review')
        self.approve(row)
        self.assertEqual(self.status()[0], 'Accepted')

    def test_shared_mask_correction_is_published_with_reviewer_attribution(self):
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        manager = AnnotationManager()
        manager.contributor_session = self.coordinator
        hive = annotation(5, 'hive')
        hive['mask'][2:4, 2:4] = 255
        manager.save_video_annotations(self.source, 'alpha', [hive])
        result = self.approve(row)
        restored = self.base / 'reviewed-shared-mask'
        restore_project(result['path'], restored)
        labels, _ = manager.load_video_annotations(restored, 'alpha')
        self.assertEqual(labels[0]['mask'][2, 2], 255)
        self.assertEqual(labels[0]['provenance']['last_modified_by']['id'], self.coordinator['id'])
        self.assertEqual(self.status()[0], 'Accepted')

    def test_approval_survives_later_multi_video_upload_with_corrected_ready_receipt(self):
        data = read_assignments(self.view())
        self.choices['beta']['contributor'] = self.alex
        save_assignments(self.source, self.configuration, self.coordinator, self.choices, data['revision'])
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        target.write_bytes(target.read_bytes() + b'\n')
        self.approve(row)
        updated = self.download(self.alex, ['alpha', 'beta'], 'both')
        self.change(updated, 'beta', 1, self.alex)
        publish_work(updated, self.configuration, self.alex, ready=['beta'])
        self.assertEqual(self.status('alpha')[0], 'Accepted')
        self.assertEqual(self.status('beta')[0], 'Ready for Review')
        # A genuine later edit is a new submission, not covered by the old approval.
        path = updated / 'annotations/json/alpha/frame_000000.json'
        path.write_bytes(path.read_bytes() + b'\n')
        publish_work(updated, self.configuration, self.alex, ready=['alpha'])
        self.assertEqual(self.status('alpha')[0], 'Ready for Review')
        next_row = next(r for r in self.queue() if r['video'] == 'alpha')
        self.prepare(next_row).apply()
        self.approve(next_row)
        self.assertEqual(self.status('alpha')[0], 'Accepted')

    def test_missing_assigned_frame_is_copied_but_wrong_image_is_not_replaced(self):
        self.submit()
        frame = self.source / 'frames/alpha/frame_000000.jpg'
        expected = frame.read_bytes()
        frame.unlink()
        self.prepare().apply()
        self.assertEqual(frame.read_bytes(), expected)
        frame.write_bytes(b'different image')
        with self.assertRaisesRegex(SyncConflictError, 'review frame'):
            self.prepare()

    def test_concurrent_local_edit_during_open_is_not_overwritten(self):
        self.submit()
        plan = self.prepare()
        target = self.source / 'annotations/json/alpha/frame_000000.json'
        target.write_bytes(target.read_bytes() + b'\n')
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Local annotations changed'):
            plan.apply()
        self.assertEqual(target.read_bytes(), before)

    def test_lost_local_receipt_after_approval_can_be_recovered(self):
        self.submit()
        row = self.queue()[0]
        self.prepare(row).apply()
        original = _write_json
        def fail_local_receipt(path, value):
            if path.name.startswith('.submission-review-'):
                raise OSError('local disk full')
            original(path, value)
        with patch('core.submission_review._write_json', side_effect=fail_local_receipt):
            result = self.approve(row)
            retry = self.approve(row)
        self.assertIn('warning', result)
        self.assertTrue(retry['unchanged'])
        self.assertIn('warning', retry)
        self.assertEqual(self.status()[0], 'Accepted')
        heads = self.view().heads()
        self.assertTrue(self.approve(row)['unchanged'])
        self.assertEqual(self.view().heads(), heads)


if __name__ == '__main__':
    unittest.main()
