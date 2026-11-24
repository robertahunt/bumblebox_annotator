"""Automatic intake of contributors' published work, on disposable projects and a disposable shared folder."""

import json
import unittest

from core.annotation import AnnotationManager
from core.assigned_work import publish_work, read_assignments, save_assignments
from core.assigned_workspace import PreparedAssignedDownload
from core.import_review import validate_coco_reviews
from core.intake import (CHANGED_HERE, REVIEW_OPEN, STATE, PreparedIntake, accept_work, held_categories, held_work, is_held,
                         prune_bundles, read_state, real_changes, release)
from core.project_delta import rollback_bundles
from core.project_sync import _inventory
from core.submission_review import PreparedSubmissionReview, approve_review, review_queue
from training.coco_video_export import export_coco_per_video
from scripts import test_assigned_work as fixture  # the module, so its own tests are not run again here


class IntakeTests(unittest.TestCase):
    setUp = fixture.AssignedWorkTests.setUp
    view = fixture.AssignedWorkTests.view
    download = fixture.AssignedWorkTests.download
    status = fixture.AssignedWorkTests.status

    def edit(self, project, video='alpha', index=0, actor=None, pixel=(0, 0)):
        manager = AnnotationManager()
        manager.contributor_session = actor or self.alex
        annotations = manager.load_frame_annotations(project, video, index)
        annotations[0]['mask'][pixel] = 255
        manager.save_frame_annotations(project, video, index, annotations)

    def edit_hive(self, project, video='alpha', actor=None):
        manager = AnnotationManager()
        manager.contributor_session = actor or self.alex
        annotations, _ = manager.load_video_annotations(project, video)
        annotations[0]['mask'][3, 3] = 255
        manager.save_video_annotations(project, video, annotations)

    def publish(self, video='alpha', actor=None, name='working', ready=False, pixel=(0, 0), hive=False):
        """A contributor edits an assigned video in their copy and uploads it. Returns the copy."""
        actor = actor or self.alex
        local = self.base / name
        if not local.exists():
            self.download(actor, [video], name)
        if hive:
            self.edit_hive(local, video, actor)
        else:
            self.edit(local, video, 0 if video == 'alpha' else 1, actor, pixel)
        publish_work(local, self.configuration, actor, ready=[video] if ready else [])
        return local

    def plan(self, **options):
        return PreparedIntake(self.source, self.configuration, self.coordinator, **options)

    def intake(self, **options):
        return self.plan(**options).apply()

    def files(self, project, video='alpha', frame=0):
        names = [f'annotations/json/{video}/frame_{frame:06d}.json', f'annotations/png/{video}/frame_{frame:06d}.png']
        return [(project / name).read_bytes() for name in names]

    def bundles(self):
        return rollback_bundles(self.source, self.configuration)

    def test_unfinished_uploaded_work_is_taken_in_and_held(self):
        theirs = self.publish()
        untouched = self.files(self.source, 'alpha', 1) + self.files(self.source, 'beta', 1)
        before = self.files(self.source)
        result = self.intake()
        self.assertTrue(result['updated'])
        self.assertEqual(result['taken'], [{'video': 'alpha', 'contributor': 'Alex'}])
        self.assertEqual(result['skipped'], {})
        self.assertEqual(self.files(self.source), self.files(theirs))
        self.assertNotEqual(self.files(self.source), before)
        self.assertEqual(self.files(self.source, 'alpha', 1) + self.files(self.source, 'beta', 1), untouched)
        self.assertEqual(self.status()[0], 'In Progress')
        self.assertEqual(held_work(self.source), {'alpha': {'frames': [0], 'shared': []}})
        bundle = self.bundles()[0]
        kept = bundle['path'] / 'before/annotations/png/alpha/frame_000000.png'
        self.assertEqual(kept.read_bytes(), before[1])
        self.assertNotIn(STATE, _inventory(self.source, False))  # the record is never published

    def test_a_run_with_nothing_new_writes_nothing(self):
        self.publish()
        self.intake()
        record = (self.source / STATE).read_bytes()
        plan = self.plan()
        self.assertEqual((plan.taken, plan.skipped), ([], {}))
        self.assertFalse(plan.apply()['updated'])
        self.assertEqual(len(self.bundles()), 1)
        self.assertEqual((self.source / STATE).read_bytes(), record)

    def test_nothing_is_recorded_before_anyone_uploads(self):
        self.assertFalse(self.intake()['updated'])
        self.assertFalse((self.source / STATE).exists())
        self.assertEqual(held_work(self.source), {})

    def test_their_next_upload_replaces_the_version_taken_in_before(self):
        theirs = self.publish()
        self.intake()
        self.publish(pixel=(1, 1))
        self.assertEqual([row['video'] for row in self.intake()['taken']], ['alpha'])
        self.assertEqual(self.files(self.source), self.files(theirs))

    def test_annotations_changed_here_are_never_written_over(self):
        self.publish()
        self.edit(self.source, actor=self.coordinator, pixel=(2, 2))
        mine = self.files(self.source)
        result = self.intake()
        self.assertFalse(result['updated'])
        self.assertEqual(result['skipped'], {'alpha': CHANGED_HERE})
        self.assertEqual(self.files(self.source), mine)
        self.assertEqual(self.bundles(), [])
        self.assertEqual(read_state(self.source)['videos']['alpha']['skipped']['reason'], CHANGED_HERE)
        self.assertEqual(self.plan().skipped, {'alpha': CHANGED_HERE})  # and it is reported again next time

    def test_a_change_here_after_an_intake_blocks_their_next_upload(self):
        self.publish()
        self.intake()
        self.edit(self.source, actor=self.coordinator, pixel=(2, 2))
        mine = self.files(self.source)
        self.publish(pixel=(1, 1))
        self.assertEqual(self.intake()['skipped'], {'alpha': CHANGED_HERE})
        self.assertEqual(self.files(self.source), mine)

    def test_the_director_can_choose_their_version_and_the_local_one_is_kept_for_rollback(self):
        theirs = self.publish()
        self.edit(self.source, actor=self.coordinator, pixel=(2, 2))
        mine = self.files(self.source)
        result = self.intake(replace={'alpha'})
        self.assertEqual([row['video'] for row in result['taken']], ['alpha'])
        self.assertEqual(self.files(self.source), self.files(theirs))
        kept = self.bundles()[0]['path'] / 'before/annotations/png/alpha/frame_000000.png'
        self.assertEqual(kept.read_bytes(), mine[1])
        self.assertIsNone(read_state(self.source)['videos']['alpha']['skipped'])

    def test_the_director_can_keep_the_local_version_until_they_upload_again(self):
        self.publish()
        self.edit(self.source, actor=self.coordinator, pixel=(2, 2))
        mine = self.files(self.source)
        self.assertEqual(self.intake(keep={'alpha'})['skipped'], {})
        self.assertEqual(self.plan().skipped, {})
        self.assertEqual(self.files(self.source), mine)
        self.publish(pixel=(1, 1))
        self.assertEqual(self.plan().skipped, {'alpha': CHANGED_HERE})

    def test_changes_here_outside_the_assigned_frames_do_not_block_intake(self):
        self.edit(self.source, 'alpha', 1, self.coordinator, (2, 2))  # alpha's assignment is frame 0 only
        mine = self.files(self.source, 'alpha', 1)
        theirs = self.publish()
        self.assertEqual([row['video'] for row in self.intake()['taken']], ['alpha'])
        self.assertEqual(self.files(self.source), self.files(theirs))
        self.assertEqual(self.files(self.source, 'alpha', 1), mine)

    def test_a_review_opens_without_a_conflict_after_an_intake(self):
        self.publish()
        self.intake()
        theirs = self.publish(pixel=(1, 1), ready=True)
        row = review_queue(self.source, self.configuration, self.coordinator)[0]
        plan = PreparedSubmissionReview(self.source, self.configuration, self.coordinator,
                                        row['video'], row['revision'], row['id'])
        self.assertEqual(plan.conflicts, [])
        plan.apply()
        self.assertEqual(self.files(self.source), self.files(theirs))

    def test_an_open_review_is_left_alone(self):
        self.publish(ready=True)
        row = review_queue(self.source, self.configuration, self.coordinator)[0]
        PreparedSubmissionReview(self.source, self.configuration, self.coordinator,
                                 row['video'], row['revision'], row['id']).apply()
        self.assertEqual(held_work(self.source), {})
        self.assertFalse(self.intake()['updated'])
        # In the project but not approved: held, and since what it changed is not known here, all of it.
        self.assertEqual(held_work(self.source),
                         {'alpha': {'frames': [0], 'shared': ['chamber', 'hive', 'pollen']}})
        self.edit(self.source, actor=self.coordinator, pixel=(2, 2))  # a correction made during the review
        mine = self.files(self.source)
        self.publish(pixel=(1, 1), ready=True)
        self.assertEqual(self.intake()['skipped'], {'alpha': REVIEW_OPEN})
        self.assertEqual(self.files(self.source), mine)

    def approve(self, correct=False):
        row = review_queue(self.source, self.configuration, self.coordinator)[0]
        PreparedSubmissionReview(self.source, self.configuration, self.coordinator,
                                 row['video'], row['revision'], row['id']).apply()
        if correct:
            self.edit(self.source, actor=self.coordinator, pixel=(2, 2))
        approve_review(self.source, self.configuration, self.coordinator, row['video'], row['revision'], row['id'])

    def test_approving_releases_the_hold_and_the_accepted_version_stays(self):
        self.publish(ready=True)
        self.intake()
        self.assertEqual(held_work(self.source), {'alpha': {'frames': [0], 'shared': []}})
        self.approve(correct=True)
        self.assertEqual(self.status()[0], 'Accepted')
        self.assertEqual(held_work(self.source), {})
        corrected = self.files(self.source)
        plan = self.plan()
        self.assertEqual((plan.taken, plan.skipped), ([], {}))
        plan.apply()
        self.assertEqual(self.files(self.source), corrected)
        self.assertEqual(held_work(self.source), {})

    def test_a_version_accepted_before_intake_existed_keeps_its_corrections(self):
        theirs = self.publish(ready=True)
        self.approve(correct=True)
        (self.source / STATE).unlink()  # as in a project whose reviews were all done before this feature
        corrected = self.files(self.source)
        self.assertNotEqual(corrected, self.files(theirs))
        plan = self.plan()
        self.assertEqual((plan.taken, plan.skipped), ([], {}))
        plan.apply()
        self.assertEqual(self.files(self.source), corrected)
        self.assertEqual(held_work(self.source), {})

    def test_accepting_with_the_older_button_also_releases_the_hold(self):
        from core.assigned_work import accept_submission
        self.publish(ready=True)
        self.intake()
        self.assertEqual(sorted(held_work(self.source)), ['alpha'])
        accept_submission(self.source, self.configuration, self.coordinator, 'alpha',
                          self.status()[1])
        self.assertEqual(self.status()[0], 'Accepted')
        self.intake()
        self.assertEqual(held_work(self.source), {})

    def test_work_continued_from_an_accepted_version_is_taken_in_and_held_again(self):
        working = self.publish(ready=True)
        self.intake()
        self.approve(correct=True)
        # They pick up the director's corrections in the copy they already have.
        PreparedAssignedDownload(self.configuration, self.project_id, self.alex, ('alpha',), working).apply()
        self.assertEqual(self.files(working), self.files(self.source))
        self.publish(pixel=(1, 1))
        self.assertNotEqual(self.status()[0], 'Accepted')  # their newer version has not been accepted
        self.assertEqual([row['video'] for row in self.intake()['taken']], ['alpha'])
        self.assertEqual(self.files(self.source), self.files(working))
        self.assertEqual(held_work(self.source), {'alpha': {'frames': [0], 'shared': []}})

    def test_several_submitted_videos_are_accepted_at_once(self):
        self.publish(ready=True)
        self.publish('beta', self.sam, 'sam-copy', ready=True)
        self.assertEqual(len(self.intake()['taken']), 2)
        self.assertEqual(held_work(self.source), {'alpha': {'frames': [0], 'shared': []},
                                                  'beta': {'frames': [1], 'shared': []}})  # the frames they edited
        accepted, refused = accept_work(self.source, self.configuration, self.coordinator, ['alpha', 'beta'])
        self.assertEqual((accepted, refused), (['alpha', 'beta'], {}))
        self.assertEqual((self.status('alpha')[0], self.status('beta')[0]), ('Accepted', 'Accepted'))
        self.assertEqual(held_work(self.source), {})
        self.assertFalse(self.intake()['updated'])

    def test_accepting_at_once_refuses_what_differs_here_or_is_not_submitted(self):
        self.publish(ready=True)
        self.publish('beta', self.sam, 'sam-copy')  # uploaded, not submitted
        self.intake()
        self.edit(self.source, actor=self.coordinator, pixel=(2, 2))
        accepted, refused = accept_work(self.source, self.configuration, self.coordinator, ['alpha', 'beta'])
        self.assertEqual(accepted, [])
        self.assertIn('differs here', refused['alpha'])
        self.assertIn('not waiting for review', refused['beta'])
        self.assertEqual(self.status('alpha')[0], 'Ready for Review')
        self.assertEqual(sorted(held_work(self.source)), ['alpha', 'beta'])
        with self.assertRaisesRegex(ValueError, 'director'):
            accept_work(self.source, self.configuration, self.alex, ['alpha'])

    def test_a_changed_video_wide_mask_holds_that_mask_and_no_frame(self):
        self.publish(hive=True)
        self.intake()
        held = held_work(self.source)
        self.assertEqual(held, {'alpha': {'frames': [], 'shared': ['hive']}})
        self.assertEqual(held_categories(held, 'alpha'), {'hive'})
        self.assertFalse(is_held(held, 'alpha', 0))
        self.assertEqual(held_categories(held, 'beta'), set())

    def test_a_re_save_that_changes_no_annotation_is_copied_but_not_held(self):
        working = self.download(self.alex, ['alpha'], 'working')
        name = 'annotations/json/alpha/frame_000000.json'
        rows = json.loads((working / name).read_text())
        for row in rows:  # what a newer app adds when it saves a frame someone only looked at
            row['provenance'] = dict(row.get('provenance') or {}, last_modified_at='2026-11-20T20:18:00+00:00')
        (working / name).write_text(json.dumps(rows))
        publish_work(working, self.configuration, self.alex)
        self.assertEqual([row['video'] for row in self.intake()['taken']], ['alpha'])
        self.assertEqual((self.source / name).read_bytes(), (working / name).read_bytes())
        self.assertEqual(held_work(self.source), {})

    def test_only_real_annotation_differences_count_as_changes(self):
        import cv2
        import numpy as np

        def png(array):
            return cv2.imencode('.png', array)[1].tobytes()

        box = {'mask_id': 1, 'bbox': [1, 2, 3, 4], 'category_id': 1}
        mask, other = np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)
        mask[1, 1], other[2, 2] = 1, 1
        hive = {'category': 'hive', 'mask_id': 5, 'area': 10}
        mine = {
            'annotations/json/v/frame_000000.json': None,                                   # they opened the frame
            'annotations/bbox/v/frame_000001.json': json.dumps([box]).encode(),             # re-saved with bookkeeping
            'annotations/png/v/frame_000002.png': png(mask),                                # the same pixels
            'annotations/bbox/v/frame_000003.json': json.dumps([box]).encode(),             # a box moved
            'annotations/png/v/frame_000004.png': png(mask),                                # a mask redrawn
            'annotations/work/v/frame_000005.json': b'{}',                                  # marked finished
            'annotations/json/v/video_annotations.json': json.dumps({'annotations': [hive]}).encode(),
            'annotations/png/v/video_annotations_hive.png': png(mask),
            'annotations/png/v/video_annotations_pollen.png': None,                         # an empty mask appears
            'annotations/png/v/frame_000006.png': b'not an image',                          # unreadable: assume changed
            # An older save of a mask row, re-saved by a newer app with derived fields and a derived box file.
            'annotations/json/v/frame_000007.json': json.dumps(
                [{'mask_id': 1, 'category_id': 1, 'area': 5, 'label': 'instance_1'}]).encode(),
            'annotations/bbox/v/frame_000007.json': None,
            'annotations/bbox/v/frame_000008.json': json.dumps([dict(box, bbox_only=True)]).encode(),  # a box deleted
        }
        theirs = dict(mine)
        theirs.update({
            'annotations/json/v/frame_000000.json': b'[]',
            'annotations/bbox/v/frame_000001.json': json.dumps(
                [dict(box, provenance={'created_by': None}, category='bee')]).encode(),
            'annotations/png/v/frame_000002.png': cv2.imencode('.png', mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes(),
            'annotations/bbox/v/frame_000003.json': json.dumps([dict(box, bbox=[9, 9, 3, 4])]).encode(),
            'annotations/png/v/frame_000004.png': png(other),
            'annotations/work/v/frame_000005.json': b'{"version": 1}',
            'annotations/json/v/video_annotations.json': json.dumps(
                {'annotations': [dict(hive, area=12, provenance={'created_by': 'x'})]}).encode(),
            'annotations/png/v/video_annotations_hive.png': png(other),
            'annotations/png/v/video_annotations_pollen.png': png(np.zeros((4, 4), np.uint8)),
            'annotations/png/v/frame_000006.png': png(mask),
            'annotations/json/v/frame_000007.json': json.dumps(
                [{'mask_id': 1, 'category_id': 1, 'area': 5, 'label': 'instance_1', 'bbox': [0, 0, 2, 2],
                  'mask_coco_rle': {'size': [4, 4], 'counts': 'abc'}}]).encode(),
            'annotations/bbox/v/frame_000007.json': json.dumps(
                [{'mask_id': 1, 'category_id': 1, 'bbox': [0, 0, 2, 2], 'from_mask': True, 'source': 'mask'}]).encode(),
            'annotations/bbox/v/frame_000008.json': b'[]',
        })
        frames, shared = real_changes(sorted(mine), mine.get, theirs.get)
        self.assertEqual((frames, shared), ({3, 4, 6, 8}, {'hive'}))
        legacy = 'annotations/png/v/video_annotations.png'  # one file for every video-wide mask: any of them may differ
        self.assertEqual(real_changes([legacy], {legacy: png(mask)}.get, {legacy: png(other)}.get),
                         (set(), {'chamber', 'hive', 'pollen'}))

    def test_a_changed_assignment_keeps_what_was_held_until_it_is_released(self):
        self.publish()
        self.intake()
        view = self.view()
        choices = dict(self.choices, alpha={'contributor': self.alex, 'frames': [0], 'task': 'bee'})
        save_assignments(self.source, self.configuration, self.coordinator, choices,
                         read_assignments(view)['revision'])
        self.intake()
        self.assertEqual(held_work(self.source), {'alpha': {'frames': [0], 'shared': []}})
        release(self.source, ['alpha'])
        self.assertEqual(held_work(self.source), {})

    def test_only_the_director_runs_intake_and_only_in_the_full_project(self):
        working = self.publish()
        with self.assertRaisesRegex(ValueError, 'director'):
            PreparedIntake(self.source, self.configuration, self.alex)
        with self.assertRaises(ValueError):
            PreparedIntake(working, self.configuration, self.coordinator)

    def test_only_the_newest_intake_bundles_are_kept(self):
        self.publish()
        self.intake()
        self.publish(pixel=(1, 1))
        self.intake()
        self.assertEqual(len(self.bundles()), 2)
        newest = self.bundles()[0]['path']
        prune_bundles(self.source, self.configuration, keep=1)
        self.assertEqual([bundle['path'] for bundle in self.bundles()], [newest])

    def export(self):
        """Export both videos; return ({video: exported frame numbers}, frames left out as held, files)."""
        held_out = []
        paths = export_coco_per_video(self.source, ['alpha', 'beta'], 'train', class_names=['bee', 'hive'],
                                      held_out=held_out)
        frames = {path.stem: sorted(image['frame_index'] for image in json.loads(path.read_text())['images'])
                  for path in paths}
        return frames, held_out, paths

    def hive_annotations(self, paths, video):
        coco = json.loads(next(path for path in paths if path.stem == video).read_text())
        hive = {category['id'] for category in coco['categories'] if category['name'] == 'hive'}
        return sum(1 for ann in coco['annotations'] if ann['category_id'] in hive)

    def test_held_frames_are_left_out_of_export_until_the_work_is_accepted(self):
        everything = {'alpha': [0, 1, 2], 'beta': [0, 1, 2]}
        self.assertEqual(self.export()[:2], (everything, []))
        self.publish(ready=True)
        self.intake()
        frames, held_out, _ = self.export()
        self.assertEqual(frames, {'alpha': [1, 2], 'beta': [0, 1, 2]})
        self.assertEqual(held_out, [('frame', 'alpha', 0)])
        accept_work(self.source, self.configuration, self.coordinator, ['alpha'])
        self.assertEqual(self.export()[:2], (everything, []))

    def test_a_held_video_wide_mask_is_left_out_of_export_and_its_frames_are_not(self):
        everything = {'alpha': [0, 1, 2], 'beta': [0, 1, 2]}
        _, _, paths = self.export()
        self.assertEqual(self.hive_annotations(paths, 'alpha'), 3)  # one per exported frame
        self.publish(hive=True)
        self.intake()
        frames, held_out, paths = self.export()
        self.assertEqual(frames, everything)  # the bee frames still go out
        self.assertEqual(held_out, [('shared', 'alpha', 'hive')])
        self.assertEqual((self.hive_annotations(paths, 'alpha'), self.hive_annotations(paths, 'beta')), (0, 3))

    def test_an_export_made_before_the_intake_cannot_carry_a_held_video_wide_mask_into_training(self):
        _, _, cached = self.export()
        self.publish(hive=True)
        self.intake()
        with self.assertRaisesRegex(ValueError, 'not been accepted'):
            validate_coco_reviews(self.source, cached)
        validate_coco_reviews(self.source, self.export()[2])

    def test_an_export_made_before_the_intake_cannot_carry_held_work_into_training(self):
        _, _, cached = self.export()
        validate_coco_reviews(self.source, cached)
        self.publish()
        self.intake()
        with self.assertRaisesRegex(ValueError, 'not been accepted'):
            validate_coco_reviews(self.source, cached)
        validate_coco_reviews(self.source, self.export()[2])


if __name__ == '__main__':
    unittest.main()
