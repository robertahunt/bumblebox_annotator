"""Policy and import review safety checks using disposable project copies."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.annotation import AnnotationManager
from core.annotation_policy import (category_policies, default_policy, policy_needs_confirmation,
                                    save_policy, temporal_support)
from core.categories import CATEGORIES
from core.contributors import new_session
from core.import_review import (REVIEW_FILE, complete_review, read_reviews, require_reviewed,
                               review_rows, review_status, validate_coco_reviews)
from core.project_import import prepare_import
from core.project_manager import ProjectManager
from core.project_sync import (_read_json, _write_json, test_destination as probe_destination,
                               restore_project, sync_project)
from scripts.test_project_import import mask_annotation
from training.coco_video_export import export_coco_per_video, export_coco_with_tracking


class PolicyFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.dest = self.root / 'source', self.root / 'dest'
        self.source_info = ProjectManager().create_project(self.source, 'Source', hive_annotation_scope='frame')
        self.dest_info = ProjectManager().create_project(self.dest, 'Destination', hive_annotation_scope='video')
        self.manager = AnnotationManager()
        self.contributor = new_session('Reviewer')
        for project in (self.source, self.dest):
            frames = project / 'frames/video'
            frames.mkdir()
            for index in (2, 5):
                cv2.imwrite(str(frames / f'frame_{index:06d}.jpg'), np.full((64, 96, 3), index, np.uint8))
            _write_json(frames / 'video_metadata.json', {'selected_frames': [2, 5], 'split': 'train',
                                                        'total_frames': 10, 'project_import': True})
        for index in (2, 5):
            self.manager.save_frame_annotations(self.source, 'video', index,
                                                [mask_annotation(offset=index)], contributor=None)

    def import_hives(self, **options):
        plan = prepare_import(self.source, self.dest, {'video': [2, 5]}, ['hive'], **options)
        self.addCleanup(plan.close)
        plan.apply()
        return plan

    def finish(self):
        for index in (2, 5):
            complete_review(self.dest, 'video', index, self.contributor)

    def exportable(self):
        for index in (2, 5):
            self.manager.save_frame_annotations(self.dest, 'video', index,
                                                [mask_annotation(8, 'bee')], contributor=None)


class AnnotationPolicyTests(PolicyFixture):
    def test_confirmation_is_based_on_all_saved_meanings_not_dialog_history(self):
        self.assertTrue(policy_needs_confirmation({}))
        self.assertFalse(policy_needs_confirmation(self.source_info))
        info = deepcopy(self.source_info)
        info['annotation_policy']['categories']['queen_brood_late']['meaning'] = 'unconfirmed'
        self.assertTrue(policy_needs_confirmation(info))
        info['annotation_policy']['categories'].pop('nectar')
        self.assertTrue(policy_needs_confirmation({'project_info': info}))
        info['annotation_policy']['categories']['queen_brood_late']['meaning'] = 'visible'
        self.assertTrue(policy_needs_confirmation(info))

    def test_new_and_legacy_intent_are_distinct(self):
        self.assertEqual(category_policies(self.source_info)['hive'], {'scope': 'frame', 'meaning': 'visible'})
        self.assertEqual(category_policies(self.dest_info)['hive'], {'scope': 'video', 'meaning': 'full_extent'})
        legacy = category_policies({})
        self.assertEqual(set(legacy), set(CATEGORIES))
        self.assertTrue(all(p['meaning'] == 'unconfirmed' for p in legacy.values()))
        self.assertEqual(legacy['hive']['scope'], 'video')

    def test_new_annotation_manager_retains_chosen_meanings(self):
        info = deepcopy(self.source_info)
        info['annotation_policy']['categories']['nectar']['meaning'] = 'full_extent'
        self.manager.new_project(info)
        self.assertEqual(category_policies(self.manager.project_info)['nectar']['meaning'], 'full_extent')
        self.manager.save_project(self.source)
        self.manager.load_project(self.source)
        self.assertEqual(category_policies(self.manager.project_info)['nectar']['meaning'], 'full_extent')

    def test_confirm_policy_preserves_scope_masks_and_unknown_fields(self):
        path = self.source / 'annotations/project.json'
        data = _read_json(path)
        data['custom'] = {'keep': True}
        _write_json(path, {'project_info': data})
        mask = self.source / 'annotations/png/video/frame_000002.png'
        before = mask.read_bytes()
        entries = category_policies(data)
        entries['hive']['meaning'] = 'full_extent'
        updated = save_policy(self.source, entries, self.contributor)['project_info']
        self.assertEqual(updated['custom'], {'keep': True})
        self.assertEqual(mask.read_bytes(), before)
        self.assertEqual(updated['hive_annotation_scope'], 'frame')
        history = updated['annotation_policy']['history'][-1]
        self.assertEqual(history['confirmed_by']['name'], 'Reviewer')
        self.assertEqual(history['previous']['hive']['meaning'], 'visible')

    def test_scope_cannot_be_silently_rewritten(self):
        entries = category_policies(self.dest_info)
        entries['hive']['scope'] = 'frame'
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            save_policy(self.dest, entries)
        self.assertEqual(_read_json(self.dest / 'annotations/project.json'), self.dest_info)

    def test_unsupported_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'version'):
            category_policies({'annotation_policy': {'version': 99}})
        entries = category_policies(self.dest_info)
        entries['bee']['meaning'] = 'invisible'
        with self.assertRaisesRegex(ValueError, 'Unknown label meaning'):
            save_policy(self.dest, entries)

    def test_temporal_support_does_not_imply_unimplemented_workflows(self):
        self.assertIn('unavailable', temporal_support('nectar'))
        self.assertIn('Experimental', temporal_support('brood_early'))
        self.assertIn('separately', temporal_support('hive'))


class ImportReviewTests(PolicyFixture):
    def test_conflicts_persist_and_completions_are_per_frame(self):
        plan = self.import_hives()
        self.assertIn('Annotation scope differs', plan.policy_conflicts[0]['reason'])
        self.assertIn('Label meaning differs', plan.policy_conflicts[0]['reason'])
        self.assertEqual([r['status'] for r in review_rows(self.dest)], ['pending', 'pending'])
        complete_review(self.dest, 'video', 2, self.contributor)
        self.assertEqual(review_status(self.dest, 'video', 2), 'complete')
        self.assertEqual(review_status(self.dest, 'video', 5), 'pending')
        completion = read_reviews(self.dest)['frames']['video']['2']['completion']
        self.assertEqual(completion['completed_by']['name'], 'Reviewer')
        complete_review(self.dest, 'video', 2, complete=False)
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')

    def test_shared_edits_reopen_all_frames_and_new_frames_require_review(self):
        self.import_hives()
        self.finish()
        self.manager.save_video_annotations(self.dest, 'video', [mask_annotation(offset=12)], contributor=None)
        self.assertEqual([r['status'] for r in review_rows(self.dest)], ['pending', 'pending'])
        cv2.imwrite(str(self.dest / 'frames/video/frame_000007.jpg'), np.zeros((64, 96, 3), np.uint8))
        self.assertEqual(review_status(self.dest, 'video', 7), 'pending')

    def test_frame_edits_reopen_only_that_frame(self):
        self.import_hives()
        self.finish()
        self.manager.save_frame_annotations(self.dest, 'video', 2, [mask_annotation(9, 'bee')], contributor=None)
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')
        self.assertEqual(review_status(self.dest, 'video', 5), 'complete')

    def test_policy_change_reopens_review_but_confirmation_timestamp_does_not(self):
        self.import_hives()
        self.finish()
        entries = category_policies(self.dest_info)
        save_policy(self.dest, entries, self.contributor)
        self.assertEqual(review_status(self.dest, 'video', 2), 'complete')
        entries['hive']['meaning'] = 'visible'
        save_policy(self.dest, entries)
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')

    def test_legacy_unknown_meaning_must_be_confirmed_before_completion(self):
        info = deepcopy(self.dest_info)
        del info['annotation_policy']
        _write_json(self.dest / 'annotations/project.json', info)
        self.import_hives()
        with self.assertRaisesRegex(ValueError, 'Confirm the destination'):
            complete_review(self.dest, 'video', 2)
        save_policy(self.dest, default_policy(info)['categories'])
        complete_review(self.dest, 'video', 2)
        self.assertEqual(review_status(self.dest, 'video', 2), 'complete')

    def test_keep_preserves_completion_and_replace_reopens(self):
        self.import_hives()
        self.finish()
        self.import_hives()
        self.assertEqual(review_status(self.dest, 'video', 2), 'complete')
        self.import_hives(replace=True)
        self.assertEqual(review_status(self.dest, 'video', 2), 'pending')

    def test_local_imports_have_independent_requirements(self):
        info = deepcopy(self.source_info)
        _write_json(self.dest / 'annotations/project.json', info)
        self.import_hives()
        records = read_reviews(self.dest)
        self.assertEqual(records['shared'], {})
        complete_review(self.dest, 'video', 2)
        self.assertEqual(review_status(self.dest, 'video', 5), 'pending')

    def test_source_pending_review_is_disclosed(self):
        self.import_hives()
        third = self.root / 'third'
        ProjectManager().create_project(third, 'Third')
        plan = prepare_import(self.dest, third, {'video': [2]}, ['hive'])
        self.addCleanup(plan.close)
        self.assertIn('Source frame still has unfinished import review', plan.policy_conflicts[0]['reason'])

    def test_completion_or_policy_change_after_preview_invalidates_import(self):
        self.import_hives()
        plan = prepare_import(self.source, self.dest, {'video': [2]}, ['hive'], replace=True)
        self.addCleanup(plan.close)
        complete_review(self.dest, 'video', 2)
        with self.assertRaisesRegex(ValueError, 'changed after'):
            plan.apply()

    def test_export_blocks_pending_without_deleting_previous_output(self):
        self.import_hives()
        output = self.dest / 'annotations/coco/train'
        output.mkdir(parents=True, exist_ok=True)
        sentinel = output / 'existing.json'
        sentinel.write_text('{}')
        for exporter in (export_coco_per_video, export_coco_with_tracking):
            with self.assertRaisesRegex(ValueError, 'need review'):
                exporter(self.dest, ['video'], 'train')
            self.assertEqual(sentinel.read_text(), '{}')
        complete_review(self.dest, 'video', 2)
        require_reviewed(self.dest, {'video': [2]})
        with self.assertRaisesRegex(ValueError, 'frame_000005'):
            require_reviewed(self.dest, {'video': [5]})

    def test_coco_reviews_round_trip_and_old_exports_cannot_bypass(self):
        self.import_hives()
        self.exportable()
        self.finish()
        paths = export_coco_per_video(self.dest, ['video'], 'train')
        validate_coco_reviews(self.dest, paths)
        data = _read_json(paths[0])
        self.assertEqual(data['info']['annotation_policy']['hive']['meaning'], 'full_extent')
        for image in data['images']:
            image.pop('import_review_signature')
            image.pop('frame_index')  # Old exports must resolve frame filenames instead.
        _write_json(paths[0], data)
        with self.assertRaisesRegex(ValueError, 'Re-export COCO'):
            validate_coco_reviews(self.dest, paths)
        complete_review(self.dest, 'video', 2, complete=False)
        with self.assertRaisesRegex(ValueError, 'need review'):
            validate_coco_reviews(self.dest, paths)

    def test_tracking_coco_uses_video_mapping(self):
        self.import_hives()
        self.exportable()
        self.finish()
        path = export_coco_with_tracking(self.dest, ['video'], 'train')
        validate_coco_reviews(self.dest, [path])
        complete_review(self.dest, 'video', 5, complete=False)
        with self.assertRaisesRegex(ValueError, 'need review'):
            validate_coco_reviews(self.dest, [path])

    def test_policy_changes_reject_cached_coco(self):
        self.import_hives()
        self.exportable()
        self.finish()
        paths = export_coco_per_video(self.dest, ['video'], 'train')
        entries = category_policies(self.dest_info)
        entries['bee']['meaning'] = 'full_extent'
        save_policy(self.dest, entries)
        with self.assertRaisesRegex(ValueError, 'policy changed'):
            validate_coco_reviews(self.dest, paths)

    def test_malformed_review_file_is_not_treated_as_reviewed(self):
        self.import_hives()
        data = read_reviews(self.dest)
        data['shared']['video']['unsupported'] = data['shared']['video'].pop('hive')
        _write_json(self.dest / REVIEW_FILE, data)
        with self.assertRaisesRegex(ValueError, 'Invalid import review'):
            require_reviewed(self.dest, {'video': [2]})

    def test_review_and_policy_survive_optional_project_sync(self):
        self.import_hives()
        self.finish()
        remote = self.root / 'external'
        remote.mkdir()
        result = sync_project(self.dest, probe_destination(remote), self.contributor)
        copied = self.root / 'restored'
        restore_project(result['path'], copied)
        self.assertEqual(read_reviews(copied), read_reviews(self.dest))
        self.assertEqual(review_status(copied, 'video', 2), 'complete')
        self.assertEqual(category_policies(_read_json(copied / 'annotations/project.json')),
                         category_policies(self.dest_info))

    def test_gui_training_workers_reject_pending_cached_coco_before_preparing_data(self):
        from training.yolo_trainer import YOLOTrainingWorker
        from training.yolo_trainer_bbox import YOLOTrainingWorkerBBox
        from training.yolo_trainer_instance_focused import YOLOTrainingWorkerInstanceFocused
        from training.sam2_trainer import SAM2TrainingWorker
        self.import_hives()
        self.exportable()
        self.finish()
        export_coco_per_video(self.dest, ['video'], 'train')
        export_coco_per_video(self.dest, ['video'], 'val')
        complete_review(self.dest, 'video', 2, complete=False)
        for worker_class in (YOLOTrainingWorker, YOLOTrainingWorkerBBox):
            with self.subTest(worker=worker_class.__name__):
                worker = worker_class(self.dest, {})
                with self.assertRaisesRegex(ValueError, 'need review'):
                    worker._prepare_yolo_dataset()
        worker = YOLOTrainingWorkerInstanceFocused(self.dest, {})
        errors = []
        worker.training_failed.connect(errors.append)
        with patch.object(worker, '_generate_instance_focused_crops') as crops:
            worker.run()
        crops.assert_not_called()
        self.assertIn('need review', errors[0])
        with patch.object(SAM2TrainingWorker, '_setup_augmentation'):
            worker = SAM2TrainingWorker(self.dest, {})
        for loader in (worker._load_training_data, worker._load_validation_data):
            with self.assertRaisesRegex(ValueError, 'need review'):
                loader()
        self.assertFalse((self.dest / 'yolo_format').exists())
        self.assertFalse((self.dest / 'yolo_bbox_format').exists())


if __name__ == '__main__':
    unittest.main()
