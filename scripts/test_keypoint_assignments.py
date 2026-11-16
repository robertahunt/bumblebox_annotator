"""Keypoints as assignable work beside masks, using disposable projects and offscreen Qt."""

from copy import deepcopy
from pathlib import Path
import unittest

from PyQt6.QtCore import Qt

from core import keypoints as kp
from core.annotation import AnnotationManager
from core.annotation_work import KEYPOINT_TASKS, mark_finished, task_label
from core.assigned_work import (publish_work, read_assignments, save_assignments,
                                shared_assignment_progress, workspace)
from core.assigned_workspace import PreparedAssignedDownload
from core.project_sync import _read_json, sync_project
from gui.annotation_work import set_keypoint_tasks, task_combo
from scripts import test_assigned_work as backend
from scripts import test_assigned_work_gui as gui


def assign(test, **tasks):
    """Give Alex alpha and Sam beta, with the given tasks."""
    choices = deepcopy(test.choices)
    for video, task in tasks.items():
        choices[video]['task'] = task
    data = read_assignments(test.view())
    save_assignments(test.source, test.configuration, test.coordinator, choices, data['revision'])


def turn_on_keypoints(test):
    kp.save_project_config(test.source, kp.default_config())
    sync_project(test.source, test.configuration, test.coordinator)


def place_keypoints(project, actor, video='alpha', frame=0, points=None):
    manager = AnnotationManager()
    manager.contributor_session = actor
    rows = manager.load_frame_annotations(project, video, frame)
    rows[0]['keypoints'] = points or {'head': [10.0, 6.0, 2], 'thorax': [14.0, 8.0, 2]}
    manager.save_frame_annotations(project, video, frame, rows)


class KeypointAssignmentTests(unittest.TestCase):
    setUp = backend.AssignedWorkTests.setUp
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    change = backend.AssignedWorkTests.change

    def test_keypoint_work_must_be_published_before_it_is_assigned(self):
        with self.assertRaisesRegex(ValueError, 'Turn on keypoints'):
            assign(self, alpha='keypoints')
        assign(self, alpha='bee')  # mask tasks are unaffected
        kp.save_project_config(self.source, kp.default_config())  # on locally, not published yet
        with self.assertRaisesRegex(ValueError, 'publish the project'):
            assign(self, alpha='bee_keypoints')
        sync_project(self.source, self.configuration, self.coordinator)
        for task in KEYPOINT_TASKS:
            assign(self, alpha=task)
            self.assertEqual(read_assignments(self.view())['videos']['alpha']['task'], task)

    def test_assigned_keypoints_finish_and_submit_separately_from_masks(self):
        turn_on_keypoints(self)
        assign(self, alpha='bee_keypoints')
        local = self.download()
        self.assertEqual(kp.read_project_config(local).names, kp.DEFAULT_KEYPOINT_NAMES)
        upload = lambda: publish_work(local, self.configuration, self.alex, auto_ready=True)
        progress = lambda: shared_assignment_progress(
            self.configuration, self.project_id, self.alex, workspace(local)['assignments'])['videos']['alpha']

        mark_finished(local, 'alpha', 0, 'bee', self.alex)  # masks done, keypoints not started
        self.assertEqual(upload()['ready'], [])
        place_keypoints(local, self.alex)
        self.assertEqual(upload()['ready'], [])  # placed, but not marked finished
        mark_finished(local, 'alpha', 0, 'keypoints', self.alex)
        self.assertEqual(upload()['ready'], ['alpha'])  # both kinds finished: ready for review
        shared = progress()  # what the director sees, from the published summary alone
        self.assertEqual((shared['status'], shared['completion']['state']), ('Ready for Review', 'finished'))

    def test_keypoint_only_assignment_ignores_mask_marks_and_reaches_an_existing_copy(self):
        assign(self, alpha='bee')
        local = self.download()  # made before the project used keypoints
        self.assertIsNone(kp.read_project_config(local))
        turn_on_keypoints(self)
        assign(self, alpha='keypoints')  # the same video, now for keypoints only
        plan = PreparedAssignedDownload(self.configuration, self.project_id, self.alex, ['alpha'], local)
        plan.apply()
        self.assertEqual(workspace(local)['assignments']['alpha']['task'], 'keypoints')
        self.assertIsNotNone(kp.read_project_config(local))

        place_keypoints(local, self.alex)
        mark_finished(local, 'alpha', 0, 'keypoints', self.alex)  # no mask mark is needed
        published = publish_work(local, self.configuration, self.alex, auto_ready=True)
        self.assertEqual(published['ready'], ['alpha'])
        index = _read_json(Path(published['path']) / 'manifest.json')['annotation_work']['videos']
        self.assertIn('keypoints', index['alpha']['categories'])
        self.assertNotIn('keypoints', index['beta']['categories'])  # not rescanned for someone else's video

    def test_task_lists_offer_keypoint_tasks_only_when_the_project_uses_them(self):
        gui.AssignedWorkGuiTests.setUpClass()
        combo = task_combo()
        tasks = lambda: [combo.itemData(i) for i in range(combo.count())]
        self.assertFalse(set(KEYPOINT_TASKS) & set(tasks()))
        set_keypoint_tasks(combo, True)
        start = tasks().index('bee_hive') + 1
        self.assertEqual(tasks()[start:start + 3], list(KEYPOINT_TASKS))
        self.assertEqual([task_label(t) for t in KEYPOINT_TASKS],
                         ['Bee keypoints', 'Bees + keypoints', 'Bees + Hive (nest) + keypoints'])
        combo.setCurrentIndex(combo.findData('keypoints'))
        changes = []
        combo.currentIndexChanged.connect(changes.append)
        set_keypoint_tasks(combo, False)  # keypoints turned off: fall back to all mask types
        self.assertEqual((combo.currentData(), len(changes)), (None, 1))
        self.assertFalse(set(KEYPOINT_TASKS) & set(tasks()))


class KeypointAssignmentGuiTests(unittest.TestCase):
    setUpClass = gui.AssignedWorkGuiTests.__dict__['setUpClass']
    setUp = gui.AssignedWorkGuiTests.setUp
    tearDown = gui.AssignedWorkGuiTests.tearDown
    view = backend.AssignedWorkTests.view
    download = backend.AssignedWorkTests.download
    make_window = gui.AssignedWorkGuiTests.make_window

    def open_video(self, w, video):
        for row in range(w.video_list.count()):
            item = w.video_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == video:
                w.video_list.setCurrentItem(item)
                w._load_video_list_item(item)
                self.app.processEvents()
                return
        self.fail('video not listed: ' + video)

    def test_keypoints_tool_follows_the_assignment_and_display_switch_is_always_there(self):
        turn_on_keypoints(self)
        choices = deepcopy(self.choices)
        choices['alpha']['task'] = 'bee_keypoints'
        choices['beta'].update(contributor=self.alex, task='hive')
        data = read_assignments(self.view())
        save_assignments(self.source, self.configuration, self.coordinator, choices, data['revision'])
        local = self.download(videos=['alpha', 'beta'])
        w = self.make_window(local)
        w.show()
        self.app.processEvents()
        toolbar, filters = w.toolbar, w.annotation_work.filters

        self.open_video(w, 'alpha')
        self.assertEqual(filters.task.currentData(), 'bee_keypoints')  # Work on follows the assignment
        self.assertFalse(toolbar.keypoint_btn.isHidden())
        self.assertFalse(toolbar.keypoints_checkbox.isHidden())
        self.assertTrue(w.keypoint_mode_action.isVisible())
        w.toolbar.set_tool('keypoint')
        self.assertEqual(w.canvas.current_tool, 'keypoint')

        self.open_video(w, 'beta')  # a masks-only assignment: no Keypoints tool
        self.assertEqual(filters.task.currentData(), 'hive')
        self.assertTrue(toolbar.keypoint_btn.isHidden())
        self.assertNotEqual(w.canvas.current_tool, 'keypoint')
        self.assertFalse(toolbar.keypoints_checkbox.isHidden())  # but keypoints can still be shown

        # The three display switches agree with each other and with the layer.
        layer = w.canvas.keypoint_layer
        toolbar.keypoints_checkbox.setChecked(False)
        self.assertEqual((layer.visible, w.keypoint_mode_action.isChecked(),
                          w.keypoint_toolbar.show_checkbox.isChecked()), (False, False, False))
        self.assertTrue(w.canvas.show_segmentations)  # masks are untouched
        w.keypoint_mode_action.trigger()
        self.assertEqual((layer.visible, toolbar.keypoints_checkbox.isChecked()), (True, True))
        self.open_video(w, 'alpha')
        self.assertFalse(toolbar.keypoint_btn.isHidden())


if __name__ == '__main__':
    unittest.main()
