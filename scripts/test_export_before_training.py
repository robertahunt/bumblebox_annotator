"""Training that exports COCO first starts by itself when the export succeeds, and never after a failed one."""

import unittest
from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QApplication, QMessageBox

from scripts import test_workspaces as fixture

FLOWS = [  # menu action, worker class it starts, extra configuration
    ('train_yolo_model_instance_focused', 'YOLOTrainingWorkerInstanceFocused', {}),
    ('train_yolo_model', 'YOLOTrainingWorker', {'model_type': 'bee'}),
    ('train_yolo_bbox_model', 'YOLOTrainingWorkerBBox', {}),
]


class ExportBeforeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        fixture.WorkspaceGuiTests.setUp(self)
        # Training needs a validation split: move beta there (its file location and its settings).
        (self.source / 'input_data/val').mkdir(parents=True)
        (self.source / 'input_data/train/beta.mp4').rename(self.source / 'input_data/val/beta.mp4')
        metadata = self.source / 'frames/beta/video_metadata.json'
        metadata.write_text(metadata.read_text().replace('"train"', '"val"'))
        self.window.load_project(self.source)
        self.information = self.stack.enter_context(patch.object(QMessageBox, 'information'))
        self.critical = self.stack.enter_context(patch.object(QMessageBox, 'critical'))
        # A question would wait for an answer forever here; none is expected once an export exists.
        self.question = self.stack.enter_context(
            patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No))

    tearDown = fixture.WorkspaceGuiTests.tearDown

    def test_the_export_asks_for_ok_only_when_nothing_follows_it(self):
        summary = self.window.export_coco_format(confirm=False)
        self.assertIn('Training: 1 videos', summary)
        self.assertIn('Validation: 1 videos', summary)
        self.information.assert_not_called()
        self.assertIn('Exported COCO format', self.window.status_label.text())
        self.assertTrue((self.source / 'annotations/coco/train/alpha.json').exists())
        self.assertEqual(self.window.export_coco_format(), summary)  # File > Export COCO Format... still confirms
        self.information.assert_called_once()

    def test_a_failed_export_reports_nothing_exported(self):
        with patch('gui.main_window.export_coco_per_video', side_effect=ValueError('imported frames need review')):
            self.assertIsNone(self.window.export_coco_format(confirm=False))
        self.critical.assert_called_once()
        self.assertIn('imported frames need review', self.critical.call_args.args[2])

    def run_flow(self, action, worker_name, extra, export_result):
        dialog = MagicMock()
        dialog.exec.return_value = True
        dialog.get_config.return_value = dict(extra, export_coco=True, epochs=1)
        progress = MagicMock(training_completed=False, training_failed=False)
        with patch('gui.main_window.TrainingConfigDialog', return_value=dialog) as self.dialog_class, \
                patch('gui.main_window.TrainingProgressDialog', return_value=progress), \
                patch(f'gui.main_window.{worker_name}') as worker, \
                patch.object(self.window, 'export_coco_format', return_value=export_result) as export, \
                patch.object(self.window, '_validate_coco_categories_for_model', return_value=True, create=True):
            getattr(self.window, action)()
        return export, worker, progress

    def test_training_starts_by_itself_after_a_successful_export(self):
        self.window.export_coco_format(confirm=False)  # so that the flows find an earlier export
        for action, worker_name, extra in FLOWS:
            with self.subTest(action):
                export, worker, progress = self.run_flow(action, worker_name, extra, 'Exported: Training: 2 videos')
                export.assert_called_once_with(confirm=False)
                worker.return_value.start.assert_called_once()
                progress.log.assert_any_call('Exported: Training: 2 videos')
        self.information.assert_not_called()
        self.question.assert_not_called()

    def test_only_whole_frame_training_offers_a_trained_model_to_start_from(self):
        self.window.export_coco_format(confirm=False)
        for action, worker_name, extra in FLOWS:
            with self.subTest(action):
                self.run_flow(action, worker_name, extra, 'Exported')
                offered = self.dialog_class.call_args.kwargs.get('start_from', False)
                self.assertEqual(offered, action == 'train_yolo_model')

    def test_training_never_starts_after_a_failed_or_cancelled_export(self):
        self.window.export_coco_format(confirm=False)  # an older export is on disk, and must not be used
        for action, worker_name, extra in FLOWS:
            with self.subTest(action):
                export, worker, _ = self.run_flow(action, worker_name, extra, None)
                export.assert_called_once_with(confirm=False)
                worker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
