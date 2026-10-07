"""Headless import-dialog and main-window refresh tests."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from core.annotation import AnnotationManager
from core.contributors import new_session
from core.project_import import prepare_import
from core.project_manager import ProjectManager
from core.project_sync import _write_json
from gui.main_window import MainWindow
from gui.project_import_dialog import ImportPreviewDialog, ProjectImportDialog
from scripts.test_project_import import mask_annotation


class ProjectImportGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source, self.dest = self.base / 'source', self.base / 'destination'
        ProjectManager().create_project(self.source, 'Source', hive_annotation_scope='frame')
        ProjectManager().create_project(self.dest, 'Destination', hive_annotation_scope='video')
        frames = self.source / 'frames/video'
        frames.mkdir()
        for index in (2, 5, 7):
            cv2.imwrite(str(frames / f'frame_{index:06d}.jpg'), np.full((64, 96, 3), index, np.uint8))
        _write_json(frames / 'video_metadata.json', {'selected_frames': [2, 5], 'split': 'train', 'total_frames': 10})
        (self.source / 'input_data/train/video.mp4').write_bytes(b'video')
        manager = AnnotationManager()
        for index, offset in ((2, 0), (5, 20)):
            manager.save_frame_annotations(self.source, 'video', index, [mask_annotation(offset=offset)], contributor=None)
        self.dialog = ProjectImportDialog(self.dest, new_session('August'))
        self.window = None
        self.stack = ExitStack()
        self.settings = QSettings(str(self.base / 'settings.ini'), QSettings.Format.IniFormat)
        self.stack.enter_context(patch('gui.main_window.QSettings', lambda: self.settings))

    def tearDown(self):
        self.dialog.close()
        if self.window:
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Discard):
                self.window.close()
        self.app.processEvents()
        self.stack.close()
        self.temp.cleanup()

    def test_source_scan_category_filter_selection_and_reference_dropdown(self):
        self.dialog.load_source(self.source)
        self.assertEqual(self.dialog.selections(), {'video': [2, 5]})
        combo = self.dialog.hive_sources['video']
        self.assertIsNone(combo.currentData())
        combo.setCurrentIndex(combo.findData(5))
        self.assertEqual(combo.currentData(), 5)
        for i in range(self.dialog.categories.count()):
            self.dialog.categories.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.assertEqual(self.dialog.selections(), {})
        self.assertFalse(self.dialog.preview_button.isEnabled())
        self.dialog.categories.item(1).setCheckState(Qt.CheckState.Checked)
        self.assertEqual(self.dialog.selections(), {'video': [2, 5]})
        self.dialog.annotated_only.setChecked(False)
        self.dialog.select_visible(True)
        self.assertEqual(self.dialog.selections(), {'video': [2, 5, 7]})
        self.dialog.select_visible(False)
        self.assertEqual(self.dialog.selections(), {})

    def test_scope_conversion_requires_explicit_confirmation_and_preview_renders(self):
        self.dialog.load_source(self.source)
        self.dialog.show()
        self.app.processEvents()
        self.dialog.grab().save('/tmp/project-import-dialog.png')
        plan = prepare_import(self.source, self.dest, {'video': [2, 5]}, ['hive'])
        try:
            preview = ImportPreviewDialog(plan)
            self.assertFalse(preview.import_button.isEnabled())
            preview.confirm_scope.setChecked(True)
            self.assertTrue(preview.import_button.isEnabled())
            preview.show()
            self.app.processEvents()
            preview.grab().save('/tmp/project-import-preview.png')
            preview.close()
        finally:
            plan.close()

    def test_replacement_requires_additional_confirmation(self):
        plan = prepare_import(self.source, self.dest, {'video': [2]}, ['hive'])
        try:
            plan.rows[0]['action'] = 'Replace'
            preview = ImportPreviewDialog(plan)
            preview.confirm_scope.setChecked(True)
            self.assertFalse(preview.import_button.isEnabled())
            preview.confirm_replace.setChecked(True)
            self.assertTrue(preview.import_button.isEnabled())
            preview.close()
        finally:
            plan.close()

    def test_cancel_preview_does_not_import(self):
        self.dialog.load_source(self.source)
        with patch.object(ImportPreviewDialog, 'exec', return_value=QDialog.DialogCode.Rejected):
            self.dialog.preview_import()
        self.assertIsNone(self.dialog.result)
        self.assertFalse((self.dest / 'frames/video').exists())

    def test_real_main_window_refresh_and_resave_preserve_imported_masks_and_creator(self):
        with patch.object(MainWindow, 'load_settings'):
            self.window = MainWindow()
        self.window.set_contributor(new_session('August'))
        self.window.load_project(self.dest)

        def perform_import(dialog):
            dialog.load_source(self.source)
            with patch.object(ImportPreviewDialog, 'exec', return_value=QDialog.DialogCode.Accepted):
                dialog.preview_import()
            return QDialog.DialogCode.Accepted

        with patch.object(ProjectImportDialog, 'exec', perform_import), \
                patch.object(QMessageBox, 'information'), \
                patch.object(QMessageBox, 'warning') as warning:
            self.window.import_project_data()
            warning.assert_not_called()
        self.assertEqual(self.window.video_list.count(), 1)
        self.assertEqual(len(self.window.frames), 2)
        self.assertTrue(self.window.preload_worker.is_alive())
        loaded = self.window.canvas.get_annotations()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]['category'], 'hive')
        np.testing.assert_array_equal(loaded[0]['mask'], mask_annotation()['mask'])
        self.window._save_annotation_sources(force=True)
        after = self.window.annotation_manager.load_video_annotations(self.dest, 'video')[0][0]
        self.assertEqual(after['provenance']['created_by']['name'], 'Original author')


if __name__ == '__main__':
    unittest.main()
