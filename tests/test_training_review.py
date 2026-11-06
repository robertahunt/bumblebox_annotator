import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

from core import training_review as review
from core.contributors import new_session
from core.project_sync import _read_json, _write_json
from gui.training_review import TrainingReviewDock
from gui.training_dialog import TrainingConfigDialog
from gui.yolo_toolbar import YOLOToolbar


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def project(tmp_path):
    for video, split in [('video-a', 'train'), ('video-b', 'val')]:
        folder = tmp_path / 'input_data' / split
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f'{video}.mp4').touch()
        folder = tmp_path / 'frames' / video
        folder.mkdir(parents=True)
        (folder / 'video_metadata.json').write_text(json.dumps({
            'split': split, 'selected_frames': [0], 'n_selected': 1,
            'total_frames': 3, 'max_mask_id': 90, 'physical_scale': {'px_cm': 205}}))
        for index in (0, 1, 2):
            name = f'frame_{index:06d}'
            cv2.imwrite(str(folder / f'{name}.jpg'), np.zeros((16, 24, 3), np.uint8))
            for kind in ('json', 'png'):
                (tmp_path / 'annotations' / kind / video).mkdir(parents=True, exist_ok=True)
            (tmp_path / 'annotations/json' / video / f'{name}.json').write_text(json.dumps([
                {'mask_id': 1, 'category': 'bee'}]))
            mask = np.zeros((16, 24), np.uint8)
            mask[3:10, 4:12] = 1
            cv2.imwrite(str(tmp_path / 'annotations/png' / video / f'{name}.png'), mask)
    return tmp_path


def test_queue_discovery_is_unselected_training_only_and_resume_does_not_reset(project):
    queue = review.create_queue(project)
    assert [(r['video'], r['frame']) for r in queue['items']] == [('video-a', 1), ('video-a', 2)]
    review.record_decision(project, 'video-a', 1, True)
    assert review.create_queue(project)['items'][0]['status'] == 'complete'


def test_complete_and_reverse_preserve_other_frames_and_metadata(project):
    review.create_queue(project)
    path = project / 'frames/video-a/video_metadata.json'
    before = _read_json(path)
    person = new_session('Reviewer')
    queue = review.record_decision(project, 'video-a', 1, True, person)
    assert _read_json(path) == dict(before, selected_frames=[0, 1], n_selected=2)
    assert queue['items'][0]['reviewed_by'] == person
    # Another selection added since the queue was opened must survive.
    _write_json(path, dict(before, selected_frames=[0, 1, 2], n_selected=3, max_mask_id=99))
    review.record_decision(project, 'video-a', 1, False, person)
    assert _read_json(path) == dict(before, selected_frames=[0, 2], n_selected=2, max_mask_id=99)
    assert review.read_queue(project)['items'][0]['status'] == 'incomplete'


def test_failed_review_write_rolls_back_selection(project, monkeypatch):
    review.create_queue(project)
    original = review._write_json
    before = _read_json(project / 'frames/video-a/video_metadata.json')
    def fail_queue(path, value):
        if Path(path).name == 'training_review.json':
            raise OSError('disk full')
        return original(path, value)
    monkeypatch.setattr(review, '_write_json', fail_queue)
    with pytest.raises(OSError, match='disk full'):
        review.record_decision(project, 'video-a', 1, True)
    assert _read_json(project / 'frames/video-a/video_metadata.json') == before
    assert review.read_queue(project)['items'][0]['status'] == 'pending'


@pytest.mark.parametrize('change', ['metadata_split', 'actual_split', 'missing_image', 'empty_mask', 'outside_queue'])
def test_invalid_or_held_out_frames_cannot_be_added(project, change):
    review.create_queue(project)
    path = project / 'frames/video-a/video_metadata.json'
    frame = 1
    if change == 'metadata_split':
        _write_json(path, dict(_read_json(path), split='val'))
    elif change == 'actual_split':
        (project / 'input_data/train/video-a.mp4').rename(project / 'input_data/val/video-a.mp4')
    elif change == 'missing_image':
        (project / 'frames/video-a/frame_000001.jpg').unlink()
    elif change == 'empty_mask':
        cv2.imwrite(str(project / 'annotations/png/video-a/frame_000001.png'), np.zeros((16, 24), np.uint8))
    else:
        frame = 0
    with pytest.raises(ValueError):
        review.record_decision(project, 'video-a', frame, True)
    assert _read_json(path)['selected_frames'] == [0]
    assert review.read_queue(project)['items'][0]['status'] == 'pending'


def test_existing_import_review_gate_is_preserved(project, monkeypatch):
    review.create_queue(project)
    monkeypatch.setattr(review, 'require_reviewed', Mock(side_effect=ValueError('needs import review')))
    with pytest.raises(ValueError, match='needs import review'):
        review.record_decision(project, 'video-a', 1, True)
    assert _read_json(project / 'frames/video-a/video_metadata.json')['selected_frames'] == [0]


def test_decisions_change_the_actual_training_export(project):
    from training.coco_video_export import export_coco_per_video
    review.create_queue(project)
    review.record_decision(project, 'video-a', 1, True)
    review.record_decision(project, 'video-a', 2, False)
    export_coco_per_video(project, ['video-a'], 'train')
    path = project / 'annotations/coco/train/video-a.json'
    assert {image['frame_index'] for image in _read_json(path)['images']} == {0, 1}
    review.record_decision(project, 'video-a', 1, False)
    export_coco_per_video(project, ['video-a'], 'train')
    assert {image['frame_index'] for image in _read_json(path)['images']} == {0}


def test_queue_validates_paths_and_never_auto_selects(project):
    with pytest.raises(ValueError):
        review.create_queue(project, [{'video': '../outside', 'frame': 1}])
    with pytest.raises(ValueError):
        review.create_queue(project, [{'video': 'video-b', 'frame': 1}])
    queue = review.create_queue(project, [{'video': 'video-a', 'frame': 2}])
    assert len(queue['items']) == 1
    assert _read_json(project / 'frames/video-a/video_metadata.json')['selected_frames'] == [0]


def test_higher_resolution_reaches_actual_current_frame_predict(app, monkeypatch):
    from gui.main_window import MainWindow
    toolbar = YOLOToolbar()
    model = Mock()
    model.predict.return_value = []
    toolbar.model = model
    assert toolbar.prediction_size_kwargs() == {}
    toolbar.input_size_combo.setCurrentIndex(toolbar.input_size_combo.findData(3072))
    window = SimpleNamespace(yolo_toolbar=toolbar, frames=[Path('/tmp/review.jpg')],
                             current_frame_idx=0, status_label=Mock(), canvas=Mock())
    monkeypatch.setattr(QMessageBox, 'information', Mock())
    MainWindow.run_yolo_inference(window)
    assert model.predict.call_args.kwargs['imgsz'] == 3072
    assert model.predict.call_args.kwargs['retina_masks'] is True
    dialog = TrainingConfigDialog()
    dialog.imgsz_combo.setCurrentText('2560')
    assert dialog.get_config()['imgsz'] == 2560
    dialog.close()
    toolbar.close()


def test_navigation_goes_to_exact_frame_and_saves_then_adds(app, project, monkeypatch):
    from gui.main_window import MainWindow
    monkeypatch.setattr(MainWindow, 'load_settings', lambda self: None)
    window = MainWindow()
    dock = TrainingReviewDock(window)
    window.project_path = project
    window.current_video_id = None
    window.current_frame_idx = -1
    window.frames = []
    window.is_playing = False
    monkeypatch.setattr(window.workflow, 'activate', Mock())
    monkeypatch.setattr(window.workflow, 'saved', Mock())
    monkeypatch.setattr(window, '_save_annotation_sources', Mock())
    monkeypatch.setattr(window, 'update_frame_list', Mock())
    def load_video(video):
        window.current_video_id = video
        window.frames = sorted((project / 'frames' / video).glob('frame_*.jpg'))
        window.current_frame_idx = 0
    def load_frame(index):
        window.current_frame_idx = index
    monkeypatch.setattr(window, 'load_video_frames', load_video)
    monkeypatch.setattr(window, 'load_frame', load_frame)
    try:
        dock.start()
        assert window.current_frame_idx == 1
        assert dock.current_matches()
        dock.decide(True)
        assert _read_json(project / 'frames/video-a/video_metadata.json')['selected_frames'] == [0, 1]
        assert window.current_frame_idx == 2
        dock.decide(False)
        assert dock.position is None
        assert [r['status'] for r in review.read_queue(project)['items']] == ['complete', 'incomplete']
        dock.filter.setCurrentIndex(dock.filter.findData('incomplete'))
        assert window.current_frame_idx == 2
        # Navigating elsewhere cannot accidentally approve the queued frame.
        window.current_frame_idx = 0
        dock.decide(True)
        assert _read_json(project / 'frames/video-a/video_metadata.json')['selected_frames'] == [0, 1]
        # A failed source save must leave the review and selection untouched.
        window.current_frame_idx = 2
        monkeypatch.setattr(window, '_save_annotation_sources', Mock(side_effect=OSError('save failed')))
        monkeypatch.setattr(QMessageBox, 'warning', Mock())
        dock.decide(True)
        assert dock.position == 1
        assert _read_json(project / 'frames/video-a/video_metadata.json')['selected_frames'] == [0, 1]
    finally:
        dock.timer.stop()
        dock.close()
        window.project_path = None
        window.current_frame_modified = False
        monkeypatch.setattr('gui.main_window.QSettings', Mock())
        window.close()
        app.processEvents()
