from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from gui.training_dialog import TrainingConfigDialog
from training.image_size import (FULL_RESOLUTION, FULL_RESOLUTION_LABEL, full_resolution_size,
                                 longest_sides, resolve_image_size)
from training.yolo_trainer import YOLOTrainingWorker
from training.yolo_trainer_bbox import YOLOTrainingWorkerBBox


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def write_images(folder, sizes):
    """Write one blank image per (width, height)."""
    folder.mkdir(parents=True)
    for index, (width, height) in enumerate(sizes):
        cv2.imwrite(str(folder / f'image_{index}.jpg'), np.zeros((height, width, 3), np.uint8))
    return folder


def test_a_number_is_used_as_it_is_without_looking_at_any_image(tmp_path):
    assert resolve_image_size(3072, [tmp_path / 'missing']) == (3072, '')
    assert resolve_image_size('1280', []) == (1280, '')


def test_full_resolution_is_the_longest_side_of_any_image(tmp_path):
    train = write_images(tmp_path / 'train', [(128, 96), (128, 90), (126, 95), (95, 104)])
    val = write_images(tmp_path / 'val', [(126, 95)])
    (train / 'notes.txt').write_text('not an image')
    assert longest_sides([train, val]) == {128: 2, 126: 2, 104: 1}
    size, note = resolve_image_size(FULL_RESOLUTION, [train, val, tmp_path / 'missing'])
    assert size == 128
    assert note == ('Full resolution: the model input is 128 px, the longest image side in this data. '
                    'Images by longest side: 128 px, 2 images, unchanged; 126 px, 2 images, enlarged 1.6%; '
                    '104 px, 1 image, enlarged 23.1%.')


def test_full_resolution_rounds_up_to_a_size_the_model_accepts(tmp_path):
    folder = write_images(tmp_path / 'train', [(100, 60)])
    size, note = resolve_image_size(FULL_RESOLUTION, [folder])
    assert size == 128
    assert 'the longest image side in this data (100 px) rounded up' in note
    assert '100 px, 1 image, enlarged 28.0%' in note
    assert full_resolution_size({4056: 1}) == 4064 and full_resolution_size({4096: 3, 3326: 1}) == 4096


def test_full_resolution_without_images_is_refused(tmp_path):
    with pytest.raises(ValueError, match='none were found'):
        resolve_image_size(FULL_RESOLUTION, [tmp_path])


def test_dialog_offers_full_resolution_for_whole_frame_training_only(app):
    dialog = TrainingConfigDialog()
    assert dialog.get_config()['imgsz'] == 640 and dialog.full_resolution_note.isHidden()
    assert dialog.get_config()['batch'] == 4
    dialog.imgsz_combo.setCurrentText(FULL_RESOLUTION_LABEL)
    config = dialog.get_config()
    assert config['imgsz'] == FULL_RESOLUTION and config['batch'] == 1
    assert not dialog.full_resolution_note.isHidden()
    dialog.batch_spin.setValue(2)
    assert dialog.get_config()['batch'] == 2
    dialog.imgsz_combo.setCurrentText('3072')
    config = dialog.get_config()
    assert config['imgsz'] == 3072 and config['batch'] == 2 and dialog.full_resolution_note.isHidden()
    dialog.close()
    crops = TrainingConfigDialog(instance_focused=True)
    assert crops.imgsz_combo.findData(FULL_RESOLUTION) == -1
    assert crops.get_config()['imgsz'] == 640
    crops.close()


def test_dialog_offers_3584_between_3072_and_full_resolution_for_whole_frames_only(app):
    dialog = TrainingConfigDialog()
    sizes = [dialog.imgsz_combo.itemText(index) for index in range(dialog.imgsz_combo.count())]
    assert sizes[-3:] == ['3072', '3584', FULL_RESOLUTION_LABEL]
    dialog.imgsz_combo.setCurrentText('3584')
    config = dialog.get_config()
    assert config['imgsz'] == 3584 and config['imgsz'] % 32 == 0  # the model needs a multiple of 32
    assert dialog.full_resolution_note.isHidden()
    dialog.close()
    crops = TrainingConfigDialog(instance_focused=True)
    assert crops.imgsz_combo.findText('3584') == -1
    crops.close()


def test_mask_training_passes_the_full_resolution_size_and_says_what_it_is(app, tmp_path):
    write_images(tmp_path / 'images' / 'train', [(128, 96), (104, 95)])
    write_images(tmp_path / 'images' / 'val', [(126, 95)])
    worker = YOLOTrainingWorker(tmp_path, {'imgsz': FULL_RESOLUTION})
    notes = []
    worker.stage_update.connect(notes.append)
    with patch('training.yolo_trainer.require_training_device', return_value=(0, 'test GPU')), \
            patch('training.yolo_trainer.CarbonTrainingRun'), patch('training.yolo_trainer.YOLO') as yolo:
        worker._train_model(tmp_path / 'dataset.yaml')
    assert yolo.return_value.train.call_args.kwargs['imgsz'] == 128
    assert notes[0].startswith('Full resolution: the model input is 128 px')
    assert notes[1] == 'Using GPU: test GPU'


def test_box_training_passes_the_full_resolution_size(app, tmp_path):
    write_images(tmp_path / 'train' / 'images', [(100, 60)])
    write_images(tmp_path / 'val' / 'images', [(90, 60)])
    weights = tmp_path / 'run' / 'weights'
    weights.mkdir(parents=True)
    (weights / 'best.pt').touch()
    worker = YOLOTrainingWorkerBBox(tmp_path, {'imgsz': FULL_RESOLUTION})
    notes = []
    worker.stage_update.connect(notes.append)
    with patch('training.yolo_trainer_bbox.require_training_device', return_value=(0, 'test GPU')), \
            patch('training.yolo_trainer_bbox.CarbonTrainingRun'), patch('training.yolo_trainer_bbox.YOLO') as yolo:
        yolo.return_value.train.return_value.save_dir = str(tmp_path / 'run')
        worker._train_model(tmp_path / 'dataset.yaml')
    assert yolo.return_value.train.call_args.kwargs['imgsz'] == 128
    assert notes[0].startswith('Full resolution: the model input is 128 px')


def test_a_chosen_number_still_reaches_training_unchanged(app, tmp_path):
    worker = YOLOTrainingWorker(tmp_path, {'imgsz': 3072})
    notes = []
    worker.stage_update.connect(notes.append)
    with patch('training.yolo_trainer.require_training_device', return_value=(0, 'test GPU')), \
            patch('training.yolo_trainer.CarbonTrainingRun'), patch('training.yolo_trainer.YOLO') as yolo:
        worker._train_model(tmp_path / 'dataset.yaml')
    assert yolo.return_value.train.call_args.kwargs['imgsz'] == 3072
    assert not any('Full resolution' in note for note in notes)
