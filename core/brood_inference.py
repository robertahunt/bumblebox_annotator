"""Optional experimental brood inference and bounded-memory per-video exports."""

import csv
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np

from core.categories import BROOD_CATEGORIES, CATEGORY_COLORS


def brood_model_classes(model):
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    classes = {int(key): str(name) for key, name in items}
    if getattr(model, 'task', None) != 'segment' or sorted(classes.values()) != sorted(BROOD_CATEGORIES):
        raise ValueError('Brood Model must be a segmentation checkpoint trained on all five brood appearance classes')
    return {key: BROOD_CATEGORIES.index(name) + 1 for key, name in classes.items()}


def brood_evidence(result, frame_shape, class_ids):
    """Conflicting stage masks are unobserved, not an arbitrary winning stage."""
    labels = np.zeros(frame_shape, np.uint8)
    if result.masks is None:
        if result.boxes is not None and len(result.boxes):
            raise ValueError('Brood model returned boxes without segmentation masks')
        return labels
    for mask, raw_class in zip(result.masks.data, result.boxes.cls):
        mask = mask.cpu().numpy() > 0.5
        if mask.shape != tuple(frame_shape):
            raise ValueError('Brood inference requires full-resolution retina_masks=True output')
        category = class_ids[int(raw_class)]
        conflict = mask & (labels != 0) & (labels != category)
        labels[mask & (labels == 0)] = category
        labels[conflict] = 255
    return labels


def paint_brood_overlay(frame, snapshots):
    overlay = frame.copy()
    for snapshot in snapshots:
        x1, y1, x2, y2 = snapshot['bbox']
        if x2 <= x1 or y2 <= y1:
            continue
        labels = cv2.resize(snapshot['labels'], (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        crop = overlay[y1:y2, x1:x2]
        colors = {i + 2: CATEGORY_COLORS[name][::-1] for i, name in enumerate(BROOD_CATEGORIES)}
        colors[7] = (180, 180, 180)
        for label, color in colors.items():
            selected = labels == label
            crop[selected] = (crop[selected] * 0.55 + np.asarray(color) * 0.45).astype(np.uint8)
    return overlay


class BroodVideoWriter:
    """One archive and one CSV per video; optional MP4 preview, no loose frames."""

    def __init__(self, folder, video_id, context, temporal_map, preview=False, preview_limit=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.video_id = video_id
        self.context = context
        self.temporal_map = temporal_map
        self.folder = folder
        self.archive = zipfile.ZipFile(folder / f'{video_id}_brood_maps.zip', 'x', zipfile.ZIP_DEFLATED)
        self.csv_file = None
        self.video = None
        self.preview = preview
        self.preview_limit = preview_limit
        self.chamber_ids = None
        self.metadata = dict(schema_version=1, experimental=True, video_id=video_id, context=context,
                             classes=list(BROOD_CATEGORIES), resolution=temporal_map.resolution,
                             labels={'unknown': 0, 'background': 1, 'brood_stage_unresolved': 7,
                                     **{name: i + 2 for i, name in enumerate(BROOD_CATEGORIES)}},
                             window_seconds=temporal_map.window_seconds,
                             max_weight=temporal_map.max_weight, min_weight=temporal_map.min_weight,
                             stage_threshold=temporal_map.stage_threshold,
                             scoring='history_plus_current', frame_count=0, complete=False,
                             chamber_layout_resets=[])

    def write(self, frame_number, frame, labels, chambers, bees, time_seconds, fps):
        ids = set(chambers)
        if self.chamber_ids is not None and self.chamber_ids != ids:
            for key in list(self.temporal_map.states):
                if key[0] == self.context:
                    del self.temporal_map.states[key]
            self.metadata['chamber_layout_resets'].append(frame_number)
        self.chamber_ids = ids
        snapshots = []
        for chamber_id, chamber in chambers.items():
            snap = self.temporal_map.update(self.context, chamber_id, chamber, labels, bees,
                                            frame.shape[:2], time_seconds)
            snapshots.append(snap)
            buffer = io.BytesIO()
            observed = np.full(self.temporal_map.resolution[::-1], 255, np.uint8)
            geom = self.temporal_map.geometry
            valid = geom._normalize_chamber_mask(chamber, frame.shape[:2])
            valid &= ~geom._normalize_bee_occlusion(bees, chamber, frame.shape[:2])
            if chamber.get('temporal_unavailable', False):
                valid[:] = False
            for index in range(6):
                observed[valid & geom._normalize_mask(labels == index, chamber, frame.shape[:2])] = index
            np.savez(buffer, **snap, observed_labels=observed)
            self.archive.writestr(f'frame_{frame_number:06d}_chamber_{chamber_id}.npz', buffer.getvalue())
            for bee in bees:
                values = self.temporal_map.bee_overlap(snap, bee, chamber, frame.shape[:2])
                if values['known_fraction'] is None:
                    continue
                row = dict(video_id=self.video_id, frame_number=frame_number,
                           time_seconds=time_seconds, chamber_id=chamber_id,
                           tracker_bee_id=bee.instance_id, **values)
                if self.csv_file is None:
                    self.csv_file = (self.folder / f'{self.video_id}_brood_overlap.csv').open('x', newline='')
                    self.csv = csv.DictWriter(self.csv_file, fieldnames=list(row))
                    self.csv.writeheader()
                self.csv.writerow(row)
        if self.preview and (self.preview_limit is None or frame_number <= self.preview_limit):
            if self.video is None:
                self.video = cv2.VideoWriter(str(self.folder / f'{self.video_id}_brood_annotated.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                if not self.video.isOpened():
                    raise RuntimeError('Cannot create brood preview video')
            self.video.write(paint_brood_overlay(frame, snapshots))
        self.metadata['frame_count'] += 1

    def close(self, complete=False):
        if self.archive.fp is None:
            return
        try:
            self.metadata['complete'] = bool(complete)
            self.archive.writestr('metadata.json', json.dumps(self.metadata, indent=2))
        finally:
            self.archive.close()
            if self.csv_file is not None:
                self.csv_file.close()
            if self.video is not None:
                self.video.release()
