"""Historical hive overlays and a compact, model-free replay archive."""

import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

from core.temporal_hive_prior import TemporalHiveSnapshot


HIVE_OVERLAY_MODES = ('current', 'temporal', 'compare', 'updated')
OVERLAY_TIMINGS = ('before_current_frame_update', 'after_current_frame_update')
PRIOR_COLOR = (170, 195, 35)  # BGR: teal
CURRENT_COLOR = (210, 70, 210)


def resolve_hive_overlay_mode(mode, prior):
    """Resolve the scoring-linked display choice before rendering or naming outputs."""
    if mode == 'scored':
        if prior is None:
            return 'current'
        return 'updated' if prior.scoring_mode == 'updated' else 'temporal'
    if mode not in HIVE_OVERLAY_MODES:
        raise ValueError(f'Unknown hive overlay mode: {mode}')
    return mode


def project_snapshot(snapshot, frame_shape):
    """Project categorical grid cells through the same chamber crop used for scoring."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = snapshot.bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Temporal snapshot bounding box is outside the source frame')
    labels = np.zeros((height, width), dtype=np.uint8)
    labels[y1:y2, x1:x2] = cv2.resize(
        snapshot.labels, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST,
    )
    return labels


def draw_temporal_hive_overlay(frame, snapshots, current_masks=None, compare=False):
    """Draw only supported hive; never substitute current detections for missing history."""
    hive = np.zeros(frame.shape[:2], dtype=bool)
    supported = False
    for snapshot in snapshots:
        labels = project_snapshot(snapshot, frame.shape)
        supported = supported or bool(np.any(labels > 0))
        hive |= labels == 2
    frame[hive] = np.rint(frame[hive] * 0.82 + np.array(PRIOR_COLOR) * 0.18).astype(np.uint8)
    contours, _ = cv2.findContours(hive.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_width = max(1, round(min(frame.shape[:2]) / 720))
    cv2.drawContours(frame, contours, -1, PRIOR_COLOR, line_width, lineType=cv2.LINE_AA)
    if compare:
        for mask in (current_masks or {}).values():
            if mask is not None and mask.shape[:2] == frame.shape[:2]:
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, CURRENT_COLOR, line_width, lineType=cv2.LINE_AA)
    return supported


def draw_hive_overlay_label(frame, mode, supported):
    label = 'Hive: temporal prior (past frames)'
    if mode == 'updated':
        label = 'Hive: history + current evidence'
    elif mode == 'compare':
        label = 'Hive: prior (teal), current (magenta)'
    if not supported:
        label += ' | Insufficient evidence' if mode == 'updated' else ' | No supported history'
    # Keep the legend legible when a full-resolution video is fit to a player.
    factor = max(1.0, min(frame.shape[:2]) / 720)
    thickness = max(1, round(factor))
    margin = round(8 * factor)
    width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * factor, thickness)[0][0]
    scale = 0.55 * factor * min(1.0, max(1, frame.shape[1] - 2 * margin) / max(1, width))
    height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][1]
    top = max(0, frame.shape[0] - height - round(12 * factor))
    cv2.rectangle(frame, (0, top), (frame.shape[1] - 1, frame.shape[0] - 1), (25, 25, 25), -1)
    cv2.putText(frame, label, (margin, frame.shape[0] - round(7 * factor)), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (245, 245, 245), thickness, cv2.LINE_AA)


class TemporalHiveOverlayWriter:
    """One ZIP per video; each member is a compressed, pickle-free frame snapshot."""

    def __init__(self, path, video_path, context_id, fps=None, provenance='live_inference', prior=None,
                 timing='before_current_frame_update'):
        if timing not in OVERLAY_TIMINGS:
            raise ValueError(f'Unsupported temporal overlay timing: {timing}')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.archive = zipfile.ZipFile(self.path, 'w', compression=zipfile.ZIP_STORED)
        self.metadata = {
            'version': 1, 'timing': timing,
            'video_path': str(Path(video_path).expanduser().resolve()), 'context_id': context_id,
            'fps': fps, 'frame_count': 0,
            'provenance': provenance,
        }
        if prior is not None:
            self.metadata['prior_settings'] = {
                'window_seconds': prior.window_seconds,
                'resolution': list(prior.resolution),
                'hive_probability_threshold': prior.hive_probability_threshold,
                'min_prior_weight': prior.min_prior_weight,
                'cleanup_kernel_size': prior.cleanup_kernel_size,
                'min_component_pixels': prior.min_component_pixels,
                'stabilize_chambers': prior.stabilize_chambers,
                'scoring_mode': prior.scoring_mode,
            }

    def write(self, frame_number, frame_shape, snapshots):
        if frame_number != self.metadata['frame_count'] + 1:
            raise ValueError('Temporal overlay frames must be consecutive and start at 1')
        if 'frame_shape' in self.metadata and list(frame_shape[:2]) != self.metadata['frame_shape']:
            raise ValueError('Temporal overlay source frame size changed')
        self.metadata['frame_shape'] = list(frame_shape[:2])
        data = io.BytesIO()
        np.savez_compressed(
            data,
            chamber_ids=np.array([s.chamber_id for s in snapshots], dtype=np.int64),
            bboxes=np.array([s.bbox for s in snapshots], dtype=np.int64).reshape(-1, 4),
            labels=(np.stack([s.labels for s in snapshots]) if snapshots
                    else np.empty((0, 0, 0), dtype=np.uint8)),
        )
        self.archive.writestr(f'frame_{frame_number:08d}.npz', data.getvalue())
        self.metadata['frame_count'] = frame_number

    def close(self, complete=True):
        if self.archive is not None:
            self.metadata['complete'] = bool(complete)
            try:
                self.archive.writestr('metadata.json', json.dumps(self.metadata))
            finally:
                self.archive.close()
                self.archive = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(complete=exc_type is None)


class TemporalHiveOverlayReader:
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        try:
            self.metadata = json.loads(self.archive.read('metadata.json'))
            if self.metadata.get('version') != 1 or self.metadata.get('timing') not in OVERLAY_TIMINGS:
                raise ValueError('Unsupported temporal overlay archive format')
        except Exception:
            self.archive.close()
            raise

    @property
    def overlay_mode(self):
        return 'updated' if self.metadata['timing'] == 'after_current_frame_update' else 'temporal'

    def read(self, frame_number):
        with np.load(io.BytesIO(self.archive.read(f'frame_{frame_number:08d}.npz')),
                     allow_pickle=False) as data:
            ids, bboxes, labels = data['chamber_ids'], data['bboxes'], data['labels']
            if (ids.ndim != 1 or bboxes.shape != (len(ids), 4) or labels.ndim != 3
                    or len(labels) != len(ids) or labels.dtype != np.uint8 or np.any(labels > 2)
                    or (len(ids) and min(labels.shape[1:]) < 1)):
                raise ValueError('Invalid temporal overlay snapshot')
            return [TemporalHiveSnapshot(int(cid), tuple(int(v) for v in box), grid.copy())
                    for cid, box, grid in zip(ids, bboxes, labels)]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.archive.close()
