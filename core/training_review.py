"""A resumable, explicit opt-in queue for previously unselected bee frames."""

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from core.assigned_work import require_full_project
from core.contributors import utc_now
from core.import_review import require_reviewed
from core.project_manager import ProjectManager
from core.project_sync import _read_json, _write_json, require_no_update
from core.validation_review import frame_has_bee_mask_annotation


QUEUE_FILE = 'annotations/training_review.json'
STATES = {'pending', 'complete', 'incomplete'}


def _key(item):
    video, frame = item['video'], item['frame']
    if (not isinstance(video, str) or video in ('', '.', '..')
            or '/' in video or '\\' in video or type(frame) is not int or frame < 0):
        raise ValueError('Invalid review frame.')
    return video, frame


def _metadata_path(project, video):
    return Path(project) / 'frames' / video / 'video_metadata.json'


def read_queue(project):
    path = Path(project) / QUEUE_FILE
    if not path.exists():
        return None
    queue = _read_json(path)
    if queue.get('version') != 1 or not isinstance(queue.get('items'), list):
        raise ValueError('Unsupported training review queue; existing progress was preserved.')
    seen = set()
    for item in queue['items']:
        key = _key(item)
        if key in seen or item.get('status') not in STATES:
            raise ValueError('Invalid training review queue; existing progress was preserved.')
        seen.add(key)
    return queue


def _require_training_frame(project, video, frame):
    metadata = _read_json(_metadata_path(project, video))
    # Check both the selection metadata and the actual split used by exports.
    if (metadata.get('split') != 'train'
            or ProjectManager(project).get_video_split(video) != 'train'):
        raise ValueError('This video is no longer in the training split. No selection changed.')
    return metadata


def discover_candidates(project):
    """List unselected training frames with saved bee masks and source images."""
    project = Path(project)
    videos = ProjectManager(project).get_videos_by_split('train')
    items = []
    for video in sorted(videos):
        meta = _read_json(_metadata_path(project, video))
        if meta.get('split') != 'train':
            continue
        selected = set(meta.get('selected_frames', []))
        for file in sorted((project / 'annotations/json' / video).glob('frame_*.json')):
            frame = int(file.stem.split('_')[1])
            if (frame not in selected
                    and (project / 'frames' / video / f'{file.stem}.jpg').is_file()
                    and frame_has_bee_mask_annotation(project, video, frame)):
                items.append({'video': video, 'frame': frame})
    return items


def create_queue(project, candidates=None, *, title='Unselected bee frames'):
    """Create once; opening the queue must never reset previous decisions."""
    project = Path(project)
    require_full_project(project)
    require_no_update(project)
    existing = read_queue(project)
    if existing is not None:
        return existing
    candidates = discover_candidates(project) if candidates is None else candidates
    items = []
    seen = set()
    for candidate in candidates:
        video, frame = _key(candidate)
        if (video, frame) in seen:
            continue
        seen.add((video, frame))
        metadata = _require_training_frame(project, video, frame)
        if frame in metadata.get('selected_frames', []):
            continue
        if not (project / 'frames' / video / f'frame_{frame:06d}.jpg').is_file():
            raise ValueError(f'Missing source image: {video}, frame {frame}')
        if not frame_has_bee_mask_annotation(project, video, frame):
            raise ValueError(f'Missing saved bee masks: {video}, frame {frame}')
        items.append({'video': video, 'frame': frame, 'status': 'pending'})
    queue = {'version': 1, 'title': title, 'created_at': utc_now(),
             'items': sorted(items, key=lambda x: (x['video'], x['frame']))}
    path = project / QUEUE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, queue)
    return queue


def _nonempty_bees(project, video, frame):
    stem = f'frame_{frame:06d}'
    data = _read_json(Path(project) / 'annotations/json' / video / f'{stem}.json')
    raster = None
    for ann in data:
        if ann.get('category', 'bee') != 'bee' or ann.get('bbox_only'):
            continue
        rle = ann.get('mask_coco_rle')
        if rle:
            rle = dict(rle)
            if isinstance(rle['counts'], str):
                rle['counts'] = rle['counts'].encode('ascii')
            if coco_mask.area(rle) > 0:
                return True
        else:
            mask_id = int(ann.get('mask_id', ann.get('instance_id', 0)) or 0)
            if mask_id <= 0:
                continue
            if raster is None:
                raster = cv2.imread(str(Path(project) / 'annotations/png' / video / f'{stem}.png'),
                                    cv2.IMREAD_UNCHANGED)
            if raster is not None and np.any(raster == mask_id):
                return True
    return False


def record_decision(project, video, frame, complete, contributor=None):
    """Change only this frame's selection, preserving every other metadata field."""
    project = Path(project)
    _key({'video': video, 'frame': frame})
    require_full_project(project)
    require_no_update(project)
    queue = read_queue(project)
    if queue is None:
        raise ValueError('No training review queue exists.')
    item = next((i for i in queue['items'] if _key(i) == (video, frame)), None)
    if item is None:
        raise ValueError('This frame is outside the review queue.')
    metadata = _require_training_frame(project, video, frame)
    if complete:
        if not (project / 'frames' / video / f'frame_{frame:06d}.jpg').is_file():
            raise ValueError('The source image is missing. No selection changed.')
        require_reviewed(project, {video: [frame]})
        if not frame_has_bee_mask_annotation(project, video, frame) or not _nonempty_bees(project, video, frame):
            raise ValueError('No nonempty saved bee masks. No selection changed.')
    selected = set(metadata.get('selected_frames', []))
    before = frame in selected
    selected.add(frame) if complete else selected.discard(frame)
    updated = deepcopy(metadata)
    updated['selected_frames'] = sorted(selected)
    updated['n_selected'] = len(selected)
    item.update(status='complete' if complete else 'incomplete', reviewed_at=utc_now(),
                reviewed_by=deepcopy(contributor), selected_before_decision=before)
    metadata_path = _metadata_path(project, video)
    _write_json(metadata_path, updated)
    try:
        _write_json(project / QUEUE_FILE, queue)
    except Exception:
        # A failed review save must not silently include an unrecorded frame.
        _write_json(metadata_path, metadata)
        raise
    return queue
