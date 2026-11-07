"""Read-only, bounded-memory previews of the two pinned annotation versions."""

from collections import Counter
import json

import cv2
import numpy as np

from core.categories import CATEGORY_COLORS
from core.coco_masks import decode_segmentation
from core.project_delta import bytes_record
from core.project_sync import _safe_child, snapshot_file


def _read(plan, name, version):
    records = plan.local if version == 'local' else plan.original['manifest']['files']
    record = records.get(name)
    if record is None:
        return None
    path = (_safe_child(plan.project, name) if version == 'local' else
            snapshot_file(plan.original['manifest_path'], plan.original['manifest'], name))
    data = path.read_bytes()
    if bytes_record(data) != record:
        raise ValueError('Annotation changed or is damaged. Reopen the comparison: ' + name)
    return data


def _annotations(data):
    if data is None:
        return []
    result = json.loads(data)
    if isinstance(result, dict):
        result = result['annotations']
    if not isinstance(result, list) or any(not isinstance(a, dict) for a in result):
        raise ValueError('Invalid annotation metadata.')
    return result


def _overlay(plan, frame, version, shape, categories, cancelled):
    height, width = shape
    overlay = np.zeros((height, width, 4), np.uint8)
    counts = Counter()
    video = plan.video
    # One label PNG and one instance mask at a time, not N full-size masks.
    for stem, shared in (('video_annotations', True), (f'frame_{frame:06d}', False)):
        annotations = _annotations(_read(plan, f'annotations/json/{video}/{stem}.json', version))
        if not shared:
            annotations += [a for a in _annotations(_read(plan, f'annotations/bbox/{video}/{stem}.json', version))
                            if a.get('bbox_only') and not a.get('from_mask')]
        label_name, label_map = None, None
        for annotation in sorted(annotations, key=lambda a: a.get('category', 'chamber' if shared else 'bee')):
            if cancelled():
                raise InterruptedError('Comparison cancelled.')
            category = annotation.get('category', 'chamber' if shared else 'bee')
            if categories is not None and category not in categories:
                continue
            color = (*CATEGORY_COLORS.get(category, (60, 175, 255)), 255)
            if annotation.get('bbox_only'):
                bbox = annotation.get('bbox')
                if bbox and len(bbox) == 4:
                    x, y, w, h = (int(v) for v in bbox)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, max(1, width // 1500))
                    counts[category] += 1
                continue
            identity = annotation.get('mask_id', annotation.get('instance_id', 0))
            if identity <= 0:
                continue
            name = f'annotations/png/{video}/{stem}_{category}.png' if shared else f'annotations/png/{video}/{stem}.png'
            records = plan.local if version == 'local' else plan.original['manifest']['files']
            if shared and name not in records:
                name = f'annotations/png/{video}/{stem}.png'
            if label_name != name:
                data = _read(plan, name, version)
                label_map = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) if data else None
                if label_map is None or label_map.ndim != 2 or label_map.shape != shape:
                    raise ValueError('Missing or incompatible mask image: ' + name)
                label_name = name
            mask = (decode_segmentation(annotation['mask_coco_rle'], height, width).astype(bool)
                    if 'mask_coco_rle' in annotation else label_map == identity)
            overlay[mask] = color
            counts[category] += 1
    return {'overlay': overlay, 'counts': dict(counts)}


def comparison_preview(plan, frame, categories=None, progress=lambda text: None, cancelled=lambda: False):
    if frame not in plan.entry['frames']:
        raise ValueError('Choose an assigned frame for comparison.')
    if cancelled():
        raise InterruptedError('Comparison cancelled.')
    name = f'frames/{plan.video}/frame_{frame:06d}.jpg'
    # Both panes always use the exact submitted image, never different originals.
    data = _read(plan, name, 'submitted')
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        raise ValueError('The submitted frame image cannot be read.')
    result = {'image': cv2.cvtColor(image, cv2.COLOR_BGR2RGB)}
    for version in ('local', 'submitted'):
        progress(f'Loading {version} annotations for {plan.video}, frame {frame}')
        try:
            result[version] = _overlay(plan, frame, version, image.shape[:2], categories, cancelled)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result[version] = {'error': str(exc)}
    return result
