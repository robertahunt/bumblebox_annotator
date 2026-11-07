"""Lightweight instance counts from metadata, without decoding mask pixels."""

import json
from pathlib import Path

from core.categories import CATEGORIES


def count_records(records, bbox_only=False):
    counts = dict.fromkeys(CATEGORIES, 0)
    seen = set()
    for index, annotation in enumerate(records or []):
        if not isinstance(annotation, dict):
            continue
        if bbox_only and (not annotation.get('bbox_only', False) or annotation.get('from_mask', False)):
            continue
        category = annotation.get('category', 'bee')
        if category not in counts:
            continue
        identity = annotation.get('mask_id', annotation.get('instance_id', annotation.get('id')))
        key = (category, identity if identity is not None else index)
        if key not in seen:
            seen.add(key)
            counts[category] += 1
    return counts


def file_counts(path, bbox_only, cache):
    path = Path(path)
    key = (str(path), bbox_only)
    try:
        stat = path.stat()
    except FileNotFoundError:
        cache.pop(key, None)
        return dict.fromkeys(CATEGORIES, 0)
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    previous = cache.get(key)
    if previous is not None and previous[0] == signature:
        return dict(previous[1])
    with path.open() as stream:
        data = json.load(stream)
    records = data.get('annotations', []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError('Invalid annotation metadata: ' + str(path))
    counts = count_records(records, bbox_only)
    cache[key] = (signature, dict(counts))
    return counts


def scan_counts(project, previous=None, *, videos=None, cancelled=lambda: False):
    """Return an isolated cache: the UI can save edits while this scan runs."""
    counts = dict.fromkeys(CATEGORIES, 0)
    cache = dict(previous or {})
    found = set()
    root = Path(project) / 'annotations'
    for folder, bbox_only in (('json', False), ('bbox', True)):
        base = root / folder
        directories = (base / video for video in videos) if videos is not None else (
            base.iterdir() if base.exists() else ())
        for directory in directories:
            if cancelled():
                raise InterruptedError('Instance count cancelled')
            if not directory.is_dir():
                continue
            paths = list(directory.glob('frame_*.json'))
            if not bbox_only:
                paths.append(directory / 'video_annotations.json')
            for path in paths:
                if cancelled():
                    raise InterruptedError('Instance count cancelled')
                found.add((str(path), bbox_only))
                for category, value in file_counts(path, bbox_only, cache).items():
                    counts[category] += value
    return counts, {key: value for key, value in cache.items() if key in found}
