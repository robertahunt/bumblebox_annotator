"""Category-specific mask inventory and explicit completion, independent of exports."""

import hashlib
import json
import os
from pathlib import Path
import re

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from core.annotation_policy import category_policies
from core.categories import CATEGORIES, STANDARD_BROOD_CATEGORIES, QUEEN_BROOD_CATEGORIES, category_label
from core.contributors import utc_now


VERSION = 1
TASKS = {c: (c,) for c in CATEGORIES}
TASKS.update(brood=STANDARD_BROOD_CATEGORIES, queen_brood=QUEEN_BROOD_CATEGORIES)
TASKS['bee_hive'] = ('bee', 'hive')


def task_label(task):
    return {'brood': 'Brood (all stages)', 'queen_brood': 'Queen brood (all stages)',
            'bee_hive': 'Bees + Hive (nest)',
            None: 'All mask types'}.get(task, category_label(task))


def task_categories(task):
    if task is None:
        return CATEGORIES
    if task not in TASKS:
        raise ValueError('Unknown annotation task: ' + str(task))
    return TASKS[task]


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _records(value):
    records = value.get('annotations', []) if isinstance(value, dict) else value
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError('Invalid annotation metadata')
    return records


class Sources:
    """Read-through cache; remote readers verify objects before returning a path."""
    def __init__(self, files, resolve):
        self.files, self.resolve = files, resolve
        self.cache = {}
        self.rasters = {str(Path(n).with_suffix('')).replace('/png/', '/json/')
                        for n in files if n.startswith('annotations/png/')}

    def json(self, name, default=None):
        if name not in self.files:
            return default
        if name not in self.cache:
            with self.resolve(name).open() as stream:
                self.cache[name] = json.load(stream)
        return self.cache[name]

    def mask(self, name):
        key = ('mask', name)
        if key not in self.cache:
            if name not in self.files:
                raise ValueError('Mask pixels unavailable: ' + name)
            pixels = cv2.imread(str(self.resolve(name)), cv2.IMREAD_UNCHANGED)
            if pixels is None or pixels.ndim != 2:
                raise ValueError('Unreadable mask: ' + name)
            self.cache[key] = pixels
        return self.cache[key]

    def digest(self, name):
        if name not in self.files:
            return None
        record = self.files[name]
        if isinstance(record, dict) and 'sha256' in record:
            return record['sha256']
        from core.project_sync import _digest
        key = ('digest', name)
        if key not in self.cache:
            self.cache[key] = _digest(self.resolve(name))
        return self.cache[key]


def _mask_evidence(source, name, category, *, shared=False):
    rows = _records(source.json(name, []))
    geometry = []
    for row in rows:
        if row.get('category', 'chamber' if shared else 'bee') != category or row.get('bbox_only', False):
            continue
        identity = row.get('mask_id', row.get('instance_id', 0))
        if not isinstance(identity, int) or identity <= 0:
            raise ValueError('Invalid mask instance ID in ' + name)
        png = name.replace('/json/', '/png/').removesuffix('.json')
        # Match the loaders: shared masks use the category raster, while frame
        # masks use exact RLE when present and require their companion PNG.
        rle = None if shared else row.get('mask_coco_rle')
        if rle is not None:
            if png + '.png' not in source.files:
                raise ValueError('Frame mask raster unavailable: ' + png + '.png')
            rle = dict(rle)
            h, w = rle['size']
            if isinstance(rle['counts'], list):
                encoded = coco_mask.frPyObjects(rle, h, w)
            else:
                encoded = dict(rle, counts=rle['counts'].encode('ascii'))
            area = int(coco_mask.area(encoded))
            digest = _hash(rle)
        else:
            if shared and png + '_' + category + '.png' in source.files:
                png += '_' + category
            elif shared and any(png + '_' + c + '.png' in source.files for c in ('chamber', 'hive', 'pollen')):
                raise ValueError('Category mask raster unavailable: ' + category)
            pixels = source.mask(png + '.png')
            binary = pixels == identity
            area = int(np.count_nonzero(binary))
            digest = hashlib.sha256(binary.tobytes()).hexdigest()
        if area:
            # Keypoint edits also change a finished frame; rows without keypoints
            # keep their existing evidence (and so their completion signature).
            if row.get('keypoints'):
                digest += ':keypoints:' + _hash(row['keypoints'])
            geometry.append(digest)
    return sorted(geometry)


def _unit(source, video, frame, category, policy):
    stem = 'video_annotations' if frame is None else f'frame_{frame:06d}'
    name = f'annotations/json/{video}/{stem}.json'
    try:
        geometry = _mask_evidence(source, name, category, shared=frame is None)
        # A raster without its category/instance metadata cannot establish absence.
        if name not in source.files and (name.removesuffix('.json') in source.rasters or
                name.removesuffix('.json') + '_' + category in source.rasters):
            raise ValueError('Mask metadata unavailable')
        signature = _hash([policy, geometry])
        record_name = 'video_work' if frame is None else f'frame_{frame:06d}'
        completed = source.json(f'annotations/work/{video}/{record_name}.json', {})
        if completed and (not isinstance(completed, dict) or completed.get('version') != VERSION or
                          not isinstance(completed.get('categories'), dict)):
            raise ValueError('Unsupported annotation completion record')
        completion = completed.get('categories', {}).get(category)
        status = 'not_checked' if geometry else 'not_started'
        if completion:
            status = 'finished' if completion.get('signature') == signature else 'changed'
        return {'count': len(geometry), 'signature': signature, 'status': status}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        return {'count': None, 'signature': None, 'status': 'unknown', 'error': str(exc)}


def _pending_imports(source, video, frames, policies):
    records = source.json('annotations/import_review.json', {})
    pending = {}
    shared = records.get('shared', {}).get(video, {})
    for index in frames:
        entry = records.get('frames', {}).get(video, {}).get(str(index), {})
        requirements = dict(shared, **entry.get('requirements', {}))
        if not requirements:
            continue
        completion = entry.get('completion')
        valid = False
        if completion:
            policy = {c: policies[c] for c in requirements if c in policies}
            names = [f'frames/{video}/frame_{index:06d}.jpg']
            names += [f'annotations/{kind}/{video}/frame_{index:06d}.{suffix}'
                      for kind, suffix in (('json', 'json'), ('bbox', 'json'), ('png', 'png'))]
            if any(p['scope'] == 'video' for p in policy.values()):
                names.append(f'annotations/json/{video}/video_annotations.json')
                names += [f'annotations/png/{video}/video_annotations_{c}.png'
                          for c, p in policy.items() if p['scope'] == 'video']
                names.append(f'annotations/png/{video}/video_annotations.png')
            signature = _hash({'requirements': requirements, 'policy': policy,
                               'files': {n: source.digest(n) for n in names}})
            valid = completion.get('signature') == signature
        if not valid:
            pending[str(index)] = list(requirements)
    return pending


def build_index(files, resolve, *, previous=None, cancelled=lambda: False,
                cached_only=False, progress=lambda message: None, task=None):
    """Build sparse summaries, reusing videos whose saved source records match."""
    if cancelled():
        raise InterruptedError('Annotation summary cancelled')
    source = Sources(files, resolve)
    policies = category_policies(source.json('annotations/project.json', {}))
    requested = task_categories(task)
    videos = {}
    video_files = {}
    for name in files:
        parts = Path(name).parts
        if len(parts) >= 3 and parts[0] == 'frames':
            video_files.setdefault(parts[1], {})[name] = files[name]
        elif len(parts) >= 4 and parts[:1] == ('annotations',) and parts[1] in ('json', 'png', 'bbox', 'work', 'pkl', 'frames'):
            video_files.setdefault(parts[2], {})[name] = files[name]
        match = re.fullmatch(r'frames/([^/]+)/frame_(\d+)\.jpg', name)
        if match:
            videos.setdefault(match[1], []).append(int(match[2]))
    old = previous.get('videos', {}) if previous and previous.get('version') == VERSION else {}
    result = {'version': VERSION, 'videos': {}}
    for position, (video, indices) in enumerate(sorted(videos.items()), 1):
        if cancelled():
            raise InterruptedError('Annotation summary cancelled')
        indices = sorted(indices)
        relevant = dict(video_files[video])
        if 'annotations/import_review.json' in files:
            relevant['annotations/import_review.json'] = files['annotations/import_review.json']
        fingerprint = _hash([policies, relevant])
        unchanged = old.get(video, {}).get('fingerprint') == fingerprint
        if unchanged and set(requested) <= set(old[video].get('categories', {})):
            result['videos'][video] = old[video]
            continue
        if cached_only:
            result['videos'][video] = {'scan_state': 'not_scanned',
                                      'error': 'Mask status has not been scanned for this published version'}
            continue
        progress(f'Scanning mask status: video {position}/{len(videos)} | {video}')
        # A local task scan is partial. Never let it stand in for unchecked
        # categories in a full publication or completion check.
        retained = dict(old[video].get('categories', {})) if unchanged else {}
        entry = {'fingerprint': fingerprint, 'frames': indices, 'categories': retained}
        scan_policies = {c: policies[c] for c in requested if c not in retained}
        try:
            entry['import_pending'] = _pending_imports(source, video, indices, policies)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            entry['error'] = 'Import review unavailable: ' + str(exc)
            entry['import_pending'] = {}
        missing_shared = set()
        for category, policy in scan_policies.items():
            units = {}
            if policy['scope'] == 'video':
                shared_name = f'annotations/json/{video}/video_annotations.json'
                try:
                    shared_rows = _records(source.json(shared_name, []))
                    if not any(r.get('category', 'chamber') == category for r in shared_rows):
                        missing_shared.add(category)
                except (OSError, ValueError, TypeError):
                    pass  # The unit itself retains the unreadable-source diagnostic.
                units['video'] = _unit(source, video, None, category, policy)
            entry['categories'][category] = {'scope': policy['scope'], 'policy': policy, 'units': units}
        source.cache = {k: v for k, v in source.cache.items() if not isinstance(k, tuple) or k[0] != 'mask'}
        # At most one frame's legacy raster is resident, regardless of video length.
        frame_policies = {c: p for c, p in scan_policies.items() if p['scope'] == 'frame'}
        scan_frames = indices if frame_policies or missing_shared else []
        for frame_position, i in enumerate(scan_frames, 1):
            if cancelled():
                raise InterruptedError('Annotation summary cancelled')
            if frame_position == 1 or frame_position % 25 == 0:
                progress(f'Scanning mask status: video {position}/{len(videos)} | {video}\n'
                         f'Frame {frame_position}/{len(indices)} (source frame {i})')
            for category, policy in frame_policies.items():
                unit = _unit(source, video, i, category, policy)
                if unit['status'] != 'not_started':
                    entry['categories'][category]['units'][str(i)] = unit
            frame_json = f'annotations/json/{video}/frame_{i:06d}.json'
            try:
                # Shared masks that exist need no frame-by-frame mask decoding.
                # Missing shared masks still need the legacy-scope safety check.
                if not missing_shared:
                    continue
                legacy = {r.get('category', 'bee') for r in _records(source.json(frame_json, [])) if not r.get('bbox_only')}
                for category in legacy & missing_shared:
                    entry['categories'][category]['units']['video'] = {
                        'count': None, 'signature': None, 'status': 'unknown',
                        'error': 'Legacy frame-local mask in a shared category; review its scope'}
            except (OSError, ValueError, TypeError):
                for category in missing_shared:
                    entry['categories'][category]['units']['video'] = {
                        'count': None, 'signature': None, 'status': 'unknown', 'error': 'Unreadable frame metadata'}
            finally:
                source.cache = {k: v for k, v in source.cache.items() if not isinstance(k, tuple) or k[0] != 'mask'}
                source.cache.pop(frame_json, None)
                source.cache.pop(f'annotations/work/{video}/frame_{i:06d}.json', None)
        if any(n.startswith((f'annotations/pkl/{video}/', f'annotations/frames/{video}/')) for n in relevant):
            entry['error'] = 'Legacy annotations need conversion before mask availability can be determined'
        result['videos'][video] = entry
        source.cache = {k: v for k, v in source.cache.items()
                        if k in ('annotations/project.json', 'annotations/import_review.json')}
    return result


def _local_video_files(project, video, cancelled):
    """Inventory only this video's summary sources, never the whole project."""
    from core.project_sync import _safe_child
    if not isinstance(video, str) or not video or Path(video).name != video or video in ('.', '..'):
        raise ValueError('Invalid video ID')
    files = {}

    def add(path):
        if cancelled():
            raise InterruptedError('Annotation summary cancelled')
        if path.is_symlink() or not path.is_file():
            raise ValueError('Annotation summaries require ordinary files: ' + str(path))
        stat = path.stat()
        files[path.relative_to(project).as_posix()] = (
            stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)

    for name in ('annotations/project.json', 'annotations/import_review.json'):
        path = _safe_child(project, name)
        if path.exists():
            add(path)
    names = [f'frames/{video}'] + [f'annotations/{kind}/{video}'
             for kind in ('json', 'png', 'bbox', 'work', 'pkl', 'frames')]
    def failed(error):
        raise error

    for name in names:
        base = _safe_child(project, name)
        if not base.exists():
            continue
        for directory, directories, filenames in os.walk(base, followlinks=False, onerror=failed):
            if cancelled():
                raise InterruptedError('Annotation summary cancelled')
            for child in directories:
                if (Path(directory) / child).is_symlink():
                    raise ValueError('Annotation summaries do not follow symbolic links')
            for filename in filenames:
                if not filename.startswith('.'):
                    add(Path(directory) / filename)
    return files


def local_index(project, *, previous=None, cancelled=lambda: False, video=None,
                videos=None, task=None, video_ready=lambda video, entry: None):
    from core.project_sync import _safe_child
    project = Path(project)
    if video is not None:
        if videos is not None:
            raise ValueError('Choose video or videos, not both')
        videos = [video]
    if videos is None:
        frames = _safe_child(project, 'frames')
        videos = sorted(p.name for p in frames.iterdir() if p.is_dir()) if frames.exists() else []
    result = {'version': VERSION, 'videos': {}}
    for name in dict.fromkeys(videos):
        if cancelled():
            raise InterruptedError('Annotation summary cancelled')
        files = _local_video_files(project, name, cancelled)
        index = build_index(files, lambda n: _safe_child(project, n), previous=previous,
                            cancelled=cancelled, task=task)
        entry = index['videos'].get(name)
        if entry is not None:
            result['videos'][name] = entry
            video_ready(name, entry)
    return result


def published_index(snapshot, *, cached_only=False, progress=lambda message: None,
                    cancelled=lambda: False):
    """Old publications can be inspected without downloading frame images."""
    from core.project_sync import snapshot_file, _digest
    manifest = snapshot['manifest']
    index = manifest.get('annotation_work')
    def resolve(name):
        if cancelled():
            raise InterruptedError('Annotation summary cancelled')
        path = snapshot_file(snapshot['manifest_path'], manifest, name)
        if _digest(path) != manifest['files'][name]['sha256']:
            raise ValueError('Published annotation metadata is damaged: ' + name)
        return path
    # Older clients may carry forward this optional field while changing files.
    # Source fingerprints, not the mere presence of an index, establish freshness.
    index = build_index(manifest['files'], resolve, previous=index, cached_only=cached_only,
                        progress=progress, cancelled=cancelled)
    if cancelled():
        raise InterruptedError('Annotation summary cancelled')
    if not cached_only:
        manifest['annotation_work'] = index
    return index


def unit_for(entry, category, frame):
    data = entry['categories'][category]
    key = 'video' if data['scope'] == 'video' else str(frame)
    if key in data['units']:
        return data['units'][key]
    return {'count': 0, 'signature': _hash([data['policy'], []]), 'status': 'not_started'}


def summarize(entry, task, frames=None):
    if entry and entry.get('scan_state') == 'not_scanned':
        return {'presence': 'unknown', 'state': 'not_scanned', 'mask_text': 'Not scanned',
                'work_text': 'Not scanned',
                'detail': 'No up-to-date published mask summary. Use Scan Mask Status to check this video.'}
    if not entry or entry.get('error'):
        return {'presence': 'unknown', 'state': 'unknown', 'mask_text': 'Unknown',
                'work_text': 'Unknown', 'detail': (entry or {}).get('error', 'Summary unavailable')}
    categories = task_categories(task)
    if not set(categories) <= set(entry.get('categories', {})):
        return {'presence': 'unknown', 'state': 'not_scanned', 'mask_text': 'Not scanned',
                'work_text': 'Not scanned', 'detail': 'This mask type has not been checked yet.'}
    frames = list(entry['frames'] if frames is None else frames)
    if not frames or not set(frames) <= set(entry['frames']):
        return summarize(None, task)
    shared_categories = [c for c in categories if entry['categories'][c]['scope'] == 'video']
    frame_categories = [c for c in categories if c not in shared_categories]
    shared = not frame_categories
    mixed_scope = bool(shared_categories and frame_categories)
    # Display summaries never use completion signatures. In particular, do not
    # hash the same empty policy for every unannotated frame/category.
    def display_unit(category, frame):
        data = entry['categories'][category]
        key = 'video' if data['scope'] == 'video' else str(frame)
        return data['units'].get(key, {'count': 0, 'status': 'not_started'})
    units = [[display_unit(c, frame) for c in categories] for frame in ([frames[0]] if shared else frames)]
    unknown = any(u['count'] is None for row in units for u in row)
    present = sum(any(u['count'] for u in row) for row in units)
    finished = sum(all(u['status'] == 'finished' for u in row) for row in units)
    pending = sum(bool(set(entry.get('import_pending', {}).get(str(i), [])) & set(categories)) for i in frames)
    total = len(units)
    state = ('unknown' if unknown else 'needs_review' if pending else 'finished' if finished == total
             else 'changed' if any(u['status'] == 'changed' for row in units for u in row)
             else 'not_checked' if present or finished else 'not_started')
    presence = 'unknown' if unknown else 'none' if not present else 'all' if present == total else 'some'
    mask_text = ('Unknown' if unknown else 'Shared mask' if shared and present else 'No masks' if not present
                 else f'Masks on {present}/{total} frames')
    work_text = {'unknown': 'Unknown', 'needs_review': f'Import review: {pending} frames',
                 'finished': 'Finished' if present else 'Checked: none present', 'changed': 'Changed; recheck',
                 'not_started': 'Not started', 'not_checked': 'Not checked' if not finished else f'{finished}/{total} finished'}[state]
    detail = 'Shared across video' if shared else 'Frame-specific'
    if mixed_scope:
        detail = ('Frame-specific: ' + ', '.join(category_label(c) for c in frame_categories) +
                  '; shared across video: ' + ', '.join(category_label(c) for c in shared_categories))
    if unknown:
        detail = '; '.join(sorted({u['error'] for row in units for u in row if u.get('error')})) or 'Mask data unavailable'
    return dict(presence=presence, state=state, mask_text=mask_text, work_text=work_text,
                finished=finished, total=total, shared=shared, mixed_scope=mixed_scope, detail=detail)


def matches(summary, show='all', presence='any'):
    state = summary['state']
    # Unverified or changed work stays discoverable without calling it untouched.
    work = (show == 'all' or show == 'finished' and state == 'finished'
            or show == 'not_started' and state == 'not_started'
            or show == 'not_checked' and state in ('not_checked', 'changed', 'needs_review', 'unknown', 'not_scanned'))
    masks = (presence == 'any' or presence == 'has' and summary['presence'] in ('some', 'all')
             or presence == 'none' and summary['presence'] == 'none'
             or presence == 'some' and summary['presence'] == 'some'
             or presence == 'unknown' and summary['presence'] == 'unknown')
    return work and masks


def mark_finished(project, video, frame, task, contributor, *, complete=True, cancelled=lambda: False):
    from core.assigned_work import require_assigned_frame
    from core.project_sync import _safe_child, _read_json, _write_json, require_working_project, require_no_update
    require_working_project(project)
    require_no_update(project)
    require_assigned_frame(project, contributor, video, frame)
    if task is None:
        raise ValueError('Choose a mask type before marking work finished')
    entry = local_index(project, video=video, task=task, cancelled=cancelled)['videos'].get(video)
    if not entry or frame not in entry['frames']:
        raise ValueError('Choose an available frame')
    if entry.get('error'):
        raise ValueError(entry['error'])
    updates = {}
    for category in task_categories(task):
        data = entry['categories'][category]
        unit = unit_for(entry, category, frame)
        if complete and (not unit['signature'] or data['policy']['meaning'] == 'unconfirmed'):
            raise ValueError('Resolve missing mask data and confirm the annotation policy before marking work finished')
        stem = 'video_work' if data['scope'] == 'video' else f'frame_{frame:06d}'
        path = _safe_child(Path(project), f'annotations/work/{video}/{stem}.json')
        if path not in updates:
            updates[path] = _read_json(path) if path.exists() else {'version': VERSION, 'categories': {}}
            if updates[path].get('version') != VERSION:
                raise ValueError('Unsupported annotation completion record')
        if complete:
            updates[path]['categories'][category] = {'signature': unit['signature'], 'at': utc_now(),
                'by': {k: contributor[k] for k in ('id', 'name')} if contributor else None,
                'outcome': 'masks' if unit['count'] else 'none_present'}
        else:
            updates[path]['categories'].pop(category, None)
    if cancelled():
        raise InterruptedError('Completion cancelled; no completion records changed')
    for path, value in updates.items():
        _safe_child(Path(project), path.parent.relative_to(project), mkdir=True)
        _write_json(path, value)
