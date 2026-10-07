"""Previewed, backed-up copies of editable annotations between local projects."""

from collections import Counter
from copy import deepcopy
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import cv2
import numpy as np

from core.annotation import AnnotationManager
from core.annotation_scope import frame_categories, merge_annotations, read_project_info
from core.categories import CATEGORIES, category_label
from core.contributors import utc_now
from core.project_manager import ProjectManager
from core.project_sync import _digest, _read_json, _safe_child, _write_json


SPLITS = ('train', 'val', 'test', 'inference')


def _records(path):
    if not path.exists():
        return []
    data = _read_json(path)
    records = data.get('annotations') if isinstance(data, dict) else data
    if not isinstance(records, list) or any(not isinstance(a, dict) for a in records):
        raise ValueError(f'Unrecognized annotation metadata: {path}')
    return records


def _frame_index(path):
    return int(path.stem.removeprefix('frame_'))


def _snapshot(project):
    """Detect edits/new files during preview without retaining any image arrays."""
    result = {}
    for name in ('annotations/json', 'annotations/bbox', 'annotations/png',
                 'annotations/pkl', 'frames', 'input_data'):
        base = _safe_child(project, name)
        for path in base.rglob('*'):
            if path.is_symlink():
                raise ValueError(f'Import does not follow symbolic links: {path}')
            if path.is_file():
                stat = path.stat()
                result[path.relative_to(project).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    for name in ('annotations/project.json', 'annotations/project_sync.json'):
        path = _safe_child(project, name)
        if path.exists():
            stat = path.stat()
            result[name] = (stat.st_size, stat.st_mtime_ns)
    return result


def inspect_project(project):
    """List frames and annotation categories using metadata, not decoded masks."""
    project = Path(project).resolve()
    if not (project / 'annotations/project.json').is_file():
        raise ValueError('Choose an annotation project containing annotations/project.json.')
    _snapshot(project)
    info = read_project_info(project)
    manager = ProjectManager(project)
    splits = manager.scan_videos()
    local_categories = frame_categories(info)
    videos = {}
    for directory in sorted((project / 'frames').glob('*')):
        if not directory.is_dir():
            continue
        video = directory.name
        if sum(video in members for members in splits.values()) > 1:
            raise ValueError(f'{video}: this video is present in multiple dataset splits.')
        metadata_file = directory / 'video_metadata.json'
        metadata = _read_json(metadata_file) if metadata_file.exists() else {}
        shared = _records(project / 'annotations/json' / video / 'video_annotations.json')
        frames = {}
        for image in sorted(directory.glob('frame_*.jpg')):
            index = _frame_index(image)
            local = _records(project / 'annotations/json' / video / f'{image.stem}.json')
            boxes = _records(project / 'annotations/bbox' / video / f'{image.stem}.json')
            local += [a for a in boxes if a.get('bbox_only') and not a.get('from_mask')]
            effective = merge_annotations(local, shared, info)
            counts = Counter(a.get('category', 'bee') for a in effective)
            frames[index] = {'path': image, 'counts': dict(counts),
                             'selected': index in metadata.get('selected_frames', []),
                             'local_hive': any(a.get('category') == 'hive' for a in local)
                             if 'hive' in local_categories else False}
        if frames:
            videos[video] = {'frames': frames, 'metadata': metadata,
                             'split': manager.get_video_split(video) or metadata.get('split', 'train'),
                             'video_path': manager.get_video_path(video)}
    if not videos:
        raise ValueError('No extracted frame_*.jpg images found in this project.')
    return {'path': project, 'info': info, 'videos': videos}


def _load_frame(manager, project, video, index):
    stem = f'frame_{index:06d}'
    records = _records(project / 'annotations/json' / video / f'{stem}.json')
    if (project / 'annotations/pkl' / video / f'{stem}.pkl').exists() and not records:
        raise ValueError(f'{video}/{stem}: convert legacy pickle annotations before importing.')
    annotations = manager.load_frame_annotations(project, video, index)
    if records and len([a for a in annotations if 'mask' in a]) != len(records):
        raise ValueError(f'{video}/{stem}: annotation masks are missing or unreadable.')
    return annotations


def _load_shared(manager, project, video):
    records = _records(project / 'annotations/json' / video / 'video_annotations.json')
    annotations, _ = manager.load_video_annotations(project, video)
    if len(records) != len(annotations):
        raise ValueError(f'{video}: unreadable video annotations.')
    for ann in annotations:
        if not ann.get('bbox_only') and 'mask' not in ann:
            raise ValueError(f'{video}: a video-wide annotation mask is missing.')
    return annotations


def _validate_annotations(annotations, shape, shared=False):
    occupied = {}
    ids = set()
    for ann in annotations:
        identity = int(ann.get('mask_id', ann.get('instance_id', 0)))
        key = (ann.get('category', 'bee'), identity)
        if not 0 < identity <= 65535 or key in ids:
            raise ValueError('Annotation IDs must be distinct integers from 1 to 65535 within each category.')
        ids.add(key)
        mask = ann.get('mask')
        if mask is None:
            if not ann.get('bbox_only'):
                raise ValueError('Annotation has neither an exact mask nor a bbox-only flag.')
            continue
        if mask.shape != shape:
            raise ValueError('Image and mask dimensions differ. Import will not resize annotations.')
        if shared:
            category = ann.get('category', 'bee')
            current = mask > 0
            previous = occupied.get(category)
            if previous is not None and np.any(previous & current):
                raise ValueError('Overlapping instances of the same video-wide category cannot be '
                                 'stored without losing pixels. Resolve them in the source first.')
            occupied[category] = current if previous is None else previous | current


def _has_geometry(annotation):
    if 'mask' in annotation:
        return bool(np.any(annotation['mask']))
    box = annotation.get('bbox', [])
    return len(box) == 4 and box[2] > 0 and box[3] > 0


class PreparedImport:
    """Temporary annotation output plus a checked, rollback-capable publication."""

    def __init__(self, source, destination):
        self.source, self.destination = source, destination
        self.temporary = tempfile.TemporaryDirectory(prefix='bumblebox-import-')
        self.staging = Path(self.temporary.name)
        self.files = {}
        self.deletions = set()
        self.rows = []
        self.warnings = []
        self.source_state = _snapshot(source)
        self.destination_state = _snapshot(destination)
        self.identifier = utc_now().replace(':', '-') + '-' + uuid4().hex[:8]
        self.closed = False

    def close(self):
        self.temporary.cleanup()
        self.closed = True

    def check_unchanged(self):
        if self.closed:
            raise ValueError('This import preview has already been closed.')
        if (_snapshot(self.source) != self.source_state or
                _snapshot(self.destination) != self.destination_state):
            raise ValueError('A project changed after the import preview. Preview again before importing.')

    def collect(self, relatives):
        for relative in relatives:
            staged = self.staging / relative
            if staged.is_file():
                self.files[relative] = staged
                self.deletions.discard(relative)
            elif (self.destination / relative).exists():
                self.deletions.add(relative)

    def apply(self, progress=None, cancelled=None):
        progress = progress or (lambda message: None)
        cancelled = cancelled or (lambda: False)
        self.check_unchanged()
        backup = _safe_child(self.destination, Path('import_backups') / self.identifier, mkdir=True)
        incoming = backup / 'incoming'
        originals = backup / 'originals'
        originals.mkdir()
        changes = sorted(set(self.files) | self.deletions)
        manifest = {'source': str(self.source), 'destination': str(self.destination),
                    'created_at': utc_now(), 'rows': self.rows, 'warnings': self.warnings,
                    'originals': {}, 'new_checksums': {}, 'status': 'preparing'}
        try:
            for number, relative in enumerate(changes, 1):
                if cancelled():
                    raise InterruptedError('Import cancelled. Destination annotations were not changed.')
                progress(f'Preparing {number}/{len(changes)}: {relative}')
                target = _safe_child(self.destination, relative)
                if target.exists():
                    old = originals / relative
                    old.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, old)
                    digest = _digest(target)
                    if _digest(old) != digest:
                        raise OSError(f'Backup verification failed: {relative}')
                    manifest['originals'][relative] = digest
                else:
                    manifest['originals'][relative] = None
                if relative in self.files:
                    new = incoming / relative
                    new.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.files[relative], new)
                    digest = _digest(self.files[relative])
                    if _digest(new) != digest:
                        raise OSError(f'Copy verification failed: {relative}')
                    manifest['new_checksums'][relative] = digest
            self.check_unchanged()
            if cancelled():
                raise InterruptedError('Import cancelled. Destination annotations were not changed.')
            manifest['status'] = 'publishing'
            _write_json(backup / 'manifest.json', manifest)
            applied = []
            try:
                for relative in changes:
                    target = _safe_child(self.destination, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    expected = manifest['originals'][relative]
                    actual = _digest(target) if target.exists() else None
                    if actual != expected:
                        raise ValueError(f'Destination changed during publication: {relative}')
                    if relative in self.files:
                        os.replace(incoming / relative, target)
                    else:
                        target.unlink()
                    applied.append(relative)
                manifest['status'] = 'complete'
                _write_json(backup / 'manifest.json', manifest)
            except Exception as exc:
                preserved_edits = []
                for relative in reversed(applied):
                    target = self.destination / relative
                    old = originals / relative
                    actual = _digest(target) if target.exists() else None
                    if actual != manifest['new_checksums'].get(relative):
                        preserved_edits.append(relative)
                        continue
                    if old.exists():
                        shutil.copy2(old, target)
                    else:
                        target.unlink(missing_ok=True)
                manifest['status'] = 'rollback_incomplete' if preserved_edits else 'rolled_back'
                manifest['preserved_external_edits'] = preserved_edits
                _write_json(backup / 'manifest.json', manifest)
                if preserved_edits:
                    raise RuntimeError(f'Another process edited imported files. Those edits were preserved; '
                                       f'review originals in {backup} before retrying.') from exc
                raise
        finally:
            if incoming.exists():
                shutil.rmtree(incoming)
        return {'backup': str(backup), 'files': len(changes), 'rows': self.rows,
                'videos': sorted({row['video'] for row in self.rows})}


def prepare_import(source, destination, selections, categories, *, split='preserve',
                   include_videos=False, replace=False, hive_frames=None, contributor=None,
                   progress=None, cancelled=None):
    """Build editable copies; no source or destination annotation files are changed."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate, non-nested projects.')
    if not (destination / 'annotations/project.json').is_file():
        raise ValueError('Open the destination annotation project first.')
    if not categories or not set(categories) <= set(CATEGORIES):
        raise ValueError('Select at least one supported annotation category.')
    if split not in ('preserve', *SPLITS):
        raise ValueError('Invalid destination split.')
    selected = {video: sorted(set(indices)) for video, indices in selections.items() if indices}
    if not selected:
        raise ValueError('Select at least one frame.')
    catalog = inspect_project(source)
    dest_info = read_project_info(destination)
    dest_local = frame_categories(dest_info)
    source_local = frame_categories(catalog['info'])
    progress = progress or (lambda message: None)
    cancelled = cancelled or (lambda: False)
    plan = PreparedImport(source, destination)
    manager = AnnotationManager()
    dest_manager = ProjectManager(destination)
    identity = source / 'annotations/project_sync.json'
    source_id = _read_json(identity).get('project_id') if identity.exists() else None
    timestamp = utc_now()
    try:
        for video, indices in selected.items():
            if video not in catalog['videos'] or any(i not in catalog['videos'][video]['frames'] for i in indices):
                raise ValueError(f'Unknown source video or frame: {video}')
            if cancelled():
                raise InterruptedError('Import preview cancelled.')
            item = catalog['videos'][video]
            target_split = item['split'] if split == 'preserve' else split
            if target_split not in SPLITS:
                raise ValueError(f'{video}: select an explicit destination split.')
            existing_split = dest_manager.get_video_split(video)
            if sum(video in members for members in dest_manager.scan_videos().values()) > 1:
                raise ValueError(f'{video}: the destination video is present in multiple dataset splits.')
            if existing_split and existing_split != target_split:
                raise ValueError(f'{video} already belongs to {existing_split}, not {target_split}. '
                                 'Choose its existing split; import never splits one video across sets.')
            if item['split'] != target_split:
                plan.warnings.append(f'{video}: {item["split"]} -> {target_split}. '
                                     'Keep this source video out of the opposite split when combining datasets.')
            source_shared = _load_shared(manager, source, video)
            dest_shared = _load_shared(manager, destination, video)
            dest_frames_dir = destination / 'frames' / video
            dest_indices = {_frame_index(p) for p in dest_frames_dir.glob('frame_*.jpg')}
            common = sorted(set(item['frames']) & dest_indices)
            if existing_split or dest_indices or dest_shared:
                if common:
                    reference = common[0]
                    a = cv2.imread(str(item['frames'][reference]['path']))
                    b = cv2.imread(str(dest_manager.get_frame_path(video, reference)))
                    if a is None or b is None or not np.array_equal(a, b):
                        raise ValueError(f'{video}: same video name but different reference image pixels.')
                else:
                    original = dest_manager.get_video_path(video)
                    if (not original or not item['video_path'] or
                            _digest(original) != _digest(item['video_path'])):
                        raise ValueError(f'{video}: cannot verify video identity. Include a source frame '
                                         'also present in the destination or matching original videos.')

            reserved = set()
            legacy_shared = set()
            for storage in ('json', 'bbox'):
                for path in (destination / 'annotations' / storage / video).glob('*.json'):
                    for ann in _records(path):
                        reserved.add(int(ann.get('mask_id', ann.get('instance_id', 0))))
                        if path.name != 'video_annotations.json' and ann.get('category', 'bee') not in dest_local:
                            legacy_shared.add(ann.get('category', 'bee'))
            id_map = {}
            def imported(ann, index, source_scope, target_scope):
                result = dict(ann)
                old_id = int(ann.get('mask_id', ann.get('instance_id', 0)))
                key = (ann.get('category', 'bee'), old_id)
                if key not in id_map:
                    new_id = old_id if 0 < old_id <= 65535 and old_id not in reserved else max(reserved | {0}) + 1
                    if new_id > 65535:
                        raise ValueError(f'{video}: no available instance IDs.')
                    reserved.add(new_id)
                    id_map[key] = new_id
                result['mask_id'] = id_map[key]
                if 'instance_id' in result:
                    result['instance_id'] = id_map[key]
                result.pop('mask_coco_rle', None)
                result.pop('mask_rle', None)
                provenance = deepcopy(result.get('provenance') or {'created_by': None, 'created_at': None})
                provenance.setdefault('import_history', []).append({
                    'project_path': str(source), 'project_id': source_id, 'video_id': video,
                    'frame_index': index, 'instance_id': old_id, 'imported_at': timestamp,
                    'imported_by': ({'id': contributor['id'], 'name': contributor['name']} if contributor else None),
                    'source_scope': source_scope, 'destination_scope': target_scope,
                    'scope_review_required': source_scope != target_scope,
                })
                result['provenance'] = provenance
                return result

            chosen_shared = {}
            expected_shape = None
            # Determine one reference frame per video-wide category; never union masks across time.
            for index in indices:
                progress(f'Previewing {video}: frame {index}')
                if cancelled():
                    raise InterruptedError('Import preview cancelled.')
                image = cv2.imread(str(item['frames'][index]['path']))
                if image is None:
                    raise ValueError(f'{video}: frame {index} is unreadable.')
                shape = image.shape[:2]
                if expected_shape and shape != expected_shape:
                    raise ValueError(f'{video}: selected frames have different dimensions.')
                expected_shape = shape
                local = _load_frame(manager, source, video, index)
                effective = merge_annotations(local, source_shared, catalog['info'])
                _validate_annotations(effective, shape)
                for category in set(categories) - dest_local:
                    specific = (hive_frames or {}).get(video) if category == 'hive' else None
                    if specific is not None and index != specific:
                        continue
                    candidates = [a for a in effective if a.get('category', 'bee') == category and _has_geometry(a)]
                    if candidates and category not in chosen_shared:
                        chosen_shared[category] = (index, candidates)
            if (hive_frames or {}).get(video) is not None and 'hive' in categories and 'hive' not in dest_local:
                if 'hive' not in chosen_shared:
                    raise ValueError(f'{video}: the chosen hive reference must be a selected frame with hive annotations.')
            replaced_shared = set()
            shared_changed = False
            for category, (index, annotations) in chosen_shared.items():
                prior = [a for a in dest_shared if a.get('category', 'bee') == category]
                conflict = bool(prior) or category in legacy_shared
                action = 'Keep destination' if conflict and not replace else 'Replace' if conflict else 'Add'
                plan.rows.append({'video': video, 'scope': 'Whole video', 'frame': index,
                                  'category': category, 'instances': len(annotations), 'action': action})
                if conflict and not replace:
                    continue
                dest_shared = [a for a in dest_shared if a.get('category', 'bee') != category]
                dest_shared += [imported(a, index, 'frame' if category in source_local else 'video', 'video')
                                for a in annotations]
                replaced_shared.add(category)
                shared_changed = True
                if category in source_local:
                    plan.warnings.append(f'{video}: {category_label(category)} from frame {index} will apply to '
                                         'the whole video. Other frames are not combined; review occlusion gaps and alignment.')
            if shared_changed:
                _validate_annotations(dest_shared, expected_shape, shared=True)
                shared_relative = f'annotations/json/{video}/video_annotations.json'
                existing_json = destination / shared_relative
                if existing_json.exists():
                    target = plan.staging / shared_relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(existing_json, target)
                manager.save_video_annotations(plan.staging, video, dest_shared, contributor=None)
                plan.collect([shared_relative] + [f'annotations/png/{video}/video_annotations_{c}.png'
                                                  for c in ('chamber', 'hive', 'pollen')])

            # Also remove superseded legacy frame copies when replacing a shared category.
            cleanup = set()
            if replaced_shared & legacy_shared:
                cleanup = {_frame_index(p) for p in (destination / 'annotations/json' / video).glob('frame_*.json')}
                cleanup |= {_frame_index(p) for p in (destination / 'annotations/bbox' / video).glob('frame_*.json')}
                plan.warnings.append(f'{video}: replacing video-wide categories also removes their old frame-local copies.')
            for index in sorted(set(indices) | cleanup):
                if cancelled():
                    raise InterruptedError('Import preview cancelled.')
                existing = _load_frame(manager, destination, video, index)
                result = [a for a in existing if a.get('category', 'bee') not in replaced_shared]
                changed = len(result) != len(existing)
                if index in indices:
                    image_path = item['frames'][index]['path']
                    relative = f'frames/{video}/{image_path.name}'
                    target = destination / relative
                    if target.exists():
                        if not np.array_equal(cv2.imread(str(image_path)), cv2.imread(str(target))):
                            raise ValueError(f'{video}: frame {index} already exists with different pixels.')
                    else:
                        plan.files[relative] = image_path
                    annotations = merge_annotations(_load_frame(manager, source, video, index),
                                                    source_shared, catalog['info'])
                    for category in set(categories) & dest_local:
                        incoming = [a for a in annotations if a.get('category', 'bee') == category and _has_geometry(a)]
                        if not incoming:
                            continue
                        prior = [a for a in result if a.get('category', 'bee') == category]
                        action = 'Keep destination' if prior and not replace else 'Replace' if prior else 'Add'
                        plan.rows.append({'video': video, 'scope': 'Frame', 'frame': index,
                                          'category': category, 'instances': len(incoming), 'action': action})
                        if prior and not replace:
                            continue
                        result = [a for a in result if a.get('category', 'bee') != category]
                        result += [imported(a, index, 'frame' if category in source_local else 'video', 'frame')
                                   for a in incoming]
                        changed = True
                        if category not in source_local:
                            plan.warnings.append(f'{video} frame {index}: copied video-wide {category_label(category)} '
                                                 'into a frame-specific mask; review visible pixels before training.')
                if changed:
                    _validate_annotations(result, expected_shape)
                    manager.save_frame_annotations(plan.staging, video, index, result, contributor=None)
                    plan.collect([f'annotations/{storage}/{video}/frame_{index:06d}.{extension}'
                                  for storage, extension in (('json', 'json'), ('png', 'png'), ('bbox', 'json'))])

            relative = f'frames/{video}/video_metadata.json'
            existing_metadata = _read_json(destination / relative) if (destination / relative).exists() else {}
            metadata = dict(existing_metadata or item['metadata'])
            available = dest_indices | set(indices)
            marked = set(existing_metadata.get('selected_frames', [])) | (
                set(item['metadata'].get('selected_frames', indices)) & set(indices))
            total = max(int(metadata.get('total_frames', 0)), max(available) + 1)
            metadata.update(split=target_split, total_frames=total, selected_frames=sorted(marked & available),
                            n_selected=len(marked & available), extracted_frame_count=len(available),
                            extraction_mode='all' if len(available) == total else 'selected')
            # The GUI recomputes its bee-only allocation cache after reopening.
            metadata.pop('max_mask_id', None)
            metadata['project_import'] = {'source_project': str(source), 'source_project_id': source_id,
                                          'imported_at': timestamp}
            target = plan.staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_json(target, metadata)
            plan.files[relative] = target
            original = dest_manager.get_video_path(video)
            if include_videos and not original:
                if not item['video_path']:
                    raise ValueError(f'{video}: original video is unavailable; uncheck Copy original videos.')
                plan.files[f'input_data/{target_split}/{item["video_path"].name}'] = item['video_path']
            plan.rows.append({'video': video, 'scope': 'Images',
                              'frame': str(indices[0]) if len(indices) == 1 else f'{indices[0]}..{indices[-1]}',
                              'frame_indices': indices,
                              'category': '', 'instances': len(indices), 'action': f'Copy/retain ({target_split})'})
        plan.warnings = list(dict.fromkeys(plan.warnings))
        plan.check_unchanged()
        return plan
    except BaseException:
        plan.close()
        raise
