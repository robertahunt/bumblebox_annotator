"""Persistent per-frame import review with fail-closed export checks."""

import hashlib
import json
from pathlib import Path

from core.annotation_scope import read_project_info
from core.annotation_policy import category_policies, policy_text
from core.categories import CATEGORIES
from core.contributors import utc_now
from core.project_sync import _read_json, _write_json, _digest


REVIEW_FILE = 'annotations/import_review.json'


def read_reviews(project):
    path = Path(project) / REVIEW_FILE
    result = _read_json(path) if path.exists() else {'version': 1, 'frames': {}, 'shared': {}}
    if result.get('version') != 1 or not isinstance(result.get('frames'), dict) or not isinstance(result.get('shared'), dict):
        raise ValueError('Unsupported import review records; review metadata must not be discarded.')
    def check_requirements(requirements):
        if not isinstance(requirements, dict) or not set(requirements) <= set(CATEGORIES):
            raise ValueError('Invalid import review categories; review metadata must not be discarded.')
        for requirement in requirements.values():
            if not isinstance(requirement, dict) or not isinstance(requirement.get('reasons'), list):
                raise ValueError('Invalid import review requirement.')
            for key in ('source_policy', 'destination_policy'):
                try:
                    policy_text(requirement[key])
                except (KeyError, TypeError) as exc:
                    raise ValueError('Invalid import review policy.') from exc
    for requirements in result['shared'].values():
        check_requirements(requirements)
    for frames in result['frames'].values():
        if not isinstance(frames, dict):
            raise ValueError('Invalid import review frames.')
        for index, entry in frames.items():
            if not str(index).isdigit() or not isinstance(entry, dict):
                raise ValueError('Invalid import review frame.')
            check_requirements(entry.get('requirements', {}))
            if entry.get('completion') is not None and not isinstance(entry['completion'], dict):
                raise ValueError('Invalid import review completion.')
    return result


def frame_task(records, video, index):
    frame = records['frames'].get(video, {}).get(str(index), {})
    shared = records['shared'].get(video, {})
    requirements = dict(shared)
    requirements.update(frame.get('requirements', {}))
    return {'requirements': requirements, 'completion': frame.get('completion')} if requirements else None


def add_review(records, video, index, category, requirement, *, shared=False):
    if shared:
        records['shared'].setdefault(video, {})[category] = requirement
        # A newer shared import supersedes older requirements for this category.
        for entry in records['frames'].get(video, {}).values():
            entry.get('requirements', {}).pop(category, None)
    else:
        entry = records['frames'].setdefault(video, {}).setdefault(str(index), {'requirements': {}})
        entry['requirements'][category] = requirement
        entry.pop('completion', None)


def review_signature(project, video, index, task, info=None):
    """Hash saved sources, not mutable canvas data or the completion record itself."""
    root = Path(project)
    policies = category_policies(info if info is not None else read_project_info(root))
    categories = task['requirements']
    policy = {c: policies[c] for c in categories if c in policies}
    payload = {'requirements': categories, 'policy': policy, 'files': {}}
    paths = [root / 'frames' / video / f'frame_{index:06d}.jpg']
    for storage, suffix in (('json', 'json'), ('bbox', 'json'), ('png', 'png')):
        paths.append(root / 'annotations' / storage / video / f'frame_{index:06d}.{suffix}')
    if any(p['scope'] == 'video' for p in policy.values()):
        paths.append(root / 'annotations/json' / video / 'video_annotations.json')
        for category, entry in policy.items():
            if entry['scope'] == 'video':
                paths.append(root / 'annotations/png' / video / f'video_annotations_{category}.png')
        paths.append(root / 'annotations/png' / video / 'video_annotations.png')
    for path in paths:
        payload['files'][str(path.relative_to(root))] = _digest(path) if path.is_file() else None
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def review_status(project, video, index, records=None, info=None):
    records = records if records is not None else read_reviews(project)
    task = frame_task(records, video, index)
    if task is None:
        return 'not_required'
    completion = task['completion']
    if completion and completion.get('signature') == review_signature(project, video, index, task, info):
        return 'complete'
    return 'pending'


def complete_review(project, video, index, contributor=None, *, complete=True):
    root = Path(project)
    if not (root / 'frames' / video / f'frame_{index:06d}.jpg').is_file():
        raise ValueError('Open an available imported frame before marking it reviewed.')
    records = read_reviews(root)
    task = frame_task(records, video, index)
    if not task:
        raise ValueError('This frame has no import review requirements.')
    policies = category_policies(read_project_info(root))
    if complete and any(policies[c]['meaning'] == 'unconfirmed' for c in task['requirements'] if c in policies):
        raise ValueError('Confirm the destination category meanings in Project > Annotation Policy first.')
    entry = records['frames'].setdefault(video, {}).setdefault(str(index), {'requirements': {}})
    if complete:
        entry['completion'] = {
            'signature': review_signature(root, video, index, task), 'completed_at': utc_now(),
            'completed_by': ({'id': contributor['id'], 'name': contributor['name']} if contributor else None),
        }
    else:
        entry.pop('completion', None)
    _write_json(root / REVIEW_FILE, records)


def review_rows(project):
    records = read_reviews(project)
    info = read_project_info(project)
    rows = []
    for video in sorted(set(records['frames']) | set(records['shared'])):
        indices = set(map(int, records['frames'].get(video, {})))
        if video in records['shared']:
            indices.update(int(p.stem.split('_')[1]) for p in (Path(project) / 'frames' / video).glob('frame_*.jpg'))
        for index in sorted(indices):
            task = frame_task(records, video, index)
            if task:
                rows.append({'video': video, 'frame': index, 'task': task,
                             'status': review_status(project, video, index, records, info)})
    return rows


def require_reviewed(project, selections):
    """Fail before replacing any exports. Training selection and review are distinct."""
    from core.assigned_work import require_full_project
    require_full_project(project)
    records = read_reviews(project)
    info = read_project_info(project)
    pending = [f'{video}/frame_{index:06d}' for video, indices in selections.items() for index in indices
               if review_status(project, video, index, records, info) == 'pending']
    if pending:
        raise ValueError('Imported annotations need review before export/training. '
                         'Open Project > Imported Frame Review, correct the labels and mark each frame complete, '
                         'or deselect those training frames.\n' + '\n'.join(pending[:20]))


def selected_frames(project, videos):
    result = {}
    for video in videos:
        path = Path(project) / 'frames' / video / 'video_metadata.json'
        result[video] = _read_json(path).get('selected_frames', []) if path.exists() else []
    return result


def export_review_metadata(project, video, index):
    task = frame_task(read_reviews(project), video, index)
    return review_signature(project, video, index, task) if task else None


def validate_coco_reviews(project, paths):
    """Cached COCO must not bypass new imports or changed review/policy decisions."""
    from core.assigned_work import require_full_project
    require_full_project(project)
    policies = category_policies(read_project_info(project))
    records = read_reviews(project)
    for path in paths:
        coco = _read_json(path)
        images = coco.get('images', [])
        video_names = {v['id']: v['name'] for v in coco.get('videos', [])}
        references = []
        for im in images:
            file = Path(im.get('file_name', ''))
            video = (video_names.get(im.get('video_id')) or coco.get('info', {}).get('video_name')
                     or (file.parent.name if file.parent.name else Path(path).stem))
            index = im.get('frame_index')
            if index is None and file.stem.startswith('frame_') and file.stem[6:].isdigit():
                index = int(file.stem[6:])
            if index is None:
                if records['frames'] or records['shared'] or im.get('import_review_signature'):
                    raise ValueError('Cannot verify imported frame in cached COCO. Re-export COCO before training.')
                continue
            references.append((im, video, int(index)))
        selections = {}
        for im, video, index in references:
            selections.setdefault(video, set()).add(index)
        require_reviewed(project, selections)
        exported_policy = coco.get('info', {}).get('annotation_policy')
        if exported_policy is not None and exported_policy != policies:
            raise ValueError('Annotation policy changed. Re-export COCO before training.')
        for im, video, index in references:
            current = export_review_metadata(project, video, index)
            if current != im.get('import_review_signature'):
                raise ValueError('Imported annotations or review changed. Re-export COCO before training.')
