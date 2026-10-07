"""Explicit, backed-up attribution of legacy annotations; dry-run by default.

Run with ``python -m scripts.attribute_project_annotations --help``. Close the
annotator before applying so cached metadata cannot overwrite the correction.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from uuid import UUID, uuid4

from core.contributors import utc_now
from core.project_sync import _safe_child, _write_json


def attribute_project(project, contributor, apply=False, app_closed=False):
    project = Path(project).resolve()
    if not (project / 'annotations/project.json').is_file():
        raise ValueError('The source is not an annotation project.')
    if apply and not app_closed:
        raise ValueError('Save and close the annotator before applying attribution.')
    actor = {'id': str(UUID(contributor['id'])), 'name': contributor['name'].strip()}
    if not actor['name']:
        raise ValueError('A contributor name is required.')
    timestamp = utc_now()
    changes = []
    counts = Counter()
    for storage in ('json', 'bbox'):
        base = _safe_child(project, Path('annotations') / storage)
        for path in sorted(base.glob('**/*')):
            if path.is_symlink():
                raise ValueError(f'Attribution refuses symbolic links: {path}')
            if path.suffix == '.pkl':
                raise ValueError('Convert legacy pickle annotations to JSON before attribution.')
            if not path.is_file() or path.suffix != '.json':
                continue
            if path.name != 'video_annotations.json' and not path.stem.startswith('frame_'):
                continue
            original = path.read_bytes()
            data = json.loads(original)
            annotations = data.get('annotations') if isinstance(data, dict) else data
            if not isinstance(annotations, list):
                raise ValueError(f'Unrecognized annotation structure: {path}')
            changed = 0
            for annotation in annotations:
                provenance = deepcopy(annotation.get('provenance') or {})
                creator = provenance.get('created_by')
                if creator and creator.get('id') != actor['id']:
                    raise ValueError(f'{path} already credits another contributor. '
                                     'No files were changed; review that attribution first.')
                if creator == actor:
                    continue
                provenance['created_by'] = actor
                provenance.setdefault('created_at', None)
                provenance.setdefault('attribution_history', []).append({
                    'action': 'assign_creator', 'assigned_by': actor,
                    'assigned_at': timestamp, 'previous_created_by': creator,
                    'reason': 'Retrospective attribution explicitly requested by contributor',
                })
                annotation['provenance'] = provenance
                counts[f'{storage}:{annotation.get("category", "bee")}'] += 1
                changed += 1
            if changed:
                changes.append((path, original, data))
    result = {'files': len(changes), 'records': sum(counts.values()),
              'counts': dict(counts), 'contributor': actor, 'applied': False, 'backup': None}
    if not apply or not changes:
        return result

    backup = _safe_child(project, Path('attribution_backups') /
                         (timestamp.replace(':', '-') + '-' + uuid4().hex[:8]), mkdir=True)
    # Finish and verify every backup before writing any annotation file.
    manifest = {'assigned_at': timestamp, 'contributor': actor, 'files': {}}
    for path, original, _ in changes:
        relative = path.relative_to(project)
        destination = _safe_child(backup, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if destination.read_bytes() != original:
            raise ValueError('Annotations changed during backup. Close the app and retry.')
        manifest['files'][relative.as_posix()] = hashlib.sha256(original).hexdigest()
    _write_json(backup / 'manifest.json', manifest)
    for path, original, _ in changes:
        if path.read_bytes() != original:
            raise ValueError('Annotations changed during backup. Close the app and retry.')
    written = []
    try:
        for path, original, data in changes:
            if path.read_bytes() != original:
                raise ValueError('Annotations changed during attribution. Close the app and retry.')
            _write_json(path, data)
            written.append((path, data))
            if json.loads(path.read_bytes()) != data:
                raise OSError(f'Attribution verification failed: {path}')
    except Exception:
        # Restore our writes only; never overwrite a subsequent external edit.
        for path, data in reversed(written):
            if json.loads(path.read_bytes()) == data:
                shutil.copy2(backup / path.relative_to(project), path)
        raise
    result.update(applied=True, backup=str(backup))
    _write_json(backup / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--contributor-name', required=True)
    parser.add_argument('--contributor-id', required=True,
                        help='Existing UUID from this installation\'s contributor profile')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--confirm-app-closed', action='store_true')
    args = parser.parse_args()
    result = attribute_project(args.project,
                               {'name': args.contributor_name, 'id': args.contributor_id},
                               apply=args.apply, app_closed=args.confirm_app_closed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
