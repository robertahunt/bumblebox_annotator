"""Read-only, file-level comparison for outgoing project publications."""

from collections import Counter
import hashlib
import json
from pathlib import PurePosixPath
import re

from core.project_sync import LOCATION_MARKERS, PROJECT_MARKER, _read_json, _safe_child


def publication_baseline(home, project_id, contributor_id):
    """Read only this contributor's latest manifest, without creating folders."""
    if home is None:
        return None
    contributors = _safe_child(home, 'contributors')
    if not contributors.exists():
        return None
    matches = []
    for folder in contributors.iterdir():
        if folder.is_symlink():
            raise ValueError('Sync refuses symbolic links in contributor histories.')
        if not folder.is_dir():
            continue
        marker = _safe_child(folder, LOCATION_MARKERS['contributor'])
        identity = _read_json(marker).get('id') if marker.is_file() else folder.name
        if identity == contributor_id:
            matches.append(folder)
    if len(matches) > 1:
        raise ValueError('Multiple sync folders have the same contributor ID.')
    if not matches:
        return None
    latest = _safe_child(matches[0], 'latest.json')
    if not latest.is_file():
        return None
    revision = _read_json(latest)['revision']
    if (not isinstance(revision, str) or not revision or PurePosixPath(revision).name != revision
            or revision.startswith('.')):
        raise ValueError('Invalid published revision name.')
    path = _safe_child(matches[0], 'revisions/' + revision + '/manifest.json')
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if (manifest.get('format') not in (1, 2, 3) or manifest.get('project_id') != project_id
            or manifest.get('contributor', {}).get('id') != contributor_id):
        raise ValueError('The review baseline belongs to a different project or contributor, or has an unsupported format.')
    files = manifest.get('files')
    if not isinstance(files, dict) or not {'annotations/project.json', 'annotations/' + PROJECT_MARKER} <= files.keys():
        raise ValueError('The published review baseline is missing project metadata.')
    # Validate manifest entries in memory; do not stat thousands of remote payload files.
    for name, record in files.items():
        parts = PurePosixPath(name).parts
        if (not parts or PurePosixPath(name).as_posix() != name or '..' in parts
                or any(part.startswith('.') for part in parts)
                or (parts[0] not in ('annotations', 'frames', 'input_data') and name != 'tracking_sequences.json')
                or not isinstance(record, dict) or not isinstance(record.get('size'), int) or record['size'] < 0
                or not isinstance(record.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', record['sha256'])):
            raise ValueError('Invalid file in the published review baseline: ' + str(name))
    return {'revision': revision, 'manifest_sha256': hashlib.sha256(payload).hexdigest(),
            'contributor_name': manifest['contributor']['name'], 'created_at': manifest.get('created_at', ''),
            'files': files}


def file_context(name):
    """Return a video ID (None for project-level files) and a display scope."""
    path = PurePosixPath(name)
    parts = path.parts
    if len(parts) >= 3 and parts[0] == 'frames':
        return parts[1], 'Video metadata' if path.name == 'video_metadata.json' else 'Frame image'
    if len(parts) >= 4 and parts[0] == 'annotations' and parts[1] == 'work':
        return parts[2], 'Annotation work completion'
    if len(parts) >= 4 and parts[0] == 'annotations' and parts[1] in ('json', 'png', 'bbox', 'pkl', 'frames'):
        return parts[2], 'Shared across video' if path.name.startswith('video_annotations') else 'Frame annotation'
    if len(parts) >= 3 and parts[0] == 'input_data':
        return path.stem, 'Original video'
    return None, 'Project settings' if name == 'annotations/project.json' else 'Project data'


def publication_changes(inventory, checksums, baseline, include_videos):
    previous = baseline['files'] if baseline else {}
    rows, unchanged = [], 0
    for name in sorted(inventory.keys() | previous.keys()):
        before = previous.get(name)
        after = inventory.get(name)
        if after is None:
            action = 'Excluded' if name.startswith('input_data/') and not include_videos else 'Removed'
        elif before is None:
            action = 'Added'
        elif before['size'] == after[0] and before['sha256'] == checksums[name]:
            unchanged += 1
            continue
        else:
            action = 'Changed'
        video, scope = file_context(name)
        rows.append({'path': name, 'video': video, 'scope': scope, 'action': action,
                     'before_bytes': before['size'] if before else None,
                     'after_bytes': after[0] if after else None})
    return {'baseline': {key: value for key, value in baseline.items() if key != 'files'} if baseline else None,
            'rows': rows, 'counts': dict(Counter(row['action'] for row in rows)), 'unchanged': unchanged}
