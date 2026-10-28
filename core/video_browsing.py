"""Lightweight video browsing metadata, without loading annotations or images."""

import os
from pathlib import Path


def experiment_assignments(config):
    setups = config.get('setups', {})
    assignments = config.get('video_assignments', {})
    if not isinstance(setups, dict) or not isinstance(assignments, dict):
        raise ValueError('Invalid imaging setup metadata')
    valid = {str(name).strip() for name, profile in setups.items()
             if str(name).strip() and isinstance(profile, dict)}
    return {str(video): str(name) for video, name in assignments.items() if str(name) in valid}


def local_modification_times(project, videos, cancelled=lambda: False):
    """Local file dates include imports/copies; they are not an edit audit trail.

    Folder dates retain evidence of deletion, including erasing the last instance.
    Original videos, extracted images, exports and calibration metadata are excluded.
    """
    project = Path(project)
    modified, errors = {}, {}
    for video in videos:
        if Path(video).name != video or video in ('.', '..'):
            raise ValueError('Invalid video identifier')
        latest = None
        try:
            for kind in ('json', 'png', 'bbox', 'work', 'pkl', 'frames'):
                if cancelled():
                    raise InterruptedError('Modification-time check cancelled')
                folder = project / 'annotations' / kind / video
                if folder.is_symlink():
                    raise ValueError('Annotation folders must not be symbolic links')
                try:
                    with os.scandir(folder) as entries:
                        latest = max(latest or 0, folder.stat().st_mtime_ns)
                        for entry in entries:
                            if cancelled():
                                raise InterruptedError('Modification-time check cancelled')
                            if not entry.name.startswith('.') and entry.is_file(follow_symlinks=False):
                                latest = max(latest or 0, entry.stat(follow_symlinks=False).st_mtime_ns)
                except FileNotFoundError:
                    continue
        except InterruptedError:
            raise
        except (OSError, ValueError) as exc:
            latest = None
            errors[video] = str(exc)
        modified[video] = latest
    return {'modified': modified, 'errors': errors}
