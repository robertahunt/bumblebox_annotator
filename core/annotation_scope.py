"""Project-level annotation scope, with legacy video-wide hive compatibility."""

import json
from pathlib import Path
from core.categories import BROOD_CATEGORIES


def frame_categories(project_info):
    info = project_info.get('project_info', project_info)
    scope = info.get('hive_annotation_scope', 'video')
    if scope not in ('frame', 'video'):
        raise ValueError(f"Unknown hive_annotation_scope: {scope!r}")
    return ({'bee', 'hive'} if scope == 'frame' else {'bee'}) | {'nectar'} | set(BROOD_CATEGORIES)


def read_project_info(project_path):
    path = Path(project_path) / 'annotations' / 'project.json'
    if not path.exists():
        return {}
    with path.open() as stream:
        return json.load(stream)


def split_annotations(annotations, project_info):
    per_frame = frame_categories(project_info)
    return (
        [ann for ann in annotations if ann.get('category', 'bee') in per_frame],
        [ann for ann in annotations if ann.get('category', 'bee') not in per_frame],
    )


def merge_annotations(frame_annotations, video_annotations, project_info):
    per_frame = frame_categories(project_info)
    # Legacy projects can contain video-level classes in old frame files. Prefer
    # the shared source for those classes, but never replace frame-specific hives.
    shared_categories = {ann.get('category', 'bee') for ann in video_annotations
                         if ann.get('category', 'bee') not in per_frame}
    return ([ann for ann in frame_annotations
             if ann.get('category', 'bee') not in shared_categories]
            + [ann for ann in video_annotations
               if ann.get('category', 'bee') not in per_frame])
