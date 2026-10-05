"""Exact-mask copying with explicit training-frame and annotation-scope guards."""

from copy import deepcopy

import numpy as np

from core.annotation_scope import frame_categories


def training_copy_targets(current_index, video_id, video_ids, splits, selected):
    """Return subsequent selected training indices in the source video only."""
    if not video_id or current_index < 0 or current_index >= len(video_ids):
        return []
    if video_ids[current_index] != video_id:
        return []
    if current_index >= len(splits) or splits[current_index] != 'train':
        return []
    return [index for index in range(current_index + 1, len(video_ids))
            if video_ids[index] == video_id
            and index < len(splits) and splits[index] == 'train'
            and index < len(selected) and selected[index]]


def annotation_key(annotation):
    return (int(annotation.get('mask_id', annotation.get('instance_id', 0))),
            annotation.get('category', 'bee'))


def selected_copy_masks(annotations, keys, project_info):
    """Snapshot category-specific masks, never the composited display mask."""
    if not keys:
        raise ValueError('Select at least one segmentation instance to copy.')
    local_categories = frame_categories(project_info)
    sources = []
    for key in dict.fromkeys(keys):
        instance_id, category = key
        if category not in local_categories:
            raise ValueError(
                f'{category.title()} labels are shared across the video in this project. '
                'They cannot be copied to training frames only. For nest labeling, '
                'use a project created with Per frame (visible nest).')
        matches = [ann for ann in annotations if annotation_key(ann) == key]
        if len(matches) != 1:
            raise ValueError(f'Cannot identify a unique {category} instance {instance_id}.')
        mask = matches[0].get('mask')
        if mask is None or np.asarray(mask).ndim != 2 or not np.any(mask):
            raise ValueError(f'{category.title()} {instance_id} has no segmentation pixels to copy.')
        source = deepcopy(matches[0])
        source['mask'] = (np.asarray(mask) > 0).astype(np.uint8) * 255
        for field in ('mask_rle', 'mask_coco_rle', 'bbox_only', 'from_mask'):
            source.pop(field, None)
        y, x = np.nonzero(source['mask'])
        source.update(mask_id=instance_id, category=category, area=int(len(x)),
                      bbox=[int(x.min()), int(y.min()),
                            int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)])
        sources.append(source)
    return sources


def merge_copied_masks(existing, sources, image_shape, *, shared=(), replace=False):
    """Prepare a frame write without mutating inputs or erasing other objects.

    Matching IDs/categories are skipped unless replacement is explicit. Other
    ID collisions and same-category pixel overlap reject the whole frame: the
    canvas stores one instance ID per pixel per category and cannot retain both.
    """
    original_keys = {annotation_key(ann) for ann in existing}
    result = list(existing)
    copied, skipped = [], []
    for source in sources:
        key = annotation_key(source)
        if source['mask'].shape != tuple(image_shape[:2]):
            raise ValueError('Frame dimensions differ; masks were not resized or copied.')
        if any(annotation_key(ann)[0] == key[0]
               and annotation_key(ann)[1] != key[1] for ann in (*result, *shared)):
            raise ValueError(f'Instance ID {key[0]} belongs to another category in this frame.')
        if key in original_keys and not replace:
            skipped.append(key)
            continue
        remaining = [ann for ann in result if annotation_key(ann) != key]
        for ann in remaining:
            mask = ann.get('mask')
            if annotation_key(ann)[1] != key[1] or mask is None:
                continue
            if np.asarray(mask).shape != source['mask'].shape:
                raise ValueError('An existing mask has incompatible frame dimensions.')
            if np.any((mask > 0) & (source['mask'] > 0)):
                raise ValueError(f'Copied {key[1]} {key[0]} overlaps another {key[1]} instance; '
                                 'existing pixels were not overwritten.')
        result = remaining + [deepcopy(source)]
        copied.append(key)
    return result, copied, skipped
