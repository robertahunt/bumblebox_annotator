"""Independent canvas snapshots for background annotation preparation.

Only category ID rasters and the active edit buffer are copied on the UI thread.
Expanding one full-image binary mask per instance happens in the save worker.
No Qt objects or writable canvas buffers cross the thread boundary.
"""

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AnnotationSnapshot:
    sources: tuple
    metadata: dict
    editing_id: int
    editing_mask: object

    @classmethod
    def capture(cls, canvas):
        metadata = deepcopy(canvas.annotation_metadata)
        editing_id = canvas.editing_instance_id
        editing_mask = canvas.editing_mask
        editing_category = metadata.get(editing_id, {}).get('category', 'bee')
        sources = []
        for category, raster in canvas._mask_sources():
            if raster is None:
                continue
            ids = list(canvas._stored_instance_ids(category, raster))
            if editing_id > 0 and editing_mask is not None and editing_category == category:
                if editing_id not in ids:
                    ids.append(editing_id)
            if ids:
                copied = raster.copy()
                copied.flags.writeable = False
                sources.append((category, copied, tuple(ids)))
        if editing_mask is not None:
            editing_mask = editing_mask.copy()
            editing_mask.flags.writeable = False
        return cls(tuple(sources), metadata, editing_id, editing_mask)

    def annotations(self):
        """Reproduce get_annotations, including empty edits and bbox-only labels."""
        annotations = []
        for category, raster, ids in self.sources:
            for instance_id in ids:
                binary = (self.editing_mask.copy()
                          if instance_id == self.editing_id and self.editing_mask is not None
                          else (raster == instance_id).astype(np.uint8) * 255)
                ann = deepcopy(self.metadata.get(instance_id, {}))
                ann.update(mask=binary, label=ann.get('label', f'instance_{instance_id}'),
                           area=int(np.count_nonzero(binary)), mask_id=int(instance_id), category=category)
                x, y, width, height = cv2.boundingRect(binary)
                if width and height:
                    ann['bbox'] = [x, y, width, height]
                annotations.append(ann)
        added = {ann['mask_id'] for ann in annotations}
        for instance_id, metadata in self.metadata.items():
            if (metadata.get('bbox_only', False) and instance_id not in added
                    and metadata.get('bbox') is not None and metadata['bbox'] != [0, 0, 0, 0]):
                ann = deepcopy(metadata)
                ann.update(mask_id=int(instance_id), bbox_only=True)
                annotations.append(ann)
        return annotations


@dataclass(frozen=True)
class SnapshotSave:
    snapshot: AnnotationSnapshot
    frame_categories: frozenset
    save_frame: bool
    save_video: bool
    revision: int
