"""Compact, pixel-exact display measurements for instance-ID masks."""

from dataclasses import dataclass
import weakref

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskGeometry:
    area: int = 0
    bbox: tuple | None = None
    centroid: tuple | None = None


def measure_mask(mask):
    if mask is None:
        return MaskGeometry()
    binary = mask if mask.dtype == np.uint8 else (mask > 0).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(binary)
    if not width or not height:
        return MaskGeometry()
    moments = cv2.moments(binary[y:y + height, x:x + width], binaryImage=True)
    area = int(moments['m00'])
    return MaskGeometry(area, (x, y, width, height),
                        (x + moments['m10'] / area, y + moments['m01'] / area))


class MaskGeometryCache:
    """Cache scalars, not pixel copies; callers invalidate after in-place writes."""

    def __init__(self):
        self._categories = {}

    def invalidate(self, category=None):
        if category is None:
            self._categories.clear()
        else:
            self._categories.pop(category, None)

    def _entry(self, category, mask):
        entry = self._categories.get(category)
        if entry is None or entry['source']() is not mask:
            entry = {'source': weakref.ref(mask), 'ids': None, 'geometry': {}}
            self._categories[category] = entry
        return entry

    def instance_ids(self, category, mask):
        if mask is None:
            self.invalidate(category)
            return ()
        entry = self._entry(category, mask)
        if entry['ids'] is None:
            entry['ids'] = tuple(int(value) for value in np.unique(mask) if value > 0)
        return entry['ids']

    def geometry(self, category, mask, instance_id):
        if mask is None:
            self.invalidate(category)
            return MaskGeometry()
        entry = self._entry(category, mask)
        if instance_id not in entry['geometry']:
            entry['geometry'][instance_id] = measure_mask(mask == instance_id)
        return entry['geometry'][instance_id]
