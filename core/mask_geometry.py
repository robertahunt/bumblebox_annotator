"""Compact, pixel-exact display measurements for instance-ID masks."""

from dataclasses import dataclass
import weakref

import cv2
import numpy as np
from scipy import ndimage

# Locating every instance in one pass keeps a table with an entry per ID up to
# the largest one, so arrays with very large IDs are measured one at a time.
MAX_LOCATED_ID = 100_000


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


def locate_instances(mask):
    """{instance ID: (row slice, column slice)} for every instance, from one pass.

    Measuring instances one at a time scans the whole array once per instance,
    which takes about a second on a frame with a hundred bees. Returns None when
    the array can't be handled this way, so the caller measures the old way.
    """
    if mask.ndim != 2 or not mask.size or not np.issubdtype(mask.dtype, np.integer):
        return None
    largest = int(mask.max())
    if largest > MAX_LOCATED_ID:
        return None
    if largest <= 0:
        return {}
    return {instance_id: box
            for instance_id, box in enumerate(ndimage.find_objects(mask, max_label=largest), 1)
            if box is not None}


def measure_located(mask, instance_id, box):
    """measure_mask(mask == instance_id), reading only the instance's bounding box."""
    y, x = box[0].start, box[1].start
    moments = cv2.moments((mask[box] == instance_id).astype(np.uint8), binaryImage=True)
    area = int(moments['m00'])
    return MaskGeometry(area, (x, y, box[1].stop - x, box[0].stop - y),
                        (x + moments['m10'] / area, y + moments['m01'] / area))


class MaskGeometryCache:
    """Cache scalars, not pixel copies; callers invalidate after in-place writes.

    A newly seen array is scanned once to find where every instance is. After an
    in-place edit only the edited instances are measured again, one at a time.
    """

    def __init__(self):
        self._categories = {}

    def invalidate(self, category=None):
        if category is None:
            self._categories.clear()
        else:
            self._categories.pop(category, None)

    def invalidate_instances(self, category, instance_ids):
        """Discard measurements only for instances affected by an in-place edit.

        IDs are rediscovered because edits may add an instance or erase the last
        pixel of an overlapped neighbor. Other scalar measurements remain valid.
        Whole-array replacements still use invalidate(category).
        """
        entry = self._categories.get(category)
        if entry is None:
            return
        for instance_id in instance_ids:
            entry['geometry'].pop(int(instance_id), None)
            if entry['boxes'] is not None:
                entry['boxes'].pop(int(instance_id), None)
        entry['ids'] = None

    def _entry(self, category, mask):
        entry = self._categories.get(category)
        if entry is None or entry['source']() is not mask:
            entry = {'source': weakref.ref(mask), 'ids': None, 'geometry': {}, 'boxes': None}
            self._categories[category] = entry
        return entry

    def _boxes(self, entry, mask):
        """Bounding slices from the one pass over a newly seen array ({} if it can't be used)."""
        if entry['boxes'] is None:
            located = locate_instances(mask)
            entry['boxes'] = located or {}
            if located is not None and entry['ids'] is None:
                entry['ids'] = tuple(located)  # ascending, as np.unique gives them
        return entry['boxes']

    def instance_ids(self, category, mask):
        if mask is None:
            self.invalidate(category)
            return ()
        entry = self._entry(category, mask)
        if entry['ids'] is None:
            self._boxes(entry, mask)
        if entry['ids'] is None:
            entry['ids'] = tuple(int(value) for value in np.unique(mask) if value > 0)
        return entry['ids']

    def geometry(self, category, mask, instance_id):
        if mask is None:
            self.invalidate(category)
            return MaskGeometry()
        entry = self._entry(category, mask)
        if instance_id not in entry['geometry']:
            box = self._boxes(entry, mask).get(instance_id)
            entry['geometry'][instance_id] = (measure_located(mask, instance_id, box) if box is not None
                                              else measure_mask(mask == instance_id))
        return entry['geometry'][instance_id]
