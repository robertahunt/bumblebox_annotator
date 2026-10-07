"""Shared pixel operations for instance and arena mask editors."""

import cv2
import numpy as np


def enclosed_mask_region(mask, point, *, filled=False, protected_mask=None):
    """Return editable pixels in the clicked enclosed component, or None.

    Fill travels through edge-sharing pixels only, so a one-pixel diagonal
    brush or eraser outline remains a barrier.
    Enclosure is checked before excluding protected pixels, so a protection zone
    cannot turn an open outline into a fillable region. The input is not changed.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional mask")
    if protected_mask is not None and protected_mask.shape != mask.shape:
        raise ValueError("Protection and segmentation masks must have the same shape")
    x, y = map(int, point)
    height, width = mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return None
    if bool(mask[y, x] > 0) != filled:
        return None
    if protected_mask is not None and protected_mask[y, x] > 0:
        return None
    source = np.where(mask > 0, 255, 0).astype(np.uint8)
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(source, flood_mask, (x, y), 128, flags=4)
    region = source == 128
    if (region[0].any() or region[-1].any()
            or region[:, 0].any() or region[:, -1].any()):
        return None
    if protected_mask is not None:
        region &= protected_mask == 0
    return region if region.any() else None
