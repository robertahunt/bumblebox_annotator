"""Standard COCO masks, distinct from the annotator's legacy row-major RLE."""

import numpy as np
from pycocotools import mask as mask_utils


def encode_mask(mask):
    binary = np.asarray(mask) > 0
    if binary.ndim != 2:
        raise ValueError('An instance mask must be two-dimensional')
    rle = mask_utils.encode(np.asfortranarray(binary, dtype=np.uint8))
    return {'size': list(rle['size']), 'counts': rle['counts'].decode('ascii')}


def segmentation_rle(segmentation, height, width):
    """Polygons or compressed/uncompressed COCO RLE as one compressed RLE, ready for pycocotools. None if empty."""
    if isinstance(segmentation, dict):
        if list(segmentation.get('size', [])) != [height, width]:
            raise ValueError('COCO mask dimensions do not match the source image')
        rle = dict(segmentation)
        if isinstance(rle['counts'], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        elif isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
    elif isinstance(segmentation, list):
        polygons = [p for p in segmentation if len(p) >= 6 and len(p) % 2 == 0]
        if not polygons:
            return None
        rle = mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))
    else:
        raise ValueError('Expected COCO polygons or RLE segmentation')
    return rle


def decode_segmentation(segmentation, height, width):
    """Decode polygons or compressed/uncompressed COCO RLE to one binary mask."""
    rle = segmentation_rle(segmentation, height, width)
    if rle is None:
        return np.zeros((height, width), dtype=np.uint8)
    return (mask_utils.decode(rle) > 0).astype(np.uint8)
