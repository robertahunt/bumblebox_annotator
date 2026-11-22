"""Model input size for training, including the full-resolution choice.

Ultralytics takes one input size and scales every image so that its longer
side equals it. "Full resolution" picks that size from the prepared images, so
that no image is reduced.
"""

from collections import Counter
from pathlib import Path

from PIL import Image


FULL_RESOLUTION = 'full'
FULL_RESOLUTION_LABEL = 'Full resolution'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def longest_sides(folders):
    """Count the images in these folders by their longer side, in pixels."""
    sides = Counter()
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                with Image.open(path) as image:  # reads the header only
                    sides[max(image.size)] += 1
    return sides


def full_resolution_size(sides, stride=32):
    """The smallest input size that reduces no image: the longest side, rounded up to the stride."""
    if not sides:
        raise ValueError('Full resolution needs prepared training images, and none were found.')
    return -(-max(sides) // stride) * stride


def describe_full_resolution(sides, size):
    longest = max(sides)
    text = f'Full resolution: the model input is {size} px, the longest image side in this data'
    if size != longest:
        text += f' ({longest} px) rounded up to a multiple the model accepts'
    groups = []
    for side in sorted(sides, reverse=True):
        count = sides[side]
        images = '1 image' if count == 1 else f'{count} images'
        change = 'unchanged' if side == size else f'enlarged {(size / side - 1) * 100:.1f}%'
        groups.append(f'{side} px, {images}, {change}')
    return f"{text}. Images by longest side: {'; '.join(groups)}."


def resolve_image_size(choice, folders, stride=32):
    """Return (pixels, note) for a size from the training dialog.

    A number is returned as it is, with an empty note. FULL_RESOLUTION is
    worked out from the images in the folders, and the note says what it came to.
    """
    if choice != FULL_RESOLUTION:
        return int(choice), ''
    sides = longest_sides(folders)
    size = full_resolution_size(sides, stride)
    return size, describe_full_resolution(sides, size)
