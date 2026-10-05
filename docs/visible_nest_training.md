# Visible Nest Training

## Annotation Policy

Create a separate project for visible-nest labels. In **New Project**, leave
**Nest / hive labels** set to **Per frame (visible nest)**. Use the existing
`hive` category and label the nest material visible in that particular frame.
Do not paint through bees or copy the reconstructed temporal nest map into the
training labels. Disconnected visible regions can belong to one instance.

These labels train the single-frame model's current visual evidence. Temporal
inference combines that evidence with history later; it is not substituted for
the manually reviewed training target. Keep train and validation videos
separate, particularly when frames are visually very similar.

New projects persist `hive_annotation_scope: frame` in
`annotations/project.json`. Only bee and hive annotations are frame-local;
chamber and pollen retain their existing video-wide scope. The explicit
**Shared across video (legacy)** choice preserves the previous behavior.
Existing projects without this field remain video-wide. Opening a project does
not migrate, delete, or rewrite its masks. Do not hand-edit this setting as a
substitute for migrating an existing dataset.

YOLO bee-tracking saves and **Delete All Bees** preserve frame-local nest labels.
Tracking stops that frame's write if a tracked bee ID would collide with an
existing nest ID. Frame-level validation reads each frame's exported nest label
in frame-specific projects, rather than substituting the shared video mask.

Unannotated frames are still skipped during segmentation dataset preparation.
This does not make partially annotated frames safe: label all visible target
regions in each image included for training. Reviewed negative-image support
is not added by this change.

## Copying A Nest Mask To Training Frames

In a per-frame nest project, select the Hive instance in the right sidebar,
then right-click it and choose **Copy to Next Training Frame...**. The copy is
saved and the destination opens with Brush selected. For several frames, use
**Copy Through Training Frames...** and choose the last training frame. Multiple
sidebar selections can also be copied together.

Only subsequent frames explicitly selected for training in the **same video**
are eligible, regardless of the current frame-list filter. Unselected frames,
validation/test frames, and other videos are excluded. The source video must
also be in the training split. Batch copying leaves the source view in place.

The exact source mask is copied, including active brush edits, holes, and
disconnected regions. This is not motion tracking or alignment. Review each
destination: erase pixels now covered by bees and label newly exposed nest.
Do not train on unreviewed copies just because the underlying nest is static.

Existing matching IDs/categories are skipped unless **Replace existing
instances with the same ID and category** is checked. Other annotations are
preserved. A conflicting category ID, incompatible image size, or overlap with
another instance of the same category causes that frame to be skipped with an
explanation. Completed copies are saved immediately, including when a longer
copy operation is canceled. Shared video-level classes cannot use this action;
legacy shared-hive projects are not automatically converted.

## Exact Masks

The old YOLO conversion kept the polygon with the most vertices, which is not
necessarily the largest region by area. It discarded other disconnected parts.
The preceding external-contour export also filled holes in masks.

COCO exports now encode masks as standard column-major COCO RLE with
`iscrowd: 0`. One annotated instance remains one target, including all its
disconnected regions and holes. Older polygon COCO files remain readable, but
lost pixels cannot be recovered from them: **re-export COCO from the saved
source annotations before training**.

Frame annotation saves retain the existing ID PNG and add an exact
`mask_coco_rle` to each JSON record. The current loader prefers this field, so
overlap between categories no longer overwrites source pixels. Old PNG/JSON
files without this field still load. Tools that read only the combined PNG
cannot recover overlapping instances; use `AnnotationManager` or the COCO RLE.

## YOLO Training

The normal segmentation and instance-focused GUI training buttons use
`training.raster_masks.RasterSegmentationTrainer`. Dataset preparation writes
per-image `labels/{split}/*.json` sidecars and a YAML
`mask_format: bumblebox-coco-rle-v1` marker. This is a BumbleBox training format,
**not ordinary YOLO polygon TXT**. Do not feed it to the unmodified `yolo train`
CLI. Bounding-box-only training is unchanged.

The adapter uses standard Ultralytics models, losses, optimizer, checkpoints,
and validation metrics. Saved weights work with normal YOLO inference.
Train and validation both receive raster masks. Each mask is resized with
nearest-neighbor interpolation and boxes are recomputed after paired geometry.
As with ordinary YOLO training, image resizing and `mask_ratio` can remove very
small details; this is not full-resolution supervision at every image size.

Paired affine transforms, horizontal/vertical flips, HSV and image-only blur
augmentation remain enabled. Polygon-only mosaic, mixup, cutmix, copy-paste, perspective and
multi-scale mixing are disabled explicitly. Separate masks use
`overlap_mask=False`; this preserves overlapping instances but can use more
host/GPU memory than a combined ID map. Start conservatively with batch size.
The GUI still requires CUDA for training; there is no new CPU fallback.

Standalone validation of these dataset sidecars must also use the adapter:

```python
from ultralytics import YOLO
from training.raster_masks import RasterSegmentationValidator

YOLO("/path/to/best.pt").val(
    validator=RasterSegmentationValidator,
    data="/path/to/yolo_format/dataset.yaml",
    overlap_mask=False,
)
```

The adapter is tested with Ultralytics 8.4.16 and Albumentations 1.4.24.
Regression tests include exact storage/COCO round trips, disconnected regions,
holes, legacy projects, paired transforms, a real loss/backward pass, and a
one-epoch CPU smoke run with checkpoint reload and standalone validation.

```bash
QT_QPA_PLATFORM=offscreen NO_ALBUMENTATIONS_UPDATE=1 \
python -m unittest scripts.test_visible_nest_training
```
