# Importing Videos and Selecting Frames

Create or open a project, then use **File > Add Video to Project** (`Ctrl+I`).
The source video is copied into `input_data/<split>/`; the original is unchanged.

- **Extract all frames** is checked by default. Every frame is saved, while
  **Frames to select** controls the evenly spaced subset marked for the chosen
  train/val/test/inference split.
- Uncheck **Extract all frames** to save only **Frames to extract**. This uses
  the same evenly spaced selection, capped at the video's frame count.
  A warning appears about GUI tracking, SAM2/YOLO propagation, and tracking
  validation on nonconsecutive frames; it disappears when all-frame extraction
  is restored.
- Sampled images retain their original source frame numbers in filenames and
  metadata. Selection is preserved when switching videos or reopening a project.
- Failed image reads/writes are reported and those frames are not marked selected.

These settings apply to new imports; they do not remove existing extracted
images or annotations. The full copied video remains available in either mode.

## Downstream Effects

YOLO image-based segmentation training does not require consecutive frames.
GUI SAM2/YOLO propagation and identity matching use extracted images, however,
so large gaps can make matches unreliable. Tracking-validation sequences expect
consecutive source frame ranges and skip missing images. Keep all frames for
those workflows. Batch video inference reads the source video directly, so
sparse annotation images do not reduce the frames available to that pipeline.

## Unfinished Annotations

The GUI's per-video COCO export skips frames without saved per-frame annotations.
The **Train YOLO Model** segmentation converter additionally excludes images
without valid segmentation polygons for the target class. Untouched frames do
not automatically become negative training examples in that path.

Partially labeled frames are different: the app has no completeness approval
gate, and a frame with one valid target mask can be included. Label all visible
objects of the target class on frames used for training or validation. Hive,
chamber, and pollen masks stored at video level are reused on included frames;
check that those masks remain accurate across the video.

Re-export COCO after annotation changes before training. These rules describe
the current GUI export and segmentation trainer, not every external COCO
consumer or the separate bbox-training path.

## Regression Checks

```bash
python scripts/test_video_import.py
```

The tests use temporary synthetic videos and masks, with no model downloads or
GPU training. They cover full/sampled import, original-file preservation, sparse
frame indexing, failed writes, sequential extraction, reopening, and exclusion
of unlabeled frames from the segmentation training dataset.
