# Nectar Source Segmentation

`Nectar source` (stored as `nectar`) is a separate instance category with a teal
mask. It is appended after the existing brood and queen-brood categories, so
existing category IDs and annotations are not renumbered. Older projects gain
the new category when opened.

## Annotate and Train

1. Choose **New Instance > Nectar source**, in the toolbar or right-click menu.
   Brush is selected automatically. Existing instances can also be reclassified
   through **Change Instance Category > Nectar source**.
2. Label the visible source pixels consistently. Masks may overlap hive or other
   categories without erasing them. Nectar annotations are frame-specific,
   including in legacy projects with video-wide hive masks. They are not
   automatically propagated into other frames.
3. Review and label all visible nectar sources in the selected training and
   validation frames. Exclude unfinished frames; choose independent videos for
   validation where possible. The current converter includes frames with target
   masks, not arbitrary unlabeled frames as negative examples.
4. Open **Train YOLO Model**, choose **Nectar source**, and keep **Export COCO**
   checked. The default run name is `nectar_segmentation`. The training labels
   contain one class, `0: nectar`, selected by COCO category name rather than a
   fixed ID. Holes and disconnected mask regions are preserved. Both splits
   need nectar segmentation annotations.

The **Nectar** checkbox controls category visibility, alongside the usual global
and individual-instance visibility controls. Nectar instances also appear in
the sidebar and automatic annotation counters.

## Test a Checkpoint

Open the **Hive, Chamber, Pollen & Nectar Toolbar** through the toolbar menu.
Use **Load Nectar Model...** followed by **Run Nectar** to add predictions to the
current frame, or accept the loading prompt after training. The loader rejects
non-segmentation checkpoints and checkpoints with classes other than `nectar`.
Review predictions before saving or reusing them as training labels.

This addition covers annotation, saving/reloading, COCO export, segmentation
training, and current-frame inference. It does not add a nectar model to Batch
Video Inference, nectar-contact measurements, or a temporal nectar map. Nectar
is not added to the eight-class brood model or the bbox-only training workflow.
