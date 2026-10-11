# Importing Frames and Annotations Between Projects

Open the **destination** project, then choose **File > Import From Project...**.
Save the source project first; the importer reads saved files, not another app's
unsaved canvas. Do not edit either project from a second process during import.

1. Choose the source project folder.
2. Select videos/frames and annotation categories. Source-selected annotated
   frames are checked initially. The annotated filter follows the chosen classes.
3. Choose a destination split. Existing videos must stay in their current split.
4. Choose **Keep destination annotations** (default), or explicitly replace the
   selected categories. Replacement affects only categories present in the
   imported annotations, not unrelated labels or blank source categories.
5. Select **Preview Import...**. Check the source frame, category, scope, counts,
   action, and source/destination label meanings for every row. Confirm the
   review requirement (plus scope conversion/replacement when applicable), then
   select **Import Copies**.
6. Open **Project > Imported Frame Review...** or **Review Queue...** below the
   image. Open a pending frame, correct its labels to match the destination
   policy, then check **Imported labels reviewed for this frame** and confirm.
   Correct labels may be approved without making an unnecessary edit.

The source is unchanged. Copies are independent, not symlinks or hard links.
Models, sync identities/settings, COCO exports, imaging-setup calibration, tracking
sequences, and source video-wide ArUco association tables are not imported.
Instance-level marker/track metadata and contributor provenance are retained.
The destination's existing video-wide tracking metadata is retained, too.

## Frame Masks to a Video-Wide Hive

When the source uses per-frame hive masks and the destination uses a shared hive
mask, each video has a **Hive reference frame** dropdown:

- **Earliest selected annotation** takes the earliest selected frame with a
  nonempty hive annotation, not necessarily frame zero.
- A specific frame can be chosen instead; it must also be selected for import.
- All hive instances from that single frame are retained as separate instances.
- Masks from different times are **not** unioned, averaged, or overwritten in
  sequence. Other categories follow their own frame/video storage rules.

The preview requires explicit confirmation that these masks will apply to the
whole video. Their original holes and disconnected regions remain, so you can
edit the shared hive mask to fill bee-occluded gaps afterward. Review camera
alignment and nest changes: one frame is a starting point, not an automatically
reconstructed full nest map. Source frame annotations remain intact.

Existing destination hive masks are kept by default. To replace them, select
**Replace selected categories**, inspect the preview, and confirm replacement.
Old frame-stored copies of a replaced video-wide category are removed throughout
that destination video so they cannot reappear as duplicate masks. Their original
files are backed up alongside the shared-mask backup.

Video-wide masks can also be copied into a frame-specific project. Those pixels
are copied exactly, not corrected for occlusion; review every selected frame
before using them as visible-surface training labels.

## Policy Conflicts and Completion

**Project > Annotation Policy...** separates where labels are stored from what
their pixels mean: visible surfaces, full object extent, or not yet confirmed.
Legacy projects retain their existing scope, with meaning unconfirmed. Confirm
the destination meanings before completing their imported-frame reviews.
Changing a meaning records intent and does not transform any mask pixels.

Each imported annotation category gets a review requirement, even if its source
and destination policies match. The preview explains scope differences, meaning
differences, unknown legacy meanings, and unfinished reviews in the source.
Copies that keep existing destination annotations do not add or reset a review
for those annotations. Image-only copies do not create label review tasks.

- Frame-specific imports require review on their selected copied frames.
- Shared-mask imports require individual review on **all extracted frames** of
  that video, including existing frames and frames extracted later. Completing
  one frame never approves the rest of the video.
- Completion is saved with contributor and time in
  `annotations/import_review.json`, included in normal project sync.
- Changes to saved frame annotation files, relevant shared annotation files,
  image pixels, or relevant policy reopen completion. Checks are conservative:
  an edit to another category in the same annotation file can also reopen review.
  Unsaved GUI edits immediately clear the displayed check; saved fingerprints
  determine persistent completion on reload.
- Uncheck the completion box to reopen a frame manually. Review and training
  selection are separate: select only frames you intend to use for training.

COCO export refuses selected pending frames before replacing existing exports.
Finish those reviews or deselect those training frames, then re-export. GUI YOLO
(segmentation, bounding-box, instance-focused) and SAM2 training also check cached
COCO review metadata, so stale exports cannot bypass a pending review for images
they contain. Models, losses, batch sizes, and resolution settings are unchanged.

This is a prospective import-review workflow, not an audit of historical manual
labels or copies made before this feature. Such old imports are not automatically
backfilled into the queue. Policy confirmation alone cannot certify those pixels;
review them separately or explicitly re-import the desired labels with replacement
and a backup. Older app versions and external training scripts do not enforce this
new review metadata.

## Frames Without Original Videos

**Copy original videos** is optional and off by default. Imported frame-only
videos are registered through their `frames/<video>/video_metadata.json`, so they
appear in the sidebar and support annotation and frame-based training export.
Frames keep their original numbers, resolution, and pixels. The source's frame
selection is preserved for the chosen subset, combined with existing destination
selections. Unselected source frames are not silently marked for training.

Video-based tools require the original video, and propagation/tracking across
nonconsecutive extracted frames has the same limitations as sampled video import.
Review completion does not change the underlying COCO eligibility rules. In particular,
the existing exporter only adds shared video masks to frames with a per-frame
annotation. A shared-hive-only import is editable, but does not by itself create
eligible training frames. Re-export COCO after reviewing imported labels.

## Safety and Provenance

- A same-named destination video must have matching reference image pixels or
  matching original video files. Existing selected frame images must match, too.
  Import refuses different image dimensions; it never rescales masks to fit.
- IDs are remapped when necessary to avoid destination collisions. Original
  instance IDs, source project/video/frame, scope conversion, and importer are
  recorded in each copied instance's `provenance.import_history`.
- Original creator and edit dates are preserved; unknown creators remain unknown.
- Re-import defaults to keeping existing category annotations, not duplicating
  objects. It is not a collaborator-mask merge or conflict-resolution system.
- Video-wide PNG storage cannot represent overlapping instances of the *same*
  category; such imports are rejected instead of discarding overlapping pixels.
  Cross-category overlap and holes/disconnected regions are preserved.
- Import detects saved project changes during preview/copying and stops rather
  than publishing a stale preview.

Before publishing, replaced files are backed up under
`import_backups/<timestamp-and-id>/originals/`, with a `manifest.json` recording
the operations and checksums. Copy/cancellation failures before publication leave
annotations unchanged; publication errors attempt rollback. After a machine crash
mid-publication, preserve that backup and inspect its manifest before retrying.
Backups are not automatically deleted or synced as annotation data. A subsequent
normal project sync publishes the imported project state as a new revision.
