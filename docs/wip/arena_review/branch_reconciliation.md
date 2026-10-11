# Annotation Branch Reconciliation

Status: combined on local branch `integration/annotation-batch`, in
`.worktrees/annotation-integration` under the batch worktree. Neither original
worktree has been switched or reset. This is a development checkpoint, not a
validated production release.

## Completed Integration

- Preserved the GUI worktree's selected local improvements in commit `3277e2f`,
  then merged the batch branch into that snapshot.
- Resolved project-opening callbacks by retaining both annotation counter reset
  and recent-project registration.
- Combined configurable ArUco parameter banks and tag filters with the GUI's
  detector defaults. Duplicate decodes of one tag no longer cause the ROI
  detector to reject it as multiple tags.
- Retained annotation visibility, navigation, dirty-saving, pollen, calibration,
  validation-review, and batch/visualization workflows.
- Incorporated temporal checkpoint/resolution work and batch refresh/merge
  utilities. Refresh model paths are explicit, and merging cannot overwrite an
  input folder. The batch CLI no longer defaults to a personal pollen checkpoint.
- Extracted shared registration and enclosed-mask operations. The local arena
  tool calls these helpers; public code does not import the private package.
- Excluded new private data, notebooks, images/videos, and checkpoints. Removed
  experiment run manifests from this branch's tracked tree, retaining local
  copies. Historical manifests and five already-published model weights remain
  in Git history; this is not a history purge.

## September 28 Progress Checkpoint

- Added project-video import controls with full-frame extraction by default,
  optional uniform sampling, and warnings about tracking/propagation on sampled
  frames. Frame selections retain their original source indices after reopening;
  failed frame writes are not counted as successful extractions.
- Added a Qt smallest/largest ArUco tag measurement dialog with frame navigation,
  zoom/pan, draggable corners, and an adjustable margin. Applying a measurement
  updates the current batch's perimeter sweeps, not per-video overrides.
- Corrected the optimizer's decoded-marker perimeter normalization to use the
  longest frame dimension, matching OpenCV and the measurement tool.
- Verified measurement-frame browsing on an actual unknown-length MJPEG clip.
  No end-to-end GPU batch validation is implied by this check.
- Re-ran all seven synthetic test groups below: 79 tests passed. The immediate
  next step is hands-on editing and representative batch-output review, followed
  by further extraction of arena-boundary and landmark editor controls.

See [video import](../../video_import.md) and
[tag measurement](../../aruco_tag_measurement.md) for the new workflows.

## October 11 Mainline Reconciliation

The policy/import-review work was checkpointed as `af40f9e` before merging
`origin/master` at `56c9a18` into `integration/annotation-batch`. The default
upstream branch is named `master`. This integration does not itself update it.

- Retained the newer brood, queen brood and nectar categories, scope-aware
  annotations, temporal workflows, project imports, contributor attribution,
  optional external sync, and per-frame import review.
- Brought in mainline's wrapping toolbars, ArUco CSV loading and identity work,
  tracking/frame-validation improvements, Project Overview, per-instance
  structure exports, and best-effort training carbon tracking.
- Routed ArUco CSV saves through the canonical save path so frame-specific hive
  and newer categories survive with their attribution and review requirements.
- Combined wrapping layouts with grouped visibility and nectar model controls.
  Project Overview now distinguishes shared-video annotations from frame-specific
  object counts, including newer categories. Counts are not review certification.
- Adopted mainline's pollen file naming: `pollen_detections.csv` is per-instance,
  while `pollen_frame_summary.csv` is per-frame. Added header checks before
  reset/append and taught hive refresh to read disconnected contour strings.
  See [CSV compatibility](../../hive_exports.md#csv-compatibility).

Focused merge checks live in `scripts/test_mainline_compatibility.py`, alongside
the existing synthetic/offscreen suites. The combined October 11 run passed
367 tests, including annotation policies, imports/review, sync, editing,
categories, exact training masks, temporal exports, ArUco measurement, CSV
compatibility, registration and validation-review helpers. Merged GUI/toolbars
were also inspected in offscreen screenshots. Tests use disposable projects and
do not alter research datasets. Physical-tablet and real GPU/batch acceptance tests
remain outstanding. Carbon estimates require the `carbontracker` dependency;
its best-effort fallback does not verify actual energy measurement here.

Next: test a disposable real project, review the combined branch, then decide
whether to merge it into `master`. No training-memory tuning is included in this
reconciliation.

## Verification and Launch

Run from the integration worktree, with the usual application dependencies:

```bash
python scripts/test_gui_editing_tools.py
python scripts/test_annotation_batch_integration.py
python -m unittest discover -s tests -p test_registration.py -v
python -m unittest discover -s tests -p test_mask_editing.py -v
python -m pytest tests/test_validation_review.py -q
python scripts/test_video_import.py
python scripts/test_aruco_measurement.py
python main.py
```

Verified 79 synthetic tests: 15 editing, 11 integration, 11 registration,
6 enclosed-mask, 3 validation-review, 15 video-import, and 18 tag-measurement tests.
Tests cover no-draw protection,
undo/redo, visibility hierarchy, view preservation across frames, calibration,
ArUco bank/filter behavior, checkpoint round trips and resume guards, category
filtering, annotation save/reopen, and offscreen main-window construction.
The local private analysis suite previously passed after helper extraction.

Physical-tablet behavior, real GPU inference/training, and an end-to-end research
batch have not been exercised here. Try a disposable project before adopting
the integration branch for ongoing annotations. The independent arena-review
workflow is still WIP, not a new menu item in the main app.

## Repositories and Worktrees

`bumblebox_annotator` and `bumblebox_annotator-gui` are two worktrees of the same
Git repository, using `augs-conference-rush` and `augusts-code-development`,
respectively. `BumbleBox` is a separate repository with its own ArUco tracking
CLI. Integrate the two annotation branches within their repository; changes to
the separate tracking repository remain separate work.

## Changes to Preserve

| Source | Functionality |
| --- | --- |
| `augusts-code-development` commits | Annotation tools, visibility and selection fixes, category/instance management, dirty saving, frame navigation, pollen training/export support |
| GUI worktree local edits | No Draw Zones, measurement and physical calibration, imaging-setup controls, bounding-box editing, and associated tests; separately review new validation/crowding modules for reusable behavior |
| `augs-conference-rush` commits | Batch inference and resumable outputs, ArUco configuration/optimization, pollen measurements, temporal hive estimates, contact/identity exports, visualization and validation work |
| Batch worktree local edits | Temporal-state checkpoints, configurable map resolution, result refresh/merge utilities, shared registration helpers |
| Local review tools | Arena reference editing, chamber-position review, landmark alignment, mask approval and compatible-result reuse |

Private analysis code, research outputs, experimental manifests, notebooks,
images, videos, and model weights are not integration inputs. Some older run
manifests are already in published history; ignore rules do not remove them.

## Overlap to Resolve

The branch tips inspected for this inventory share ancestor `fe21929`. Files
modified on both committed branches are:

- `gui/main_window.py`: preserve annotation controls alongside newer batch and
  validation entrypoints, including signal connections and settings.
- `core/marker_detector.py`: preserve annotation-facing behavior alongside
  parameter overrides, filtering, and batch optimization support.
- `gui/frame_level_validation_worker.py`: combine validation improvements and
  check category handling and visualization outputs.

Also reconcile uncommitted edits and new dependencies; Git's committed-file
overlap does not account for them. In particular, do not replace either
`gui/canvas.py` or `gui/main_window.py` wholesale with the other version.

## Remaining Review and Extraction

1. Manually check tablet
   editing, instance switching, mask/bbox toggles, undo/redo, no-draw protection,
   frame navigation, measurement, and saving/reopening a disposable project.
2. Exercise real inference and validation/visualization on a small local dataset.
3. Extract additional editor controls against that combined implementation.
   Keep local experiment code as a caller of shared modules, not a dependency
   of the application.
4. Review the final diff and file list before publishing the integration branch.

## Temporal Map Resolution

`--temporal-resolution WIDTHxHEIGHT` controls the grid used by the rolling hive
prior. Chamber masks and detections are normalized into that grid. The current
default is `256x256`; rectangular grids such as `800x1500` are accepted.

The temporal prior uses NumPy arrays on the CPU, independently of YOLO's input
image size. Larger grids retain more spatial detail but cost more RAM and CPU
time. Doubling both dimensions quadruples the number of cells. Finer sampling
does not improve inaccurate detections or fix image misalignment.

This existing temporal prior normalizes the chamber bounding-box crop. It is
distinct from the landmark-based image-registration workflow being extracted.
