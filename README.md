# Bumblebox Annotator

A GUI-based annotation tool for video instance segmentation with human-in-the-loop training.

Written using Claude Sonnet 4.5

## Features

- **Video Processing**: Convert videos to frames for annotation
- **SAM2 Integration**: Interactive prompting for instance segmentation
- **Mask Editing**: Brush tools for refining segmentation masks
- **Zoom & Pan**: Navigate large images easily
- **Human-in-the-Loop Training**: Iteratively train instance segmentation models
- **Model Support**: YOLO, Mask2Former, and other instance segmentation architectures

## Installation

Follow the [new-computer setup guide](docs/setup.md) for the pinned Linux/NVIDIA
environment, CPU-only alternative, a partly tested Apple Silicon Mac recipe, CUDA
verification, optional SAM2, and project transfer steps. Clone `integration/annotation-batch` to use this integration
version; the upstream default branch is still `master`.

Use a fresh environment and the supplied `constraints-linux-py310.txt`. Install
the matching PyTorch/torchvision pair first, then the constrained requirements.
Do not use the legacy `install.sh` or `install.ps1` for this version: they target
an older CUDA setup and install unpinned SAM2.

## Segmentation Models

[BeeWhere segmentation models!](https://drive.google.com/drive/folders/1EDx2Gp3tX8OQCpdNEyMcNlSrJFp3Y0RC?usp=sharing) contains the `models` folder from the `2026_CV4E` project. Download the checkpoints you need and load them through the app's model controls.

## Usage

On a configured Linux computer, install an application-menu and optional desktop
shortcut once from the checkout you want to run:

```bash
conda activate bee_annotator
python scripts/desktop_launcher.py install --desktop
```

Then open **Bumblebox Annotator** from the applications menu or desktop and select
your profile. Students do not need a terminal, Conda activation, or CUDA to draw
masks. See [desktop launcher setup and troubleshooting](docs/setup.md#desktop-launcher).

The image canvas has a gray, display-only margin on every side. Use **Pan** to
bring bees at the edge of the image toward the center of the screen. Brush and
eraser strokes cannot start in the margin, and image/mask dimensions stay
unchanged. **Fit to Window** (F) recenters and fits the image, not the margin.

The mouse wheel zooms in and out around the point under the cursor. Zooming out goes
on past the fitted size, to half of it, where the whole image is in view wherever you
were working. Point at any other part of the reduced image and scroll to zoom in there.

A **magnifier** in the top-left corner of the canvas shows a close-up of whatever is
under the cursor (image, masks, boxes, keypoints), with a mark at the exact cursor
position. It starts as a 100-pixel frame at three times the main view's zoom, so it
zooms with the view. Change either under **View > Magnifier Size and Zoom** (frames
up to 600 pixels), or turn it off with **View > Show Magnifier** (Ctrl+Shift+M).

Below it, a **zoomed-out view** shows the surroundings of the cursor, so the close-up,
the main view and the wider picture are on screen together. It outlines the part the
main view shows and marks the cursor. It starts as a 200-pixel frame reaching twice
as far across as the main view; **View > Zoomed-Out View Size and Reach** sets the
frame size and how far it reaches, from 1.05 to 2 times the main view or the whole
frame, and **View > Show Zoomed-Out View** (Ctrl+Shift+O) turns it off. Number labels
and keypoint markers are left out of it, because they keep their size on screen and
would cover a small picture.

Both size menus end with **Half the canvas height**, which also says how many pixels
that is for the window as it stands. With both frames set to it they are equal squares
that fill the left side of the canvas, and they resize with the window.

Both remember their settings on that computer and let clicks pass through. They move
to another corner while the cursor is under them, and sit side by side when the canvas
is too short to stack them.

**Play** steps through the frames in view, at up to ten frames a second, and stops
on the last one. Pressing **Play** on the last frame starts again from the first.
While playing, the Instances panel waits; it refreshes when you pause.

Assignments can use **Bees + Hive (nest)** to request both mask types together.
**Mark Finished** checks off both, using the project's saved scope for each:
for example, frame-specific bees and a video-wide hive mask. Automatic review
submission waits for both types on every assigned frame. Update all annotating
computers before using this task; older versions do not recognize it.

For terminal use, the existing entry point remains available:

```bash
python main.py
```

Proposed reusable arena editing, chamber-position review, and image alignment tools are described in the [arena review work-in-progress outline](docs/wip/arena_review/README.md).

New projects show each category's annotation scope and label meaning. **Project > Annotation Policy...** reopens these settings; old projects keep their original scopes and show unconfirmed meanings. See the [annotation-policy status and roadmap](docs/wip/annotation_policy.md) for working features and remaining experimental/planned functionality.

See [video import and frame selection](docs/video_import.md) for full-video or sampled extraction and how unfinished annotations affect training.

**Distinct neighboring bees**, beside **Opacity**, gives nearby bee masks and
boxes contrasting display colors. It is on by default; uncheck it to restore the
original instance colors. The checkbox preference is remembered on this computer.
Colors stay fixed while drawing or hiding instances; the adjacent refresh icon
recalculates contrast after large edits. New frames favor previous colors within
the same video where contrast allows. This affects the annotation canvas only,
not annotation pixels, IDs, training data, or exported video colors.

In Brush or Eraser mode, **Fill to Neighboring Masks** in the right-click menu
uses other masks of the same category to complete an outline. For example, draw
the visible part of a bee up to an overlapping bee's mask, then right-click in
the empty interior and fill. New pixels stop exactly at the neighboring mask;
hidden masks still count, but bounding boxes and other categories do not.
The outline must meet the neighboring mask without a gap. Only the clicked
region is filled, protected areas remain untouched, and the fill is one undo
step. Existing brush overlaps are not trimmed. **Fill Enclosed Area** retains
its original behavior, using only the selected instance's outline.

When editing a hive instance, **Fill to Bee Masks** also lets bee masks complete
the outline. Right-click the empty hive area enclosed by your outline and the
bee masks. Bee pixels are excluded (even for hidden bees), and neither the bee
masks nor existing hive pixels are changed. The fill supports undo/redo and
protected areas. This labels **visible hive**, not hive hidden behind bees:
when your project's policy requires the complete hive, fill those occlusion
gaps separately before marking the annotation finished.

See [importing between projects](docs/project_import.md) to copy selected frames and annotations, including converting frame-specific hive masks into an editable video-wide mask.
Import previews disclose scope/meaning conflicts. **Project > Imported Frame Review...** opens the review queue; correct each affected frame and check **Imported labels reviewed for this frame** before training/export.

Training and export use only the frames selected for a video's split. **Project > Review Bee Training Candidates...** lists training-split frames that have saved bee masks but are not selected: **Complete** adds a frame to training and **Incomplete** leaves it out. The queue is made once and keeps every decision. Each time it is opened, frames that have gained bee masks since then join it as pending, and the **Pending, added later** view shows only those.

In the training dialog, **Image Size** is the model input: every image is scaled so that its longer side equals it. For whole frames the numbered sizes go up to 3584. **Full resolution**, offered for whole-frame mask and box training, uses the longest image side found in the training and validation images, rounded up to a multiple of 32, so no image is reduced. Images with a shorter long side are enlarged to match, and the training log lists the size and each enlargement. It needs the most memory, so choosing it sets **Batch Size** to 1; raise that only after a run at 1 has worked. Measured on four crowded 4096-pixel frames with an NVIDIA RTX A5000 at batch size 1: about 19 GB of graphics memory at full resolution, against 12 GB at 3072. Preparing a frame also takes ordinary memory: 5.4 GB at 3072 for a frame with 107 bees, which scales to about 9 GB at 4096. An Apple GPU trains at full precision. The first full-resolution run on one (a Mac Studio with 96 GB, batch size 1) took 4 to 6.6 seconds a step, held 108 GB of graphics memory, which is more than the Mac has, and stopped in its fourth epoch on a fault in PyTorch's Apple GPU support. The app has since been changed for both problems (see [what the first training run on a Mac showed](docs/setup.md#what-the-first-training-run-on-a-mac-showed)); a full-resolution run with those changes has not been reported from a Mac.

See [ArUco tag size measurement](docs/aruco_tag_measurement.md) to measure minimum and maximum tag sizes before batch optimization.

Batch ArUco **Detection workers** runs the parameter bank concurrently within each
frame (default ceiling: 8). In `batch_video_inference_cli.py`, use
`--aruco-detection-workers 8`, or `1` for serial execution. This is separate from
**Optimization workers** / `--aruco-workers`, which controls the initial sweep.
The actual concurrency is limited by the bank size, available CPUs, and OpenCV's
existing thread count, and is printed in the run log. More workers can use more
RAM. All parameter sets and their merge order are preserved; changing this option
does not invalidate completed results for resume. Stop finishes the current
frame, then cleans up the workers. OpenCV's global threading and training settings
are not changed; interactive annotation detection remains serial.

See [batch hive masks and exports](docs/hive_exports.md) for the difference between video overlays, per-video summaries, and temporal hive priors.

For CLI batch inference, `--fps-override 4.3` uses a measured acquisition rate
instead of container metadata for frame timing and annotated playback. It does not
resample video or change the 1-based CSV frame numbers. Without this option the
container rate is used. Choose a fresh output folder when changing timing;
resume treats a changed override as an analysis configuration change.
`bee_velocity.csv` remains in pixels/frame: multiply by FPS and `--pixel-size-mm`
to convert to mm/second.

See [nectar source segmentation](docs/nectar_segmentation.md) to annotate nectar sources, train a nectar-only model, and run it on the current frame.

See [contributors and optional project sync](docs/project_sync.md) for session attribution,
external-folder setup, deduplicated versioned backups, previewed changed-file
updates, local rollback history, and collaboration safeguards.

For smaller contributor laptops, [Assigned Work](docs/assigned_work.md) lets a
coordinator assign videos, download only selected frame images, and publish
annotation changes without copying the whole project.

Contributor identities can be [shared between computers](docs/shared_profiles.md)
through an optional shared-folder profile directory, without exchanging files manually.

See [Project Home and workspaces](docs/workspaces.md) for the streamlined startup,
annotation/review/training views, local autosave, and shared-project update status.

## Project Structure

```
bee_annotator/
├── main.py                 # Application entry point
├── gui/
│   ├── main_window.py     # Main application window
│   ├── canvas.py          # Image canvas with zoom/pan
│   ├── toolbar.py         # Tool buttons and controls
│   └── dialogs.py         # Various dialog windows
├── core/
│   ├── video_processor.py # Video to frames conversion
│   ├── sam2_integrator.py # SAM2 model integration
│   ├── mask_editor.py     # Mask editing operations
│   └── annotation.py      # Annotation data structures
├── training/
│   ├── trainer.py         # Human-in-the-loop training
│   ├── models.py          # Model definitions
│   └── dataset.py         # Dataset preparation
└── utils/
    ├── io.py              # File I/O operations
    └── visualization.py   # Visualization utilities
```

## Experimental Brood Workflow

See [Brood Segmentation](docs/brood_segmentation.md) for visible-only brood labels,
eight-class training (including queen brood), and optional experimental history-informed batch maps.

## License

MIT
