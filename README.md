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
environment, CPU-only alternative, CUDA verification, optional SAM2, and project
transfer steps. Clone `integration/annotation-batch` to use this integration
version; the upstream default branch is still `master`.

Use a fresh environment and the supplied `constraints-linux-py310.txt`. Install
the matching PyTorch/torchvision pair first, then the constrained requirements.
Do not use the legacy `install.sh` or `install.ps1` for this version: they target
an older CUDA setup and install unpinned SAM2.

## Segmentation Models

[BeeWhere segmentation models!](https://drive.google.com/drive/folders/1EDx2Gp3tX8OQCpdNEyMcNlSrJFp3Y0RC?usp=sharing) contains the `models` folder from the `2026_CV4E` project. Download the checkpoints you need and load them through the app's model controls.

## Usage

```bash
python main.py
```

Proposed reusable arena editing, chamber-position review, and image alignment tools are described in the [arena review work-in-progress outline](docs/wip/arena_review/README.md).

New projects show each category's annotation scope and label meaning. **Project > Annotation Policy...** reopens these settings; old projects keep their original scopes and show unconfirmed meanings. See the [annotation-policy status and roadmap](docs/wip/annotation_policy.md) for working features and remaining experimental/planned functionality.

See [video import and frame selection](docs/video_import.md) for full-video or sampled extraction and how unfinished annotations affect training.

See [importing between projects](docs/project_import.md) to copy selected frames and annotations, including converting frame-specific hive masks into an editable video-wide mask.
Import previews disclose scope/meaning conflicts. **Project > Imported Frame Review...** opens the review queue; correct each affected frame and check **Imported labels reviewed for this frame** before training/export.

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
external-folder setup, versioned backups, and collaboration safeguards.

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
