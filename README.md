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

```bash
# Create conda environment
conda create -n bee_annotator python=3.10 -y
conda activate bee_annotator

# Install dependencies
pip install -r requirements.txt
```

## Segmentation Models

[BeeWhere segmentation models!](https://drive.google.com/drive/folders/1EDx2Gp3tX8OQCpdNEyMcNlSrJFp3Y0RC?usp=sharing) contains the `models` folder from the `2026_CV4E` project. Download the checkpoints you need and load them through the app's model controls.

## Usage

```bash
python main.py
```

Proposed reusable arena editing, chamber-position review, and image alignment tools are described in the [arena review work-in-progress outline](docs/wip/arena_review/README.md).

See [video import and frame selection](docs/video_import.md) for full-video or sampled extraction and how unfinished annotations affect training.

See [importing between projects](docs/project_import.md) to copy selected frames and annotations, including converting frame-specific hive masks into an editable video-wide mask.

See [ArUco tag size measurement](docs/aruco_tag_measurement.md) to measure minimum and maximum tag sizes before batch optimization.

See [batch hive masks and exports](docs/hive_exports.md) for the difference between video overlays, per-video summaries, and temporal hive priors.

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
five-class training, and optional history-informed batch maps.

## License

MIT
