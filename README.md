# Bee Annotator

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

## Usage

```bash
python main.py
```

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

## License

MIT
