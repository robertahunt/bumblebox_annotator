# Training Guide

## Overview

The BeeWhere Annotator now includes a complete training pipeline for fine-tuning Detectron2 models on your annotated bee videos with instance tracking support.

## Features

- **Non-blocking training**: Train models in the background while continuing to annotate
- **Live progress monitoring**: Real-time loss curves, validation metrics, and training logs
- **Validation visualization**: View ground truth vs. predictions side-by-side
- **Pause/resume/stop**: Full control over the training process
- **Automatic best model tracking**: Best checkpoint is automatically saved based on validation AP
- **Instance tracking integration**: Models are trained with video tracking metadata for temporal consistency

## Prerequisites

Install Detectron2 and dependencies:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2 (adjust for your CUDA version)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Additional requirements
pip install opencv-python pycocotools matplotlib
```

## Training Workflow

### 1. Prepare Your Data

Annotate videos using the main annotation interface:
- Use the video sidebar to switch between videos
- Add instance segmentation masks for bees
- Assign unique Instance IDs to track individual bees across frames
- Mark videos as 'train' or 'val' in the video list

### 2. Export and Configure

**Menu: Training → Start Training**

This will:
1. Export annotations to COCO format with tracking metadata
2. Show dataset statistics (train/val split, number of instances)
3. Open the training configuration dialog

### 3. Configure Training Parameters

**Model Architecture:**
- R50-FPN: ResNet-50 backbone (faster, less accurate)
- R101-FPN: ResNet-101 backbone (balanced)
- X101-FPN: ResNeXt-101 backbone (slower, more accurate)

**Hyperparameters:**
- **Batch Size**: Number of images per iteration (reduce if GPU memory limited)
- **Learning Rate**: Typically 0.00025 for fine-tuning (auto-adjusted for batch size)
- **Max Iterations**: Total training steps (auto-calculated: ~30-50 epochs for small datasets)
- **Validation Period**: Run validation every N iterations (default: 500)

**Data Augmentation:**
- Random crops (80% of images)
- Horizontal flips
- Multi-scale training (resize range)

### 4. Monitor Training

The **Training Progress Dialog** shows:

- **Progress Bar**: Current iteration / max iterations
- **Live Loss Curve**: Total loss plotted in real-time with matplotlib
- **Current Metrics**: Loss value and learning rate
- **Validation Results Table**: Segmentation AP, AP50, AP75, bbox AP for each validation run
- **Best Model Tracker**: Highlighted when a new best checkpoint is saved
- **Training Log**: Detailed status messages with auto-scroll

**Controls:**
- **Pause/Resume**: Pause training to check results or free GPU memory
- **Stop**: Stop training early (saves current checkpoint)
- **View Validation Predictions**: See ground truth vs. predictions on validation samples

### 5. View Validation Predictions

Click **"View Validation Predictions"** after the first validation step to:
- Compare ground truth annotations with model predictions
- See confidence scores for each predicted instance
- Navigate through multiple validation samples
- Assess model quality visually

### 6. Training Completion

When training finishes:
- Best model checkpoint is saved in `models/training/model_best.pth`
- Final validation metrics are displayed
- You can load the trained model for inference (coming soon)

## Tips

**For Small Datasets (<100 images):**
- Use R50-FPN for faster iteration
- Increase max iterations (50+ epochs)
- Enable all data augmentation
- Lower learning rate (0.0001)

**For Large Datasets (>1000 images):**
- Use X101-FPN for best accuracy
- Standard settings (30 epochs)
- Can disable some augmentation
- Standard learning rate (0.00025)

**GPU Memory Issues:**
- Reduce batch size (try 1 or 2)
- Use R50-FPN instead of X101-FPN
- Reduce image size in config (advanced)

**Validation Frequency:**
- Small datasets: Validate every 100-200 iterations
- Large datasets: Validate every 500-1000 iterations

## Output Files

Training outputs are saved to `<project>/models/training/`:

```
models/training/
├── config.yaml              # Full training configuration
├── model_best.pth          # Best checkpoint (highest validation AP)
├── model_final.pth         # Final checkpoint at end of training
├── model_0001000.pth       # Periodic checkpoints
├── model_0002000.pth
├── ...
├── metrics.json            # Training metrics log
└── events.out.tfevents.*   # TensorBoard logs
```

## Next Steps

After training:
1. **Run Inference** (coming soon): Apply trained model to new videos
2. **Fine-tune Further**: Resume training from best checkpoint with lower learning rate
3. **Export Model**: Save model for use in other applications

## Troubleshooting

**Training won't start:**
- Check that detectron2 is installed: `python -c "import detectron2; print(detectron2.__version__)"`
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure you have at least 8GB of GPU memory

**Training is very slow:**
- Reduce batch size to 1 or 2
- Use R50-FPN instead of X101-FPN
- Close other GPU applications

**Validation AP is low:**
- Check validation predictions viewer for errors
- Verify annotations are correct
- Increase training iterations
- Enable more data augmentation

**GPU Out of Memory:**
- Reduce batch size to 1
- Switch to smaller model (R50-FPN)
- Close other applications using GPU

## Technical Details

### COCO Format with Tracking

The exported COCO JSON includes additional fields for instance tracking:

```json
{
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation": [...],
      "bbox": [x, y, width, height],
      "area": 1234.5,
      "iscrowd": 0,
      "track_id": 10001,        // video_id * 10000 + instance_id
      "video_id": 1,            // unique video identifier
      "frame_index": 42         // frame number within video
    }
  ],
  "videos": [
    {
      "id": 1,
      "name": "col_1-2021-06-08_17-44-01.mjpeg",
      "num_frames": 1800,
      "frame_indices": [1, 5, 10, ...]
    }
  ]
}
```

This structure enables:
- Training models with temporal consistency
- Instance tracking across frames
- Video-aware data augmentation
- Future integration with tracking algorithms

### Custom Detectron2 Hooks

Three custom hooks integrate with Detectron2's training loop:

1. **ProgressHook**: Emits training metrics every 0.5s (throttled to avoid GUI lag)
2. **ValidationHook**: Runs COCO evaluation, tracks best AP, saves checkpoints
3. **VisualizationHook**: Generates prediction samples for visual inspection

All hooks communicate with the GUI via Qt signals for thread-safe updates.
