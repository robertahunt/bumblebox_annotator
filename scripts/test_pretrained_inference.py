#!/usr/bin/env python3
"""
Test inference using Detectron2 models (pretrained COCO or custom trained)
This shows what different models detect on your images
"""

import cv2
import sys
import argparse
from pathlib import Path
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


def setup_predictor(model_path=None, num_classes=None, score_threshold=0.5):
    """
    Setup predictor with either pretrained COCO weights or custom checkpoint
    
    Args:
        model_path: Path to custom model checkpoint. If None, uses COCO pretrained weights.
        num_classes: Number of classes for custom model. If None, uses 80 (COCO classes).
        score_threshold: Confidence threshold for detections
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    if model_path:
        # Custom trained model
        print(f"Loading custom model from: {model_path}")
        cfg.MODEL.WEIGHTS = str(model_path)
        
        # Set number of classes for custom model
        if num_classes is not None:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        else:
            # Default to 1 for single-class models (like bee detection)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    else:
        # Use pretrained COCO weights
        print("Loading pretrained COCO model...")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    
    # Set threshold for detection
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    
    # Set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return DefaultPredictor(cfg), cfg


def run_inference_on_image(image_path, output_path, model_path=None, num_classes=None, 
                          class_names=None, score_threshold=0.5):
    """
    Run inference and save visualization
    
    Args:
        image_path: Path to input image
        output_path: Path to save visualization
        model_path: Path to custom model checkpoint (None for COCO pretrained)
        num_classes: Number of classes for custom model
        class_names: List of class names for custom model
        score_threshold: Confidence threshold
    """
    print(f"Loading image: {image_path}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False
    
    print(f"Image shape: {img.shape}")
    
    # Setup predictor
    if model_path:
        print(f"Setting up custom model predictor...")
    else:
        print("Setting up pretrained COCO model predictor...")
    
    predictor, cfg = setup_predictor(model_path, num_classes, score_threshold)
    
    # Determine metadata and class names
    if model_path:
        # Custom model
        if class_names is None:
            class_names = ["bee"] if num_classes == 1 else [f"class_{i}" for i in range(num_classes or 1)]
        
        # Register custom metadata
        from detectron2.data import MetadataCatalog
        metadata = MetadataCatalog.get("custom_dataset")
        metadata.thing_classes = class_names
    else:
        # COCO model
        metadata = MetadataCatalog.get("coco_2017_val")
        class_names = metadata.thing_classes
    
    print("Running inference...")
    
    # Run inference
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    
    # Get predictions
    num_instances = len(instances)
    print(f"\nDetected {num_instances} instances")
    
    if num_instances > 0:
        # Get classes and scores
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        
        # Check if masks are available
        has_masks = instances.has("pred_masks")
        print(f"Has segmentation masks: {has_masks}")
        
        print("\nDetections:")
        for i, (cls, score) in enumerate(zip(classes, scores)):
            class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            print(f"  {i+1}. {class_name} (confidence: {score:.3f})")
    else:
        if model_path:
            print("No instances detected - model may need more training or lower threshold")
        else:
            print("No instances detected (expected - COCO model doesn't know about bees)")
    
    # Create visualization with enhanced mask visibility
    print("\nCreating visualization...")
    v = Visualizer(
        img[:, :, ::-1],  # BGR to RGB
        metadata=metadata,
        scale=1.0,  # Full scale for better visibility
        instance_mode=ColorMode.IMAGE  # Use IMAGE mode to show masks more clearly
    )
    
    out = v.draw_instance_predictions(instances)
    vis_img = out.get_image()[:, :, ::-1]  # RGB to BGR
    
    # Save visualization
    cv2.imwrite(str(output_path), vis_img)
    print(f"✓ Visualization saved to: {output_path}")
    
    # Also create a side-by-side comparison
    # Ensure both images have the same dimensions
    if img.shape != vis_img.shape:
        # Resize vis_img to match original image
        vis_img = cv2.resize(vis_img, (img.shape[1], img.shape[0]))
    
    comparison = cv2.hconcat([img, vis_img])
    comparison_path = output_path.parent / f"{output_path.stem}_comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"✓ Side-by-side comparison saved to: {comparison_path}")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run Detectron2 inference on images with pretrained or custom models"
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, help="Path to save visualization")
    parser.add_argument("--model", "-m", type=str, 
                       help="Path to custom model checkpoint (.pth file). If not provided, uses COCO pretrained model.")
    parser.add_argument("--num-classes", type=int, default=1,
                       help="Number of classes for custom model (default: 1 for bee detection)")
    parser.add_argument("--class-names", type=str, nargs="+", 
                       help="Names of classes for custom model (e.g., --class-names bee)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Confidence threshold for detections (default: 0.5)")
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        prefix = "custom" if args.model else "pretrained"
        output_path = image_path.parent / f"{prefix}_inference_{image_path.name}"
    
    print("=" * 70)
    print("DETECTRON2 MODEL INFERENCE TEST")
    print("=" * 70)
    print()
    
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"ERROR: Model checkpoint not found: {model_path}")
            sys.exit(1)
        print(f"Using custom model: {model_path}")
        print(f"Number of classes: {args.num_classes}")
        if args.class_names:
            print(f"Class names: {', '.join(args.class_names)}")
    else:
        model_path = None
        print("Using pretrained COCO model")
        print("COCO is trained on 80 common object classes (person, car, etc.)")
        print("It does NOT know about bees, so may detect nothing or mis-classify them.")
    
    print(f"Confidence threshold: {args.threshold}")
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Using CPU (will be slower)")
    
    print()
    
    # Run inference
    success = run_inference_on_image(
        image_path, 
        output_path,
        model_path=args.model,
        num_classes=args.num_classes if args.model else None,
        class_names=args.class_names,
        score_threshold=args.threshold
    )
    
    if success:
        print()
        print("=" * 70)
        print("DONE!")
        print("=" * 70)
        if not args.model:
            print()
            print("TIP: To test your custom trained model, use:")
            print(f"  python test_pretrained_inference.py {image_path} \\")
            print("    --model projects/your_project/models/training/model_final.pth \\")
            print("    --threshold 0.3")


if __name__ == "__main__":
    main()
