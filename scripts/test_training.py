#!/usr/bin/env python3
"""
Simple script to test Detectron2 training and inference
Run this to verify your dataset and training setup works
"""

import os
import sys
from pathlib import Path

# Set memory allocator before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils


def load_and_fix_dataset(json_file, image_root):
    """Load COCO data and fix dimensions/category IDs"""
    print(f"Loading dataset from {json_file}...")
    dataset_dicts = load_coco_json(json_file, image_root)
    
    print(f"Found {len(dataset_dicts)} images")
    
    # Fix dimensions and category IDs
    for i, d in enumerate(dataset_dicts):
        img_path = d['file_name']
        
        # Load actual image to get real dimensions
        try:
            with Image.open(img_path) as img:
                d['height'] = img.height
                d['width'] = img.width
        except Exception as e:
            print(f"ERROR: Could not read {img_path}: {e}")
            continue
        
        # Remap category_id to 0 (Detectron2 expects 0-indexed)
        if 'annotations' in d:
            for anno in d['annotations']:
                if 'category_id' in anno:
                    anno['category_id'] = 0
        
        # Print first image info
        if i == 0:
            print(f"\nFirst image info:")
            print(f"  File: {img_path}")
            print(f"  Dimensions: {d['width']} x {d['height']}")
            print(f"  Annotations: {len(d.get('annotations', []))}")
            if d.get('annotations'):
                print(f"  First annotation category: {d['annotations'][0].get('category_id')}")
    
    return dataset_dicts


def test_dataloader(cfg):
    """Test if dataloader can iterate without errors"""
    print("\n=== Testing DataLoader ===")
    
    try:
        data_loader = build_detection_train_loader(cfg)
        print(f"✓ DataLoader created successfully")
        
        print("Fetching first batch...")
        data_iter = iter(data_loader)
        batch = next(data_iter)
        
        print(f"✓ Got batch with {len(batch)} images")
        print(f"  Image shape: {batch[0]['image'].shape}")
        print(f"  Instances: {batch[0]['instances']}")
        
        return True
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training(cfg, max_iter=5):
    """Test if training can run for a few iterations"""
    print(f"\n=== Testing Training ({max_iter} iterations) ===")
    
    cfg.SOLVER.MAX_ITER = max_iter
    
    try:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        
        print("✓ Trainer created, starting training...")
        trainer.train()
        
        print(f"✓ Training completed {max_iter} iterations successfully!")
        return True
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Get project path from command line
    if len(sys.argv) < 2:
        print("Usage: python test_training.py <project_path>")
        print()
        print("Example: python test_training.py projects/my_project")
        print()
        print("The project directory should contain:")
        print("  - coco_train.json (training annotations)")
        print("  - coco_val.json (validation annotations)")
        print("  - frame images")
        sys.exit(1)
    
    # Configuration
    PROJECT_PATH = Path(sys.argv[1])
    TRAIN_JSON = PROJECT_PATH / "coco_train.json"
    VAL_JSON = PROJECT_PATH / "coco_val.json"
    
    # Check project directory exists
    if not PROJECT_PATH.exists():
        print(f"ERROR: Project directory not found: {PROJECT_PATH}")
        sys.exit(1)
    
    if not TRAIN_JSON.exists():
        print(f"ERROR: Training COCO file not found: {TRAIN_JSON}")
        print(f"Expected to find coco_train.json in {PROJECT_PATH}")
        sys.exit(1)
    
    if not VAL_JSON.exists():
        print(f"ERROR: Validation COCO file not found: {VAL_JSON}")
        print(f"Expected to find coco_val.json in {PROJECT_PATH}")
        sys.exit(1)
    
    print("="*60)
    print("Detectron2 Training Test Script")
    print("="*60)
    print(f"\nProject: {PROJECT_PATH}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  Free memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    else:
        print("\n✗ CUDA not available, training will be very slow")
    
    # Load and register datasets
    print("\n=== Loading Datasets ===")
    
    train_data = load_and_fix_dataset(str(TRAIN_JSON), str(PROJECT_PATH))
    val_data = load_and_fix_dataset(str(VAL_JSON), str(PROJECT_PATH))
    
    # Register with Detectron2
    DatasetCatalog.register("test_train", lambda: train_data)
    DatasetCatalog.register("test_val", lambda: val_data)
    MetadataCatalog.get("test_train").thing_classes = ["bee"]
    MetadataCatalog.get("test_val").thing_classes = ["bee"]
    
    print("\n✓ Datasets registered")
    
    # Setup configuration
    print("\n=== Configuring Model ===")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    cfg.DATASETS.TRAIN = ("test_train",)
    cfg.DATASETS.TEST = ("test_val",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Memory-saving settings
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Reduce image sizes for limited memory
    cfg.INPUT.MIN_SIZE_TRAIN = (384, 416, 448, 480)
    cfg.INPUT.MAX_SIZE_TRAIN = 768
    cfg.INPUT.MIN_SIZE_TEST = 480
    cfg.INPUT.MAX_SIZE_TEST = 768
    
    # Enable FP16
    cfg.SOLVER.AMP.ENABLED = True
    
    # Reduce RPN proposals
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 500
    
    # Reduce FPN channels
    cfg.MODEL.FPN.OUT_CHANNELS = 128
    
    # Output directory
    cfg.OUTPUT_DIR = str(PROJECT_PATH / "models" / "test_training")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Configuration complete")
    print(f"  Model: Mask R-CNN R50-FPN")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Image sizes: {cfg.INPUT.MIN_SIZE_TRAIN} - {cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"  FP16: {cfg.SOLVER.AMP.ENABLED}")
    
    # Test dataloader
    if not test_dataloader(cfg):
        print("\n✗ DataLoader test failed, stopping here")
        sys.exit(1)
    
    # Test training
    if not test_training(cfg, max_iter=5):
        print("\n✗ Training test failed")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYour dataset and training setup are working correctly.")
    print(f"Model checkpoints saved to: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
