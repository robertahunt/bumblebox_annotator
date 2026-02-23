"""
YOLO SAHI (Sliced Aided Hyper Inference) Training Worker
Trains on full images with enhanced augmentation and configures for sliced inference
"""

import json
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from time import sleep
from PyQt6.QtCore import QThread, pyqtSignal

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOTrainingWorkerSAHI(QThread):
    """Background worker for YOLO model training with SAHI support"""
    
    # Signals
    progress_update = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    training_complete = pyqtSignal(str, dict)  # model_path, final_metrics
    training_failed = pyqtSignal(str)  # error_message
    stage_update = pyqtSignal(str)  # stage description
    
    def __init__(self, project_path, training_config):
        """
        Initialize YOLO SAHI training worker
        
        Args:
            project_path: Path to project directory
            training_config: Dict with training parameters
        """
        super().__init__()
        self.project_path = Path(project_path)
        self.config = training_config
        self.should_stop = False
        
    def stop(self):
        """Request training to stop"""
        self.should_stop = True
        
    def run(self):
        """Run training workflow"""
        try:
            if not YOLO_AVAILABLE:
                self.training_failed.emit("Ultralytics YOLO not installed. Install with: pip install ultralytics")
                return
                
            # Step 1: Prepare data
            self.stage_update.emit("Preparing training data (full images)...")
            yolo_dir, dataset_yaml = self._prepare_yolo_dataset()
            
            if self.should_stop:
                return
                
            # Step 2: Backup current model
            self.stage_update.emit("Backing up current model...")
            self._backup_current_model()
            
            if self.should_stop:
                return
                
            # Step 3: Train model with enhanced augmentation
            self.stage_update.emit("Training model with SAHI-optimized augmentation...")
            model_path = self._train_model(dataset_yaml)
            
            if self.should_stop:
                return
                
            # Step 4: Get final metrics
            final_metrics = self._get_final_metrics(model_path)
            
            self.training_complete.emit(str(model_path), final_metrics)
            
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.training_failed.emit(error_msg)
            
    def _prepare_yolo_dataset(self):
        """Convert COCO format to YOLO format with tiled crops for SAHI training"""
        # Look for per-video JSON files in train and val folders
        train_dir = self.project_path / 'annotations/coco/train'
        val_dir = self.project_path / 'annotations/coco/val'
        
        train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
        val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
        
        if not train_jsons and not val_jsons:
            raise FileNotFoundError("COCO format annotations not found. Export annotations first.")
        
        # Create YOLO format directory
        yolo_dir = self.project_path / 'yolo_sahi'
        yolo_dir.mkdir(exist_ok=True)

        # Clean previous outputs to avoid mixing stale frames with new splits
        images_root = yolo_dir / 'images'
        labels_root = yolo_dir / 'labels'
        if images_root.exists():
            shutil.rmtree(images_root)
        if labels_root.exists():
            shutil.rmtree(labels_root)
        
        # Get crop configuration
        crop_size = self.config.get('crop_size', 640)  # Match SAHI slice size
        max_crops_per_image = self.config.get('max_crops_per_image', 10)  # (Unused - kept for compatibility)
        include_full_images = self.config.get('include_full_images', True)  # Also include full images
        
        # Convert train and val sets with tiled cropping
        train_full, train_crops = 0, 0
        for json_file in train_jsons:
            full, crops = self._coco_to_yolo_with_crops(
                json_file, yolo_dir, 'train',
                crop_size, max_crops_per_image, include_full_images
            )
            train_full += full
            train_crops += crops
            
        val_full, val_crops = 0, 0
        for json_file in val_jsons:
            full, crops = self._coco_to_yolo_with_crops(
                json_file, yolo_dir, 'val',
                crop_size, max_crops_per_image, include_full_images
            )
            val_full += full
            val_crops += crops
        
        # Create dataset YAML
        dataset_yaml = yolo_dir / 'dataset.yaml'
        yaml_content = {
            'path': str(yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'bee'}
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        total_train = train_full + train_crops
        total_val = val_full + val_crops
        
        self.stage_update.emit(
            f"Dataset prepared: train ({train_full} full, {train_crops} crops, {total_train} total), "
            f"val ({val_full} full, {val_crops} crops, {total_val} total)"
        )
        
        # Warn if validation set is very small (may not generate validation batch plots)
        if total_val < self.config.get('batch', 8) * 3:
            self.stage_update.emit(
                f"⚠ Warning: Validation set has only {total_val} images. "
                f"Ultralytics may not generate validation batch plots (val_batch*.jpg) "
                f"if there are fewer than 3 batches of size {self.config.get('batch', 8)}."
            )
        
        return yolo_dir, dataset_yaml
        
    def _coco_to_yolo_with_crops(self, coco_json_path, output_dir, split='train',
                                   crop_size=640, max_crops=10, include_full=True):
        """
        Convert COCO format to YOLO with tiled crops that cover the entire image
        Crops are arranged in a grid pattern with random position and size variations
        
        Args:
            coco_json_path: Path to COCO JSON file
            output_dir: Output directory
            split: 'train' or 'val'
            crop_size: Base size for crops (will match SAHI slice size)
            max_crops: (Unused - kept for API compatibility)
            include_full: Whether to also include full images
        
        Returns:
            Tuple of (num_full_images, num_crops)
        """
        with open(coco_json_path, 'r') as f:
            coco = json.load(f)
        
        # Create output directories
        images_dir = output_dir / 'images' / split
        labels_dir = output_dir / 'labels' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Build image mapping
        images_dict = {img['id']: img for img in coco['images']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        full_count = 0
        crop_count = 0
        
        for img_id, img_info in images_dict.items():
            if self.should_stop:
                break
                
            src_img_path = Path(img_info['file_name'])
            if not src_img_path.is_absolute():
                src_img_path = self.project_path / img_info['file_name']
            
            if not src_img_path.exists():
                continue
            
            # Load image with cv2 for cropping
            img = cv2.imread(str(src_img_path))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            annotations = img_annotations.get(img_id, [])
            
            # Create unique base filename
            video_name = src_img_path.parent.name
            original_name = src_img_path.stem
            extension = src_img_path.suffix
            
            # Save full image if requested
            if include_full:
                full_filename = f"{video_name}_{original_name}{extension}"
                full_label_name = f"{video_name}_{original_name}.txt"
                
                dst_img_path = images_dir / full_filename
                cv2.imwrite(str(dst_img_path), img)
                
                # Create label file for full image
                label_file = labels_dir / full_label_name
                if self._write_yolo_labels(label_file, annotations, img_w, img_h):
                    full_count += 1
            
            # Generate tiled crops that cover the entire image with randomness
            if annotations:
                # Calculate number of rows and columns needed to cover the image
                n_cols = int(np.ceil(img_w / crop_size))
                n_rows = int(np.ceil(img_h / crop_size))
                
                # Calculate the step size for tile centers
                step_x = img_w / n_cols
                step_y = img_h / n_rows
                
                crop_idx = 0
                for row in range(n_rows):
                    for col in range(n_cols):
                        if self.should_stop:
                            break
                        
                        # Calculate base center position for this tile
                        base_center_x = (col + 0.5) * step_x
                        base_center_y = (row + 0.5) * step_y
                        
                        # Add random noise to center position (up to 20% of step size)
                        noise_x = np.random.uniform(-0.2 * step_x, 0.2 * step_x)
                        noise_y = np.random.uniform(-0.2 * step_y, 0.2 * step_y)
                        
                        center_x = base_center_x + noise_x
                        center_y = base_center_y + noise_y
                        
                        # Add random noise to crop size (±10%)
                        noise_factor = np.random.uniform(0.9, 1.1)
                        noisy_crop_size = int(crop_size * noise_factor)
                        
                        # Calculate crop coordinates
                        crop_x = int(center_x - noisy_crop_size / 2)
                        crop_y = int(center_y - noisy_crop_size / 2)
                        
                        # Ensure crop is within image bounds
                        crop_x = max(0, min(crop_x, img_w - noisy_crop_size))
                        crop_y = max(0, min(crop_y, img_h - noisy_crop_size))
                        
                        # Adjust crop size if it extends beyond image bounds
                        actual_crop_w = min(noisy_crop_size, img_w - crop_x)
                        actual_crop_h = min(noisy_crop_size, img_h - crop_y)
                        
                        if actual_crop_w < 320 or actual_crop_h < 320:  # Skip very small crops
                            continue
                        
                        # Extract crop
                        crop_img = img[crop_y:crop_y+actual_crop_h, crop_x:crop_x+actual_crop_w].copy()
                        
                        # Transform annotations to crop coordinates
                        crop_annotations = self._transform_annotations_to_crop(
                            annotations, crop_x, crop_y, actual_crop_w, actual_crop_h
                        )
                        
                        # Only save crop if it contains annotations
                        if crop_annotations:
                            crop_filename = f"{video_name}_{original_name}_tile_r{row}_c{col}{extension}"
                            crop_label_name = f"{video_name}_{original_name}_tile_r{row}_c{col}.txt"
                            
                            crop_img_path = images_dir / crop_filename
                            cv2.imwrite(str(crop_img_path), crop_img)
                            
                            crop_label_file = labels_dir / crop_label_name
                            if self._write_yolo_labels(crop_label_file, crop_annotations, 
                                                       actual_crop_w, actual_crop_h):
                                crop_count += 1
                                crop_idx += 1
        
        sleep(2)  # Brief pause to ensure all file operations are complete
        return full_count, crop_count
    
    def _transform_annotations_to_crop(self, annotations, crop_x, crop_y, crop_w, crop_h):
        """Transform annotations to crop coordinate space"""
        crop_annotations = []
        
        for ann in annotations:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            for seg in ann['segmentation']:
                if len(seg) < 6:
                    continue
                
                # Transform segmentation points to crop coordinates
                transformed_seg = []
                all_inside = True
                any_inside = False
                
                for i in range(0, len(seg), 2):
                    x = seg[i] - crop_x
                    y = seg[i+1] - crop_y
                    
                    # Check if point is inside crop
                    if 0 <= x <= crop_w and 0 <= y <= crop_h:
                        any_inside = True
                    else:
                        all_inside = False
                    
                    # Clip to crop bounds
                    x = max(0, min(x, crop_w))
                    y = max(0, min(y, crop_h))
                    
                    transformed_seg.extend([x, y])
                
                # Only include annotation if at least some part is in the crop
                if any_inside and len(transformed_seg) >= 6:
                    crop_ann = ann.copy()
                    crop_ann['segmentation'] = [transformed_seg]
                    crop_annotations.append(crop_ann)
                    break  # Only take first segmentation per annotation
        
        return crop_annotations
    
    def _write_yolo_labels(self, label_file, annotations, img_width, img_height):
        """Write YOLO format labels to file"""
        if img_width == 0 or img_height == 0:
            return False
        
        labels_written = False
        with open(label_file, 'w') as f:
            for ann in annotations:
                class_id = 0  # Single class
                
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                for seg in ann['segmentation']:
                    if len(seg) < 6:
                        continue
                    
                    # Normalize coordinates
                    normalized_seg = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / img_width
                        y = seg[i+1] / img_height
                        # Ensure coordinates are in valid range
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        normalized_seg.extend([x, y])
                    
                    if len(normalized_seg) >= 6:
                        line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_seg])
                        f.write(line + "\n")
                        labels_written = True
        
        return labels_written
    
    def _coco_to_yolo(self, coco_json_path, output_dir, split='train'):
        """Legacy method - convert COCO format to YOLO segmentation format (full images only)"""
        with open(coco_json_path, 'r') as f:
            coco = json.load(f)
        
        # Create output directories
        images_dir = output_dir / 'images' / split
        labels_dir = output_dir / 'labels' / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Build image mapping
        images_dict = {img['id']: img for img in coco['images']}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        converted_count = 0
        for img_id, img_info in images_dict.items():
            src_img_path = Path(img_info['file_name'])
            if not src_img_path.is_absolute():
                src_img_path = self.project_path / img_info['file_name']
            
            if not src_img_path.exists():
                continue
            
            # Get actual image dimensions
            try:
                with Image.open(src_img_path) as img:
                    img_width = img.width
                    img_height = img.height
            except Exception:
                img_width = img_info.get('width', 0)
                img_height = img_info.get('height', 0)
                if img_width == 0 or img_height == 0:
                    continue
            
            # Create unique filename
            video_name = src_img_path.parent.name
            original_name = src_img_path.stem
            extension = src_img_path.suffix
            unique_filename = f"{video_name}_{original_name}{extension}"
            
            # Copy image
            dst_img_path = images_dir / unique_filename
            shutil.copy2(src_img_path, dst_img_path)
            
            # Create label file
            label_file = labels_dir / f"{video_name}_{original_name}.txt"
            annotations = img_annotations.get(img_id, [])
            
            with open(label_file, 'w') as f:
                for ann in annotations:
                    class_id = 0  # Single class
                    
                    if 'segmentation' not in ann or not ann['segmentation']:
                        continue
                    
                    for seg in ann['segmentation']:
                        if len(seg) < 6:
                            continue
                        
                        # Normalize coordinates
                        normalized_seg = []
                        for i in range(0, len(seg), 2):
                            x = seg[i] / img_width
                            y = seg[i+1] / img_height
                            normalized_seg.extend([x, y])
                        
                        line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_seg])
                        f.write(line + "\n")
            
            converted_count += 1
            
        return converted_count
        
    def _backup_current_model(self):
        """Backup the currently used model"""
        current_model = self.config.get('current_model_path')
        if current_model and Path(current_model).exists():
            backup_path = Path(current_model).parent / 'backup_sahi_best.pt'
            shutil.copy2(current_model, backup_path)
            self.stage_update.emit(f"Current model backed up to: {backup_path.name}")
        
    def _train_model(self, dataset_yaml):
        """Train YOLO model with SAHI-optimized parameters"""
        # Initialize model
        model = YOLO('yolo26s-seg.pt')
        
        # Training parameters with enhanced augmentation for SAHI
        output_dir = self.project_path / 'models' / 'yolo_sahi_runs'
        training_params = {
            'data': str(dataset_yaml),
            'epochs': self.config.get('epochs', 100),
            'imgsz': self.config.get('imgsz', 640),  # Larger image size for better detail
            'batch': self.config.get('batch', 8),  # Smaller batch due to larger image size
            'name': self.config.get('name', 'bee_segmentation_sahi'),
            'project': str(output_dir.absolute()),
            'patience': self.config.get('patience', 20),
            'save': True,
            'save_period': -1,
            'val': True,  # Enable validation during training
            'plots': True,  # Save validation prediction plots
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'optimizer': 'AdamW',
            'lr0': self.config.get('lr0', 0.001),
            'lrf': 0.01,
            'warmup_epochs': 3.0,
            'overlap_mask': False,  # Prevent overlapping masks for close-together bees
            'verbose': True,
            
            # Enhanced augmentation parameters
            'hsv_h': 0.015,  # Image HSV-Hue augmentation (fraction)
            'hsv_s': 0.7,    # Image HSV-Saturation augmentation (fraction)
            'hsv_v': 0.4,    # Image HSV-Value augmentation (fraction)
            'degrees': 10.0, # Image rotation (+/- deg)
            'translate': 0.2, # Image translation (+/- fraction)
            'scale': 0.9,    # Image scale (+/- gain)
            'shear': 5.0,    # Image shear (+/- deg)
            'perspective': 0.0005, # Image perspective (+/- fraction)
            'flipud': 0.5,   # Image flip up-down (probability)
            'fliplr': 0.5,   # Image flip left-right (probability)
            'mosaic': 0.1,   # Image mosaic (probability)
            'mixup': 0.05,   # Image mixup (probability)
            'cutmix': 0.05,   # Image cutmix (probability)
            'copy_paste': 0.05, # Segment copy-paste (probability) - similar to cutmix
            'copy_paste_mode': 'mixup',  # Copy-paste mode (blend or simple)
            'erasing': 0.1,  # Random erasing augmentation (probability)
            
            # Multi-scale training
            'rect': False,   # Rectangular training (for varied aspect ratios)
            'close_mosaic': 10,  # Disable mosaic augmentation for final N epochs
            'visualize': True,  # Disable visualization during training to speed up
        }
        
        # Train with callback for progress
        class ProgressCallback:
            def __init__(self, worker):
                self.worker = worker
                self.current_metrics = {}
                self.current_epoch = 0
                
            def on_train_epoch_end(self, trainer):
                if self.worker.should_stop:
                    trainer.stop = True
                    return
                    
                self.current_epoch = trainer.epoch + 1
                
                # Get training loss
                if hasattr(trainer, 'tloss'):
                    # tloss can be a tensor or scalar
                    try:
                        if hasattr(trainer.tloss, 'item'):
                            # It's a tensor - convert to scalar
                            self.current_metrics['train_loss'] = trainer.tloss.item()
                        else:
                            self.current_metrics['train_loss'] = float(trainer.tloss)
                    except (ValueError, RuntimeError):
                        # If it's a multi-element tensor, take the mean
                        if hasattr(trainer.tloss, 'mean'):
                            self.current_metrics['train_loss'] = trainer.tloss.mean().item()
                        else:
                            self.current_metrics['train_loss'] = 0.0
                
                # Note: Validation metrics will be updated in on_val_end callback
                
            def on_val_end(self, validator):
                """Called after validation completes - this is where we get validation metrics"""
                if self.worker.should_stop:
                    return
                
                # Get validation metrics
                if hasattr(validator, 'metrics') and validator.metrics is not None:
                    val_metrics = validator.metrics
                    
                    # For segmentation models - box metrics
                    if hasattr(val_metrics, 'box') and val_metrics.box is not None:
                        box_metrics = val_metrics.box
                        if hasattr(box_metrics, 'map50'):
                            self.current_metrics['mAP50'] = float(box_metrics.map50)
                        if hasattr(box_metrics, 'map'):
                            self.current_metrics['mAP50-95'] = float(box_metrics.map)
                    
                    # For segmentation models - mask metrics
                    if hasattr(val_metrics, 'seg') and val_metrics.seg is not None:
                        seg_metrics = val_metrics.seg
                        if hasattr(seg_metrics, 'map50'):
                            self.current_metrics['mAP50_mask'] = float(seg_metrics.map50)
                        if hasattr(seg_metrics, 'map'):
                            self.current_metrics['mAP50-95_mask'] = float(seg_metrics.map)
                
                # Get total epochs from validator's parent trainer
                total_epochs = validator.args.epochs if hasattr(validator, 'args') else 50
                
                # Emit progress update with both training and validation metrics
                self.worker.progress_update.emit(
                    self.current_epoch,
                    total_epochs,
                    self.current_metrics.copy()
                )
                
                # Clear metrics for next epoch
                self.current_metrics = {}
        
        # Add callbacks
        callback = ProgressCallback(self)
        model.add_callback('on_train_epoch_end', callback.on_train_epoch_end)
        model.add_callback('on_val_end', callback.on_val_end)
        
        # Train
        results = model.train(**training_params)
        
        # Return best model path
        best_model_path = output_dir / self.config.get('name', 'bee_segmentation_sahi') / 'weights' / 'best.pt'
        return best_model_path
        
    def _get_final_metrics(self, model_path):
        """Get final validation metrics"""
        try:
            model = YOLO(str(model_path))
            # Get results directory
            results_dir = model_path.parent.parent
            results_file = results_dir / 'results.csv'
            
            if results_file.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(results_file)
                    last_row = df.iloc[-1]
                    
                    return {
                        'mAP50': float(last_row.get('metrics/mAP50(B)', 0)),
                        'mAP50-95': float(last_row.get('metrics/mAP50-95(B)', 0)),
                        'precision': float(last_row.get('metrics/precision(B)', 0)),
                        'recall': float(last_row.get('metrics/recall(B)', 0)),
                    }
                except ImportError:
                    print("pandas not available, skipping metrics parsing")
            
        except Exception as e:
            print(f"Could not load final metrics: {e}")
            
        return {}
