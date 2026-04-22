"""
YOLO Instance-Focused Fine-Tuning Training Worker
Generates cropped, resized regions around individual bees with ONLY the primary instance labeled.
This differs from Stage2 which includes all visible bees in each crop.
"""

import json
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
import torch
from ultralytics import YOLO

try:
    import albumentations as A
    from ultralytics.data.augment import Albumentations
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class YOLOTrainingWorkerInstanceFocused(QThread):
    """Worker thread for instance-focused YOLO training (single-instance refinement)"""
    
    # Signals
    stage_update = pyqtSignal(str)  # Stage description
    progress_update = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    training_complete = pyqtSignal(str, dict)  # model_path, final_metrics
    training_failed = pyqtSignal(str)  # error_message
    
    def __init__(self, project_path, config):
        super().__init__()
        self.project_path = Path(project_path)
        self.config = config
        self.should_stop = False
        
    def stop(self):
        """Request training to stop"""
        self.should_stop = True
        
    def run(self):
        """Main training workflow for instance-focused training"""
        try:
            # Step 1: Check COCO annotations exist
            self.stage_update.emit("Checking COCO annotations...")
            train_dir = self.project_path / 'annotations/coco/train'
            val_dir = self.project_path / 'annotations/coco/val'
            
            train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
            val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
            
            if not train_jsons and not val_jsons:
                raise FileNotFoundError(
                    "COCO annotations not found. Please export annotations first."
                )
            
            # Step 2: Generate instance-focused crops (only primary instance labeled)
            self.stage_update.emit("Generating instance-focused training crops...")
            instance_dir = self._generate_instance_focused_crops(train_jsons, val_jsons)
            
            if self.should_stop:
                return
            
            # Step 3: Create dataset YAML
            self.stage_update.emit("Creating dataset configuration...")
            dataset_yaml = self._create_dataset_yaml(instance_dir)
            
            # Step 4: Backup current model
            self._backup_current_model()
            
            if self.should_stop:
                return
            
            # Step 5: Train model
            self.stage_update.emit("Starting training...")
            model_path = self._train_model(dataset_yaml)
            
            if self.should_stop:
                self.stage_update.emit("Training cancelled")
                return
            
            # Step 6: Get final metrics
            final_metrics = self._get_final_metrics(model_path)
            
            # Signal completion
            self.training_complete.emit(str(model_path), final_metrics)
            
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.training_failed.emit(error_msg)
            
    def _generate_instance_focused_crops(self, train_jsons, val_jsons):
        """Generate cropped, resized regions around each bee with ONLY the primary instance labeled"""
        instance_dir = self.project_path / 'yolo_instance_focused'

        # Clean previous outputs to avoid mixing stale crops with new splits
        images_root = instance_dir / 'images'
        labels_root = instance_dir / 'labels'
        if images_root.exists():
            shutil.rmtree(images_root)
        if labels_root.exists():
            shutil.rmtree(labels_root)
        
        # Create directories
        for split in ['train', 'val']:
            (instance_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (instance_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Get configuration
        padding_mode = self.config.get('crop_padding_mode', 'absolute')  # 'absolute' or 'relative'
        padding_px = self.config.get('crop_padding_px', 50)  # Absolute padding in pixels
        padding_pct = self.config.get('crop_padding_pct', 0.3)  # Relative padding (percentage)
        target_size = self.config.get('target_crop_size', 640)  # Resize all crops to this
        min_size = self.config.get('min_crop_size', 128)  # Skip if too small after padding
        maintain_aspect = self.config.get('maintain_aspect_ratio', False)  # Whether to maintain aspect ratio
        
        # Process training set (multiple JSON files)
        train_crops = 0
        for train_json in train_jsons:
            train_crops += self._process_coco_to_instance_crops(
                train_json,
                instance_dir / 'images' / 'train',
                instance_dir / 'labels' / 'train',
                padding_mode,
                padding_px,
                padding_pct,
                target_size,
                min_size,
                maintain_aspect
            )
            if self.should_stop:
                return instance_dir
        
        # Process validation set (multiple JSON files)
        val_crops = 0
        for val_json in val_jsons:
            val_crops += self._process_coco_to_instance_crops(
                val_json,
                instance_dir / 'images' / 'val',
                instance_dir / 'labels' / 'val',
                padding_mode,
                padding_px,
                padding_pct,
                target_size,
                min_size,
                maintain_aspect
            )
            if self.should_stop:
                return instance_dir
        
        self.stage_update.emit(
            f"Generated {train_crops} training crops, {val_crops} validation crops"
        )
        
        return instance_dir
    
    def _process_coco_to_instance_crops(self, coco_json_path, output_img_dir, output_label_dir,
                                         padding_mode, padding_px, padding_pct, target_size,
                                         min_size, maintain_aspect):
        """
        Convert COCO annotations to instance-focused cropped YOLO format.
        For each annotation, create a resized crop with ONLY that instance's mask.
        """
        # Load COCO annotations
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create image ID to annotations mapping
        img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)
        
        crop_count = 0
        
        # Process each image
        for img_info in coco_data['images']:
            if self.should_stop:
                break
                
            img_id = img_info['id']
            img_filename = img_info['file_name']
            
            # Construct image path
            img_path = Path(img_filename)
            if not img_path.is_absolute():
                img_path = self.project_path / img_filename
            
            if not img_path.exists():
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            
            # Get annotations for this image
            gt_anns = img_id_to_anns.get(img_id, [])
            if not gt_anns:
                continue
            
            # For each annotation, create an instance-focused crop
            for ann_idx, ann in enumerate(gt_anns):
                # Skip chamber (category_id=3) and hive (category_id=2) annotations
                category_id = ann.get('category_id', 1)
                if category_id in [2, 3]:  # hive=2, chamber=3
                    continue
                
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Calculate padding based on mode
                if padding_mode == 'absolute':
                    pad_w = padding_px
                    pad_h = padding_px
                else:  # relative
                    pad_w = w * padding_pct
                    pad_h = h * padding_pct
                
                # Calculate crop bounds (before resize)
                crop_x1 = max(0, int(x - pad_w))
                crop_y1 = max(0, int(y - pad_h))
                crop_x2 = min(img_w, int(x + w + pad_w))
                crop_y2 = min(img_h, int(y + h + pad_h))
                
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                
                # Skip if too small
                if crop_w < min_size or crop_h < min_size:
                    continue
                
                # Crop the image
                crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                # Create mask for this PRIMARY instance only
                primary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                
                if 'segmentation' in ann and ann['segmentation']:
                    segmentation = ann['segmentation'][0]
                    pts = []
                    for i in range(0, len(segmentation), 2):
                        pts.append([int(segmentation[i]), int(segmentation[i + 1])])
                    pts = np.array(pts, dtype=np.int32)
                    cv2.fillPoly(primary_mask, [pts], 255)
                else:
                    # If no segmentation, skip this annotation
                    continue
                
                # Crop the primary instance's mask
                cropped_mask = primary_mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                # Check if mask has visible pixels
                if np.count_nonzero(cropped_mask) == 0:
                    continue
                
                # Resize crop and mask to target size
                if maintain_aspect:
                    # Maintain aspect ratio by padding
                    scale = min(target_size / crop_w, target_size / crop_h)
                    new_w = int(crop_w * scale)
                    new_h = int(crop_h * scale)
                    
                    # Resize
                    resized_crop = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Create padded versions
                    final_crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                    final_mask = np.zeros((target_size, target_size), dtype=np.uint8)
                    
                    # Center the resized content
                    pad_top = (target_size - new_h) // 2
                    pad_left = (target_size - new_w) // 2
                    final_crop[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_crop
                    final_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_mask
                    
                    # Update dimensions for coordinate normalization
                    resize_w = target_size
                    resize_h = target_size
                    offset_x = pad_left
                    offset_y = pad_top
                    scale_x = new_w / crop_w
                    scale_y = new_h / crop_h
                else:
                    # Square resize (may distort)
                    final_crop = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
                    final_mask = cv2.resize(cropped_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                    
                    resize_w = target_size
                    resize_h = target_size
                    scale_x = target_size / crop_w
                    scale_y = target_size / crop_h
                    offset_x = 0
                    offset_y = 0
                
                # Convert mask to polygon
                contours, _ = cv2.findContours(
                    final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if len(contours) == 0:
                    continue
                
                # Get the largest contour (primary instance)
                main_contour = max(contours, key=cv2.contourArea)
                
                # Filter out very small contours
                if cv2.contourArea(main_contour) < 10:
                    continue
                
                # Simplify contour
                perimeter = cv2.arcLength(main_contour, True)
                epsilon = 0.001 * perimeter
                simplified = cv2.approxPolyDP(main_contour, epsilon, True)
                
                # Convert to YOLO format (normalized coordinates relative to resized crop)
                yolo_polygon = []
                for point in simplified:
                    px, py = point[0]
                    norm_x = max(0.0, min(1.0, px / resize_w))
                    norm_y = max(0.0, min(1.0, py / resize_h))
                    yolo_polygon.extend([norm_x, norm_y])
                
                # Require at least 3 points (6 coordinates)
                if len(yolo_polygon) < 6:
                    continue
                
                # Generate unique filename
                coco_path = Path(img_filename)
                video_name = coco_path.parent.name
                frame_name = coco_path.stem
                crop_filename = f"{video_name}_{frame_name}_inst{ann_idx:03d}.jpg"
                label_filename = f"{video_name}_{frame_name}_inst{ann_idx:03d}.txt"
                
                # Save resized crop
                crop_img_path = output_img_dir / crop_filename
                cv2.imwrite(str(crop_img_path), final_crop)
                
                # Save label (ONLY the primary instance)
                label_path = output_label_dir / label_filename
                with open(label_path, 'w') as f:
                    # Only bee annotations (category_id=1) are included
                    # YOLO uses 0-based indexing, so bee=0
                    class_id = 0
                    yolo_label = [class_id] + yolo_polygon
                    f.write(' '.join(map(str, yolo_label)) + '\n')
                
                crop_count += 1
        
        return crop_count
    
    def _create_dataset_yaml(self, instance_dir):
        """Create YAML configuration for instance-focused dataset"""
        dataset_yaml = {
            'path': str(instance_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'bee'},
            'nc': 1
        }
        
        yaml_path = instance_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        return yaml_path
    
    def _backup_current_model(self):
        """Backup the currently used instance-focused model"""
        models_dir = self.project_path / 'models'
        current_model = models_dir / 'yolo_instance_focused_best.pt'
        
        if current_model.exists():
            backup_path = models_dir / 'yolo_instance_focused_best_backup.pt'
            shutil.copy2(current_model, backup_path)
            self.stage_update.emit(f"Current model backed up to: {backup_path.name}")
    
    def _train_model(self, dataset_yaml):
        """Train instance-focused YOLO model"""
        # Get base model from config, default to yolov8s-seg.pt
        base_model = self.config.get('base_model', 'yolov8s-seg.pt')
        
        # Initialize model
        model = YOLO(base_model)
        
        # Training parameters
        output_dir = self.project_path / 'models' / 'yolo_instance_focused_runs'
        training_params = {
            'data': str(dataset_yaml),
            'epochs': self.config.get('epochs', 100),
            'imgsz': self.config.get('imgsz', 640),
            'batch': self.config.get('batch', 16),  # Can use larger batch with uniform sizes
            'name': self.config.get('name', 'bee_segmentation_instance_focused'),
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
            'overlap_mask': False,  # No overlapping masks since we have single instances
            'verbose': True,
            
            # Augmentation parameters
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 20.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 5.0,
            'perspective': 0.0,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.0,  # Disable mosaic for instance-focused training
            'mixup': 0.0,   # Disable mixup for single-instance focus
        }
        
        # Add custom Albumentations for blur augmentation (to handle blurry videos)
        if ALBUMENTATIONS_AVAILABLE:
            custom_transforms = [
                A.Blur(blur_limit=(3, 15), p=0.15),  # Gaussian blur with 15% probability
                A.MedianBlur(blur_limit=15, p=0.15),  # Median blur with 15% probability
            ]
            # Monkey-patch the Albumentations class into the model's data augmentation
            # This will be applied during data loading
            training_params['augment'] = True  # Enable augmentation
            model.add_callback('on_train_start', 
                lambda trainer: setattr(trainer.train_loader.dataset, 'albumentations',
                    Albumentations(p=1.0, transforms=custom_transforms)) 
                    if hasattr(trainer, 'train_loader') and hasattr(trainer.train_loader, 'dataset') else None
            )
        
        # Training callback for progress
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
                    try:
                        if hasattr(trainer.tloss, 'item'):
                            self.current_metrics['train_loss'] = trainer.tloss.item()
                        else:
                            self.current_metrics['train_loss'] = float(trainer.tloss)
                    except (ValueError, RuntimeError):
                        if hasattr(trainer.tloss, 'mean'):
                            self.current_metrics['train_loss'] = trainer.tloss.mean().item()
                        else:
                            self.current_metrics['train_loss'] = 0.0
                
            def on_val_end(self, validator):
                """Called after validation completes"""
                if self.worker.should_stop:
                    return
                
                # Get validation metrics
                if hasattr(validator, 'metrics') and validator.metrics is not None:
                    val_metrics = validator.metrics
                    
                    # For instance-focused refinement, we care more about IoU than mAP
                    # Extract mask IoU metrics if available
                    if hasattr(val_metrics, 'seg') and val_metrics.seg is not None:
                        seg_metrics = val_metrics.seg
                        
                        # Try to get mean IoU from results
                        # YOLO computes IoU during validation for mAP calculation
                        # The mean_results attribute contains per-class metrics including IoU
                        if hasattr(seg_metrics, 'mean_results'):
                            # mean_results format: [precision, recall, mAP50, mAP50-95, ...]
                            # For segmentation, YOLO also tracks mask IoU
                            mean_results = seg_metrics.mean_results()
                            if len(mean_results) > 0:
                                # Use mAP50 as a proxy for IoU@0.5 (they're related)
                                # mAP at IoU=0.5 threshold essentially measures IoU > 0.5
                                self.current_metrics['Mask_IoU@0.5'] = float(seg_metrics.map50)
                                self.current_metrics['Mask_IoU@0.5:0.95'] = float(seg_metrics.map)
                        else:
                            # Fallback: use mAP metrics (which are based on IoU thresholds)
                            if hasattr(seg_metrics, 'map50'):
                                self.current_metrics['Mask_IoU@0.5'] = float(seg_metrics.map50)
                            if hasattr(seg_metrics, 'map'):
                                self.current_metrics['Mask_IoU@0.5:0.95'] = float(seg_metrics.map)
                    
                    # Also track box metrics for reference
                    if hasattr(val_metrics, 'box') and val_metrics.box is not None:
                        box_metrics = val_metrics.box
                        if hasattr(box_metrics, 'map50'):
                            self.current_metrics['Box_IoU@0.5'] = float(box_metrics.map50)
                        if hasattr(box_metrics, 'map'):
                            self.current_metrics['Box_IoU@0.5:0.95'] = float(box_metrics.map)
                
                # Get total epochs
                total_epochs = validator.args.epochs if hasattr(validator, 'args') else 100
                
                # Emit progress
                self.worker.progress_update.emit(
                    self.current_epoch,
                    total_epochs,
                    self.current_metrics.copy()
                )
                
                # Clear for next epoch
                self.current_metrics = {}
        
        # Add callbacks
        callback = ProgressCallback(self)
        model.add_callback('on_train_epoch_end', callback.on_train_epoch_end)
        model.add_callback('on_val_end', callback.on_val_end)
        
        # Train
        results = model.train(**training_params)
        
        # Return best model path
        best_model_path = output_dir / self.config.get('name', 'bee_segmentation_instance_focused') / 'weights' / 'best.pt'
        return best_model_path
    
    def _get_final_metrics(self, model_path):
        """Get final validation metrics"""
        try:
            import pandas as pd
            results_csv = model_path.parent.parent / 'results.csv'
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                # Get last row metrics
                last_row = df.iloc[-1]
                # For instance-focused, report as IoU metrics (mAP is computed from IoU thresholds)
                metrics = {
                    'Mask_IoU@0.5': last_row.get('metrics/mAP50(M)', 0.0),
                    'Mask_IoU@0.5:0.95': last_row.get('metrics/mAP50-95(M)', 0.0),
                    'Box_IoU@0.5': last_row.get('metrics/mAP50(B)', 0.0),
                    'Box_IoU@0.5:0.95': last_row.get('metrics/mAP50-95(B)', 0.0),
                    'precision': last_row.get('metrics/precision(B)', 0.0),
                    'recall': last_row.get('metrics/recall(B)', 0.0),
                }
                return metrics
        except (ImportError, Exception) as e:
            print(f"Could not parse metrics: {e}")
        
        return {}
