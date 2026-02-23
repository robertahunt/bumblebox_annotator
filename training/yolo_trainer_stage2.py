"""
YOLO Stage 2 (Fine-Grained) Training Worker
Generates crops around detected bees and trains a refinement model
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


class YOLOTrainingWorkerStage2(QThread):
    """Worker thread for Stage 2 YOLO training (fine-grained refinement)"""
    
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
        """Main training workflow for Stage 2"""
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
            
            # Step 2: Generate Stage 2 crops (with all visible bees labeled)
            self.stage_update.emit("Generating training crops...")
            stage2_dir = self._generate_stage2_crops(train_jsons, val_jsons)
            
            if self.should_stop:
                return
            
            # Step 3: Create dataset YAML
            self.stage_update.emit("Creating dataset configuration...")
            dataset_yaml = self._create_dataset_yaml(stage2_dir)
            
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
            
    def _generate_stage2_crops(self, train_jsons, val_jsons):
        """Generate cropped regions around each bee with all visible bees labeled"""
        stage2_dir = self.project_path / 'yolo_stage2'

        # Clean previous outputs to avoid mixing stale crops with new splits
        images_root = stage2_dir / 'images'
        labels_root = stage2_dir / 'labels'
        if images_root.exists():
            shutil.rmtree(images_root)
        if labels_root.exists():
            shutil.rmtree(labels_root)
        
        # Create directories
        for split in ['train', 'val']:
            (stage2_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (stage2_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Get configuration
        padding = self.config.get('crop_padding', 0.3)
        min_size = self.config.get('min_crop_size', 128)
        
        # Process training set (multiple JSON files)
        train_crops = 0
        for train_json in train_jsons:
            train_crops += self._process_coco_to_crops(
                train_json,
                stage2_dir / 'images' / 'train',
                stage2_dir / 'labels' / 'train',
                padding,
                min_size
            )
            if self.should_stop:
                return stage2_dir
        
        # Process validation set (multiple JSON files)
        val_crops = 0
        for val_json in val_jsons:
            val_crops += self._process_coco_to_crops(
                val_json,
                stage2_dir / 'images' / 'val',
                stage2_dir / 'labels' / 'val',
                padding,
                min_size
            )
            if self.should_stop:
                return stage2_dir
        
        self.stage_update.emit(
            f"Generated {train_crops} training crops, {val_crops} validation crops"
        )
        
        return stage2_dir
    
    def _process_coco_to_crops(self, coco_json_path, output_img_dir, output_label_dir, 
                                padding, min_size):
        """
        Convert COCO annotations to cropped YOLO format
        For each annotation, create a crop containing ALL visible bees in that region
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
            
            # Create pixel masks for all annotations
            annotation_masks = []
            for ann in gt_anns:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                
                segmentation = ann['segmentation'][0]
                pts = []
                for i in range(0, len(segmentation), 2):
                    pts.append([int(segmentation[i]), int(segmentation[i + 1])])
                pts = np.array(pts, dtype=np.int32)
                
                cv2.fillPoly(mask, [pts], 255)
                
                annotation_masks.append({
                    'mask': mask,
                    'category_id': ann['category_id'],
                    'annotation_id': ann['id']
                })
            
            # For each annotation, create a crop
            for ann_idx, ann in enumerate(gt_anns):
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Add padding
                pad_w = w * padding
                pad_h = h * padding
                
                # Calculate crop bounds
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
                
                # Find all masks visible in this crop
                crop_labels = []
                for mask_data in annotation_masks:
                    # Crop the mask
                    cropped_mask = mask_data['mask'][crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    
                    # Check if mask has visible pixels
                    if np.count_nonzero(cropped_mask) == 0:
                        continue
                    
                    # Convert mask to polygon
                    contours, _ = cv2.findContours(
                        cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if len(contours) == 0:
                        continue
                    
                    # Process ALL contours (not just the largest)
                    # Filter out very small contours (noise)
                    min_contour_area = 10  # pixels
                    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
                    
                    if not valid_contours:
                        continue
                    
                    # keep only main contour
                    valid_contours = [max(valid_contours, key=cv2.contourArea)]

                    # Create a YOLO annotation for each contour
                    for contour in valid_contours:
                        # Simplify contour
                        perimeter = cv2.arcLength(contour, True)
                        epsilon = 0.001 * perimeter
                        simplified = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Convert to YOLO format (normalized coordinates)
                        yolo_polygon = []
                        for point in simplified:
                            px, py = point[0]
                            norm_x = max(0.0, min(1.0, px / crop_w))
                            norm_y = max(0.0, min(1.0, py / crop_h))
                            yolo_polygon.extend([norm_x, norm_y])
                        
                        # Require at least 3 points (6 coordinates)
                        if len(yolo_polygon) >= 6:
                            category_id = mask_data['category_id'] - 1  # YOLO uses 0-based
                            yolo_label = [category_id] + yolo_polygon
                            crop_labels.append(yolo_label)
                
                # Skip if no instances found
                if not crop_labels:
                    continue
                
                # Generate unique filename
                coco_path = Path(img_filename)
                video_name = coco_path.parent.name
                frame_name = coco_path.stem
                crop_filename = f"{video_name}_{frame_name}_crop{ann_idx:03d}.jpg"
                label_filename = f"{video_name}_{frame_name}_crop{ann_idx:03d}.txt"
                
                # Save crop
                crop_img_path = output_img_dir / crop_filename
                cv2.imwrite(str(crop_img_path), crop_img)
                
                # Save labels (all visible bees in this crop)
                label_path = output_label_dir / label_filename
                with open(label_path, 'w') as f:
                    for yolo_label in crop_labels:
                        f.write(' '.join(map(str, yolo_label)) + '\n')
                
                crop_count += 1
        
        return crop_count
    
    def _create_dataset_yaml(self, stage2_dir):
        """Create YAML configuration for Stage 2 dataset"""
        dataset_yaml = {
            'path': str(stage2_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'bee'},
            'nc': 1
        }
        
        yaml_path = stage2_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        return yaml_path
    
    def _backup_current_model(self):
        """Backup the currently used fine model"""
        current_model = self.config.get('current_model_path')
        if current_model and Path(current_model).exists():
            backup_path = Path(current_model).parent / 'backup_best.pt'
            shutil.copy2(current_model, backup_path)
            self.stage_update.emit(f"Current model backed up to: {backup_path.name}")
    
    def _train_model(self, dataset_yaml):
        """Train Stage 2 YOLO model"""
        # Initialize model
        model = YOLO('yolov8s-seg.pt')
        
        # Training parameters
        output_dir = self.project_path / 'models' / 'yolo_runs'
        training_params = {
            'data': str(dataset_yaml),
            'epochs': self.config.get('epochs', 50),
            'imgsz': self.config.get('imgsz', 640),
            'batch': self.config.get('batch', 8),
            'name': self.config.get('name', 'bee_segmentation_stage2'),
            'project': str(output_dir.absolute()),
            'patience': self.config.get('patience', 10),
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
        }
        
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
                    
                    # Box metrics (bounding boxes)
                    if hasattr(val_metrics, 'box') and val_metrics.box is not None:
                        box_metrics = val_metrics.box
                        if hasattr(box_metrics, 'map50'):
                            self.current_metrics['mAP50'] = float(box_metrics.map50)
                        if hasattr(box_metrics, 'map'):
                            self.current_metrics['mAP50-95'] = float(box_metrics.map)
                    
                    # Segmentation mask metrics
                    if hasattr(val_metrics, 'seg') and val_metrics.seg is not None:
                        seg_metrics = val_metrics.seg
                        if hasattr(seg_metrics, 'map50'):
                            self.current_metrics['mAP50_mask'] = float(seg_metrics.map50)
                        if hasattr(seg_metrics, 'map'):
                            self.current_metrics['mAP50-95_mask'] = float(seg_metrics.map)
                
                # Get total epochs
                total_epochs = validator.args.epochs if hasattr(validator, 'args') else 50
                
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
        best_model_path = output_dir / self.config.get('name', 'bee_segmentation_stage2') / 'weights' / 'best.pt'
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
                metrics = {
                    'mAP50': last_row.get('metrics/mAP50(B)', 0.0),
                    'mAP50-95': last_row.get('metrics/mAP50-95(B)', 0.0),
                    'precision': last_row.get('metrics/precision(B)', 0.0),
                    'recall': last_row.get('metrics/recall(B)', 0.0),
                }
                return metrics
        except (ImportError, Exception) as e:
            print(f"Could not parse metrics: {e}")
        
        return {}
