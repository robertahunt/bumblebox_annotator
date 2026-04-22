"""
YOLO bbox training worker for GUI integration
"""

import json
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import albumentations as A
    from ultralytics.data.augment import Albumentations
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class YOLOTrainingWorkerBBox(QThread):
    """Background worker for YOLO bbox detection model training"""
    
    # Signals
    progress_update = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    training_complete = pyqtSignal(str, dict)  # model_path, final_metrics
    training_failed = pyqtSignal(str)  # error_message
    stage_update = pyqtSignal(str)  # stage description
    
    def __init__(self, project_path, training_config):
        """
        Initialize YOLO bbox training worker
        
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
            self.stage_update.emit("Preparing training data...")
            yolo_dir, dataset_yaml = self._prepare_yolo_dataset()
            
            if self.should_stop:
                return
                
            # Step 2: Backup current model
            self.stage_update.emit("Backing up current model...")
            self._backup_current_model()
            
            if self.should_stop:
                return
                
            # Step 3: Train model  
            self.stage_update.emit("Training model...")
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
        """Convert COCO format to YOLO format (bbox only)"""
        # Look for per-video JSON files in train and val folders
        train_dir = self.project_path / 'annotations/coco/train'
        val_dir = self.project_path / 'annotations/coco/val'
        
        train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
        val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
        
        if not train_jsons and not val_jsons:
            raise FileNotFoundError("COCO format annotations not found. Export annotations first.")
        
        # Create YOLO format directory
        yolo_dir = self.project_path / 'yolo_bbox_format'
        yolo_dir.mkdir(exist_ok=True)
        
        # Process train and val splits
        for split_name, json_files in [('train', train_jsons), ('val', val_jsons)]:
            if not json_files:
                continue
                
            split_dir = yolo_dir / split_name
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each video's annotations
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Get video name to make filenames unique
                video_name = coco_data.get('info', {}).get('video_name', json_file.stem)
                
                # Create image id to file mapping
                image_map = {img['id']: img for img in coco_data['images']}
                
                # Create lookup for annotations by image_id
                annotations_by_image = {}
                for ann in coco_data['annotations']:
                    image_id = ann['image_id']
                    if image_id not in annotations_by_image:
                        annotations_by_image[image_id] = []
                    annotations_by_image[image_id].append(ann)
                
                # Convert each image and its annotations
                for image_id, image_info in image_map.items():
                    # Copy image with unique name (include video name to avoid collisions)
                    src_image_path = Path(image_info['file_name'])
                    if not src_image_path.is_absolute():
                        src_image_path = self.project_path / src_image_path
                    
                    if not src_image_path.exists():
                        print(f"Warning: Image not found: {src_image_path}")
                        continue
                    
                    # Create unique filename: videoname_framename.jpg
                    unique_filename = f"{video_name}_{src_image_path.name}"
                    dst_image_path = images_dir / unique_filename
                    label_file = labels_dir / f"{Path(unique_filename).stem}.txt"
                    
                    # Skip if both image and label already exist (avoid partial updates)
                    if dst_image_path.exists() and label_file.exists():
                        continue
                    
                    # Copy image file
                    if not dst_image_path.exists():
                        shutil.copy(src_image_path, dst_image_path)
                    
                    # Convert annotations to YOLO format (bbox only)
                    img_width = image_info['width']
                    img_height = image_info['height']
                    
                    with open(label_file, 'w') as f:
                        if image_id in annotations_by_image:
                            for ann in annotations_by_image[image_id]:
                                # Skip chamber (category_id=3) and hive (category_id=2) annotations
                                category_id = ann.get('category_id', 1)
                                if category_id in [2, 3]:  # hive=2, chamber=3
                                    continue
                                
                                # Get bbox in COCO format [x, y, width, height]
                                if 'bbox' in ann:
                                    x, y, w, h = ann['bbox']
                                else:
                                    # Skip if no bbox
                                    continue
                                
                                # Convert to YOLO format (normalized center x, center y, width, height)
                                x_center = (x + w / 2) / img_width
                                y_center = (y + h / 2) / img_height
                                norm_width = w / img_width
                                norm_height = h / img_height
                                
                                # Only bee annotations (category_id=1) are included
                                # YOLO uses 0-based indexing, so bee=0
                                class_id = 0
                                
                                # Write YOLO annotation (class_id x_center y_center width height)
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        # Create dataset YAML - only bee class (chamber/hive excluded)
        dataset_yaml = yolo_dir / 'dataset.yaml'
        yaml_content = {
            'path': str(yolo_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'bee'},
            'nc': 1
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Dataset prepared at {yolo_dir}")
        return yolo_dir, dataset_yaml
    
    def _backup_current_model(self):
        """Backup current bbox model if it exists"""
        models_dir = self.project_path / 'models'
        current_model = models_dir / 'yolo_bbox_best.pt'
        
        if current_model.exists():
            backup_path = models_dir / 'yolo_bbox_best_backup.pt'
            shutil.copy(current_model, backup_path)
            print(f"Backed up current model to {backup_path}")
    
    def _train_model(self, dataset_yaml):
        """Train YOLO bbox detection model"""
        # Get base model from config, default to yolov8s.pt (detection)
        base_model = self.config.get('base_model', 'yolov8s.pt')
        
        # Initialize model
        model = YOLO(base_model)
        
        # Training parameters
        output_dir = self.project_path / 'models' / 'yolo_bbox_runs'
        training_params = {
            'data': str(dataset_yaml),
            'epochs': self.config.get('epochs', 1000),
            'imgsz': self.config.get('imgsz', 1024),
            'batch': self.config.get('batch', 8),
            'name': self.config.get('name', 'bee_bbox_detection'),
            'project': str(output_dir.absolute()),
            'patience': self.config.get('patience', 100),
            'save': True,
            'save_period': -1,
            'val': True,
            'plots': True,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'optimizer': 'AdamW',
            'lr0': self.config.get('lr0', 0.001),
            'lrf': 0.01,
            'warmup_epochs': 3.0,
            'verbose': True,
            
            # Enhanced augmentation parameters
            'hsv_v': 0.2,
            'degrees': 20.0,
            'translate': 0.2,
            'scale': 0.4,
            'shear': 5.0,
            'perspective': 0.0005,
            'flipud': 0.5,
            'fliplr': 0.5,
            'mosaic': 0.1,
            'mixup': 0.05,
            'cutmix': 0.05,
            'copy_paste': 0.05,
            'copy_paste_mode': 'mixup',
            'erasing': 0.05,
            
            # Multi-scale training
            'rect': False,
            'close_mosaic': 10,
            'visualize': True,
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
                
                # Get validation metrics (detection only - no mask metrics)
                if hasattr(validator, 'metrics') and validator.metrics is not None:
                    val_metrics = validator.metrics
                    
                    # For detection models - box metrics
                    if hasattr(val_metrics, 'box') and val_metrics.box is not None:
                        box_metrics = val_metrics.box
                        if hasattr(box_metrics, 'map50'):
                            self.current_metrics['mAP50'] = float(box_metrics.map50)
                        if hasattr(box_metrics, 'map'):
                            self.current_metrics['mAP50-95'] = float(box_metrics.map)
                        if hasattr(box_metrics, 'map75'):
                            self.current_metrics['mAP75'] = float(box_metrics.map75)
                
                # Get total epochs
                total_epochs = validator.args.epochs if hasattr(validator, 'args') else 50
                
                # Emit progress update
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
        print(f"Starting training with params: {training_params}")
        results = model.train(**training_params)
        
        # Find best model - get actual save directory from results
        # YOLO appends numbers to avoid overwriting, so use the actual save_dir
        run_dir = Path(results.save_dir)
        best_model = run_dir / 'weights' / 'best.pt'
        print(f"Looking for best model at: {best_model}")
        
        # Copy to standard location
        models_dir = self.project_path / 'models'
        models_dir.mkdir(exist_ok=True)
        final_path = models_dir / 'yolo_bbox_best.pt'
        
        if best_model.exists():
            shutil.copy(best_model, final_path)
            print(f"Model saved to {final_path}")
            return final_path
        else:
            raise FileNotFoundError(f"Best model not found at {best_model}")
    
    def _get_final_metrics(self, model_path):
        """Extract final metrics from trained model"""
        try:
            # Load the model
            model = YOLO(str(model_path))
            
            # Get validation results
            if hasattr(model, 'metrics') and model.metrics is not None:
                metrics = {}
                val_metrics = model.metrics
                
                if hasattr(val_metrics, 'box') and val_metrics.box is not None:
                    box_metrics = val_metrics.box
                    if hasattr(box_metrics, 'map50'):
                        metrics['mAP50'] = float(box_metrics.map50)
                    if hasattr(box_metrics, 'map'):
                        metrics['mAP50-95'] = float(box_metrics.map)
                    if hasattr(box_metrics, 'map75'):
                        metrics['mAP75'] = float(box_metrics.map75)
                
                return metrics
            
            return {}
            
        except Exception as e:
            print(f"Could not extract final metrics: {e}")
            return {}
