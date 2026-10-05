"""
YOLO training worker for GUI integration
"""

from core.categories import training_categories
import numpy as np
import json
import shutil
import yaml
from pathlib import Path
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal
from training.gpu import require_cuda_device
from core.coco_masks import decode_segmentation

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class YOLOTrainingWorker(QThread):
    """Background worker for YOLO model training"""
    
    # Signals
    progress_update = pyqtSignal(int, int, dict)  # epoch, total_epochs, metrics
    training_complete = pyqtSignal(str, dict)  # model_path, final_metrics
    training_failed = pyqtSignal(str)  # error_message
    stage_update = pyqtSignal(str)  # stage description
    
    def __init__(self, project_path, training_config):
        """
        Initialize YOLO training worker
        
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
        """Convert COCO format to YOLO format"""
        if self.config.get('model_type') == 'brood' and not self.config.get('brood_reviewed', False):
            raise ValueError('Confirm that all visible brood stages are labeled in the selected training and validation frames')
        # Look for per-video JSON files in train and val folders
        train_dir = self.project_path / 'annotations/coco/train'
        val_dir = self.project_path / 'annotations/coco/val'
        
        train_jsons = list(train_dir.glob('*.json')) if train_dir.exists() else []
        val_jsons = list(val_dir.glob('*.json')) if val_dir.exists() else []
        
        if not train_jsons and not val_jsons:
            raise FileNotFoundError("COCO format annotations not found. Export annotations first.")
        
        # Create YOLO format directory
        yolo_dir = self.project_path / 'yolo_format'
        yolo_dir.mkdir(exist_ok=True)

        # Clean previous outputs to avoid mixing stale frames with new splits
        images_root = yolo_dir / 'images'
        labels_root = yolo_dir / 'labels'
        if images_root.exists():
            shutil.rmtree(images_root)
        if labels_root.exists():
            shutil.rmtree(labels_root)
        
        # Convert train and val sets (processing multiple JSON files)
        model_type = self.config.get('model_type', 'bee')
        train_count = sum(self._coco_to_yolo(json_file, yolo_dir, 'train', model_type) for json_file in train_jsons)
        val_count = sum(self._coco_to_yolo(json_file, yolo_dir, 'val', model_type) for json_file in val_jsons)

        if train_count == 0:
            raise ValueError(
                f"No training images with {model_type} segmentation annotations were found. "
                "Make sure this annotation type exists in the selected training videos, "
                "then re-export COCO annotations before training."
            )
        if val_count == 0:
            raise ValueError(
                f"No validation images with {model_type} segmentation annotations were found. "
                "Make sure this annotation type exists in the selected validation videos, "
                "then re-export COCO annotations before training."
            )
        
        # Create dataset YAML - single class model for the selected type
        dataset_yaml = yolo_dir / 'dataset.yaml'
        yaml_content = {
            'path': str(yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': dict(enumerate(training_categories(model_type))),
            'nc': len(training_categories(model_type)),
            'mask_format': 'bumblebox-coco-rle-v1',
        }
        
        with open(dataset_yaml, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        self.stage_update.emit(f"Dataset prepared for {model_type.upper()} model: {train_count} train, {val_count} val")
        
        return yolo_dir, dataset_yaml
        
    def _coco_to_yolo(self, coco_json_path, output_dir, split='train', model_type='bee'):
        """Convert COCO masks to lossless labels for the raster YOLO trainer.
        
        Args:
            coco_json_path: Path to COCO JSON file
            output_dir: Output directory for YOLO format
            split: 'train' or 'val'
            model_type: 'bee', 'chamber', 'hive', or 'pollen' - filters annotations to this type
        """
        from training.raster_masks import mask_record, write_mask_labels

        with open(coco_json_path, 'r') as f:
            coco = json.load(f)
        
        # Map model type to COCO category ID. Prefer the exported COCO
        # categories so this stays correct if class order ever changes.
        category_map = {
            cat.get('name'): cat.get('id')
            for cat in coco.get('categories', [])
            if cat.get('name') and cat.get('id') is not None
        }
        if not category_map:
            category_map = {
                'bee': 1,
                'hive': 2,
                'chamber': 3,
                'pollen': 4,
            }
        selected_categories = training_categories(model_type)
        missing = [name for name in selected_categories if name not in category_map]
        if missing:
            raise ValueError(
                f"COCO file {Path(coco_json_path).name} does not define category "
                f"{', '.join(missing)!r}. Re-export COCO annotations with Export COCO checked.")
        target_classes = {category_map[name]: index
                          for index, name in enumerate(selected_categories)}

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
            label_file = labels_dir / f"{video_name}_{original_name}.json"
            annotations = img_annotations.get(img_id, [])
            
            # Collect valid segmentation annotation lines before writing
            # Only process annotations that have actual segmentation masks (not bbox-only)
            instances = []
            brood_stage_pixels = np.zeros((img_height, img_width), np.uint8) if model_type == 'brood' else None
            
            for ann in annotations:
                # Filter by category - only include annotations matching the model type
                if ann.get('category_id') not in target_classes:
                    continue
                
                # Skip bbox-only annotations as they don't have real masks.
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                mask = decode_segmentation(ann['segmentation'], img_height, img_width)
                class_id = target_classes[ann['category_id']]
                if brood_stage_pixels is not None:
                    occupied = mask > 0
                    if np.any(occupied & (brood_stage_pixels != 0) & (brood_stage_pixels != class_id + 1)):
                        raise ValueError(
                            f'Conflicting brood stages overlap in {src_img_path.name}. '
                            'Label only the visible top surface with one brood appearance stage; hive overlap is allowed.')
                    brood_stage_pixels[occupied] = class_id + 1
                instance = mask_record(mask, class_id=class_id)
                if instance is not None:
                    instances.append(instance)
            
            # Only write label file and include image if there are valid segmentation annotations
            if instances:
                write_mask_labels(label_file, (img_height, img_width), instances)
                converted_count += 1
            else:
                # Remove copied image if no segmentation annotations
                # (image may have only bbox-only annotations or no annotations at all)
                if dst_img_path.exists():
                    dst_img_path.unlink()
            
        return converted_count
        
    def _backup_current_model(self):
        """Backup the currently used model"""
        current_model = self.config.get('current_model_path')
        if current_model and Path(current_model).exists():
            backup_path = Path(current_model).parent / 'backup_best.pt'
            shutil.copy2(current_model, backup_path)
            self.stage_update.emit(f"Current model backed up to: {backup_path.name}")
        
    def _train_model(self, dataset_yaml):
        """Train YOLO model"""
        device, device_label = require_cuda_device()
        self.stage_update.emit(f"Using GPU: {device_label}")
        print(f"YOLO training using GPU: {device_label}")

        # Initialize model
        model = YOLO('yolov8s-seg.pt')
        
        # Training parameters
        output_dir = self.project_path / 'models' / 'yolo_runs'
        training_params = {
            'data': str(dataset_yaml),
            'epochs': self.config.get('epochs', 50),
            'imgsz': self.config.get('imgsz', 640),
            'batch': self.config.get('batch', 8),
            'name': self.config.get('name', 'bee_segmentation'),
            'project': str(output_dir.absolute()),
            'patience': self.config.get('patience', 10),
            'save': True,
            'save_period': -1,
            'val': True,  # Enable validation during training
            'plots': True,  # Save validation prediction plots
            'device': device,
            'optimizer': 'AdamW',
            'lr0': self.config.get('lr0', 0.001),
            'lrf': 0.01,
            'warmup_epochs': 3.0,
            # Keep segmentation training memory sane. overlap_mask=True is the
            # Ultralytics default and is much lighter than separate full-size
            # binary masks for every bee in high-resolution frames.
            'overlap_mask': True,
            'amp': True,
            'cache': False,
            'verbose': True,
            'workers': self.config.get('workers', 0),  # Stable in GUI: avoid orphaned dataloader processes
            #'freeze':10,
            
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
            'visualize': False,
            'augment': True,  # Image-only blur in the raster dataset
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
        from training.raster_masks import RasterSegmentationTrainer, raster_training_options
        training_params.update(raster_training_options())
        self.stage_update.emit(
            "Using exact instance masks (holes and disconnected regions preserved). "
            "Paired affine/flips and color augmentation enabled; mosaic, mixup, "
            "copy-paste, cutmix and perspective disabled."
        )
        results = model.train(trainer=RasterSegmentationTrainer, **training_params)
        
        # Return best model path
        best_model_path = output_dir / self.config.get('name', 'bee_segmentation') / 'weights' / 'best.pt'
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
