"""
SAM2 Fine-tuning Training Worker
Based on: https://towardsdatascience.com/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3/
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Install with: pip install albumentations")


class SAM2TrainingWorker(QThread):
    """Background worker for SAM2 fine-tuning"""
    
    # Signals
    progress_update = pyqtSignal(int, int, dict)  # step, total_steps, metrics
    training_complete = pyqtSignal(str, dict)  # model_path, final_metrics
    training_failed = pyqtSignal(str)  # error_message
    stage_update = pyqtSignal(str)  # stage description
    
    def __init__(self, project_path, training_config):
        """
        Initialize SAM2 training worker
        
        Args:
            project_path: Path to project directory
            training_config: Dict with training parameters including:
                - checkpoint_path: Path to pre-trained SAM2 checkpoint
                - config_name: SAM2 config (e.g., 'sam2_hiera_s.yaml')
                - num_steps: Number of training steps (default: 25000)
                - learning_rate: Learning rate (default: 1e-5)
                - weight_decay: Weight decay (default: 4e-5)
                - save_interval: Steps between model saves (default: 1000)
                - train_image_encoder: Whether to train image encoder (default: False)
        """
        super().__init__()
        self.project_path = Path(project_path)
        self.config = training_config
        self.should_stop = False
        self.data = []  # List of training data entries
        self.val_data = []  # List of validation data entries
        
        # Setup data augmentation pipeline
        self._setup_augmentation()
        
    def stop(self):
        """Request training to stop"""
        self.should_stop = True
    
    def _setup_augmentation(self):
        """Setup data augmentation pipeline using albumentations"""
        if not ALBUMENTATIONS_AVAILABLE:
            self.transform = None
            print("Data augmentation disabled (albumentations not installed)")
            return
        
        # Get augmentation settings from config (with defaults)
        use_augmentation = self.config.get('use_augmentation', True)
        
        if not use_augmentation:
            self.transform = None
            print("Data augmentation disabled by config")
            return
        
        # Create augmentation pipeline matching YOLO instance-focused parameters
        # Note: albumentations automatically handles mask and bounding box transformations
        self.transform = A.Compose([
            # Geometric transformations (matching YOLO: flipud=0.5, fliplr=0.5)
            A.HorizontalFlip(p=0.5),  # fliplr
            A.VerticalFlip(p=0.5),    # flipud
            
            # Affine transformations (matching YOLO: degrees=20, translate=0.1, scale=0.5, shear=5.0)
            A.ShiftScaleRotate(
                shift_limit=0.1,     # translate: 10% shift (matching YOLO)
                scale_limit=0.5,     # scale: 50% scale range (matching YOLO)
                rotate_limit=20,     # degrees: ±20 degrees rotation (matching YOLO)
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.7
            ),
            
            # Shear transform (matching YOLO: shear=5.0)
            A.Affine(
                shear={'x': (-5, 5), 'y': (-5, 5)},  # ±5 degrees shear
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0,
                p=0.5
            ),
            
            # HSV color augmentations (matching YOLO: hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
            A.HueSaturationValue(
                hue_shift_limit=int(0.015 * 179),    # hsv_h: 0.015 * 179 = ~2.7 degrees
                sat_shift_limit=int(0.7 * 255),      # hsv_s: 0.7 * 255 = ~178
                val_shift_limit=int(0.4 * 255),      # hsv_v: 0.4 * 255 = ~102
                p=0.7
            ),
            
        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # [x_min, y_min, x_max, y_max]
            label_fields=['bbox_labels'],
            min_visibility=0.3  # Keep bbox if at least 30% visible after transform
        ))
        
        print("Data augmentation enabled (matching YOLO instance-focused parameters)")
        
    def run(self):
        """Run training workflow"""
        try:
            if not SAM2_AVAILABLE:
                self.training_failed.emit("SAM2 not installed. Install from: https://github.com/facebookresearch/segment-anything-2")
                return
            
            # Step 1: Load and prepare data
            self.stage_update.emit("Loading training data...")
            self._load_training_data()
            
            if self.should_stop or not self.data:
                self.training_failed.emit("No training data found. Please annotate some frames first.")
                return
            
            # Load validation data
            self.stage_update.emit("Loading validation data...")
            self._load_validation_data()
            print(f"Loaded {len(self.val_data)} validation samples")
            
            # Step 2: Load SAM2 model
            self.stage_update.emit("Loading SAM2 model...")
            predictor = self._load_sam2_model()
            
            if self.should_stop:
                return
            
            # Step 3: Setup training
            self.stage_update.emit("Setting up training parameters...")
            optimizer, scaler, scheduler = self._setup_training(predictor)
            
            if self.should_stop:
                return
            
            # Step 4: Training loop
            self.stage_update.emit("Starting training...")
            model_path = self._train_model(predictor, optimizer, scaler, scheduler)
            
            if self.should_stop:
                return
            
            # Step 5: Complete
            final_metrics = {"training_steps": self.config.get('num_steps', 25000)}
            self.training_complete.emit(str(model_path), final_metrics)
            
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
            self.training_failed.emit(error_msg)
    
    def _load_training_data(self):
        """Load training data from COCO annotations"""
        # Look for COCO annotations in train folder
        train_dir = self.project_path / 'annotations/coco/train'
        
        if not train_dir.exists():
            raise FileNotFoundError("Training annotations not found. Export COCO annotations first.")
        
        train_jsons = list(train_dir.glob('*.json'))
        if not train_jsons:
            raise FileNotFoundError("No COCO JSON files found in training directory.")
        
        # Load all training data
        for json_path in train_jsons:
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            
            # Create image lookup
            images_by_id = {img['id']: img for img in coco_data['images']}
            
            # Process annotations
            for ann in coco_data['annotations']:
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                image_info = images_by_id.get(ann['image_id'])
                if not image_info:
                    continue
                
                # Get image path (handle both absolute and relative paths)
                image_path = Path(image_info['file_name'])
                if not image_path.is_absolute():
                    image_path = self.project_path / image_info['file_name']
                
                if not image_path.exists():
                    continue
                
                # Store data entry
                self.data.append({
                    'image_path': str(image_path),
                    'annotation': ann
                })
        
        print(f"Loaded {len(self.data)} training samples from {len(train_jsons)} video(s)")
    
    def _load_validation_data(self):
        """Load validation data from COCO annotations"""
        # Look for COCO annotations in val folder
        val_dir = self.project_path / 'annotations/coco/val'
        
        if not val_dir.exists():
            print("Warning: No validation directory found. Skipping validation.")
            return
        
        val_jsons = list(val_dir.glob('*.json'))
        if not val_jsons:
            print("Warning: No COCO JSON files found in validation directory.")
            return
        
        # Load all validation data
        for json_path in val_jsons:
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            
            # Create image lookup
            images_by_id = {img['id']: img for img in coco_data['images']}
            
            # Process annotations
            for ann in coco_data['annotations']:
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                image_info = images_by_id.get(ann['image_id'])
                if not image_info:
                    continue
                
                # Get image path (handle both absolute and relative paths)
                image_path = Path(image_info['file_name'])
                if not image_path.is_absolute():
                    image_path = self.project_path / image_info['file_name']
                
                if not image_path.exists():
                    continue
                
                # Store data entry
                self.val_data.append({
                    'image_path': str(image_path),
                    'annotation': ann
                })
        
        print(f"Loaded {len(self.val_data)} validation samples from {len(val_jsons)} video(s)")
    
    def _load_sam2_model(self):
        """Load SAM2 model"""
        checkpoint_path = self.config['checkpoint_path']
        config_name = self.config.get('config_name', 'sam2_hiera_l.yaml')
        
        # Build SAM2 model
        sam2_model = build_sam2(config_name, checkpoint_path, device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)
        
        return predictor
    
    def _setup_training(self, predictor):
        """Setup training parameters"""
        # Enable training for mask decoder and prompt encoder (as per tutorial)
        predictor.model.sam_mask_decoder.train(True)
        predictor.model.sam_prompt_encoder.train(True)
        
        # Setup optimizer with different learning rates for different components
        learning_rate = self.config.get('learning_rate', 1e-4)  # Balanced LR for fine-tuning
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Group parameters for different learning rates
        param_groups = [
            {
                'params': predictor.model.sam_mask_decoder.parameters(),
                'lr': learning_rate,
                'name': 'mask_decoder'
            },
            {
                'params': predictor.model.sam_prompt_encoder.parameters(),
                'lr': learning_rate,
                'name': 'prompt_encoder'
            }
        ]
        
        # Optionally train image encoder with lower learning rate
        if self.config.get('train_image_encoder', False):
            predictor.model.image_encoder.train(True)
            param_groups.append({
                'params': predictor.model.image_encoder.parameters(),
                'lr': learning_rate * 0.1,  # 10x lower LR for image encoder
                'name': 'image_encoder'
            })
            print(f"Training image encoder with LR: {learning_rate * 0.1:.2e}")
        else:
            predictor.model.image_encoder.train(False)
        
        print(f"Training with LR: {learning_rate:.2e}, Weight decay: {weight_decay:.2e}")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            params=param_groups,
            weight_decay=weight_decay
        )
        
        # Setup mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()
        
        # Setup learning rate scheduler with warmup and cosine annealing
        num_steps = self.config.get('num_steps', 25000)
        warmup_steps = self.config.get('warmup_steps', 500)  # Shorter warmup for faster convergence
        min_lr_ratio = self.config.get('min_lr_ratio', 0.1)  # Don't decay below 10% of initial LR
        
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup from 0.1 to 1.0 (not from 0)
                return 0.1 + 0.9 * (step / warmup_steps)
            else:
                # Cosine annealing with minimum LR
                progress = (step - warmup_steps) / (num_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return optimizer, scaler, scheduler
    
    def _generate_random_crop_around_annotation(self, img, ann, crop_size):
        """Generate random crop that contains the annotation
        
        Args:
            img: Full image (numpy array, RGB)
            ann: COCO annotation dict
            crop_size: Size of crop (e.g., 640 or 1024)
            
        Returns:
            crop_img: Cropped image
            crop_mask: Binary mask in crop coordinates
            crop_point: Random point inside mask in crop coordinates
            crop_label: Point label (1 for positive)
            Or (None, None, None, None) if crop generation fails
        """
        img_h, img_w = img.shape[:2]
        
        # Get bounding box of annotation
        if 'bbox' in ann and ann['bbox']:
            x, y, w, h = ann['bbox']
            center_x = x + w / 2
            center_y = y + h / 2
        else:
            # If no bbox, try to get from segmentation
            if isinstance(ann['segmentation'], list) and ann['segmentation']:
                seg = ann['segmentation'][0]  # Use first polygon
                if len(seg) >= 6:
                    xs = seg[0::2]
                    ys = seg[1::2]
                    center_x = np.mean(xs)
                    center_y = np.mean(ys)
                else:
                    # Invalid segmentation, skip
                    return None, None, None, None
            else:
                # No valid way to get center, skip
                return None, None, None, None
        
        # Add some randomness to crop position (±20% of crop size)
        offset_x = np.random.uniform(-0.2 * crop_size, 0.2 * crop_size)
        offset_y = np.random.uniform(-0.2 * crop_size, 0.2 * crop_size)
        
        # Calculate crop position centered on annotation (with offset)
        crop_x = int(center_x + offset_x - crop_size / 2)
        crop_y = int(center_y + offset_y - crop_size / 2)
        
        # Clip to image bounds
        crop_x = max(0, min(crop_x, img_w - crop_size))
        crop_y = max(0, min(crop_y, img_h - crop_size))
        
        # Handle edge cases where image is smaller than crop
        actual_crop_w = min(crop_size, img_w - crop_x)
        actual_crop_h = min(crop_size, img_h - crop_y)
        
        # Extract crop
        crop_img = img[crop_y:crop_y+actual_crop_h, crop_x:crop_x+actual_crop_w].copy()
        
        # Pad if necessary to reach crop_size
        if actual_crop_w < crop_size or actual_crop_h < crop_size:
            padded = np.zeros((crop_size, crop_size, 3), dtype=crop_img.dtype)
            padded[:actual_crop_h, :actual_crop_w] = crop_img
            crop_img = padded
        
        # Convert segmentation to mask in crop coordinates
        crop_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        
        if isinstance(ann['segmentation'], list):
            # Polygon format
            for seg in ann['segmentation']:
                if len(seg) >= 6:  # Valid polygon
                    poly = np.array(seg).reshape(-1, 2)
                    # Transform to crop coordinates
                    poly[:, 0] -= crop_x
                    poly[:, 1] -= crop_y
                    # Clip to crop bounds
                    poly[:, 0] = np.clip(poly[:, 0], 0, crop_size - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, crop_size - 1)
                    poly = poly.astype(np.int32)
                    cv2.fillPoly(crop_mask, [poly], 1)
        else:
            # RLE format (not commonly used in this app, but handle it)
            from pycocotools import mask as mask_utils
            rle = ann['segmentation']
            full_mask = mask_utils.decode(rle)
            # Extract crop region
            crop_mask_region = full_mask[crop_y:crop_y+actual_crop_h, crop_x:crop_x+actual_crop_w]
            crop_mask[:actual_crop_h, :actual_crop_w] = crop_mask_region
        
        # Get bounding box of mask in crop coordinates
        coords = np.argwhere(crop_mask > 0)
        if len(coords) == 0:
            # No valid mask in crop, return None to skip
            return None, None, None, None
        
        # Get tight bounding box from mask
        y_coords, x_coords = coords[:, 0], coords[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Add random noise to bounding box (±5-15% of box dimensions)
        width = x_max - x_min
        height = y_max - y_min
        
        # Random noise scale (5-15% of dimension)
        noise_scale_x = np.random.uniform(0.05, 0.15)
        noise_scale_y = np.random.uniform(0.05, 0.15)
        
        # Apply noise (can expand or shrink the box)
        noise_x_min = np.random.uniform(-noise_scale_x * width, noise_scale_x * width)
        noise_y_min = np.random.uniform(-noise_scale_y * height, noise_scale_y * height)
        noise_x_max = np.random.uniform(-noise_scale_x * width, noise_scale_x * width)
        noise_y_max = np.random.uniform(-noise_scale_y * height, noise_scale_y * height)
        
        # Apply noise and clip to crop bounds
        x_min_noisy = np.clip(x_min + noise_x_min, 0, crop_size - 1)
        y_min_noisy = np.clip(y_min + noise_y_min, 0, crop_size - 1)
        x_max_noisy = np.clip(x_max + noise_x_max, 0, crop_size - 1)
        y_max_noisy = np.clip(y_max + noise_y_max, 0, crop_size - 1)
        
        # Ensure valid box (min < max)
        if x_min_noisy >= x_max_noisy:
            x_min_noisy, x_max_noisy = x_max_noisy - 1, x_min_noisy + 1
        if y_min_noisy >= y_max_noisy:
            y_min_noisy, y_max_noisy = y_max_noisy - 1, y_min_noisy + 1
        
        # Format as [1, 4] array: [x1, y1, x2, y2]
        crop_box = np.array([[x_min_noisy, y_min_noisy, x_max_noisy, y_max_noisy]], dtype=np.float32)
        
        return crop_img, crop_mask, crop_box, None
    
    def _read_batch(self):
        """Read random training batch with random crops around annotations"""
        # Select random entry
        entry = self.data[np.random.randint(len(self.data))]
        
        # Read full image
        img = cv2.imread(entry['image_path'])[..., ::-1]  # BGR to RGB
        
        # Get crop size from config (default 1024 to match SAM2 max size)
        crop_size = self.config.get('crop_size', 1024)
        
        # Get annotation
        ann = entry['annotation']
        
        # Generate random crop around annotation
        crop_img, crop_mask, crop_box, _ = self._generate_random_crop_around_annotation(
            img, ann, crop_size
        )
        
        # Skip if crop generation failed
        if crop_img is None:
            return None, None, None, None
        
        # Apply data augmentation if available
        if self.transform is not None:
            try:
                # Prepare bbox in pascal_voc format: [x_min, y_min, x_max, y_max]
                bbox = crop_box[0].tolist()  # Remove batch dimension for albumentations
                
                # Apply augmentation
                transformed = self.transform(
                    image=crop_img,
                    mask=crop_mask,
                    bboxes=[bbox],
                    bbox_labels=[0]  # Dummy label (required by albumentations)
                )
                
                crop_img = transformed['image']
                crop_mask = transformed['mask']
                
                # Get transformed bbox
                if len(transformed['bboxes']) > 0:
                    # Bbox survived the transformation
                    bbox_transformed = transformed['bboxes'][0]
                    crop_box = np.array([bbox_transformed], dtype=np.float32)
                else:
                    # Bbox was lost during transformation (e.g., rotated out of view)
                    # Skip this sample
                    return None, None, None, None
                    
            except Exception as e:
                # If augmentation fails, continue with original crop
                print(f"Warning: Augmentation failed: {e}. Using original crop.")
        
        # Resize crop if it exceeds 1024 (SAM2 requirement)
        if crop_size > 1024:
            r = 1024 / crop_size
            crop_img = cv2.resize(crop_img, (1024, 1024))
            crop_mask = cv2.resize(crop_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
            # Scale box coordinates
            crop_box = crop_box * r
        
        return crop_img, crop_mask[np.newaxis, ...], crop_box, None  # Add batch dimension to mask
    
    def _dice_loss(self, pred, target, smooth=1.0):
        """Dice loss for segmentation
        
        Args:
            pred: Predicted mask probabilities [batch, H, W]
            target: Ground truth mask [batch, H, W]
            smooth: Smoothing factor
            
        Returns:
            Dice loss value
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def _focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance
        
        Args:
            pred: Predicted mask probabilities [batch, H, W]
            target: Ground truth mask [batch, H, W]
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            
        Returns:
            Focal loss value
        """
        bce = -target * torch.log(pred + 1e-7) - (1 - target) * torch.log(1 - pred + 1e-7)
        
        # Apply focal weighting
        focal_weight = target * (1 - pred) ** gamma + (1 - target) * pred ** gamma
        focal_loss = alpha * focal_weight * bce
        
        return focal_loss.mean()
    
    def _train_model(self, predictor, optimizer, scaler, scheduler):
        """Main training loop with improved loss and training strategies"""
        num_steps = self.config.get('num_steps', 25000)
        save_interval = self.config.get('save_interval', 1000)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 2)  # Effective batch size = 2
        max_grad_norm = self.config.get('max_grad_norm', 1.0)  # Gradient clipping for stability
        
        # Create output directory
        output_dir = self.project_path / 'models' / 'sam2_finetuned'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create debug directory for visualizations
        debug_dir = self.project_path / 'tmp' / 'sam2_training_debug'
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving training visualizations to: {debug_dir}")
        print(f"Check this folder to see example training crops\n")
        
        # Create validation visualization directory
        val_debug_dir = self.project_path / 'tmp' / 'sam2_validation_debug'
        val_debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving validation visualizations to: {val_debug_dir}\n")
        
        mean_iou = 0.0
        val_loss = 0.0
        val_iou = 0.0
        
        # Track best model (using validation IOU, higher is better)
        best_val_iou = 0.0
        best_val_loss = float('inf')
        best_model_step = 0
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 5000)  # Stop if no improvement for 5000 steps
        
        # Exponential moving average for smoother metrics
        ema_alpha = 0.99  # Smoothing factor
        ema_iou = 0.0
        
        # Run initial validation at step 0 if we have validation data
        if len(self.val_data) > 0:
            print("\n=== Running initial validation (step 0) ===")
            val_metrics = self._validate_model(predictor, val_debug_dir, 0, num_examples=100)
            val_loss = val_metrics['val_loss']
            val_iou = val_metrics['val_iou']
            best_val_iou = val_iou
            best_val_loss = val_loss
            print(f"Initial validation - Loss: {val_loss:.4f}, IOU: {val_iou:.4f}\n")
            print(f"Set initial best: IOU={best_val_iou:.4f}, Loss={best_val_loss:.4f}\n")
        else:
            print("\nWarning: No validation data found. Validation metrics will not be available.")
            print("To enable validation, add videos to the 'val' split and annotate some frames.\n")
        
        print(f"Training configuration:")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(f"  Effective batch size: {gradient_accumulation_steps}")
        print(f"  Warmup steps: {self.config.get('warmup_steps', 500)}")
        print(f"  Max gradient norm: {max_grad_norm}")
        print(f"  Early stopping patience: {patience} steps\n")
        
        # Initialize gradient accumulation
        optimizer.zero_grad()
        accumulated_loss = 0.0
        
        for itr in range(num_steps):
            if self.should_stop:
                break
            
            with torch.cuda.amp.autocast():  # Mixed precision
                # Load data batch
                image, mask, input_box, _ = self._read_batch()
                
                # Skip invalid batches
                if image is None:
                    print(f"Step {itr}: Skipped invalid batch")
                    continue
                
                # Save debug visualization for first 20 steps
                if itr < 20:
                    self._save_training_example(image, mask, input_box, debug_dir, itr)
                
                # Apply SAM image encoder to the image
                predictor.set_image(image)
                
                # Process input box using prompt encoder
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    point_coords=None, point_labels=None, box=input_box, mask_logits=None, normalize_coords=True
                )
                
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels) if unnorm_coords is not None else None,
                    boxes=unnorm_box,
                    masks=None,
                )
                
                # Predict masks using mask decoder
                # Determine batched mode from coords (if points) or box (if box prompt)
                if unnorm_coords is not None:
                    batched_mode = unnorm_coords.shape[0] > 1
                elif unnorm_box is not None:
                    batched_mode = unnorm_box.shape[0] > 1
                else:
                    batched_mode = False
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                
                # Upscale masks to original image resolution
                prd_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[-1]
                )
                
                # Calculate loss
                # Select the best mask based on predicted scores (instead of always using first)
                prd_masks_sigmoid = torch.sigmoid(prd_masks)  # [batch, 3, H, W]
                best_mask_idx = prd_scores.argmax(dim=1)  # [batch]
                prd_mask = prd_masks_sigmoid[0, best_mask_idx[0]]  # Get best mask, shape: [H, W]
                if prd_mask.ndim == 2:
                    prd_mask = prd_mask.unsqueeze(0)  # Add batch dimension: [1, H, W]
                
                # Ensure gt_mask has batch dimension to match prd_mask
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                if gt_mask.ndim == 2:
                    gt_mask = gt_mask.unsqueeze(0)  # Add batch dimension: [1, H, W]
                
                # Debug: Check mask statistics
                if itr < 5:
                    print(f"\nStep {itr} - Mask stats:")
                    print(f"  GT mask shape: {gt_mask.shape}, min: {gt_mask.min():.3f}, max: {gt_mask.max():.3f}, mean: {gt_mask.mean():.3f}")
                    print(f"  GT mask positive pixels: {(gt_mask > 0).sum().item()}")
                    print(f"  Pred mask shape: {prd_mask.shape}, min: {prd_mask.min():.3f}, max: {prd_mask.max():.3f}, mean: {prd_mask.mean():.3f}")
                    print(f"  Pred mask positive pixels (>0.5): {(prd_mask > 0.5).sum().item()}")
                    print(f"  Best mask index: {best_mask_idx[0].item()}")
                
                # Improved loss function: Dice + BCE (simpler, faster convergence)
                dice_loss = self._dice_loss(prd_mask, gt_mask)
                bce_loss = (
                    -gt_mask * torch.log(prd_mask + 1e-7) - 
                    (1 - gt_mask) * torch.log(1 - prd_mask + 1e-7)
                ).mean()
                
                # Combine losses (Dice is better for segmentation, BCE for gradient stability)
                seg_loss = dice_loss * 0.6 + bce_loss * 0.4
                
                # Score loss (IOU-based) - use flattened tensors for robust calculation
                gt_flat = gt_mask.reshape(gt_mask.shape[0], -1)
                pred_flat = (prd_mask > 0.5).reshape(prd_mask.shape[0], -1)
                
                inter = (gt_flat * pred_flat).sum(1)
                union = gt_flat.sum(1) + pred_flat.sum(1) - inter
                iou = inter / (union + 1e-6)  # Add epsilon to avoid division by zero
                score_loss = torch.abs(prd_scores[0, best_mask_idx[0]] - iou).mean()
                
                # Combined loss
                loss = seg_loss + score_loss * 0.05
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Debug: Check loss values
                if itr < 5:
                    print(f"  Dice loss: {dice_loss.item():.6f}, BCE loss: {bce_loss.item():.6f}")
                    print(f"  Seg loss: {seg_loss.item():.6f}, Score loss: {score_loss.item():.6f}, Total loss: {(loss.item() * gradient_accumulation_steps):.6f}")
                    print(f"  IOU: {iou.mean().item():.3f}")
            
            # Backpropagation with gradient accumulation
            scaler.scale(loss).backward()
            accumulated_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights every gradient_accumulation_steps
            if (itr + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()
                
                # Average the accumulated loss
                loss_value = accumulated_loss / gradient_accumulation_steps
                accumulated_loss = 0.0
            else:
                # Use accumulated loss for reporting
                loss_value = loss.item() * gradient_accumulation_steps
            
            # Update metrics with EMA
            current_iou = iou.mean().cpu().detach().numpy()
            if itr == 0:
                ema_iou = current_iou
            else:
                ema_iou = ema_alpha * ema_iou + (1 - ema_alpha) * current_iou
            mean_iou = ema_iou  # Use EMA for reporting
            
            # Run validation every 100 steps (but not at 0 since we already did that)
            if itr > 0 and itr % 1000 == 0 and len(self.val_data) > 0:
                try:
                    print(f"\n{'='*60}")
                    print(f"Running validation at step {itr}")
                    print(f"{'='*60}")
                    val_metrics = self._validate_model(predictor, val_debug_dir, itr, num_examples=100)
                    val_loss = val_metrics['val_loss']
                    val_iou = val_metrics['val_iou']
                    print(f"Validation complete - Loss: {val_loss:.4f}, IOU: {val_iou:.4f}")
                    
                    # Save model if it's the best so far (based on IOU, higher is better)
                    improved = False
                    if val_iou > best_val_iou:
                        best_val_iou = val_iou
                        best_val_loss = val_loss
                        best_model_step = itr
                        improved = True
                        patience_counter = 0  # Reset patience
                        model_path = output_dir / "sam2_finetuned_best.pt"
                        # Save in SAM2 checkpoint format (dict with 'model' key)
                        torch.save({'model': predictor.model.state_dict()}, model_path)
                        print(f"*** NEW BEST MODEL! Saved checkpoint: {model_path}")
                        print(f"*** Best: IOU={best_val_iou:.4f}, Loss={best_val_loss:.4f} at step {best_model_step}")
                    else:
                        patience_counter += 1000  # Increment by validation interval
                        print(f"No improvement (best: IOU={best_val_iou:.4f} at step {best_model_step})")
                        print(f"Patience: {patience_counter}/{patience}")
                        
                        # Check for early stopping
                        if patience_counter >= patience:
                            print(f"\n*** Early stopping triggered! No improvement for {patience} steps ***")
                            print(f"Best model at step {best_model_step}: IOU={best_val_iou:.4f}, Loss={best_val_loss:.4f}")
                            break
                    
                    print(f"{'='*60}\n")
                except Exception as e:
                    print(f"ERROR: Validation failed at step {itr}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Print diagnostic warnings if training is too slow
            if itr == 1000 and mean_iou < 0.3:
                print(f"\n{'!'*60}")
                print(f"WARNING: Training seems slow (IOU={mean_iou:.3f} at step {itr})")
                print(f"Consider:")
                print(f"  - Increasing learning_rate (current: {optimizer.param_groups[0]['lr']:.2e})")
                print(f"  - Reducing gradient_accumulation_steps (current: {gradient_accumulation_steps})")
                print(f"  - Checking if data augmentation is too aggressive")
                print(f"{'!'*60}\n")
            elif itr == 5000 and mean_iou < 0.6:
                print(f"\n{'!'*60}")
                print(f"WARNING: Training slower than expected (IOU={mean_iou:.3f} at step {itr})")
                print(f"Target: IOU > 0.8 by step 5000")
                print(f"Current LR: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"Consider adjusting hyperparameters if performance doesn't improve")
                print(f"{'!'*60}\n")
            
            # Report progress
            if itr % 10 == 0:
                # Add validation status indicator
                val_status = f"(last updated: step {(itr // 100) * 100})" if itr > 0 else "(initial)"
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                
                metrics = {
                    'train_loss': loss_value,  # Training loss (unscaled)
                    'val_loss': val_loss,  # Validation loss (updated every 100 steps)
                    'mAP50': float(mean_iou),  # Training IOU
                    'mAP50-95': val_iou,  # Validation IOU (updated every 100 steps)
                    # Also include original names for logging
                    'loss': loss_value,
                    'seg_loss': seg_loss.item(),
                    'score_loss': score_loss.item(),
                    'mean_iou': float(mean_iou),
                    'learning_rate': current_lr
                }
                self.progress_update.emit(itr, num_steps, metrics)
                print(f"Step {itr}/{num_steps}, LR: {current_lr:.2e}, Train Loss: {loss_value:.4f}, Train IOU: {mean_iou:.4f}, Val Loss: {val_loss:.4f} {val_status}, Val IOU: {val_iou:.4f} {val_status}")
        
        # Training complete
        print(f"\nTraining complete! Best model at step {best_model_step}:")
        print(f"  Best validation IOU: {best_val_iou:.4f}")
        print(f"  Best validation Loss: {best_val_loss:.4f}")
        
        # Return best model path
        best_model_path = output_dir / "sam2_finetuned_best.pt"
        if not best_model_path.exists():
            # Fallback: save current model if no best model was saved (e.g., no validation data)
            # Save in SAM2 checkpoint format (dict with 'model' key)
            torch.save({'model': predictor.model.state_dict()}, best_model_path)
            print(f"Saved final model: {best_model_path}")
        
        return best_model_path
    
    def _save_training_example(self, image, mask, input_box, debug_dir, step):
        """Save visualization of training example
        
        Args:
            image: RGB image (numpy array)
            mask: Ground truth mask (numpy array with batch dimension)
            input_box: Bounding box coordinates (numpy array, shape [1, 4])
            debug_dir: Directory to save visualizations
            step: Training step number
        """
        import matplotlib.pyplot as plt
        
        # Remove batch dimension from mask if present
        if mask.ndim == 3:
            mask_2d = mask[0]
        else:
            mask_2d = mask
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Original image with box
        axes[0].imshow(image)
        if input_box is not None and len(input_box) > 0:
            # Extract box coordinates (format: [batch, 4] = [x1, y1, x2, y2])
            x1, y1, x2, y2 = input_box[0]
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
        axes[0].set_title(f'Step {step}: Input Image + Box')
        axes[0].axis('off')
        
        # Plot 2: Ground truth mask
        axes[1].imshow(mask_2d, cmap='gray')
        axes[1].set_title(f'Ground Truth Mask\n(pixels: {(mask_2d > 0).sum()})')
        axes[1].axis('off')
        
        # Plot 3: Overlay
        axes[2].imshow(image)
        axes[2].imshow(mask_2d, alpha=0.5, cmap='Reds')
        if input_box is not None and len(input_box) > 0:
            x1, y1, x2, y2 = input_box[0]
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
            axes[2].add_patch(rect)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        output_path = debug_dir / f'training_step_{step:04d}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        if step == 0:
            print(f"Saved first training example to: {output_path}")
    
    def _validate_model(self, predictor, val_debug_dir, step, num_examples=100):
        """Run validation on samples and visualize a few predictions
        
        Args:
            predictor: SAM2 predictor
            val_debug_dir: Directory to save validation visualizations
            step: Current training step
            num_examples: Number of validation samples to evaluate (default: 100)
            
        Returns:
            dict: Validation metrics (val_loss, val_iou)
        """
        if len(self.val_data) == 0:
            print("Warning: No validation data available, skipping validation")
            return {'val_loss': 0.0, 'val_iou': 0.0}
        
        import matplotlib.pyplot as plt
        
        print(f"Starting validation with {len(self.val_data)} total validation samples...")
        
        # Sample random validation examples
        num_samples = min(num_examples, len(self.val_data))
        val_indices = np.random.choice(len(self.val_data), num_samples, replace=False)
        print(f"Selected {num_samples} random samples for validation (visualizing first 3)")
        
        total_loss = 0.0
        total_iou = 0.0
        valid_samples = 0
        
        # Visualize first few examples
        examples_to_visualize = []
        
        with torch.no_grad():  # No gradients needed for validation
            for idx in val_indices:
                entry = self.val_data[idx]
                
                try:
                    # Load image
                    image = cv2.imread(entry['image_path'])
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Get annotation
                    ann = entry['annotation']
                    
                    # Generate crop and mask (similar to training)
                    crop_data = self._generate_random_crop_around_annotation(
                        image, ann, self.config.get('crop_size', 1024)
                    )
                    
                    if crop_data is None:
                        continue
                    
                    crop_image, crop_mask, crop_box, _ = crop_data
                    
                    # Use box prompt - already in correct shape [1, 4] from _generate_random_crop_around_annotation
                    input_box = crop_box.astype(np.float32)  # Already shape [1, 4]
                    
                    with torch.cuda.amp.autocast():
                        # Set image and predict
                        predictor.set_image(crop_image)
                        
                        # Process input box
                        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                            point_coords=None, point_labels=None, box=input_box, mask_logits=None, normalize_coords=True
                        )
                        
                        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                            points=(unnorm_coords, labels) if unnorm_coords is not None else None,
                            boxes=unnorm_box,
                            masks=None,
                        )
                        
                        # Predict masks
                        # Determine batched mode from coords (if points) or box (if box prompt)
                        if unnorm_coords is not None:
                            batched_mode = unnorm_coords.shape[0] > 1
                        elif unnorm_box is not None:
                            batched_mode = unnorm_box.shape[0] > 1
                        else:
                            batched_mode = False
                        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                        
                        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=True,
                            repeat_image=batched_mode,
                            high_res_features=high_res_features,
                        )
                        
                        # Upscale masks
                        prd_masks = predictor._transforms.postprocess_masks(
                            low_res_masks, predictor._orig_hw[-1]
                        )
                        
                        # Calculate loss - use first mask from the 3 predictions
                        prd_mask = torch.sigmoid(prd_masks[:, 0])  # Shape: [batch, H, W]
                        
                        # Ensure gt_mask has batch dimension to match prd_mask
                        gt_mask = torch.tensor(crop_mask.astype(np.float32)).cuda()
                        if gt_mask.ndim == 2:
                            gt_mask = gt_mask.unsqueeze(0)  # Add batch dimension: [1, H, W]
                        
                        # Segmentation loss
                        seg_loss = (
                            -gt_mask * torch.log(prd_mask + 0.00001) - 
                            (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)
                        ).mean()
                        
                        # IOU calculation - flatten spatial dimensions
                        # Reshape to [batch, H*W] for easier calculation
                        gt_flat = gt_mask.reshape(gt_mask.shape[0], -1)
                        pred_flat = (prd_mask > 0.5).reshape(prd_mask.shape[0], -1)
                        
                        inter = (gt_flat * pred_flat).sum(1)
                        union = gt_flat.sum(1) + pred_flat.sum(1) - inter
                        iou = inter / (union + 1e-6)  # Add epsilon to avoid division by zero
                        
                        # Accumulate metrics
                        total_loss += seg_loss.item()
                        total_iou += iou.mean().item()
                        valid_samples += 1
                        
                        # Store for visualization (limit to 3 to keep output manageable)
                        if len(examples_to_visualize) < 3:
                            # Get the first prediction from batch
                            pred_mask_np = prd_mask[0].cpu().numpy() if prd_mask.shape[0] > 0 else prd_mask.cpu().numpy()
                            # Extract box coordinates from shape [1, 4] -> [x1, y1, x2, y2]
                            box_coords = crop_box[0] if crop_box.ndim == 2 else crop_box
                            examples_to_visualize.append({
                                'image': crop_image,
                                'gt_mask': crop_mask if crop_mask.ndim == 2 else crop_mask[0],
                                'pred_mask': pred_mask_np,
                                'box': box_coords,  # Now [x1, y1, x2, y2] as 1D array
                                'iou': iou.mean().item()
                            })
                
                except Exception as e:
                    print(f"  Validation sample {idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Calculate average metrics
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
            avg_iou = total_iou / valid_samples
            print(f"  Validated on {valid_samples}/{num_samples} samples")
            print(f"  Avg Loss: {avg_loss:.4f}, Avg IOU: {avg_iou:.4f}")
        else:
            avg_loss = 0.0
            avg_iou = 0.0
            print(f"  Warning: No valid samples processed (0/{num_samples})")
        
        # Visualize examples
        if examples_to_visualize:
            fig, axes = plt.subplots(len(examples_to_visualize), 4, figsize=(16, 4 * len(examples_to_visualize)))
            if len(examples_to_visualize) == 1:
                axes = axes.reshape(1, -1)
            
            for i, example in enumerate(examples_to_visualize):
                # Plot 1: Input image with box
                axes[i, 0].imshow(example['image'])
                x1, y1, x2, y2 = example['box']
                width = x2 - x1
                height = y2 - y1
                rect = plt.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='red', facecolor='none')
                axes[i, 0].add_patch(rect)
                axes[i, 0].set_title(f'Input Image + Box\nIOU: {example["iou"]:.3f}')
                axes[i, 0].axis('off')
                
                # Plot 2: Ground truth mask
                axes[i, 1].imshow(example['gt_mask'], cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                # Plot 3: Predicted mask
                axes[i, 2].imshow(example['pred_mask'] > 0.5, cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
                
                # Plot 4: Overlay comparison
                axes[i, 3].imshow(example['image'])
                # Show GT in green, prediction in red
                gt_overlay = np.zeros_like(example['image'])
                gt_overlay[:, :, 1] = example['gt_mask'] * 255  # Green for GT
                pred_overlay = np.zeros_like(example['image'])
                pred_overlay[:, :, 0] = (example['pred_mask'] > 0.5) * 255  # Red for prediction
                axes[i, 3].imshow(gt_overlay, alpha=0.3)
                axes[i, 3].imshow(pred_overlay, alpha=0.3)
                axes[i, 3].set_title('Overlay (GT=Green, Pred=Red)')
                axes[i, 3].axis('off')
            
            plt.suptitle(f'Validation Results - Step {step}\nAvg Loss: {avg_loss:.4f}, Avg IOU: {avg_iou:.4f}', 
                        fontsize=14, y=0.995)
            plt.tight_layout()
            
            # Save figure
            output_path = val_debug_dir / f'validation_step_{step:05d}.png'
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved validation visualization to: {output_path}")
        
        return {'val_loss': avg_loss, 'val_iou': avg_iou}
