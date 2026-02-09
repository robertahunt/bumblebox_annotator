"""
SAM2 integration for interactive segmentation
"""

import torch
import numpy as np
from pathlib import Path


class SAM2Integrator:
    """SAM2 model integration for interactive segmentation"""
    
    def __init__(self, checkpoint_path, config_name='sam2_hiera_l.yaml', device=None):
        """
        Initialize SAM2 model
        
        Args:
            checkpoint_path: Path to SAM2 checkpoint
            config_name: Model configuration name
            device: Device to run on (cuda/cpu)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_name = config_name
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading SAM2 on {self.device}...")
        
        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Build model for image prediction
            sam2_model = build_sam2(self.config_name, str(self.checkpoint_path))
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            # Build video predictor separately
            self.video_predictor = build_sam2_video_predictor(self.config_name, str(self.checkpoint_path))
            
            print("SAM2 loaded successfully")
            
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM2: {str(e)}")
            
        self.current_image = None
        self.features_computed = False
        
        # Video tracking state
        self.video_state = None
        self.frame_idx = 0
        
    def set_image(self, image):
        """
        Set image for prediction
        
        Args:
            image: numpy array (H, W, 3) in RGB format
        """
        self.current_image = image
        self.predictor.set_image(image)
        self.features_computed = True
        
    def predict_with_point(self, image, x, y, is_positive=True, multimask=False):
        """
        Predict mask from point prompt
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            x, y: Point coordinates
            is_positive: Whether point is foreground (True) or background (False)
            multimask: Whether to return multiple masks
            
        Returns:
            Binary mask (H, W)
        """
        # Check if image has changed by comparing object identity
        if self.current_image is None or not self.features_computed or self.current_image is not image:
            self.set_image(image)
            
        # Prepare point prompt
        point = np.array([[x, y]], dtype=np.float32)
        label = np.array([1 if is_positive else 0], dtype=np.int32)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=multimask
        )
        
        # Return best mask
        if multimask:
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8) * 255
        else:
            return masks[0].astype(np.uint8) * 255
            
    def predict_with_points(self, image, positive_points, negative_points, multimask=False):
        """
        Predict mask from multiple point prompts
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            positive_points: List of (x, y) tuples for foreground points
            negative_points: List of (x, y) tuples for background points
            multimask: Whether to return multiple masks
            
        Returns:
            Binary mask (H, W)
        """
        # Always set image to ensure we're working with the current frame
        # Check if image has changed by comparing object identity
        if self.current_image is None or not self.features_computed or self.current_image is not image:
            self.set_image(image)
            
        # Combine points and labels
        all_points = []
        all_labels = []
        
        for x, y in positive_points:
            all_points.append([x, y])
            all_labels.append(1)
            
        for x, y in negative_points:
            all_points.append([x, y])
            all_labels.append(0)
            
        if not all_points:
            # No points provided, return empty mask
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        points = np.array(all_points, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask
        )
        
        # Return best mask
        if multimask:
            best_idx = np.argmax(scores)
            return masks[best_idx].astype(np.uint8) * 255
        else:
            return masks[0].astype(np.uint8) * 255
            
    def predict_with_box(self, image, x1, y1, x2, y2):
        """
        Predict mask from box prompt
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            x1, y1, x2, y2: Box coordinates
            
        Returns:
            Binary mask (H, W)
        """
        # Check if image has changed by comparing object identity
        if self.current_image is None or not self.features_computed or self.current_image is not image:
            self.set_image(image)
            
        # Prepare box prompt (xyxy format)
        box = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False
        )
        
        return masks[0].astype(np.uint8) * 255
        
    def predict_with_points_and_box(self, image, points, point_labels, box=None):
        """
        Predict mask with multiple prompts
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            points: numpy array of shape (N, 2) with point coordinates
            point_labels: numpy array of shape (N,) with point labels (1=positive, 0=negative)
            box: Optional box prompt (x1, y1, x2, y2)
            
        Returns:
            Binary mask (H, W)
        """
        # Check if image has changed by comparing object identity
        if self.current_image is None or not self.features_computed or self.current_image is not image:
            self.set_image(image)
            
        # Prepare prompts
        kwargs = {
            'point_coords': points.astype(np.float32),
            'point_labels': point_labels.astype(np.int32),
            'multimask_output': False
        }
        
        if box is not None:
            kwargs['box'] = np.array([box], dtype=np.float32)
            
        # Predict
        masks, scores, logits = self.predictor.predict(**kwargs)
        
        return masks[0].astype(np.uint8) * 255
        
    def refine_mask(self, image, current_mask, new_points, new_labels):
        """
        Refine existing mask with additional prompts
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            current_mask: Current binary mask
            new_points: Additional points
            new_labels: Labels for new points
            
        Returns:
            Refined binary mask (H, W)
        """
        # Check if image has changed by comparing object identity
        if self.current_image is None or not self.features_computed or self.current_image is not image:
            self.set_image(image)
            
        # Use current mask as prompt
        mask_input = current_mask[None, :, :].astype(np.float32)
        
        # Predict with additional points
        masks, scores, logits = self.predictor.predict(
            point_coords=new_points.astype(np.float32),
            point_labels=new_labels.astype(np.int32),
            mask_input=mask_input,
            multimask_output=False
        )
        
        return masks[0].astype(np.uint8) * 255
        
    def init_video_state(self, first_frame):
        """
        Initialize video tracking state with first frame
        
        Args:
            first_frame: First frame image (H, W, 3) in RGB format
        """
        self.video_state = self.video_predictor.init_state(first_frame)
        self.frame_idx = 0
        
    def add_masks_to_video_state(self, frame_idx, masks):
        """
        Add masks to video state for tracking
        
        Args:
            frame_idx: Frame index
            masks: List of binary masks to track
            
        Returns:
            List of object IDs assigned to each mask
        """
        if self.video_state is None:
            raise RuntimeError("Video state not initialized. Call init_video_state first.")
            
        object_ids = []
        for i, mask in enumerate(masks):
            # Add mask as new object to track
            obj_id = len(object_ids)
            _, _, _ = self.video_predictor.add_new_mask(
                self.video_state,
                frame_idx,
                obj_id,
                mask > 0
            )
            object_ids.append(obj_id)
            
        return object_ids
        
    def propagate_masks_to_frame(self, current_frame_idx, current_masks, next_frame):
        """
        Propagate masks from current frame to next frame using SAM2 video tracking
        
        Args:
            current_frame_idx: Index of current frame
            current_masks: List of masks from current frame
            next_frame: Next frame image (H, W, 3) in RGB format
            
        Returns:
            List of propagated masks for next frame
        """
        propagated_masks = []
        
        # Set next frame in predictor
        self.predictor.set_image(next_frame)
        
        for i, mask in enumerate(current_masks):
            try:
                # Check if mask is empty
                if not np.any(mask > 0):
                    print(f"Mask {i} is empty, skipping")
                    propagated_masks.append(None)
                    continue
                
                # Extract multiple points from the mask to use as prompts
                # Sample points from inside the mask region
                y_coords, x_coords = np.where(mask > 0)
                
                # Use centroid as primary point
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                
                # Sample additional points spread across the mask
                num_samples = min(10, len(x_coords))
                if num_samples > 1:
                    indices = np.random.choice(len(x_coords), num_samples - 1, replace=False)
                    sample_points = [(x_coords[i], y_coords[i]) for i in indices]
                else:
                    sample_points = []
                
                # Combine all points
                all_points = [(center_x, center_y)] + sample_points
                point_coords = np.array(all_points, dtype=np.float32)
                point_labels = np.ones(len(all_points), dtype=np.int32)  # All positive
                
                # Predict on next frame using multiple points from the mask
                # This tells SAM2 where the object was and helps it find it in the new frame
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False
                )
                
                # Check if propagated mask is valid (not empty)
                propagated_mask = masks[0].astype(np.uint8) * 255
                if not np.any(propagated_mask > 0):
                    print(f"Mask {i} propagation resulted in empty mask")
                    propagated_masks.append(None)
                else:
                    propagated_masks.append(propagated_mask)
                
            except Exception as e:
                print(f"Error propagating mask {i}: {e}")
                import traceback
                traceback.print_exc()
                propagated_masks.append(None)
                
        return propagated_masks


