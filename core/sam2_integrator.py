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
    
    def _propagate_mask_chunk_video(self, temp_dir, mask_indices_and_masks, start_obj_id):
        """
        Helper method to propagate a chunk of masks using video predictor.
        
        Args:
            temp_dir: Directory containing frame files (0.jpg, 1.jpg)
            mask_indices_and_masks: List of (original_index, mask) tuples to process
            start_obj_id: Starting object ID for this chunk
            
        Returns:
            Dictionary mapping original_index to propagated mask (or None for failures)
        """
        import torch
        
        results = {}
        
        try:
            # Initialize video state with temporary directory
            inference_state = self.video_predictor.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True
            )
            
            # Add each mask from current frame as an object to track
            object_ids = []
            next_obj_id = start_obj_id
            
            for original_idx, mask in mask_indices_and_masks:
                try:
                    # Check if mask is empty
                    if not np.any(mask > 0):
                        print(f"Mask {original_idx} is empty, skipping")
                        results[original_idx] = None
                        continue
                    
                    # Convert mask to binary (0 or 1)
                    binary_mask = (mask > 0).astype(np.uint8)
                    
                    # Add mask to first frame (frame_idx=0) with unique obj_id
                    frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,  # Current frame is index 0
                        obj_id=next_obj_id,  # Use unique sequential ID
                        mask=binary_mask
                    )
                    
                    object_ids.append((original_idx, next_obj_id))
                    next_obj_id += 1
                    
                except Exception as e:
                    print(f"Error adding mask {original_idx} to video predictor: {e}")
                    results[original_idx] = None
                    continue
            
            if not object_ids:
                print("No valid masks to propagate in this chunk")
                for original_idx, _ in mask_indices_and_masks:
                    results[original_idx] = None
                return results
            
            print(f"Processing chunk: {len(object_ids)} masks with obj_ids {[obj_id for _, obj_id in object_ids]}")
            
            # Propagate masks to all frames (just frame 1 in our case)
            video_segments = {}
            for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0
            ):
                # Handle obj_ids that might be arrays/lists
                processed_masks = {}
                for i, obj_id_raw in enumerate(obj_ids):
                    # Convert obj_id to int (might be array/list)
                    if isinstance(obj_id_raw, (list, np.ndarray)):
                        obj_id = int(obj_id_raw[0]) if len(obj_id_raw) > 0 else i
                    else:
                        obj_id = int(obj_id_raw)
                    
                    # Move to CPU and convert to numpy immediately to free GPU memory
                    processed_masks[obj_id] = (mask_logits[i] > 0.0).cpu().numpy()
                
                video_segments[frame_idx] = processed_masks
                
                # Clear CUDA cache after processing each frame
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Extract masks for next frame (frame_idx=1)
            next_frame_masks = video_segments.get(1, {})
            
            # Build result dictionary maintaining original indices
            for original_idx, obj_id in object_ids:
                if obj_id in next_frame_masks:
                    # Get propagated mask
                    propagated_mask = next_frame_masks[obj_id].squeeze()
                    
                    # Convert to uint8 (0 or 255)
                    propagated_mask = (propagated_mask > 0).astype(np.uint8) * 255
                    
                    # Check if mask is valid
                    if np.any(propagated_mask > 0):
                        results[original_idx] = propagated_mask
                    else:
                        print(f"Mask {original_idx} propagation resulted in empty mask")
                        results[original_idx] = None
                else:
                    print(f"Mask {original_idx} not found in propagation results")
                    results[original_idx] = None
            
            # Clean up video state to free memory
            try:
                self.video_predictor.reset_state(inference_state)
            except Exception as e:
                print(f"Warning: Failed to reset video predictor state: {e}")
            
            # Delete inference state explicitly
            try:
                del inference_state
            except:
                pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache after cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"Chunk cleanup complete. Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        except Exception as e:
            print(f"Error in chunk propagation: {e}")
            import traceback
            traceback.print_exc()
            # Return None for all masks in this chunk on error
            for original_idx, _ in mask_indices_and_masks:
                results[original_idx] = None
        
        return results
    
    def propagate_masks_to_frame_video(self, current_frame, current_masks, next_frame):
        """
        Propagate masks from current frame to next frame using SAM2's video propagation.
        This uses the official propagate_in_video API which is more accurate but memory intensive.
        For more than 10 masks, processes them in chunks to manage memory.
        
        Args:
            current_frame: Current frame image (H, W, 3) in RGB format
            current_masks: List of masks from current frame (each mask is H x W, values 0 or 255)
            next_frame: Next frame image (H, W, 3) in RGB format
            
        Returns:
            List of propagated masks for next frame (or None for failed propagations)
        """
        import tempfile
        import os
        import torch
        
        # Clear CUDA cache before starting to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Cleared CUDA cache. Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        try:
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save frames as temporary files (SAM2 expects numeric names: 0.jpg, 1.jpg, etc.)
                frame0_path = os.path.join(temp_dir, "0.jpg")
                frame1_path = os.path.join(temp_dir, "1.jpg")
                
                import cv2
                # Convert RGB to BGR for cv2.imwrite
                cv2.imwrite(frame0_path, cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(frame1_path, cv2.cvtColor(next_frame, cv2.COLOR_RGB2BGR))
                
                # Process masks in chunks if there are more than 10
                chunk_size = 10
                num_masks = len(current_masks)
                
                if num_masks > chunk_size:
                    print(f"Processing {num_masks} masks in chunks of {chunk_size}")
                    
                    # Process in chunks
                    all_results = {}
                    for chunk_start in range(0, num_masks, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, num_masks)
                        chunk_num = chunk_start // chunk_size + 1
                        total_chunks = (num_masks + chunk_size - 1) // chunk_size
                        
                        print(f"\nProcessing chunk {chunk_num}/{total_chunks} (masks {chunk_start}-{chunk_end-1})")
                        
                        # Prepare chunk data: (original_index, mask) tuples
                        chunk_data = [(i, current_masks[i]) for i in range(chunk_start, chunk_end)]
                        
                        # Process this chunk
                        chunk_results = self._propagate_mask_chunk_video(
                            temp_dir, 
                            chunk_data, 
                            start_obj_id=0  # Reset obj_id for each chunk
                        )
                        
                        # Merge into all results
                        all_results.update(chunk_results)
                        
                        print(f"Chunk {chunk_num} complete. Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
                    # Build final result list in original order
                    propagated_masks = [all_results.get(i) for i in range(num_masks)]
                    
                else:
                    # Process all masks at once if there are 10 or fewer
                    print(f"Processing {num_masks} masks in a single batch")
                    chunk_data = [(i, mask) for i, mask in enumerate(current_masks)]
                    all_results = self._propagate_mask_chunk_video(
                        temp_dir,
                        chunk_data,
                        start_obj_id=0
                    )
                    propagated_masks = [all_results.get(i) for i in range(num_masks)]
                
                # Final cleanup
                import gc
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"Final cleanup. Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            # Temp directory and files are automatically cleaned up here
            
        except Exception as e:
            print(f"Error in video propagation: {e}")
            import traceback
            traceback.print_exc()
            # Return None for all masks on error
            propagated_masks = [None] * len(current_masks)
        
        finally:
            # Ensure garbage collection happens even if exception occurs
            import gc
            gc.collect()
        
        return propagated_masks
