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
    
    def _propagate_mask_chunk_video(self, temp_dir, mask_indices_and_masks, start_obj_id, mask_threshold=0.0, DEBUG=False):
        """
        Helper method to propagate a chunk of masks using video predictor.
        
        Args:
            temp_dir: Directory containing frame files (0.jpg, 1.jpg)
            mask_indices_and_masks: List of (original_index, mask) tuples to process
            start_obj_id: Starting object ID for this chunk
            mask_threshold: Threshold for converting logits to binary masks (default: 0.0)
            
        Returns:
            Dictionary mapping original_index to propagated mask (or None for failures)
        """
        import torch
        
        results = {}
        
        # Create mapping from original_idx to mask for comparison later
        original_masks_map = {idx: mask for idx, mask in mask_indices_and_masks}
        
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
            print(f"  Starting propagation from frame 0...")
            video_segments = {}
            for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0
            ):
                print(f"  Received propagation for frame {frame_idx}: {len(obj_ids)} masks, logits shape: {mask_logits.shape}")
                # Handle obj_ids that might be arrays/lists
                processed_masks = {}
                for i, obj_id_raw in enumerate(obj_ids):
                    # Convert obj_id to int (might be array/list)
                    if isinstance(obj_id_raw, (list, np.ndarray)):
                        obj_id = int(obj_id_raw[0]) if len(obj_id_raw) > 0 else i
                    else:
                        obj_id = int(obj_id_raw)
                    
                    # Move to CPU and convert to numpy immediately to free GPU memory
                    mask_logits_np = mask_logits[i].cpu().numpy()
                    print(f"    obj_id {obj_id}: logits range [{mask_logits_np.min():.2f}, {mask_logits_np.max():.2f}]")
                    processed_masks[obj_id] = (mask_logits[i] > mask_threshold).cpu().numpy()
                
                video_segments[frame_idx] = processed_masks
                
                # Clear CUDA cache after processing each frame
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if DEBUG:
                print(f"  Total frames in video_segments: {list(video_segments.keys())}")
            
            # Extract masks for next frame (frame_idx=1)
            next_frame_masks = video_segments.get(1, {})
            if DEBUG:
                print(f"  Masks available for frame 1: {len(next_frame_masks)}")
            
            
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
                        
                        # Debug: Compare with original mask to check if it actually changed
                        if DEBUG:
                            if original_idx in original_masks_map:
                                original_mask = original_masks_map[original_idx]
                                if original_mask is not None and propagated_mask.shape == original_mask.shape:
                                    # Calculate overlap/difference
                                    same_pixels = np.sum((original_mask > 0) == (propagated_mask > 0))
                                    total_pixels = original_mask.size
                                    similarity = same_pixels / total_pixels * 100
                                    print(f"    Mask {original_idx + 1}: {similarity:.1f}% similar to input")
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
    
    def _calculate_mask_iou(self, mask1, mask2):
        """
        Calculate IoU (Intersection over Union) between two masks.
        
        Args:
            mask1: First mask (numpy array, values 0 or 255)
            mask2: Second mask (numpy array, values 0 or 255)
            
        Returns:
            float: IoU score between 0 and 1
        """
        if mask1 is None or mask2 is None:
            return 0.0
        
        # Convert to binary
        binary1 = (mask1 > 0).astype(np.bool_)
        binary2 = (mask2 > 0).astype(np.bool_)
        
        # Calculate intersection and union
        intersection = np.logical_and(binary1, binary2).sum()
        union = np.logical_or(binary1, binary2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def _find_overlapping_groups(self, masks, overlap_threshold=0.3):
        """
        Find groups of masks that significantly overlap with each other.
        
        Args:
            masks: List of masks (each can be None or numpy array)
            overlap_threshold: Minimum IoU to consider masks as overlapping
            
        Returns:
            List of lists, where each inner list contains indices of overlapping masks
        """
        num_masks = len(masks)
        if num_masks <= 1:
            return []
        
        # Build adjacency graph of overlapping masks
        overlaps = set()
        for i in range(num_masks):
            if masks[i] is None:
                continue
            for j in range(i + 1, num_masks):
                if masks[j] is None:
                    continue
                
                iou = self._calculate_mask_iou(masks[i], masks[j])
                if iou >= overlap_threshold:
                    overlaps.add((i, j))
                    # Report with instance IDs (1-based)
                    instance_i = i + 1
                    instance_j = j + 1
                    print(f"Found overlap: instances {instance_i} and {instance_j} have IoU {iou:.3f}")
        
        if not overlaps:
            return []
        
        # Find connected components using union-find
        parent = list(range(num_masks))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union overlapping pairs
        for i, j in overlaps:
            union(i, j)
        
        # Group by root parent
        groups = {}
        for i in range(num_masks):
            if masks[i] is not None:
                root = find(i)
                if root not in groups:
                    groups[root] = []
                groups[root].append(i)
        
        # Return only groups with 2+ members (overlapping groups)
        overlapping_groups = [group for group in groups.values() if len(group) >= 2]
        
        return overlapping_groups
    
    def _resolve_mask_overlaps(self, propagated_masks, original_masks, method='distance'):
        """
        Resolve overlaps between propagated masks by assigning overlapping pixels
        to the most likely instance.
        
        Args:
            propagated_masks: List of propagated masks with overlaps
            original_masks: List of original masks from current frame (for reference)
            method: Method to resolve overlaps:
                - 'area': Assign to mask that needs pixels to match original area
                - 'distance': Assign to nearest original mask centroid
                - 'morphology': Use erosion to separate
                - 'watershed': Use watershed algorithm with original masks as seeds
                
        Returns:
            List of masks with overlaps resolved
        """
        import cv2
        from scipy import ndimage
        
        resolved_masks = []
        
        # Get image dimensions from first non-None mask
        img_shape = None
        for mask in propagated_masks:
            if mask is not None:
                img_shape = mask.shape
                break
        
        if img_shape is None:
            return propagated_masks
        
        # Create overlap map to identify problematic regions
        overlap_map = np.zeros(img_shape, dtype=np.uint8)
        for mask in propagated_masks:
            if mask is not None:
                overlap_map += (mask > 0).astype(np.uint8)
        
        # No overlaps - return as is
        if overlap_map.max() <= 1:
            return propagated_masks
        
        print(f"  Resolving overlaps using method: {method}")
        overlap_pixels = (overlap_map > 1).sum()
        total_pixels = (overlap_map > 0).sum()
        print(f"  Overlap region: {overlap_pixels} pixels ({100*overlap_pixels/max(total_pixels,1):.1f}% of masked area)")
        
        if method == 'area':
            # Method: Area-preserving assignment
            # Assign overlapping pixels to masks that need them to maintain original area
            
            # Calculate target areas from original masks
            original_areas = []
            for mask in original_masks:
                if mask is not None:
                    area = np.sum(mask > 0)
                    original_areas.append(area)
                else:
                    original_areas.append(0)
            
            print(f"  Original areas: {original_areas}")
            
            # Initialize resolved masks and track current areas (excluding overlaps)
            resolved_masks = []
            current_areas = []
            for i, mask in enumerate(propagated_masks):
                if mask is not None:
                    # Count only non-overlapping pixels for now
                    non_overlap = (mask > 0) & (overlap_map == 1)
                    current_areas.append(np.sum(non_overlap))
                    resolved_masks.append(mask.copy())
                else:
                    current_areas.append(0)
                    resolved_masks.append(None)
            
            print(f"  Current areas (before overlap resolution): {current_areas}")
            
            # Get all overlapping pixels
            overlap_coords = np.argwhere(overlap_map > 1)
            
            # For each overlapping pixel, assign to mask with largest area deficit
            for coord in overlap_coords:
                y, x = coord
                
                # Find which masks claim this pixel
                claiming_masks = []
                for i, mask in enumerate(propagated_masks):
                    if mask is not None and mask[y, x] > 0:
                        claiming_masks.append(i)
                
                if len(claiming_masks) == 0:
                    continue
                
                # Calculate area deficit for each claiming mask
                # Deficit = original_area - current_area (positive means needs more pixels)
                deficits = []
                for i in claiming_masks:
                    deficit = original_areas[i] - current_areas[i]
                    deficits.append(deficit)
                
                # Assign pixel to mask with largest deficit
                winner_idx = claiming_masks[np.argmax(deficits)]
                
                # Remove from all other masks
                for i in claiming_masks:
                    if i != winner_idx:
                        resolved_masks[i][y, x] = 0
                
                # Update current area for winner
                current_areas[winner_idx] += 1
            
            final_areas = [np.sum(m > 0) if m is not None else 0 for m in resolved_masks]
            print(f"  Final areas (after overlap resolution): {final_areas}")
            area_diffs = [abs(final_areas[i] - original_areas[i]) for i in range(len(original_areas))]
            print(f"  Area differences from original: {area_diffs} pixels")
        
        elif method == 'distance':
            # Method 1: Distance-based assignment
            # Calculate centroids of original masks
            centroids = []
            for i, mask in enumerate(original_masks):
                if mask is not None and np.any(mask > 0):
                    coords = np.argwhere(mask > 0)
                    centroid = coords.mean(axis=0)  # [y, x]
                    centroids.append(centroid)
                else:
                    centroids.append(None)
            
            # For each propagated mask, resolve overlaps
            for i, mask in enumerate(propagated_masks):
                if mask is None or centroids[i] is None:
                    resolved_masks.append(mask)
                    continue
                
                # Create resolved mask
                resolved = mask.copy()
                
                # Find overlapping regions for this mask
                mask_binary = (mask > 0)
                is_overlap = (overlap_map > 1)
                overlap_region = mask_binary & is_overlap
                
                if overlap_region.any():
                    # Get coordinates of overlapping pixels
                    overlap_coords = np.argwhere(overlap_region)
                    
                    # Calculate distance from this mask's centroid
                    my_centroid = centroids[i]
                    my_distances = np.sqrt(((overlap_coords - my_centroid) ** 2).sum(axis=1))
                    
                    # Check distances from other masks' centroids
                    for j, other_mask in enumerate(propagated_masks):
                        if j == i or other_mask is None or centroids[j] is None:
                            continue
                        
                        other_mask_binary = (other_mask > 0)
                        other_overlap = other_mask_binary & overlap_region
                        
                        if other_overlap.any():
                            # Calculate distance from other centroid
                            other_centroid = centroids[j]
                            other_distances = np.sqrt(((overlap_coords - other_centroid) ** 2).sum(axis=1))
                            
                            # Remove pixels that are closer to other mask
                            remove_pixels = other_distances < my_distances
                            remove_coords = overlap_coords[remove_pixels]
                            resolved[remove_coords[:, 0], remove_coords[:, 1]] = 0
                
                resolved_masks.append(resolved)
        
        elif method == 'morphology':
            # Method 2: Morphological erosion/dilation
            # Erode each mask slightly to create separation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            for mask in propagated_masks:
                if mask is not None and np.any(mask > 0):
                    # Erode to separate
                    eroded = cv2.erode(mask, kernel, iterations=2)
                    # Dilate back (but now separated)
                    dilated = cv2.dilate(eroded, kernel, iterations=2)
                    resolved_masks.append(dilated)
                else:
                    resolved_masks.append(mask)
        
        elif method == 'watershed':
            # Method 3: Watershed segmentation
            # Use original masks to create seeds/markers
            markers = np.zeros(img_shape, dtype=np.int32)
            
            for i, mask in enumerate(original_masks):
                if mask is not None and np.any(mask > 0):
                    # Erode original mask to get core region (marker)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    core = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=2)
                    markers[core > 0] = i + 1  # Markers are 1-indexed
            
            # Create combined mask from all propagated masks
            combined = np.zeros(img_shape, dtype=np.uint8)
            for mask in propagated_masks:
                if mask is not None:
                    combined = np.maximum(combined, mask)
            
            # Apply watershed
            # Convert to 3-channel for watershed
            combined_3ch = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(combined_3ch, markers)
            
            # Extract individual masks from watershed result
            for i in range(len(propagated_masks)):
                if propagated_masks[i] is not None:
                    resolved = ((markers == (i + 1)).astype(np.uint8) * 255)
                    resolved_masks.append(resolved)
                else:
                    resolved_masks.append(None)
        
        else:
            # Unknown method, return original
            return propagated_masks
        
        # Calculate improvement
        new_overlap_map = np.zeros(img_shape, dtype=np.uint8)
        for mask in resolved_masks:
            if mask is not None:
                new_overlap_map += (mask > 0).astype(np.uint8)
        
        new_overlap_pixels = (new_overlap_map > 1).sum()
        print(f"  After resolution: {new_overlap_pixels} overlapping pixels (reduced by {overlap_pixels - new_overlap_pixels})")
        
        return resolved_masks
    
    def propagate_masks_to_frame_video(self, current_frame, current_masks, next_frame, 
                                      overlap_threshold=0.3, enable_overlap_correction=True,
                                      resolve_overlaps=True, overlap_resolution_method='area',
                                      mask_threshold=0.0):
        """
        Propagate masks from current frame to next frame using SAM2's video propagation.
        This uses the official propagate_in_video API which is more accurate but memory intensive.
        For more than 10 masks, processes them in chunks to manage memory.
        
        If instances overlap significantly after chunk-based propagation, they are re-propagated
        together in the same chunk to prevent one instance from overtaking another.
        
        Additionally, overlapping regions can be resolved using various post-processing methods.
        
        Args:
            current_frame: Current frame image (H, W, 3) in RGB format
            current_masks: List of masks from current frame (each mask is H x W, values 0 or 255)
            next_frame: Next frame image (H, W, 3) in RGB format
            overlap_threshold: IoU threshold for detecting overlaps (default: 0.3)
            enable_overlap_correction: Whether to re-propagate overlapping instances (default: True)
            resolve_overlaps: Whether to apply post-processing to resolve remaining overlaps (default: True)
            overlap_resolution_method: Method for resolving overlaps (default: 'area')
                - 'area': Assign overlapping pixels to maintain original mask areas (recommended)
                - 'distance': Assign overlapping pixels based on distance to original mask centroids
                - 'morphology': Use morphological erosion/dilation to separate masks
                - 'watershed': Use watershed algorithm with original masks as seeds
            mask_threshold: Threshold for converting SAM2 logits to binary masks (default: 0.0)
                Higher values (e.g., 0.5-2.0) produce more conservative masks with less overlap
            
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
        
        # Report threshold if non-default
        if mask_threshold != 0.0:
            print(f"Using mask threshold: {mask_threshold} (higher = more conservative masks)")
        
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
                            start_obj_id=0,  # Reset obj_id for each chunk
                            mask_threshold=mask_threshold
                        )
                        
                        # Merge into all results
                        all_results.update(chunk_results)
                        
                        print(f"Chunk {chunk_num} complete. Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
                    # Build final result list in original order
                    propagated_masks = [all_results.get(i) for i in range(num_masks)]
                    
                    # Check for overlaps and re-propagate if needed
                    if enable_overlap_correction:
                        overlapping_groups = self._find_overlapping_groups(propagated_masks, overlap_threshold)
                        
                        if overlapping_groups:
                            print(f"\n{'='*60}")
                            print(f"Found {len(overlapping_groups)} overlapping groups after chunk propagation")
                            print(f"Re-propagating these instances together to prevent overtaking...")
                            print(f"{'='*60}\n")
                            
                            # Re-propagate each overlapping group
                            for group_num, group_indices in enumerate(overlapping_groups, 1):
                                # Convert to instance IDs (1-based) for display
                                instance_ids = [idx + 1 for idx in group_indices]
                                print(f"\nRe-propagating overlapping group {group_num}/{len(overlapping_groups)}: instances {instance_ids}")
                                
                                # Check if original masks overlap (helps with debugging)
                                print(f"  Checking original masks (current frame):...")
                                for i in range(len(group_indices)):
                                    for j in range(i + 1, len(group_indices)):
                                        idx_i = group_indices[i]
                                        idx_j = group_indices[j]
                                        if current_masks[idx_i] is not None and current_masks[idx_j] is not None:
                                            original_iou = self._calculate_mask_iou(current_masks[idx_i], current_masks[idx_j])
                                            print(f"    Instances {idx_i + 1} & {idx_j + 1}: {original_iou:.3f} (original frame)")
                                
                                # Calculate pairwise IoUs BEFORE re-propagation
                                print(f"  IoU before re-propagation:")
                                old_ious = {}
                                for i in range(len(group_indices)):
                                    for j in range(i + 1, len(group_indices)):
                                        idx_i = group_indices[i]
                                        idx_j = group_indices[j]
                                        if propagated_masks[idx_i] is not None and propagated_masks[idx_j] is not None:
                                            iou_before = self._calculate_mask_iou(propagated_masks[idx_i], propagated_masks[idx_j])
                                            old_ious[(idx_i, idx_j)] = iou_before
                                            print(f"    Instances {idx_i + 1} & {idx_j + 1}: {iou_before:.3f}")
                                
                                # Prepare data for this group
                                group_data = [(i, current_masks[i]) for i in group_indices]
                                
                                # Re-propagate this group together
                                try:
                                    group_results = self._propagate_mask_chunk_video(
                                        temp_dir,
                                        group_data,
                                        start_obj_id=0,
                                        mask_threshold=mask_threshold
                                    )
                                    
                                    # Update results with re-propagated masks
                                    for idx in group_indices:
                                        if idx in group_results:
                                            propagated_masks[idx] = group_results[idx]
                                    
                                    # Calculate pairwise IoUs AFTER re-propagation
                                    print(f"  IoU after re-propagation:")
                                    for i in range(len(group_indices)):
                                        for j in range(i + 1, len(group_indices)):
                                            idx_i = group_indices[i]
                                            idx_j = group_indices[j]
                                            if propagated_masks[idx_i] is not None and propagated_masks[idx_j] is not None:
                                                iou_after = self._calculate_mask_iou(propagated_masks[idx_i], propagated_masks[idx_j])
                                                iou_before = old_ious.get((idx_i, idx_j), 0.0)
                                                improvement = iou_before - iou_after
                                                improvement_pct = (improvement / max(iou_before, 0.001)) * 100
                                                status = "✓ improved" if improvement > 0.01 else "⚠ no change" if abs(improvement) < 0.01 else "✗ worse"
                                                print(f"    Instances {idx_i + 1} & {idx_j + 1}: {iou_after:.3f} (change: {improvement:+.3f}, {improvement_pct:+.1f}%) {status}")
                                    
                                except Exception as e:
                                    print(f"  Warning: Re-propagation failed for group {instance_ids}: {e}")
                                    # Keep original propagations on error
                            
                            print(f"\n{'='*60}")
                            print(f"Overlap correction complete")
                            print(f"{'='*60}\n")
                    
                else:
                    # Process all masks at once if there are 10 or fewer
                    print(f"Processing {num_masks} masks in a single batch")
                    chunk_data = [(i, mask) for i, mask in enumerate(current_masks)]
                    all_results = self._propagate_mask_chunk_video(
                        temp_dir,
                        chunk_data,
                        start_obj_id=0,
                        mask_threshold=mask_threshold
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
        
        # Apply final overlap resolution if requested and propagation succeeded
        if resolve_overlaps and propagated_masks:
            # Check if any propagated masks are valid (not None)
            valid_masks = [m for m in propagated_masks if m is not None]
            if len(valid_masks) > 1:
                print(f"\n{'='*60}")
                print(f"Applying final overlap resolution using '{overlap_resolution_method}' method")
                print(f"{'='*60}")
                
                # Call the resolution method
                try:
                    propagated_masks = self._resolve_mask_overlaps(
                        propagated_masks,
                        current_masks, 
                        method=overlap_resolution_method
                    )
                    print(f"✓ Overlap resolution complete")
                except Exception as e:
                    print(f"✗ Overlap resolution failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with unresolved masks on error
                
                print(f"{'='*60}\n")
        
        return propagated_masks
