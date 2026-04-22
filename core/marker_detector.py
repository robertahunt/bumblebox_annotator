"""
ArUco marker and QR code detection within segmentation masks.

This module provides functionality to detect and decode visual markers (ArUco/QR)
within bee segmentation masks for robust tracking and identification.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class MarkerDetection:
    """Result of marker detection"""
    marker_type: str  # 'aruco' | 'qr'
    marker_id: Union[int, str]  # int for ArUco, str for QR
    confidence: float  # Detection confidence (0-1)
    corners: np.ndarray  # Corner coordinates (4, 2) for ArUco, variable for QR
    dict_type: Optional[str] = None  # ArUco dictionary type (e.g., '4x4_50')
    center: Optional[Tuple[float, float]] = None  # Marker center (x, y)


class MarkerDetector:
    """Detect and decode ArUco markers and QR codes within segmentation masks"""
    
    # ArUco dictionaries to try (in order of preference)
    ARUCO_DICTS = {
        '4x4_50': cv2.aruco.DICT_4X4_50,
        '4x4_100': cv2.aruco.DICT_4X4_100,
        '4x4_250': cv2.aruco.DICT_4X4_250,
        '4x4_1000': cv2.aruco.DICT_4X4_1000,
        '5x5_50': cv2.aruco.DICT_5X5_50,
        '5x5_100': cv2.aruco.DICT_5X5_100,
        '5x5_250': cv2.aruco.DICT_5X5_250,
        '5x5_1000': cv2.aruco.DICT_5X5_1000,
        '6x6_50': cv2.aruco.DICT_6X6_50,
        '6x6_100': cv2.aruco.DICT_6X6_100,
        '6x6_250': cv2.aruco.DICT_6X6_250,
        '6x6_1000': cv2.aruco.DICT_6X6_1000,
        '7x7_50': cv2.aruco.DICT_7X7_50,
        '7x7_100': cv2.aruco.DICT_7X7_100,
        '7x7_250': cv2.aruco.DICT_7X7_250,
        '7x7_1000': cv2.aruco.DICT_7X7_1000,
    }
    
    def __init__(
        self,
        aruco_dicts: Optional[List[str]] = None,
        min_confidence: float = 0.2,
        enable_aruco: bool = True,
        enable_qr: bool = True,
        debug: bool = False,
        debug_folder: Optional[str] = None
    ):
        """
        Initialize marker detector.
        
        Args:
            aruco_dicts: List of ArUco dictionary names to try (None = try common ones)
            min_confidence: Minimum detection confidence threshold (0-1, default 0.2 for small markers)
            enable_aruco: Enable ArUco marker detection
            enable_qr: Enable QR code detection
        """
        self.min_confidence = min_confidence
        self.enable_aruco = enable_aruco
        self.enable_qr = enable_qr
        self.debug = debug
        self.debug_folder = debug_folder
        self.debug_counter = 0  # Counter for unique debug filenames
        
        # Create debug folder if debug is enabled
        if self.debug:
            print(f"[MarkerDetector] Debug mode enabled")
            if self.debug_folder:
                from pathlib import Path
                Path(self.debug_folder).mkdir(parents=True, exist_ok=True)
                print(f"[MarkerDetector] Debug images will be saved to: {self.debug_folder}")
            else:
                print(f"[MarkerDetector] WARNING: debug_folder not set yet, images won't be saved until set_debug_folder() is called")
        
        # Set up ArUco dictionaries to try
        if aruco_dicts is None:
            # Default: try common small dictionaries first
            self.aruco_dicts_to_try = [
                '4x4_50', '4x4_100', '4x4_250', '4x4_1000',
                '5x5_50', '5x5_100', '5x5_250',
                '6x6_50', '6x6_100'
            ]
        else:
            self.aruco_dicts_to_try = aruco_dicts
        
        # Initialize ArUco detectors for each dictionary
        self.aruco_detectors = {}
        if self.enable_aruco:
            for dict_name in self.aruco_dicts_to_try:
                if dict_name in self.ARUCO_DICTS:
                    aruco_dict = cv2.aruco.getPredefinedDictionary(
                        self.ARUCO_DICTS[dict_name]
                    )
                    detector_params = cv2.aruco.DetectorParameters()
                    # Adjust parameters for better detection
                    #detector_params.adaptiveThreshConstant = 7
                    #detector_params.minMarkerPerimeterRate = 0.03
                    #detector_params.maxMarkerPerimeterRate = 4.0
                    
                    self.aruco_detectors[dict_name] = cv2.aruco.ArucoDetector(
                        aruco_dict, detector_params
                    )
        
        # Initialize QR code detector
        self.qr_detector = None
        if self.enable_qr:
            self.qr_detector = cv2.QRCodeDetector()
    
    def set_debug_folder(self, folder_path: str):
        """Set or update the debug folder path
        
        Args:
            folder_path: Path to folder where debug images will be saved
        """
        from pathlib import Path
        self.debug_folder = folder_path
        if self.debug and folder_path:
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            print(f"[MarkerDetector] Debug folder updated: {folder_path}")
    
    def detect_in_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prefer_aruco: bool = True,
        instance_id: Optional[int] = None
    ) -> Optional[MarkerDetection]:
        """
        Detect markers within a segmentation mask region.
        
        Args:
            image: Full RGB/BGR image (H, W, 3)
            mask: Binary segmentation mask (H, W) with 255 for instance region
            prefer_aruco: Try ArUco first before QR (faster detection)
            
        Returns:
            MarkerDetection object if marker found, None otherwise
        """
        if image is None or mask is None:
            return None
        
        # Ensure image is BGR for OpenCV
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
        # Ensure mask is binary
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        # Extract ROI using bounding box for efficiency
        y_indices, x_indices = np.where(mask_binary > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            if self.debug:
                print(f"  [Instance {instance_id}] Empty mask, skipping")
            return None
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # Add small padding
        pad = 5
        y_min = max(0, y_min - pad)
        y_max = min(image.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(image.shape[1], x_max + pad)
        
        if self.debug:
            print(f"  [Instance {instance_id}] ROI: ({x_min}, {y_min}) to ({x_max}, {y_max}), size: {x_max-x_min}x{y_max-y_min}")
            if self.debug_folder:
                print(f"  [Instance {instance_id}] Debug folder: {self.debug_folder}")
            else:
                print(f"  [Instance {instance_id}] WARNING: debug_folder is not set, images will not be saved")
        
        # Extract ROI
        roi_image = image[y_min:y_max, x_min:x_max].copy()
        roi_mask = mask_binary[y_min:y_max, x_min:x_max]
        
        # IMPORTANT: Don't mask the image for ArUco detection!
        # ArUco's adaptive thresholding works better on the original image.
        # We use the mask only to verify the marker center is within the region.
        detection_image = roi_image.copy()
        
        # Save debug images if enabled
        if self.debug and self.debug_folder:
            print(f"  [Instance {instance_id}] Saving debug images...")
            # For debug, create a masked version to visualize what region we're checking
            masked_roi = roi_image.copy()
            masked_roi[roi_mask == 0] = 255
            self._save_debug_images(roi_image, roi_mask, masked_roi, instance_id)
        
        # Try detection in order of preference
        detections = []
        
        if prefer_aruco:
            if self.enable_aruco:
                aruco_result = self._detect_aruco(detection_image, roi_mask, x_min, y_min, instance_id)
                if aruco_result:
                    detections.append(aruco_result)
                    if self.debug:
                        print(f"  [Instance {instance_id}] ArUco detected: ID={aruco_result.marker_id}, conf={aruco_result.confidence:.2f}")
            if self.enable_qr:
                qr_result = self._detect_qr(detection_image, roi_mask, x_min, y_min, instance_id)
                if qr_result:
                    detections.append(qr_result)
                    if self.debug:
                        print(f"  [Instance {instance_id}] QR detected: ID={qr_result.marker_id}, conf={qr_result.confidence:.2f}")
        else:
            if self.enable_qr:
                qr_result = self._detect_qr(detection_image, roi_mask, x_min, y_min, instance_id)
                if qr_result:
                    detections.append(qr_result)
                    if self.debug:
                        print(f"  [Instance {instance_id}] QR detected: ID={qr_result.marker_id}, conf={qr_result.confidence:.2f}")
            if self.enable_aruco:
                aruco_result = self._detect_aruco(detection_image, roi_mask, x_min, y_min, instance_id)
                if aruco_result:
                    detections.append(aruco_result)
                    if self.debug:
                        print(f"  [Instance {instance_id}] ArUco detected: ID={aruco_result.marker_id}, conf={aruco_result.confidence:.2f}")
        
        # Return best detection (highest confidence)
        if detections:
            best_detection = max(detections, key=lambda d: d.confidence)
            if best_detection.confidence >= self.min_confidence:
                if self.debug:
                    print(f"  [Instance {instance_id}] ✓ Best detection: {best_detection.marker_type} ID={best_detection.marker_id}")
                return best_detection
            elif self.debug:
                print(f"  [Instance {instance_id}] ✗ Detection below confidence threshold: {best_detection.confidence:.2f} < {self.min_confidence}")
        elif self.debug:
            print(f"  [Instance {instance_id}] ✗ No markers detected")
        
        return None
    
    def _save_debug_images(self, roi_image, roi_mask, masked_roi, instance_id):
        """Save debug images for troubleshooting"""
        try:
            from pathlib import Path
            debug_path = Path(self.debug_folder)
            
            # Ensure debug folder exists
            debug_path.mkdir(parents=True, exist_ok=True)
            
            # Save original ROI
            roi_file = debug_path / f"instance_{instance_id}_{self.debug_counter:03d}_roi.jpg"
            success1 = cv2.imwrite(str(roi_file), roi_image)
            
            # Save mask
            mask_file = debug_path / f"instance_{instance_id}_{self.debug_counter:03d}_mask.jpg"
            success2 = cv2.imwrite(str(mask_file), roi_mask)
            
            # Save masked ROI (what we search for markers in)
            masked_file = debug_path / f"instance_{instance_id}_{self.debug_counter:03d}_masked.jpg"
            success3 = cv2.imwrite(str(masked_file), masked_roi)
            
            if success1 and success2 and success3:
                print(f"  [Debug] Saved debug images to {debug_path}")
            else:
                print(f"  [Debug] WARNING: Failed to save some images (roi:{success1}, mask:{success2}, masked:{success3})")
            
            self.debug_counter += 1
        except Exception as e:
            print(f"  [Debug] ERROR saving debug images: {e}")
            import traceback
            traceback.print_exc()
    
    def _detect_aruco(
        self,
        roi_image: np.ndarray,
        roi_mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        instance_id: Optional[int] = None,
        reject_multiple: bool = False
    ) -> Optional[MarkerDetection]:
        """
        Detect ArUco markers in ROI.
        
        Args:
            roi_image: ROI image (BGR)
            roi_mask: Binary mask for ROI
            offset_x: X offset to convert ROI coords to full image coords
            offset_y: Y offset to convert ROI coords to full image coords
            reject_multiple: If True, return None if multiple markers detected
            
        Returns:
            MarkerDetection if found, None otherwise
        """
        best_detection = None
        best_confidence = 0.0
        valid_markers_count = 0  # Track how many valid markers found in mask
        
        # Try each ArUco dictionary
        for dict_name, detector in self.aruco_detectors.items():
            try:
                # Detect markers
                corners, ids, rejected = detector.detectMarkers(roi_image)
                
                if ids is not None and len(ids) > 0:
                    # Check each detected marker
                    for i, marker_id in enumerate(ids):
                        marker_corners = corners[i][0]  # Shape: (4, 2)
                        
                        # Check if marker is within the mask region
                        center_x = int(np.mean(marker_corners[:, 0]))
                        center_y = int(np.mean(marker_corners[:, 1]))
                        
                        if (0 <= center_y < roi_mask.shape[0] and 
                            0 <= center_x < roi_mask.shape[1] and
                            roi_mask[center_y, center_x] > 0):
                            
                            valid_markers_count += 1
                            
                            # If rejecting multiple and we found more than one, return None
                            if reject_multiple and valid_markers_count > 1:
                                if self.debug and instance_id is not None:
                                    print(f"  [Instance {instance_id}] ✗ Multiple ArUco codes detected, rejecting")
                                return None
                            
                            # Calculate confidence based on corner clarity and size
                            # Larger markers with clear corners = higher confidence
                            area = cv2.contourArea(marker_corners)
                            perimeter = cv2.arcLength(marker_corners, True)
                            
                            # Normalize confidence (0-1)
                            # Scale confidence so that even small markers (30x30) get reasonable scores
                            # A 30x30 marker (area ~900) should get ~0.4 confidence
                            # A 50x50 marker (area ~2500) should get ~0.7 confidence
                            confidence = min(1.0, np.sqrt(area) / 70.0)
                            
                            if confidence > best_confidence:
                                # Adjust corners to full image coordinates
                                full_corners = marker_corners.copy()
                                full_corners[:, 0] += offset_x
                                full_corners[:, 1] += offset_y
                                
                                center = (
                                    float(center_x + offset_x),
                                    float(center_y + offset_y)
                                )
                                
                                best_detection = MarkerDetection(
                                    marker_type='aruco',
                                    marker_id=int(marker_id[0]),
                                    confidence=confidence,
                                    corners=full_corners,
                                    dict_type=dict_name,
                                    center=center
                                )
                                best_confidence = confidence
                                
                                # Early exit if we found a high-confidence marker (but only if not rejecting multiple)
                                if not reject_multiple and confidence > 0.9:
                                    return best_detection
            
            except Exception as e:
                # Skip this dictionary if detection fails
                continue
        
        return best_detection
    
    def _detect_qr(
        self,
        roi_image: np.ndarray,
        roi_mask: np.ndarray,
        offset_x: int,
        offset_y: int,
        instance_id: Optional[int] = None
    ) -> Optional[MarkerDetection]:
        """
        Detect QR codes in ROI.
        
        Args:
            roi_image: ROI image (BGR)
            roi_mask: Binary mask for ROI
            offset_x: X offset to convert ROI coords to full image coords
            offset_y: Y offset to convert ROI coords to full image coords
            
        Returns:
            MarkerDetection if found, None otherwise
        """
        if self.qr_detector is None:
            return None
        
        try:
            # Detect and decode QR code
            data, points, straight_qrcode = self.qr_detector.detectAndDecode(roi_image)
            
            if data and points is not None and len(points) > 0:
                # QR code found
                points = points[0]  # Shape: (4, 2)
                
                # Check if QR code center is within mask
                center_x = int(np.mean(points[:, 0]))
                center_y = int(np.mean(points[:, 1]))
                
                if (0 <= center_y < roi_mask.shape[0] and 
                    0 <= center_x < roi_mask.shape[1] and
                    roi_mask[center_y, center_x] > 0):
                    
                    # Calculate confidence based on QR code quality
                    # Successful decode = high confidence
                    confidence = 0.95  # QR codes either work or don't
                    
                    # Adjust corners to full image coordinates
                    full_corners = points.copy()
                    full_corners[:, 0] += offset_x
                    full_corners[:, 1] += offset_y
                    
                    center = (
                        float(center_x + offset_x),
                        float(center_y + offset_y)
                    )
                    
                    return MarkerDetection(
                        marker_type='qr',
                        marker_id=data,  # QR data as string
                        confidence=confidence,
                        corners=full_corners,
                        dict_type=None,
                        center=center
                    )
        
        except Exception as e:
            # QR detection failed
            pass
        
        return None
    
    def detect_in_annotations(
        self,
        image: np.ndarray,
        annotations: List[Dict[str, Any]]
    ) -> Dict[int, MarkerDetection]:
        """
        Detect markers in all annotations.
        
        Args:
            image: Full image (H, W, 3)
            annotations: List of annotation dicts with 'mask' field
            
        Returns:
            Dict mapping instance_id to MarkerDetection object
        """
        detections = {}
        
        for ann in annotations:
            if 'mask' not in ann:
                continue
            
            # Get instance ID
            instance_id = ann.get('mask_id', ann.get('instance_id', 0))
            if instance_id <= 0:
                continue
            
            # Try to detect marker in this mask
            detection = self.detect_in_mask(image, ann['mask'])
            
            if detection:
                detections[instance_id] = detection
                print(f"  Found {detection.marker_type} marker "
                      f"ID={detection.marker_id} "
                      f"(confidence={detection.confidence:.2f}) "
                      f"in instance {instance_id}")
        
        return detections
    
    def detect_aruco_in_bee_instances(
        self,
        image: np.ndarray,
        annotations: List[Dict[str, Any]],
        reject_multiple: bool = True
    ) -> Dict[int, MarkerDetection]:
        """
        Detect ArUco markers on full image, then match to bee instances.
        Rejects instances with multiple ArUco codes detected.
        
        Args:
            image: Full image (H, W, 3)
            annotations: List of annotation dicts
            reject_multiple: Reject instances with multiple ArUco codes (default: True)
            
        Returns:
            Dict mapping instance_id to MarkerDetection object
        """
        detections = {}
        stats = {
            'total_bees': 0,
            'total_markers': 0,
            'detected': 0,
            'rejected_multiple': 0,
            'rejected_none': 0
        }
        
        if self.debug:
            print("[MarkerDetector] Starting ArUco detection on full image...")
            print(f"[MarkerDetector] Input: {len(annotations)} total annotation(s)")
        
        # Always print basic info for debugging
        print(f"[ArUco Detection] Image shape: {image.shape if image is not None else 'None'}")
        print(f"[ArUco Detection] Total annotations: {len(annotations)}")
        
        # Filter bee annotations and build lookup structures
        bee_annotations = []
        for ann in annotations:
            if ann.get('category', 'bee') != 'bee':
                if self.debug:
                    print(f"  Skipping non-bee annotation: category={ann.get('category', 'bee')}")
                continue
            
            instance_id = ann.get('mask_id', ann.get('instance_id', 0))
            if instance_id <= 0:
                continue
            
            bbox = ann.get('bbox', None)
            if bbox is None or len(bbox) != 4:
                continue
            
            bee_annotations.append({
                'instance_id': instance_id,
                'bbox': bbox,
                'mask': ann.get('mask', None),
                'has_mask': 'mask' in ann and ann['mask'] is not None
            })
        
        stats['total_bees'] = len(bee_annotations)
        
        print(f"[ArUco Detection] Found {stats['total_bees']} bee annotation(s) to process")
        
        if stats['total_bees'] == 0:
            if self.debug:
                print("[MarkerDetector] No valid bee annotations found")
            return detections
        
        if self.debug:
            print(f"[MarkerDetector] Processing {stats['total_bees']} bee annotation(s)")
        
        # Step 1: Run ArUco detection ONCE on full image (4x4 dictionaries only)
        all_markers = []  # List of (marker_id, center, corners, dict_name)
        
        # Only use 4x4 dictionaries as specified
        dicts_to_try = ['4x4_50', '4x4_100', '4x4_250', '4x4_1000']
        
        if self.debug:
            print(f"[MarkerDetector] Available detectors: {list(self.aruco_detectors.keys())}")
            print(f"[MarkerDetector] Will try: {dicts_to_try}")
        
        print(f"[ArUco Detection] Available detectors: {list(self.aruco_detectors.keys())}")
        
        for dict_name in dicts_to_try:
            if dict_name not in self.aruco_detectors:
                if self.debug:
                    print(f"  ✗ {dict_name}: detector not available")
                continue
            
            detector = self.aruco_detectors[dict_name]
            corners, ids, rejected = detector.detectMarkers(image)
            
            if ids is not None and len(ids) > 0:
                if self.debug:
                    print(f"  {dict_name}: Found {len(ids)} marker(s) - IDs: {ids.flatten().tolist()}")
                
                for i, marker_id in enumerate(ids):
                    marker_corners = corners[i][0]  # Shape: (4, 2)
                    center = marker_corners.mean(axis=0)
                    
                    # Calculate confidence
                    area = cv2.contourArea(marker_corners)
                    confidence = min(1.0, np.sqrt(area) / 70.0)
                    
                    all_markers.append({
                        'marker_id': int(marker_id[0]),
                        'center': center,
                        'corners': marker_corners,
                        'dict_name': dict_name,
                        'confidence': confidence
                    })
        
        stats['total_markers'] = len(all_markers)
        
        # Deduplicate markers (same marker detected by multiple dictionaries)
        # Group by marker_id and keep only unique positions
        unique_markers = []
        seen_markers = {}  # marker_id -> list of (center, marker_dict)
        
        for marker in all_markers:
            marker_id = marker['marker_id']
            center = marker['center']
            
            if marker_id not in seen_markers:
                seen_markers[marker_id] = []
            
            # Check if we've seen this marker at this position before
            is_duplicate = False
            for existing_center, existing_marker in seen_markers[marker_id]:
                # If centers are within 5 pixels, it's the same marker
                dist = np.linalg.norm(center - existing_center)
                if dist < 5.0:
                    # It's a duplicate - keep the one with higher confidence
                    if marker['confidence'] > existing_marker['confidence']:
                        # Replace with higher confidence version
                        unique_markers.remove(existing_marker)
                        unique_markers.append(marker)
                        # Update the seen list
                        idx = seen_markers[marker_id].index((existing_center, existing_marker))
                        seen_markers[marker_id][idx] = (center, marker)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_markers.append(marker)
                seen_markers[marker_id].append((center, marker))
        
        if self.debug:
            print(f"\n[MarkerDetector] Detected {len(all_markers)} marker(s) total in full image")
            print(f"[MarkerDetector] After deduplication: {len(unique_markers)} unique marker(s)")
            unique_ids = sorted(set(m['marker_id'] for m in unique_markers))
            print(f"[MarkerDetector] Unique marker IDs: {unique_ids}")
            print(f"[MarkerDetector] Matching markers to {len(bee_annotations)} bee instance(s)...")
        else:
            # Always print summary even when not in debug mode
            if len(unique_markers) > 0:
                unique_ids = sorted(set(m['marker_id'] for m in unique_markers))
                print(f"[ArUco Detection] Detected {len(unique_markers)} unique marker(s) in full image: {unique_ids}")
        
        # Use deduplicated markers from here on
        all_markers = unique_markers
        stats['total_markers'] = len(all_markers)
        
        # Step 2: Match each marker to bee instances
        # Group markers by bee instance
        bee_markers = {}  # instance_id -> list of markers
        
        for marker in all_markers:
            cx, cy = marker['center']
            
            # Check which bee(s) this marker falls within
            for bee_ann in bee_annotations:
                instance_id = bee_ann['instance_id']
                bbox = bee_ann['bbox']
                x, y, w, h = bbox
                
                # Check if marker center is in bbox (with small tolerance for edge cases)
                if x <= cx <= (x + w) and y <= cy <= (y + h):
                    # If there's a segmentation mask, verify the marker is within it
                    if bee_ann['has_mask']:
                        mask = bee_ann['mask']
                        cy_int, cx_int = int(cy), int(cx)
                        
                        # Check if center is in mask
                        center_in_mask = (0 <= cy_int < mask.shape[0] and 
                                          0 <= cx_int < mask.shape[1] and 
                                          mask[cy_int, cx_int] > 0)
                        
                        # If center is not in mask, check if any corner is
                        # This handles edge cases where marker is at boundary
                        marker_in_instance = center_in_mask
                        if not center_in_mask:
                            corners = marker['corners']
                            for corner in corners:
                                corner_x, corner_y = int(corner[0]), int(corner[1])
                                if (0 <= corner_y < mask.shape[0] and 
                                    0 <= corner_x < mask.shape[1] and 
                                    mask[corner_y, corner_x] > 0):
                                    marker_in_instance = True
                                    break
                        
                        if marker_in_instance:
                            # Marker is within segmentation
                            if instance_id not in bee_markers:
                                bee_markers[instance_id] = []
                            bee_markers[instance_id].append(marker)
                            
                            if self.debug:
                                match_type = "center" if center_in_mask else "corner"
                                print(f"  ✓ Marker {marker['marker_id']} matched to Instance {instance_id} (in segmentation, via {match_type})")
                    else:
                        # No mask, just use bbox
                        if instance_id not in bee_markers:
                            bee_markers[instance_id] = []
                        bee_markers[instance_id].append(marker)
                        
                        if self.debug:
                            print(f"  ✓ Marker {marker['marker_id']} matched to Instance {instance_id} (in bbox)")
        
        # Step 3: Process matches and apply reject_multiple rule
        for instance_id, markers in bee_markers.items():
            if len(markers) == 0:
                continue
            elif len(markers) == 1:
                # Single marker - create MarkerDetection
                marker = markers[0]
                detections[instance_id] = MarkerDetection(
                    marker_type='aruco',
                    marker_id=marker['marker_id'],
                    confidence=marker['confidence'],
                    corners=marker['corners'],
                    dict_type=marker['dict_name'],
                    center=tuple(marker['center'])
                )
                stats['detected'] += 1
                
                if self.debug:
                    print(f"  [Instance {instance_id}] ✓ ArUco ID={marker['marker_id']} assigned (conf={marker['confidence']:.2f})")
            else:
                # Multiple markers in same bee
                if reject_multiple:
                    stats['rejected_multiple'] += 1
                    marker_ids = [m['marker_id'] for m in markers]
                    # Always print rejection info (not just in debug mode) to alert user
                    print(f"  [Instance {instance_id}] ✗ REJECTED: {len(markers)} markers detected (IDs: {marker_ids})")
                    if self.debug:
                        for m in markers:
                            print(f"    - ArUco {m['marker_id']} at ({m['center'][0]:.1f}, {m['center'][1]:.1f}), conf={m['confidence']:.2f}")
                else:
                    # Take the best (highest confidence)
                    best_marker = max(markers, key=lambda m: m['confidence'])
                    detections[instance_id] = MarkerDetection(
                        marker_type='aruco',
                        marker_id=best_marker['marker_id'],
                        confidence=best_marker['confidence'],
                        corners=best_marker['corners'],
                        dict_type=best_marker['dict_name'],
                        center=tuple(best_marker['center'])
                    )
                    stats['detected'] += 1
                    
                    if self.debug:
                        print(f"  [Instance {instance_id}] ✓ Best marker {best_marker['marker_id']} selected from {len(markers)} (conf={best_marker['confidence']:.2f})")
        
        # Count bees with no markers
        stats['rejected_none'] = stats['total_bees'] - stats['detected'] - stats['rejected_multiple']
        
        # Print summary
        if self.debug:
            print(f"\n[MarkerDetector] Detection complete:")
            print(f"  Total markers in image: {stats['total_markers']}")
            print(f"  Total bees: {stats['total_bees']}")
            print(f"  ArUco codes matched: {stats['detected']}")
            print(f"  Rejected (multiple codes): {stats['rejected_multiple']}")
            print(f"  Bees with no markers: {stats['rejected_none']}")
        
        return detections
