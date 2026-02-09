"""
Mask editing operations
"""

import numpy as np
import cv2
from scipy import ndimage


class MaskEditor:
    """Mask editing utilities"""
    
    @staticmethod
    def brush_stroke(mask, x1, y1, x2, y2, brush_size, value=255):
        """
        Apply brush stroke to mask
        
        Args:
            mask: Binary mask array
            x1, y1, x2, y2: Line endpoints
            brush_size: Brush diameter
            value: Value to paint (255 for add, 0 for erase)
            
        Returns:
            Modified mask
        """
        mask = mask.copy()
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 
                value, brush_size, lineType=cv2.LINE_AA)
        return mask
        
    @staticmethod
    def flood_fill(mask, x, y, value=255):
        """
        Flood fill from point
        
        Args:
            mask: Binary mask array
            x, y: Seed point
            value: Fill value
            
        Returns:
            Modified mask
        """
        mask = mask.copy()
        h, w = mask.shape
        seed_point = (int(x), int(y))
        
        if 0 <= seed_point[0] < w and 0 <= seed_point[1] < h:
            cv2.floodFill(mask, None, seed_point, value)
            
        return mask
        
    @staticmethod
    def dilate(mask, kernel_size=5):
        """Dilate mask"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
        
    @staticmethod
    def erode(mask, kernel_size=5):
        """Erode mask"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(mask, kernel, iterations=1)
        
    @staticmethod
    def smooth(mask, kernel_size=5):
        """Smooth mask edges"""
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        smoothed = cv2.filter2D(mask.astype(np.float32), -1, kernel)
        return (smoothed > 127).astype(np.uint8) * 255
        
    @staticmethod
    def remove_small_objects(mask, min_size=100):
        """Remove small connected components"""
        # Label connected components
        labeled, num_features = ndimage.label(mask > 0)
        
        # Calculate sizes
        sizes = ndimage.sum(mask > 0, labeled, range(num_features + 1))
        
        # Remove small objects
        mask_sizes = sizes < min_size
        remove_pixel = mask_sizes[labeled]
        labeled[remove_pixel] = 0
        
        return (labeled > 0).astype(np.uint8) * 255
        
    @staticmethod
    def fill_holes(mask):
        """Fill holes in mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill contours
        filled = mask.copy()
        for contour in contours:
            cv2.drawContours(filled, [contour], 0, 255, -1)
            
        return filled
        
    @staticmethod
    def combine_masks(masks, operation='union'):
        """
        Combine multiple masks
        
        Args:
            masks: List of binary masks
            operation: 'union', 'intersection', or 'difference'
            
        Returns:
            Combined mask
        """
        if not masks:
            return None
            
        result = masks[0].copy()
        
        for mask in masks[1:]:
            if operation == 'union':
                result = cv2.bitwise_or(result, mask)
            elif operation == 'intersection':
                result = cv2.bitwise_and(result, mask)
            elif operation == 'difference':
                result = cv2.bitwise_and(result, cv2.bitwise_not(mask))
                
        return result
        
    @staticmethod
    def mask_to_polygon(mask, epsilon=2.0):
        """
        Convert mask to polygon contour
        
        Args:
            mask: Binary mask
            epsilon: Approximation accuracy
            
        Returns:
            List of contour points
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
            
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = epsilon / 100.0 * cv2.arcLength(largest_contour, True)
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return polygon.reshape(-1, 2).tolist()
        
    @staticmethod
    def polygon_to_mask(polygon, shape):
        """
        Convert polygon to mask
        
        Args:
            polygon: List of (x, y) points
            shape: Mask shape (height, width)
            
        Returns:
            Binary mask
        """
        mask = np.zeros(shape, dtype=np.uint8)
        points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask
