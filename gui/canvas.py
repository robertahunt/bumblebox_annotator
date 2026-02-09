"""
Image canvas with zoom, pan, and annotation capabilities
"""

import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (QPixmap, QImage, QPen, QBrush, QColor, QPainter,
                        QTransform, QCursor, QPolygonF)
from pathlib import Path
import cv2


class ImageCanvas(QGraphicsView):
    """Image canvas with zoom, pan, and annotation capabilities"""
    
    point_clicked = pyqtSignal(int, int, bool)  # x, y, is_positive
    box_drawn = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2
    annotation_changed = pyqtSignal()
    masks_visibility_changed = pyqtSignal(bool)  # Signal when mask visibility changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Setup view
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Image
        self.image_item = None
        self.current_image = None
        self.image_path = None
        
        # Tool state
        self.current_tool = 'pan'
        self.brush_size = 10
        self.mask_opacity = 128  # Default opacity for masks (0-255)
        
        # Annotations
        self.masks = []  # List of mask arrays
        self.mask_items = []  # List of QGraphicsPixmapItem for visualization
        self.mask_colors = []  # List of colors for each mask
        self.mask_ids = []  # List of persistent IDs for each mask
        self.next_mask_id = 1  # Counter for assigning unique IDs
        self.selected_mask_idx = -1
        self.masks_visible = True  # Track mask visibility state
        
        # SAM2 prompts
        self.positive_points = []  # List of (x, y) tuples
        self.negative_points = []  # List of (x, y) tuples
        self.prompt_items = []  # List of graphics items for visualization
        self.active_sam2_mask_idx = -1  # Index of mask being edited with SAM2 prompts
        
        # Selection border
        self.selection_border_item = None
        
        # Drawing state
        self.is_drawing = False
        self.drawing_start = None
        self.temp_item = None
        
        # Undo/redo
        self.history = []
        self.history_idx = -1
        
        self.zoom_factor = 1.0
        
        # Enable focus to receive keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            # Hide masks while spacebar is held
            if self.masks_visible:
                self.hide_masks()
        elif event.key() == Qt.Key.Key_F and not event.isAutoRepeat():
            # Fit image to window (reset zoom)
            self.fit_to_window()
        else:
            super().keyPressEvent(event)
    
    def keyReleaseEvent(self, event):
        """Handle key release events"""
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            # Show masks when spacebar is released
            if not self.masks_visible:
                self.show_masks()
        else:
            super().keyReleaseEvent(event)
    
    def hide_masks(self):
        """Temporarily hide all masks"""
        self.masks_visible = False
        for item in self.mask_items:
            item.setVisible(False)
        # Also hide selection border
        if self.selection_border_item:
            self.selection_border_item.setVisible(False)
        self.masks_visibility_changed.emit(False)
    
    def show_masks(self):
        """Show all masks again"""
        self.masks_visible = True
        for item in self.mask_items:
            item.setVisible(True)
        # Show selection border if there's a selection
        if self.selection_border_item and self.selected_mask_idx >= 0:
            self.selection_border_item.setVisible(True)
        self.masks_visibility_changed.emit(True)
        
    def load_image(self, image_path_or_array):
        """Load an image file or numpy array"""
        # Check if input is a numpy array
        if isinstance(image_path_or_array, np.ndarray):
            # Already a numpy array (in RGB format)
            self.current_image = image_path_or_array
            self.image_path = None
        else:
            # It's a file path
            self.image_path = Path(image_path_or_array)
            
            # Load with OpenCV for processing
            self.current_image = cv2.imread(str(image_path_or_array))
            
            # Check if image was loaded successfully
            if self.current_image is None or self.current_image.size == 0:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(None, "Error", f"Failed to load image: {image_path_or_array}")
                return
            
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Convert to QPixmap for display
        h, w, ch = self.current_image.shape
        bytes_per_line = ch * w
        q_image = QImage(self.current_image.data, w, h, bytes_per_line, 
                        QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Clear scene
        self.scene.clear()
        self.masks = []
        self.mask_ids = []
        self.next_mask_id = 1
        self.mask_items = []
        self.mask_colors = []
        self.positive_points = []
        self.negative_points = []
        self.prompt_items = []
        self.active_sam2_mask_idx = -1
        self.selection_border_item = None
        
        # Add image to scene
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)
        
        # Fit to window - use QTimer to ensure widget is properly sized
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self.fit_to_window)
    
    def clear_image(self):
        """Clear the current image and all annotations"""
        self.scene.clear()
        self.current_image = None
        self.image_path = None
        self.image_item = None
        self.masks = []
        self.mask_ids = []
        self.next_mask_id = 1
        self.mask_items = []
        self.mask_colors = []
        self.positive_points = []
        self.negative_points = []
        self.prompt_items = []
        self.active_sam2_mask_idx = -1
        self.selection_border_item = None
        
    def fit_to_window(self):
        """Fit image to window"""
        if self.image_item:
            self.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_factor = 1.0
            
    def resizeEvent(self, event):
        """Handle resize event to fit image to new window size"""
        super().resizeEvent(event)
        if self.image_item and self.zoom_factor == 1.0:
            # Only auto-fit if not zoomed in/out
            self.fit_to_window()
            
    def zoom_in(self):
        """Zoom in"""
        self.scale(1.1, 1.1)
        self.zoom_factor *= 1.1
        
    def zoom_out(self):
        """Zoom out with limit to prevent zooming smaller than fit-to-window"""
        if not self.image_item:
            return
        
        # Get current transform scale
        current_transform = self.transform()
        current_scale = current_transform.m11()
        
        # Calculate what scale we'd need to fit the whole image
        viewport_rect = self.viewport().rect()
        image_rect = self.image_item.boundingRect()
        
        scale_x = viewport_rect.width() / image_rect.width()
        scale_y = viewport_rect.height() / image_rect.height()
        min_scale = min(scale_x, scale_y)
        
        # Only zoom out if we're more zoomed in than fit-to-window
        if current_scale > min_scale * 0.95:  # 0.95 adds small tolerance
            self.scale(1/1.1, 1/1.1)
            self.zoom_factor /= 1.1
        
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()
            
    def set_tool(self, tool_name):
        """Set current tool"""
        self.current_tool = tool_name
        
        if tool_name == 'pan':
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            
            if tool_name == 'sam2_prompt':
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif tool_name == 'brush':
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif tool_name == 'eraser':
                self.setCursor(Qt.CursorShape.CrossCursor)
            elif tool_name == 'sam2_box':
                self.setCursor(Qt.CursorShape.CrossCursor)
                
    def mousePressEvent(self, event):
        """Handle mouse press"""
        if self.current_image is None:
            return
            
        # Get scene position
        scene_pos = self.mapToScene(event.pos())
        
        if self.current_tool == 'sam2_prompt':
            # SAM2 point prompting
            x, y = int(scene_pos.x()), int(scene_pos.y())
            is_positive = event.button() == Qt.MouseButton.LeftButton
            
            # Store and visualize the point
            if is_positive:
                self.positive_points.append((x, y))
            else:
                self.negative_points.append((x, y))
            self.add_prompt_marker(x, y, is_positive)
            
            # Emit signal for SAM2 prediction
            self.point_clicked.emit(x, y, is_positive)
            
        elif self.current_tool == 'sam2_box':
            # Start drawing box
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing = True
                self.drawing_start = scene_pos
                
        elif self.current_tool == 'brush' or self.current_tool == 'eraser':
            # Start drawing/erasing
            if event.button() == Qt.MouseButton.LeftButton:
                self.is_drawing = True
                self.last_point = scene_pos
                
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move"""
        if self.current_image is None:
            return
            
        scene_pos = self.mapToScene(event.pos())
        
        if self.is_drawing:
            if self.current_tool == 'sam2_box':
                # Update box preview
                if self.temp_item:
                    self.scene.removeItem(self.temp_item)
                    
                from PyQt6.QtWidgets import QGraphicsRectItem
                rect = QRectF(self.drawing_start, scene_pos).normalized()
                self.temp_item = QGraphicsRectItem(rect)
                self.temp_item.setPen(QPen(QColor(0, 255, 0), 2))
                self.scene.addItem(self.temp_item)
                
            elif self.current_tool == 'brush' or self.current_tool == 'eraser':
                # Draw/erase on mask
                self.draw_on_mask(self.last_point, scene_pos)
                self.last_point = scene_pos
                
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if self.current_image is None:
            return
            
        scene_pos = self.mapToScene(event.pos())
        
        if self.is_drawing:
            if self.current_tool == 'sam2_box':
                # Emit box drawn signal
                if self.temp_item:
                    self.scene.removeItem(self.temp_item)
                    self.temp_item = None
                    
                x1, y1 = int(self.drawing_start.x()), int(self.drawing_start.y())
                x2, y2 = int(scene_pos.x()), int(scene_pos.y())
                self.box_drawn.emit(x1, y1, x2, y2)
                
            self.is_drawing = False
        else:
            super().mouseReleaseEvent(event)
            
    def draw_on_mask(self, start_pos, end_pos):
        """Draw on the current mask with brush/eraser"""
        if self.selected_mask_idx < 0 or self.selected_mask_idx >= len(self.masks):
            # Create new mask if none selected
            h, w = self.current_image.shape[:2]
            new_mask = np.zeros((h, w), dtype=np.uint8)
            self.masks.append(new_mask)
            self.selected_mask_idx = len(self.masks) - 1
            
        mask = self.masks[self.selected_mask_idx]
        
        # Convert to image coordinates
        x1, y1 = int(start_pos.x()), int(start_pos.y())
        x2, y2 = int(end_pos.x()), int(end_pos.y())
        
        # Draw line on mask
        value = 255 if self.current_tool == 'brush' else 0
        cv2.line(mask, (x1, y1), (x2, y2), value, self.brush_size)
        
        # Update visualization with highlighting
        self.update_mask_visualization(self.selected_mask_idx)
        self.annotation_changed.emit()
        
    def add_mask(self, mask, mask_id=None):
        """Add a new mask from SAM2 prediction"""
        if isinstance(mask, np.ndarray):
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)
            self.masks.append(mask)
            
            # Assign or use provided mask ID
            if mask_id is None:
                mask_id = self.next_mask_id
                self.next_mask_id += 1
            else:
                # Update next_mask_id if necessary to avoid conflicts
                if mask_id >= self.next_mask_id:
                    self.next_mask_id = mask_id + 1
            
            self.mask_ids.append(mask_id)
            self.add_mask_visualization(mask)
            self.selected_mask_idx = len(self.masks) - 1
            
    def add_mask_visualization(self, mask):
        """Add mask visualization overlay"""
        # Create colored overlay
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Random color for this mask
        color = np.random.randint(0, 255, 3)
        self.mask_colors.append(color)
        overlay[mask > 0] = [color[0], color[1], color[2], self.mask_opacity]
        
        # Convert to QPixmap
        q_image = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Add to scene
        mask_item = QGraphicsPixmapItem(pixmap)
        mask_item.setZValue(1)  # Above image
        mask_item.setVisible(self.masks_visible)  # Respect current visibility state
        self.scene.addItem(mask_item)
        self.mask_items.append(mask_item)
        
    def update_mask_visualization(self, idx):
        """Update visualization for a specific mask"""
        if idx < len(self.mask_items):
            # Remove old item
            old_item = self.mask_items[idx]
            self.scene.removeItem(old_item)
            
            # Create new item
            mask = self.masks[idx]
            h, w = mask.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Use stored color if available
            if idx < len(self.mask_colors):
                color = self.mask_colors[idx]
            else:
                color = np.random.randint(0, 255, 3)
                self.mask_colors.append(color)
            
            # Apply highlighting if this is the selected instance
            # Boost opacity for selected mask (add 72 to base opacity, capped at 255)
            alpha = min(self.mask_opacity + 72, 255) if idx == self.selected_mask_idx else self.mask_opacity
            overlay[mask > 0] = [color[0], color[1], color[2], alpha]
            
            q_image = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(q_image)
            
            mask_item = QGraphicsPixmapItem(pixmap)
            mask_item.setZValue(1)
            mask_item.setVisible(self.masks_visible)  # Respect current visibility state
            self.scene.addItem(mask_item)
            self.mask_items[idx] = mask_item
            
            # Add border around selected instance
            if idx == self.selected_mask_idx:
                self.add_selection_border(mask)
            
    def set_annotations(self, annotations):
        """Load annotations for current frame"""
        # Clear existing masks
        for item in self.mask_items:
            self.scene.removeItem(item)
        self.masks = []
        self.mask_items = []
        self.mask_colors = []  # Reset colors for consistent visualization
        self.mask_ids = []
        self.next_mask_id = 1
        
        # Clear selection border
        if self.selection_border_item:
            self.scene.removeItem(self.selection_border_item)
            self.selection_border_item = None
        
        # Reset selection
        self.selected_mask_idx = -1
        self.active_sam2_mask_idx = -1
        
        # Load masks from annotations
        for ann in annotations:
            if 'mask' in ann:
                mask = ann['mask']
                if isinstance(mask, np.ndarray):
                    # Load mask with its persistent ID if available
                    mask_id = ann.get('mask_id', None)
                    self.add_mask(mask, mask_id)
                    
    def get_annotations(self):
        """Get current annotations"""
        annotations = []
        for i, mask in enumerate(self.masks):
            ann = {
                'mask': mask,
                'label': f'instance_{i}',
                'area': int(np.sum(mask > 0)),
                'mask_id': self.mask_ids[i] if i < len(self.mask_ids) else i+1
            }
            annotations.append(ann)
        return annotations
        
    def set_selected_instance(self, idx):
        """Set selected instance and zoom to it"""
        self.selected_mask_idx = idx
        
        # Zoom to the selected instance
        if 0 <= idx < len(self.masks):
            self.zoom_to_instance(idx)
    
    def get_mask_bbox(self, mask):
        """
        Get bounding box of a mask
        
        Args:
            mask: numpy array representing the mask
            
        Returns:
            tuple: (x, y, width, height) or None if mask is empty
        """
        if mask is None or not np.any(mask):
            return None
        
        # Find coordinates where mask is non-zero
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        
        # Get min/max coordinates (note: coords are in [row, col] format)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        return (int(x_min), int(y_min), int(width), int(height))
    
    def zoom_to_rect(self, x, y, width, height, padding=50):
        """
        Zoom and center view on a specific rectangle
        
        Args:
            x, y: Top-left corner of rectangle
            width, height: Size of rectangle
            padding: Extra space around the rectangle (in pixels)
        """
        if not self.image_item:
            return
        
        # Add padding
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        width_padded = width + 2 * padding
        height_padded = height + 2 * padding
        
        # Create QRectF for the region
        rect = QRectF(x_padded, y_padded, width_padded, height_padded)
        
        # Fit the view to this rectangle
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        
        # Update zoom factor based on the transform
        transform = self.transform()
        self.zoom_factor = transform.m11()  # Get scale factor from transform matrix
    
    def zoom_to_instance(self, idx, padding=50):
        """
        Zoom to a specific instance
        
        Args:
            idx: Index of the instance/mask to zoom to
            padding: Extra space around the instance (in pixels)
        """
        if not (0 <= idx < len(self.masks)):
            return
        
        mask = self.masks[idx]
        bbox = self.get_mask_bbox(mask)
        
        if bbox is None:
            return
        
        x, y, width, height = bbox
        self.zoom_to_rect(x, y, width, height, padding)
    
    def highlight_instance(self, idx):
        """Highlight a specific instance by making others semi-transparent"""
        if idx < 0 or idx >= len(self.masks):
            return
            
        # Update all mask visualizations with different opacity
        for i in range(len(self.masks)):
            self.update_mask_visualization(i)
            
    def refresh_all_visualizations(self):
        """Refresh all mask visualizations with current selection highlighting"""
        for i in range(len(self.masks)):
            self.update_mask_visualization(i)
            
    def add_selection_border(self, mask):
        """Add a border around the selected mask"""
        # Remove previous border if it exists
        if self.selection_border_item:
            self.scene.removeItem(self.selection_border_item)
            self.selection_border_item = None
        
        # Find contours of the mask
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return
        
        # Draw contours as polygon
        from PyQt6.QtWidgets import QGraphicsPathItem
        from PyQt6.QtGui import QPainterPath
        
        path = QPainterPath()
        for contour in contours:
            if len(contour) > 0:
                # Start at first point
                first_point = contour[0][0]
                path.moveTo(float(first_point[0]), float(first_point[1]))
                
                # Draw lines to other points
                for point in contour[1:]:
                    pt = point[0]
                    path.lineTo(float(pt[0]), float(pt[1]))
                
                # Close the path
                path.closeSubpath()
        
        # Create graphics item with thick yellow border
        border_item = QGraphicsPathItem(path)
        border_item.setPen(QPen(QColor(255, 255, 0), 3))  # Yellow, 3px thick
        border_item.setZValue(10)  # Above masks
        self.scene.addItem(border_item)
        self.selection_border_item = border_item
        
    def resolve_all_overlaps(self):
        """Resolve all overlapping masks by priority (later instances win)"""
        if len(self.masks) < 2:
            return 0  # No overlaps possible
        
        overlap_count = 0
        # Process masks in reverse order so later instances have priority
        for i in range(len(self.masks) - 1, 0, -1):
            current_mask = self.masks[i]
            # Remove current mask pixels from all previous masks
            for j in range(i):
                overlap_pixels = np.logical_and(self.masks[j] > 0, current_mask > 0)
                if np.any(overlap_pixels):
                    overlap_count += np.sum(overlap_pixels)
                    self.masks[j][overlap_pixels] = 0
                    self.update_mask_visualization(j)
        
        return overlap_count
    
    def clear_selected_instance(self):
        """Clear the mask for the selected instance (keep the instance, just zero the mask)"""
        if 0 <= self.selected_mask_idx < len(self.masks):
            # Zero out the mask
            self.masks[self.selected_mask_idx].fill(0)
            
            # Update visualization
            self.update_mask_visualization(self.selected_mask_idx)
            
            # Clear prompt points (visual markers only, don't delete masks)
            for item in self.prompt_items:
                self.scene.removeItem(item)
            self.positive_points = []
            self.negative_points = []
            self.prompt_items = []
            
            # Reset the active SAM2 mask index so it doesn't interfere
            self.active_sam2_mask_idx = -1
            
            # Keep the instance selected so user can re-annotate it
            self.annotation_changed.emit()
    
    def delete_selected_instance(self):
        """Delete the currently selected instance"""
        if 0 <= self.selected_mask_idx < len(self.masks):
            # Remove from lists
            del self.masks[self.selected_mask_idx]
            
            # Remove visual item
            if self.selected_mask_idx < len(self.mask_items):
                item = self.mask_items[self.selected_mask_idx]
                self.scene.removeItem(item)
                del self.mask_items[self.selected_mask_idx]
            
            # Remove color
            if self.selected_mask_idx < len(self.mask_colors):
                del self.mask_colors[self.selected_mask_idx]
            
            # Remove mask ID (preserve persistent IDs by not reusing them)
            if self.selected_mask_idx < len(self.mask_ids):
                del self.mask_ids[self.selected_mask_idx]
            
            # Remove border if it exists
            if self.selection_border_item:
                self.scene.removeItem(self.selection_border_item)
                self.selection_border_item = None
            
            # Reset active_sam2_mask_idx if it was pointing to this instance
            if self.active_sam2_mask_idx == self.selected_mask_idx:
                self.active_sam2_mask_idx = -1
            elif self.active_sam2_mask_idx > self.selected_mask_idx:
                # Shift down if pointing to later instance
                self.active_sam2_mask_idx -= 1
            
            # Update selection
            if len(self.masks) == 0:
                # No more instances
                self.selected_mask_idx = -1
                self.active_sam2_mask_idx = -1
            elif self.selected_mask_idx >= len(self.masks):
                self.selected_mask_idx = len(self.masks) - 1
                
            self.annotation_changed.emit()
        
    def undo(self):
        """Undo last action"""
        # TODO: Implement proper undo/redo
        pass
        
    def redo(self):
        """Redo last action"""
        # TODO: Implement proper undo/redo
        pass
        
    def set_brush_size(self, size):
        """Set brush size"""
        self.brush_size = size
    
    def set_mask_opacity(self, opacity):
        """Set mask opacity and refresh all visualizations"""
        self.mask_opacity = opacity
        # Update all mask visualizations
        for i in range(len(self.masks)):
            self.update_mask_visualization(i)
        
    def add_prompt_marker(self, x, y, is_positive):
        """Add a visual marker for SAM2 prompt point"""
        from PyQt6.QtWidgets import QGraphicsEllipseItem
        
        # Create circular marker
        radius = 5
        color = QColor(0, 255, 0) if is_positive else QColor(255, 0, 0)
        
        marker = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
        marker.setPen(QPen(color, 2))
        marker.setBrush(QBrush(color))
        marker.setZValue(10)  # Above everything
        
        self.scene.addItem(marker)
        self.prompt_items.append(marker)
        
    def clear_prompt_points(self):
        """Clear all SAM2 prompt points and remove active mask (only if it's a new temporary mask)"""
        # Remove visual markers
        for item in self.prompt_items:
            self.scene.removeItem(item)
        
        # Clear storage
        self.positive_points = []
        self.negative_points = []
        self.prompt_items = []
        
        # Remove the active SAM2 mask ONLY if it's not a selected instance
        # (i.e., it's a temporary mask being built, not an existing one being edited)
        if 0 <= self.active_sam2_mask_idx < len(self.masks):
            # Only delete if it's NOT the selected instance (selected instances should be preserved)
            if self.active_sam2_mask_idx != self.selected_mask_idx:
                # Remove visual item
                if self.active_sam2_mask_idx < len(self.mask_items):
                    item = self.mask_items[self.active_sam2_mask_idx]
                    self.scene.removeItem(item)
                    del self.mask_items[self.active_sam2_mask_idx]
                
                # Remove from lists
                del self.masks[self.active_sam2_mask_idx]
                if self.active_sam2_mask_idx < len(self.mask_colors):
                    del self.mask_colors[self.active_sam2_mask_idx]
                    
                # Adjust selected_mask_idx if needed
                if self.selected_mask_idx > self.active_sam2_mask_idx:
                    self.selected_mask_idx -= 1
        
        # Remove selection border
        if self.selection_border_item:
            self.scene.removeItem(self.selection_border_item)
            self.selection_border_item = None
        
        # Reset active mask index
        self.active_sam2_mask_idx = -1
        
    def start_new_sam2_instance(self):
        """Start annotating a new SAM2 instance"""
        # Clear prompts and reset for new instance
        for item in self.prompt_items:
            self.scene.removeItem(item)
        
        self.positive_points = []
        self.negative_points = []
        self.prompt_items = []
        self.active_sam2_mask_idx = -1
        self.selected_mask_idx = -1
        
        # Remove selection border
        if self.selection_border_item:
            self.scene.removeItem(self.selection_border_item)
            self.selection_border_item = None
        
    def get_prompt_points(self):
        """Get all accumulated prompt points for SAM2"""
        return {
            'positive': self.positive_points.copy(),
            'negative': self.negative_points.copy()
        }
