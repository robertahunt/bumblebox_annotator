"""
Image canvas with zoom, pan, and annotation capabilities
"""

import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import (QPixmap, QImage, QPen, QBrush, QColor, QPainter,
                        QTransform, QCursor, QPolygonF, QFont)
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
        
        # Annotations - Using integer segmentation format
        self.combined_mask = None  # Single H×W int32 array with instance IDs
        self.mask_items = []  # List of QGraphicsPixmapItem for visualization
        self.mask_colors = {}  # Dict mapping instance_id -> (r,g,b) color
        self.next_mask_id = 1  # Counter for assigning unique IDs
        self.selected_mask_idx = -1
        self.masks_visible = True  # Track mask visibility state
        self.label_items = []  # List of QGraphicsTextItem for instance labels
        self.labels_visible = False  # Track label visibility state
        
        # Editing isolation - temporary mask for current edit
        self.editing_mask = None  # Binary H×W mask for instance being edited
        self.editing_instance_id = -1  # ID of instance being edited
        self.editing_mask_item = None  # Visualization item for editing mask
        self._editing_overlay_cache = None  # Cached RGBA overlay for editing mask
        self._is_actively_drawing = False  # Flag to skip expensive updates during drawing
        
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
        
        # Inference box (for box inference mode)
        self.inference_box_rect = None  # QRectF or None
        self.inference_box_item = None  # QGraphicsRectItem
        self.inference_box_handles = []  # Corner/edge handle items
        self.dragging_box = False
        self.dragging_handle = None  # 'tl', 'tr', 'bl', 'br', 'move'
        self.drag_offset = None
        
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
        # Hide editing mask if present
        if self.editing_mask_item:
            self.editing_mask_item.setVisible(False)
        # Also hide selection border
        if self.selection_border_item:
            self.selection_border_item.setVisible(False)
        self.masks_visibility_changed.emit(False)
    
    def show_masks(self):
        """Show all masks again"""
        self.masks_visible = True
        for item in self.mask_items:
            item.setVisible(True)
        # Show editing mask if present
        if self.editing_mask_item:
            self.editing_mask_item.setVisible(True)
        # Show selection border if there's a selection
        if self.selection_border_item and self.selected_mask_idx >= 0:
            self.selection_border_item.setVisible(True)
        self.masks_visibility_changed.emit(True)
        
    def load_image(self, image_path_or_array):
        """Load an image file or numpy array"""
        # Explicitly clear old resources before loading new image
        if self.image_item:
            self.scene.removeItem(self.image_item)
            self.image_item = None
        
        # Clear all visualization items
        for item in self.mask_items:
            self.scene.removeItem(item)
        self.mask_items.clear()
        
        for item in self.label_items:
            self.scene.removeItem(item)
        self.label_items.clear()
        
        for item in self.prompt_items:
            self.scene.removeItem(item)
        self.prompt_items.clear()
        
        if self.selection_border_item:
            self.scene.removeItem(self.selection_border_item)
            self.selection_border_item = None
        
        if self.editing_mask_item:
            self.scene.removeItem(self.editing_mask_item)
            self.editing_mask_item = None
        
        # Clear cached overlays to free memory
        self._cached_overlay = None
        self._editing_overlay_cache = None
        
        # Delete old image data
        old_image = self.current_image
        self.current_image = None
        del old_image
        
        # Check if input is a numpy array
        if isinstance(image_path_or_array, np.ndarray):
            # Already a numpy array (grayscale H×W or RGB H×W×3)
            self.current_image = image_path_or_array
            self.image_path = None
        else:
            # It's a file path
            self.image_path = Path(image_path_or_array)
            
            # Load as grayscale to save memory (3× reduction)
            self.current_image = cv2.imread(str(image_path_or_array), cv2.IMREAD_GRAYSCALE)
            
            # Check if image was loaded successfully
            if self.current_image is None or self.current_image.size == 0:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(None, "Error", f"Failed to load image: {image_path_or_array}")
                return
        
        # Convert to QPixmap for display
        # If grayscale (H×W), convert to RGB (H×W×3) for display only
        if len(self.current_image.shape) == 2:
            # Grayscale: convert to RGB for display
            h, w = self.current_image.shape
            display_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
            bytes_per_line = 3 * w
            # Convert display image to bytes and immediately create QImage, then pixmap
            # Using tobytes() creates a copy, so QImage doesn't reference display_image
            pixmap = QPixmap.fromImage(
                QImage(display_image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            )
            # Explicitly delete temporary display image
            del display_image
        else:
            # Already RGB
            h, w, ch = self.current_image.shape
            bytes_per_line = ch * w
            pixmap = QPixmap.fromImage(
                QImage(self.current_image.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
            )
        
        # Clear remaining scene items
        self.scene.clear()
        self.combined_mask = None
        self.next_mask_id = 1
        self.mask_colors = {}
        self.positive_points = []
        self.negative_points = []
        self.active_sam2_mask_idx = -1
        
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
        self.combined_mask = None  # Single H×W array with integer instance IDs (0=background)
        self.next_mask_id = 1
        self.mask_items = []  # QGraphicsPixmapItems for visualization 
        self.mask_colors = {}  # Dict mapping instance_id -> (r,g,b) color
        self.label_items = []  # Clear instance label items
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
            elif tool_name == 'inference_box':
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
                
        elif self.current_tool == 'inference_box':
            # Check if clicking on inference box or its handles
            if event.button() == Qt.MouseButton.LeftButton:
                if self.inference_box_rect:
                    # Check if clicking on a handle
                    handle = self._get_handle_at_pos(scene_pos)
                    if handle:
                        self.dragging_box = True
                        self.dragging_handle = handle
                        self.drag_offset = scene_pos
                    elif self.inference_box_rect.contains(scene_pos):
                        # Clicking inside box - move it
                        self.dragging_box = True
                        self.dragging_handle = 'move'
                        self.drag_offset = QPointF(scene_pos.x() - self.inference_box_rect.x(),
                                                   scene_pos.y() - self.inference_box_rect.y())
                    else:
                        # Start drawing new box
                        self.is_drawing = True
                        self.drawing_start = scene_pos
                        self.clear_inference_box()
                else:
                    # Start drawing new box
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
                
            elif self.current_tool == 'inference_box':
                # Update inference box preview
                if self.temp_item:
                    self.scene.removeItem(self.temp_item)
                    
                from PyQt6.QtWidgets import QGraphicsRectItem
                rect = QRectF(self.drawing_start, scene_pos).normalized()
                self.temp_item = QGraphicsRectItem(rect)
                self.temp_item.setPen(QPen(QColor(255, 165, 0), 3))  # Orange
                self.temp_item.setZValue(1000)  # Draw on top
                self.scene.addItem(self.temp_item)
                
            elif self.current_tool == 'brush' or self.current_tool == 'eraser':
                # Draw/erase on mask
                self._is_actively_drawing = True
                self.draw_on_mask(self.last_point, scene_pos)
                self.last_point = scene_pos
                
        elif self.dragging_box and self.current_tool == 'inference_box':
            # Handle dragging/resizing inference box
            if self.dragging_handle == 'move':
                # Move entire box
                new_x = scene_pos.x() - self.drag_offset.x()
                new_y = scene_pos.y() - self.drag_offset.y()
                self.inference_box_rect.moveTo(new_x, new_y)
                self._update_inference_box_display()
            elif self.dragging_handle in ['tl', 'tr', 'bl', 'br']:
                # Resize from corner
                rect = self.inference_box_rect
                if self.dragging_handle == 'tl':
                    rect.setTopLeft(scene_pos)
                elif self.dragging_handle == 'tr':
                    rect.setTopRight(scene_pos)
                elif self.dragging_handle == 'bl':
                    rect.setBottomLeft(scene_pos)
                elif self.dragging_handle == 'br':
                    rect.setBottomRight(scene_pos)
                self.inference_box_rect = rect.normalized()
                self._update_inference_box_display()
                
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
                
            elif self.current_tool == 'inference_box':
                # Finish drawing inference box
                if self.temp_item:
                    self.scene.removeItem(self.temp_item)
                    self.temp_item = None
                    
                # Store the box
                rect = QRectF(self.drawing_start, scene_pos).normalized()
                self.set_inference_box(rect)
                self.box_drawn.emit(int(rect.x()), int(rect.y()),
                                   int(rect.right()), int(rect.bottom()))
                
            elif self.current_tool == 'brush' or self.current_tool == 'eraser':
                # Finish drawing - update border now
                self._is_actively_drawing = False
                if self.editing_mask is not None and self.selected_mask_idx == self.editing_instance_id:
                    self.add_selection_border(self.editing_mask)
                
            self.is_drawing = False
        elif self.dragging_box:
            # Finish dragging inference box
            self.dragging_box = False
            self.dragging_handle = None
            self.drag_offset = None
        else:
            super().mouseReleaseEvent(event)
    
    def set_inference_box(self, rect):
        """Set the inference box rectangle and display it with handles"""
        self.inference_box_rect = rect
        self._update_inference_box_display()
    
    def clear_inference_box(self):
        """Remove the inference box from display"""
        if self.inference_box_item:
            self.scene.removeItem(self.inference_box_item)
            self.inference_box_item = None
        for handle in self.inference_box_handles:
            self.scene.removeItem(handle)
        self.inference_box_handles = []
        self.inference_box_rect = None
    
    def get_inference_box(self):
        """Get the current inference box coordinates as (x1, y1, x2, y2)"""
        if self.inference_box_rect:
            return (int(self.inference_box_rect.x()), int(self.inference_box_rect.y()),
                    int(self.inference_box_rect.right()), int(self.inference_box_rect.bottom()))
        return None
    
    def _update_inference_box_display(self):
        """Update the visual display of the inference box with handles"""
        # Remove old graphics items
        if self.inference_box_item:
            self.scene.removeItem(self.inference_box_item)
        for handle in self.inference_box_handles:
            self.scene.removeItem(handle)
        self.inference_box_handles = []
        
        if not self.inference_box_rect:
            return
        
        from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsEllipseItem
        
        # Draw main box
        self.inference_box_item = QGraphicsRectItem(self.inference_box_rect)
        self.inference_box_item.setPen(QPen(QColor(255, 165, 0), 3))  # Orange
        self.inference_box_item.setZValue(1000)
        self.scene.addItem(self.inference_box_item)
        
        # Draw corner handles
        handle_size = 10
        corners = [
            ('tl', self.inference_box_rect.topLeft()),
            ('tr', self.inference_box_rect.topRight()),
            ('bl', self.inference_box_rect.bottomLeft()),
            ('br', self.inference_box_rect.bottomRight())
        ]
        
        for corner_name, point in corners:
            handle = QGraphicsEllipseItem(
                point.x() - handle_size/2, point.y() - handle_size/2,
                handle_size, handle_size
            )
            handle.setPen(QPen(QColor(255, 165, 0), 2))
            handle.setBrush(QBrush(QColor(255, 255, 255)))
            handle.setZValue(1001)
            handle.setData(0, corner_name)  # Store corner name in item data
            self.scene.addItem(handle)
            self.inference_box_handles.append(handle)
    
    def _get_handle_at_pos(self, pos):
        """Check if position is near a handle or inside box, return handle type"""
        if not self.inference_box_rect:
            return None
        
        handle_size = 15  # Slightly larger hit area
        
        # Check corners
        corners = {
            'tl': self.inference_box_rect.topLeft(),
            'tr': self.inference_box_rect.topRight(),
            'bl': self.inference_box_rect.bottomLeft(),
            'br': self.inference_box_rect.bottomRight()
        }
        
        for corner_name, point in corners.items():
            dx = abs(pos.x() - point.x())
            dy = abs(pos.y() - point.y())
            if dx < handle_size and dy < handle_size:
                return corner_name
        
        return None
            
    def draw_on_mask(self, start_pos, end_pos):
        """Draw on the current mask with brush/eraser"""
        if self.combined_mask is None:
            # Handle both grayscale (H,W) and RGB (H,W,3) images
            if len(self.current_image.shape) == 2:
                h, w = self.current_image.shape
            else:
                h, w = self.current_image.shape[:2]
            self.combined_mask = np.zeros((h, w), dtype=np.int32)
            
        if self.selected_mask_idx < 0:
            # Create new instance
            self.selected_mask_idx = self.next_mask_id
            self.next_mask_id += 1
            # Ensure color exists for new instance
            if self.selected_mask_idx not in self.mask_colors:
                self.mask_colors[self.selected_mask_idx] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Start editing mode if not already active
        if self.editing_instance_id != self.selected_mask_idx:
            self.start_editing_instance(self.selected_mask_idx)
            
        instance_id = self.selected_mask_idx
        
        # Convert to image coordinates
        x1, y1 = int(start_pos.x()), int(start_pos.y())
        x2, y2 = int(end_pos.x()), int(end_pos.y())
        
        # Create stroke mask
        temp_mask = np.zeros(self.editing_mask.shape, dtype=np.uint8)
        cv2.line(temp_mask, (x1, y1), (x2, y2), 1, self.brush_size)
        stroke_pixels = temp_mask > 0
        
        # Apply tool to editing mask (no overlap protection needed - isolated layer)
        if self.current_tool == 'brush':
            self.editing_mask[stroke_pixels] = 255
        else:  # eraser
            self.editing_mask[stroke_pixels] = 0
        
        # Update editing mask visualization incrementally
        self._update_editing_visualization_incremental(stroke_pixels)
        self.annotation_changed.emit()
        
    def add_mask(self, mask, mask_id=None, color=None, rebuild_viz=True):
        """Add a new mask to the combined segmentation mask
        
        Args:
            mask: Binary mask array (H×W with 0/255 or True/False values)
            mask_id: Optional persistent ID for this mask
            color: Optional (r,g,b) color tuple. If None, generates random color
            rebuild_viz: If True, rebuild visualization immediately. Set False for batch operations.
        """
        if not isinstance(mask, np.ndarray):
            return
            
        # Initialize combined mask if needed
        if self.combined_mask is None:
            h, w = mask.shape[:2]
            self.combined_mask = np.zeros((h, w), dtype=np.int32)
            self._cached_overlay = None  # Clear cache
        
        # Convert binary mask to boolean
        binary_mask = mask > 0 if mask.dtype != bool else mask
        
        # Assign or use provided mask ID
        if mask_id is None:
            mask_id = self.next_mask_id
            self.next_mask_id += 1
        else:
            # Update next_mask_id if necessary to avoid conflicts
            if mask_id >= self.next_mask_id:
                self.next_mask_id = mask_id + 1
        
        # Add to combined mask (overwrites overlapping pixels)
        self.combined_mask[binary_mask] = mask_id
        
        # Store color
        if color is None:
            color = tuple(np.random.randint(0, 255, 3).tolist())
        self.mask_colors[mask_id] = color
        
        # Rebuild visualization only if requested
        if rebuild_viz:
            self.rebuild_visualizations()
        self.selected_mask_idx = mask_id
        
        # Emit signal to notify that annotations have changed
        self.annotation_changed.emit()
            
    def rebuild_visualizations(self):
        """Rebuild all mask visualizations from combined mask"""
        # Clear existing visualization items
        for item in self.mask_items:
            self.scene.removeItem(item)
        self.mask_items = []
        
        if self.combined_mask is None:
            self._cached_overlay = None
            return
        
        # Get unique instance IDs (excluding background 0)
        instance_ids = np.unique(self.combined_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        if len(instance_ids) == 0:
            self._cached_overlay = None
            return
        
        # Create composite overlay for all instances
        h, w = self.combined_mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        for instance_id in instance_ids:
            # Get color for this instance
            if instance_id not in self.mask_colors:
                self.mask_colors[instance_id] = tuple(np.random.randint(0, 255, 3).tolist())
            color = self.mask_colors[instance_id]
            
            # Apply color with appropriate opacity
            mask_pixels = self.combined_mask == instance_id
            alpha = self.mask_opacity
            overlay[mask_pixels] = [color[0], color[1], color[2], alpha]
        
        # Cache the overlay for incremental updates
        self._cached_overlay = overlay
        
        # Convert to QPixmap and add to scene
        q_image = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image.copy())  # copy to prevent memory leak
        
        mask_item = QGraphicsPixmapItem(pixmap)
        mask_item.setZValue(1)
        mask_item.setVisible(self.masks_visible)
        self.scene.addItem(mask_item)
        self.mask_items.append(mask_item)
        
        # Add border around selected instance
        if self.selected_mask_idx > 0:
            selected_mask_binary = (self.combined_mask == self.selected_mask_idx).astype(np.uint8) * 255
            self.add_selection_border(selected_mask_binary)
        
        # Update instance labels
        self.update_instance_labels()
    
    def _update_overlay_region(self, affected_pixels, instance_id):
        """Incrementally update only the affected region of the visualization
        
        Args:
            affected_pixels: Boolean mask of pixels that changed
            instance_id: ID of instance being drawn, or None to erase (set to background)
        """
        if self.combined_mask is None:
            return
        
        # Initialize cached overlay if needed
        if self._cached_overlay is None:
            self.rebuild_visualizations()
            return
        
        # Update only affected pixels in cached overlay
        if instance_id is not None:
            # Drawing: set pixels to instance color
            if instance_id not in self.mask_colors:
                self.mask_colors[instance_id] = tuple(np.random.randint(0, 255, 3).tolist())
            color = self.mask_colors[instance_id]
            alpha = self.mask_opacity
            self._cached_overlay[affected_pixels] = [color[0], color[1], color[2], alpha]
        else:
            # Erasing: set pixels to transparent or underlying instance color
            # Check what's actually in the combined mask at these positions
            erase_mask = affected_pixels & (self.combined_mask == 0)
            self._cached_overlay[erase_mask] = [0, 0, 0, 0]
            
            # Update any pixels that now show a different instance
            reveal_mask = affected_pixels & (self.combined_mask > 0)
            if np.any(reveal_mask):
                revealed_ids = np.unique(self.combined_mask[reveal_mask])
                for revealed_id in revealed_ids:
                    if revealed_id > 0:
                        color = self.mask_colors.get(revealed_id, (255, 255, 255))
                        alpha = self.mask_opacity
                        reveal_pixels = reveal_mask & (self.combined_mask == revealed_id)
                        self._cached_overlay[reveal_pixels] = [color[0], color[1], color[2], alpha]
        
        # Update the pixmap from cached overlay
        h, w = self._cached_overlay.shape[:2]
        q_image = QImage(self._cached_overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image.copy())  # copy to prevent memory leak
        
        # Update existing mask item or create new one
        if self.mask_items:
            self.mask_items[0].setPixmap(pixmap)
        else:
            mask_item = QGraphicsPixmapItem(pixmap)
            mask_item.setZValue(1)
            mask_item.setVisible(self.masks_visible)
            self.scene.addItem(mask_item)
            self.mask_items.append(mask_item)
        
        # Update selection border if we modified the selected instance
        if self.selected_mask_idx > 0 and (instance_id == self.selected_mask_idx or instance_id is None):
            # Refresh border for selected instance
            selected_mask_binary = (self.combined_mask == self.selected_mask_idx).astype(np.uint8) * 255
            self.add_selection_border(selected_mask_binary)
    
    def add_mask_visualization(self, mask, color=None):
        """Add mask visualization overlay
        
        Args:
            mask: Binary mask array
            color: Optional (r,g,b) color tuple. If None, generates random color
        """
        # Create colored overlay
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Use provided color or generate random one
        if color is None:
            color = np.random.randint(0, 255, 3)
        self.mask_colors.append(color)
        overlay[mask > 0] = [color[0], color[1], color[2], self.mask_opacity]
        
        # Convert to QPixmap (make copy to avoid memory leak)
        q_image = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image.copy())
        
        # Add to scene
        mask_item = QGraphicsPixmapItem(pixmap)
        mask_item.setZValue(1)  # Above image
        mask_item.setVisible(self.masks_visible)  # Respect current visibility state
        self.scene.addItem(mask_item)
        self.mask_items.append(mask_item)
        
    def update_mask_visualization(self, idx):
        """DEPRECATED: Use rebuild_visualizations() instead"""
        # This method is no longer used with integer mask format
        # Just call rebuild_visualizations to update everything
        self.rebuild_visualizations()
    
    def update_instance_labels(self):
        """Create or update text labels showing instance numbers on the canvas"""
        # Clear existing labels
        for label_item in self.label_items:
            self.scene.removeItem(label_item)
        self.label_items = []
        
        if self.combined_mask is None or not self.labels_visible:
            return
        
        # Get unique instance IDs (excluding background 0)
        instance_ids = np.unique(self.combined_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        for idx, instance_id in enumerate(instance_ids, start=1):
            # Get the mask for this instance
            instance_mask = (self.combined_mask == instance_id)
            
            # Find centroid of the mask
            y_coords, x_coords = np.where(instance_mask)
            if len(y_coords) == 0:
                continue
            
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))
            
            # Get color for this instance
            color = self.mask_colors.get(instance_id, (255, 255, 255))
            
            # Create text label with the actual instance ID (not enumeration index)
            label_text = str(instance_id)
            text_item = QGraphicsTextItem(label_text)
            
            # Style the text
            font = QFont("Arial", 16, QFont.Weight.Bold)
            text_item.setFont(font)
            
            # Set text color to match mask color
            text_color = QColor(color[0], color[1], color[2])
            text_item.setDefaultTextColor(text_color)
            
            # Add black outline for better visibility
            # We'll do this by drawing the text multiple times with offset
            text_item.setHtml(
                f'<div style="color: rgb({color[0]}, {color[1]}, {color[2]}); '
                f'text-shadow: -2px -2px 0 #000, 2px -2px 0 #000, -2px 2px 0 #000, 2px 2px 0 #000, '
                f'-2px 0px 0 #000, 2px 0px 0 #000, 0px -2px 0 #000, 0px 2px 0 #000; '
                f'font-size: 50px; font-weight: bold;">{label_text}</div>'
            )
            
            # Position at centroid (offset by half text width/height for centering)
            text_rect = text_item.boundingRect()
            text_item.setPos(centroid_x - text_rect.width() / 2, 
                           centroid_y - text_rect.height() / 2)
            
            # Set high z-value to appear above masks
            text_item.setZValue(10)
            
            # Add to scene
            self.scene.addItem(text_item)
            self.label_items.append(text_item)
    
    def set_labels_visible(self, visible):
        """Show or hide instance number labels
        
        Args:
            visible: True to show labels, False to hide
        """
        self.labels_visible = visible
        
        if visible:
            # Update and show labels
            self.update_instance_labels()
        else:
            # Hide all labels
            for label_item in self.label_items:
                self.scene.removeItem(label_item)
            self.label_items = []
    
    def toggle_labels(self):
        """Toggle visibility of instance number labels"""
        self.set_labels_visible(not self.labels_visible)
            
    def set_annotations(self, annotations, mask_colors=None):
        """Load annotations for current frame
        
        Args:
            annotations: List of annotation dictionaries
            mask_colors: Optional dict mapping mask_id to (r,g,b) color tuple
        """
        # Commit any pending edits first
        if self.editing_instance_id > 0:
            self.commit_editing()
        
        # Clear existing masks and free memory
        # Explicitly clear the combined mask array
        if self.combined_mask is not None:
            del self.combined_mask
            self.combined_mask = None
        
        # Clear cached overlay
        if self._cached_overlay is not None:
            del self._cached_overlay
            self._cached_overlay = None
        
        for item in self.mask_items:
            self.scene.removeItem(item)
            # Explicitly delete the item to free Qt resources
            try:
                item.setPixmap(None)  # Release pixmap data
            except:
                pass
            del item
        
        # Clear mask references
        self.mask_items = []
        self.mask_colors = {}  # Reset to dict
        self.next_mask_id = 1
        
        # Clear selection border
        if self.selection_border_item:
            self.scene.removeItem(self.selection_border_item)
            try:
                self.selection_border_item.setPixmap(None)
            except:
                pass
            del self.selection_border_item
            self.selection_border_item = None
        
        # Reset selection
        self.selected_mask_idx = -1
        self.active_sam2_mask_idx = -1
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        if not annotations:
            return
        
        # Determine combined mask size from first annotation
        if annotations and 'mask' in annotations[0]:
            h, w = annotations[0]['mask'].shape[:2]
            self.combined_mask = np.zeros((h, w), dtype=np.int32)
        else:
            return
        
        # Build combined mask from individual annotations (no rebuild per annotation)
        for ann in annotations:
            if 'mask' not in ann:
                continue
            
            mask = ann['mask']
            if not isinstance(mask, np.ndarray):
                continue
            
            # Get instance ID
            mask_id = ann.get('mask_id', self.next_mask_id)
            if mask_id >= self.next_mask_id:
                self.next_mask_id = mask_id + 1
            
            # Add to combined mask
            binary_mask = mask > 0 if mask.dtype != bool else mask
            self.combined_mask[binary_mask] = mask_id
            
            # Store or assign color
            if mask_colors and mask_id in mask_colors:
                self.mask_colors[mask_id] = mask_colors[mask_id]
            else:
                self.mask_colors[mask_id] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Rebuild visualizations once at the end
        self.rebuild_visualizations()
                    
    def get_annotations(self):
        """Get current annotations - extracts binary masks from combined mask
        
        Includes the editing mask if currently editing an instance.
        """
        if self.combined_mask is None:
            return []
        
        annotations = []
        
        # Get unique instance IDs from combined mask (excluding background 0)
        instance_ids = np.unique(self.combined_mask)
        instance_ids = instance_ids[instance_ids > 0].tolist()
        
        # Add editing instance ID if currently editing
        if self.editing_instance_id > 0 and self.editing_mask is not None:
            if self.editing_instance_id not in instance_ids:
                instance_ids.append(self.editing_instance_id)
        
        for instance_id in instance_ids:
            # Check if this is the editing instance
            if instance_id == self.editing_instance_id and self.editing_mask is not None:
                # Use editing mask
                binary_mask = self.editing_mask.copy()
            else:
                # Extract binary mask from combined mask
                binary_mask = (self.combined_mask == instance_id).astype(np.uint8) * 255
            
            ann = {
                'mask': binary_mask,
                'label': f'instance_{instance_id}',
                'area': int(np.sum(binary_mask > 0)),
                'mask_id': int(instance_id)
            }
            annotations.append(ann)
        
        return annotations
        
    def set_selected_instance(self, idx):
        """Set selected instance by ID and zoom to it"""
        self.selected_mask_idx = idx
        
        # Zoom to the selected instance (idx is now an instance ID, not index)
        if self.combined_mask is not None and idx > 0:
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
        """Zoom to a specific instance by its mask_id
        
        Args:
            idx: Instance ID (mask_id) to zoom to
            padding: Extra space around the instance (in pixels)
        """
        if self.combined_mask is None or idx <= 0:
            return
        
        # Extract binary mask for this instance
        binary_mask = (self.combined_mask == idx).astype(np.uint8) * 255
        if not np.any(binary_mask > 0):
            return
        
        bbox = self.get_mask_bbox(binary_mask)
        if bbox is None:
            return
        
        x, y, width, height = bbox
        self.zoom_to_rect(x, y, width, height, padding)
    
    def highlight_instance(self, idx):
        """Highlight a specific instance by ID"""
        if self.combined_mask is None or idx <= 0:
            return
        
        # Commit any edits to previous instance before switching
        if self.editing_instance_id > 0 and self.editing_instance_id != idx:
            self.commit_editing()
        
        self.selected_mask_idx = idx
        self.rebuild_visualizations()
            
    def refresh_all_visualizations(self):
        """Refresh all mask visualizations with current selection highlighting"""
        self.rebuild_visualizations()
            
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
        """No-op: overlaps are impossible with integer mask format"""
        return 0  # No overlaps possible
    
    def start_editing_instance(self, instance_id):
        """Start editing an instance in isolation mode
        
        Extracts the instance to a temporary editing mask so edits don't affect other instances
        until committed.
        
        Args:
            instance_id: ID of instance to edit
        """
        # Commit any previous edits
        if self.editing_instance_id > 0:
            self.commit_editing()
        
        # Initialize editing mask
        if self.combined_mask is not None:
            h, w = self.combined_mask.shape
            self.editing_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Extract this instance to editing mask (remove from combined mask temporarily)
            instance_pixels = self.combined_mask == instance_id
            self.editing_mask[instance_pixels] = 255
            self.combined_mask[instance_pixels] = 0
        else:
            # No combined mask yet, create fresh
            h, w = self.current_image.shape[:2]
            self.editing_mask = np.zeros((h, w), dtype=np.uint8)
            self.combined_mask = np.zeros((h, w), dtype=np.int32)
        
        self.editing_instance_id = instance_id
        
        # Rebuild base visualization (without the editing instance)
        self.rebuild_visualizations()
        
        # Show editing mask on top
        self._update_editing_visualization()
    
    def commit_editing(self):
        """Commit editing mask back to combined mask
        
        Merges the temporary editing mask back into the main combined mask.
        """
        if self.editing_instance_id <= 0 or self.editing_mask is None:
            return
        
        # Check if combined_mask exists
        if self.combined_mask is None:
            return
        
        # Merge editing mask back into combined mask
        editing_pixels = self.editing_mask > 0
        self.combined_mask[editing_pixels] = self.editing_instance_id
        
        # Clear editing state and free memory
        old_editing_mask = self.editing_mask
        self.editing_mask = None
        self.editing_instance_id = -1
        del old_editing_mask
        
        # Clear editing overlay cache
        if self._editing_overlay_cache is not None:
            del self._editing_overlay_cache
            self._editing_overlay_cache = None
        
        # Remove editing visualization
        if self.editing_mask_item:
            self.scene.removeItem(self.editing_mask_item)
            try:
                self.editing_mask_item.setPixmap(None)
            except:
                pass
            del self.editing_mask_item
            self.editing_mask_item = None
        
        # Rebuild full visualization
        self.rebuild_visualizations()
        
        self.annotation_changed.emit()
    
    def _update_editing_visualization(self):
        """Update visualization of the editing mask
        
        Shows the editing mask as an overlay on top of other instances.
        Rebuilds the entire overlay - use for initialization or major changes.
        """
        if self.editing_mask is None or self.editing_instance_id <= 0:
            return
        
        # Remove previous editing visualization
        if self.editing_mask_item:
            self.scene.removeItem(self.editing_mask_item)
            self.editing_mask_item = None
        
        # Create or rebuild cached overlay for editing mask
        h, w = self.editing_mask.shape
        
        # Get color for this instance
        if self.editing_instance_id not in self.mask_colors:
            self.mask_colors[self.editing_instance_id] = tuple(np.random.randint(0, 255, 3).tolist())
        color = self.mask_colors[self.editing_instance_id]
        
        # Create full overlay and cache it
        self._editing_overlay_cache = np.zeros((h, w, 4), dtype=np.uint8)
        mask_pixels = self.editing_mask > 0
        alpha = min(self.mask_opacity, 255)
        self._editing_overlay_cache[mask_pixels] = [color[0], color[1], color[2], alpha]
        
        # Convert to QPixmap (use QImage.copy() to prevent memory leak)
        q_image = QImage(self._editing_overlay_cache.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image.copy())
        
        # Add to scene on top of everything
        self.editing_mask_item = QGraphicsPixmapItem(pixmap)
        self.editing_mask_item.setZValue(2)  # Above regular masks (which are at 1)
        self.editing_mask_item.setVisible(self.masks_visible)
        self.scene.addItem(self.editing_mask_item)
        
        # Update selection border for editing mask (skip if actively drawing)
        if not self._is_actively_drawing and self.selected_mask_idx == self.editing_instance_id:
            self.add_selection_border(self.editing_mask)
    
    def _update_editing_visualization_incremental(self, affected_pixels):
        """Update only the affected region of the editing visualization
        
        Much faster than full rebuild - updates only changed pixels.
        
        Args:
            affected_pixels: Boolean mask of pixels that changed
        """
        if self.editing_mask is None or self.editing_instance_id <= 0:
            return
        
        # Initialize cache if needed
        if self._editing_overlay_cache is None:
            self._update_editing_visualization()
            return
        
        # Get color for this instance
        color = self.mask_colors.get(self.editing_instance_id, (255, 255, 255))
        alpha = min(self.mask_opacity, 255)
        
        # Update only affected pixels in the cached overlay
        # For brush: set pixels to color where editing_mask is non-zero
        # For eraser: set pixels to transparent where editing_mask is zero
        brush_pixels = affected_pixels & (self.editing_mask > 0)
        eraser_pixels = affected_pixels & (self.editing_mask == 0)
        
        self._editing_overlay_cache[brush_pixels] = [color[0], color[1], color[2], alpha]
        self._editing_overlay_cache[eraser_pixels] = [0, 0, 0, 0]
        
        # Update the pixmap from cached overlay
        # Use QImage.copy() to prevent memory leak from buffer references
        h, w = self._editing_overlay_cache.shape[:2]
        q_image = QImage(self._editing_overlay_cache.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(q_image.copy())  # copy() creates independent image
        
        # Update existing item
        if self.editing_mask_item:
            self.editing_mask_item.setPixmap(pixmap)
        else:
            # Create item if it doesn't exist
            self.editing_mask_item = QGraphicsPixmapItem(pixmap)
            self.editing_mask_item.setZValue(2)
            self.editing_mask_item.setVisible(self.masks_visible)
            self.scene.addItem(self.editing_mask_item)
    
    def clear_selected_instance(self):
        """Remove the selected instance from the combined mask"""
        if self.selected_mask_idx <= 0:
            return
        
        # If editing this instance, just clear the editing mask
        if self.editing_instance_id == self.selected_mask_idx:
            self.editing_mask = np.zeros_like(self.editing_mask)
            self._update_editing_visualization()
        else:
            # Not in editing mode, directly clear from combined mask
            if self.combined_mask is not None:
                self.combined_mask[self.combined_mask == self.selected_mask_idx] = 0
        
        # Remove color entry
        if self.selected_mask_idx in self.mask_colors:
            del self.mask_colors[self.selected_mask_idx]
        
        # Rebuild visualization
        if self.editing_instance_id <= 0:
            self.rebuild_visualizations()
        
        # Clear prompt points
        for item in self.prompt_items:
            self.scene.removeItem(item)
        self.positive_points = []
        self.negative_points = []
        self.prompt_items = []
        
        # Reset selection
        self.active_sam2_mask_idx = -1
        self.selected_mask_idx = -1
        
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
        # Update all visualizations with new opacity
        self.rebuild_visualizations()
        
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
        if self.active_sam2_mask_idx > 0 and self.combined_mask is not None:
            # Only delete if it's NOT the selected instance (selected instances should be preserved)
            if self.active_sam2_mask_idx != self.selected_mask_idx:
                # Remove this instance from combined mask
                self.combined_mask[self.combined_mask == self.active_sam2_mask_idx] = 0
                
                # Remove color entry
                if self.active_sam2_mask_idx in self.mask_colors:
                    del self.mask_colors[self.active_sam2_mask_idx]
                
                # Rebuild visualizations
                self.rebuild_visualizations()
        
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
