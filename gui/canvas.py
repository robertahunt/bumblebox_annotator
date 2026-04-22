"""
Image canvas with zoom, pan, and annotation capabilities
"""

import numpy as np
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QTimer
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
        self.mask_opacity = 64  # Default opacity for masks (0-255) - 25%
        
        # Annotation type - determines which category is being added
        self.current_annotation_type = 'bee'  # 'bee', 'chamber', or 'hive'
        
        # Annotations - Separate masks allow overlapping between bee/chamber/hive
        self.bee_mask = None  # H×W int32 array with bee instance IDs
        self.chamber_mask = None  # H×W int32 array with chamber instance IDs
        self.hive_mask = None  # H×W int32 array with hive instance IDs
        
        # Legacy compatibility: provide combined view of all masks (readonly, dynamically generated)
        self._combined_mask_cache = None
        self._combined_mask_dirty = True
        
        self.mask_items = []  # List of QGraphicsPixmapItem for visualization
        self.mask_colors = {}  # Dict mapping instance_id -> (r,g,b) color
        self.next_mask_id = 1  # Counter for assigning unique IDs
        self.selected_mask_idx = -1
        self.masks_visible = True  # Track mask visibility state
        self.label_items = []  # List of QGraphicsTextItem for instance labels
        self.labels_visible = False  # Track label visibility state
        self.labels_visible_before_hiding = False  # Remember label state when spacebar is pressed
        self.annotation_metadata = {}  # Dict mapping instance_id -> metadata dict (for marker data, etc.)
        
        # Annotation type visibility and colors
        self.annotation_type_colors = {
            'bee': None,  # Random colors for bee instances
            'chamber': (255, 0, 0),  # Red for chamber
            'hive': (255, 255, 0)  # Yellow for hive
        }
        # Initialize visibility to match toolbar checkbox defaults (bees=True, others=False)
        self.annotation_type_visibility = {
            'bee': True,
            'chamber': False,
            'hive': False
        }
        
        # Performance optimization: cache instance IDs and bounding boxes
        self._cached_instance_ids = None  # Cached sorted list of instance IDs
        self._cached_bboxes = {}  # Dict mapping instance_id -> (x, y, w, h)
        
        # Performance optimization: cache instance IDs and bounding boxes
        self._cached_instance_ids = None  # Cached sorted list of instance IDs
        self._cached_bboxes = {}  # Dict mapping instance_id -> (x, y, w, h)
        
        # Performance optimization: cache instance IDs and bounding boxes
        self._cached_instance_ids = None  # Cached sorted list of instance IDs
        self._cached_bboxes = {}  # Dict mapping instance_id -> (x, y, w, h)
        
        # Editing isolation - temporary mask for current edit
        self.editing_mask = None  # Binary H×W mask for instance being edited
        self.editing_instance_id = -1  # ID of instance being edited
        self.editing_mask_item = None  # Visualization item for editing mask
        self._editing_overlay_cache = None  # Cached RGBA overlay for editing mask
        self._is_actively_drawing = False  # Flag to skip expensive updates during drawing
        
        # Throttling for brush/eraser to improve performance
        self._pending_viz_update = False  # Flag indicating viz update is pending
        self._dirty_pixels = None  # Accumulate dirty pixels for batched update
        self._viz_update_timer = QTimer()
        self._viz_update_timer.setSingleShot(True)
        self._viz_update_timer.setInterval(16)  # ~60fps max update rate
        self._viz_update_timer.timeout.connect(self._flush_pending_viz_update)
        
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
        
        # View toggles
        self.show_segmentations = True  # Whether to show segmentation masks
        self.show_bboxes = False  # Whether to show bounding boxes
        
        # Bbox editing (for bbox annotation mode)
        self.bbox_items_map = {}  # Dict mapping mask_id -> QGraphicsRectItem
        self.drawing_new_bbox = False  # Whether currently drawing a new bbox
        self.selected_bbox_id = None  # ID of currently selected bbox
        self.bbox_handles = []  # Corner handle items for selected bbox
        self.dragging_bbox = False
        self.dragging_bbox_handle = None  # 'tl', 'tr', 'bl', 'br', 'move'
        self.bbox_drag_offset = None
        self.bbox_original_rect = None  # Original rect before editing
        
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
        # Flush any pending updates immediately to avoid lag
        if self._viz_update_timer.isActive():
            self._viz_update_timer.stop()
            self._flush_pending_viz_update()
        
        self.masks_visible = False
        for item in self.mask_items:
            item.setVisible(False)
        # Hide editing mask if present
        if self.editing_mask_item:
            self.editing_mask_item.setVisible(False)
        # Also hide selection border
        if self.selection_border_item:
            self.selection_border_item.setVisible(False)
        
        # Remember if labels were visible and hide them too
        self.labels_visible_before_hiding = self.labels_visible
        if self.labels_visible:
            for label_item in self.label_items:
                label_item.setVisible(False)
        
        self.masks_visibility_changed.emit(False)
    
    def show_masks(self):
        """Show all masks again"""
        # Flush any pending updates immediately to avoid lag
        if self._viz_update_timer.isActive():
            self._viz_update_timer.stop()
            self._flush_pending_viz_update()
        
        self.masks_visible = True
        for item in self.mask_items:
            item.setVisible(True)
        # Show editing mask if present
        if self.editing_mask_item:
            self.editing_mask_item.setVisible(True)
        # Show selection border if there's a selection
        if self.selection_border_item and self.selected_mask_idx >= 0:
            self.selection_border_item.setVisible(True)
        
        # Restore labels if they were visible before hiding
        if self.labels_visible_before_hiding:
            for label_item in self.label_items:
                label_item.setVisible(True)
        
        self.masks_visibility_changed.emit(True)
        
    def load_image(self, image_path_or_array):
        """Load an image file or numpy array"""
        # Explicitly clear old resources before loading new image
        if self.image_item:
            try:
                if self.image_item.scene() == self.scene:
                    self.scene.removeItem(self.image_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
            self.image_item = None
        
        # Clear all visualization items
        for item in self.mask_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        self.mask_items.clear()
        
        for item in self.label_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        self.label_items.clear()
        
        for item in self.prompt_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        self.prompt_items.clear()
        
        if self.selection_border_item:
            try:
                if self.selection_border_item.scene() == self.scene:
                    self.scene.removeItem(self.selection_border_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
            self.selection_border_item = None
        
        if self.editing_mask_item:
            try:
                if self.editing_mask_item.scene() == self.scene:
                    self.scene.removeItem(self.editing_mask_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
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
        self.bee_mask = None
        self.chamber_mask = None
        self.hive_mask = None
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
        self.bee_mask = None
        self.chamber_mask = None
        self.hive_mask = None
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
        # Flush any pending visualization updates when switching tools
        if self._viz_update_timer.isActive():
            self._viz_update_timer.stop()
            self._flush_pending_viz_update()
        
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
                # Only check for instance selection if NOT already in editing mode
                if self.editing_instance_id <= 0:
                    # If an instance is already selected (from sidebar), start editing it
                    if self.selected_mask_idx > 0:
                        # Start editing the selected instance
                        self.start_editing_instance(self.selected_mask_idx)
                    else:
                        # No instance selected yet - check what's under the click
                        # Check if clicking on a bbox (when bboxes are visible)
                        clicked_bbox_id = self._get_bbox_at_pos(scene_pos)
                        if clicked_bbox_id:
                            # Clicked on a bbox - start editing that instance
                            self.start_editing_instance(clicked_bbox_id)
                            self.selected_mask_idx = clicked_bbox_id
                        else:
                            # Check if clicking on an existing instance in the active annotation
                            # type's mask only — don't pick up instances from other types
                            x, y = int(scene_pos.x()), int(scene_pos.y())
                            instance_id = self._find_instance_at_point(x, y,
                                          category=self.current_annotation_type)
                            
                            # If clicked on an instance, start editing mode
                            if instance_id > 0:
                                self.start_editing_instance(instance_id)
                                # Update instance list selection
                                self.selected_mask_idx = instance_id
                
                self.is_drawing = True
                self.last_point = scene_pos
        
        elif self.current_tool == 'bbox':
            # Start drawing new bbox
            if event.button() == Qt.MouseButton.LeftButton:
                # If there's a selected instance, check if it already has a bbox
                if self.selected_mask_idx > 0:
                    if self.selected_mask_idx in self.bbox_items_map:
                        # Instance has a bbox - select it for editing
                        self._select_bbox(self.selected_mask_idx)
                        # Check if clicking on handle
                        handle = self._get_bbox_handle_at_pos(scene_pos)
                        if handle:
                            self.dragging_bbox_handle = handle
                            self.dragging_bbox = True
                            self.bbox_drag_offset = scene_pos
                        elif self.bbox_items_map[self.selected_mask_idx].rect().contains(scene_pos):
                            # Clicking inside bbox - start dragging
                            self.dragging_bbox = True
                            self.dragging_bbox_handle = 'move'
                            rect = self.bbox_items_map[self.selected_mask_idx].rect()
                            self.bbox_drag_offset = QPointF(scene_pos.x() - rect.x(), scene_pos.y() - rect.y())
                        return
                    else:
                        # Instance doesn't have a bbox yet - allow drawing one
                        self.is_drawing = True
                        self.drawing_new_bbox = True
                        self.drawing_start = scene_pos
                        return
                
                # No selected instance - check general bbox interactions
                # Check if clicking on existing bbox handle
                if self.selected_bbox_id:
                    handle = self._get_bbox_handle_at_pos(scene_pos)
                    if handle:
                        self.dragging_bbox_handle = handle
                        self.dragging_bbox = True
                        self.bbox_drag_offset = scene_pos
                        return
                
                # Check if clicking inside existing bbox
                bbox_id = self._get_bbox_at_pos(scene_pos)
                if bbox_id:
                    self._select_bbox(bbox_id)
                    self.dragging_bbox = True
                    self.dragging_bbox_handle = 'move'
                    rect = self.bbox_items_map[bbox_id].rect()
                    self.bbox_drag_offset = QPointF(scene_pos.x() - rect.x(), scene_pos.y() - rect.y())
                else:
                    # Start drawing new bbox
                    self.is_drawing = True
                    self.drawing_new_bbox = True
                    self.drawing_start = scene_pos
                
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """Handle mouse move"""
        if self.current_image is None:
            return
            
        scene_pos = self.mapToScene(event.pos())
        
        if self.is_drawing:
            if self.drawing_new_bbox:
                # Update bbox preview
                if self.temp_item:
                    try:
                        self.scene.removeItem(self.temp_item)
                    except RuntimeError:
                        pass
                
                # Draw temporary rectangle
                from PyQt6.QtWidgets import QGraphicsRectItem
                from PyQt6.QtGui import QPen, QColor
                rect = QRectF(self.drawing_start, scene_pos).normalized()
                self.temp_item = QGraphicsRectItem(rect)
                pen = QPen(QColor(255, 255, 0))  # Yellow
                pen.setWidth(3)
                pen.setStyle(Qt.PenStyle.DashLine)
                self.temp_item.setPen(pen)
                self.temp_item.setZValue(150)
                self.scene.addItem(self.temp_item)
            elif self.current_tool == 'sam2_box':
                # Update box preview
                if self.temp_item:
                    try:
                        self.scene.removeItem(self.temp_item)
                    except RuntimeError:
                        pass
                    
                from PyQt6.QtWidgets import QGraphicsRectItem
                rect = QRectF(self.drawing_start, scene_pos).normalized()
                self.temp_item = QGraphicsRectItem(rect)
                self.temp_item.setPen(QPen(QColor(0, 255, 0), 2))
                self.scene.addItem(self.temp_item)
                
            elif self.current_tool == 'inference_box':
                # Update inference box preview
                if self.temp_item:
                    try:
                        self.scene.removeItem(self.temp_item)
                    except RuntimeError:
                        pass
                    
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
        
        elif self.dragging_bbox and self.show_bboxes:
            # Handle dragging/resizing bbox
            if self.dragging_bbox_handle == 'move':
                # Move entire bbox
                if self.selected_bbox_id and self.selected_bbox_id in self.bbox_items_map:
                    rect_item = self.bbox_items_map[self.selected_bbox_id]
                    rect = rect_item.rect()
                    new_x = scene_pos.x() - self.bbox_drag_offset.x()
                    new_y = scene_pos.y() - self.bbox_drag_offset.y()
                    rect.moveTo(new_x, new_y)
                    rect_item.setRect(rect)
                    self._update_bbox_handles()
            elif self.dragging_bbox_handle in ['tl', 'tr', 'bl', 'br']:
                # Resize from corner
                if self.selected_bbox_id and self.selected_bbox_id in self.bbox_items_map:
                    rect_item = self.bbox_items_map[self.selected_bbox_id]
                    rect = rect_item.rect()
                    if self.dragging_bbox_handle == 'tl':
                        rect.setTopLeft(scene_pos)
                    elif self.dragging_bbox_handle == 'tr':
                        rect.setTopRight(scene_pos)
                    elif self.dragging_bbox_handle == 'bl':
                        rect.setBottomLeft(scene_pos)
                    elif self.dragging_bbox_handle == 'br':
                        rect.setBottomRight(scene_pos)
                    rect = rect.normalized()
                    rect_item.setRect(rect)
                    self._update_bbox_handles()
                
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if self.current_image is None:
            return
            
        scene_pos = self.mapToScene(event.pos())
        
        if self.is_drawing:
            if self.drawing_new_bbox:
                # Finish drawing new bbox
                if self.temp_item:
                    try:
                        self.scene.removeItem(self.temp_item)
                    except RuntimeError:
                        pass
                    self.temp_item = None
                
                # Create the new bbox
                rect = QRectF(self.drawing_start, scene_pos).normalized()
                
                # Only create if bbox has minimum size
                if rect.width() > 10 and rect.height() > 10:
                    self._create_new_bbox(rect)
                
                self.drawing_new_bbox = False
                self.is_drawing = False
                self.setToolTip("Click on a bbox to select and edit")
                
            elif self.current_tool == 'sam2_box':
                # Emit box drawn signal
                if self.temp_item:
                    try:
                        self.scene.removeItem(self.temp_item)
                    except RuntimeError:
                        pass
                    self.temp_item = None
                    
                x1, y1 = int(self.drawing_start.x()), int(self.drawing_start.y())
                x2, y2 = int(scene_pos.x()), int(scene_pos.y())
                self.box_drawn.emit(x1, y1, x2, y2)
                
            elif self.current_tool == 'inference_box':
                # Finish drawing inference box
                if self.temp_item:
                    try:
                        self.scene.removeItem(self.temp_item)
                    except RuntimeError:
                        pass
                    self.temp_item = None
                    
                # Store the box
                rect = QRectF(self.drawing_start, scene_pos).normalized()
                self.set_inference_box(rect)
                self.box_drawn.emit(int(rect.x()), int(rect.y()),
                                   int(rect.right()), int(rect.bottom()))
                
            elif self.current_tool == 'brush' or self.current_tool == 'eraser':
                # Flush any pending visualization updates before finishing
                if self._viz_update_timer.isActive():
                    self._viz_update_timer.stop()
                    self._flush_pending_viz_update()
                
                # Finish drawing - update border now
                self._is_actively_drawing = False
                if self.editing_mask is not None and self.selected_mask_idx == self.editing_instance_id:
                    self.add_selection_border(self.editing_mask)
                
                # Emit annotation_changed now that drawing is complete
                self.annotation_changed. emit()
                
            self.is_drawing = False
        elif self.dragging_box:
            # Finish dragging inference box
            self.dragging_box = False
            self.dragging_handle = None
            self.drag_offset = None
        elif self.dragging_bbox:
            # Finish dragging/resizing bbox
            if self.selected_bbox_id and self.selected_bbox_id in self.bbox_items_map:
                # Update the annotation metadata with new bbox coordinates
                rect_item = self.bbox_items_map[self.selected_bbox_id]
                rect = rect_item.rect()
                self.annotation_metadata[self.selected_bbox_id]['bbox'] = [
                    int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
                ]
                # Update labels to reflect new position
                self.update_instance_labels()
                # Emit annotation changed signal
                self.annotation_changed.emit()
            self.dragging_bbox = False
            self.dragging_bbox_handle = None
            self.bbox_drag_offset = None
        else:
            super().mouseReleaseEvent(event)
    
    def set_inference_box(self, rect):
        """Set the inference box rectangle and display it with handles"""
        self.inference_box_rect = rect
        self._update_inference_box_display()
    
    def clear_inference_box(self):
        """Remove the inference box from display"""
        if self.inference_box_item:
            try:
                self.scene.removeItem(self.inference_box_item)
            except RuntimeError:
                pass
            self.inference_box_item = None
        for handle in self.inference_box_handles:
            try:
                self.scene.removeItem(handle)
            except RuntimeError:
                pass
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
            try:
                self.scene.removeItem(self.inference_box_item)
            except RuntimeError:
                pass
        for handle in self.inference_box_handles:
            try:
                self.scene.removeItem(handle)
            except RuntimeError:
                pass
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
        # Initialize appropriate mask if needed
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
        else:
            return
        
        # Initialize appropriate mask based on annotation type
        if self.current_annotation_type == 'chamber':
            if self.chamber_mask is None:
                self.chamber_mask = np.zeros((h, w), dtype=np.int32)
        elif self.current_annotation_type == 'hive':
            if self.hive_mask is None:
                self.hive_mask = np.zeros((h, w), dtype=np.int32)
        else:  # bee
            if self.bee_mask is None:
                self.bee_mask = np.zeros((h, w), dtype=np.int32)
            
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
        
        # Draw directly on editing mask (much faster - no temp array allocation)
        if self.current_tool == 'brush':
            cv2.line(self.editing_mask, (x1, y1), (x2, y2), 255, self.brush_size)
        else:  # eraser
            cv2.line(self.editing_mask, (x1, y1), (x2, y2), 0, self.brush_size)
        
        # Calculate bounding box of stroke for dirty region (much more memory efficient)
        half_brush = self.brush_size // 2 + 1
        min_x = max(0, min(x1, x2) - half_brush)
        max_x = min(self.editing_mask.shape[1], max(x1, x2) + half_brush)
        min_y = max(0, min(y1, y2) - half_brush)
        max_y = min(self.editing_mask.shape[0], max(y1, y2) + half_brush)
        
        # Create dirty region mask for this stroke
        stroke_dirty = np.zeros(self.editing_mask.shape, dtype=bool)
        stroke_dirty[min_y:max_y, min_x:max_x] = True
        
        # Accumulate dirty pixels for batched update (major performance improvement)
        if self._dirty_pixels is None:
            self._dirty_pixels = stroke_dirty
        else:
            self._dirty_pixels |= stroke_dirty
        
        # Schedule throttled visualization update
        if not self._viz_update_timer.isActive():
            self._viz_update_timer.start()
        
        # Only emit annotation_changed for final stroke (on release) to avoid slowdown
        # The visualization update happens independently via timer
        
    def add_mask(self, mask, mask_id=None, color=None, rebuild_viz=True, category=None):
        """Add a new mask based on current annotation type
        
        Args:
            mask: Binary mask array (H×W with 0/255 or True/False values)
            mask_id: Optional persistent ID for bee masks (ignored for chamber/hive)
            color: Optional (r,g,b) color tuple. If None, uses default for type
            rebuild_viz: If True, rebuild visualization immediately
            category: Override annotation type ('bee', 'chamber', 'hive'). Defaults to current_annotation_type.
        """
        if not isinstance(mask, np.ndarray):
            return
        
        # Resolve effective category
        effective_category = category if category is not None else self.current_annotation_type

        # Convert binary mask to boolean
        binary_mask = mask > 0 if mask.dtype != bool else mask
        
        if effective_category == 'chamber':
            # Update chamber mask
            self._add_chamber_mask(binary_mask, mask_id, color, rebuild_viz)
        elif effective_category == 'hive':
            # Update hive mask
            self._add_hive_mask(binary_mask, mask_id, color, rebuild_viz)
        else:
            # Default: bee annotation (multi-instance)
            self._add_bee_mask(binary_mask, mask_id, color, rebuild_viz)
    
    def _add_bee_mask(self, binary_mask, mask_id=None, color=None, rebuild_viz=True, category='bee'):
        """Add an instance mask to the appropriate mask array (bee, chamber, or hive)
        
        Args:
            binary_mask: Boolean array indicating instance pixels
            mask_id: Optional persistent ID for this instance
            color: Optional (r,g,b) color tuple
            rebuild_viz: If True, rebuild visualization immediately
            category: Instance category ('bee', 'chamber', or 'hive')
        """
        # Get reference to the appropriate mask based on category
        if category == 'chamber':
            mask_array_name = 'chamber_mask'
        elif category == 'hive':
            mask_array_name = 'hive_mask'
        else:
            mask_array_name = 'bee_mask'
        
        # Initialize mask if needed
        if getattr(self, mask_array_name) is None:
            h, w = binary_mask.shape[:2]
            setattr(self, mask_array_name, np.zeros((h, w), dtype=np.int32))
            self._cached_overlay = None  # Clear cache
        
        mask_array = getattr(self, mask_array_name)
        
        # Guard against stale callbacks (e.g. async SAM2) where the mask shape
        # no longer matches the re-initialised mask array (different video frame size).
        if binary_mask.shape[:2] != mask_array.shape[:2]:
            print(f"Warning: add_mask ignored – binary_mask shape {binary_mask.shape[:2]} "
                  f"doesn't match {mask_array_name} shape {mask_array.shape[:2]} "
                  f"(likely stale async result from previous video)")
            return
        
        # Assign or use provided mask ID
        if mask_id is None:
            mask_id = self.next_mask_id
            self.next_mask_id += 1
        else:
            # Update next_mask_id if necessary to avoid conflicts
            if mask_id >= self.next_mask_id:
                self.next_mask_id = mask_id + 1
        
        # Add to the appropriate mask array
        mask_array[binary_mask] = mask_id
        
        # Mark combined mask cache as dirty
        self._combined_mask_dirty = True
        
        # Invalidate cached instance IDs since we added a new one
        self._cached_instance_ids = None
        
        # Store color based on category
        if color is None:
            if category == 'chamber':
                color = self.annotation_type_colors['chamber']  # Red
            elif category == 'hive':
                color = self.annotation_type_colors['hive']  # Yellow
            else:  # bee
                color = tuple(np.random.randint(0, 255, 3).tolist())
        self.mask_colors[mask_id] = color
        
        # Initialize metadata and store category
        if mask_id not in self.annotation_metadata:
            self.annotation_metadata[mask_id] = {}
        self.annotation_metadata[mask_id]['category'] = category
        
        # Rebuild visualization only if requested
        if rebuild_viz:
            self.rebuild_visualizations()
        self.selected_mask_idx = mask_id
        
        # Emit signal to notify that annotations have changed
        self.annotation_changed.emit()
    
    def _add_chamber_mask(self, binary_mask, mask_id=None, color=None, rebuild_viz=True):
        """Add chamber instance to combined mask with category='chamber'
        
        Args:
            binary_mask: Boolean array indicating chamber pixels
            mask_id: Optional persistent ID for this instance
            color: Optional (r,g,b) color tuple. If None, uses default red for chamber
            rebuild_viz: If True, rebuild visualization immediately
        """
        # Add as a regular instance with chamber category
        self._add_bee_mask(binary_mask, mask_id=mask_id, color=color, rebuild_viz=rebuild_viz, category='chamber')
    
    def _add_hive_mask(self, binary_mask, mask_id=None, color=None, rebuild_viz=True):
        """Add hive instance to combined mask with category='hive'
        
        Args:
            binary_mask: Boolean array indicating hive pixels
            mask_id: Optional persistent ID for this instance
            color: Optional (r,g,b) color tuple. If None, uses default yellow for hive
            rebuild_viz: If True, rebuild visualization immediately
        """
        # Add as a regular instance with hive category
        self._add_bee_mask(binary_mask, mask_id=mask_id, color=color, rebuild_viz=rebuild_viz, category='hive')
    
    def _get_mask_array_for_instance(self, instance_id):
        """Get the appropriate mask array for a given instance ID
        
        Args:
            instance_id: Instance ID to look up
            
        Returns:
            Tuple of (mask_array, category) or (None, None) if not found
        """
        if instance_id <= 0:
            return None, None
        
        # Check metadata for category
        category = self.annotation_metadata.get(instance_id, {}).get('category', 'bee')
        
        if category == 'chamber':
            return self.chamber_mask, 'chamber'
        elif category == 'hive':
            return self.hive_mask, 'hive'
        else:
            return self.bee_mask, 'bee'
    
    def _find_instance_at_point(self, x, y, category=None):
        """Find which instance (if any) is at the given point

        Args:
            x: X coordinate
            y: Y coordinate
            category: If given ('bee', 'chamber', or 'hive'), only search that
                      type's mask. Otherwise all three masks are searched.

        Returns:
            Instance ID at that point, or 0 if no instance
        """
        if category == 'chamber':
            masks_to_search = [self.chamber_mask]
        elif category == 'hive':
            masks_to_search = [self.hive_mask]
        elif category == 'bee':
            masks_to_search = [self.bee_mask]
        else:
            masks_to_search = [self.bee_mask, self.chamber_mask, self.hive_mask]

        for mask_array in masks_to_search:
            if (mask_array is not None and
                0 <= x < mask_array.shape[1] and
                0 <= y < mask_array.shape[0]):
                instance_id = int(mask_array[y, x])
                if instance_id > 0:
                    return instance_id
        return 0
    
    def _get_binary_mask_for_instance(self, instance_id):
        """Get binary mask for a specific instance ID
        
        Args:
            instance_id: Instance ID to extract
            
        Returns:
            Binary mask (H×W uint8) or None if instance not found
        """
        if instance_id <= 0:
            return None
        
        # Find which mask contains this instance
        mask_array, category = self._get_mask_array_for_instance(instance_id)
        if mask_array is not None:
            return (mask_array == instance_id).astype(np.uint8) * 255
        
        return None
    
    @property
    def combined_mask(self):
        """Legacy compatibility property: provides a readonly combined view of all masks
        
        This dynamically generates a single mask with all instances for backward compatibility.
        Note: This is readonly - modifications won't affect the underlying separate masks.
        Use the appropriate mask (bee_mask, chamber_mask, hive_mask) directly for modifications.
        
        Returns:
            H×W int32 array with all instance IDs, or None if no masks exist
        """
        # Return None if no masks exist
        if self.bee_mask is None and self.chamber_mask is None and self.hive_mask is None:
            return None
        
        # Use cached version if available and not dirty
        if self._combined_mask_cache is not None and not self._combined_mask_dirty:
            return self._combined_mask_cache
        
        #Determine dimensions
        if self.bee_mask is not None:
            h, w = self.bee_mask.shape
        elif self.chamber_mask is not None:
            h, w = self.chamber_mask.shape
        elif self.hive_mask is not None:
            h, w = self.hive_mask.shape
        else:
            return None
        
        # Create combined view (simple approach: bee mask as base since it's most common)
        combined = np.zeros((h, w), dtype=np.int32)
        
        # Layer masks: bee first (base layer), then chamber, then hive
        # Later layers overwrite earlier ones where they overlap
        if self.bee_mask is not None:
            combined[self.bee_mask > 0] = self.bee_mask[self.bee_mask > 0]
        if self.chamber_mask is not None:
            if self.chamber_mask.shape[:2] == (h, w):
                combined[self.chamber_mask > 0] = self.chamber_mask[self.chamber_mask > 0]
            else:
                print(f"Warning: combined_mask skipping chamber_mask "
                      f"(shape {self.chamber_mask.shape[:2]} != {(h, w)})")
        if self.hive_mask is not None:
            if self.hive_mask.shape[:2] == (h, w):
                combined[self.hive_mask > 0] = self.hive_mask[self.hive_mask > 0]
            else:
                print(f"Warning: combined_mask skipping hive_mask "
                      f"(shape {self.hive_mask.shape[:2]} != {(h, w)})")
        
        # Cache for future reads
        self._combined_mask_cache = combined
        self._combined_mask_dirty = False
        
        return combined
    
    @combined_mask.setter
    def combined_mask(self, value):
        """Setter for backward compatibility - mark cache as dirty"""
        # Setting to None clears all masks
        if value is None:
            self.bee_mask = None
            self.chamber_mask = None
            self.hive_mask = None
            self._combined_mask_cache = None
            self._combined_mask_dirty = True
        else:
            # For other values, just mark cache as dirty
            # Direct assignment not supported - use set_annotations instead
            self._combined_mask_dirty = True
            
    def rebuild_visualizations(self):
        """Rebuild all visualizations (masks and/or bboxes) based on display flags"""
        # Clear existing visualization items
        for item in self.mask_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        self.mask_items = []
        
        # Clear bbox items properly before clearing the map
        for mask_id, rect_item in list(self.bbox_items_map.items()):
            try:
                if rect_item.scene() == self.scene:
                    self.scene.removeItem(rect_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        self.bbox_items_map = {}
        
        # Check if any masks exist
        has_masks = (self.bee_mask is not None or 
                     self.chamber_mask is not None or 
                     self.hive_mask is not None)
        
        if not has_masks:
            self._cached_overlay = None
            # Still draw bbox-only annotations if we have them
            if self.show_bboxes:
                self._draw_bboxes_from_metadata()
            return
        
        # Collect all instance IDs from all masks
        all_instance_ids = []
        mask_lookup = {}  # Map instance_id -> (mask_type, mask_array)
        
        if self.bee_mask is not None and self.annotation_type_visibility.get('bee', True):
            bee_ids = np.unique(self.bee_mask)
            bee_ids = bee_ids[bee_ids > 0]
            for instance_id in bee_ids:
                mask_lookup[instance_id] = ('bee', self.bee_mask)
        
        if self.chamber_mask is not None and self.annotation_type_visibility.get('chamber', True):
            chamber_ids = np.unique(self.chamber_mask)
            chamber_ids = chamber_ids[chamber_ids > 0]
            for instance_id in chamber_ids:
                mask_lookup[instance_id] = ('chamber', self.chamber_mask)
        
        if self.hive_mask is not None and self.annotation_type_visibility.get('hive', True):
            hive_ids = np.unique(self.hive_mask)
            hive_ids = hive_ids[hive_ids > 0]
            for instance_id in hive_ids:
                mask_lookup[instance_id] = ('hive', self.hive_mask)
        
        # Build rendering order: chamber and hive first (background), bees last (on top)
        chamber_ids_visible = [iid for iid, (cat, _) in mask_lookup.items() if cat == 'chamber']
        hive_ids_visible = [iid for iid, (cat, _) in mask_lookup.items() if cat == 'hive']
        bee_ids_visible = [iid for iid, (cat, _) in mask_lookup.items() if cat == 'bee']
        all_instance_ids = chamber_ids_visible + hive_ids_visible + bee_ids_visible
        
        # Draw segmentation masks if enabled
        if self.show_segmentations and len(all_instance_ids) > 0:
            # Get dimensions from the first non-None mask
            if self.bee_mask is not None:
                h, w = self.bee_mask.shape
            elif self.chamber_mask is not None:
                h, w = self.chamber_mask.shape
            else:
                h, w = self.hive_mask.shape
            
            # Create composite overlay for all instances
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Render each instance with category-based color
            for instance_id in all_instance_ids:
                # Get category and mask array
                category, mask_array = mask_lookup[instance_id]
                
                # Get metadata
                metadata = self.annotation_metadata.get(instance_id, {})
                
                # Get color for this instance
                if instance_id not in self.mask_colors:
                    # Assign color based on category
                    if category == 'chamber':
                        self.mask_colors[instance_id] = self.annotation_type_colors['chamber']
                    elif category == 'hive':
                        self.mask_colors[instance_id] = self.annotation_type_colors['hive']
                    else:  # bee
                        self.mask_colors[instance_id] = tuple(np.random.randint(0, 255, 3).tolist())
                
                color = self.mask_colors[instance_id]
                
                # Apply color with appropriate opacity
                mask_pixels = mask_array == instance_id
                if mask_pixels.shape[:2] != (h, w):
                    print(f"Warning: rebuild_visualizations skipping instance {instance_id} – "
                          f"mask shape {mask_pixels.shape[:2]} != overlay shape {(h, w)}")
                    continue
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
            
            # Add border around selected instance using the optimized method
            if self.selected_mask_idx > 0:
                self.update_selection_border()
            
            # Draw outlines for hive and chamber instances (helps distinguish overlapping regions)
            self._draw_instance_outlines(all_instance_ids, mask_lookup)
        else:
            self._cached_overlay = None
        
        # Draw bounding boxes if enabled
        if self.show_bboxes:
            self._draw_bboxes()
        
        # Update instance labels
        self.update_instance_labels()
    
    def _draw_bboxes(self):
        """Draw bounding boxes for all instances from masks and metadata"""
        from PyQt6.QtWidgets import QGraphicsRectItem
        from PyQt6.QtGui import QPen, QColor
        from PyQt6.QtCore import Qt
        
        # Draw bboxes for all instance types from their separate masks
        mask_sources = [
            ('bee', self.bee_mask),
            ('chamber', self.chamber_mask),
            ('hive', self.hive_mask)
        ]
        
        for category, mask_array in mask_sources:
            if mask_array is None:
                continue
            if not self.annotation_type_visibility.get(category, True):
                continue
            
            instance_ids = np.unique(mask_array)
            instance_ids = instance_ids[instance_ids > 0]
            
            for instance_id in instance_ids:
                if instance_id in self.bbox_items_map:
                    continue  # Already drawn
                # Get bbox from mask
                mask = (mask_array == instance_id)
                bbox = self.get_mask_bbox(mask.astype(np.uint8) * 255)
                
                if bbox is not None:
                    x, y, w, h = bbox
                    color = self.mask_colors.get(instance_id, (255, 255, 255))
                    
                    rect_item = QGraphicsRectItem(x, y, w, h)
                    pen = QPen(QColor(color[0], color[1], color[2]))
                    pen.setWidth(4)
                    rect_item.setPen(pen)
                    rect_item.setZValue(100)
                    
                    self.scene.addItem(rect_item)
                    self.mask_items.append(rect_item)
                    self.bbox_items_map[instance_id] = rect_item
        
        # Also draw bbox-only annotations from metadata
        self._draw_bboxes_from_metadata()
    
    def _draw_bboxes_from_metadata(self):
        """Draw bounding boxes from bbox-only annotations in metadata"""
        from PyQt6.QtWidgets import QGraphicsRectItem
        from PyQt6.QtGui import QPen, QColor
        from PyQt6.QtCore import Qt
        
        # Draw bbox-only annotations (those not already in bbox_items_map)
        for instance_id, metadata in self.annotation_metadata.items():
            # Skip if already drawn from a mask
            if instance_id in self.bbox_items_map:
                continue
            
            # Skip if this instance is currently being edited
            if instance_id == self.editing_instance_id:
                continue
            
            # Use the instance's own category for visibility check
            category = metadata.get('category', 'bee')
            if not self.annotation_type_visibility.get(category, True):
                continue
            
            # Only draw if it's a bbox-only annotation with a valid bbox
            if 'bbox' in metadata and metadata['bbox'] != [0, 0, 0, 0]:
                x, y, w, h = metadata['bbox']
                color = self.mask_colors.get(instance_id, (255, 255, 255))
                
                rect_item = QGraphicsRectItem(x, y, w, h)
                pen = QPen(QColor(color[0], color[1], color[2]))
                pen.setWidth(4)
                rect_item.setPen(pen)
                rect_item.setZValue(100)
                
                self.scene.addItem(rect_item)
                self.mask_items.append(rect_item)
                self.bbox_items_map[instance_id] = rect_item
    
    def _draw_instance_outlines(self, instance_ids, mask_lookup):
        """Draw contour outlines for hive and chamber instances to better distinguish overlapping regions
        
        Args:
            instance_ids: List of instance IDs to potentially draw outlines for
            mask_lookup: Dict mapping instance_id -> (category, mask_array)
        """
        from PyQt6.QtWidgets import QGraphicsPathItem
        from PyQt6.QtGui import QPainterPath, QPen, QColor
        
        # Only draw outlines for hive and chamber instances
        categories_to_outline = {'hive', 'chamber'}
        
        for instance_id in instance_ids:
            if instance_id not in mask_lookup:
                continue
            
            category, mask_array = mask_lookup[instance_id]
            
            # Only draw outlines for hive and chamber
            if category not in categories_to_outline:
                continue
            
            # Check if this category is visible
            if not self.annotation_type_visibility.get(category, True):
                continue
            
            # Extract binary mask for this instance
            binary_mask = (mask_array == instance_id).astype(np.uint8)
            
            # Find contours (both external and internal/holes)
            contours, hierarchy = cv2.findContours(
                binary_mask,
                cv2.RETR_CCOMP,  # Retrieve both external and internal contours
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
            
            # Get instance color for outline
            color = self.mask_colors.get(instance_id, (255, 255, 255))
            
            # Create path for all contours (external and internal)
            path = QPainterPath()
            
            if hierarchy is not None:
                hierarchy = hierarchy[0]  # Reshape from (1, n, 4) to (n, 4)
                
                for i, contour in enumerate(contours):
                    if len(contour) < 3:  # Skip degenerate contours
                        continue
                    
                    # Start at first point
                    first_point = contour[0][0]
                    path.moveTo(float(first_point[0]), float(first_point[1]))
                    
                    # Draw lines to other points
                    for point in contour[1:]:
                        pt = point[0]
                        path.lineTo(float(pt[0]), float(pt[1]))
                    
                    # Close the path
                    path.closeSubpath()
            else:
                # No hierarchy, just draw all contours
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    
                    first_point = contour[0][0]
                    path.moveTo(float(first_point[0]), float(first_point[1]))
                    
                    for point in contour[1:]:
                        pt = point[0]
                        path.lineTo(float(pt[0]), float(pt[1]))
                    
                    path.closeSubpath()
            
            # Create graphics item with contrasting outline
            # Use white outline with black shadow for good visibility
            outline_item = QGraphicsPathItem(path)
            
            # Use a brighter/contrasting color for the outline
            # White outline is most visible on colored backgrounds
            pen = QPen(QColor(255, 255, 255), 2)  # White, 2px thick
            outline_item.setPen(pen)
            outline_item.setZValue(5)  # Above masks (1) but below selection border (10)
            
            self.scene.addItem(outline_item)
            self.mask_items.append(outline_item)
    
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
            try:
                if label_item.scene() == self.scene:
                    self.scene.removeItem(label_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        self.label_items = []
        
        if not self.labels_visible:
            return
        
        # Show labels for bboxes if bbox view is enabled
        if self.show_bboxes and self.bbox_items_map:
            # Clean up any deleted items while iterating
            to_remove = []
            
            for instance_id, rect_item in self.bbox_items_map.items():
                try:
                    # Check category visibility - only show labels for visible categories
                    category = self.annotation_metadata.get(instance_id, {}).get('category', 'bee')
                    if not self.annotation_type_visibility.get(category, True):
                        continue  # Skip this instance if its category is not visible
                    
                    rect = rect_item.rect()
                    color = self.mask_colors.get(instance_id, (255, 255, 255))
                    
                    # Create text label with instance ID
                    label_text = str(instance_id)
                    text_item = QGraphicsTextItem(label_text)
                    
                    # Style the text with black outline
                    text_item.setHtml(
                        f'<div style="color: rgb({color[0]}, {color[1]}, {color[2]}); '
                        f'text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, '
                        f'-1px 0px 0 #000, 1px 0px 0 #000, 0px -1px 0 #000, 0px 1px 0 #000; '
                        f'font-size: 30px; font-weight: bold;">{label_text}</div>'
                    )
                    
                    # Position at top-left corner of bbox
                    text_item.setPos(rect.x() + 5, rect.y() + 5)
                    
                    # Set high z-value to appear above bboxes
                    text_item.setZValue(102)
                    
                    # Add to scene
                    self.scene.addItem(text_item)
                    self.label_items.append(text_item)
                    
                except (RuntimeError, AttributeError):
                    # Item has been deleted by Qt or is invalid
                    to_remove.append(instance_id)
            
            # Clean up deleted items from map
            for instance_id in to_remove:
                if instance_id in self.bbox_items_map:
                    del self.bbox_items_map[instance_id]
        
        # For mask mode, show labels at centroids (only if segmentations are shown)
        if not self.show_segmentations or self.combined_mask is None:
            return
        
        # Get unique instance IDs (excluding background 0)
        instance_ids = np.unique(self.combined_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        for idx, instance_id in enumerate(instance_ids, start=1):
            # Check category visibility - only show labels for visible categories
            category = self.annotation_metadata.get(instance_id, {}).get('category', 'bee')
            if not self.annotation_type_visibility.get(category, True):
                continue  # Skip this instance if its category is not visible
            
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
                f'text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000, '
                f'-1px 0px 0 #000, 1px 0px 0 #000, 0px -1px 0 #000, 0px 1px 0 #000; '
                f'font-size: 30px; font-weight: bold;">{label_text}</div>'
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
                try:
                    if label_item.scene() == self.scene:
                        self.scene.removeItem(label_item)
                except (RuntimeError, AttributeError):
                    # Item already deleted by Qt or invalid
                    pass
            self.label_items = []
    
    def toggle_labels(self):
        """Toggle visibility of instance number labels"""
        self.set_labels_visible(not self.labels_visible)
            
    def set_annotations(self, annotations, mask_colors=None):
        """Load annotations for current frame
        
        Args:
            annotations: List of annotation dictionaries (all types: bee, chamber, hive)
            mask_colors: Optional dict mapping mask_id to (r,g,b) color tuple
        """
        # Commit any pending edits first
        if self.editing_instance_id > 0:
            self.commit_editing()
        
        # Clear existing masks and free memory
        for mask_attr in ['bee_mask', 'chamber_mask', 'hive_mask']:
            if getattr(self, mask_attr) is not None:
                delattr(self, mask_attr)
                setattr(self, mask_attr, None)
        
        # Clear cached overlay
        if self._cached_overlay is not None:
            del self._cached_overlay
            self._cached_overlay = None
        
        for item in self.mask_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
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
        self.annotation_metadata = {}  # Clear metadata
        
        # Clear selection border
        if self.selection_border_item:
            try:
                if self.selection_border_item.scene() == self.scene:
                    self.scene.removeItem(self.selection_border_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
            try:
                self.selection_border_item.setPixmap(None)
            except:
                pass
            del self.selection_border_item
            self.selection_border_item = None
        
        # Reset selection
        self.selected_mask_idx = -1
        self.active_sam2_mask_idx = -1
        
        # Clear cached data for performance optimization
        self._cached_instance_ids = None
        self._cached_bboxes = {}
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        if not annotations:
            return
        
        # Determine dimensions from current image or first mask
        has_masks = any('mask' in ann for ann in annotations)
        h, w = None, None
        
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
        elif has_masks:
            first_mask_ann = next((ann for ann in annotations if 'mask' in ann), None)
            if first_mask_ann:
                h, w = first_mask_ann['mask'].shape[:2]
        
        if h is None or w is None:
            return  # Can't determine size
        
        # Initialize separate masks for each category
        self.bee_mask = np.zeros((h, w), dtype=np.int32)
        self.chamber_mask = np.zeros((h, w), dtype=np.int32)
        self.hive_mask = np.zeros((h, w), dtype=np.int32)
        
        # Build separate masks from annotations
        for ann in annotations:
            # Get instance ID
            mask_id = ann.get('mask_id', self.next_mask_id)
            if mask_id >= self.next_mask_id:
                self.next_mask_id = mask_id + 1
            
            # Get category (defaults to 'bee' for backward compatibility)
            category = ann.get('category', 'bee')
            
            # Determine which mask array to use
            if category == 'chamber':
                mask_array = self.chamber_mask
                color = self.annotation_type_colors['chamber']
            elif category == 'hive':
                mask_array = self.hive_mask
                color = self.annotation_type_colors['hive']
            else:  # bee
                mask_array = self.bee_mask
                color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Store or assign color
            if mask_colors and mask_id in mask_colors:
                self.mask_colors[mask_id] = mask_colors[mask_id]
            else:
                self.mask_colors[mask_id] = color
            
            if 'mask' in ann:
                # Annotation with mask
                mask = ann['mask']
                if not isinstance(mask, np.ndarray):
                    continue
                
                # Validate mask dimensions
                if mask.shape[:2] != (h, w):
                    print(f"Warning: Skipping instance {mask_id} - mask dimensions {mask.shape[:2]} don't match image dimensions {(h, w)}")
                    # Still store metadata for bbox-only fallback
                    metadata = {k: v for k, v in ann.items() if k != 'mask'}
                    metadata['bbox_only'] = True
                    metadata['category'] = category
                    self.annotation_metadata[mask_id] = metadata
                    continue
                
                # Store metadata (everything except mask)
                metadata = {k: v for k, v in ann.items() if k != 'mask'}
                metadata['category'] = category
                self.annotation_metadata[mask_id] = metadata
                
                # Add to the appropriate mask array
                binary_mask = mask > 0 if mask.dtype != bool else mask
                mask_array[binary_mask] = mask_id
            else:
                # Bbox-only annotation - store metadata for later drawing
                metadata = ann.copy()
                metadata['category'] = category
                self.annotation_metadata[mask_id] = metadata
        
        # Mark combined mask cache as dirty since we modified masks
        self._combined_mask_dirty = True
        
        # Rebuild visualizations once at the end
        self.rebuild_visualizations()
    
    def _set_bbox_annotations(self, bbox_annotations, mask_colors=None):
        """DEPRECATED: Bbox-only annotations are now handled in main set_annotations flow
        
        This method is kept for backward compatibility but should not be called
        since bbox-only annotations now go through the unified annotation path.
        """
        # This method is deprecated - bbox annotations are handled in set_annotations
        pass
    
    def get_annotations(self):
        """Get all current annotations (bee, chamber, and hive)
        
        Includes the editing mask if currently editing.
        Also includes bbox-only annotations if present.
        Preserves metadata including category.
        """
        # Flush any pending visualization updates from throttled drawing
        self._flush_pending_viz_update()
        
        annotations = []
        
        # Process all three mask types
        mask_sources = [
            ('bee', self.bee_mask),
            ('chamber', self.chamber_mask),
            ('hive', self.hive_mask)
        ]
        
        for category, mask_array in mask_sources:
            if mask_array is None:
                continue
            
            # Get unique instance IDs from this mask (excluding background 0)
            instance_ids = np.unique(mask_array)
            instance_ids = instance_ids[instance_ids > 0].tolist()
            
            # Add editing instance ID if currently editing and matches this category
            if (self.editing_instance_id > 0 and 
                self.editing_mask is not None and 
                self.annotation_metadata.get(self.editing_instance_id, {}).get('category', 'bee') == category):
                if self.editing_instance_id not in instance_ids:
                    instance_ids.append(self.editing_instance_id)
            
            for instance_id in instance_ids:
                # Check if this is the editing instance
                if instance_id == self.editing_instance_id and self.editing_mask is not None:
                    # Use editing mask
                    binary_mask = self.editing_mask.copy()
                else:
                    # Extract binary mask from the appropriate mask array
                    binary_mask = (mask_array == instance_id).astype(np.uint8) * 255
                
                # Start with stored metadata if available
                if instance_id in self.annotation_metadata:
                    ann = self.annotation_metadata[instance_id].copy()
                else:
                    ann = {}
                
                # Always update/add these core fields
                ann['mask'] = binary_mask
                ann['label'] = ann.get('label', f'instance_{instance_id}')
                ann['area'] = int(np.sum(binary_mask > 0))
                ann['mask_id'] = int(instance_id)
                ann['category'] = category
                
                # Auto-compute bbox from mask
                bbox = self.get_mask_bbox(binary_mask)
                if bbox is not None:
                    ann['bbox'] = list(bbox)  # [x, y, width, height]
                
                annotations.append(ann)
        
        # Also return bbox-only annotations (stored in metadata but not in combined_mask)
        # These don't have masks, just bboxes
        # Track which IDs were already added from combined_mask to avoid duplicates
        added_ids = {ann['mask_id'] for ann in annotations}
        
        for instance_id, metadata in self.annotation_metadata.items():
            if metadata.get('bbox_only', False) and instance_id not in added_ids:
                # Skip placeholder bboxes (0,0,0,0)
                if 'bbox' not in metadata or metadata['bbox'] == [0, 0, 0, 0]:
                    continue
                
                # This is a bbox-only annotation that wasn't already added
                ann = metadata.copy()
                ann['mask_id'] = int(instance_id)
                # Make sure bbox_only flag is set
                ann['bbox_only'] = True
                annotations.append(ann)
        
        return annotations
    
    def get_instance_ids(self):
        """
        Get list of all instance IDs (cached for performance)
        
        Returns:
            Sorted list of instance IDs
        """
        # Return cached list if available
        if self._cached_instance_ids is not None:
            return self._cached_instance_ids
        
        # Build instance ID list
        instance_ids_set = set()
        
        # 1. From all three mask arrays (instances with segmentations)
        for mask_array in [self.bee_mask, self.chamber_mask, self.hive_mask]:
            if mask_array is not None:
                mask_instance_ids = np.unique(mask_array)
                mask_instance_ids = mask_instance_ids[mask_instance_ids > 0]  # Exclude background (0)
                instance_ids_set.update(mask_instance_ids.tolist())
        
        # 2. From annotation_metadata (bbox-only instances)
        instance_ids_set.update(self.annotation_metadata.keys())
        
        # 3. Add editing instance if currently in editing mode
        if self.editing_instance_id > 0 and self.editing_mask is not None:
            instance_ids_set.add(self.editing_instance_id)
        
        # Cache and return sorted list
        self._cached_instance_ids = sorted(list(instance_ids_set))
        return self._cached_instance_ids
    
    def reassign_instance_id(self, old_id, new_id):
        """Reassign an instance from old_id to new_id (for ArUco tracking)
        
        Args:
            old_id: Current instance ID
            new_id: New instance ID to assign
            
        Returns:
            bool: True if successful, False if failed
        """
        if old_id == new_id:
            return True
        
        # Find which mask array contains this instance
        mask_array, category = self._get_mask_array_for_instance(old_id)
        
        if mask_array is None:
            # Try metadata-only (bbox-only annotation)
            if old_id not in self.annotation_metadata:
                return False
            mask_array = None  # Will handle metadata-only below
        
        # Check if new_id already exists
        existing_ids = self.get_instance_ids()
        if new_id in existing_ids and new_id != old_id:
            # Merge instances: new_id exists, so we're merging old_id into it
            print(f"  ArUco reassignment: Merging instance {old_id} into existing instance {new_id}")
            
            if mask_array is not None:
                # Get target mask array for new_id
                target_mask_array, target_category = self._get_mask_array_for_instance(new_id)
                
                if target_mask_array is not None and target_category == category:
                    # Merge masks: pixels from old_id become new_id
                    target_mask_array[mask_array == old_id] = new_id
                    # Clear old_id from original mask
                    mask_array[mask_array == old_id] = 0
                else:
                    print(f"  Warning: Cannot merge - target instance {new_id} not found or category mismatch")
                    return False
            
            # Merge metadata (prefer existing new_id metadata, but add marker info from old_id if present)
            if old_id in self.annotation_metadata:
                old_metadata = self.annotation_metadata.pop(old_id)
                # Only transfer marker info if new_id doesn't have it
                if new_id in self.annotation_metadata:
                    if 'marker' in old_metadata and 'marker' not in self.annotation_metadata[new_id]:
                        self.annotation_metadata[new_id]['marker'] = old_metadata['marker']
            
            # Keep color of new_id
            if old_id in self.mask_colors:
                self.mask_colors.pop(old_id)
                
        else:
            # Simple reassignment: old_id becomes new_id
            if mask_array is not None:
                # Update mask array
                mask_array[mask_array == old_id] = new_id
            
            # Update metadata
            if old_id in self.annotation_metadata:
                metadata = self.annotation_metadata.pop(old_id)
                self.annotation_metadata[new_id] = metadata
            
            # Update colors
            if old_id in self.mask_colors:
                color = self.mask_colors.pop(old_id)
                self.mask_colors[new_id] = color
            
            # Update next_mask_id if needed
            if new_id >= self.next_mask_id:
                self.next_mask_id = new_id + 1
        
        # Clear caches
        self._cached_instance_ids = None
        self._cached_bboxes = {}
        if self._cached_overlay is not None:
            del self._cached_overlay
            self._cached_overlay = None
        
        # Update selection if needed
        if self.selected_mask_idx == old_id:
            self.selected_mask_idx = new_id
        if self.editing_instance_id == old_id:
            self.editing_instance_id = new_id
        
        return True
    
    def get_instance_bbox_cached(self, instance_id):
        """
        Get bounding box for an instance (cached for performance)
        
        Args:
            instance_id: Instance ID
            
        Returns:
            tuple: (x, y, width, height) or None if mask is empty/invalid
        """
        # Check cache first
        if instance_id in self._cached_bboxes:
            return self._cached_bboxes[instance_id]
        
        bbox = None
        
        # Try to get bbox from editing mask
        if instance_id == self.editing_instance_id and self.editing_mask is not None:
            bbox = self.get_mask_bbox(self.editing_mask)
        # Try to get bbox from combined_mask
        elif self.combined_mask is not None and instance_id > 0:
            instance_mask = (self.combined_mask == instance_id)
            if np.any(instance_mask):
                bbox = self.get_mask_bbox(instance_mask)
        
        # Try to get bbox from metadata (for bbox-only annotations)
        if bbox is None and instance_id in self.annotation_metadata:
            metadata = self.annotation_metadata[instance_id]
            if 'bbox' in metadata and metadata['bbox'] != [0, 0, 0, 0]:
                bbox = tuple(metadata['bbox'])
        
        # Cache the result (even if None)
        if bbox is not None:
            self._cached_bboxes[instance_id] = bbox
        
        return bbox
    
    def zoom_to_instance_fast(self, idx, padding=50):
        """Fast version of zoom_to_instance using cached bounding box
        
        Args:
            idx: Instance ID (mask_id) to zoom to
            padding: Extra space around the instance (in pixels)
        """
        bbox = self.get_instance_bbox_cached(idx)
        if bbox:
            x, y, width, height = bbox
            self.zoom_to_rect(x, y, width, height, padding)
        
    def set_selected_instance(self, idx):
        """Set selected instance by ID and zoom to it"""
        self.selected_mask_idx = idx
        
        # If brush/eraser tool is active and instance has a segmentation, start editing mode
        # For bbox-only instances, wait until user actually starts drawing
        if self.current_tool in ['brush', 'eraser']:
            # Check if instance has segmentation data
            has_segmentation = (
                self.combined_mask is not None and 
                np.any(self.combined_mask == idx)
            )
            if has_segmentation:
                # Zoom to instance before entering editing mode using fast cached version
                self.zoom_to_instance_fast(idx)
                
                # Now start editing
                self.start_editing_instance(idx)
                return
            # For bbox-only, just select and wait for user to start drawing
        
        # Try to zoom to the instance using cached bbox (much faster)
        # If bbox graphics item exists (bboxes are visible), use it
        if idx in self.bbox_items_map:
            self._select_bbox(idx)
        else:
            # Use cached bbox for fast zooming
            self.zoom_to_instance_fast(idx)
    
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
        # Commit any edits to previous instance before switching
        if self.editing_instance_id > 0 and self.editing_instance_id != idx:
            self.commit_editing()

        # In bbox view mode, don't rebuild visualizations (would clear bbox selection)
        if self.show_bboxes:
            self.selected_mask_idx = idx
            return
        
        self.selected_mask_idx = idx
        
        # PERFORMANCE OPTIMIZATION: Only update the selection border instead of rebuilding everything
        # This is much faster when there are many instances
        self.update_selection_border()
            
    def refresh_all_visualizations(self):
        """Refresh all mask visualizations with current selection highlighting"""
        self.rebuild_visualizations()
            
    def update_selection_border(self):
        """Update the selection border for the currently selected instance
        
        This is a lightweight method that only updates the border without rebuilding
        all visualizations. Much faster for instance switching.
        """
        # Remove old border
        if self.selection_border_item:
            try:
                if self.selection_border_item.scene() == self.scene:
                    self.scene.removeItem(self.selection_border_item)
            except (RuntimeError, AttributeError):
                pass
            self.selection_border_item = None
        
        # Add new border if we have a valid selection
        if self.selected_mask_idx > 0 and self.combined_mask is not None:
            # Check if the selected instance's category is visible
            metadata = self.annotation_metadata.get(self.selected_mask_idx, {})
            category = metadata.get('category', 'bee')
            if not self.annotation_type_visibility.get(category, True):
                # Category is hidden, don't show selection border
                return
            # Check if instance exists in combined_mask or is being edited
            if self.selected_mask_idx == self.editing_instance_id and self.editing_mask is not None:
                # Use editing mask
                binary_mask = self.editing_mask
            elif np.any(self.combined_mask == self.selected_mask_idx):
                # Use combined mask
                binary_mask = (self.combined_mask == self.selected_mask_idx).astype(np.uint8) * 255
            else:
                # No mask to highlight (bbox-only instance)
                return
            
            self.add_selection_border(binary_mask)
    
    def add_selection_border(self, mask):
        """Add a border around the selected mask"""
        # Remove previous border if it exists
        if self.selection_border_item:
            try:
                if self.selection_border_item.scene() == self.scene:
                    self.scene.removeItem(self.selection_border_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
            self.selection_border_item = None
        
        # Check if the selected instance's category is visible
        if self.selected_mask_idx > 0:
            metadata = self.annotation_metadata.get(self.selected_mask_idx, {})
            category = metadata.get('category', 'bee')
            if not self.annotation_type_visibility.get(category, True):
                # Category is hidden, don't show selection border
                return
        
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
        
        # Get the appropriate mask array for this instance
        mask_array, category = self._get_mask_array_for_instance(instance_id)
        
        # Ensure mask exists
        if mask_array is None:
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                # Create the appropriate mask based on category
                if category == 'chamber' or self.current_annotation_type == 'chamber':
                    if self.chamber_mask is None:
                        self.chamber_mask = np.zeros((h, w), dtype=np.int32)
                    mask_array = self.chamber_mask
                elif category == 'hive' or self.current_annotation_type == 'hive':
                    if self.hive_mask is None:
                        self.hive_mask = np.zeros((h, w), dtype=np.int32)
                    mask_array = self.hive_mask
                else:  # bee
                    if self.bee_mask is None:
                        self.bee_mask = np.zeros((h, w), dtype=np.int32)
                    mask_array = self.bee_mask
            else:
                return
        
        h, w = mask_array.shape
        self.editing_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Check if this instance exists in mask_array or is bbox-only
        if np.any(mask_array == instance_id):
            # Instance has segmentation - extract it
            instance_pixels = mask_array == instance_id
            self.editing_mask[instance_pixels] = 255
            mask_array[instance_pixels] = 0
            # Mark combined mask cache as dirty
            self._combined_mask_dirty = True
        else:
            # Instance is bbox-only or new - start with empty editing mask
            # User will draw segmentation with brush or SAM2
            pass
        
        self.editing_instance_id = instance_id
        self.selected_mask_idx = instance_id  # Sync selected_mask_idx with editing_instance_id
        
        # Stamp the correct category into annotation_metadata for new instances.
        # _get_mask_array_for_instance defaults to 'bee' when the instance is brand-new
        # (not yet in metadata), so we must record the actual current type here so
        # that get_annotations() and commit_editing() both resolve the right category.
        if instance_id not in self.annotation_metadata:
            self.annotation_metadata[instance_id] = {}
        if 'category' not in self.annotation_metadata[instance_id]:
            self.annotation_metadata[instance_id]['category'] = self.current_annotation_type
        
        # Rebuild base visualization (without the editing instance)
        self.rebuild_visualizations()
        
        # Show editing mask on top
        self._update_editing_visualization()
    
    def commit_editing(self):
        """Commit editing mask back to combined mask
        
        Merges the temporary editing mask back into the main combined mask.
        """
        # Flush any pending visualization updates first
        if self._viz_update_timer.isActive():
            self._viz_update_timer.stop()
            self._flush_pending_viz_update()
        
        if self.editing_instance_id <= 0 or self.editing_mask is None:
            return
        
        # Check if editing mask is empty (no pixels drawn)
        has_pixels = np.any(self.editing_mask > 0)
        
        if not has_pixels:
            # Empty editing mask - don't commit, just exit editing mode
            # This preserves bbox-only instances that were selected but not drawn on
            self.editing_mask = None
            self.editing_instance_id = -1
            
            # Clear editing overlay cache
            if self._editing_overlay_cache is not None:
                del self._editing_overlay_cache
                self._editing_overlay_cache = None
            
            # Remove editing visualization
            if self.editing_mask_item:
                try:
                    if self.editing_mask_item.scene() == self.scene:
                        self.scene.removeItem(self.editing_mask_item)
                except (RuntimeError, AttributeError):
                    # Item already deleted by Qt or invalid
                    pass
                try:
                    self.editing_mask_item.setPixmap(None)
                except:
                    pass
                del self.editing_mask_item
                self.editing_mask_item = None
            
            # Rebuild visualization to restore bbox-only instances
            self.rebuild_visualizations()
            return
        
        # Create appropriate mask if it doesn't exist (e.g., when converting bbox-only to mask)
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
        else:
            return
        
        # Get category from metadata or use current annotation type
        category = self.annotation_metadata.get(self.editing_instance_id, {}).get('category', self.current_annotation_type)
        
        # Get or create the appropriate mask array
        if category == 'chamber':
            if self.chamber_mask is None:
                self.chamber_mask = np.zeros((h, w), dtype=np.int32)
            mask_array = self.chamber_mask
        elif category == 'hive':
            if self.hive_mask is None:
                self.hive_mask = np.zeros((h, w), dtype=np.int32)
            mask_array = self.hive_mask
        else:  # bee
            if self.bee_mask is None:
                self.bee_mask = np.zeros((h, w), dtype=np.int32)
            mask_array = self.bee_mask
        
        # Merge editing mask back into appropriate mask array
        editing_pixels = self.editing_mask > 0
        mask_array[editing_pixels] = self.editing_instance_id
        
        # Mark combined mask cache as dirty
        self._combined_mask_dirty = True
        
        # Invalidate cached instance IDs (in case this was a new instance)
        self._cached_instance_ids = None
        
        # Persist the resolved category in annotation_metadata so subsequent
        # get_annotations() calls always return the right category.
        if self.editing_instance_id not in self.annotation_metadata:
            self.annotation_metadata[self.editing_instance_id] = {}
        self.annotation_metadata[self.editing_instance_id]['category'] = category
        
        # If this was a bbox-only annotation, remove the bbox_only flag from metadata
        if self.editing_instance_id in self.annotation_metadata:
                if 'bbox_only' in self.annotation_metadata[self.editing_instance_id]:
                    del self.annotation_metadata[self.editing_instance_id]['bbox_only']
        
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
            try:
                if self.editing_mask_item.scene() == self.scene:
                    self.scene.removeItem(self.editing_mask_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
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
            try:
                if self.editing_mask_item.scene() == self.scene:
                    self.scene.removeItem(self.editing_mask_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
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
        # Always show editing mask when in editing mode, regardless of show_segmentations
        self.editing_mask_item.setVisible(True)
        self.scene.addItem(self.editing_mask_item)
        
        # Update selection border for editing mask (skip if actively drawing)
        if not self._is_actively_drawing and self.selected_mask_idx == self.editing_instance_id:
            self.add_selection_border(self.editing_mask)
    
    def _flush_pending_viz_update(self):
        """Flush pending visualization update (called by timer)"""
        if self._dirty_pixels is not None:
            self._update_editing_visualization_incremental(self._dirty_pixels)
            self._dirty_pixels = None
    
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
        
        # Update the pixmap from cached overlay (optimized: reuse buffer)
        h, w = self._editing_overlay_cache.shape[:2]
        # Create QImage directly from buffer without copy for better performance
        # The buffer is kept alive by self._editing_overlay_cache
        q_image = QImage(self._editing_overlay_cache.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        # Only copy when creating pixmap to avoid tearing
        pixmap = QPixmap.fromImage(q_image)
        
        # Update existing item
        if self.editing_mask_item:
            self.editing_mask_item.setPixmap(pixmap)
        else:
            # Create item if it doesn't exist
            self.editing_mask_item = QGraphicsPixmapItem(pixmap)
            self.editing_mask_item.setZValue(2)
            # Always show editing mask when in editing mode
            self.editing_mask_item.setVisible(True)
            self.scene.addItem(self.editing_mask_item)
    
    def clear_selected_instance(self):
        """Remove the selected instance from the combined mask"""
        if self.selected_mask_idx <= 0:
            return
        
        instance_to_clear = self.selected_mask_idx
        
        # If editing this instance, clear the editing mask and exit editing mode
        if self.editing_instance_id == instance_to_clear:
            self.editing_mask = None
            self.editing_instance_id = -1
            # Remove editing visualization
            if self.editing_mask_item:
                try:
                    if self.editing_mask_item.scene() == self.scene:
                        self.scene.removeItem(self.editing_mask_item)
                except (RuntimeError, AttributeError):
                    # Item already deleted by Qt or invalid
                    pass
                self.editing_mask_item = None
        else:
            # Not in editing mode, directly clear from the appropriate mask array
            # Find which mask array contains this instance
            mask_array, category = self._get_mask_array_for_instance(instance_to_clear)
            if mask_array is not None:
                mask_array[mask_array == instance_to_clear] = 0
                # Mark combined mask cache as dirty
                self._combined_mask_dirty = True
                # Invalidate cached instance IDs since we removed one
                self._cached_instance_ids = None
        
        # Remove from annotation_metadata (for bbox-only instances or metadata)
        if instance_to_clear in self.annotation_metadata:
            del self.annotation_metadata[instance_to_clear]
        
        # Remove color entry
        if instance_to_clear in self.mask_colors:
            del self.mask_colors[instance_to_clear]
        
        # Remove from bbox_items_map if present
        if instance_to_clear in self.bbox_items_map:
            rect_item = self.bbox_items_map[instance_to_clear]
            try:
                if rect_item.scene() == self.scene:
                    self.scene.removeItem(rect_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
            del self.bbox_items_map[instance_to_clear]
        
        # Rebuild visualization
        self.rebuild_visualizations()
        
        # Clear prompt points
        for item in self.prompt_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
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
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        
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
            try:
                if self.selection_border_item.scene() == self.scene:
                    self.scene.removeItem(self.selection_border_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
            self.selection_border_item = None
        
        # Reset active mask index
        self.active_sam2_mask_idx = -1
    
    def start_new_sam2_instance(self):
        """Start annotating a new SAM2 instance"""
        # Clear prompts and reset for new instance
        for item in self.prompt_items:
            try:
                if item.scene() == self.scene:
                    self.scene.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
        
        self.positive_points = []
        self.negative_points = []
        self.prompt_items = []
        self.active_sam2_mask_idx = -1
        self.selected_mask_idx = -1
        
        # Remove selection border
        if self.selection_border_item:
            try:
                if self.selection_border_item.scene() == self.scene:
                    self.scene.removeItem(self.selection_border_item)
            except (RuntimeError, AttributeError):
                # Item already deleted by Qt or invalid
                pass
            self.selection_border_item = None
    
    def get_prompt_points(self):
        """Get all accumulated prompt points for SAM2"""
        return {
            'positive': self.positive_points.copy(),
            'negative': self.negative_points.copy()
        }
    
    def set_show_segmentations(self, show):
        """Enable or disable segmentation mask display"""
        self.show_segmentations = show
        self.rebuild_visualizations()
    
    def set_show_bboxes(self, show):
        """Enable or disable bounding box display"""
        self.show_bboxes = show
        if not show:
            self._deselect_bbox()
        self.rebuild_visualizations()
    
    def set_bbox_mode(self, enabled):
        """DEPRECATED: Use set_show_bboxes instead"""
        self.set_show_bboxes(enabled)
    
    def _get_bbox_at_pos(self, pos):
        """Get bbox ID at scene position
        
        Args:
            pos: QPointF scene position
            
        Returns:
            mask_id of bbox at position, or None
        """
        # Clean up any deleted items while iterating
        to_remove = []
        
        for mask_id, rect_item in self.bbox_items_map.items():
            try:
                if rect_item.rect().contains(pos):
                    return mask_id
            except (RuntimeError, AttributeError):
                # Item has been deleted by Qt or is invalid
                to_remove.append(mask_id)
        
        # Clean up deleted items from map
        for mask_id in to_remove:
            del self.bbox_items_map[mask_id]
        
        return None
    
    def _select_bbox(self, mask_id):
        """Select a bbox for editing
        
        Args:
            mask_id: ID of bbox to select
        """
        if mask_id not in self.bbox_items_map:
            return
        
        # Deselect previous bbox
        if self.selected_bbox_id and self.selected_bbox_id != mask_id:
            self._deselect_bbox()
        
        self.selected_bbox_id = mask_id
        self.selected_mask_idx = mask_id  # Keep instance selection in sync
        rect_item = self.bbox_items_map[mask_id]
        
        # Highlight selected bbox by increasing pen width
        color = self.mask_colors.get(mask_id, (255, 255, 0))
        pen = QPen(QColor(color[0], color[1], color[2]))
        pen.setWidth(6)  # Thicker for selected bbox
        rect_item.setPen(pen)
        rect_item.setZValue(101)  # Bring to front
        
        # Always allow bbox editing — the bbox is independent metadata
        # even when the instance also has a segmentation mask
        self._create_bbox_handles()
        self.setToolTip("Drag corners to resize, drag center to move bbox")
        
        # Zoom to bbox
        rect = rect_item.rect()
        self.zoom_to_rect(int(rect.x()), int(rect.y()), 
                         int(rect.width()), int(rect.height()), padding=100)
    
    def _deselect_bbox(self):
        """Deselect currently selected bbox"""
        if self.selected_bbox_id and self.selected_bbox_id in self.bbox_items_map:
            rect_item = self.bbox_items_map[self.selected_bbox_id]
            
            # Reset pen to normal
            color = self.mask_colors.get(self.selected_bbox_id, (255, 255, 0))
            pen = QPen(QColor(color[0], color[1], color[2]))
            pen.setWidth(4)  # Normal width
            rect_item.setPen(pen)
            rect_item.setZValue(100)  # Normal z-order
        
        # Remove handles
        for handle in self.bbox_handles:
            try:
                self.scene.removeItem(handle)
            except RuntimeError:
                pass
        self.bbox_handles = []
        
        self.selected_bbox_id = None
        if self.show_bboxes:
            self.setToolTip("Click on a bbox to select and edit")
    
    def _create_new_bbox(self, rect):
        """Create a new bbox annotation from a drawn rectangle
        
        Args:
            rect: QRectF of the drawn bbox
        """
        from PyQt6.QtWidgets import QGraphicsRectItem
        from PyQt6.QtGui import QPen, QColor
        
        # Use selected instance ID if available, otherwise create new
        if self.selected_mask_idx > 0:
            mask_id = self.selected_mask_idx
        else:
            mask_id = self.next_mask_id
            self.next_mask_id += 1
        
        # Assign color if not already assigned (use category-appropriate color)
        if mask_id not in self.mask_colors:
            if self.current_annotation_type == 'chamber':
                color = self.annotation_type_colors['chamber']  # Red
            elif self.current_annotation_type == 'hive':
                color = self.annotation_type_colors['hive']  # Yellow
            else:
                color = tuple(np.random.randint(0, 255, 3).tolist())
            self.mask_colors[mask_id] = color
        else:
            color = self.mask_colors[mask_id]
        
        # Create bbox metadata
        bbox = [int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())]
        
        # Update or create metadata for this instance
        if mask_id not in self.annotation_metadata:
            self.annotation_metadata[mask_id] = {}
        
        # Preserve existing category if set; otherwise use current annotation type
        existing_category = self.annotation_metadata[mask_id].get('category', self.current_annotation_type)
        
        self.annotation_metadata[mask_id].update({
            'mask_id': mask_id,
            'bbox': bbox,
            'bbox_only': True,  # Mark as bbox-only until segmentation is added
            'category': existing_category
        })
        
        # Invalidate cached instance IDs since we created a new one
        self._cached_instance_ids = None
        
        # Create rectangle item for visualization
        rect_item = QGraphicsRectItem(rect)
        pen = QPen(QColor(color[0], color[1], color[2]))
        pen.setWidth(4)
        rect_item.setPen(pen)
        rect_item.setZValue(100)
        
        self.scene.addItem(rect_item)
        self.mask_items.append(rect_item)
        self.bbox_items_map[mask_id] = rect_item
        
        # Update labels if visible
        self.update_instance_labels()
        
        # Select the newly created bbox
        self._select_bbox(mask_id)
        
        # Emit annotation changed signal
        self.annotation_changed.emit()
        
        print(f"Created new bbox with ID {mask_id}: {bbox}")
    
    def _create_bbox_handles(self):
        """Create corner handles for selected bbox"""
        from PyQt6.QtWidgets import QGraphicsEllipseItem
        
        # Remove old handles
        for handle in self.bbox_handles:
            try:
                self.scene.removeItem(handle)
            except RuntimeError:
                pass
        self.bbox_handles = []
        
        if not self.selected_bbox_id or self.selected_bbox_id not in self.bbox_items_map:
            return
        
        rect = self.bbox_items_map[self.selected_bbox_id].rect()
        handle_size = 10
        handle_color = QColor(255, 255, 0)  # Yellow handles
        
        # Create corner handles
        corners = {
            'tl': rect.topLeft(),
            'tr': rect.topRight(),
            'bl': rect.bottomLeft(),
            'br': rect.bottomRight()
        }
        
        for corner_id, corner_pos in corners.items():
            handle = QGraphicsEllipseItem(
                corner_pos.x() - handle_size/2,
                corner_pos.y() - handle_size/2,
                handle_size, handle_size
            )
            handle.setBrush(QBrush(handle_color))
            handle.setPen(QPen(QColor(0, 0, 0), 1))
            handle.setZValue(102)  # Above bbox
            handle.setData(0, corner_id)  # Store corner ID
            self.scene.addItem(handle)
            self.bbox_handles.append(handle)
    
    def _update_bbox_handles(self):
        """Update corner handle positions"""
        if not self.selected_bbox_id or self.selected_bbox_id not in self.bbox_items_map:
            return
        
        rect = self.bbox_items_map[self.selected_bbox_id].rect()
        handle_size = 10
        
        corners = {
            'tl': rect.topLeft(),
            'tr': rect.topRight(),
            'bl': rect.bottomLeft(),
            'br': rect.bottomRight()
        }
        
        for handle in self.bbox_handles:
            corner_id = handle.data(0)
            if corner_id in corners:
                corner_pos = corners[corner_id]
                handle.setRect(
                    corner_pos.x() - handle_size/2,
                    corner_pos.y() - handle_size/2,
                    handle_size, handle_size
                )
    
    def _get_bbox_handle_at_pos(self, pos):
        """Get bbox handle at scene position
        
        Args:
            pos: QPointF scene position
            
        Returns:
            Handle ID ('tl', 'tr', 'bl', 'br') or None
        """
        for handle in self.bbox_handles:
            try:
                if handle.contains(pos):
                    return handle.data(0)
            except RuntimeError:
                # Handle has been deleted by Qt
                continue
        return None
    
    
    def set_annotation_type_visibility(self, annotation_type, visible, rebuild=True):
        """Set visibility for a specific annotation type
        
        Args:
            annotation_type: Annotation type ('bee', 'hive', or 'chamber')
            visible: Boolean visibility state
            rebuild: If True, rebuild visualizations immediately (default: True)
        """
        if annotation_type in self.annotation_type_visibility:
            self.annotation_type_visibility[annotation_type] = visible
            if rebuild:
                self.rebuild_visualizations()
    
    
    # Chamber and Hive mask management methods
    
    def set_annotation_type(self, annotation_type, rebuild=True):
        """Set the current annotation type (bee/chamber/hive)
        
        Args:
            annotation_type: 'bee', 'chamber', or 'hive'
            rebuild: If True, rebuild visualizations immediately (default: True)
        """
        if annotation_type not in ['bee', 'chamber', 'hive']:
            return

        if annotation_type == self.current_annotation_type:
            return  # No change — nothing to do

        # Commit any active editing before switching type so the
        # edited mask is written back to the correct array and the
        # editing overlay is cleared.
        if self.editing_instance_id > 0:
            self.commit_editing()

        # Deselect current instance — it belongs to the old type
        self.selected_mask_idx = -1
        self.active_sam2_mask_idx = -1

        self.current_annotation_type = annotation_type

        # Rebuild so only the new type's overlay is shown
        if rebuild:
            self.rebuild_visualizations()
    
    def get_annotation_type(self):
        """Get the current annotation type
        
        Returns:
            String: 'bee', 'chamber', or 'hive'
        """
        return self.current_annotation_type
