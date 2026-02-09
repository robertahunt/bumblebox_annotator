"""
Validation visualization viewer
"""

import cv2
import numpy as np
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QScrollArea, QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from pathlib import Path


class ValidationViewer(QDialog):
    """Dialog to view validation predictions"""
    
    def __init__(self, validation_results, parent=None):
        super().__init__(parent)
        self.validation_results = validation_results
        self.current_index = 0
        
        self.setWindowTitle("Validation Predictions")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(700)
        
        self.init_ui()
        self.show_current_sample()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title showing current sample
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(self.title_label)
        
        # Scroll area for images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.images_container = QWidget()
        self.images_layout = QHBoxLayout()
        self.images_container.setLayout(self.images_layout)
        scroll.setWidget(self.images_container)
        
        layout.addWidget(scroll)
        
        # Info labels
        self.info_layout = QVBoxLayout()
        self.gt_info_label = QLabel()
        self.pred_info_label = QLabel()
        self.info_layout.addWidget(self.gt_info_label)
        self.info_layout.addWidget(self.pred_info_label)
        layout.addLayout(self.info_layout)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("◀ Previous")
        self.prev_button.clicked.connect(self.show_previous)
        button_layout.addWidget(self.prev_button)
        
        self.sample_label = QLabel()
        self.sample_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(self.sample_label)
        
        self.next_button = QPushButton("Next ▶")
        self.next_button.clicked.connect(self.show_next)
        button_layout.addWidget(self.next_button)
        
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def show_current_sample(self):
        """Display current validation sample"""
        if not self.validation_results or self.current_index >= len(self.validation_results):
            return
        
        sample = self.validation_results[self.current_index]
        
        # Update title
        image_path = Path(sample['image_path'])
        self.title_label.setText(f"Sample {self.current_index + 1}: {image_path.name}")
        
        # Update sample counter
        self.sample_label.setText(f"{self.current_index + 1} / {len(self.validation_results)}")
        
        # Enable/disable navigation buttons
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.validation_results) - 1)
        
        # Clear previous images
        while self.images_layout.count():
            child = self.images_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Get image and annotations
        image = sample['image']
        ground_truth = sample['ground_truth']
        predictions = sample['predictions']
        
        # Create ground truth visualization
        gt_image = self.visualize_ground_truth(image, ground_truth)
        gt_widget = self.create_image_widget(gt_image, "Ground Truth")
        self.images_layout.addWidget(gt_widget)
        
        # Create predictions visualization
        pred_image = self.visualize_predictions(image, predictions)
        pred_widget = self.create_image_widget(pred_image, "Predictions")
        self.images_layout.addWidget(pred_widget)
        
        # Update info labels
        num_gt = len(ground_truth)
        num_pred = len(predictions)
        
        self.gt_info_label.setText(f"Ground Truth: {num_gt} instances")
        
        if num_pred > 0:
            scores = predictions.scores.numpy()
            avg_conf = np.mean(scores)
            self.pred_info_label.setText(
                f"Predictions: {num_pred} instances (avg confidence: {avg_conf:.2f})"
            )
        else:
            self.pred_info_label.setText("Predictions: 0 instances")
    
    def visualize_ground_truth(self, image, annotations):
        """Draw ground truth annotations"""
        vis_image = image.copy()
        
        for ann in annotations:
            if 'segmentation' in ann:
                # Draw mask
                mask = self.polygon_to_mask(ann['segmentation'], image.shape[:2])
                color = np.random.randint(50, 255, 3).tolist()
                vis_image[mask > 0] = vis_image[mask > 0] * 0.5 + np.array(color) * 0.5
            
            if 'bbox' in ann:
                # Draw bounding box
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return vis_image
    
    def visualize_predictions(self, image, predictions):
        """Draw predicted instances"""
        vis_image = image.copy()
        
        if len(predictions) == 0:
            return vis_image
        
        # Get masks and boxes
        masks = predictions.pred_masks.numpy()
        boxes = predictions.pred_boxes.tensor.numpy()
        scores = predictions.scores.numpy()
        
        for i in range(len(predictions)):
            mask = masks[i]
            box = boxes[i]
            score = scores[i]
            
            # Draw mask
            color = np.random.randint(50, 255, 3).tolist()
            vis_image[mask] = vis_image[mask] * 0.5 + np.array(color) * 0.5
            
            # Draw box with confidence
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add confidence label
            label = f"{score:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_image
    
    def polygon_to_mask(self, segmentation, shape):
        """Convert polygon segmentation to binary mask"""
        mask = np.zeros(shape, dtype=np.uint8)
        
        for polygon in segmentation:
            if len(polygon) >= 6:  # At least 3 points
                pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        
        return mask
    
    def create_image_widget(self, image, title):
        """Create widget containing image with title"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Image
        image_label = QLabel()
        
        # Convert to QPixmap
        h, w = image.shape[:2]
        
        # Resize if too large
        max_width = 550
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            h, w = new_h, new_w
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        q_image = QImage(image_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        image_label.setPixmap(pixmap)
        layout.addWidget(image_label)
        
        widget.setLayout(layout)
        return widget
    
    def show_next(self):
        """Show next sample"""
        if self.current_index < len(self.validation_results) - 1:
            self.current_index += 1
            self.show_current_sample()
    
    def show_previous(self):
        """Show previous sample"""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_sample()
