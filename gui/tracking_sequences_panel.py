"""
UI panel for managing tracking sequences
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QListWidget, QListWidgetItem, QLabel, QMessageBox,
                             QDialog, QFormLayout, QSpinBox, QTextEdit, QCheckBox,
                             QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from pathlib import Path

from core.tracking_sequence_manager import TrackingSequence


class CreateSequenceDialog(QDialog):
    """Dialog for creating a new tracking sequence"""
    
    def __init__(self, video_id: str, current_frame: int, parent=None):
        super().__init__(parent)
        self.video_id = video_id
        self.current_frame = current_frame
        self.setWindowTitle("Create Tracking Sequence")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Info
        info_label = QLabel(
            f"<b>Video:</b> {self.video_id}<br>"
            f"<b>Current Frame:</b> {self.current_frame}<br><br>"
            "Create a tracking sequence from consecutive annotated frames. "
            "The sequence will be used to evaluate tracking algorithms."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Frame range
        frame_group = QGroupBox("Frame Range")
        frame_layout = QFormLayout()
        
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 999999)
        self.start_spin.setValue(self.current_frame)
        self.start_spin.valueChanged.connect(self.update_length)
        frame_layout.addRow("Start Frame:", self.start_spin)
        
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 999999)
        self.end_spin.setValue(self.current_frame + 1)  # Default to 2 frames
        self.end_spin.valueChanged.connect(self.update_length)
        frame_layout.addRow("End Frame:", self.end_spin)
        
        self.length_label = QLabel("2 frames")
        frame_layout.addRow("Length:", self.length_label)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # Notes
        notes_group = QGroupBox("Notes (Optional)")
        notes_layout = QVBoxLayout()
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.setPlaceholderText("e.g., 'High activity period', 'Bees entering/exiting', etc.")
        notes_layout.addWidget(self.notes_edit)
        
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)
        
        # Warning
        warning_label = QLabel(
            "⚠️ Make sure all frames in the range have ground truth annotations!\n"
            "The system will check this when you create the sequence."
        )
        warning_label.setStyleSheet("color: #b8860b; padding: 10px; background-color: #fff9e6; border-radius: 5px;")
        warning_label.setWordWrap(True)
        layout.addWidget(warning_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.create_btn = QPushButton("Create Sequence")
        self.create_btn.clicked.connect(self.accept)
        self.create_btn.setDefault(True)
        button_layout.addWidget(self.create_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.update_length()
    
    def update_length(self):
        """Update length label"""
        length = self.end_spin.value() - self.start_spin.value() + 1
        if length < 1:
            self.length_label.setText(f"<span style='color: red;'>Invalid (end must be >= start)</span>")
            self.create_btn.setEnabled(False)
        else:
            self.length_label.setText(f"{length} frames")
            self.create_btn.setEnabled(True)
    
    def get_config(self):
        """Get configuration from dialog"""
        return {
            'start_frame': self.start_spin.value(),
            'end_frame': self.end_spin.value(),
            'notes': self.notes_edit.toPlainText().strip()
        }


class EditSequenceDialog(QDialog):
    """Dialog for editing an existing tracking sequence"""
    
    def __init__(self, sequence: TrackingSequence, parent=None):
        super().__init__(parent)
        self.sequence = sequence
        self.setWindowTitle("Edit Tracking Sequence")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Info
        info_label = QLabel(
            f"<b>Video:</b> {self.sequence.video_id}<br>"
            f"<b>Sequence ID:</b> {self.sequence.sequence_id}<br>"
            f"<b>Created:</b> {self.sequence.created_date[:10]}<br><br>"
            "Modify the frame range or notes for this sequence."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Frame range
        frame_group = QGroupBox("Frame Range")
        frame_layout = QFormLayout()
        
        self.start_spin = QSpinBox()
        self.start_spin.setRange(0, 999999)
        self.start_spin.setValue(self.sequence.start_frame)
        self.start_spin.valueChanged.connect(self.update_length)
        frame_layout.addRow("Start Frame:", self.start_spin)
        
        self.end_spin = QSpinBox()
        self.end_spin.setRange(0, 999999)
        self.end_spin.setValue(self.sequence.end_frame)
        self.end_spin.valueChanged.connect(self.update_length)
        frame_layout.addRow("End Frame:", self.end_spin)
        
        self.length_label = QLabel(f"{self.sequence.length} frames")
        frame_layout.addRow("Length:", self.length_label)
        
        frame_group.setLayout(frame_layout)
        layout.addWidget(frame_group)
        
        # Notes
        notes_group = QGroupBox("Notes (Optional)")
        notes_layout = QVBoxLayout()
        
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        self.notes_edit.setPlaceholderText("e.g., 'High activity period', 'Bees entering/exiting', etc.")
        self.notes_edit.setText(self.sequence.notes)
        notes_layout.addWidget(self.notes_edit)
        
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)
        
        # Info about annotation checking
        info_label2 = QLabel(
            "💡 The system will check if all frames in the updated range have annotations."
        )
        info_label2.setStyleSheet("color: #555; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        info_label2.setWordWrap(True)
        layout.addWidget(info_label2)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Changes")
        self.save_btn.clicked.connect(self.accept)
        self.save_btn.setDefault(True)
        button_layout.addWidget(self.save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.update_length()
    
    def update_length(self):
        """Update length label"""
        length = self.end_spin.value() - self.start_spin.value() + 1
        if length < 1:
            self.length_label.setText(f"<span style='color: red;'>Invalid (end must be >= start)</span>")
            self.save_btn.setEnabled(False)
        else:
            self.length_label.setText(f"{length} frames")
            self.save_btn.setEnabled(True)
    
    def get_config(self):
        """Get configuration from dialog"""
        return {
            'start_frame': self.start_spin.value(),
            'end_frame': self.end_spin.value(),
            'notes': self.notes_edit.toPlainText().strip()
        }


class TrackingSequencesPanel(QWidget):
    """Panel for managing tracking sequences"""
    
    # Signals
    sequence_created = pyqtSignal(str)  # sequence_id
    sequence_deleted = pyqtSignal(str)  # sequence_id
    sequence_selected = pyqtSignal(str, int)  # sequence_id, frame_idx (navigate to frame)
    validation_requested = pyqtSignal(list)  # List of sequence_ids to validate
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sequence_manager = None
        self.annotation_manager = None
        self.current_video_id = None
        self.current_frame = 0  # Initialize current frame
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_label = QLabel("<b>Tracking Sequences</b>")
        header_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(header_label)
        
        # Video info
        self.video_label = QLabel("No video loaded")
        self.video_label.setStyleSheet("color: gray; padding: 5px;")
        self.video_label.setWordWrap(True)
        layout.addWidget(self.video_label)
        
        # Sequence list
        self.sequence_list = QListWidget()
        self.sequence_list.itemDoubleClicked.connect(self.on_sequence_double_clicked)
        self.sequence_list.itemSelectionChanged.connect(self.update_button_states)
        layout.addWidget(self.sequence_list)
        
        # Buttons
        btn_layout = QVBoxLayout()
        
        self.new_btn = QPushButton("➕ New Sequence from Current Frame")
        self.new_btn.clicked.connect(self.create_sequence)
        self.new_btn.setEnabled(False)
        btn_layout.addWidget(self.new_btn)
        
        self.goto_btn = QPushButton("→ Go to Sequence")
        self.goto_btn.clicked.connect(self.goto_sequence)
        self.goto_btn.setEnabled(False)
        btn_layout.addWidget(self.goto_btn)
        
        self.edit_btn = QPushButton("✏️ Edit Sequence")
        self.edit_btn.clicked.connect(self.edit_sequence)
        self.edit_btn.setEnabled(False)
        btn_layout.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton("🗑 Delete Sequence")
        self.delete_btn.clicked.connect(self.delete_sequence)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)
        
        self.validate_btn = QPushButton("📊 Validate Sequences")
        self.validate_btn.clicked.connect(self.validate_sequences)
        self.validate_btn.setEnabled(False)
        self.validate_btn.setStyleSheet("font-weight: bold; background-color: #2196F3; color: white; padding: 8px;")
        btn_layout.addWidget(self.validate_btn)
        
        layout.addLayout(btn_layout)
        
        # Summary
        self.summary_label = QLabel("0 sequences total")
        self.summary_label.setStyleSheet("color: gray; font-size: 11px; padding: 5px;")
        layout.addWidget(self.summary_label)
    
    def set_managers(self, sequence_manager, annotation_manager):
        """Set manager references"""
        self.sequence_manager = sequence_manager
        self.annotation_manager = annotation_manager
    
    def set_video(self, video_id: str):
        """Set current video and refresh list"""
        self.current_video_id = video_id
        self.video_label.setText(f"Video: {video_id}")
        self.new_btn.setEnabled(True)
        self.refresh()
    
    def set_current_frame(self, frame_idx: int):
        """Update current frame (for creating sequences)"""
        self.current_frame = frame_idx
    
    def refresh(self):
        """Refresh sequence list"""
        self.sequence_list.clear()
        
        if not self.sequence_manager:
            return
        
        sequences = self.sequence_manager.get_sequences_for_video(self.current_video_id)
        
        for seq in sequences:
            # Check annotation status
            is_complete, missing_frames = self.sequence_manager.check_sequence_annotations(
                seq, self.annotation_manager
            )
            
            # Create item text
            status_icon = "✓" if is_complete else "⚠"
            status_text = "Complete" if is_complete else f"{len(missing_frames)} frames missing"
            enabled_text = "" if seq.enabled else " [DISABLED]"
            
            item_text = (
                f"{status_icon} Frames {seq.start_frame}-{seq.end_frame} "
                f"({seq.length} frames) - {status_text}{enabled_text}"
            )
            
            if seq.notes:
                item_text += f"\n   Note: {seq.notes[:50]}{'...' if len(seq.notes) > 50 else ''}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, seq.sequence_id)
            
            # Color code by status
            if not seq.enabled:
                item.setForeground(Qt.GlobalColor.gray)
            elif not is_complete:
                item.setForeground(Qt.GlobalColor.darkYellow)
            
            # Make checkable
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if seq.enabled else Qt.CheckState.Unchecked)
            
            self.sequence_list.addItem(item)
        
        # Update summary
        all_sequences = self.sequence_manager.get_all_sequences()
        enabled_count = len(self.sequence_manager.get_enabled_sequences())
        self.summary_label.setText(f"{len(sequences)} sequences for this video | {len(all_sequences)} total ({enabled_count} enabled)")
        
        self.update_button_states()
    
    def update_button_states(self):
        """Update button enabled states"""
        has_selection = len(self.sequence_list.selectedItems()) > 0
        has_checked = any(
            self.sequence_list.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.sequence_list.count())
        )
        
        # Enable validate button if any sequences exist in database (any video)
        all_sequences = self.sequence_manager.get_all_sequences()
        has_any_sequences = len(all_sequences) > 0
        
        self.goto_btn.setEnabled(has_selection)
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.validate_btn.setEnabled(has_any_sequences)
    
    def create_sequence(self):
        """Create a new tracking sequence"""
        if not self.current_video_id or not hasattr(self, 'current_frame'):
            QMessageBox.warning(self, "Error", "No video or frame loaded")
            return
        
        dialog = CreateSequenceDialog(self.current_video_id, self.current_frame, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            
            # Check if frames have annotations
            sequence = self.sequence_manager.add_sequence(
                self.current_video_id,
                config['start_frame'],
                config['end_frame'],
                config['notes']
            )
            
            is_complete, missing_frames = self.sequence_manager.check_sequence_annotations(
                sequence, self.annotation_manager
            )
            
            if not is_complete:
                QMessageBox.warning(
                    self,
                    "Missing Annotations",
                    f"Sequence created but {len(missing_frames)} frames are missing annotations:\n"
                    f"{missing_frames[:10]}{'...' if len(missing_frames) > 10 else ''}\n\n"
                    "Please annotate these frames before running validation."
                )
            else:
                QMessageBox.information(
                    self,
                    "Sequence Created",
                    f"Tracking sequence created successfully!\n"
                    f"Frames: {config['start_frame']}-{config['end_frame']} ({sequence.length} frames)"
                )
            
            self.refresh()
            self.sequence_created.emit(sequence.sequence_id)
    
    def edit_sequence(self):
        """Edit selected tracking sequence"""
        selected_items = self.sequence_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        sequence_id = item.data(Qt.ItemDataRole.UserRole)
        sequence = self.sequence_manager.get_sequence(sequence_id)
        
        if not sequence:
            QMessageBox.warning(self, "Error", "Sequence not found")
            return
        
        dialog = EditSequenceDialog(sequence, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            
            # Update sequence with new frame range and notes
            success = self.sequence_manager.update_sequence(
                sequence_id,
                start_frame=config['start_frame'],
                end_frame=config['end_frame'],
                notes=config['notes']
            )
            
            if not success:
                QMessageBox.warning(self, "Error", "Failed to update sequence")
                return
            
            # Re-fetch the sequence (it may have a new ID if frames changed)
            updated_sequence = self.sequence_manager.get_sequence_by_frames(
                sequence.video_id,
                config['start_frame'],
                config['end_frame']
            )
            
            if not updated_sequence:
                # Fallback to original sequence_id
                updated_sequence = self.sequence_manager.get_sequence(sequence_id)
            
            # Check if frames have annotations
            is_complete, missing_frames = self.sequence_manager.check_sequence_annotations(
                updated_sequence, self.annotation_manager
            )
            
            if not is_complete:
                QMessageBox.warning(
                    self,
                    "Missing Annotations",
                    f"Sequence updated but {len(missing_frames)} frames are missing annotations:\n"
                    f"{missing_frames[:10]}{'...' if len(missing_frames) > 10 else ''}\n\n"
                    "Please annotate these frames before running validation."
                )
            else:
                QMessageBox.information(
                    self,
                    "Sequence Updated",
                    f"Tracking sequence updated successfully!\n"
                    f"Frames: {config['start_frame']}-{config['end_frame']} ({updated_sequence.length} frames)"
                )
            
            self.refresh()
    
    def delete_sequence(self):
        """Delete selected sequence"""
        selected_items = self.sequence_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        sequence_id = item.data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            "Delete this tracking sequence?\n\n"
            "(Ground truth annotations will NOT be deleted)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.sequence_manager.remove_sequence(sequence_id):
                self.refresh()
                self.sequence_deleted.emit(sequence_id)
    
    def goto_sequence(self):
        """Navigate to selected sequence"""
        selected_items = self.sequence_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        sequence_id = item.data(Qt.ItemDataRole.UserRole)
        sequence = self.sequence_manager.get_sequence(sequence_id)
        
        if sequence:
            # Navigate to first frame of sequence
            self.sequence_selected.emit(sequence_id, sequence.start_frame)
    
    def validate_sequences(self):
        """Request validation of checked sequences"""
        # Check for checked sequences in current video list
        checked_sequences = []
        for i in range(self.sequence_list.count()):
            item = self.sequence_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                sequence_id = item.data(Qt.ItemDataRole.UserRole)
                checked_sequences.append(sequence_id)
        
        # If none checked in current video, check all sequences across all videos
        if not checked_sequences:
            all_sequences = self.sequence_manager.get_all_sequences()
            checked_sequences = [
                seq.sequence_id for seq in all_sequences 
                if getattr(seq, 'enabled', True)
            ]
        
        if checked_sequences:
            self.validation_requested.emit(checked_sequences)
        else:
            QMessageBox.information(
                self,
                "No Sequences Available",
                "Please create at least one sequence to validate.\n\n"
                "Use the '+' button to create a tracking sequence."
            )
    
    def on_sequence_double_clicked(self, item):
        """Handle double click on sequence"""
        self.goto_sequence()
