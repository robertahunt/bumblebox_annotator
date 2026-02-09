"""
Training progress dialog with live metrics and visualization
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QProgressBar, QTableWidget,
                             QTableWidgetItem, QGroupBox, QTextEdit, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class MetricsPlot(FigureCanvasQTAgg):
    """Matplotlib widget for plotting training metrics"""
    
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        
        self.iterations = []
        self.losses = []
        
        self.axes.set_xlabel('Iteration')
        self.axes.set_ylabel('Loss')
        self.axes.set_title('Training Loss')
        self.axes.grid(True, alpha=0.3)
        
    def add_data_point(self, iteration, loss):
        """Add a data point and update plot"""
        self.iterations.append(iteration)
        self.losses.append(loss)
        
        self.axes.clear()
        self.axes.plot(self.iterations, self.losses, 'b-', linewidth=2)
        self.axes.set_xlabel('Iteration')
        self.axes.set_ylabel('Loss')
        self.axes.set_title('Training Loss')
        self.axes.grid(True, alpha=0.3)
        
        self.draw()


class TrainingProgressDialog(QDialog):
    """Non-modal dialog showing training progress"""
    
    stop_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    show_validation_requested = pyqtSignal()
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.is_paused = False
        self.validation_results = []
        self.best_ap = 0.0
        self.best_iter = 0
        
        self.setWindowTitle("Training Progress")
        self.setMinimumWidth(800)
        self.setMinimumHeight(700)
        
        # Make non-modal so user can interact with main window
        self.setModal(False)
        
        self.init_ui()
        
        # Add initial message to log
        self.log_text.append("=" * 60)
        self.log_text.append("TRAINING PROGRESS LOG")
        self.log_text.append("=" * 60)
        self.log_text.append("Initializing...")
        
    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        
        # Progress bar
        progress_group = QGroupBox("Overall Progress")
        progress_layout = QVBoxLayout()
        
        self.iter_label = QLabel("Iteration: 0/1000")
        progress_layout.addWidget(self.iter_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Status: Initializing...")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Split view for metrics and validation
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Training metrics
        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QVBoxLayout()
        
        # Current metrics display
        metrics_text_layout = QHBoxLayout()
        self.loss_label = QLabel("Loss: --")
        self.lr_label = QLabel("LR: --")
        metrics_text_layout.addWidget(self.loss_label)
        metrics_text_layout.addWidget(self.lr_label)
        metrics_text_layout.addStretch()
        metrics_layout.addLayout(metrics_text_layout)
        
        # Loss plot
        self.metrics_plot = MetricsPlot(self, width=7, height=3)
        metrics_layout.addWidget(self.metrics_plot)
        
        metrics_group.setLayout(metrics_layout)
        splitter.addWidget(metrics_group)
        
        # Validation results
        validation_group = QGroupBox("Validation Results")
        validation_layout = QVBoxLayout()
        
        self.validation_table = QTableWidget()
        self.validation_table.setColumnCount(5)
        self.validation_table.setHorizontalHeaderLabels([
            'Iteration', 'segm AP', 'AP50', 'AP75', 'bbox AP'
        ])
        self.validation_table.setMaximumHeight(200)
        validation_layout.addWidget(self.validation_table)
        
        self.best_model_label = QLabel("Best: -- (iter --)")
        validation_layout.addWidget(self.best_model_label)
        
        validation_group.setLayout(validation_layout)
        splitter.addWidget(validation_group)
        
        layout.addWidget(splitter)
        
        # Log output
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.view_val_button = QPushButton("View Validation Predictions")
        self.view_val_button.clicked.connect(self.show_validation_requested.emit)
        self.view_val_button.setEnabled(False)
        button_layout.addWidget(self.view_val_button)
        
        button_layout.addStretch()
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def update_iteration(self, iteration, max_iter, metrics):
        """Update progress from iteration"""
        progress = int((iteration / max_iter) * 100)
        self.progress_bar.setValue(progress)
        self.iter_label.setText(f"Iteration: {iteration}/{max_iter}")
        
        # Update metrics display
        if 'total_loss' in metrics:
            loss = metrics['total_loss']
            self.loss_label.setText(f"Loss: {loss:.4f}")
            
            # Add to plot
            self.metrics_plot.add_data_point(iteration, loss)
        
        if 'lr' in metrics:
            self.lr_label.setText(f"LR: {metrics['lr']:.6f}")
        
        # Log detailed metrics
        log_msg = f"Iter {iteration}: "
        log_msg += f"loss={metrics.get('total_loss', 0):.4f}"
        
        if 'loss_cls' in metrics:
            log_msg += f", cls={metrics['loss_cls']:.4f}"
        if 'loss_box_reg' in metrics:
            log_msg += f", box={metrics['loss_box_reg']:.4f}"
        if 'loss_mask' in metrics:
            log_msg += f", mask={metrics['loss_mask']:.4f}"
        
        self.append_log(log_msg)
    
    def update_validation_started(self, iteration):
        """Update when validation starts"""
        self.status_label.setText(f"Status: Running validation at iter {iteration}...")
        self.append_log(f"\n▶ Validation started at iteration {iteration}")
    
    def update_validation_completed(self, iteration, results):
        """Update when validation completes"""
        self.status_label.setText("Status: Training...")
        
        # Add to table
        row = self.validation_table.rowCount()
        self.validation_table.insertRow(row)
        
        self.validation_table.setItem(row, 0, QTableWidgetItem(str(iteration)))
        self.validation_table.setItem(row, 1, QTableWidgetItem(f"{results.get('segm_AP', 0):.3f}"))
        self.validation_table.setItem(row, 2, QTableWidgetItem(f"{results.get('segm_AP50', 0):.3f}"))
        self.validation_table.setItem(row, 3, QTableWidgetItem(f"{results.get('segm_AP75', 0):.3f}"))
        self.validation_table.setItem(row, 4, QTableWidgetItem(f"{results.get('bbox_AP', 0):.3f}"))
        
        # Scroll to bottom
        self.validation_table.scrollToBottom()
        
        # Log results
        self.append_log(
            f"✓ Validation: segm AP={results.get('segm_AP', 0):.3f}, "
            f"AP50={results.get('segm_AP50', 0):.3f}, "
            f"AP75={results.get('segm_AP75', 0):.3f}"
        )
        
        # Store for visualization
        self.validation_results.append((iteration, results))
        
        # Enable view button
        self.view_val_button.setEnabled(True)
    
    def update_best_model(self, iteration, ap, checkpoint_path):
        """Update when best model is found"""
        self.best_ap = ap
        self.best_iter = iteration
        self.best_model_label.setText(f"Best: {ap:.3f} (iter {iteration})")
        self.append_log(f"⭐ New best model! AP={ap:.3f} at iteration {iteration}")
    
    def append_log(self, message):
        """Append message to log"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(f"Status: {message}")
        self.append_log(message)
    
    def toggle_pause(self):
        """Toggle pause/resume"""
        if self.is_paused:
            self.pause_button.setText("Pause")
            self.is_paused = False
            self.resume_requested.emit()
            self.update_status("Training resumed")
        else:
            self.pause_button.setText("Resume")
            self.is_paused = True
            self.pause_requested.emit()
            self.update_status("Training paused")
    
    def training_finished(self, success, final_metrics):
        """Handle training completion"""
        self.pause_button.setEnabled(False)
        self.stop_button.setText("Close")
        
        if success:
            self.status_label.setText("Status: Training completed successfully!")
            self.append_log(
                f"\n✓ Training finished!\n"
                f"Total iterations: {final_metrics.get('iterations', 0)}\n"
                f"Best model: AP={final_metrics.get('best_ap', 0):.3f} "
                f"at iteration {final_metrics.get('best_iter', 0)}"
            )
        else:
            self.status_label.setText("Status: Training stopped")
            self.append_log("\n✗ Training stopped or failed")
