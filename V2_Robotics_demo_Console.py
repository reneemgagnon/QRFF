"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   QRFF ROBOTICS TRAINING CONSOLE - PyQt6 Edition                              ║
║   ═══════════════════════════════════════════════════════════════════════     ║
║   Neural Robotics Training Interface with NVIDIA PhysicalAI Dataset           ║
║                                                                               ║
║   INNOVATIVE FEATURES:                                                        ║
║   ├── 🔍 Smart HDF5 Dataset Locator with HuggingFace Integration             ║
║   ├── 📊 Real-time Training Metrics Dashboard                                ║
║   ├── 🌡️ QRFF Field Thermodynamics Visualization                            ║
║   ├── 🤖 Live Pose Prediction Preview                                        ║
║   ├── 📈 Interactive Loss/Accuracy Curves                                    ║
║   └── 🎛️ Dynamic Hyperparameter Control                                      ║
║                                                                               ║
║   Dataset: nvidia/PhysicalAI-Robotics-Manipulation-Augmented                  ║
║   Architecture: Quaternion Ruffle Field (QRFF) for SE(3) Learning            ║
║                                                                               ║
║   Author: Built on QRFF by Renee M Gagnon                                     ║
║   License: Research & Development Use                                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from queue import Queue
import traceback

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def check_dependencies() -> Dict[str, bool]:
    """
    Check availability of all required dependencies.
    
    Returns:
        Dictionary mapping package names to availability status.
    """
    deps = {}
    
    # PyQt6
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        deps['PyQt6'] = True
    except ImportError:
        deps['PyQt6'] = False
    
    # PyTorch
    try:
        import torch
        deps['torch'] = True
        deps['cuda'] = torch.cuda.is_available()
    except ImportError:
        deps['torch'] = False
        deps['cuda'] = False
    
    # NumPy
    try:
        import numpy as np
        deps['numpy'] = True
    except ImportError:
        deps['numpy'] = False
    
    # h5py for HDF5
    try:
        import h5py
        deps['h5py'] = True
    except ImportError:
        deps['h5py'] = False
    
    # HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download
        deps['huggingface_hub'] = True
    except ImportError:
        deps['huggingface_hub'] = False
    
    # Matplotlib for plotting
    try:
        import matplotlib
        deps['matplotlib'] = True
    except ImportError:
        deps['matplotlib'] = False
    
    return deps


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

DEPS = check_dependencies()

if not DEPS.get('PyQt6'):
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  ERROR: PyQt6 is required but not installed.                      ║")
    print("║  Install with: pip install PyQt6                                  ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    sys.exit(1)

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QLineEdit, QTextEdit, QProgressBar,
    QTabWidget, QFrame, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QFileDialog, QMessageBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QSlider, QCheckBox, QScrollArea, QSizePolicy, QStatusBar
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation, 
    QEasingCurve, QObject
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QBrush, QLinearGradient, QPainter, 
    QFontDatabase, QIcon, QPixmap, QPen
)

# Scientific imports
if DEPS.get('numpy'):
    import numpy as np

if DEPS.get('torch'):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split

if DEPS.get('h5py'):
    import h5py

if DEPS.get('matplotlib'):
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsoleConfig:
    """
    Configuration for the Robotics Training Console.
    
    Attributes:
        dataset_name: HuggingFace dataset identifier
        hdf5_files: Available HDF5 file options
        cache_dir: Local cache directory for datasets
        default_epochs: Default training epochs
        default_batch_size: Default batch size
        default_learning_rate: Default learning rate
        default_hidden_dim: Default model hidden dimension
        default_n_neurons: Default QRFF neuron count
        theme: UI theme ('dark' or 'light')
    """
    dataset_name: str = "nvidia/PhysicalAI-Robotics-Manipulation-Augmented"
    hdf5_files: List[str] = field(default_factory=lambda: [
        "mimic_dataset_1k.hdf5",
        "cosmos_dataset_1k.hdf5"
    ])
    cache_dir: str = "./nvidia_manipulation_cache"
    default_epochs: int = 50
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    default_hidden_dim: int = 128
    default_n_neurons: int = 64
    theme: str = "dark"


# ═══════════════════════════════════════════════════════════════════════════════
# STYLESHEET
# ═══════════════════════════════════════════════════════════════════════════════

DARK_STYLESHEET = """
/* ═══════════════════════════════════════════════════════════════════════════
   QRFF ROBOTICS CONSOLE - DARK THEME
   Cyberpunk-inspired neural network training interface
   ═══════════════════════════════════════════════════════════════════════════ */

QMainWindow {
    background-color: #0a0e14;
}

QWidget {
    background-color: #0a0e14;
    color: #b3b1ad;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
}

/* ═══════════════ GROUP BOXES ═══════════════ */
QGroupBox {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 12px;
    padding: 15px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: #58a6ff;
    background-color: #0d1117;
}

/* ═══════════════ BUTTONS ═══════════════ */
QPushButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    color: #c9d1d9;
    font-weight: bold;
    min-height: 30px;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #8b949e;
}

QPushButton:pressed {
    background-color: #161b22;
}

QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}

QPushButton#primaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #238636, stop:1 #2ea043);
    border: none;
    color: white;
}

QPushButton#primaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #2ea043, stop:1 #3fb950);
}

QPushButton#dangerButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #da3633, stop:1 #f85149);
    border: none;
    color: white;
}

QPushButton#accentButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:1 #388bfd);
    border: none;
    color: white;
}

/* ═══════════════ INPUTS ═══════════════ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px;
    color: #c9d1d9;
    selection-background-color: #1f6feb;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #58a6ff;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid #8b949e;
    margin-right: 10px;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    selection-background-color: #1f6feb;
}

/* ═══════════════ TEXT EDIT / CONSOLE ═══════════════ */
QTextEdit {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px;
    color: #7ee787;
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 11px;
}

/* ═══════════════ PROGRESS BAR ═══════════════ */
QProgressBar {
    background-color: #21262d;
    border: none;
    border-radius: 4px;
    height: 12px;
    text-align: center;
    color: white;
    font-weight: bold;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:0.5 #58a6ff, stop:1 #79c0ff);
    border-radius: 4px;
}

/* ═══════════════ TABS ═══════════════ */
QTabWidget::pane {
    border: 1px solid #21262d;
    border-radius: 6px;
    background-color: #0d1117;
}

QTabBar::tab {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 20px;
    margin-right: 2px;
    color: #8b949e;
}

QTabBar::tab:selected {
    background-color: #0d1117;
    color: #58a6ff;
    border-color: #30363d;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #21262d;
    color: #c9d1d9;
}

/* ═══════════════ TABLES ═══════════════ */
QTableWidget {
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    gridline-color: #21262d;
}

QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #21262d;
}

QTableWidget::item:selected {
    background-color: #1f6feb;
}

QHeaderView::section {
    background-color: #161b22;
    color: #58a6ff;
    padding: 8px;
    border: none;
    border-bottom: 2px solid #30363d;
    font-weight: bold;
}

/* ═══════════════ SCROLLBARS ═══════════════ */
QScrollBar:vertical {
    background-color: #0d1117;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #0d1117;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    border-radius: 6px;
    min-width: 30px;
}

/* ═══════════════ SLIDERS ═══════════════ */
QSlider::groove:horizontal {
    background-color: #21262d;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #58a6ff;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #79c0ff;
}

/* ═══════════════ LABELS ═══════════════ */
QLabel#titleLabel {
    color: #58a6ff;
    font-size: 24px;
    font-weight: bold;
}

QLabel#subtitleLabel {
    color: #8b949e;
    font-size: 12px;
}

QLabel#metricValue {
    color: #7ee787;
    font-size: 28px;
    font-weight: bold;
}

QLabel#metricLabel {
    color: #8b949e;
    font-size: 11px;
}

QLabel#warningLabel {
    color: #d29922;
}

QLabel#errorLabel {
    color: #f85149;
}

QLabel#successLabel {
    color: #3fb950;
}

/* ═══════════════ FRAMES ═══════════════ */
QFrame#metricCard {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 15px;
}

QFrame#separator {
    background-color: #21262d;
    max-height: 1px;
}

/* ═══════════════ STATUS BAR ═══════════════ */
QStatusBar {
    background-color: #161b22;
    border-top: 1px solid #21262d;
    color: #8b949e;
}

QStatusBar::item {
    border: none;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetLocator:
    """
    Smart HDF5 Dataset Locator with HuggingFace Integration.
    
    Scans local directories for HDF5 files and can download
    from NVIDIA's PhysicalAI dataset on HuggingFace.
    """
    
    def __init__(self, config: ConsoleConfig):
        """
        Initialize the dataset locator.
        
        Args:
            config: Console configuration object
        """
        self.config = config
        self.found_datasets: List[Dict[str, Any]] = []
    
    def scan_local_directories(self, 
                               search_paths: Optional[List[str]] = None
                               ) -> List[Dict[str, Any]]:
        """
        Scan local directories for HDF5 datasets.
        
        Args:
            search_paths: List of directories to search (None = defaults)
            
        Returns:
            List of found dataset info dictionaries
        """
        if search_paths is None:
            search_paths = [
                ".",
                "./data",
                "./datasets",
                self.config.cache_dir,
                str(Path.home() / "datasets"),
                str(Path.home() / ".cache" / "huggingface"),
            ]
        
        self.found_datasets = []
        
        for search_path in search_paths:
            path = Path(search_path)
            if not path.exists():
                continue
            
            # Search for HDF5 files
            for hdf5_file in path.rglob("*.hdf5"):
                dataset_info = self._analyze_hdf5(hdf5_file)
                if dataset_info:
                    self.found_datasets.append(dataset_info)
            
            # Also check .h5 extension
            for h5_file in path.rglob("*.h5"):
                dataset_info = self._analyze_hdf5(h5_file)
                if dataset_info:
                    self.found_datasets.append(dataset_info)
        
        return self.found_datasets
    
    def _analyze_hdf5(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Analyze an HDF5 file to extract metadata.
        
        Args:
            path: Path to HDF5 file
            
        Returns:
            Dataset info dictionary or None if invalid
        """
        if not DEPS.get('h5py'):
            return None
        
        try:
            with h5py.File(path, 'r') as f:
                # Check if it looks like a robotics dataset
                has_data = 'data' in f
                
                # Count demonstrations if applicable
                n_demos = 0
                if has_data:
                    n_demos = len([k for k in f['data'].keys() 
                                  if k.startswith('demo_')])
                
                # Get file size
                file_size = path.stat().st_size / (1024 * 1024)  # MB
                
                return {
                    'path': str(path),
                    'name': path.name,
                    'n_demos': n_demos,
                    'size_mb': round(file_size, 2),
                    'is_nvidia': 'mimic' in path.name.lower() or 
                                 'cosmos' in path.name.lower(),
                    'valid': has_data and n_demos > 0
                }
        except Exception as e:
            return None
    
    def download_from_huggingface(self, 
                                  filename: str,
                                  progress_callback: Optional[callable] = None
                                  ) -> Optional[Path]:
        """
        Download dataset from HuggingFace.
        
        Args:
            filename: HDF5 filename to download
            progress_callback: Optional callback for progress updates
            
        Returns:
            Path to downloaded file or None on failure
        """
        if not DEPS.get('huggingface_hub'):
            return None
        
        from huggingface_hub import hf_hub_download
        
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=self.config.dataset_name,
                filename=filename,
                repo_type="dataset",
                cache_dir=str(cache_dir),
                local_dir=str(cache_dir),
            )
            return Path(downloaded_path)
        except Exception as e:
            print(f"Download failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingWorker(QThread):
    """
    Background worker thread for model training.
    
    Signals:
        progress: Emitted with (epoch, metrics_dict) during training
        log_message: Emitted with log messages for console
        finished: Emitted when training completes
        error: Emitted with error message on failure
    """
    
    progress = pyqtSignal(int, dict)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, 
                 dataset_path: str,
                 config: Dict[str, Any],
                 parent=None):
        """
        Initialize training worker.
        
        Args:
            dataset_path: Path to HDF5 dataset
            config: Training configuration dictionary
            parent: Parent QObject
        """
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.config = config
        self._stop_requested = False
    
    def request_stop(self):
        """Request graceful stop of training."""
        self._stop_requested = True
    
    def run(self):
        """Execute training in background thread."""
        try:
            self._run_training()
        except Exception as e:
            self.error.emit(f"Training failed: {str(e)}\n{traceback.format_exc()}")
    
    def _run_training(self):
        """Internal training loop."""
        self.log_message.emit("═" * 60)
        self.log_message.emit("  QRFF ROBOTICS TRAINING INITIATED")
        self.log_message.emit("═" * 60)
        
        # Get device
        if DEPS.get('torch'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.log_message.emit(f"  Device: {device}")
        else:
            self.error.emit("PyTorch not available")
            return
        
        # Load data
        self.log_message.emit("\n[1/4] Loading dataset...")
        demonstrations = self._load_demonstrations()
        if demonstrations is None:
            return
        
        self.log_message.emit(f"  Loaded {len(demonstrations)} demonstrations")
        
        # Create dataset
        self.log_message.emit("\n[2/4] Preparing training data...")
        train_loader, val_loader, input_dim = self._create_dataloaders(
            demonstrations, device
        )
        
        if train_loader is None:
            return
        
        self.log_message.emit(f"  Training batches: {len(train_loader)}")
        self.log_message.emit(f"  Validation batches: {len(val_loader)}")
        self.log_message.emit(f"  Input dimension: {input_dim}")
        
        # Create model
        self.log_message.emit("\n[3/4] Initializing QRFF model...")
        model = self._create_model(input_dim, device)
        if model is None:
            return
        
        n_params = sum(p.numel() for p in model.parameters())
        self.log_message.emit(f"  Model parameters: {n_params:,}")
        
        # Train
        self.log_message.emit("\n[4/4] Starting training...")
        self.log_message.emit("-" * 50)
        
        results = self._train_loop(model, train_loader, val_loader, device)
        
        self.log_message.emit("\n" + "═" * 60)
        self.log_message.emit("  TRAINING COMPLETE")
        self.log_message.emit("═" * 60)
        
        self.finished.emit(results)
    
    def _load_demonstrations(self) -> Optional[List[Dict[str, np.ndarray]]]:
        """Load demonstrations from HDF5 file."""
        if not DEPS.get('h5py'):
            self.error.emit("h5py not available")
            return None
        
        demonstrations = []
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                data_group = f['data']
                demo_keys = sorted([k for k in data_group.keys() 
                                   if k.startswith('demo_')])
                
                max_demos = self.config.get('max_demos', 100)
                demo_keys = demo_keys[:max_demos]
                
                for demo_key in demo_keys:
                    demo = data_group[demo_key]
                    demo_data = {}
                    
                    # Actions
                    if 'actions' in demo:
                        demo_data['actions'] = demo['actions'][:]
                    
                    # Observations
                    obs = demo['obs']
                    
                    # Extract various observation types
                    for key in ['robot0_joint_pos', 'joint_pos']:
                        if key in obs:
                            demo_data['joint_pos'] = obs[key][:]
                            break
                    
                    for key in ['robot0_eef_pos', 'eef_pos']:
                        if key in obs:
                            demo_data['eef_pos'] = obs[key][:]
                            break
                    
                    for key in ['robot0_eef_quat', 'eef_quat']:
                        if key in obs:
                            demo_data['eef_quat'] = obs[key][:]
                            break
                    
                    for key in ['robot0_gripper_qpos', 'gripper_qpos']:
                        if key in obs:
                            demo_data['gripper'] = obs[key][:]
                            break
                    
                    for key in ['object', 'object_pose']:
                        if key in obs:
                            demo_data['object_poses'] = obs[key][:]
                            break
                    
                    demonstrations.append(demo_data)
                
        except Exception as e:
            self.log_message.emit(f"  Using synthetic data: {str(e)}")
            demonstrations = self._create_synthetic_data()
        
        return demonstrations
    
    def _create_synthetic_data(self, n_demos: int = 100) -> List[Dict]:
        """Create synthetic manipulation data for testing."""
        demonstrations = []
        
        for _ in range(n_demos):
            T = 100  # timesteps
            demonstrations.append({
                'joint_pos': np.random.randn(T, 7).astype(np.float32) * 0.1,
                'eef_pos': np.random.randn(T, 3).astype(np.float32) * 0.1 + 
                          np.array([0.5, 0, 0.3]),
                'eef_quat': np.tile([1, 0, 0, 0], (T, 1)).astype(np.float32),
                'gripper': np.random.rand(T, 2).astype(np.float32) * 0.04,
                'object_poses': np.random.randn(T, 21).astype(np.float32) * 0.1,
                'actions': np.random.randn(T, 7).astype(np.float32) * 0.01,
            })
        
        return demonstrations
    
    def _create_dataloaders(self, 
                           demonstrations: List[Dict],
                           device: torch.device
                           ) -> Tuple[DataLoader, DataLoader, int]:
        """Create PyTorch dataloaders from demonstrations."""
        
        # Build samples
        samples = []
        seq_len = self.config.get('sequence_length', 10)
        
        for demo in demonstrations:
            if 'eef_pos' not in demo or 'eef_quat' not in demo:
                continue
            
            T = len(demo['eef_pos'])
            
            for t in range(T - 1):
                # Build state vector
                state_parts = [demo['eef_pos'][t], demo['eef_quat'][t]]
                
                if 'joint_pos' in demo:
                    state_parts.append(demo['joint_pos'][t])
                if 'gripper' in demo:
                    state_parts.append(demo['gripper'][t])
                if 'object_poses' in demo:
                    state_parts.append(demo['object_poses'][t])
                
                state = np.concatenate(state_parts)
                
                # Target: next EE pose
                target = np.concatenate([
                    demo['eef_pos'][t + 1],
                    demo['eef_quat'][t + 1]
                ])
                
                samples.append((state, target))
        
        if not samples:
            self.error.emit("No valid samples created")
            return None, None, None
        
        # Convert to tensors
        states = torch.tensor(
            np.array([s[0] for s in samples]), 
            dtype=torch.float32
        )
        targets = torch.tensor(
            np.array([s[1] for s in samples]), 
            dtype=torch.float32
        )
        
        input_dim = states.shape[1]
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(states, targets)
        
        # Split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        batch_size = self.config.get('batch_size', 32)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size
        )
        
        return train_loader, val_loader, input_dim
    
    def _create_model(self, input_dim: int, device: torch.device) -> nn.Module:
        """Create QRFF manipulation model."""
        hidden_dim = self.config.get('hidden_dim', 128)
        n_neurons = self.config.get('n_neurons', 64)
        
        class QRFFManipulationModel(nn.Module):
            """Simplified QRFF model for pose prediction."""
            
            def __init__(self, input_dim, hidden_dim, n_neurons):
                super().__init__()
                
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
                
                # Quaternion field simulation
                self.field_neurons = nn.Parameter(
                    torch.randn(n_neurons, 4) * 0.1
                )
                self.field_positions = nn.Parameter(
                    torch.randn(n_neurons, 4) * 0.5
                )
                
                self.field_attention = nn.MultiheadAttention(
                    hidden_dim, num_heads=4, batch_first=True
                )
                
                # Output heads
                self.position_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 3)
                )
                
                self.quaternion_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 4)
                )
                
                # Field state tracking
                self.register_buffer('temperature', torch.tensor(1.0))
                self.register_buffer('coherence', torch.tensor(0.5))
                self.register_buffer('energy', torch.tensor(0.0))
            
            def forward(self, x):
                # Encode
                h = self.encoder(x)
                
                # Field dynamics (simplified)
                h = h.unsqueeze(1)
                h, _ = self.field_attention(h, h, h)
                h = h.squeeze(1)
                
                # Output
                pos = self.position_head(h)
                quat = self.quaternion_head(h)
                quat = F.normalize(quat, dim=-1)
                
                return torch.cat([pos, quat], dim=-1)
            
            def update_field_dynamics(self, loss_value):
                """Update field thermodynamic state."""
                with torch.no_grad():
                    self.energy = 0.95 * self.energy + 0.05 * loss_value
                    self.temperature = 0.1 + 0.9 * torch.sigmoid(self.energy)
                    self.coherence = torch.clamp(
                        self.coherence + 0.01 * (1 - loss_value), 0.1, 0.99
                    )
            
            def get_field_state(self):
                """Get current field state."""
                return {
                    'temperature': self.temperature.item(),
                    'coherence': self.coherence.item(),
                    'energy': self.energy.item()
                }
        
        model = QRFFManipulationModel(input_dim, hidden_dim, n_neurons)
        return model.to(device)
    
    def _train_loop(self, 
                   model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   device: torch.device) -> Dict:
        """Execute training loop."""
        
        epochs = self.config.get('epochs', 50)
        lr = self.config.get('learning_rate', 1e-3)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'position_error_cm': [],
            'rotation_error_deg': [],
            'temperature': [],
            'coherence': [],
            'energy': []
        }
        
        for epoch in range(epochs):
            if self._stop_requested:
                self.log_message.emit("\n⚠️ Training stopped by user")
                break
            
            # Training
            model.train()
            train_loss = 0
            
            for batch_states, batch_targets in train_loader:
                batch_states = batch_states.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_states)
                
                # SE(3) loss
                pos_loss = F.mse_loss(predictions[:, :3], batch_targets[:, :3])
                
                # Geodesic rotation loss
                pred_quat = predictions[:, 3:]
                target_quat = batch_targets[:, 3:]
                dot = torch.abs((pred_quat * target_quat).sum(dim=-1))
                dot = torch.clamp(dot, -1.0, 1.0)
                rot_loss = 2 * torch.acos(dot).mean()
                
                loss = pos_loss + 0.5 * rot_loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                model.update_field_dynamics(loss.item())
            
            scheduler.step()
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0
            pos_errors = []
            rot_errors = []
            
            with torch.no_grad():
                for batch_states, batch_targets in val_loader:
                    batch_states = batch_states.to(device)
                    batch_targets = batch_targets.to(device)
                    
                    predictions = model(batch_states)
                    
                    # Position error in cm
                    pos_err = (predictions[:, :3] - batch_targets[:, :3]).norm(dim=-1)
                    pos_errors.extend((pos_err * 100).cpu().tolist())
                    
                    # Rotation error in degrees
                    pred_quat = predictions[:, 3:]
                    target_quat = batch_targets[:, 3:]
                    dot = torch.abs((pred_quat * target_quat).sum(dim=-1))
                    dot = torch.clamp(dot, -1.0, 1.0)
                    rot_err = 2 * torch.acos(dot) * 180 / 3.14159
                    rot_errors.extend(rot_err.cpu().tolist())
                    
                    loss = F.mse_loss(predictions[:, :3], batch_targets[:, :3])
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            avg_pos_error = np.mean(pos_errors)
            avg_rot_error = np.mean(rot_errors)
            
            # Get field state
            field_state = model.get_field_state()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['position_error_cm'].append(avg_pos_error)
            history['rotation_error_deg'].append(avg_rot_error)
            history['temperature'].append(field_state['temperature'])
            history['coherence'].append(field_state['coherence'])
            history['energy'].append(field_state['energy'])
            
            # Log progress
            self.log_message.emit(
                f"  Epoch {epoch+1:3d}/{epochs} │ "
                f"Loss: {train_loss:.4f} │ "
                f"Pos: {avg_pos_error:.2f}cm │ "
                f"Rot: {avg_rot_error:.1f}° │ "
                f"T: {field_state['temperature']:.3f}"
            )
            
            # Emit progress
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'position_error_cm': avg_pos_error,
                'rotation_error_deg': avg_rot_error,
                'temperature': field_state['temperature'],
                'coherence': field_state['coherence'],
                'energy': field_state['energy'],
                'learning_rate': scheduler.get_last_lr()[0]
            }
            self.progress.emit(epoch + 1, metrics)
        
        return {
            'history': history,
            'final_position_error_cm': history['position_error_cm'][-1],
            'final_rotation_error_deg': history['rotation_error_deg'][-1],
            'epochs_completed': len(history['train_loss'])
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsPlotWidget(FigureCanvas if DEPS.get('matplotlib') else QWidget):
    """
    Real-time training metrics visualization widget.
    
    Displays:
        - Loss curves (training and validation)
        - Position/rotation errors
        - QRFF field thermodynamics
    """
    
    def __init__(self, parent=None):
        """Initialize the metrics plot widget."""
        if DEPS.get('matplotlib'):
            self.fig = Figure(figsize=(12, 6), dpi=100, facecolor='#0d1117')
            super().__init__(self.fig)
            self.setParent(parent)
            self._setup_axes()
        else:
            super().__init__(parent)
            self.setMinimumHeight(300)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'position_error': [],
            'rotation_error': [],
            'temperature': [],
            'coherence': []
        }
    
    def _setup_axes(self):
        """Set up the plot axes."""
        self.fig.clear()
        
        # Create 2x2 subplot grid
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_errors = self.fig.add_subplot(gs[0, 1])
        self.ax_field = self.fig.add_subplot(gs[1, 0])
        self.ax_thermo = self.fig.add_subplot(gs[1, 1])
        
        # Style all axes
        for ax in [self.ax_loss, self.ax_errors, self.ax_field, self.ax_thermo]:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            ax.spines['bottom'].set_color('#30363d')
            ax.spines['top'].set_color('#30363d')
            ax.spines['left'].set_color('#30363d')
            ax.spines['right'].set_color('#30363d')
            ax.xaxis.label.set_color('#8b949e')
            ax.yaxis.label.set_color('#8b949e')
            ax.title.set_color('#c9d1d9')
        
        self.ax_loss.set_title('Loss Curves')
        self.ax_errors.set_title('Prediction Errors')
        self.ax_field.set_title('Field Coherence')
        self.ax_thermo.set_title('Field Temperature')
    
    def update_metrics(self, epoch: int, metrics: Dict):
        """
        Update plots with new metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metric values
        """
        self.history['train_loss'].append(metrics.get('train_loss', 0))
        self.history['val_loss'].append(metrics.get('val_loss', 0))
        self.history['position_error'].append(metrics.get('position_error_cm', 0))
        self.history['rotation_error'].append(metrics.get('rotation_error_deg', 0))
        self.history['temperature'].append(metrics.get('temperature', 1))
        self.history['coherence'].append(metrics.get('coherence', 0.5))
        
        self._refresh_plots()
    
    def _refresh_plots(self):
        """Refresh all plot displays."""
        if not DEPS.get('matplotlib'):
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        self.ax_loss.clear()
        self.ax_loss.plot(epochs, self.history['train_loss'], 
                         color='#58a6ff', label='Train', linewidth=2)
        self.ax_loss.plot(epochs, self.history['val_loss'], 
                         color='#f0883e', label='Val', linewidth=2)
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss Curves', color='#c9d1d9')
        self.ax_loss.legend(facecolor='#21262d', edgecolor='#30363d', 
                           labelcolor='#c9d1d9')
        self.ax_loss.grid(True, alpha=0.2, color='#30363d')
        self.ax_loss.set_facecolor('#161b22')
        
        # Error curves
        self.ax_errors.clear()
        ax2 = self.ax_errors.twinx()
        l1 = self.ax_errors.plot(epochs, self.history['position_error'], 
                                color='#7ee787', label='Position (cm)', linewidth=2)
        l2 = ax2.plot(epochs, self.history['rotation_error'], 
                     color='#f778ba', label='Rotation (°)', linewidth=2)
        self.ax_errors.set_xlabel('Epoch')
        self.ax_errors.set_ylabel('Position Error (cm)', color='#7ee787')
        ax2.set_ylabel('Rotation Error (°)', color='#f778ba')
        self.ax_errors.set_title('Prediction Errors', color='#c9d1d9')
        self.ax_errors.grid(True, alpha=0.2, color='#30363d')
        self.ax_errors.set_facecolor('#161b22')
        ax2.set_facecolor('#161b22')
        ax2.spines['right'].set_color('#f778ba')
        ax2.tick_params(colors='#f778ba')
        
        # Field coherence
        self.ax_field.clear()
        self.ax_field.fill_between(epochs, 0, self.history['coherence'],
                                   color='#238636', alpha=0.3)
        self.ax_field.plot(epochs, self.history['coherence'], 
                          color='#3fb950', linewidth=2)
        self.ax_field.set_xlabel('Epoch')
        self.ax_field.set_ylabel('Coherence')
        self.ax_field.set_title('Field Coherence', color='#c9d1d9')
        self.ax_field.set_ylim(0, 1)
        self.ax_field.grid(True, alpha=0.2, color='#30363d')
        self.ax_field.set_facecolor('#161b22')
        
        # Temperature
        self.ax_thermo.clear()
        self.ax_thermo.fill_between(epochs, 0, self.history['temperature'],
                                    color='#da3633', alpha=0.3)
        self.ax_thermo.plot(epochs, self.history['temperature'], 
                           color='#f85149', linewidth=2)
        self.ax_thermo.set_xlabel('Epoch')
        self.ax_thermo.set_ylabel('Temperature')
        self.ax_thermo.set_title('Field Temperature', color='#c9d1d9')
        self.ax_thermo.grid(True, alpha=0.2, color='#30363d')
        self.ax_thermo.set_facecolor('#161b22')
        
        self.fig.tight_layout()
        self.draw()
    
    def clear_history(self):
        """Clear all history data."""
        for key in self.history:
            self.history[key] = []
        if DEPS.get('matplotlib'):
            self._setup_axes()
            self.draw()


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC CARD WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class MetricCard(QFrame):
    """
    Stylized metric display card.
    
    Shows a single metric value with label and optional icon/color.
    """
    
    def __init__(self, 
                 title: str, 
                 value: str = "—", 
                 color: str = "#58a6ff",
                 parent=None):
        """
        Initialize metric card.
        
        Args:
            title: Metric title/label
            value: Initial value to display
            color: Accent color for the value
            parent: Parent widget
        """
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setMinimumHeight(80)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        
        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.value_label = QLabel(value)
        self.value_label.setObjectName("metricValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setStyleSheet(f"color: {color};")
        
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
    
    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.setText(value)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONSOLE WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class RoboticsConsole(QMainWindow):
    """
    Main PyQt6 Robotics Training Console Window.
    
    Features:
        - Dataset locator and browser
        - Training configuration panel
        - Real-time metrics dashboard
        - Console log output
        - Interactive controls
    """
    
    def __init__(self):
        """Initialize the robotics console."""
        super().__init__()
        
        self.config = ConsoleConfig()
        self.dataset_locator = DatasetLocator(self.config)
        self.training_worker: Optional[TrainingWorker] = None
        self.selected_dataset: Optional[str] = None
        
        self._setup_ui()
        self._connect_signals()
        self._scan_for_datasets()
    
    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("QRFF Robotics Training Console")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(DARK_STYLESHEET)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        self._create_header(main_layout)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)
        
        # Left panel - Configuration
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Visualization
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter sizes
        splitter.setSizes([450, 950])
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def _create_header(self, parent_layout: QVBoxLayout):
        """Create the header section."""
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        # Title
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)
        
        title = QLabel("🤖 QRFF Robotics Training Console")
        title.setObjectName("titleLabel")
        
        subtitle = QLabel("NVIDIA PhysicalAI-Robotics-Manipulation-Augmented Dataset")
        subtitle.setObjectName("subtitleLabel")
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addWidget(title_container)
        
        header_layout.addStretch()
        
        # Status indicators
        self.device_label = QLabel()
        self._update_device_status()
        header_layout.addWidget(self.device_label)
        
        parent_layout.addWidget(header)
    
    def _update_device_status(self):
        """Update the device status indicator."""
        if DEPS.get('torch'):
            if DEPS.get('cuda'):
                import torch
                device_name = torch.cuda.get_device_name(0)
                self.device_label.setText(f"🟢 GPU: {device_name}")
                self.device_label.setStyleSheet("color: #3fb950;")
            else:
                self.device_label.setText("🟡 CPU Mode")
                self.device_label.setStyleSheet("color: #d29922;")
        else:
            self.device_label.setText("🔴 PyTorch Not Available")
            self.device_label.setStyleSheet("color: #f85149;")
    
    def _create_left_panel(self) -> QWidget:
        """Create the left configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        
        # Dataset section
        dataset_group = QGroupBox("📂 Dataset")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # Dataset combo
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumHeight(35)
        self.dataset_combo.addItem("No datasets found...")
        dataset_layout.addWidget(self.dataset_combo)
        
        # Dataset buttons
        btn_layout = QHBoxLayout()
        
        self.scan_btn = QPushButton("🔍 Scan")
        self.scan_btn.clicked.connect(self._scan_for_datasets)
        btn_layout.addWidget(self.scan_btn)
        
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.clicked.connect(self._browse_for_dataset)
        btn_layout.addWidget(self.browse_btn)
        
        self.download_btn = QPushButton("⬇️ Download")
        self.download_btn.setObjectName("accentButton")
        self.download_btn.clicked.connect(self._download_dataset)
        btn_layout.addWidget(self.download_btn)
        
        dataset_layout.addLayout(btn_layout)
        
        # Dataset info
        self.dataset_info = QLabel("Select a dataset to view details")
        self.dataset_info.setWordWrap(True)
        self.dataset_info.setStyleSheet("color: #8b949e; padding: 8px;")
        dataset_layout.addWidget(self.dataset_info)
        
        layout.addWidget(dataset_group)
        
        # Training config section
        config_group = QGroupBox("⚙️ Training Configuration")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(10)
        
        # Epochs
        config_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.config.default_epochs)
        config_layout.addWidget(self.epochs_spin, 0, 1)
        
        # Batch size
        config_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(self.config.default_batch_size)
        config_layout.addWidget(self.batch_spin, 1, 1)
        
        # Learning rate
        config_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(self.config.default_learning_rate)
        config_layout.addWidget(self.lr_spin, 2, 1)
        
        # Hidden dimension
        config_layout.addWidget(QLabel("Hidden Dim:"), 3, 0)
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(32, 1024)
        self.hidden_spin.setSingleStep(32)
        self.hidden_spin.setValue(self.config.default_hidden_dim)
        config_layout.addWidget(self.hidden_spin, 3, 1)
        
        # QRFF neurons
        config_layout.addWidget(QLabel("Field Neurons:"), 4, 0)
        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(8, 256)
        self.neurons_spin.setSingleStep(8)
        self.neurons_spin.setValue(self.config.default_n_neurons)
        config_layout.addWidget(self.neurons_spin, 4, 1)
        
        # Max demos
        config_layout.addWidget(QLabel("Max Demos:"), 5, 0)
        self.demos_spin = QSpinBox()
        self.demos_spin.setRange(10, 1000)
        self.demos_spin.setValue(100)
        config_layout.addWidget(self.demos_spin, 5, 1)
        
        layout.addWidget(config_group)
        
        # Control buttons
        controls_group = QGroupBox("🎮 Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.train_btn = QPushButton("▶️  START TRAINING")
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.setMinimumHeight(50)
        self.train_btn.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.train_btn.clicked.connect(self._toggle_training)
        controls_layout.addWidget(self.train_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        controls_layout.addWidget(self.progress_bar)
        
        layout.addWidget(controls_group)
        
        # Console output
        console_group = QGroupBox("💻 Console")
        console_layout = QVBoxLayout(console_group)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(200)
        self.console.setFont(QFont("Consolas", 10))
        console_layout.addWidget(self.console)
        
        layout.addWidget(console_group)
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 0, 0, 0)
        
        # Metrics cards row
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)
        
        self.epoch_card = MetricCard("Epoch", "0 / 0", "#58a6ff")
        self.loss_card = MetricCard("Loss", "—", "#f0883e")
        self.pos_card = MetricCard("Position Error", "— cm", "#7ee787")
        self.rot_card = MetricCard("Rotation Error", "— °", "#f778ba")
        self.temp_card = MetricCard("Field Temp", "—", "#f85149")
        self.coherence_card = MetricCard("Coherence", "—", "#3fb950")
        
        cards_layout.addWidget(self.epoch_card)
        cards_layout.addWidget(self.loss_card)
        cards_layout.addWidget(self.pos_card)
        cards_layout.addWidget(self.rot_card)
        cards_layout.addWidget(self.temp_card)
        cards_layout.addWidget(self.coherence_card)
        
        layout.addLayout(cards_layout)
        
        # Plots
        plots_group = QGroupBox("📈 Training Metrics")
        plots_layout = QVBoxLayout(plots_group)
        
        if DEPS.get('matplotlib'):
            self.metrics_plot = MetricsPlotWidget()
            plots_layout.addWidget(self.metrics_plot)
        else:
            placeholder = QLabel("Matplotlib not available.\n"
                                "Install with: pip install matplotlib")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #d29922; padding: 50px;")
            plots_layout.addWidget(placeholder)
            self.metrics_plot = None
        
        layout.addWidget(plots_group, 1)
        
        return panel
    
    def _connect_signals(self):
        """Connect widget signals to slots."""
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
    
    def _scan_for_datasets(self):
        """Scan for local HDF5 datasets."""
        self.console.append("🔍 Scanning for local datasets...")
        
        datasets = self.dataset_locator.scan_local_directories()
        
        self.dataset_combo.clear()
        
        if datasets:
            for ds in datasets:
                label = f"{ds['name']} ({ds['n_demos']} demos, {ds['size_mb']} MB)"
                if ds['is_nvidia']:
                    label = "🟢 " + label
                self.dataset_combo.addItem(label, ds['path'])
            
            self.console.append(f"✅ Found {len(datasets)} dataset(s)")
        else:
            self.dataset_combo.addItem("No datasets found")
            self.console.append("⚠️ No local datasets found. Try Download or Browse.")
    
    def _browse_for_dataset(self):
        """Open file dialog to browse for HDF5 file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 Dataset",
            str(Path.home()),
            "HDF5 Files (*.hdf5 *.h5);;All Files (*)"
        )
        
        if file_path:
            info = self.dataset_locator._analyze_hdf5(Path(file_path))
            if info and info['valid']:
                self.dataset_combo.addItem(
                    f"📁 {info['name']} ({info['n_demos']} demos)",
                    file_path
                )
                self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)
                self.console.append(f"✅ Added dataset: {file_path}")
            else:
                QMessageBox.warning(self, "Invalid Dataset",
                    "Selected file is not a valid robotics HDF5 dataset.")
    
    def _download_dataset(self):
        """Download dataset from HuggingFace."""
        if not DEPS.get('huggingface_hub'):
            QMessageBox.warning(self, "Missing Dependency",
                "huggingface_hub not installed.\n"
                "Install with: pip install huggingface_hub")
            return
        
        # Show download options
        msg = QMessageBox(self)
        msg.setWindowTitle("Download Dataset")
        msg.setText("Select dataset to download from NVIDIA PhysicalAI:")
        msg.setIcon(QMessageBox.Icon.Question)
        
        mimic_btn = msg.addButton("Mimic (1K demos)", QMessageBox.ButtonRole.ActionRole)
        cosmos_btn = msg.addButton("Cosmos (1K demos)", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        
        msg.exec()
        
        if msg.clickedButton() == mimic_btn:
            filename = "mimic_dataset_1k.hdf5"
        elif msg.clickedButton() == cosmos_btn:
            filename = "cosmos_dataset_1k.hdf5"
        else:
            return
        
        self.console.append(f"⬇️ Downloading {filename}...")
        self.console.append("   This may take a while (~35GB total dataset)...")
        
        # Download in background (simplified - would use thread in production)
        path = self.dataset_locator.download_from_huggingface(filename)
        
        if path:
            self.console.append(f"✅ Downloaded to: {path}")
            self._scan_for_datasets()
        else:
            self.console.append("❌ Download failed. Check internet connection.")
    
    def _on_dataset_selected(self, index: int):
        """Handle dataset selection change."""
        path = self.dataset_combo.itemData(index)
        if path:
            self.selected_dataset = path
            info = self.dataset_locator._analyze_hdf5(Path(path))
            if info:
                self.dataset_info.setText(
                    f"📊 Demonstrations: {info['n_demos']}\n"
                    f"💾 Size: {info['size_mb']} MB\n"
                    f"📍 Path: {info['path']}"
                )
            else:
                self.dataset_info.setText("Unable to read dataset info")
        else:
            self.selected_dataset = None
            self.dataset_info.setText("Select a dataset to view details")
    
    def _toggle_training(self):
        """Start or stop training."""
        if self.training_worker and self.training_worker.isRunning():
            # Stop training
            self.training_worker.request_stop()
            self.train_btn.setText("⏳ STOPPING...")
            self.train_btn.setEnabled(False)
        else:
            # Start training
            self._start_training()
    
    def _start_training(self):
        """Start the training process."""
        if not self.selected_dataset:
            QMessageBox.warning(self, "No Dataset",
                "Please select a dataset before training.")
            return
        
        if not DEPS.get('torch'):
            QMessageBox.warning(self, "Missing Dependency",
                "PyTorch not available.\n"
                "Install with: pip install torch")
            return
        
        # Clear previous
        self.console.clear()
        if self.metrics_plot:
            self.metrics_plot.clear_history()
        
        # Get config
        training_config = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'hidden_dim': self.hidden_spin.value(),
            'n_neurons': self.neurons_spin.value(),
            'max_demos': self.demos_spin.value(),
        }
        
        # Create and start worker
        self.training_worker = TrainingWorker(
            self.selected_dataset,
            training_config,
            self
        )
        
        self.training_worker.progress.connect(self._on_training_progress)
        self.training_worker.log_message.connect(self._on_log_message)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.error.connect(self._on_training_error)
        
        self.training_worker.start()
        
        # Update UI
        self.train_btn.setText("⏹️  STOP TRAINING")
        self.train_btn.setObjectName("dangerButton")
        self.train_btn.setStyleSheet(
            self.train_btn.styleSheet()  # Force style refresh
        )
        self.progress_bar.setMaximum(training_config['epochs'])
    
    def _on_training_progress(self, epoch: int, metrics: Dict):
        """Handle training progress update."""
        total_epochs = self.epochs_spin.value()
        
        # Update progress bar
        self.progress_bar.setValue(epoch)
        
        # Update metric cards
        self.epoch_card.set_value(f"{epoch} / {total_epochs}")
        self.loss_card.set_value(f"{metrics['train_loss']:.4f}")
        self.pos_card.set_value(f"{metrics['position_error_cm']:.2f} cm")
        self.rot_card.set_value(f"{metrics['rotation_error_deg']:.1f} °")
        self.temp_card.set_value(f"{metrics['temperature']:.3f}")
        self.coherence_card.set_value(f"{metrics['coherence']:.3f}")
        
        # Update plots
        if self.metrics_plot:
            self.metrics_plot.update_metrics(epoch, metrics)
        
        # Update status
        self.statusBar().showMessage(
            f"Training: Epoch {epoch}/{total_epochs} | "
            f"Loss: {metrics['train_loss']:.4f} | "
            f"LR: {metrics.get('learning_rate', 0):.6f}"
        )
    
    def _on_log_message(self, message: str):
        """Handle log message from training worker."""
        self.console.append(message)
        # Auto-scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _on_training_finished(self, results: Dict):
        """Handle training completion."""
        self.train_btn.setText("▶️  START TRAINING")
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.setEnabled(True)
        self.train_btn.setStyleSheet(self.train_btn.styleSheet())
        
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        self.statusBar().showMessage(
            f"Training complete! Final Position Error: "
            f"{results['final_position_error_cm']:.2f} cm | "
            f"Rotation Error: {results['final_rotation_error_deg']:.1f}°"
        )
        
        QMessageBox.information(self, "Training Complete",
            f"✅ Training finished successfully!\n\n"
            f"Final Position Error: {results['final_position_error_cm']:.2f} cm\n"
            f"Final Rotation Error: {results['final_rotation_error_deg']:.1f}°\n"
            f"Epochs Completed: {results['epochs_completed']}")
    
    def _on_training_error(self, error_msg: str):
        """Handle training error."""
        self.train_btn.setText("▶️  START TRAINING")
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.setEnabled(True)
        self.train_btn.setStyleSheet(self.train_btn.styleSheet())
        
        self.console.append(f"\n❌ ERROR: {error_msg}")
        self.statusBar().showMessage("Training failed - see console for details")
        
        QMessageBox.critical(self, "Training Error",
            f"Training failed:\n\n{error_msg[:500]}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Main entry point for the QRFF Robotics Training Console.
    
    Launches the PyQt6 application with the training interface.
    """
    # Print startup banner
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║   QRFF ROBOTICS TRAINING CONSOLE                                  ║")
    print("║   PyQt6 Edition - NVIDIA PhysicalAI Dataset                       ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    
    # Check dependencies
    deps = check_dependencies()
    print("║   Dependencies:                                                   ║")
    for name, available in deps.items():
        status = "✓" if available else "✗"
        print(f"║     {status} {name:<20}                                       ║")
    
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Create and show main window
    window = RoboticsConsole()
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()