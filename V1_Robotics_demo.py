"""
QRFF Robotics Demo: NVIDIA PhysicalAI Manipulation Dataset
============================================================
Trains a Quaternion Ruffle Field model on NVIDIA's Franka Panda
cube-stacking manipulation dataset for behavior cloning.

DATASET: nvidia/PhysicalAI-Robotics-Manipulation-Augmented
    - 1000 demonstrations of 3-cube stacking task
    - Franka Panda robot arm
    - Full state information: joints, EE pose, object poses
    - Actions: 6D relative EE motion + 1D gripper

TASK: Behavior Cloning via 6-DOF Pose Prediction
    Input:  Current robot state + object poses
    Output: Next end-effector pose (position + quaternion)
    
    The QRFF learns to map the current manipulation context to
    the appropriate gripper pose, leveraging:
        - Geodesic loss for proper SO(3) rotation learning
        - Motor memory for recalling similar manipulation primitives
        - Field dynamics as confidence/uncertainty indicator

FEATURES DEMONSTRATED:
    1. HuggingFace dataset loading with HDF5 extraction
    2. State/action preprocessing for QRFF
    3. Training with SE(3) geodesic loss
    4. Visualization of learned pose manifold
    5. Field dynamics evolution during training

REQUIREMENTS:
    pip install torch numpy h5py datasets matplotlib tqdm

Author: Built on Renee M Gagnon's QRFF Architecture
        Using NVIDIA PhysicalAI Dataset (CC-BY-4.0)
Date: January 2026
"""

import os
import sys
import math
import time
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

import numpy as np

# Handle optional imports gracefully
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("h5py not available. Install with: pip install h5py")

try:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("HuggingFace not available. Install with: pip install datasets huggingface_hub")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs): return x


# =============================================================================
# QRFF IMPORTS (with fallback for standalone testing)
# =============================================================================

try:
    from V5_Quantum_Ruffle_Field import (
        QuaternionFieldConfig,
        get_device,
        QuaternionRuffleField,
        QuaternionRuffleOptimizer,
        QuaternionSignalProcessor,
        QuaternionMemoryTracer,
        quaternion_normalize,
        quaternion_multiply,
        quaternion_slerp,
        quaternion_geodesic_distance,
        quaternion_inner_product,
        quaternion_conjugate,
    )
    QRFF_AVAILABLE = True
except ImportError:
    QRFF_AVAILABLE = False
    print("QRFF not available. Ensure V5_Quantum_Ruffle_Field.py is in path.")
    
    # Minimal fallbacks for testing
    def get_device():
        if TORCH_AVAILABLE:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return None
    
    def quaternion_normalize(q, eps=1e-8):
        return q / (q.norm(dim=-1, keepdim=True) + eps)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NVIDIAManipulationConfig:
    """
    Configuration for the NVIDIA Manipulation demo.
    
    Attributes:
        dataset_name: HuggingFace dataset identifier
        hdf5_filename: Which HDF5 file to use (mimic or cosmos)
        cache_dir: Local cache directory for downloaded data
        max_demos: Maximum demonstrations to load (None = all)
        sequence_length: Length of state sequences for training
        train_split: Fraction of data for training
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Optimizer learning rate
        hidden_dim: QRFF hidden dimension
        n_neurons: Number of neurons in quaternion field
        use_object_poses: Include cube poses in input
        use_gripper_state: Include gripper open/close in input
        position_weight: Loss weight for position component
        rotation_weight: Loss weight for rotation component
    """
    # Dataset
    dataset_name: str = "nvidia/PhysicalAI-Robotics-Manipulation-Augmented"
    hdf5_filename: str = "mimic_dataset_1k.hdf5"  # or "cosmos_dataset_1k.hdf5"
    cache_dir: str = "./nvidia_manipulation_cache"
    max_demos: Optional[int] = 100  # Limit for quick testing
    sequence_length: int = 10       # Timesteps per training sample
    
    # Training
    train_split: float = 0.8
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    
    # Model
    hidden_dim: int = 128
    n_neurons: int = 64
    
    # Input features
    use_object_poses: bool = True   # Include cube positions
    use_gripper_state: bool = True  # Include gripper open/close
    
    # Loss weights
    position_weight: float = 1.0
    rotation_weight: float = 1.0


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def download_nvidia_dataset(config: NVIDIAManipulationConfig) -> Path:
    """
    Download the NVIDIA manipulation dataset from HuggingFace.
    
    Args:
        config: Configuration object
        
    Returns:
        Path to downloaded HDF5 file
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    local_path = cache_dir / config.hdf5_filename
    
    if local_path.exists():
        print(f"Using cached dataset: {local_path}")
        return local_path
    
    if not HF_AVAILABLE:
        raise RuntimeError("HuggingFace hub not available. Install: pip install huggingface_hub")
    
    print(f"Downloading {config.hdf5_filename} from {config.dataset_name}...")
    print("(This may take a while for the first download ~35GB total)")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=config.dataset_name,
            filename=config.hdf5_filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
            local_dir=str(cache_dir),
        )
        return Path(downloaded_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Falling back to synthetic data for demo...")
        return None


def load_hdf5_demonstrations(
    hdf5_path: Path,
    max_demos: Optional[int] = None,
    verbose: bool = True
) -> List[Dict[str, np.ndarray]]:
    """
    Load demonstrations from HDF5 file.
    
    The NVIDIA dataset structure:
        /data/demo_0/
            /actions          - (T, 7) relative EE actions
            /obs/
                /robot0_joint_pos    - (T, 7) joint positions
                /robot0_joint_vel    - (T, 7) joint velocities  
                /robot0_eef_pos      - (T, 3) EE position
                /robot0_eef_quat     - (T, 4) EE quaternion [x,y,z,w] or [w,x,y,z]
                /robot0_gripper_qpos - (T, 2) gripper state
                /object              - (T, N*7) object poses
                
    Args:
        hdf5_path: Path to HDF5 file
        max_demos: Maximum demos to load
        verbose: Print progress
        
    Returns:
        List of demonstration dictionaries
    """
    if not H5PY_AVAILABLE:
        raise RuntimeError("h5py not available. Install: pip install h5py")
    
    demonstrations = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get demo keys
        data_group = f['data']
        demo_keys = sorted([k for k in data_group.keys() if k.startswith('demo_')])
        
        if max_demos is not None:
            demo_keys = demo_keys[:max_demos]
        
        if verbose:
            print(f"Loading {len(demo_keys)} demonstrations...")
            demo_iter = tqdm(demo_keys, desc="Loading demos")
        else:
            demo_iter = demo_keys
        
        for demo_key in demo_iter:
            demo = data_group[demo_key]
            
            # Extract data (handle different possible key names)
            demo_data = {}
            
            # Actions
            if 'actions' in demo:
                demo_data['actions'] = demo['actions'][:]
            
            # Observations
            obs = demo['obs']
            
            # Joint positions (try different key names)
            for key in ['robot0_joint_pos', 'joint_pos', 'robot_joint_pos']:
                if key in obs:
                    demo_data['joint_pos'] = obs[key][:]
                    break
            
            # Joint velocities
            for key in ['robot0_joint_vel', 'joint_vel', 'robot_joint_vel']:
                if key in obs:
                    demo_data['joint_vel'] = obs[key][:]
                    break
            
            # End-effector position
            for key in ['robot0_eef_pos', 'eef_pos', 'ee_pos']:
                if key in obs:
                    demo_data['eef_pos'] = obs[key][:]
                    break
            
            # End-effector quaternion
            for key in ['robot0_eef_quat', 'eef_quat', 'ee_quat']:
                if key in obs:
                    demo_data['eef_quat'] = obs[key][:]
                    break
            
            # Gripper state
            for key in ['robot0_gripper_qpos', 'gripper_qpos', 'gripper']:
                if key in obs:
                    demo_data['gripper'] = obs[key][:]
                    break
            
            # Object poses (cubes)
            for key in ['object', 'object_pose', 'cube_poses']:
                if key in obs:
                    demo_data['object_poses'] = obs[key][:]
                    break
            
            demonstrations.append(demo_data)
    
    if verbose:
        print(f"Loaded {len(demonstrations)} demonstrations")
        if demonstrations:
            sample = demonstrations[0]
            print("Sample shapes:")
            for k, v in sample.items():
                print(f"  {k}: {v.shape}")
    
    return demonstrations


def create_synthetic_manipulation_data(
    n_demos: int = 100,
    timesteps_per_demo: int = 100,
    seed: int = 42
) -> List[Dict[str, np.ndarray]]:
    """
    Create synthetic manipulation data for testing when real data unavailable.
    
    Simulates a simplified cube-stacking task with:
        - 7-DOF robot arm (Franka-like)
        - 3 cubes to stack
        - Smooth trajectories between pick/place poses
        
    Args:
        n_demos: Number of demonstrations
        timesteps_per_demo: Timesteps per demonstration
        seed: Random seed
        
    Returns:
        List of demonstration dictionaries
    """
    np.random.seed(seed)
    demonstrations = []
    
    # Workspace bounds
    ws_center = np.array([0.5, 0.0, 0.1])
    ws_range = np.array([0.3, 0.3, 0.3])
    
    # Cube colors/order: blue (bottom), red (middle), green (top)
    cube_stack_z = [0.025, 0.075, 0.125]  # 5cm cubes
    
    for demo_idx in range(n_demos):
        T = timesteps_per_demo
        
        # Initialize arrays
        joint_pos = np.zeros((T, 7))
        joint_vel = np.zeros((T, 7))
        eef_pos = np.zeros((T, 3))
        eef_quat = np.zeros((T, 4))
        gripper = np.zeros((T, 2))
        object_poses = np.zeros((T, 21))  # 3 cubes × 7 (pos + quat)
        actions = np.zeros((T, 7))
        
        # Random initial cube positions (on table)
        cube_positions = [
            ws_center + np.array([np.random.uniform(-0.1, 0.1), 
                                  np.random.uniform(-0.1, 0.1), 
                                  0.025])
            for _ in range(3)
        ]
        
        # Generate trajectory phases: approach → grasp → lift → place (repeated)
        phase_length = T // 6
        
        for t in range(T):
            phase = t // phase_length
            phase_progress = (t % phase_length) / phase_length
            
            # Smooth interpolation
            alpha = 0.5 * (1 - np.cos(np.pi * phase_progress))
            
            if phase == 0:  # Approach cube 0
                start = ws_center + np.array([0, 0, 0.3])
                end = cube_positions[0] + np.array([0, 0, 0.1])
                eef_pos[t] = start + alpha * (end - start)
                gripper[t] = [0.04, 0.04]  # Open
                
            elif phase == 1:  # Grasp cube 0
                eef_pos[t] = cube_positions[0] + np.array([0, 0, 0.05])
                gripper[t] = [0.04 - alpha * 0.03, 0.04 - alpha * 0.03]
                
            elif phase == 2:  # Lift cube 0
                start = cube_positions[0] + np.array([0, 0, 0.05])
                end = cube_positions[1] + np.array([0, 0, 0.15])
                eef_pos[t] = start + alpha * (end - start)
                gripper[t] = [0.01, 0.01]  # Closed
                
            elif phase == 3:  # Place on cube 1
                eef_pos[t] = cube_positions[1] + np.array([0, 0, 0.08])
                gripper[t] = [0.01 + alpha * 0.03, 0.01 + alpha * 0.03]
                
            elif phase == 4:  # Retreat
                start = cube_positions[1] + np.array([0, 0, 0.08])
                end = ws_center + np.array([0, 0, 0.3])
                eef_pos[t] = start + alpha * (end - start)
                gripper[t] = [0.04, 0.04]
                
            else:  # Hold
                eef_pos[t] = ws_center + np.array([0, 0, 0.3])
                gripper[t] = [0.04, 0.04]
            
            # Generate quaternion (pointing down with slight variation)
            angle = np.random.randn() * 0.1
            eef_quat[t] = [np.cos(np.pi/4 + angle/2), 
                          np.sin(np.pi/4 + angle/2), 0, 0]  # [w,x,y,z]
            
            # Simple IK approximation for joints
            joint_pos[t] = np.array([
                np.arctan2(eef_pos[t, 1], eef_pos[t, 0]),  # Base rotation
                np.arcsin(np.clip((eef_pos[t, 2] - 0.3) / 0.5, -1, 1)),  # Shoulder
                0.0,  # Elbow
                -np.pi/2 + np.random.randn() * 0.1,  # Wrist 1
                0.0,  # Wrist 2
                np.pi/4,  # Wrist 3
                gripper[t, 0] * 10,  # Finger
            ])
            
            # Joint velocities (finite difference)
            if t > 0:
                joint_vel[t] = (joint_pos[t] - joint_pos[t-1]) * 30  # 30Hz
            
            # Object poses (static for simplicity)
            for i, pos in enumerate(cube_positions):
                object_poses[t, i*7:i*7+3] = pos
                object_poses[t, i*7+3:i*7+7] = [1, 0, 0, 0]  # Identity quat
            
            # Actions (relative EE motion)
            if t > 0:
                actions[t, :3] = eef_pos[t] - eef_pos[t-1]
                # Relative rotation (simplified)
                actions[t, 3:6] = 0
                actions[t, 6] = gripper[t, 0] - gripper[t-1, 0]
        
        demonstrations.append({
            'joint_pos': joint_pos.astype(np.float32),
            'joint_vel': joint_vel.astype(np.float32),
            'eef_pos': eef_pos.astype(np.float32),
            'eef_quat': eef_quat.astype(np.float32),
            'gripper': gripper.astype(np.float32),
            'object_poses': object_poses.astype(np.float32),
            'actions': actions.astype(np.float32),
        })
    
    print(f"Generated {n_demos} synthetic demonstrations")
    return demonstrations


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class ManipulationDataset(Dataset):
    """
    PyTorch Dataset for manipulation demonstrations.
    
    Creates (state, next_pose) pairs for behavior cloning.
    State includes: joint positions, EE pose, gripper, object poses
    Target: Next timestep's end-effector pose
    """
    
    def __init__(
        self,
        demonstrations: List[Dict[str, np.ndarray]],
        config: NVIDIAManipulationConfig,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            demonstrations: List of demonstration dictionaries
            config: Configuration object
            device: Torch device
        """
        self.config = config
        self.device = device or get_device()
        
        # Build samples from demonstrations
        self.samples = []
        
        for demo in demonstrations:
            T = len(demo['eef_pos'])
            
            for t in range(T - 1):  # -1 because we predict next pose
                # Build state vector
                state_parts = []
                
                # Joint positions (7D)
                if 'joint_pos' in demo:
                    state_parts.append(demo['joint_pos'][t])
                
                # Current EE pose (7D: pos + quat)
                state_parts.append(demo['eef_pos'][t])
                state_parts.append(demo['eef_quat'][t])
                
                # Gripper state (1D or 2D)
                if config.use_gripper_state and 'gripper' in demo:
                    g = demo['gripper'][t]
                    if len(g.shape) == 0 or len(g) == 1:
                        state_parts.append(np.array([g.item() if len(g.shape)==0 else g[0]]))
                    else:
                        state_parts.append(np.array([g[0]]))  # Just first finger
                
                # Object poses (21D for 3 cubes)
                if config.use_object_poses and 'object_poses' in demo:
                    state_parts.append(demo['object_poses'][t])
                
                state = np.concatenate(state_parts).astype(np.float32)
                
                # Target: next EE pose
                target_pos = demo['eef_pos'][t + 1]
                target_quat = demo['eef_quat'][t + 1]
                
                # Ensure quaternion is [w,x,y,z] format and normalized
                if np.abs(np.linalg.norm(target_quat) - 1.0) > 0.01:
                    target_quat = target_quat / (np.linalg.norm(target_quat) + 1e-8)
                
                target = np.concatenate([target_pos, target_quat]).astype(np.float32)
                
                self.samples.append((state, target))
        
        # Compute input dimension
        self.input_dim = self.samples[0][0].shape[0]
        print(f"Dataset: {len(self.samples)} samples, input_dim={self.input_dim}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        state, target = self.samples[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


# =============================================================================
# QRFF MODEL FOR MANIPULATION
# =============================================================================

class QRFFManipulationModel(nn.Module):
    """
    QRFF-based model for robot manipulation pose prediction.
    
    Architecture:
        Input Encoder → QRFF Processor → Pose Decoder
        
    Learns to predict the next end-effector pose given current
    robot and environment state.
    """
    
    def __init__(
        self,
        input_dim: int,
        config: NVIDIAManipulationConfig,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.config = config
        self.device = device or get_device()
        
        # Input encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        
        # QRFF components (if available)
        if QRFF_AVAILABLE:
            field_config = QuaternionFieldConfig(
                n_neurons=config.n_neurons,
                enable_memory=True
            )
            self.ruffle_field = QuaternionRuffleField(
                n_neurons=config.n_neurons,
                device=self.device,
                enable_memory=True,
                config=field_config
            )
            self.signal_processor = QuaternionSignalProcessor(
                self.ruffle_field,
                hidden_dim=config.hidden_dim,
                use_attention=True,
                use_sequence_memory=True
            )
            self.has_qrff = True
        else:
            # Fallback: simple transformer-like processing
            self.processor = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.has_qrff = False
        
        # Position head (3D)
        self.position_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, 3)
        )
        
        # Quaternion head (4D)
        self.quaternion_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, 4)
        )
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: State tensor (batch, input_dim)
            
        Returns:
            Pose tensor (batch, 7) = [x, y, z, w, qx, qy, qz]
        """
        # Encode
        h = self.encoder(x)
        
        # Process through QRFF or fallback
        if self.has_qrff:
            # Add sequence dimension for QRFF
            h = h.unsqueeze(1)  # (batch, 1, hidden)
            h = self.signal_processor(h)
            h = h.squeeze(1)    # (batch, hidden)
        else:
            h = self.processor(h)
        
        # Decode pose
        position = self.position_head(h)
        quaternion = self.quaternion_head(h)
        
        # Normalize quaternion
        quaternion = quaternion / (quaternion.norm(dim=-1, keepdim=True) + 1e-8)
        
        return torch.cat([position, quaternion], dim=-1)
    
    def get_field_state(self) -> Dict[str, float]:
        """Get QRFF field state if available."""
        if self.has_qrff:
            with torch.no_grad():
                energy = self.ruffle_field.compute_folding_energy().item()
            return {
                'energy': energy,
                'temperature': self.ruffle_field.field_temperature.item(),
                'coherence': self.ruffle_field.coherence_factor.item(),
            }
        return {'energy': 0, 'temperature': 1, 'coherence': 1}


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class SE3GeodesicLoss(nn.Module):
    """
    SE(3) loss with geodesic distance for rotation.
    
    L = w_p * ||pos_pred - pos_target||² + w_r * arccos(|q_pred · q_target|)²
    """
    
    def __init__(self, position_weight: float = 1.0, rotation_weight: float = 1.0):
        super().__init__()
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.
        
        Args:
            pred: (batch, 7) predicted poses
            target: (batch, 7) target poses
            
        Returns:
            Dict with loss components
        """
        # Split
        pred_pos = pred[:, :3]
        pred_quat = pred[:, 3:]
        target_pos = target[:, :3]
        target_quat = target[:, 3:]
        
        # Normalize quaternions
        pred_quat = pred_quat / (pred_quat.norm(dim=-1, keepdim=True) + 1e-8)
        target_quat = target_quat / (target_quat.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Position loss (MSE)
        pos_error = (pred_pos - target_pos).pow(2).sum(dim=-1)
        pos_loss = pos_error.mean()
        
        # Rotation loss (geodesic)
        dot = (pred_quat * target_quat).sum(dim=-1).abs()
        dot = torch.clamp(dot, 0, 1 - 1e-7)
        rot_error = torch.acos(dot).pow(2)
        rot_loss = rot_error.mean()
        
        # Total
        total_loss = self.position_weight * pos_loss + self.rotation_weight * rot_loss
        
        return {
            'loss': total_loss,
            'position_loss': pos_loss,
            'rotation_loss': rot_loss,
            'position_error_cm': torch.sqrt(pos_error).mean() * 100,
            'rotation_error_deg': torch.rad2deg(torch.sqrt(rot_error)).mean()
        }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: NVIDIAManipulationConfig,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Train the manipulation model.
    
    Args:
        model: The QRFF model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration
        device: Torch device
        
    Returns:
        Training history dictionary
    """
    # Loss and optimizer
    loss_fn = SE3GeodesicLoss(
        position_weight=config.position_weight,
        rotation_weight=config.rotation_weight
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # QRFF optimizer (if available)
    qrff_optimizer = None
    if hasattr(model, 'has_qrff') and model.has_qrff and QRFF_AVAILABLE:
        qrff_optimizer = QuaternionRuffleOptimizer(
            model.ruffle_field,
            fold_threshold=1.5,
            ruffle_scale=0.05
        )
    
    # History
    history = {
        'train_loss': [],
        'val_loss': [],
        'position_error_cm': [],
        'rotation_error_deg': [],
        'temperature': [],
        'coherence': []
    }
    
    best_val_loss = float('inf')
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_state, batch_target in train_loader:
            batch_state = batch_state.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            
            pred = model(batch_state)
            loss_dict = loss_fn(pred, batch_target)
            loss = loss_dict['loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # QRFF dynamics
            if qrff_optimizer is not None:
                qrff_optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        pos_errors = []
        rot_errors = []
        
        with torch.no_grad():
            for batch_state, batch_target in val_loader:
                batch_state = batch_state.to(device)
                batch_target = batch_target.to(device)
                
                pred = model(batch_state)
                loss_dict = loss_fn(pred, batch_target)
                
                val_losses.append(loss_dict['loss'].item())
                pos_errors.append(loss_dict['position_error_cm'].item())
                rot_errors.append(loss_dict['rotation_error_deg'].item())
        
        # Update scheduler
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['position_error_cm'].append(np.mean(pos_errors))
        history['rotation_error_deg'].append(np.mean(rot_errors))
        
        field_state = model.get_field_state()
        history['temperature'].append(field_state['temperature'])
        history['coherence'].append(field_state['coherence'])
        
        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                  f"Train: {history['train_loss'][-1]:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Pos: {np.mean(pos_errors):.2f}cm | "
                  f"Rot: {np.mean(rot_errors):.2f}° | "
                  f"T: {field_state['temperature']:.3f}")
    
    return history


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_results(
    history: Dict[str, List[float]],
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str = "./nvidia_manipulation_results"
) -> None:
    """
    Create visualization plots.
    
    Args:
        history: Training history
        model: Trained model
        val_loader: Validation data
        device: Torch device
        output_dir: Output directory
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Training curves
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train', alpha=0.8)
    ax.plot(history['val_loss'], label='Validation', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Position error
    ax = axes[0, 1]
    ax.plot(history['position_error_cm'], color='blue', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Position Error (cm)')
    ax.set_title('Position Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 3. Rotation error
    ax = axes[0, 2]
    ax.plot(history['rotation_error_deg'], color='orange', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rotation Error (degrees)')
    ax.set_title('Rotation Accuracy')
    ax.grid(True, alpha=0.3)
    
    # 4. Field temperature
    ax = axes[1, 0]
    ax.plot(history['temperature'], color='red', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Temperature')
    ax.set_title('QRFF Field Temperature')
    ax.grid(True, alpha=0.3)
    
    # 5. Field coherence
    ax = axes[1, 1]
    ax.plot(history['coherence'], color='green', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Coherence')
    ax.set_title('QRFF Field Coherence')
    ax.grid(True, alpha=0.3)
    
    # 6. Prediction visualization (sample)
    ax = axes[1, 2]
    model.eval()
    with torch.no_grad():
        sample_states, sample_targets = next(iter(val_loader))
        sample_states = sample_states[:20].to(device)
        sample_targets = sample_targets[:20].to(device)
        predictions = model(sample_states)
        
        # Plot predicted vs actual positions
        pred_pos = predictions[:, :3].cpu().numpy()
        true_pos = sample_targets[:, :3].cpu().numpy()
        
        ax.scatter(true_pos[:, 0], true_pos[:, 1], c='blue', label='True', alpha=0.7, s=50)
        ax.scatter(pred_pos[:, 0], pred_pos[:, 1], c='red', label='Predicted', alpha=0.7, s=50, marker='x')
        
        # Draw lines connecting predictions to targets
        for i in range(len(pred_pos)):
            ax.plot([true_pos[i, 0], pred_pos[i, 0]], 
                   [true_pos[i, 1], pred_pos[i, 1]], 
                   'gray', alpha=0.3)
    
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Position Predictions (Sample)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


# =============================================================================
# MAIN DEMO
# =============================================================================

def run_nvidia_manipulation_demo(
    config: Optional[NVIDIAManipulationConfig] = None,
    use_synthetic: bool = False
) -> Dict[str, Any]:
    """
    Run the full NVIDIA manipulation demo.
    
    Args:
        config: Configuration object
        use_synthetic: Force use of synthetic data
        
    Returns:
        Results dictionary
    """
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required. Install with: pip install torch")
        return {}
    
    config = config or NVIDIAManipulationConfig()
    device = get_device()
    
    print("=" * 70)
    print("QRFF ROBOTICS: NVIDIA PhysicalAI Manipulation Demo")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dataset: {config.dataset_name}")
    print(f"QRFF Available: {QRFF_AVAILABLE}")
    print()
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("=" * 60)
    print("DATA LOADING")
    print("=" * 60)
    
    demonstrations = None
    
    if not use_synthetic:
        try:
            # Try to download real dataset
            hdf5_path = download_nvidia_dataset(config)
            if hdf5_path is not None and hdf5_path.exists():
                demonstrations = load_hdf5_demonstrations(
                    hdf5_path, 
                    max_demos=config.max_demos
                )
        except Exception as e:
            print(f"Could not load real dataset: {e}")
    
    if demonstrations is None:
        print("\nUsing synthetic data for demonstration...")
        demonstrations = create_synthetic_manipulation_data(
            n_demos=config.max_demos or 100,
            timesteps_per_demo=100
        )
    
    # =========================================================================
    # Create Dataset
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)
    
    dataset = ManipulationDataset(demonstrations, config, device)
    
    # Split
    n_train = int(len(dataset) * config.train_split)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size
    )
    
    # =========================================================================
    # Create Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("MODEL CREATION")
    print("=" * 60)
    
    model = QRFFManipulationModel(
        input_dim=dataset.input_dim,
        config=config,
        device=device
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Input dimension: {dataset.input_dim}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Field neurons: {config.n_neurons}")
    
    # =========================================================================
    # Train
    # =========================================================================
    history = train_model(model, train_loader, val_loader, config, device)
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    final_pos_error = history['position_error_cm'][-1]
    final_rot_error = history['rotation_error_deg'][-1]
    
    print(f"Final Position Error: {final_pos_error:.2f} cm")
    print(f"Final Rotation Error: {final_rot_error:.2f} degrees")
    
    field_state = model.get_field_state()
    print(f"\nFinal Field State:")
    print(f"  Temperature: {field_state['temperature']:.4f}")
    print(f"  Coherence: {field_state['coherence']:.4f}")
    print(f"  Energy: {field_state['energy']:.4f}")
    
    # =========================================================================
    # Visualize
    # =========================================================================
    visualize_results(history, model, val_loader, device)
    
    print("\n" + "=" * 70)
    print("✅ NVIDIA Manipulation Demo Complete!")
    print("=" * 70)
    
    return {
        'model': model,
        'history': history,
        'config': config,
        'final_position_error_cm': final_pos_error,
        'final_rotation_error_deg': final_rot_error,
        'field_state': field_state
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="QRFF Robotics Demo with NVIDIA PhysicalAI Dataset"
    )
    parser.add_argument(
        '--synthetic', 
        action='store_true',
        help='Use synthetic data instead of downloading real dataset'
    )
    parser.add_argument(
        '--max-demos',
        type=int,
        default=100,
        help='Maximum demonstrations to load (default: 100)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    
    args = parser.parse_args()
    
    config = NVIDIAManipulationConfig(
        max_demos=args.max_demos,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    results = run_nvidia_manipulation_demo(
        config=config,
        use_synthetic=args.synthetic
    )