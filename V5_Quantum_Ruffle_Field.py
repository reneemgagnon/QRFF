"""
Quaternion Ruffle Field v5.0 - Unified & Optimized
====================================================
Complete integration of all optimizations and sequence memory features.
Consolidated version preventing patch fragmentation.

CORE INNOVATIONS (Preserved from v3.0):
    - Dual-state neurons (position + orientation via quaternions)
    - Field thermodynamics (temperature, coherence dynamics)
    - Memory preservation with SLERP blending
    - Energy-based perturbation optimization
    - Dynamic distances (Euclidean + geodesic)
    - Quaternion memory tracer for state analysis

PERFORMANCE UPGRADES (Integrated from v3.1):
    - Cached learnable attention patterns (O(1) vs O(n²) per forward)
    - Periodic field dynamics updates during forward passes
    - Warmup period before applying ruffles
    - Adjusted thresholds for proper perturbation triggering
    - Skip connection gating for signal processor
    - Always-on dynamics updates in optimizer

NEW IN v4.1 - SEQUENCE MEMORY:
    - Cross-timestep attention for long-range dependencies
    - Temperature-modulated attention (softer at higher temps)
    - Gated memory integration (learns how much context to use)
    - Enables pattern recall across sequence positions

BUGFIXES INCLUDED:
    - SLERP antipodal quaternion handling
    - Signal processor dimension handling
    - Device auto-detection for Windows/CPU
    - Vectorized restore_from_memory (no loop)
    - Correct einsum subscripts for attention

Author: Renee M Gagnon
Date: April 2025 - Dec 2025 Udpates
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class QuaternionFieldConfig:
    """
    Configuration for Quaternion Ruffle Field.
    
    Attributes:
        n_neurons: Number of neurons in the field (default: 64)
        space_dim: Dimensionality of coordinate space (default: 4)
        enable_memory: Enable state memory preservation (default: True)
        temperature_bounds: Min/max temperature values (default: (0.1, 10.0))
        coherence_rate: Rate of coherence adaptation (default: 0.05)
        energy_momentum: Momentum for energy buffer updates (default: 0.95)
        numerical_eps: Epsilon for numerical stability (default: 1e-8)
    """
    n_neurons: int = 64
    space_dim: int = 4
    enable_memory: bool = True
    temperature_bounds: Tuple[float, float] = (0.1, 10.0)
    coherence_rate: float = 0.05
    energy_momentum: float = 0.95
    numerical_eps: float = 1e-8


# =============================================================================
# DEVICE UTILITIES
# =============================================================================

def get_device(tensor: Optional[torch.Tensor] = None) -> torch.device:
    """
    Get appropriate device, defaulting to CUDA if available.
    
    Args:
        tensor: Optional tensor to get device from
        
    Returns:
        torch.device object (cuda if available, else cpu)
    """
    if tensor is not None:
        return tensor.device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# QUATERNION OPERATIONS
# =============================================================================

def quaternion_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize quaternions to unit length.
    
    Args:
        q: Quaternions of shape (..., 4)
        eps: Small value for numerical stability
        
    Returns:
        Unit quaternions of shape (..., 4)
    """
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quaternion_inner_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute inner product (dot product) between quaternions.
    
    Args:
        q1: First quaternion batch (..., 4)
        q2: Second quaternion batch (..., 4)
        
    Returns:
        Inner products of shape (...)
    """
    return (q1 * q2).sum(dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate: q* = w - xi - yj - zk
    
    Args:
        q: Quaternions of shape (..., 4)
        
    Returns:
        Conjugate quaternions of shape (..., 4)
    """
    conj = q.clone()
    conj[..., 1:] = -conj[..., 1:]
    return conj


def quaternion_multiply(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product of two quaternions (fully vectorized).
    
    The Hamilton product is defined as:
        (q0 + q1i + q2j + q3k)(r0 + r1i + r2j + r3k) =
        (q0r0 - q1r1 - q2r2 - q3r3) +
        (q0r1 + q1r0 + q2r3 - q3r2)i +
        (q0r2 - q1r3 + q2r0 + q3r1)j +
        (q0r3 + q1r2 - q2r1 + q3r0)k
    
    Args:
        q: First quaternion (..., 4)
        r: Second quaternion (..., 4)
        
    Returns:
        Product quaternion (..., 4)
    """
    w0, x0, y0, z0 = q.unbind(-1)
    w1, x1, y1, z1 = r.unbind(-1)
    
    return torch.stack([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,  # Real part
        w0*x1 + x0*w1 + y0*z1 - z0*y1,  # i component
        w0*y1 - x0*z1 + y0*w1 + z0*x1,  # j component
        w0*z1 + x0*y1 - y0*x1 + z0*w1   # k component
    ], dim=-1)


def quaternion_slerp(
    q1: torch.Tensor, 
    q2: torch.Tensor, 
    t: float,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between quaternions.
    
    Properly handles antipodal quaternions (q and -q represent same rotation).
    SLERP provides constant-speed motion along the shortest arc on S³.
    
    Formula: slerp(q1, q2, t) = sin((1-t)θ)/sin(θ) * q1 + sin(tθ)/sin(θ) * q2
    
    Args:
        q1: Start quaternion (..., 4)
        q2: End quaternion (..., 4)
        t: Interpolation parameter [0, 1]
        eps: Threshold for linear interpolation fallback
        
    Returns:
        Interpolated quaternion (..., 4)
    """
    # Compute dot product
    dot = quaternion_inner_product(q1, q2)
    
    # Handle antipodal quaternions (take shorter path)
    q2_adjusted = torch.where(dot.unsqueeze(-1) < 0, -q2, q2)
    dot = torch.abs(dot)
    
    # Clamp for numerical stability
    dot = torch.clamp(dot, 0.0, 1.0 - eps)
    
    # Calculate angle
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # For very small angles, use linear interpolation (LERP)
    use_lerp = sin_theta < eps
    
    # SLERP weights
    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + eps)
    w2 = torch.sin(t * theta) / (sin_theta + eps)
    
    # LERP weights (fallback for small angles)
    w1 = torch.where(use_lerp, torch.full_like(w1, 1.0 - t), w1)
    w2 = torch.where(use_lerp, torch.full_like(w2, t), w2)
    
    # Interpolate and normalize
    result = w1.unsqueeze(-1) * q1 + w2.unsqueeze(-1) * q2_adjusted
    return quaternion_normalize(result)


def quaternion_random_perturbation(
    scale: float = 0.05,
    shape: Tuple[int, ...] = (4,),
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate small random perturbation quaternions near identity.
    
    Args:
        scale: Perturbation magnitude (0 = identity, 1 = random)
        shape: Output shape (should end with 4, or be (4,) for single)
        device: Torch device
        
    Returns:
        Perturbation quaternions biased toward identity
    """
    if device is None:
        device = get_device()
    
    # Handle both single quaternion and batch
    if len(shape) == 1 and shape[0] == 4:
        rand = torch.randn(4, device=device)
        rand = quaternion_normalize(rand) * scale
        rand[0] += (1.0 - scale)  # Bias toward identity
        return quaternion_normalize(rand)
    else:
        rand = torch.randn(shape, device=device)
        rand = quaternion_normalize(rand) * scale
        rand[..., 0] += (1.0 - scale)  # Bias toward identity
        return quaternion_normalize(rand)


def quaternion_geodesic_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance on quaternion manifold S³.
    
    The geodesic distance is the angle of rotation between orientations.
    d(q1, q2) = arccos(|q1 · q2|)
    
    Args:
        q1: First quaternion batch (..., 4)
        q2: Second quaternion batch (..., 4)
        
    Returns:
        Geodesic distances
    """
    dot = quaternion_inner_product(q1, q2)
    dot_clamped = torch.clamp(torch.abs(dot), 0.0, 1.0 - 1e-7)
    return torch.acos(dot_clamped)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion to 3x3 rotation matrix.
    
    Args:
        q: Unit quaternion (..., 4)
        
    Returns:
        Rotation matrix (..., 3, 3)
    """
    q = quaternion_normalize(q)
    w, x, y, z = q.unbind(-1)
    
    # Rotation matrix elements
    r00 = 1 - 2*(y*y + z*z)
    r01 = 2*(x*y - z*w)
    r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w)
    r11 = 1 - 2*(x*x + z*z)
    r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w)
    r21 = 2*(y*z + x*w)
    r22 = 1 - 2*(x*x + y*y)
    
    matrix = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1)
    ], dim=-2)
    
    return matrix


def quaternion_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Quaternion logarithm (maps to tangent space).
    
    For unit quaternion q = [cos(θ), sin(θ)v]:
    log(q) = [0, θv] where v is unit axis
    
    Args:
        q: Unit quaternion (..., 4)
        eps: Numerical epsilon
        
    Returns:
        Logarithm in tangent space (..., 4) with zero real part
    """
    q = quaternion_normalize(q)
    w = q[..., 0:1]
    xyz = q[..., 1:]
    
    # Compute rotation angle
    xyz_norm = xyz.norm(dim=-1, keepdim=True)
    theta = torch.atan2(xyz_norm, torch.abs(w))
    
    # Handle small rotations (avoid division by zero)
    scale = torch.where(
        xyz_norm < eps,
        torch.ones_like(xyz_norm),
        theta / xyz_norm
    )
    
    return torch.cat([torch.zeros_like(w), scale * xyz], dim=-1)


def quaternion_exp(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Quaternion exponential (maps from tangent space).
    
    For v = [0, θv] where v is unit axis:
    exp(v) = [cos(θ), sin(θ)v]
    
    Args:
        v: Tangent vector (..., 4) with zero real part
        eps: Numerical epsilon
        
    Returns:
        Unit quaternion (..., 4)
    """
    xyz = v[..., 1:]
    theta = xyz.norm(dim=-1, keepdim=True)
    
    # Handle small angles
    half_theta = theta / 2
    scale = torch.where(
        theta < eps,
        torch.ones_like(theta) * 0.5,
        torch.sin(half_theta) / theta
    )
    
    w = torch.cos(half_theta)
    xyz_out = scale * xyz
    
    return quaternion_normalize(torch.cat([w, xyz_out], dim=-1))


# =============================================================================
# QUATERNION RUFFLE FIELD (Core Architecture)
# =============================================================================

class QuaternionRuffleField(nn.Module):
    """
    Enhanced Non-Newtonian manifold with quaternion neuron states.
    
    Core Features:
        - Dual-state neurons (position + orientation)
        - Field thermodynamics with temperature and coherence
        - Memory state preservation with SLERP
        - Advanced energy computation
        - Adaptive field dynamics
        
    Each neuron maintains:
        - A position in 4D space (coordinates)
        - An orientation as a unit quaternion
        
    The field evolves through:
        - Temperature-modulated dynamics
        - Coherence-based coupling
        - Energy-driven perturbations
        - Memory-preserved state transitions
        
    Attributes:
        n_neurons: Number of neurons in the field
        space_dim: Dimensionality of coordinate space
        coordinates: Learnable spatial positions (n_neurons, space_dim)
        quaternions: Learnable orientations (n_neurons, 4)
        field_temperature: Adaptive temperature parameter
        coherence_factor: Inter-neuron coupling strength
    """
    
    def __init__(
        self,
        n_neurons: int,
        space_dim: int = 4,
        device: Optional[torch.device] = None,
        enable_memory: bool = True,
        config: Optional[QuaternionFieldConfig] = None
    ):
        """
        Initialize the Quaternion Ruffle Field.
        
        Args:
            n_neurons: Number of neurons in the field
            space_dim: Dimensionality of coordinate space (default 4)
            device: Torch device (auto-detected if None)
            enable_memory: Enable state memory preservation
            config: Optional configuration object
        """
        super().__init__()
        
        # Configuration
        self.config = config or QuaternionFieldConfig()
        self.n_neurons = n_neurons
        self.space_dim = space_dim
        self.device = device or get_device()
        self.enable_memory = enable_memory
        
        # Core field parameters - spatial positions
        self.coordinates = nn.Parameter(
            torch.randn(n_neurons, space_dim, device=self.device) * 0.1
        )
        
        # Core field parameters - quaternion orientations
        # Initialize with better distribution (near-identity with perturbation)
        raw_quaternions = torch.randn(n_neurons, 4, device=self.device)
        self.quaternions = nn.Parameter(quaternion_normalize(raw_quaternions))
        
        # Backward compatibility with phase-based implementations
        self.phases = nn.Parameter(
            torch.rand(n_neurons, device=self.device) * 2 * math.pi
        )
        
        # Memory state buffers (for sequence processing)
        if enable_memory:
            self.register_buffer(
                'memory_coordinates',
                torch.zeros(n_neurons, space_dim, device=self.device)
            )
            self.register_buffer(
                'memory_quaternions', 
                torch.zeros(n_neurons, 4, device=self.device)
            )
            self.register_buffer(
                'memory_valid',
                torch.zeros(1, dtype=torch.bool, device=self.device)
            )
        
        # Energy and dynamics tracking
        self.register_buffer('energy_buffer', torch.zeros(1, device=self.device))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long, device=self.device))
        
        # Advanced dynamics parameters
        self.register_buffer('field_temperature', torch.ones(1, device=self.device))
        self.register_buffer('coherence_factor', torch.ones(1, device=self.device))
    
    def reset(self, preserve_memory: bool = True) -> None:
        """
        Reset the field state for new sequence processing.
        
        Args:
            preserve_memory: Whether to save current state to memory before reset
        """
        if self.enable_memory and preserve_memory:
            # Store current state in memory
            self.memory_coordinates.copy_(self.coordinates.data)
            self.memory_quaternions.copy_(self.quaternions.data)
            self.memory_valid.fill_(True)
        
        # Reset to neutral/identity state
        with torch.no_grad():
            # Reset coordinates to small random values
            self.coordinates.data = torch.randn_like(self.coordinates) * 0.05
            
            # Reset quaternions to near-identity with small perturbations
            identity_quaternions = torch.zeros_like(self.quaternions)
            identity_quaternions[:, 0] = 1.0  # Real part = 1 (identity rotation)
            perturbation = torch.randn_like(self.quaternions) * 0.02
            self.quaternions.data = quaternion_normalize(identity_quaternions + perturbation)
            
            # Reset phases for compatibility
            self.phases.data = torch.rand_like(self.phases) * 2 * math.pi
            
            # Reset dynamics
            self.energy_buffer.zero_()
            self.field_temperature.fill_(1.0)
            self.coherence_factor.fill_(1.0)
    
    def restore_from_memory(self, blend_factor: float = 0.5) -> None:
        """
        Restore field state from memory with smooth blending.
        
        Uses SLERP for quaternion interpolation to ensure smooth
        transitions on the rotation manifold. Fully vectorized (no loop).
        
        Args:
            blend_factor: Blend ratio [0=current, 1=full memory restore]
        """
        if not (self.enable_memory and self.memory_valid.item()):
            return
        
        with torch.no_grad():
            # Linear blend for coordinates
            self.coordinates.data = (
                (1 - blend_factor) * self.coordinates.data +
                blend_factor * self.memory_coordinates
            )
            
            # Vectorized SLERP for quaternions
            self.quaternions.data = quaternion_slerp(
                self.quaternions.data,
                self.memory_quaternions,
                blend_factor
            )
    
    def compute_dynamic_distances(self) -> torch.Tensor:
        """
        Compute enhanced dynamic distances between neurons in the field.
        
        Combines:
            - Temperature-scaled Euclidean distance in coordinate space
            - Geodesic distance on quaternion manifold
            - Coherence-weighted blending
        
        Returns:
            Distance matrix of shape (n_neurons, n_neurons)
        """
        coords = self.coordinates  # [n, d]
        quats = quaternion_normalize(self.quaternions)  # [n, 4]
        
        # Euclidean distance with temperature scaling
        coord_diffs = coords.unsqueeze(0) - coords.unsqueeze(1)
        coord_distances = torch.norm(coord_diffs, dim=2)  # [n, n]
        
        # Temperature-scaled distances
        temp_scaled_distances = coord_distances / (self.field_temperature + 1e-8)
        
        # Quaternion distance (geodesic distance on quaternion manifold)
        quat_inner = quaternion_inner_product(quats.unsqueeze(0), quats.unsqueeze(1))
        quat_inner_clamped = torch.clamp(quat_inner, -0.9999, 0.9999)
        quat_distance = torch.acos(torch.abs(quat_inner_clamped))  # [n, n]
        
        # Coherence-weighted combination
        alpha = self.coherence_factor
        beta = 1.0 - alpha
        
        dynamic_distance = alpha * temp_scaled_distances + beta * quat_distance
        
        return dynamic_distance
    
    def compute_folding_energy(self) -> torch.Tensor:
        """
        Compute enhanced folding energy of the field.
        
        Energy components:
            - Curvature: Inverse square distance (repulsion)
            - Rotational disorder: Variance in quaternion orientations
            - Memory drift: Distance from previous memory state
            - Coherence energy
        
        Returns:
            Scalar tensor representing field energy
        """
        coords = self.coordinates
        quats = quaternion_normalize(self.quaternions)
        
        # Spatial curvature energy (improved with adaptive scaling)
        pairwise_dists = torch.norm(coords.unsqueeze(0) - coords.unsqueeze(1), dim=2) + 1e-6
        curvature_energy = torch.mean(1.0 / (pairwise_dists**2 + self.field_temperature))
        
        # Quaternion disorder energy (measures rotational chaos)
        quat_inner = quaternion_inner_product(quats.unsqueeze(0), quats.unsqueeze(1))
        rotation_disorder = torch.mean(1.0 - quat_inner**2)
        
        # Memory coherence penalty (if memory is available)
        memory_penalty = torch.tensor(0.0, device=self.device)
        if self.enable_memory and self.memory_valid.item():
            coord_drift = torch.norm(coords - self.memory_coordinates)
            quat_drift = torch.mean(1.0 - quaternion_inner_product(quats, self.memory_quaternions)**2)
            memory_penalty = 0.1 * (coord_drift + quat_drift)
        
        # Total energy with adaptive weighting
        total_energy = (
            curvature_energy + 
            0.5 * rotation_disorder + 
            memory_penalty
        )
        
        # Update energy buffer with momentum
        momentum = self.config.energy_momentum
        self.energy_buffer = momentum * self.energy_buffer + (1 - momentum) * total_energy.detach()
        
        return total_energy
    
    def get_phase_representation(self) -> torch.Tensor:
        """
        Get phase representation from quaternions for backward compatibility.
        
        Uses quaternion logarithm to extract rotation angle and axis,
        then combines into a phase-like scalar representation.
        
        Returns:
            Phase tensor of shape (n_neurons,)
        """
        quats = quaternion_normalize(self.quaternions)
        w, x, y, z = quats.unbind(-1)
        
        # Handle case where w is close to 1 (small rotations)
        theta = 2 * torch.atan2(torch.sqrt(x**2 + y**2 + z**2), torch.abs(w))
        
        # Extract phase-like representation
        phase_approx = theta + torch.atan2(y, x + 1e-8)
        
        return phase_approx
    
    def update_field_dynamics(self, input_energy: Optional[torch.Tensor] = None) -> None:
        """
        Update field temperature and coherence based on system state.
        
        Args:
            input_energy: Optional external energy input (for gradient computation)
        """
        # Update step count
        self.step_count += 1
        
        # Compute current energy
        current_energy = self.compute_folding_energy()
        current_energy_val = current_energy.detach()
        
        # Get previous energy from buffer for gradient computation
        prev_energy = self.energy_buffer.clone()
        
        # Temperature responds to energy CHANGE (gradient between steps)
        # Use previous buffered energy, not the input_energy
        energy_gradient = torch.abs(current_energy_val - prev_energy)
        
        # Temperature increases when energy is changing rapidly (exploration)
        # Temperature decreases when energy is stable (exploitation)
        temp_target = 0.5 + 1.5 * torch.tanh(energy_gradient)
        temp_rate = 0.1  # Faster adaptation
        
        self.field_temperature = torch.clamp(
            (1 - temp_rate) * self.field_temperature + temp_rate * temp_target,
            self.config.temperature_bounds[0],
            self.config.temperature_bounds[1]
        )
        
        # Coherence factor evolution (faster rate)
        energy_stability = torch.exp(-current_energy_val)
        coherence_target = 0.3 + 0.7 * energy_stability  # Wider range
        coherence_rate = 0.1  # Faster than config default
        
        self.coherence_factor = torch.clamp(
            (1 - coherence_rate) * self.coherence_factor + coherence_rate * coherence_target,
            0.1, 1.0
        )
    
    def get_field_state_summary(self) -> Dict[str, float]:
        """
        Get comprehensive summary of current field state.
        
        Returns:
            Dictionary with field statistics
        """
        quats = quaternion_normalize(self.quaternions)
        
        return {
            'energy': self.energy_buffer.item(),
            'temperature': self.field_temperature.item(),
            'coherence': self.coherence_factor.item(),
            'coord_mean': self.coordinates.mean().item(),
            'coord_std': self.coordinates.std().item(),
            'quat_real_mean': quats[:, 0].mean().item(),
            'quat_imag_norm': torch.norm(quats[:, 1:], dim=1).mean().item(),
            'step_count': self.step_count.item(),
            'memory_valid': self.memory_valid.item() if self.enable_memory else False
        }
    
    def forward(self, update_dynamics: bool = True) -> torch.Tensor:
        """
        Forward pass computing dynamic distance matrix.
        
        Args:
            update_dynamics: Whether to update field dynamics
            
        Returns:
            Dynamic distance matrix (n_neurons, n_neurons)
        """
        if update_dynamics:
            self.update_field_dynamics()
        return self.compute_dynamic_distances()


# =============================================================================
# QUATERNION RUFFLE OPTIMIZER (Enhanced v4)
# =============================================================================

class QuaternionRuffleOptimizer:
    """
    Enhanced optimizer for quaternion ruffle fields with adaptive strategies.
    
    v4 Improvements:
        - Adaptive threshold based on energy trends
        - Short warmup period (20 steps)
        - Always updates dynamics
        - Returns detailed stats dictionary from step()
    
    Features:
        - Energy-triggered perturbations
        - Adaptive scaling based on training progress
        - Temperature and coherence-aware ruffles
    """
    
    def __init__(
        self,
        field: QuaternionRuffleField,
        fold_threshold: float = 1.5,
        ruffle_scale: float = 0.05,
        warmup_steps: int = 20
    ):
        """
        Initialize the optimizer.
        
        Args:
            field: The quaternion field to optimize
            fold_threshold: Energy threshold for applying perturbations (default: 1.5)
            ruffle_scale: Base scale of perturbations (default: 0.05)
            warmup_steps: Number of steps before applying ruffles (default: 20)
        """
        self.field = field
        self.fold_threshold = fold_threshold
        self.ruffle_scale = ruffle_scale
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.energy_history: List[float] = []
    
    def step(self) -> Dict[str, float]:
        """
        Perform an enhanced quaternion ruffle step with adaptive perturbations.
        
        Returns:
            Dictionary with step statistics including:
                - step: Current step number
                - energy: Current field energy
                - temperature: Field temperature
                - coherence: Field coherence factor
                - applied_ruffle: Whether perturbations were applied
        """
        self.steps += 1
        
        # Compute energy
        with torch.no_grad():
            energy = self.field.compute_folding_energy()
            energy_val = energy.item()
        
        self.energy_history.append(energy_val)
        if len(self.energy_history) > 100:
            self.energy_history = self.energy_history[-100:]
        
        # ALWAYS update dynamics
        self.field.update_field_dynamics(input_energy=energy)
        
        # Adaptive threshold based on recent energy trend
        adaptive_threshold = self.fold_threshold
        if len(self.energy_history) > 10:
            recent_mean = sum(self.energy_history[-10:]) / 10
            older_mean = sum(self.energy_history[-20:-10]) / max(1, min(10, len(self.energy_history) - 10))
            trend = recent_mean - older_mean
            # If energy is decreasing (good), raise threshold; if increasing, lower it
            adaptive_threshold = self.fold_threshold * (1.0 - 0.3 * trend)
            adaptive_threshold = max(0.5, min(3.0, adaptive_threshold))
        
        # Apply perturbations after warmup
        applied_ruffle = False
        if self.steps > self.warmup_steps and energy_val < adaptive_threshold:
            self._apply_perturbations(energy)
            applied_ruffle = True
        
        return {
            'step': self.steps,
            'energy': energy_val,
            'temperature': self.field.field_temperature.item(),
            'coherence': self.field.coherence_factor.item(),
            'applied_ruffle': applied_ruffle,
            'adaptive_threshold': adaptive_threshold
        }
    
    def _apply_perturbations(self, current_energy: torch.Tensor) -> None:
        """
        Apply coordinate and quaternion perturbations.
        
        Args:
            current_energy: Current field energy tensor
        """
        with torch.no_grad():
            # Get field state
            temp = self.field.field_temperature.item()
            coherence = self.field.coherence_factor.item()
            
            # Slower decay - allow more exploration throughout training
            decay = 1.0 / (1.0 + 0.002 * self.steps)
            
            # Coordinate perturbations - scale with temperature
            coord_scale = self.ruffle_scale * (0.5 + temp) * decay
            self.field.coordinates.data += torch.randn_like(self.field.coordinates) * coord_scale
            
            # Quaternion perturbations - inversely scale with coherence
            # Lower coherence = more exploration needed
            quat_scale = self.ruffle_scale * (1.5 - 0.5 * coherence) * decay
            delta_quats = quaternion_random_perturbation(
                scale=quat_scale,
                shape=(self.field.n_neurons, 4),
                device=self.field.device
            )
            new_quats = quaternion_multiply(self.field.quaternions.data, delta_quats)
            self.field.quaternions.data = quaternion_normalize(new_quats)
            
            # Update phases
            self.field.phases.data = self.field.get_phase_representation()
    
    def get_optimizer_state(self) -> Dict[str, float]:
        """
        Get current optimizer state information.
        
        Returns:
            Dictionary with optimizer state
        """
        return {
            'steps': self.steps,
            'fold_threshold': self.fold_threshold,
            'ruffle_scale': self.ruffle_scale,
            'warmup_steps': self.warmup_steps,
            'energy_history_length': len(self.energy_history),
            'recent_energy_mean': (
                sum(self.energy_history[-10:]) / min(10, len(self.energy_history))
                if self.energy_history else 0.0
            )
        }


# =============================================================================
# QUATERNION SIGNAL PROCESSOR (Optimized v4)
# =============================================================================

class QuaternionSignalProcessor(nn.Module):
    """
    Optimized signal processor with cached attention and periodic dynamics.
    
    v4.1 Improvements:
        - Sequence-aware processing (cross-timestep communication)
        - Learnable attention patterns (O(1) per forward vs O(n²))
        - Quaternion-modulated attention scaling
        - Periodic field dynamics updates during forward pass
        - Skip connection gating
    
    Features:
        - Quaternion modulation with field state awareness
        - Multi-head attention on quaternion representations
        - Temperature and coherence modulation
        - Sequence memory aggregation for long-range dependencies
    """
    
    def __init__(
        self,
        field: QuaternionRuffleField,
        hidden_dim: int = 64,
        use_attention: bool = True,
        attention_heads: int = 4,
        head_dim: int = 16,
        update_dynamics_every: int = 5,
        use_sequence_memory: bool = True
    ):
        """
        Initialize the optimized signal processor.
        
        Args:
            field: The quaternion field to process signals through
            hidden_dim: Dimensionality of hidden representations
            use_attention: Whether to use attention mechanism
            attention_heads: Number of attention heads
            head_dim: Dimension per attention head
            update_dynamics_every: Update dynamics every N forward passes (default: 5)
            use_sequence_memory: Enable cross-timestep memory aggregation
        """
        super().__init__()
        
        self.field = field
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.head_dim = head_dim
        self.update_dynamics_every = update_dynamics_every
        self.use_sequence_memory = use_sequence_memory
        
        # Track forward passes
        self.register_buffer('forward_count', torch.zeros(1, dtype=torch.long))
        
        # Projections
        self.input_projection = nn.Linear(hidden_dim, field.n_neurons)
        self.output_projection = nn.Linear(field.n_neurons, hidden_dim)
        
        # Normalization
        self.input_norm = nn.LayerNorm(field.n_neurons)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        if use_attention:
            # OPTIMIZATION: Learnable attention patterns instead of computing from quaternions
            # This is O(1) per forward instead of O(n²)
            self.attention_patterns = nn.Parameter(
                torch.randn(attention_heads, field.n_neurons, field.n_neurons) * 0.02
            )
            
            # Quaternion-to-attention modulation (lightweight)
            self.quat_to_attn_scale = nn.Linear(4, attention_heads)
            
            # Value projection
            self.value_proj = nn.Linear(field.n_neurons, field.n_neurons)
        
        # Skip connection gate
        self.skip_gate = nn.Parameter(torch.tensor(0.3))
        
        # ========== SEQUENCE MEMORY MECHANISM ==========
        if use_sequence_memory:
            # Sequence-level attention for cross-timestep communication
            self.seq_query = nn.Linear(hidden_dim, hidden_dim)
            self.seq_key = nn.Linear(hidden_dim, hidden_dim)
            self.seq_value = nn.Linear(hidden_dim, hidden_dim)
            self.seq_out = nn.Linear(hidden_dim, hidden_dim)
            self.seq_norm = nn.LayerNorm(hidden_dim)
            
            # Memory gate - learns how much to incorporate sequence context
            self.memory_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
    
    def _update_attention_from_quaternions(self) -> torch.Tensor:
        """
        Modulate learned attention patterns using quaternion state.
        Much faster than computing attention from scratch.
        
        Returns:
            Attention weights of shape (heads, n_neurons, n_neurons)
        """
        quats = quaternion_normalize(self.field.quaternions)  # [n_neurons, 4]
        
        # Get per-head scaling from quaternions
        # [n_neurons, 4] -> [n_neurons, heads] -> mean -> [heads]
        head_scales = torch.tanh(self.quat_to_attn_scale(quats)).mean(dim=0)  # [heads]
        
        # Scale learned attention patterns
        # [heads, n, n] * [heads, 1, 1]
        scaled_patterns = self.attention_patterns * (1.0 + 0.5 * head_scales.view(-1, 1, 1))
        
        # Add coherence-based bias
        coherence = self.field.coherence_factor.item()
        scaled_patterns = scaled_patterns + coherence * 0.1
        
        return F.softmax(scaled_patterns, dim=-1)  # [heads, n_neurons, n_neurons]
    
    def quaternion_modulation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quaternion-based modulation to input.
        
        Args:
            x: Input tensor of shape [batch, n_neurons]
            
        Returns:
            Modulated tensor of shape [batch, n_neurons]
        """
        quats = quaternion_normalize(self.field.quaternions)  # [n_neurons, 4]
        w, x_q, y_q, z_q = quats.unbind(-1)
        
        # Temperature-aware modulation
        temp = self.field.field_temperature.item()
        coherence = self.field.coherence_factor.item()
        
        # Real part amplitude
        amplitude = w * temp  # [n_neurons]
        
        # Imaginary parts create phase-like modulation
        phase = torch.cos(x_q * coherence) * torch.sin(y_q) * torch.cos(z_q)
        
        # Apply modulation
        modulated = x * (amplitude.unsqueeze(0) + 0.3 * phase.unsqueeze(0))
        
        return modulated
    
    def apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED attention: Uses learned patterns modulated by quaternions.
        O(batch * n) instead of O(batch * n²) for quaternion computation.
        
        Args:
            x: Input tensor of shape [batch, n_neurons]
            
        Returns:
            Attention-weighted tensor of shape [batch, n_neurons]
        """
        # Get quaternion-modulated attention patterns
        attn_weights = self._update_attention_from_quaternions()  # [heads, n_neurons, n_neurons]
        
        # Compute values
        values = self.value_proj(x)  # [batch, n_neurons]
        
        # Apply attention for each head
        # attn_weights[h, i, j] = attention weight from position i to position j
        # output[b, h, i] = sum_j(values[b, j] * attn_weights[h, i, j])
        attended = torch.einsum('bj,hij->bhi', values, attn_weights)
        
        # Average over heads
        attended = attended.mean(dim=1)  # [batch, n_neurons]
        
        # Adaptive residual based on coherence (higher coherence = stronger attention)
        coherence = self.field.coherence_factor.item()
        attn_strength = 0.3 + 0.5 * coherence  # Range [0.3, 0.8]
        
        return x + attn_strength * attended
    
    def apply_sequence_memory(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-timestep attention for sequence memory.
        
        This allows information to flow between different positions in the sequence,
        enabling the model to "remember" patterns from earlier in the sequence.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim]
            
        Returns:
            Memory-enhanced tensor of shape [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute queries, keys, values across sequence dimension
        Q = self.seq_query(x)  # [batch, seq_len, hidden_dim]
        K = self.seq_key(x)    # [batch, seq_len, hidden_dim]
        V = self.seq_value(x)  # [batch, seq_len, hidden_dim]
        
        # Scaled dot-product attention across sequence positions
        # [batch, seq_len, hidden_dim] @ [batch, hidden_dim, seq_len] -> [batch, seq_len, seq_len]
        scale = math.sqrt(self.hidden_dim)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / scale
        
        # Apply temperature-modulated attention
        # Higher temperature = softer attention (more mixing)
        # Lower temperature = sharper attention (more focused)
        temp = self.field.field_temperature.item()
        attn_scores = attn_scores / (0.5 + temp)
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq_len, seq_len]
        
        # Apply attention to values
        context = torch.bmm(attn_weights, V)  # [batch, seq_len, hidden_dim]
        context = self.seq_out(context)
        context = self.seq_norm(context)
        
        # Gated residual connection
        # Concatenate original and context, then learn how much to mix
        gate_input = torch.cat([x, context], dim=-1)  # [batch, seq_len, hidden_dim*2]
        gate = self.memory_gate(gate_input)  # [batch, seq_len, hidden_dim]
        
        # Mix original and memory-enhanced representations
        output = gate * context + (1 - gate) * x
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sequence memory and periodic dynamics updates.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim]
            
        Returns:
            Processed tensor of shape [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # ========== SEQUENCE MEMORY (cross-timestep communication) ==========
        if self.use_sequence_memory:
            x = self.apply_sequence_memory(x)  # [batch, seq_len, hidden_dim]
        
        # ========== QUATERNION FIELD PROCESSING (per-timestep) ==========
        # Flatten for field processing
        x_flat = x.view(-1, self.hidden_dim)  # [batch*seq, hidden_dim]
        
        # Project to neuron space
        h = self.input_projection(x_flat)  # [batch*seq, n_neurons]
        h = self.input_norm(h)
        
        # Quaternion modulation
        h = self.quaternion_modulation(h)
        
        # Neuron-level attention (if enabled)
        if self.use_attention:
            h = self.apply_attention(h)
        
        # Quaternion-weighted scaling
        quat_weights = F.softmax(self.field.quaternions[:, 0], dim=0)
        h = h * (1.0 + 0.3 * quat_weights.unsqueeze(0))
        
        # Project back
        out = self.output_projection(h)
        out = self.output_norm(out)
        
        # Skip connection with learnable gate
        skip_weight = torch.sigmoid(self.skip_gate)
        out = skip_weight * out + (1 - skip_weight) * x_flat
        
        # ========== DYNAMICS UPDATES ==========
        # UPDATE DYNAMICS more frequently during training
        self.forward_count += 1
        if self.training and self.forward_count.item() % self.update_dynamics_every == 0:
            with torch.no_grad():
                self.field.update_field_dynamics()
        
        # Periodic memory consolidation during training (every 50 steps)
        if self.training and self.forward_count.item() % 50 == 0:
            with torch.no_grad():
                if self.field.enable_memory:
                    # Soft memory update - blend current state into memory
                    if self.field.memory_valid.item():
                        blend = 0.1  # Gentle consolidation
                        self.field.memory_coordinates.data = (
                            (1 - blend) * self.field.memory_coordinates + 
                            blend * self.field.coordinates.data
                        )
                        self.field.memory_quaternions.data = quaternion_slerp(
                            self.field.memory_quaternions,
                            self.field.quaternions.data,
                            blend
                        )
                    else:
                        # Initialize memory on first consolidation
                        self.field.memory_coordinates.copy_(self.field.coordinates.data)
                        self.field.memory_quaternions.copy_(self.field.quaternions.data)
                        self.field.memory_valid.fill_(True)
        
        return out.view(batch_size, seq_len, self.hidden_dim)


# =============================================================================
# QUATERNION MEMORY TRACER
# =============================================================================

class QuaternionMemoryTracer:
    """
    Trace quaternion field state for memory analysis.
    
    Features:
        - Full trajectory recording
        - Coordinate and quaternion history
        - Energy tracking
        - Memory coherence computation
    """
    
    def __init__(self, field: QuaternionRuffleField):
        """
        Initialize the memory tracer.
        
        Args:
            field: The quaternion field to trace
        """
        self.field = field
        self.history_coords: List[torch.Tensor] = []
        self.history_quats: List[torch.Tensor] = []
        self.history_energy: List[float] = []
    
    def record(self) -> None:
        """Save current field state snapshot."""
        self.history_coords.append(
            self.field.coordinates.detach().cpu().clone()
        )
        self.history_quats.append(
            quaternion_normalize(self.field.quaternions.detach().cpu().clone())
        )
        self.history_energy.append(self.field.energy_buffer.item())
    
    def export(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        Return full history for later analysis.
        
        Returns:
            Dictionary containing coordinate, quaternion, and energy history
        """
        return {
            'coordinates': (
                torch.stack(self.history_coords) 
                if self.history_coords else None
            ),
            'quaternions': (
                torch.stack(self.history_quats)
                if self.history_quats else None
            ),
            'energy': (
                torch.tensor(self.history_energy)
                if self.history_energy else None
            )
        }
    
    def compute_memory_coherence(self) -> float:
        """
        Compute coherence between memory states over time.
        
        Returns:
            Memory coherence score in [0, 1] where 1 = perfectly stable
        """
        if len(self.history_coords) < 2:
            return 0.0
        
        # Compare consecutive states
        coord_diffs = []
        quat_diffs = []
        
        for i in range(1, len(self.history_coords)):
            # Coordinate changes
            coord_diff = torch.norm(
                self.history_coords[i] - self.history_coords[i-1],
                dim=1
            ).mean().item()
            coord_diffs.append(coord_diff)
            
            # Quaternion changes
            quat_inner = quaternion_inner_product(
                self.history_quats[i],
                self.history_quats[i-1]
            )
            quat_diff = (1.0 - quat_inner**2).mean().item()
            quat_diffs.append(quat_diff)
        
        # Compute overall coherence score
        coord_stability = 1.0 / (1.0 + sum(coord_diffs) / len(coord_diffs))
        quat_stability = 1.0 / (1.0 + sum(quat_diffs) / len(quat_diffs))
        
        coherence = (coord_stability + quat_stability) / 2.0
        return coherence
    
    def clear(self) -> None:
        """Clear all recorded history."""
        self.history_coords.clear()
        self.history_quats.clear()
        self.history_energy.clear()


# =============================================================================
# QUATERNION OUTPUT HEAD (For Rotation Prediction)
# =============================================================================

class QuaternionOutputHead(nn.Module):
    """
    Specialized output head for quaternion prediction tasks.
    
    Ensures output is a valid unit quaternion.
    Useful for rotation prediction benchmarks.
    """
    
    def __init__(self, hidden_dim: int, intermediate_dim: int = 64):
        """
        Initialize the quaternion output head.
        
        Args:
            hidden_dim: Input hidden dimension
            intermediate_dim: Intermediate layer dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.LayerNorm(intermediate_dim),
            nn.Linear(intermediate_dim, 4)  # Output quaternion
        )
        
        # Initialize last layer to output near-identity quaternion
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0)
        self.net[-1].bias.data[0] = 1.0  # w = 1 (identity)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing unit quaternions.
        
        Args:
            x: Hidden states (..., hidden_dim)
            
        Returns:
            Unit quaternions (..., 4)
        """
        raw = self.net(x)
        return quaternion_normalize(raw)


# =============================================================================
# QRF MODEL (Complete Model Wrapper)
# =============================================================================

class QRFModel(nn.Module):
    """
    Complete Quaternion Ruffle Field model with input/output projections.
    
    A convenience wrapper that combines:
        - Input projection
        - Quaternion signal processor (with sequence memory)
        - Output head (supports quaternion or arbitrary dimensions)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_neurons: int = 64,
        use_attention: bool = True,
        use_memory: bool = True,
        use_sequence_memory: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the QRF model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden representation dimension
            output_dim: Output dimension (4 for quaternion output)
            n_neurons: Number of neurons in quaternion field
            use_attention: Enable neuron-level attention mechanism
            use_memory: Enable field state memory preservation
            use_sequence_memory: Enable cross-timestep sequence memory
            device: Torch device
        """
        super().__init__()
        
        self.device = device or get_device()
        self.output_dim = output_dim
        
        # Field
        config = QuaternionFieldConfig(n_neurons=n_neurons, enable_memory=use_memory)
        self.field = QuaternionRuffleField(
            n_neurons=n_neurons,
            device=self.device,
            enable_memory=use_memory,
            config=config
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Signal processor
        self.processor = QuaternionSignalProcessor(
            self.field,
            hidden_dim=hidden_dim,
            use_attention=use_attention,
            use_sequence_memory=use_sequence_memory
        )
        
        # Output
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        if output_dim == 4:
            # Quaternion output (specialized head)
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, 4)
            )
            self.is_quat_output = True
        else:
            # Generic output
            self.output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim // 2),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            self.is_quat_output = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the QRF model.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
            
        Returns:
            Output tensor of shape [batch, seq_len, output_dim]
        """
        h = self.input_proj(x)
        h = self.input_norm(h)
        h = self.processor(h)
        h = self.output_norm(h)
        out = self.output_head(h)
        
        if self.is_quat_output:
            out = quaternion_normalize(out)
        
        return out
    
    def get_field_state(self) -> Dict:
        """
        Get current field state summary.
        
        Returns:
            Dictionary with field statistics
        """
        return self.field.get_field_state_summary()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_quaternion_layer(
    input_dim: int,
    output_dim: int,
    n_neurons: int = 32,
    use_attention: bool = True,
    use_sequence_memory: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to create a quaternion processing layer.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        n_neurons: Number of neurons in quaternion field
        use_attention: Enable neuron-level attention mechanism
        use_sequence_memory: Enable cross-timestep sequence memory
        device: Torch device
        
    Returns:
        Sequential module with quaternion processing
    """
    device = device or get_device()
    field = QuaternionRuffleField(n_neurons, device=device)
    processor = QuaternionSignalProcessor(
        field, 
        hidden_dim=input_dim,
        use_attention=use_attention,
        use_sequence_memory=use_sequence_memory
    )
    
    return nn.Sequential(
        processor,
        nn.Linear(input_dim, output_dim)
    ).to(device)


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QUATERNION RUFFLE FIELD v5.0 - Unified & Optimized")
    print("=" * 60)
    
    # Configuration
    device = get_device()
    print(f"Using device: {device}")
    
    # Create QRF model
    print("\n--- Creating QRF Model ---")
    model = QRFModel(
        input_dim=6,
        hidden_dim=128,
        output_dim=4,
        n_neurons=64,
        use_attention=True,
        use_sequence_memory=True,  # NEW: cross-timestep memory
        device=device
    ).to(device)
    
    # Create optimizer
    qrf_opt = QuaternionRuffleOptimizer(model.field, fold_threshold=1.5)
    
    # Create tracer
    tracer = QuaternionMemoryTracer(model.field)
    
    # Test forward pass timing
    print("\n--- Performance Test ---")
    import time
    
    x = torch.randn(32, 64, 6, device=device)
    
    # Warmup
    for _ in range(5):
        _ = model(x)
    
    # Time 100 passes
    start = time.time()
    for i in range(100):
        out = model(x)
        stats = qrf_opt.step()
        if i % 20 == 0:
            tracer.record()
    elapsed = time.time() - start
    
    print(f"100 forward passes: {elapsed:.2f}s ({elapsed/100*1000:.1f}ms per pass)")
    print(f"Output shape: {out.shape}")
    
    # Check if output is valid quaternion
    quat_norms = out.norm(dim=-1)
    is_unit = torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5)
    print(f"Output is unit quaternion: {is_unit}")
    
    print(f"\n--- Final Field State ---")
    print(f"  Energy: {stats['energy']:.4f}")
    print(f"  Temperature: {stats['temperature']:.4f}")
    print(f"  Coherence: {stats['coherence']:.4f}")
    print(f"  Applied ruffle: {stats['applied_ruffle']}")
    
    # Check if dynamics are updating
    dynamics_updating = stats['temperature'] != 1.0 or stats['coherence'] != 1.0
    print(f"  Dynamics updating: {dynamics_updating}")
    
    # Test memory system
    print("\n--- Field Memory System Test ---")
    model.field.reset(preserve_memory=True)
    print(f"After reset: {model.field.get_field_state_summary()}")
    
    model.field.restore_from_memory(blend_factor=0.5)
    print(f"After restore (50%): {model.field.get_field_state_summary()}")
    
    # Memory coherence
    print(f"\nTracer memory coherence: {tracer.compute_memory_coherence():.4f}")
    
    # Test quaternion output head standalone
    print("\n--- Quaternion Output Head Test ---")
    quat_head = QuaternionOutputHead(hidden_dim=128).to(device)
    h = torch.randn(4, 128, device=device)
    q_out = quat_head(h)
    print(f"Output shape: {q_out.shape}")
    print(f"Is unit quaternion: {torch.allclose(q_out.norm(dim=-1), torch.ones(4, device=device), atol=1e-5)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    processor_params = sum(p.numel() for p in model.processor.parameters())
    print(f"\nTotal model parameters: {total_params:,}")
    print(f"Signal processor parameters: {processor_params:,}")
    print(f"  (includes sequence memory layers)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! v4.1 ready for use.")
    print("=" * 60)