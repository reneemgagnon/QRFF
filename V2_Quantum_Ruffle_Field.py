"""
Quaternion Ruffle Field v2.0
============================
A novel neural architecture combining spatial coordinates with quaternion orientations
to create dynamic, evolving manifold-based neural processing.

Author: [Your Name]
Date: April 2025
License: MIT

Key Innovations:
    - Dual-state neurons (position + orientation)
    - Field thermodynamics with adaptive temperature/coherence
    - Memory-blended SLERP state transitions
    - Energy-based perturbation optimization
    - Quaternion-weighted attention mechanisms

Mathematical Foundation:
    - Quaternions: q = w + xi + yj + zk where i² = j² = k² = ijk = -1
    - Hamilton product for composition
    - SLERP for smooth interpolation on S³ manifold
    - Geodesic distance: d(q1, q2) = arccos(|q1 · q2|)
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
    """Configuration for Quaternion Ruffle Field."""
    n_neurons: int = 64
    space_dim: int = 4
    enable_memory: bool = True
    temperature_bounds: Tuple[float, float] = (0.1, 10.0)
    coherence_rate: float = 0.05
    energy_momentum: float = 0.95
    numerical_eps: float = 1e-8


# =============================================================================
# QUATERNION OPERATIONS (Optimized & Vectorized)
# =============================================================================

def get_device(tensor: Optional[torch.Tensor] = None) -> torch.device:
    """Get appropriate device, defaulting to CUDA if available."""
    if tensor is not None:
        return tensor.device
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    # q and -q represent the same rotation, so flip if needed
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
    w1 = torch.sin((1.0 - t) * theta) / sin_theta
    w2 = torch.sin(t * theta) / sin_theta
    
    # LERP weights (fallback)
    w1 = torch.where(use_lerp, torch.full_like(w1, 1.0 - t), w1)
    w2 = torch.where(use_lerp, torch.full_like(w2, t), w2)
    
    # Interpolate and normalize
    result = w1.unsqueeze(-1) * q1 + w2.unsqueeze(-1) * q2_adjusted
    return quaternion_normalize(result)


def quaternion_slerp_batch(
    q1: torch.Tensor,
    q2: torch.Tensor, 
    t: float,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Vectorized SLERP for batch processing (no loops).
    
    Args:
        q1: Batch of start quaternions (N, 4)
        q2: Batch of end quaternions (N, 4)
        t: Interpolation parameter
        eps: Numerical epsilon
        
    Returns:
        Interpolated quaternions (N, 4)
    """
    return quaternion_slerp(q1, q2, t, eps)


def quaternion_random_unit(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    Generate uniformly distributed random unit quaternions.
    
    Uses the method from K. Shoemake, "Uniform random rotations"
    
    Args:
        shape: Output shape (should end with 4)
        device: Torch device
        
    Returns:
        Random unit quaternions
    """
    # Generate random values
    u = torch.rand(shape[:-1] + (3,), device=device)
    u1, u2, u3 = u[..., 0], u[..., 1], u[..., 2]
    
    # Convert to quaternion (Shoemake method)
    sqrt_u1 = torch.sqrt(u1)
    sqrt_1_u1 = torch.sqrt(1.0 - u1)
    
    q = torch.stack([
        sqrt_1_u1 * torch.sin(2 * math.pi * u2),
        sqrt_1_u1 * torch.cos(2 * math.pi * u2),
        sqrt_u1 * torch.sin(2 * math.pi * u3),
        sqrt_u1 * torch.cos(2 * math.pi * u3)
    ], dim=-1)
    
    return q


def quaternion_random_perturbation(
    scale: float = 0.05,
    shape: Tuple[int, ...] = (4,),
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate small random perturbation quaternions near identity.
    
    Args:
        scale: Perturbation magnitude (0 = identity, 1 = random)
        shape: Output shape
        device: Torch device
        
    Returns:
        Perturbation quaternions biased toward identity
    """
    if device is None:
        device = get_device()
        
    # Start with random quaternion
    rand = torch.randn(shape, device=device)
    rand = quaternion_normalize(rand) * scale
    
    # Bias toward identity [1, 0, 0, 0]
    rand[..., 0] = rand[..., 0] + (1.0 - scale)
    
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
    Non-Newtonian manifold with quaternion neuron states.
    
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
        
        # Core field parameters - quaternion orientations (properly initialized)
        initial_quats = quaternion_random_unit((n_neurons, 4), self.device)
        self.quaternions = nn.Parameter(initial_quats)
        
        # Legacy phase support (backward compatibility)
        self.phases = nn.Parameter(
            torch.rand(n_neurons, device=self.device) * 2 * math.pi
        )
        
        # Memory state buffers
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
        
        # Dynamics tracking
        self.register_buffer('energy_buffer', torch.zeros(1, device=self.device))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long, device=self.device))
        self.register_buffer('field_temperature', torch.ones(1, device=self.device))
        self.register_buffer('coherence_factor', torch.ones(1, device=self.device))
    
    def reset(self, preserve_memory: bool = True) -> None:
        """
        Reset field state for new sequence processing.
        
        Args:
            preserve_memory: Whether to save current state to memory
        """
        if self.enable_memory and preserve_memory:
            self.memory_coordinates.copy_(self.coordinates.data)
            self.memory_quaternions.copy_(self.quaternions.data)
            self.memory_valid.fill_(True)
        
        with torch.no_grad():
            # Reset coordinates to small random values
            self.coordinates.data = torch.randn_like(self.coordinates) * 0.05
            
            # Reset quaternions to near-identity
            identity = torch.zeros_like(self.quaternions)
            identity[:, 0] = 1.0  # Real part = 1 (identity rotation)
            perturbation = torch.randn_like(self.quaternions) * 0.02
            self.quaternions.data = quaternion_normalize(identity + perturbation)
            
            # Reset phases
            self.phases.data = torch.rand_like(self.phases) * 2 * math.pi
            
            # Reset dynamics
            self.energy_buffer.zero_()
            self.field_temperature.fill_(1.0)
            self.coherence_factor.fill_(1.0)
    
    def restore_from_memory(self, blend_factor: float = 0.5) -> None:
        """
        Restore field state from memory with smooth blending.
        
        Uses SLERP for quaternion interpolation to ensure smooth
        transitions on the rotation manifold.
        
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
            
            # Vectorized SLERP for quaternions (no loop!)
            self.quaternions.data = quaternion_slerp_batch(
                self.quaternions.data,
                self.memory_quaternions,
                blend_factor
            )
    
    def compute_dynamic_distances(self) -> torch.Tensor:
        """
        Compute enhanced dynamic distances between neurons.
        
        Combines:
            - Temperature-scaled Euclidean distance in coordinate space
            - Geodesic distance on quaternion manifold
            - Coherence-weighted blending
        
        Returns:
            Distance matrix of shape (n_neurons, n_neurons)
        """
        coords = self.coordinates
        quats = quaternion_normalize(self.quaternions)
        
        # Euclidean distance with temperature scaling
        coord_diffs = coords.unsqueeze(0) - coords.unsqueeze(1)
        coord_distances = torch.norm(coord_diffs, dim=2)
        temp_scaled = coord_distances / (self.field_temperature + 1e-8)
        
        # Geodesic distance on quaternion manifold
        quat_distances = quaternion_geodesic_distance(
            quats.unsqueeze(0),
            quats.unsqueeze(1)
        )
        
        # Coherence-weighted combination
        alpha = self.coherence_factor
        return alpha * temp_scaled + (1.0 - alpha) * quat_distances
    
    def compute_folding_energy(self) -> torch.Tensor:
        """
        Compute total field energy (curvature + disorder + memory drift).
        
        Energy components:
            - Curvature: Inverse square distance (repulsion)
            - Rotational disorder: Variance in quaternion orientations
            - Memory drift: Distance from previous memory state
        
        Returns:
            Scalar energy tensor
        """
        coords = self.coordinates
        quats = quaternion_normalize(self.quaternions)
        
        # Spatial curvature energy
        pairwise_dists = torch.norm(
            coords.unsqueeze(0) - coords.unsqueeze(1), dim=2
        ) + 1e-6
        curvature_energy = torch.mean(
            1.0 / (pairwise_dists**2 + self.field_temperature)
        )
        
        # Quaternion disorder (rotational chaos)
        quat_inner = quaternion_inner_product(
            quats.unsqueeze(0),
            quats.unsqueeze(1)
        )
        rotation_disorder = torch.mean(1.0 - quat_inner**2)
        
        # Memory coherence penalty
        memory_penalty = torch.tensor(0.0, device=self.device)
        if self.enable_memory and self.memory_valid.item():
            coord_drift = torch.norm(coords - self.memory_coordinates)
            quat_drift = torch.mean(
                1.0 - quaternion_inner_product(quats, self.memory_quaternions)**2
            )
            memory_penalty = 0.1 * (coord_drift + quat_drift)
        
        # Total energy
        total_energy = curvature_energy + 0.5 * rotation_disorder + memory_penalty
        
        # Update energy buffer with momentum
        momentum = self.config.energy_momentum
        self.energy_buffer = (
            momentum * self.energy_buffer + 
            (1 - momentum) * total_energy.detach()
        )
        
        return total_energy
    
    def get_phase_representation(self) -> torch.Tensor:
        """
        Extract phase representation from quaternions.
        
        Uses quaternion logarithm to extract rotation angle and axis,
        then combines into a phase-like scalar representation.
        
        Returns:
            Phase tensor of shape (n_neurons,)
        """
        quats = quaternion_normalize(self.quaternions)
        w, x, y, z = quats.unbind(-1)
        
        # Rotation angle from quaternion
        theta = 2 * torch.atan2(
            torch.sqrt(x**2 + y**2 + z**2),
            torch.abs(w)
        )
        
        # Combine angle with axis direction (avoid atan2(0,0))
        phase = theta + torch.atan2(y, x + 1e-8)
        
        return phase
    
    def update_field_dynamics(self, input_energy: Optional[torch.Tensor] = None) -> None:
        """
        Update field temperature and coherence based on energy state.
        
        Args:
            input_energy: Optional external energy input
        """
        self.step_count += 1
        current_energy = self.compute_folding_energy()
        
        # Adaptive temperature
        if input_energy is not None:
            energy_gradient = torch.abs(current_energy - input_energy)
            temp_update = 0.01 * energy_gradient
            self.field_temperature = torch.clamp(
                self.field_temperature + temp_update,
                self.config.temperature_bounds[0],
                self.config.temperature_bounds[1]
            )
        
        # Coherence evolution
        energy_stability = torch.exp(-current_energy)
        coherence_target = 0.5 + 0.5 * energy_stability
        self.coherence_factor = (
            (1 - self.config.coherence_rate) * self.coherence_factor +
            self.config.coherence_rate * coherence_target
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
# QUATERNION RUFFLE OPTIMIZER
# =============================================================================

class QuaternionRuffleOptimizer:
    """
    Optimizer for quaternion ruffle fields with adaptive perturbations.
    
    Applies "ruffles" (perturbations) when field energy drops below threshold,
    helping escape local minima on the quaternion manifold.
    """
    
    def __init__(
        self,
        field: QuaternionRuffleField,
        fold_threshold: float = 1.2,
        ruffle_scale: float = 0.04
    ):
        """
        Initialize the optimizer.
        
        Args:
            field: The quaternion field to optimize
            fold_threshold: Energy threshold for applying perturbations
            ruffle_scale: Base scale of perturbations
        """
        self.field = field
        self.fold_threshold = fold_threshold
        self.ruffle_scale = ruffle_scale
        self.steps = 0
        self.energy_history: List[float] = []
    
    def step(self) -> None:
        """Perform one optimization step with adaptive perturbations."""
        energy = self.field.compute_folding_energy()
        self.steps += 1
        self.energy_history.append(energy.item())
        
        # Maintain bounded history
        if len(self.energy_history) > 100:
            self.energy_history = self.energy_history[-100:]
        
        # Adaptive threshold based on energy trends
        if len(self.energy_history) > 10:
            recent = sum(self.energy_history[-10:]) / 10
            older = sum(self.energy_history[-20:-10]) / max(1, min(10, len(self.energy_history) - 10))
            energy_trend = recent - older
            adaptive_threshold = self.fold_threshold * (1.0 + 0.1 * energy_trend)
        else:
            adaptive_threshold = self.fold_threshold
        
        # Apply perturbations when energy is low
        if energy < adaptive_threshold:
            self._apply_perturbations()
    
    def _apply_perturbations(self) -> None:
        """Apply coordinate and quaternion perturbations."""
        # Coordinate perturbations
        coord_scale = (
            self.ruffle_scale * 
            self.field.field_temperature.item() /
            (1.0 + 0.01 * self.steps)
        )
        perturb_coords = torch.randn_like(self.field.coordinates) * coord_scale
        self.field.coordinates.data += perturb_coords
        
        # Quaternion perturbations (vectorized)
        quat_scale = (
            self.ruffle_scale * 
            (2.0 - self.field.coherence_factor.item()) /
            (1.0 + 0.005 * self.steps)
        )
        delta_quats = quaternion_random_perturbation(
            scale=quat_scale,
            shape=(self.field.n_neurons, 4),
            device=self.field.device
        )
        
        new_quats = quaternion_multiply(self.field.quaternions.data, delta_quats)
        self.field.quaternions.data = quaternion_normalize(new_quats)
        
        # Update phases for compatibility
        self.field.phases.data = self.field.get_phase_representation()
        
        # Trigger dynamics update
        self.field.update_field_dynamics(input_energy=self.field.energy_buffer)
    
    def get_optimizer_state(self) -> Dict[str, float]:
        """Get current optimizer state information."""
        return {
            'steps': self.steps,
            'fold_threshold': self.fold_threshold,
            'ruffle_scale': self.ruffle_scale,
            'energy_history_length': len(self.energy_history),
            'recent_energy_mean': (
                sum(self.energy_history[-10:]) / min(10, len(self.energy_history))
                if self.energy_history else 0.0
            )
        }


# =============================================================================
# QUATERNION SIGNAL PROCESSOR (Neural Network Layer)
# =============================================================================

class QuaternionSignalProcessor(nn.Module):
    """
    Signal processor with adaptive quaternion modulation and attention.
    
    Processes input signals through the quaternion field, applying:
        - Quaternion-based amplitude and phase modulation
        - Multi-head attention on quaternion representations
        - Field-aware weighting and pooling
    """
    
    def __init__(
        self,
        field: QuaternionRuffleField,
        hidden_dim: int = 64,
        use_attention: bool = True,
        attention_heads: int = 4,
        head_dim: int = 16
    ):
        """
        Initialize the signal processor.
        
        Args:
            field: Quaternion ruffle field for processing
            hidden_dim: Hidden representation dimensionality
            use_attention: Enable multi-head attention
            attention_heads: Number of attention heads
            head_dim: Dimension per attention head
        """
        super().__init__()
        
        self.field = field
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.head_dim = head_dim
        
        # Projections
        self.input_projection = nn.Linear(hidden_dim, field.n_neurons)
        self.output_projection = nn.Linear(field.n_neurons, hidden_dim)
        
        # Normalization layers
        self.input_norm = nn.LayerNorm(field.n_neurons)
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        # Attention layers
        if use_attention:
            attn_dim = attention_heads * head_dim
            self.attention_key = nn.Linear(4, attn_dim)
            self.attention_query = nn.Linear(4, attn_dim)
            self.attention_value = nn.Linear(4, attn_dim)
            self.attention_output = nn.Linear(attn_dim, 4)
            self.attention_scale = math.sqrt(head_dim)
    
    def quaternion_modulation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quaternion-based signal modulation.
        
        Args:
            x: Input tensor (batch, n_neurons)
            
        Returns:
            Modulated tensor (batch, n_neurons)
        """
        quats = quaternion_normalize(self.field.quaternions)
        w, qx, qy, qz = quats.unbind(-1)
        
        temp = self.field.field_temperature.item()
        coherence = self.field.coherence_factor.item()
        
        # Amplitude modulation from real part
        amplitude = w.unsqueeze(0) * temp
        
        # Phase modulation from imaginary parts
        phase = (
            torch.cos(qx.unsqueeze(0) * coherence) *
            torch.sin(qy.unsqueeze(0) * coherence) *
            torch.cos(qz.unsqueeze(0) * coherence)
        )
        
        return x * (amplitude + 0.3 * phase)
    
    def apply_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention using quaternion representations.
        
        Args:
            x: Input tensor (batch, n_neurons)
            
        Returns:
            Attention-weighted tensor (batch, n_neurons)
        """
        batch_size = x.shape[0]
        quats = quaternion_normalize(self.field.quaternions)
        quats_expanded = quats.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Compute Q, K, V
        keys = self.attention_key(quats_expanded)
        queries = self.attention_query(quats_expanded)
        values = self.attention_value(quats_expanded)
        
        # Reshape for multi-head attention
        n = self.field.n_neurons
        keys = keys.view(batch_size, n, self.attention_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, n, self.attention_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, n, self.attention_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores with coherence bias
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.attention_scale
        scores = scores + self.field.coherence_factor.item() * 0.1
        weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(weights, values)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, n, self.attention_heads * self.head_dim)
        attended = self.attention_output(attended)
        
        # Modulate input with attended values
        mod_weights = F.softmax(attended[:, :, 0], dim=-1)
        return x * (1.0 + mod_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the signal processor.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            
        Returns:
            Processed tensor (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Flatten for processing
        x_flat = x.view(-1, self.hidden_dim)
        
        # Project to neuron space
        projected = self.input_projection(x_flat)
        projected = self.input_norm(projected)
        
        # Apply quaternion modulation
        modulated = self.quaternion_modulation(projected)
        
        # Apply attention if enabled
        if self.use_attention:
            modulated = self.apply_attention(modulated)
        
        # Quaternion-weighted pooling
        quat_weights = F.softmax(self.field.quaternions[:, 0], dim=0)
        pooled = (modulated * quat_weights.unsqueeze(0)).sum(dim=1)
        
        # Output projection
        output = self.output_projection(pooled.unsqueeze(1))
        output = self.output_norm(output.squeeze(1))
        
        return output.view(batch_size, seq_len, self.hidden_dim)


# =============================================================================
# QUATERNION MEMORY TRACER (Analysis Tool)
# =============================================================================

class QuaternionMemoryTracer:
    """
    Trace and analyze quaternion field state evolution over time.
    
    Records snapshots of field state for post-hoc analysis of:
        - Coordinate trajectories
        - Quaternion rotation paths
        - Energy landscape evolution
        - Memory coherence metrics
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
        Export full history for analysis.
        
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
        Compute coherence between consecutive memory states.
        
        Returns:
            Coherence score in [0, 1] where 1 = perfectly stable
        """
        if len(self.history_coords) < 2:
            return 0.0
        
        coord_diffs = []
        quat_diffs = []
        
        for i in range(1, len(self.history_coords)):
            # Coordinate change
            coord_diff = torch.norm(
                self.history_coords[i] - self.history_coords[i-1],
                dim=1
            ).mean().item()
            coord_diffs.append(coord_diff)
            
            # Quaternion change (geodesic)
            quat_inner = quaternion_inner_product(
                self.history_quats[i],
                self.history_quats[i-1]
            )
            quat_diff = (1.0 - quat_inner**2).mean().item()
            quat_diffs.append(quat_diff)
        
        # Compute stability scores
        coord_stability = 1.0 / (1.0 + sum(coord_diffs) / len(coord_diffs))
        quat_stability = 1.0 / (1.0 + sum(quat_diffs) / len(quat_diffs))
        
        return (coord_stability + quat_stability) / 2.0
    
    def clear(self) -> None:
        """Clear all recorded history."""
        self.history_coords.clear()
        self.history_quats.clear()
        self.history_energy.clear()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_quaternion_layer(
    input_dim: int,
    output_dim: int,
    n_neurons: int = 32,
    use_attention: bool = True,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Factory function to create a quaternion processing layer.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        n_neurons: Number of neurons in quaternion field
        use_attention: Enable attention mechanism
        device: Torch device
        
    Returns:
        Sequential module with quaternion processing
    """
    device = device or get_device()
    field = QuaternionRuffleField(n_neurons, device=device)
    processor = QuaternionSignalProcessor(
        field, 
        hidden_dim=input_dim,
        use_attention=use_attention
    )
    
    return nn.Sequential(
        processor,
        nn.Linear(input_dim, output_dim)
    ).to(device)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Configuration
    device = get_device()
    print(f"Using device: {device}")
    
    # Create field
    field = QuaternionRuffleField(n_neurons=64, device=device)
    print(f"Field created with {field.n_neurons} neurons")
    
    # Create optimizer
    optimizer = QuaternionRuffleOptimizer(field)
    
    # Create signal processor
    processor = QuaternionSignalProcessor(field, hidden_dim=128)
    processor = processor.to(device)
    
    # Create tracer
    tracer = QuaternionMemoryTracer(field)
    
    # Example forward pass
    batch_size, seq_len, hidden_dim = 4, 16, 128
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    # Process
    tracer.record()
    output = processor(x)
    optimizer.step()
    tracer.record()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Field state: {field.get_field_state_summary()}")
    print(f"Memory coherence: {tracer.compute_memory_coherence():.4f}")