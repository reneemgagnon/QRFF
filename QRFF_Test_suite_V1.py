"""
Quaternion Ruffle Field Benchmark Suite
=======================================
Comprehensive benchmarks comparing QRF against standard neural architectures.

Benchmarks:
    1. Parameter Efficiency - QRF vs Dense vs Transformer
    2. Rotation Equivariance - 3D rotation prediction tasks
    3. Sequence Modeling - Memory and temporal patterns
    4. Training Dynamics - Convergence and stability
    5. Computational Cost - FLOPs and memory usage

Author: [Your Name]
Date: April 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

# Import QRF (assumes quaternion_ruffle_field_v2.py is in same directory)
try:
    from V2_Quantum_Ruffle_Field import (
        QuaternionRuffleField,
        QuaternionSignalProcessor,
        QuaternionRuffleOptimizer,
        QuaternionMemoryTracer,
        get_device,
        quaternion_normalize,
        quaternion_multiply,
        quaternion_to_rotation_matrix
    )
    QRF_AVAILABLE = True
except ImportError:
    QRF_AVAILABLE = False
    warnings.warn("QRF module not found. Run from same directory as quaternion_ruffle_field_v2.py")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    # Data parameters
    batch_size: int = 32
    seq_length: int = 64
    hidden_dim: int = 128
    n_classes: int = 10
    
    # Training parameters
    n_epochs: int = 50
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    
    # Model parameters
    n_neurons: int = 64  # For QRF
    n_heads: int = 4     # For Transformer
    n_layers: int = 2    # For Transformer
    
    # Benchmark parameters
    n_trials: int = 3
    seed: int = 42
    
    # Device
    device: str = "auto"


# =============================================================================
# BASELINE MODELS
# =============================================================================

class DenseBaseline(nn.Module):
    """Simple dense (MLP) baseline."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 2):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        batch, seq, feat = x.shape
        x_flat = x.view(batch * seq, feat)
        out = self.network(x_flat)
        return out.view(batch, seq, -1)


class TransformerBaseline(nn.Module):
    """Standard Transformer encoder baseline."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.norm(x)
        return self.output_proj(x)


class LSTMBaseline(nn.Module):
    """LSTM baseline for sequence modeling."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.norm(out)
        return self.output_proj(out)


class QRFModel(nn.Module):
    """Quaternion Ruffle Field model wrapper."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_neurons: int = 64,
        use_attention: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.device = device or get_device()
        
        # Create quaternion field
        self.field = QuaternionRuffleField(
            n_neurons=n_neurons,
            device=self.device
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # QRF processor
        self.qrf_processor = QuaternionSignalProcessor(
            self.field,
            hidden_dim=hidden_dim,
            use_attention=use_attention
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.qrf_processor(x)
        x = self.norm(x)
        return self.output_proj(x)


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def generate_rotation_data(
    n_samples: int,
    seq_length: int,
    noise_scale: float = 0.1,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic 3D rotation prediction data.
    
    Task: Given a sequence of 3D points, predict the rotation applied.
    This tests rotation equivariance - QRF should excel here.
    """
    device = device or get_device()
    
    # Generate random 3D point clouds
    points = torch.randn(n_samples, seq_length, 3, device=device)
    
    # Generate random rotation quaternions (targets)
    u = torch.rand(n_samples, 3, device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    
    sqrt_u1 = torch.sqrt(u1)
    sqrt_1_u1 = torch.sqrt(1.0 - u1)
    
    target_quats = torch.stack([
        sqrt_1_u1 * torch.sin(2 * np.pi * u2),
        sqrt_1_u1 * torch.cos(2 * np.pi * u2),
        sqrt_u1 * torch.sin(2 * np.pi * u3),
        sqrt_u1 * torch.cos(2 * np.pi * u3)
    ], dim=-1)
    
    # Apply rotations to points
    rot_matrices = quaternion_to_rotation_matrix(target_quats)  # (n_samples, 3, 3)
    rotated_points = torch.bmm(points.view(n_samples * seq_length, 1, 3).expand(-1, 3, -1).transpose(1, 2),
                                rot_matrices.unsqueeze(1).expand(-1, seq_length, -1, -1).reshape(-1, 3, 3))
    rotated_points = rotated_points[:, :, 0].view(n_samples, seq_length, 3)
    
    # Add noise
    rotated_points = rotated_points + torch.randn_like(rotated_points) * noise_scale
    
    # Input: concatenate original and rotated points
    inputs = torch.cat([points, rotated_points], dim=-1)  # (n_samples, seq_length, 6)
    
    return inputs, target_quats


def generate_memory_sequence_data(
    n_samples: int,
    seq_length: int,
    hidden_dim: int,
    memory_gap: int = 10,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for testing memory capabilities.
    
    Task: Remember a pattern from early in sequence, ignore noise,
    then reproduce/classify at the end.
    """
    device = device or get_device()
    
    # Create patterns at the start
    n_patterns = 4
    patterns = torch.randn(n_patterns, hidden_dim, device=device)
    
    # Assign pattern labels
    labels = torch.randint(0, n_patterns, (n_samples,), device=device)
    
    # Create sequences
    sequences = torch.randn(n_samples, seq_length, hidden_dim, device=device) * 0.1
    
    # Insert pattern at position 0
    for i in range(n_samples):
        sequences[i, 0] = patterns[labels[i]]
    
    # Insert recall cue at position (seq_length - memory_gap)
    recall_cue = torch.ones(hidden_dim, device=device)
    sequences[:, -memory_gap] = recall_cue
    
    return sequences, labels


def generate_classification_data(
    n_samples: int,
    seq_length: int,
    hidden_dim: int,
    n_classes: int = 10,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate standard sequence classification data.
    """
    device = device or get_device()
    
    # Create class prototypes
    prototypes = torch.randn(n_classes, hidden_dim, device=device)
    
    # Generate labels
    labels = torch.randint(0, n_classes, (n_samples,), device=device)
    
    # Generate sequences based on prototypes with noise
    sequences = torch.randn(n_samples, seq_length, hidden_dim, device=device) * 0.5
    
    for i in range(n_samples):
        # Add prototype signal throughout sequence
        sequences[i] = sequences[i] + prototypes[labels[i]].unsqueeze(0) * 0.3
    
    return sequences, labels


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    task_name: str
    accuracy: float
    loss: float
    train_time: float
    inference_time: float
    n_parameters: int
    memory_mb: float
    convergence_epoch: int
    final_metrics: Dict


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory(model: nn.Module, input_shape: Tuple, device: torch.device) -> float:
    """Measure GPU memory usage in MB."""
    if device.type != 'cuda':
        return 0.0
    
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(*input_shape, device=device)
    _ = model(x)
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_regression: bool = False
) -> Tuple[float, float]:
    """Train for one epoch, return loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if is_regression:
            # For quaternion prediction, use last timestep
            outputs = outputs[:, -1, :4]  # Get quaternion prediction
            outputs = quaternion_normalize(outputs)
            # Quaternion distance loss
            dot = (outputs * targets).sum(dim=-1)
            loss = (1 - dot.abs()).mean()
        else:
            # Classification: use mean pooling over sequence
            outputs = outputs.mean(dim=1)
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        if not is_regression:
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        else:
            # For regression, compute angular error
            dot = (outputs * targets).sum(dim=-1)
            correct += (dot.abs() > 0.95).sum().item()  # ~18 degree threshold
        
        total += inputs.size(0)
    
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_regression: bool = False
) -> Tuple[float, float]:
    """Evaluate model, return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if is_regression:
                outputs = outputs[:, -1, :4]
                outputs = quaternion_normalize(outputs)
                dot = (outputs * targets).sum(dim=-1)
                loss = (1 - dot.abs()).mean()
                correct += (dot.abs() > 0.95).sum().item()
            else:
                outputs = outputs.mean(dim=1)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            
            total_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
    
    return total_loss / total, correct / total


def run_benchmark(
    model: nn.Module,
    model_name: str,
    task_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: BenchmarkConfig,
    device: torch.device,
    is_regression: bool = False
) -> BenchmarkResult:
    """Run complete benchmark for a model on a task."""
    
    model = model.to(device)
    
    # Count parameters
    n_params = count_parameters(model)
    
    # Measure memory
    sample_input = next(iter(train_loader))[0]
    memory_mb = measure_memory(model, sample_input.shape, device)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    train_accs = []
    best_acc = 0.0
    convergence_epoch = config.n_epochs
    
    start_time = time.time()
    
    for epoch in range(config.n_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, is_regression
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        if train_acc > best_acc + 0.01:  # 1% improvement threshold
            best_acc = train_acc
            convergence_epoch = epoch + 1
    
    train_time = time.time() - start_time
    
    # Evaluation
    start_time = time.time()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, is_regression)
    inference_time = time.time() - start_time
    
    return BenchmarkResult(
        model_name=model_name,
        task_name=task_name,
        accuracy=test_acc,
        loss=test_loss,
        train_time=train_time,
        inference_time=inference_time,
        n_parameters=n_params,
        memory_mb=memory_mb,
        convergence_epoch=convergence_epoch,
        final_metrics={
            'train_losses': train_losses,
            'train_accs': train_accs
        }
    )


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

class BenchmarkSuite:
    """Complete benchmark suite for comparing models."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        
        if self.config.device == "auto":
            self.device = get_device() if QRF_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        
        self.results: List[BenchmarkResult] = []
        
        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
    
    def create_models(self, input_dim: int, output_dim: int) -> Dict[str, nn.Module]:
        """Create all models to benchmark."""
        models = {
            'Dense': DenseBaseline(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=output_dim,
                n_layers=self.config.n_layers
            ),
            'Transformer': TransformerBaseline(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=output_dim,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers
            ),
            'LSTM': LSTMBaseline(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=output_dim,
                n_layers=self.config.n_layers
            ),
        }
        
        if QRF_AVAILABLE:
            models['QRF'] = QRFModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=output_dim,
                n_neurons=self.config.n_neurons,
                device=self.device
            )
            models['QRF_NoAttn'] = QRFModel(
                input_dim=input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=output_dim,
                n_neurons=self.config.n_neurons,
                use_attention=False,
                device=self.device
            )
        
        return models
    
    def run_rotation_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark on rotation prediction task."""
        print("\n" + "="*60)
        print("BENCHMARK: Rotation Prediction (3D Point Cloud)")
        print("="*60)
        
        # Generate data
        n_train, n_test = 2000, 500
        train_inputs, train_targets = generate_rotation_data(
            n_train, self.config.seq_length, device=self.device
        )
        test_inputs, test_targets = generate_rotation_data(
            n_test, self.config.seq_length, device=self.device
        )
        
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_targets),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_inputs, test_targets),
            batch_size=self.config.batch_size
        )
        
        # Create models (output = 4 for quaternion)
        models = self.create_models(input_dim=6, output_dim=4)
        
        results = []
        for name, model in models.items():
            print(f"\nTraining {name}...")
            result = run_benchmark(
                model, name, "Rotation",
                train_loader, test_loader,
                self.config, self.device,
                is_regression=True
            )
            results.append(result)
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  Parameters: {result.n_parameters:,}")
            print(f"  Train time: {result.train_time:.2f}s")
        
        self.results.extend(results)
        return results
    
    def run_memory_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark on long-term memory task."""
        print("\n" + "="*60)
        print("BENCHMARK: Long-Term Memory (Pattern Recall)")
        print("="*60)
        
        n_train, n_test = 2000, 500
        train_inputs, train_targets = generate_memory_sequence_data(
            n_train, self.config.seq_length, self.config.hidden_dim,
            memory_gap=20, device=self.device
        )
        test_inputs, test_targets = generate_memory_sequence_data(
            n_test, self.config.seq_length, self.config.hidden_dim,
            memory_gap=20, device=self.device
        )
        
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_targets),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_inputs, test_targets),
            batch_size=self.config.batch_size
        )
        
        models = self.create_models(
            input_dim=self.config.hidden_dim,
            output_dim=4  # 4 pattern classes
        )
        
        results = []
        for name, model in models.items():
            print(f"\nTraining {name}...")
            result = run_benchmark(
                model, name, "Memory",
                train_loader, test_loader,
                self.config, self.device
            )
            results.append(result)
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  Parameters: {result.n_parameters:,}")
        
        self.results.extend(results)
        return results
    
    def run_classification_benchmark(self) -> List[BenchmarkResult]:
        """Benchmark on sequence classification task."""
        print("\n" + "="*60)
        print("BENCHMARK: Sequence Classification")
        print("="*60)
        
        n_train, n_test = 3000, 500
        train_inputs, train_targets = generate_classification_data(
            n_train, self.config.seq_length, self.config.hidden_dim,
            n_classes=self.config.n_classes, device=self.device
        )
        test_inputs, test_targets = generate_classification_data(
            n_test, self.config.seq_length, self.config.hidden_dim,
            n_classes=self.config.n_classes, device=self.device
        )
        
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_targets),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(test_inputs, test_targets),
            batch_size=self.config.batch_size
        )
        
        models = self.create_models(
            input_dim=self.config.hidden_dim,
            output_dim=self.config.n_classes
        )
        
        results = []
        for name, model in models.items():
            print(f"\nTraining {name}...")
            result = run_benchmark(
                model, name, "Classification",
                train_loader, test_loader,
                self.config, self.device
            )
            results.append(result)
            print(f"  Accuracy: {result.accuracy:.4f}")
            print(f"  Parameters: {result.n_parameters:,}")
        
        self.results.extend(results)
        return results
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.run_rotation_benchmark()
        self.run_memory_benchmark()
        self.run_classification_benchmark()
        return self.results
    
    def print_summary(self) -> None:
        """Print formatted summary of all results."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Group by task
        tasks = set(r.task_name for r in self.results)
        
        for task in tasks:
            print(f"\n{task} Task:")
            print("-" * 70)
            print(f"{'Model':<15} {'Accuracy':<12} {'Params':<12} {'Train(s)':<12} {'Memory(MB)':<12}")
            print("-" * 70)
            
            task_results = [r for r in self.results if r.task_name == task]
            task_results.sort(key=lambda x: x.accuracy, reverse=True)
            
            for r in task_results:
                print(f"{r.model_name:<15} {r.accuracy:<12.4f} {r.n_parameters:<12,} {r.train_time:<12.2f} {r.memory_mb:<12.2f}")
        
        # Overall winner analysis
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        
        if QRF_AVAILABLE:
            qrf_results = [r for r in self.results if r.model_name == 'QRF']
            other_results = [r for r in self.results if r.model_name != 'QRF' and 'QRF' not in r.model_name]
            
            for task in tasks:
                qrf_task = [r for r in qrf_results if r.task_name == task]
                best_other = max([r for r in other_results if r.task_name == task], key=lambda x: x.accuracy)
                
                if qrf_task:
                    qrf = qrf_task[0]
                    diff = qrf.accuracy - best_other.accuracy
                    print(f"\n{task}: QRF vs {best_other.model_name}")
                    print(f"  Accuracy difference: {diff:+.4f} ({'QRF better' if diff > 0 else 'Baseline better'})")
                    print(f"  Parameter ratio: {qrf.n_parameters / best_other.n_parameters:.2f}x")
    
    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        results_data = []
        for r in self.results:
            data = {
                'model_name': r.model_name,
                'task_name': r.task_name,
                'accuracy': r.accuracy,
                'loss': r.loss,
                'train_time': r.train_time,
                'inference_time': r.inference_time,
                'n_parameters': r.n_parameters,
                'memory_mb': r.memory_mb,
                'convergence_epoch': r.convergence_epoch
            }
            results_data.append(data)
        
        with open(filepath, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': results_data
            }, f, indent=2)
        
        print(f"\nResults saved to {filepath}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Run the complete benchmark suite."""
    print("="*80)
    print("QUATERNION RUFFLE FIELD BENCHMARK SUITE")
    print("="*80)
    
    if not QRF_AVAILABLE:
        print("\nWARNING: QRF module not available. Running baselines only.")
        print("Place this file in the same directory as quaternion_ruffle_field_v2.py")
    
    # Configuration
    config = BenchmarkConfig(
        batch_size=32,
        seq_length=64,
        hidden_dim=128,
        n_epochs=30,
        learning_rate=1e-3,
        n_neurons=64,
        n_heads=4,
        n_layers=2,
        n_trials=1
    )
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Sequence length: {config.seq_length}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Epochs: {config.n_epochs}")
    
    # Run benchmarks
    suite = BenchmarkSuite(config)
    suite.run_all_benchmarks()
    
    # Print summary
    suite.print_summary()
    
    # Save results
    suite.save_results("benchmark_results.json")
    
    return suite


if __name__ == "__main__":
    suite = main()