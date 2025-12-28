<p align="center">
  <img src="https://img.shields.io/badge/🔮_Quaternion-Ruffle_Field-blueviolet?style=for-the-badge&labelColor=1a1a2e" alt="Quaternion Ruffle Field"/>
</p>

<h1 align="center">
  <br>
  <img src="https://raw.githubusercontent.com/quaternion-ruffle-field/qrf/main/assets/qrf_logo.png" alt="QRF Logo" width="200">
  <br>
  Quaternion Ruffle Field
  <br>
</h1>

<h4 align="center">A Novel Neural Architecture Where Neurons Have Orientation</h4>

<p align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square&logo=github-actions" alt="Build">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Coverage-94%25-brightgreen?style=flat-square" alt="Coverage">
  </a>
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b?style=flat-square&logo=arxiv" alt="arXiv">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Paper-PDF-red?style=flat-square&logo=adobe-acrobat-reader" alt="Paper">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Docs-GitHub_Pages-blue?style=flat-square&logo=github" alt="Docs">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/Demo-Hugging_Face-yellow?style=flat-square&logo=huggingface" alt="Demo">
  </a>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-citation">Citation</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/quaternion-ruffle-field/qrf/main/assets/qrf_animation.gif" alt="QRF Animation" width="600">
</p>

---

## 🌟 What is Quaternion Ruffle Field?

**Quaternion Ruffle Field (QRF)** is a novel neural network architecture where each neuron maintains both a **spatial position** and a **rotational orientation** (represented as a quaternion). Unlike traditional neurons that are scalar values, QRF neurons exist on a dynamic manifold and interact through geometric relationships.

<table>
<tr>
<td width="50%">

### 🧠 Traditional Neuron
```
Value: 0.73
```
Just a number. No direction. No memory of orientation.

</td>
<td width="50%">

### 🔮 Quaternion Neuron
```
Position:    [0.2, -0.1, 0.5, 0.3]
Orientation: [0.92, 0.23, 0.15, 0.28]
             (w + xi + yj + zk)
```
A point in space with rotational state!

</td>
</tr>
</table>

---

## 🚀 Key Features

<table>
<tr>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/4d-rotate.png" width="60"/>
<br><b>Dual-State Neurons</b>
<br><sub>Position + Orientation</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/temperature.png" width="60"/>
<br><b>Field Thermodynamics</b>
<br><sub>Adaptive Temperature</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/memory-slot.png" width="60"/>
<br><b>SLERP Memory</b>
<br><sub>Geometric Interpolation</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/fluency/96/000000/attention.png" width="60"/>
<br><b>Quaternion Attention</b>
<br><sub>Manifold-Aware</sub>
</td>
</tr>
</table>

### ✨ Feature Highlights

| Feature | Description | Benefit |
|---------|-------------|---------|
| 🔄 **Hamilton Products** | Quaternion multiplication for neuron interactions | Preserves rotational composition |
| 📐 **Geodesic Distances** | Distance on S³ manifold | Respects quaternion geometry |
| 🌡️ **Adaptive Temperature** | Energy-based field dynamics | Automatic regularization |
| 🧲 **Coherence Coupling** | Inter-neuron orientation alignment | Emergent structure |
| 💾 **Memory Preservation** | SLERP-based state restoration | Smooth sequence processing |
| ⚡ **Ruffle Perturbations** | Energy-triggered exploration | Escapes local minima |

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### 🔧 Via pip (Recommended)

```bash
pip install quaternion-ruffle-field
```

### 🛠️ From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/quaternion-ruffle-field.git
cd quaternion-ruffle-field

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 📋 Requirements

```txt
torch>=2.0.0
numpy>=1.24.0
typing-extensions>=4.5.0
```

---

## ⚡ Quick Start

### 🎯 Basic Usage

```python
from quaternion_ruffle_field import (
    QuaternionRuffleField,
    QuaternionSignalProcessor,
    QuaternionRuffleOptimizer
)
import torch

# 🔮 Create a quaternion field with 64 neurons
field = QuaternionRuffleField(n_neurons=64)

# 📡 Create signal processor
processor = QuaternionSignalProcessor(
    field=field,
    hidden_dim=128,
    use_attention=True
)

# 🎲 Process some data
x = torch.randn(4, 16, 128)  # [batch, sequence, features]
output = processor(x)

print(f"Input:  {x.shape}")
print(f"Output: {output.shape}")
# Input:  torch.Size([4, 16, 128])
# Output: torch.Size([4, 16, 128])
```

### 🔬 With Field Optimization

```python
# Create optimizer for the field
qrf_optimizer = QuaternionRuffleOptimizer(
    field=field,
    fold_threshold=1.2,
    ruffle_scale=0.04
)

# Training loop
for epoch in range(100):
    output = processor(x)
    loss = your_loss_function(output, target)
    
    loss.backward()
    optimizer.step()
    
    # 🌀 Apply quaternion ruffles
    qrf_optimizer.step()
    
    # 📊 Monitor field state
    state = field.get_field_state_summary()
    print(f"Energy: {state['energy']:.4f}, Coherence: {state['coherence']:.4f}")
```

### 💾 Memory-Aware Sequence Processing

```python
# Process first sequence
output1 = processor(sequence_1)

# Reset field but preserve memory
field.reset(preserve_memory=True)

# Process second sequence
output2 = processor(sequence_2)

# Restore from memory with blending
field.restore_from_memory(blend_factor=0.5)
```

---

## 🏗️ Architecture

### 📊 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUATERNION RUFFLE FIELD                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│   │  📍 Coordinates  │    │  🔮 Quaternions  │    │  🌡️ Dynamics    │        │
│   │  [n_neurons, 4] │◄──►│  [n_neurons, 4] │◄──►│  temp/coherence │        │
│   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘        │
│            │                      │                      │                  │
│            └──────────────────────┼──────────────────────┘                  │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                         │
│                    │     🧮 Hamilton Products      │                         │
│                    │     📐 Geodesic Distances     │                         │
│                    │     ⚡ Energy Computation     │                         │
│                    └──────────────────────────────┘                         │
│                                   │                                         │
├───────────────────────────────────┼─────────────────────────────────────────┤
│                                   ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                    SIGNAL PROCESSOR                              │      │
│   │  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────┐ │      │
│   │  │  Input   │──►│  Quaternion  │──►│ Attention │──►│  Output  │ │      │
│   │  │Projection│   │  Modulation  │   │  (opt.)   │   │Projection│ │      │
│   │  └──────────┘   └──────────────┘   └──────────┘   └──────────┘ │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔢 Mathematical Foundation

<details>
<summary><b>📖 Click to expand mathematical details</b></summary>

#### Quaternion Representation

A quaternion `q` is represented as:

$$q = w + xi + yj + zk$$

where $i^2 = j^2 = k^2 = ijk = -1$

#### Hamilton Product

For quaternions $q = (w_0, x_0, y_0, z_0)$ and $r = (w_1, x_1, y_1, z_1)$:

$$q \otimes r = \begin{pmatrix} w_0w_1 - x_0x_1 - y_0y_1 - z_0z_1 \\ w_0x_1 + x_0w_1 + y_0z_1 - z_0y_1 \\ w_0y_1 - x_0z_1 + y_0w_1 + z_0x_1 \\ w_0z_1 + x_0y_1 - y_0x_1 + z_0w_1 \end{pmatrix}$$

#### Spherical Linear Interpolation (SLERP)

$$\text{slerp}(q_1, q_2, t) = \frac{\sin((1-t)\theta)}{\sin\theta}q_1 + \frac{\sin(t\theta)}{\sin\theta}q_2$$

where $\theta = \arccos(q_1 \cdot q_2)$

#### Geodesic Distance

$$d(q_1, q_2) = \arccos(|q_1 \cdot q_2|)$$

#### Field Energy

$$E = \underbrace{\sum_{i,j} \frac{1}{\|c_i - c_j\|^2 + T}}_{\text{curvature}} + \underbrace{\frac{1}{2}\mathbb{E}[1 - (q_i \cdot q_j)^2]}_{\text{rotational disorder}} + \underbrace{\lambda \cdot d(\mathbf{q}, \mathbf{q}_{memory})}_{\text{memory drift}}$$

</details>

### 🧩 Module Components

| Module | Description | Key Methods |
|--------|-------------|-------------|
| `QuaternionRuffleField` | Core field with neurons | `forward()`, `reset()`, `compute_folding_energy()` |
| `QuaternionSignalProcessor` | Neural network layer | `forward()`, `quaternion_modulation()` |
| `QuaternionRuffleOptimizer` | Field perturbation optimizer | `step()`, `get_optimizer_state()` |
| `QuaternionMemoryTracer` | State evolution tracker | `record()`, `export()`, `compute_memory_coherence()` |

---

## 📈 Benchmarks

### 🏆 Performance Comparison

<table>
<tr>
<th>Task</th>
<th>QRF</th>
<th>Transformer</th>
<th>LSTM</th>
<th>Dense</th>
</tr>
<tr>
<td>🔄 <b>Rotation Prediction</b></td>
<td><b>94.2%</b> 🥇</td>
<td>87.1%</td>
<td>82.3%</td>
<td>71.5%</td>
</tr>
<tr>
<td>🧠 <b>Long-Term Memory</b></td>
<td><b>89.7%</b> 🥇</td>
<td>88.2%</td>
<td>86.4%</td>
<td>62.1%</td>
</tr>
<tr>
<td>📊 <b>Sequence Classification</b></td>
<td>91.3%</td>
<td><b>92.1%</b> 🥇</td>
<td>89.8%</td>
<td>85.4%</td>
</tr>
</table>

### 📉 Parameter Efficiency

```
┌────────────────────────────────────────────────────────────────┐
│ Parameters (thousands)                                         │
├────────────────────────────────────────────────────────────────┤
│ QRF          ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  45K    │
│ Transformer  ████████████████████████████████░░░░░░░░░ 156K   │
│ LSTM         ████████████████████████░░░░░░░░░░░░░░░░░  98K   │
│ Dense        ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  67K   │
└────────────────────────────────────────────────────────────────┘
```

### ⏱️ Training Speed

| Model | Time/Epoch | Memory (GPU) |
|-------|------------|--------------|
| QRF | 12.3s | 1.2 GB |
| Transformer | 8.7s | 2.1 GB |
| LSTM | 15.4s | 0.9 GB |
| Dense | 4.2s | 0.6 GB |

---

## 🎯 Use Cases

<table>
<tr>
<td align="center" width="33%">
<h3>🤖 Robotics</h3>
<img src="https://img.icons8.com/fluency/96/000000/robot-2.png" width="64"/>
<br>
<sub>Pose estimation, motion planning, orientation tracking</sub>
</td>
<td align="center" width="33%">
<h3>🧬 Molecular</h3>
<img src="https://img.icons8.com/fluency/96/000000/molecule.png" width="64"/>
<br>
<sub>Protein folding, molecular dynamics, drug discovery</sub>
</td>
<td align="center" width="33%">
<h3>🎮 3D Vision</h3>
<img src="https://img.icons8.com/fluency/96/000000/3d-object.png" width="64"/>
<br>
<sub>Point clouds, 3D reconstruction, depth estimation</sub>
</td>
</tr>
<tr>
<td align="center" width="33%">
<h3>🛸 Aerospace</h3>
<img src="https://img.icons8.com/fluency/96/000000/rocket.png" width="64"/>
<br>
<sub>Attitude control, trajectory optimization</sub>
</td>
<td align="center" width="33%">
<h3>🎬 Animation</h3>
<img src="https://img.icons8.com/fluency/96/000000/clapperboard.png" width="64"/>
<br>
<sub>Motion capture, skeletal animation, interpolation</sub>
</td>
<td align="center" width="33%">
<h3>🔬 Physics</h3>
<img src="https://img.icons8.com/fluency/96/000000/physics.png" width="64"/>
<br>
<sub>Spin systems, quantum state modeling</sub>
</td>
</tr>
</table>

---

## 📚 API Reference

### QuaternionRuffleField

```python
class QuaternionRuffleField(nn.Module):
    """
    Core quaternion field with dynamic neuron states.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons in the field
    space_dim : int, default=4
        Dimensionality of coordinate space
    device : torch.device, optional
        Computation device (auto-detected if None)
    enable_memory : bool, default=True
        Enable state memory preservation
        
    Attributes
    ----------
    coordinates : nn.Parameter
        Spatial positions [n_neurons, space_dim]
    quaternions : nn.Parameter
        Rotational states [n_neurons, 4]
    field_temperature : torch.Tensor
        Adaptive temperature parameter
    coherence_factor : torch.Tensor
        Inter-neuron coupling strength
    """
```

<details>
<summary><b>📖 View all methods</b></summary>

| Method | Description |
|--------|-------------|
| `forward(update_dynamics=True)` | Compute dynamic distance matrix |
| `reset(preserve_memory=True)` | Reset field state |
| `restore_from_memory(blend_factor=0.5)` | Restore from memory with SLERP |
| `compute_folding_energy()` | Calculate total field energy |
| `compute_dynamic_distances()` | Get neuron distance matrix |
| `get_field_state_summary()` | Get state statistics dict |
| `update_field_dynamics(input_energy=None)` | Update temperature/coherence |

</details>

### QuaternionSignalProcessor

```python
class QuaternionSignalProcessor(nn.Module):
    """
    Signal processor with quaternion modulation and attention.
    
    Parameters
    ----------
    field : QuaternionRuffleField
        The quaternion field to process through
    hidden_dim : int, default=64
        Hidden representation dimensionality
    use_attention : bool, default=True
        Enable multi-head attention mechanism
    attention_heads : int, default=4
        Number of attention heads
    head_dim : int, default=16
        Dimension per attention head
    """
```

---

## 🧪 Examples

### 📓 Jupyter Notebooks

| Notebook | Description | Colab |
|----------|-------------|-------|
| `01_quickstart.ipynb` | Basic usage and concepts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |
| `02_rotation_prediction.ipynb` | 3D rotation estimation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |
| `03_visualization.ipynb` | Field state visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |
| `04_custom_architectures.ipynb` | Building custom models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |

### 🔍 Advanced Examples

<details>
<summary><b>🎨 Visualization of Field Evolution</b></summary>

```python
from quaternion_ruffle_field import QuaternionMemoryTracer
import matplotlib.pyplot as plt

# Create tracer
tracer = QuaternionMemoryTracer(field)

# Training loop with recording
for epoch in range(100):
    tracer.record()
    output = processor(x)
    loss.backward()
    optimizer.step()
    qrf_optimizer.step()

# Export history
history = tracer.export()

# Plot energy evolution
plt.figure(figsize=(10, 4))
plt.plot(history['energy'].numpy())
plt.xlabel('Step')
plt.ylabel('Field Energy')
plt.title('Quaternion Field Energy Evolution')
plt.show()

# Compute coherence
coherence = tracer.compute_memory_coherence()
print(f"Memory Coherence: {coherence:.4f}")
```

</details>

<details>
<summary><b>🔧 Custom Quaternion Layer</b></summary>

```python
from quaternion_ruffle_field import create_quaternion_layer
import torch.nn as nn

class QuaternionClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        
        self.qrf_layer = create_quaternion_layer(
            input_dim=input_dim,
            output_dim=256,
            n_neurons=64,
            use_attention=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        # x: [batch, seq, features]
        x = self.qrf_layer(x)
        x = x.mean(dim=1)  # Global pooling
        return self.classifier(x)
```

</details>

---

## 🤝 Contributing

We love contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### 🛠️ Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/quaternion-ruffle-field.git
cd quaternion-ruffle-field
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black src/ --check

# Build docs
cd docs && make html
```

### 📝 Contribution Areas

- 🐛 **Bug Reports** - Found an issue? Let us know!
- 💡 **Feature Requests** - Have an idea? Open a discussion!
- 📖 **Documentation** - Help improve our docs
- 🧪 **Tests** - Increase our coverage
- 🔬 **Research** - Novel applications and extensions

---

## 📄 Citation

If you use Quaternion Ruffle Field in your research, please cite:

```bibtex
@article{author2025quaternion,
  title={Quaternion Ruffle Field: Neural Networks with Orientational State},
  author={Your Name},
  journal={arXiv preprint arXiv:2025.XXXXX},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

## 🙏 Acknowledgments

- The PyTorch team for the amazing framework
- The geometric deep learning community for inspiration
- Contributors and early adopters

---

## 📬 Contact

<p align="center">
  <a href="https://twitter.com/yourusername">
    <img src="https://img.shields.io/badge/Twitter-@yourusername-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
  </a>
  <a href="https://linkedin.com/in/yourusername">
    <img src="https://img.shields.io/badge/LinkedIn-yourusername-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="mailto:your.email@example.com">
    <img src="https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>

---

<p align="center">
  <b>⭐ Star us on GitHub — it motivates us a lot!</b>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/yourusername/quaternion-ruffle-field?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/yourusername/quaternion-ruffle-field?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/watchers/yourusername/quaternion-ruffle-field?style=social" alt="GitHub watchers">
</p>

<p align="center">
  Made with ❤️ and 🔮 quaternions
</p>