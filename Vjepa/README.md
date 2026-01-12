# Tri-Temporal V-JEPA (NOW, SOON, EVENTUALLY) for Automotive World Models

A V-JEPA-faithful training scaffold that adds **three coupled time scales** to self-supervised video world modeling for driving:

- **NOW**: dense perception of the present (short context window)
- **SOON**: near-term latent forecasting (seconds)
- **EVENTUALLY**: sparse outcome-basin reasoning (longer horizon, abstract modes)

This folder is designed to be a clean place to experiment with **hierarchical temporal abstraction** while keeping the core V-JEPA training mechanics intact (masked latent prediction with an EMA teacher).

---

## What you get

- V-JEPA-style **tubelet tokenization** (spatiotemporal patches)
- **Context encoder (online)** + **Target encoder (EMA teacher)**
- **Predictor** that fills in masked token embeddings (latent space, not pixels)
- Multi-mask training (short and long spatial masks)
- A tri-temporal extension:
  - **SOON**: forecast masking over a near-future time slice
  - **EVENTUALLY**: outcome embedding aligned to far-future teacher features (InfoNCE)

---

## Why tri-temporal

Driving is not a single-horizon problem. A useful world model must do all of this at once:

- Understand what is happening now (lane geometry, actors, signals)
- Anticipate what happens next (brake lights, merges, cut-ins)
- Track what the scenario is trending toward (exit, stop, yield, lane change)

EVENTUALLY is intentionally not a pixel predictor. It is a **basin-of-outcomes** representation: a compact embedding that captures which high-level mode the scene is flowing toward.

---

## V-JEPA correctness checklist

This implementation follows the core V-JEPA training pattern:

1. Tokenize a video into tubelets and add 3D positional information
2. Create masks over token indices
3. **Target encoder (EMA teacher)** encodes the full unmasked token sequence
4. **Context encoder (online)** encodes only visible tokens
5. The **predictor** receives:
   - context embeddings
   - learned mask tokens + masked positions (positional embeddings)
6. Optimize an **L1 regression loss in representation space** on masked tokens only
7. Update the teacher via **EMA**

If you swap the backbone and predictor with the official Meta implementation later, the tri-temporal objectives remain the same.

---

## Tri-temporal objectives

### 1) Base V-JEPA masked prediction (spatial)
A multi-block spatial mask is repeated across time. The model learns to infer masked patch embeddings from visible patches.

### 2) SOON masked prediction (temporal forecast)
The **SOON** time slice is masked, and only the **NOW** slice is provided as visible context. The predictor learns to infer near-future token embeddings in latent space.

### 3) EVENTUALLY basin alignment (outcome embedding)
An **OutcomeAbstractor** produces a sparse embedding from NOW context tokens. A teacher embedding from a far-future window acts as the target. Training uses **InfoNCE** to pull matching (same clip) outcome pairs together and push mismatches apart within the batch.

This produces an outcome-space representation that is more stable than frame-level prediction.

---

## Folder structure

This README is meant to live in:

```
QRFF/
  Vjepa/
    tri_temporal_vjepa.py
    README.md
```

If your filenames differ, update the commands below accordingly.

---

## Quickstart

### Requirements
- Python 3.10+
- PyTorch 2.0+
- torchvision (optional, for video loading)

Install:
```bash
pip install torch torchvision
```

### Data layout
Expected:
```
data_root/
  videos/
    clip_0001.mp4
    clip_0002.mp4
  annotations/
    clip_0001.json   # optional
    clip_0002.json
```

Annotations are optional in this scaffold. If video loading fails, the dataset falls back to random tensors so you can validate the pipeline end-to-end.

### Train
```bash
python tri_temporal_vjepa.py --data_root /path/to/data_root --steps 1000 --device cuda
```

You should see logs like:
- total loss
- vjepa loss
- soon loss
- event loss
- learning rate

---

## Configuration

Edit `TriTemporalVJEPAConfig` inside `tri_temporal_vjepa.py`.

Key settings:

- `num_frames`: total frames per training clip
- `tubelet_t`, `patch`: tubelet shape (default 2 x 16 x 16)
- `enc_dim`, `pred_dim`: encoder and predictor dimensions
- `now_frames`, `soon_frames`, `eventually_frames`: temporal partitioning
- masking controls:
  - `short_num_blocks`, `short_scale`
  - `long_num_blocks`, `long_scale`

Current scaffold assumes:
- `num_frames == now_frames + soon_frames + eventually_frames`
- each time slice divisible by `tubelet_t`

---

## Practical notes for automotive datasets

- EVENTUALLY often needs a longer horizon than a short clip.
  - A common pattern is to sample a longer clip and treat NOW, SOON, EVENTUALLY as sliding windows.
- If you have weak maneuver labels, consider adding **supervised contrastive** on the outcome embedding.
- If you care about lane-level reasoning, avoid early global pooling and keep a set of spatial tokens deeper into the model.

---

## Safety disclaimer

This is research code. It is not a driving policy and is not safe to deploy in real vehicles without extensive validation, safety engineering, and regulatory compliance.

---

## Roadmap ideas

- Swap the scaffold backbone for the official V-JEPA 2 encoder and predictor
- Replace 3D sin-cos position embedding with the exact positional scheme used in your chosen V-JEPA variant
- Add multi-mask per clip to amortize the teacher forward pass
- Add action conditioning (steering, speed) for controllable forecasting
- Add evaluation hooks (lane change prediction, trajectory consistency, scene graph probes)

---

## License

Pick a license before publishing widely. Common choices:
- MIT
- Apache-2.0

---

## Credits

Conceptual foundation: JEPA / V-JEPA family of self-supervised world models by Meta AI and collaborators.
