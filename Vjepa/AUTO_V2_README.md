# VJepa_AUTO_V2

A single-file, monolithic training scaffold for a tri-temporal V-JEPA style video world model.

This script implements the core V-JEPA mechanics:
- Online (context) encoder processes visible tokens only
- Target encoder is an EMA teacher that processes the full, unmasked video (stop-grad)
- A predictor reconstructs masked latent tokens in representation space
- Loss is computed on masked tokens (L1)

It also adds tri-temporal objectives:
- NOW: standard masked latent prediction over the full clip
- SOON: predict near-future tokens using only NOW context
- EVENTUALLY: predict an outcome embedding from NOW context and align it to far-future teacher features with an InfoNCE loss

File:
- `VJepa_AUTO_V2.py`


## Why this exists

If you are building automotive or robotics world models and you want a readable starting point, this is a clean scaffold that:
- runs as a single Python file
- is easy to modify in place
- has clear tensor shapes and checks
- supports AMP, torch.compile, and torchrun DDP


## Requirements

- Python 3.10+
- PyTorch 2.x
- Optional: `torchvision` (video loading via `torchvision.io.read_video`)
- Optional: `decord` (alternative video loader)

Install examples:
```bash
pip install torch torchvision
# optional
pip install decord
```


## Dataset layout

Default expectation:
```
data_root/
  videos/
    *.mp4
```

The loader will:
- decode frames
- resize to `image_size x image_size`
- uniformly sample or pad to `num_frames`

If no videos are found or decoding fails, the dataset falls back to random tensors so the training loop can still run.


## Quick start

Single GPU:
```bash
python VJepa_AUTO_V2.py --data_root /path/to/data_root --steps 1000 --device cuda --amp
```

CPU debug:
```bash
python VJepa_AUTO_V2.py --data_root /path/to/data_root --steps 10 --device cpu
```

Change batch size and workers:
```bash
python VJepa_AUTO_V2.py --data_root /path/to/data_root --steps 1000 --batch_size 2 --num_workers 2 --amp
```


## Multi GPU (torchrun)

Example with 2 GPUs:
```bash
torchrun --nproc_per_node=2 VJepa_AUTO_V2.py --data_root /path/to/data_root --steps 10000 --device cuda --amp
```

Notes:
- The script initializes torch.distributed when launched via torchrun.
- A DistributedSampler is used for the dataset.
- EMA teacher updates run every step.


## What the model is doing

### Tokenization
Video is patchified into tubelets using a Conv3D tokenizer:
- Tubelet size: `tubelet_t x patch x patch` (default `2 x 16 x 16`)
- Output tokens: `[B, L, enc_dim]`
- Grid: `(t', h', w')` where:
  - `t' = num_frames / tubelet_t`
  - `h' = image_size / patch`
  - `w' = image_size / patch`

With defaults:
- `num_frames=84`, `tubelet_t=2` -> `t'=42`
- `image_size=224`, `patch=16` -> `h'=w'=14`
- `L = 42 * 14 * 14 = 8232` tokens per sample

A fixed 3D sin-cos positional embedding is added to tokens.


### V-JEPA latent prediction (NOW)
For each step:
1) Target encoder (EMA teacher) encodes full tokens, producing `teacher_full: [B, L, D]`
2) Two spatial masks are sampled and repeated across time:
   - short mask: many small blocks
   - long mask: few large blocks
3) Online encoder only sees visible tokens, producing `z_vis`
4) Predictor reconstructs embeddings for masked tokens
5) Loss is L1 between predicted masked embeddings and teacher masked embeddings

This produces `loss_vjepa`.


### SOON
Tokens are partitioned by tubelet time index into NOW, SOON, EVENTUALLY segments.
- Online encoder sees only NOW tokens
- Predictor reconstructs SOON token embeddings
- Loss is L1 to teacher SOON token embeddings

This produces `loss_soon`.


### EVENTUALLY
- Online encoder sees only NOW tokens
- An Outcome head (learnable queries attending to NOW tokens) produces a compact vector `out_pred: [B, D]`
- Teacher target is the mean of EVENTUALLY tokens: `out_tgt: [B, D]`
- InfoNCE aligns `out_pred` to `out_tgt` across the batch

This produces `loss_eventually`.


### Total loss
```text
loss_total =
  lambda_vjepa * loss_vjepa
+ lambda_soon  * loss_soon
+ lambda_eventually * loss_eventually
```


## Configuration

All key settings live in the `VJepaAutoV2Config` dataclass inside `VJepa_AUTO_V2.py`.

Important constraints (enforced in code):
- `num_frames == now_frames + soon_frames + eventually_frames`
- each of `now_frames`, `soon_frames`, `eventually_frames` must be divisible by `tubelet_t`

Useful knobs:
- Memory and speed:
  - `batch_size`
  - `enc_dim`
  - `enc_depth`, `pred_depth`
  - `image_size`, `patch`, `tubelet_t`
- Masking:
  - `short_num_blocks`, `short_scale`
  - `long_num_blocks`, `long_scale`
- Teacher stability:
  - `ema_momentum`
- Loss weighting:
  - `lambda_vjepa`, `lambda_soon`, `lambda_eventually`


## Outputs

The script writes to `out_dir` (default `runs/vjepa_auto_v2`):
- `config.json`
- `checkpoints/ckpt_<step>.pt`
- `checkpoints/ckpt_final.pt`

Checkpoints contain:
- online encoder
- predictor
- outcome head
- target encoder
- optimizer and AMP scaler state


## Troubleshooting

### Out of memory
Defaults are intentionally heavy.
Try:
- `--batch_size 1`
- reduce `enc_dim` (for example 384)
- reduce `enc_depth` and `pred_depth` (for example 6 and 6)
- reduce `image_size` (for example 128)
- reduce `num_frames` while keeping the tri-temporal sum constraint

### No videos found
Make sure you have:
`data_root/videos/*.mp4`

If not, the loader will use random data. You will still see losses, but it will not learn anything meaningful.

### Video decoding issues
If `torchvision` decoding fails on your codec, install `decord` and try again, or preconvert your videos to a widely supported MP4/H.264 profile.


## Distillation and ViT compatibility

If you want to distill a ViT or video-ViT encoder into a different backbone (for example your ruffle encoder), this scaffold is compatible because:
- the objective is latent token prediction, not pixel reconstruction
- any encoder that emits `[B, L, D]` patch or tubelet latents can be used for the online/target backbone

A common workflow:
1) distill a pretrained encoder into your custom encoder so it matches the teacher token space
2) run JEPA training using an EMA teacher of your custom encoder


## License

Add your project license here.
