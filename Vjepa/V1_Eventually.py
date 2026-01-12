"""
tri_temporal_vjepa.py

Tri-Temporal V-JEPA Training Scaffold (Automotive World Models)

This file implements a V-JEPA-faithful training loop:
- Context encoder + predictor operate on a masked token sequence
- Target encoder (EMA teacher) processes the full unmasked token sequence
- Predictor receives context embeddings + (mask token + 3D pos embed) and predicts masked token embeddings
- Loss is L1 in latent space on masked tokens only (stop-grad on teacher)

Tri-temporal extensions:
- SOON objective: "forecast mask" masks near-future time slices to force latent prediction of near-future patches
- EVENTUALLY objective: outcome abstractor predicts a sparse outcome embedding from early context,
  aligned (InfoNCE) to teacher features from a far-future window (basin-of-outcomes style)

Notes:
- This is a scaffold. Swap the backbone with facebookresearch/vjepa2 or facebookresearch/jepa components
  once you wire your data loader and tokenizer to match the official shapes/positional encoding.
"""

from __future__ import annotations

import math
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("tri_temporal_vjepa")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -----------------------------
# Config
# -----------------------------

@dataclass
class TriTemporalVJEPAConfig:
    # Video sampling
    image_size: int = 224
    num_frames: int = 84             # must be divisible by tubelet_t (default 2)
    frame_rate: int = 4              # for your own mental mapping only

    # Tubelets (V-JEPA paper uses 2x16x16 tubelets)
    tubelet_t: int = 2
    patch: int = 16

    # Token dims
    enc_dim: int = 768
    pred_dim: int = 384
    enc_depth: int = 12
    pred_depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # Tri-temporal segmentation (in raw frames)
    now_frames: int = 4
    soon_frames: int = 16
    eventually_frames: int = 64

    # Masking (spatial blocks repeated across time)
    short_num_blocks: int = 8
    short_scale: float = 0.15
    long_num_blocks: int = 2
    long_scale: float = 0.70
    aspect_min: float = 0.75
    aspect_max: float = 1.50

    # Training
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    total_steps: int = 200000
    grad_clip: float = 1.0

    # EMA teacher
    ema_momentum: float = 0.996

    # Loss weights
    lambda_vjepa: float = 1.0
    lambda_soon: float = 1.0
    lambda_eventually: float = 0.25

    # EVENTUALLY (outcome) head
    outcome_queries: int = 8
    outcome_temperature: float = 0.2


# -----------------------------
# Utilities
# -----------------------------

@torch.no_grad()
def ema_update_(target: nn.Module, online: nn.Module, m: float) -> None:
    """Polyak EMA update: target = m*target + (1-m)*online."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))


def cosine_schedule(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def info_nce_logits(
    q: torch.Tensor,  # [B, D]
    k: torch.Tensor,  # [B, D]
    temperature: float
) -> torch.Tensor:
    """Compute BxB logits where diagonal is positive."""
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    return (q @ k.t()) / temperature


# -----------------------------
# 3D sin-cos position embedding
# (V-JEPA paper uses absolute 3D sin-cos; V-JEPA 2 uses 3D RoPE, but this is faithful to V-JEPA.)
# -----------------------------

def _sincos_1d(n: int, d: int, device: torch.device) -> torch.Tensor:
    """
    Returns [n, d] 1D sincos with d even.
    """
    if d % 2 != 0:
        raise ValueError("pos dim must be even for sincos")
    pos = torch.arange(n, device=device).float().unsqueeze(1)             # [n, 1]
    omega = torch.arange(d // 2, device=device).float()
    omega = 1.0 / (10000 ** (omega / (d // 2)))                           # [d/2]
    out = pos * omega.unsqueeze(0)                                        # [n, d/2]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)             # [n, d]


def make_3d_sincos_pos_embed(t: int, h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """
    Returns [L, dim] where L = t*h*w, flattened in (t, h, w) order.
    """
    # Split dim across (t,h,w) and keep each even
    dt = dim // 3
    dh = dim // 3
    dw = dim - dt - dh
    # force even
    dt -= dt % 2
    dh -= dh % 2
    dw -= dw % 2
    # if we lost dims due to even rounding, add to dw
    rem = dim - (dt + dh + dw)
    dw += rem
    if dw % 2 != 0:
        dw -= 1
        dt += 1  # keep total; dt might become odd, fix below
    if dt % 2 != 0:
        dt += 1
        dw -= 1
    if dh % 2 != 0:
        dh += 1
        dw -= 1
    if dt <= 0 or dh <= 0 or dw <= 0 or (dt + dh + dw) != dim:
        raise ValueError("bad pos split for dim")

    pt = _sincos_1d(t, dt, device)  # [t, dt]
    ph = _sincos_1d(h, dh, device)  # [h, dh]
    pw = _sincos_1d(w, dw, device)  # [w, dw]

    # Broadcast to grid then concat
    # grid shapes:
    #   pt: [t, 1, 1, dt]
    #   ph: [1, h, 1, dh]
    #   pw: [1, 1, w, dw]
    ptg = pt[:, None, None, :]
    phg = ph[None, :, None, :]
    pwg = pw[None, None, :, :]
    pos = torch.cat([ptg.expand(t, h, w, dt),
                     phg.expand(t, h, w, dh),
                     pwg.expand(t, h, w, dw)], dim=-1)  # [t,h,w,dim]
    return pos.reshape(t * h * w, dim)                   # [L, dim]


# -----------------------------
# Masking (3D multi-block strategy, spatial blocks repeated over time)
# -----------------------------

def sample_spatial_block_mask(
    h: int,
    w: int,
    num_blocks: int,
    scale: float,
    aspect_min: float,
    aspect_max: float,
    device: torch.device
) -> torch.Tensor:
    """
    Returns spatial mask [h, w] where True means masked (to be predicted).
    Blocks may overlap; union is taken.
    """
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    area = h * w

    for _ in range(num_blocks):
        target_area = max(1.0, scale * area)
        aspect = float(torch.empty(1, device=device).uniform_(aspect_min, aspect_max).item())
        bh = int(round(math.sqrt(target_area * aspect)))
        bw = int(round(math.sqrt(target_area / aspect)))
        bh = max(1, min(h, bh))
        bw = max(1, min(w, bw))
        top = int(torch.randint(0, max(1, h - bh + 1), (1,), device=device).item())
        left = int(torch.randint(0, max(1, w - bw + 1), (1,), device=device).item())
        mask[top:top + bh, left:left + bw] = True

    return mask


def build_3d_mask_from_spatial(spatial_mask: torch.Tensor, t: int) -> torch.Tensor:
    """
    spatial_mask: [h, w] bool
    returns: [t, h, w] bool, repeated across time
    """
    return spatial_mask.unsqueeze(0).expand(t, -1, -1)


def flatten_3d_mask(mask_3d: torch.Tensor) -> torch.Tensor:
    """
    mask_3d: [t,h,w] bool
    returns: [L] bool in (t,h,w) flatten order
    """
    return mask_3d.reshape(-1)


def token_time_indices(t: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Returns [L] int time index per token in flattened order.
    """
    tt = torch.arange(t, device=device)[:, None, None].expand(t, h, w)
    return tt.reshape(-1)


# -----------------------------
# Backbone and Predictor (V-JEPA style)
# -----------------------------

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class TubeletTokenizer(nn.Module):
    """
    Patchify video into tubelet tokens via Conv3D.
    Input: [B, T, C, H, W]
    Output: tokens [B, L, D] plus grid shape (t', h', w')
    """
    def __init__(self, tubelet_t: int, patch: int, in_ch: int, dim: int):
        super().__init__()
        self.tubelet_t = tubelet_t
        self.patch = patch
        self.proj = nn.Conv3d(
            in_channels=in_ch,
            out_channels=dim,
            kernel_size=(tubelet_t, patch, patch),
            stride=(tubelet_t, patch, patch),
            padding=0
        )

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        b, t, c, h, w = video.shape
        x = video.permute(0, 2, 1, 3, 4)          # [B, C, T, H, W]
        x = self.proj(x)                          # [B, D, t', h', w']
        _, d, tp, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, L, D]
        return x, (tp, hp, wp)


class VJEPABackbone(nn.Module):
    """
    Tokenizer + Transformer encoder (context encoder or target encoder).
    """
    def __init__(self, cfg: TriTemporalVJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = TubeletTokenizer(cfg.tubelet_t, cfg.patch, in_ch=3, dim=cfg.enc_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.enc_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.enc_depth)
        ])
        self.norm = nn.LayerNorm(cfg.enc_dim)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        tokens, grid = self.tokenizer(video)
        return self.forward_tokens(tokens), grid


class VJEPAPredictor(nn.Module):
    """
    Predictor receives:
    - context embeddings z_ctx [B, N, enc_dim] produced by context encoder on visible tokens
    - mask token embeddings for masked positions: (shared mask vector + pos embed) in pred_dim
    It outputs predicted embeddings in enc_dim for masked tokens.
    """
    def __init__(self, cfg: TriTemporalVJEPAConfig):
        super().__init__()
        self.cfg = cfg

        self.ctx_proj = nn.Linear(cfg.enc_dim, cfg.pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.pred_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.pred_depth)
        ])
        self.norm = nn.LayerNorm(cfg.pred_dim)
        self.out_proj = nn.Linear(cfg.pred_dim, cfg.enc_dim)

        self.pos_proj = nn.Linear(cfg.enc_dim, cfg.pred_dim)  # project 3D pos embed to pred_dim

    def forward(
        self,
        z_ctx: torch.Tensor,            # [B, N, enc_dim]
        pos_masked: torch.Tensor        # [M, enc_dim]
    ) -> torch.Tensor:
        b, n, _ = z_ctx.shape
        m = pos_masked.shape[0]

        ctx = self.ctx_proj(z_ctx)                          # [B, N, pred_dim]
        mask = self.mask_token.expand(b, m, -1)             # [B, M, pred_dim]
        mask = mask + self.pos_proj(pos_masked).unsqueeze(0)  # add pos to mask tokens

        x = torch.cat([ctx, mask], dim=1)                   # [B, N+M, pred_dim]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred_mask = x[:, n:, :]                             # [B, M, pred_dim]
        return self.out_proj(pred_mask)                     # [B, M, enc_dim]


class OutcomeAbstractor(nn.Module):
    """
    EVENTUALLY: produce a sparse outcome embedding from a set of tokens.
    Uses learnable queries that attend to tokens, then pools.
    """
    def __init__(self, cfg: TriTemporalVJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.queries = nn.Parameter(torch.randn(1, cfg.outcome_queries, cfg.enc_dim) * 0.02)
        self.attn = nn.MultiheadAttention(cfg.enc_dim, cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.enc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.enc_dim, cfg.enc_dim),
            nn.GELU(),
            nn.Linear(cfg.enc_dim, cfg.enc_dim)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, D]
        returns outcome embedding [B, D]
        """
        b = tokens.shape[0]
        q = self.queries.expand(b, -1, -1)  # [B, Q, D]
        y, _ = self.attn(q, tokens, tokens, need_weights=False)
        y = self.norm(y)
        y = y + self.mlp(y)
        return y.mean(dim=1)  # [B, D]


# -----------------------------
# Tri-temporal V-JEPA model
# -----------------------------

class TriTemporalVJEPA(nn.Module):
    def __init__(self, cfg: TriTemporalVJEPAConfig):
        super().__init__()
        self.cfg = cfg

        self.online = VJEPABackbone(cfg)
        self.target = VJEPABackbone(cfg)
        self.predictor = VJEPAPredictor(cfg)
        self.outcome = OutcomeAbstractor(cfg)

        # Initialize target as copy of online
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        ema_update_(self.target, self.online, self.cfg.ema_momentum)

    def tokenize_with_pos(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        """
        Returns:
          tokens_with_pos: [B, L, D]
          pos_embed:       [L, D]
          grid: (t', h', w')
        """
        tokens, grid = self.online.tokenizer(video)  # use online tokenizer for shape
        tp, hp, wp = grid
        pos = make_3d_sincos_pos_embed(tp, hp, wp, self.cfg.enc_dim, device=video.device)  # [L, D]
        tokens = tokens + pos.unsqueeze(0)
        return tokens, pos, grid

    def vjepa_masked_loss(
        self,
        tokens_with_pos: torch.Tensor,    # [B, L, D], pos already added
        pos_embed: torch.Tensor,          # [L, D]
        grid: Tuple[int, int, int],
        masked_idx: torch.Tensor,         # [M] long
        target_tokens_full: torch.Tensor  # [B, L, D] teacher outputs (already encoded)
    ) -> torch.Tensor:
        """
        One V-JEPA loss for a given mask:
        - context encoder on visible tokens
        - predictor on (z_ctx + mask tokens with pos) to predict masked tokens
        - L1 to teacher tokens at masked positions
        """
        device = tokens_with_pos.device
        L = tokens_with_pos.shape[1]
        mask_bool = torch.zeros((L,), dtype=torch.bool, device=device)
        mask_bool[masked_idx] = True
        visible_idx = torch.nonzero(~mask_bool, as_tuple=False).squeeze(1)  # [N]

        # Context encoder on visible tokens only
        x_vis = tokens_with_pos[:, visible_idx, :]                 # [B, N, D]
        z_vis = self.online.forward_tokens(x_vis)                  # [B, N, D]

        # Predictor uses masked pos embeds
        pos_masked = pos_embed[masked_idx]                         # [M, D]
        pred_masked = self.predictor(z_vis, pos_masked)            # [B, M, D]

        # Teacher targets
        with torch.no_grad():
            tgt_masked = target_tokens_full[:, masked_idx, :]      # [B, M, D]

        return F.l1_loss(pred_masked, tgt_masked)

    def forward_train(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute tri-temporal losses for one batch.
        """
        cfg = self.cfg
        device = video.device

        # Patchify + add pos once
        tokens_with_pos, pos_embed, grid = self.tokenize_with_pos(video)
        tp, hp, wp = grid
        L = tp * hp * wp

        # Teacher tokens (full unmasked token sequence)
        with torch.no_grad():
            # Target encoder must process full (unmasked) tokens_with_pos
            tgt_full = self.target.forward_tokens(tokens_with_pos)  # [B, L, D]

        # ---- Multi-mask V-JEPA denoising losses (short and long) ----
        spatial_short = sample_spatial_block_mask(hp, wp, cfg.short_num_blocks, cfg.short_scale,
                                                  cfg.aspect_min, cfg.aspect_max, device)
        spatial_long = sample_spatial_block_mask(hp, wp, cfg.long_num_blocks, cfg.long_scale,
                                                 cfg.aspect_min, cfg.aspect_max, device)
        mask_short = flatten_3d_mask(build_3d_mask_from_spatial(spatial_short, tp))  # [L]
        mask_long = flatten_3d_mask(build_3d_mask_from_spatial(spatial_long, tp))    # [L]

        masked_idx_short = torch.nonzero(mask_short, as_tuple=False).squeeze(1)
        masked_idx_long = torch.nonzero(mask_long, as_tuple=False).squeeze(1)

        loss_vjepa_short = self.vjepa_masked_loss(tokens_with_pos, pos_embed, grid, masked_idx_short, tgt_full)
        loss_vjepa_long = self.vjepa_masked_loss(tokens_with_pos, pos_embed, grid, masked_idx_long, tgt_full)
        loss_vjepa = 0.5 * (loss_vjepa_short + loss_vjepa_long)

        # ---- SOON objective: forecast mask (mask near-future time slice) ----
        # Convert frame counts to tubelet-time counts
        # Each tubelet consumes cfg.tubelet_t frames.
        if (cfg.now_frames % cfg.tubelet_t) != 0 or (cfg.soon_frames % cfg.tubelet_t) != 0 or (cfg.eventually_frames % cfg.tubelet_t) != 0:
            raise ValueError("now/soon/eventually frames must be divisible by tubelet_t")

        now_t = cfg.now_frames // cfg.tubelet_t
        soon_t = cfg.soon_frames // cfg.tubelet_t
        evt_t = cfg.eventually_frames // cfg.tubelet_t

        if now_t + soon_t + evt_t != tp:
            # This keeps the scaffold strict. You can relax by sampling or cropping.
            raise ValueError(f"num_frames mismatch: tp={tp}, but now+soon+event={now_t+soon_t+evt_t}. "
                             f"Adjust cfg.num_frames or temporal splits.")

        t_idx = token_time_indices(tp, hp, wp, device)  # [L]
        soon_time_mask = (t_idx >= now_t) & (t_idx < (now_t + soon_t))  # tokens in SOON window
        masked_idx_soon = torch.nonzero(soon_time_mask, as_tuple=False).squeeze(1)

        # Context uses only NOW window tokens as visible (drop everything else from context encoder)
        # We implement this by additionally masking tokens not in NOW, and letting predictor handle only SOON masked indices.
        now_time_mask = (t_idx < now_t)
        visible_idx_now = torch.nonzero(now_time_mask, as_tuple=False).squeeze(1)

        # Encode only NOW tokens
        x_now = tokens_with_pos[:, visible_idx_now, :]
        z_now = self.online.forward_tokens(x_now)

        # Predict SOON masked tokens from NOW context
        pos_soon = pos_embed[masked_idx_soon]
        pred_soon = self.predictor(z_now, pos_soon)  # [B, M_soon, D]
        with torch.no_grad():
            tgt_soon = tgt_full[:, masked_idx_soon, :]
        loss_soon = F.l1_loss(pred_soon, tgt_soon)

        # ---- EVENTUALLY objective: basin alignment (InfoNCE) ----
        # Outcome prediction from NOW context tokens (online), aligned to far-future teacher outcome
        # Online outcome uses NOW tokens only (unmasked tokens_with_pos subset, then online encoder)
        z_now_full = self.online.forward_tokens(tokens_with_pos[:, visible_idx_now, :])  # [B, N_now, D]
        out_pred = self.outcome(z_now_full)  # [B, D]

        # Teacher outcome from EVENTUALLY time window
        evt_time_mask = (t_idx >= (now_t + soon_t)) & (t_idx < (now_t + soon_t + evt_t))
        evt_idx = torch.nonzero(evt_time_mask, as_tuple=False).squeeze(1)
        with torch.no_grad():
            out_tgt = tgt_full[:, evt_idx, :].mean(dim=1)  # [B, D] (simple pool; replace with queries if you want)

        logits = info_nce_logits(out_pred, out_tgt, temperature=cfg.outcome_temperature)  # [B,B]
        labels = torch.arange(video.shape[0], device=device)
        loss_eventually = F.cross_entropy(logits, labels)

        return {
            "loss_vjepa": loss_vjepa,
            "loss_soon": loss_soon,
            "loss_eventually": loss_eventually,
            "loss_total": (cfg.lambda_vjepa * loss_vjepa
                           + cfg.lambda_soon * loss_soon
                           + cfg.lambda_eventually * loss_eventually)
        }


# -----------------------------
# Dataset (scaffold)
# -----------------------------

class AutomotiveVideoDataset(Dataset):
    """
    Expects:
      data_root/videos/*.mp4
      data_root/annotations/*.json (optional)
    If you do not have videos wired yet, it will fall back to random tensors.
    """
    def __init__(self, data_root: str, cfg: TriTemporalVJEPAConfig, split: str = "train"):
        self.root = Path(data_root)
        self.cfg = cfg
        self.split = split
        self.video_paths = sorted((self.root / "videos").glob("*.mp4"))
        self.ann_dir = self.root / "annotations"
        logger.info(f"[{split}] found {len(self.video_paths)} mp4 files under {self.root/'videos'}")

    def __len__(self) -> int:
        # If no data is present, expose some length so the scaffold can run.
        return max(1, len(self.video_paths))

    def _load_video(self, path: Path) -> torch.Tensor:
        """
        Returns float tensor [T, C, H, W] in [0,1]
        Uses torchvision.io.read_video if available.
        """
        try:
            import torchvision
            frames, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")  # [T, H, W, C], uint8
            frames = frames.float() / 255.0
            frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        except Exception:
            # Fallback: random frames
            frames = torch.rand(self.cfg.num_frames, 3, self.cfg.image_size, self.cfg.image_size)
            return frames

        # Resize and temporal sample
        frames = F.interpolate(frames, size=(self.cfg.image_size, self.cfg.image_size), mode="bilinear", align_corners=False)

        # Uniform sample or pad
        T = frames.shape[0]
        if T >= self.cfg.num_frames:
            idx = torch.linspace(0, T - 1, self.cfg.num_frames).long()
            frames = frames[idx]
        else:
            pad = self.cfg.num_frames - T
            frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)
        return frames

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.video_paths) == 0:
            frames = torch.rand(self.cfg.num_frames, 3, self.cfg.image_size, self.cfg.image_size)
            return {"frames": frames}

        path = self.video_paths[idx % len(self.video_paths)]
        frames = self._load_video(path)

        return {"frames": frames}


# -----------------------------
# Trainer
# -----------------------------

class Trainer:
    def __init__(self, cfg: TriTemporalVJEPAConfig, model: TriTemporalVJEPA, loader: DataLoader, device: str):
        self.cfg = cfg
        self.model = model.to(device)
        self.loader = loader
        self.device = device

        self.opt = torch.optim.AdamW(self.model.online.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
        self.step = 0

    def train(self, max_steps: int) -> None:
        self.model.train()
        it = iter(self.loader)

        for _ in range(max_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)

            lr = cosine_schedule(self.step, self.cfg.warmup_steps, self.cfg.total_steps, self.cfg.lr)
            for pg in self.opt.param_groups:
                pg["lr"] = lr

            video = batch["frames"].to(self.device)  # [T,C,H,W]
            video = video.unsqueeze(0) if video.dim() == 4 else video  # ensure [B,T,C,H,W]
            if video.dim() == 5 and video.shape[2] != 3:
                # if accidental [B,T,H,W,C]
                raise ValueError("expected video as [B,T,C,H,W]")

            losses = self.model.forward_train(video)

            self.opt.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.online.parameters(), self.cfg.grad_clip)
            self.opt.step()

            # EMA teacher update
            self.model.update_target()

            if self.step % 50 == 0:
                logger.info(
                    f"step={self.step} lr={lr:.2e} "
                    f"total={losses['loss_total'].item():.4f} "
                    f"vjepa={losses['loss_vjepa'].item():.4f} "
                    f"soon={losses['loss_soon'].item():.4f} "
                    f"event={losses['loss_eventually'].item():.4f}"
                )

            self.step += 1


# -----------------------------
# Entry point
# -----------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser("Tri-Temporal V-JEPA scaffold")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()

    cfg = TriTemporalVJEPAConfig()

    # Hard guard: the temporal split must match the tubelet time grid exactly in this scaffold.
    if cfg.num_frames != (cfg.now_frames + cfg.soon_frames + cfg.eventually_frames):
        raise ValueError("cfg.num_frames must equal now_frames + soon_frames + eventually_frames in this scaffold")

    ds = AutomotiveVideoDataset(args.data_root, cfg, split="train")
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    model = TriTemporalVJEPA(cfg)
    logger.info(f"online params: {sum(p.numel() for p in model.online.parameters()):,}")
    logger.info(f"predictor params: {sum(p.numel() for p in model.predictor.parameters()):,}")
    logger.info(f"outcome params: {sum(p.numel() for p in model.outcome.parameters()):,}")

    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    trainer = Trainer(cfg, model, dl, device=dev)
    trainer.train(args.steps)


if __name__ == "__main__":
    main()
