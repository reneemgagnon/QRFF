"""
VJepa_AUTO_V2.py

Tri-Temporal V-JEPA (V-JEPA style) training scaffold for video world models.

What this is
- Online (context) encoder + predictor trained to predict masked latent tokens.
- Target encoder is an EMA teacher (stop-grad) that encodes the full, unmasked video.
- Loss is computed in latent space on masked tokens only (L1).
- Tri-temporal extensions:
  - SOON: predict near-future tokens using only NOW context
  - EVENTUALLY: predict a compact "outcome" embedding from NOW context and align it to far-future teacher features (InfoNCE)

What this is not
- This is not the official facebookresearch/vjepa2 codebase. It is a faithful scaffold with clear shapes,
  useful for wiring your own data loader and later swapping in official components.

Directory expectation (default)
  data_root/
    videos/*.mp4

Run
  python VJepa_AUTO_V2.py --data_root /path/to/data --steps 1000

Optional
  python VJepa_AUTO_V2.py --data_root /path/to/data --steps 100 --amp --compile

Notes
- If no videos are found or decoding fails, it falls back to random tensors so the loop can run.
- This script is intentionally monolithic for ease of editing.
"""

from __future__ import annotations

import os
import math
import json
import time
import random
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("VJepa_AUTO_V2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class VJepaAutoV2Config:
    # Video sampling
    image_size: int = 224
    num_frames: int = 84          # must be divisible by tubelet_t
    channels: int = 3

    # Tubelets (paper often uses 2x16x16 tubelets)
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
    seed: int = 1337

    # EMA teacher
    ema_momentum: float = 0.996

    # Loss weights
    lambda_vjepa: float = 1.0
    lambda_soon: float = 1.0
    lambda_eventually: float = 0.25

    # EVENTUALLY (outcome) head
    outcome_queries: int = 8
    outcome_temperature: float = 0.2

    # Runtime
    log_every: int = 50
    ckpt_every: int = 1000
    out_dir: str = "runs/vjepa_auto_v2"
    fallback_length: int = 2048  # number of samples to expose when no videos found


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def info_nce_logits(q: torch.Tensor, k: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute BxB logits where diagonal is positive."""
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    return (q @ k.t()) / temperature


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    return torch.distributed.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        torch.distributed.barrier()


# -----------------------------------------------------------------------------
# 3D sin-cos position embedding (absolute)
# -----------------------------------------------------------------------------

def _sincos_1d(n: int, d: int, device: torch.device) -> torch.Tensor:
    if d % 2 != 0:
        raise ValueError("pos dim must be even for sincos")
    pos = torch.arange(n, device=device).float().unsqueeze(1)  # [n, 1]
    omega = torch.arange(d // 2, device=device).float()
    omega = 1.0 / (10000 ** (omega / (d // 2)))
    out = pos * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def make_3d_sincos_pos_embed(t: int, h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """Returns [L, dim] where L = t*h*w, flattened in (t, h, w) order."""
    dt = dim // 3
    dh = dim // 3
    dw = dim - dt - dh

    dt -= dt % 2
    dh -= dh % 2
    dw -= dw % 2

    rem = dim - (dt + dh + dw)
    dw += rem

    # ensure even splits
    if dw % 2 != 0:
        dw -= 1
        dt += 1
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

    ptg = pt[:, None, None, :]
    phg = ph[None, :, None, :]
    pwg = pw[None, None, :, :]

    pos = torch.cat(
        [
            ptg.expand(t, h, w, dt),
            phg.expand(t, h, w, dh),
            pwg.expand(t, h, w, dw),
        ],
        dim=-1,
    )
    return pos.reshape(t * h * w, dim)


# -----------------------------------------------------------------------------
# Masking (spatial blocks repeated across time)
# -----------------------------------------------------------------------------

def sample_spatial_block_mask(
    h: int,
    w: int,
    num_blocks: int,
    scale: float,
    aspect_min: float,
    aspect_max: float,
    device: torch.device
) -> torch.Tensor:
    """Returns spatial mask [h, w] where True means masked."""
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
    return spatial_mask.unsqueeze(0).expand(t, -1, -1)


def flatten_3d_mask(mask_3d: torch.Tensor) -> torch.Tensor:
    return mask_3d.reshape(-1)


def token_time_indices(t: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    tt = torch.arange(t, device=device)[:, None, None].expand(t, h, w)
    return tt.reshape(-1)


# -----------------------------------------------------------------------------
# Backbone and predictor (V-JEPA style)
# -----------------------------------------------------------------------------

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
    Input:  [B, T, C, H, W]
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
            padding=0,
        )

    def forward(self, video: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        b, t, c, h, w = video.shape
        x = video.permute(0, 2, 1, 3, 4)     # [B, C, T, H, W]
        x = self.proj(x)                     # [B, D, t', h', w']
        _, d, tp, hp, wp = x.shape
        x = x.flatten(2).transpose(1, 2)     # [B, L, D]
        return x, (tp, hp, wp)


class VJEPABackbone(nn.Module):
    """
    Tokenizer + Transformer encoder (context encoder or target encoder).
    """
    def __init__(self, cfg: VJepaAutoV2Config):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = TubeletTokenizer(cfg.tubelet_t, cfg.patch, in_ch=cfg.channels, dim=cfg.enc_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.enc_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.enc_depth)]
        )
        self.norm = nn.LayerNorm(cfg.enc_dim)

    def forward_tokens(self, tokens_with_pos: torch.Tensor) -> torch.Tensor:
        x = tokens_with_pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def patchify_with_pos(self, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
        tokens, grid = self.tokenizer(video)  # [B,L,D]
        tp, hp, wp = grid
        pos = make_3d_sincos_pos_embed(tp, hp, wp, self.cfg.enc_dim, device=video.device)
        tokens = tokens + pos.unsqueeze(0)
        return tokens, pos, grid


class VJEPAPredictor(nn.Module):
    """
    Predictor receives:
      - context embeddings z_ctx [B, N, enc_dim] from the online encoder on visible tokens
      - pos_masked [M, enc_dim]
    It outputs predicted embeddings in enc_dim for masked tokens.
    """
    def __init__(self, cfg: VJepaAutoV2Config):
        super().__init__()
        self.cfg = cfg
        self.ctx_proj = nn.Linear(cfg.enc_dim, cfg.pred_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.pred_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.pred_depth)]
        )
        self.norm = nn.LayerNorm(cfg.pred_dim)
        self.out_proj = nn.Linear(cfg.pred_dim, cfg.enc_dim)
        self.pos_proj = nn.Linear(cfg.enc_dim, cfg.pred_dim)

    def forward(self, z_ctx: torch.Tensor, pos_masked: torch.Tensor) -> torch.Tensor:
        b, n, _ = z_ctx.shape
        m = pos_masked.shape[0]
        ctx = self.ctx_proj(z_ctx)                                   # [B, N, pred_dim]
        mask = self.mask_token.expand(b, m, -1)                      # [B, M, pred_dim]
        mask = mask + self.pos_proj(pos_masked).unsqueeze(0)         # add pos to mask tokens
        x = torch.cat([ctx, mask], dim=1)                            # [B, N+M, pred_dim]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pred_mask = x[:, n:, :]                                      # [B, M, pred_dim]
        return self.out_proj(pred_mask)                              # [B, M, enc_dim]


class OutcomeAbstractor(nn.Module):
    """
    EVENTUALLY head: produces a compact outcome embedding from a set of tokens.
    Learnable queries attend to tokens, then we pool queries.
    """
    def __init__(self, cfg: VJepaAutoV2Config):
        super().__init__()
        self.cfg = cfg
        self.queries = nn.Parameter(torch.randn(1, cfg.outcome_queries, cfg.enc_dim) * 0.02)
        self.attn = nn.MultiheadAttention(cfg.enc_dim, cfg.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.enc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.enc_dim, cfg.enc_dim),
            nn.GELU(),
            nn.Linear(cfg.enc_dim, cfg.enc_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        q = self.queries.expand(b, -1, -1)  # [B, Q, D]
        y, _ = self.attn(q, tokens, tokens, need_weights=False)
        y = self.norm(y)
        y = y + self.mlp(y)
        return y.mean(dim=1)                # [B, D]


# -----------------------------------------------------------------------------
# Tri-temporal V-JEPA model
# -----------------------------------------------------------------------------

class TriTemporalVJEPA(nn.Module):
    def __init__(self, cfg: VJepaAutoV2Config):
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

    def _vjepa_masked_loss(
        self,
        tokens_o: torch.Tensor,           # [B, L, D] online tokens with pos
        pos_embed: torch.Tensor,          # [L, D]
        masked_idx: torch.Tensor,         # [M]
        teacher_full: torch.Tensor        # [B, L, D] teacher outputs (stop-grad)
    ) -> torch.Tensor:
        device = tokens_o.device
        L = tokens_o.shape[1]
        mask_bool = torch.zeros((L,), dtype=torch.bool, device=device)
        mask_bool[masked_idx] = True
        visible_idx = torch.nonzero(~mask_bool, as_tuple=False).squeeze(1)  # [N]

        # Online context encoder on visible tokens only
        z_vis = self.online.forward_tokens(tokens_o[:, visible_idx, :])     # [B, N, D]

        # Predictor predicts masked token embeddings
        pos_masked = pos_embed[masked_idx]                                  # [M, D]
        pred_masked = self.predictor(z_vis, pos_masked)                     # [B, M, D]

        # Teacher targets for masked indices
        tgt_masked = teacher_full[:, masked_idx, :]                         # [B, M, D]
        return F.l1_loss(pred_masked, tgt_masked)

    def forward_train(self, video: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        video: [B, T, C, H, W]
        returns losses dict
        """
        cfg = self.cfg
        device = video.device

        # Basic temporal sanity checks
        if cfg.num_frames != (cfg.now_frames + cfg.soon_frames + cfg.eventually_frames):
            raise ValueError("cfg.num_frames must equal now_frames + soon_frames + eventually_frames in this scaffold")
        if (cfg.now_frames % cfg.tubelet_t) != 0 or (cfg.soon_frames % cfg.tubelet_t) != 0 or (cfg.eventually_frames % cfg.tubelet_t) != 0:
            raise ValueError("now/soon/eventually frames must be divisible by tubelet_t")

        # Online patchify + pos
        tokens_o, pos_embed, grid_o = self.online.patchify_with_pos(video)   # [B,L,D], [L,D], (tp,hp,wp)
        tp, hp, wp = grid_o

        # Teacher patchify + pos and encode full tokens with EMA teacher
        with torch.no_grad():
            tokens_t, pos_t, grid_t = self.target.patchify_with_pos(video)
            if grid_t != grid_o:
                raise ValueError(f"grid mismatch online={grid_o} target={grid_t}. Ensure identical tokenizer strides.")
            # Use teacher's own pos embedding (identical by construction), but keep the variable consistent
            teacher_full = self.target.forward_tokens(tokens_t)              # [B, L, D]

        # Multi-mask V-JEPA latent prediction losses (short and long)
        spatial_short = sample_spatial_block_mask(hp, wp, cfg.short_num_blocks, cfg.short_scale, cfg.aspect_min, cfg.aspect_max, device)
        spatial_long = sample_spatial_block_mask(hp, wp, cfg.long_num_blocks, cfg.long_scale, cfg.aspect_min, cfg.aspect_max, device)

        mask_short = flatten_3d_mask(build_3d_mask_from_spatial(spatial_short, tp))  # [L]
        mask_long = flatten_3d_mask(build_3d_mask_from_spatial(spatial_long, tp))   # [L]

        masked_idx_short = torch.nonzero(mask_short, as_tuple=False).squeeze(1)
        masked_idx_long = torch.nonzero(mask_long, as_tuple=False).squeeze(1)

        loss_vjepa_short = self._vjepa_masked_loss(tokens_o, pos_embed, masked_idx_short, teacher_full)
        loss_vjepa_long = self._vjepa_masked_loss(tokens_o, pos_embed, masked_idx_long, teacher_full)
        loss_vjepa = 0.5 * (loss_vjepa_short + loss_vjepa_long)

        # Tri-temporal indexing in tubelet-time units
        now_t = cfg.now_frames // cfg.tubelet_t
        soon_t = cfg.soon_frames // cfg.tubelet_t
        evt_t = cfg.eventually_frames // cfg.tubelet_t
        if now_t + soon_t + evt_t != tp:
            raise ValueError(f"num_frames mismatch: tp={tp}, but now+soon+event={now_t+soon_t+evt_t}. Adjust splits.")

        t_idx = token_time_indices(tp, hp, wp, device)  # [L]

        # SOON: predict near-future tokens using only NOW context
        soon_time_mask = (t_idx >= now_t) & (t_idx < (now_t + soon_t))
        masked_idx_soon = torch.nonzero(soon_time_mask, as_tuple=False).squeeze(1)

        now_time_mask = (t_idx < now_t)
        visible_idx_now = torch.nonzero(now_time_mask, as_tuple=False).squeeze(1)

        z_now = self.online.forward_tokens(tokens_o[:, visible_idx_now, :])         # [B, N_now, D]
        pos_soon = pos_embed[masked_idx_soon]                                        # [M_soon, D]
        pred_soon = self.predictor(z_now, pos_soon)                                  # [B, M_soon, D]
        tgt_soon = teacher_full[:, masked_idx_soon, :]
        loss_soon = F.l1_loss(pred_soon, tgt_soon)

        # EVENTUALLY: basin alignment (InfoNCE)
        out_pred = self.outcome(z_now)                                               # [B, D]
        evt_time_mask = (t_idx >= (now_t + soon_t)) & (t_idx < (now_t + soon_t + evt_t))
        evt_idx = torch.nonzero(evt_time_mask, as_tuple=False).squeeze(1)

        with torch.no_grad():
            out_tgt = teacher_full[:, evt_idx, :].mean(dim=1)                        # [B, D]

        logits = info_nce_logits(out_pred, out_tgt, temperature=cfg.outcome_temperature)
        labels = torch.arange(video.shape[0], device=device)
        loss_eventually = F.cross_entropy(logits, labels)

        loss_total = (cfg.lambda_vjepa * loss_vjepa
                      + cfg.lambda_soon * loss_soon
                      + cfg.lambda_eventually * loss_eventually)

        return {
            "loss_vjepa": loss_vjepa,
            "loss_soon": loss_soon,
            "loss_eventually": loss_eventually,
            "loss_total": loss_total,
        }


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class AutomotiveVideoDataset(Dataset):
    """
    Expects:
      data_root/videos/*.mp4

    Fallback behavior:
      If no videos are found or decoding fails, returns random tensors.
    """
    def __init__(
        self,
        data_root: str,
        cfg: VJepaAutoV2Config,
        split: str = "train",
        video_glob: str = "*.mp4",
    ):
        self.root = Path(data_root)
        self.cfg = cfg
        self.split = split
        self.video_paths = sorted((self.root / "videos").glob(video_glob))
        logger.info(f"[{split}] found {len(self.video_paths)} videos under {self.root/'videos'} (glob={video_glob})")

        # Try to locate a decoder stack once so __getitem__ is cheap
        self._decoder = self._pick_decoder()

    def __len__(self) -> int:
        if len(self.video_paths) > 0:
            return len(self.video_paths)
        return int(self.cfg.fallback_length)

    def _pick_decoder(self) -> str:
        # Prefer torchvision (common) if it can be imported
        try:
            import torchvision  # noqa: F401
            return "torchvision"
        except Exception:
            pass

        # Optional: decord
        try:
            import decord  # noqa: F401
            return "decord"
        except Exception:
            pass

        return "random"

    def _resize_frames(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        # frames: [T, C, H, W]
        if frames_tchw.shape[-1] == self.cfg.image_size and frames_tchw.shape[-2] == self.cfg.image_size:
            return frames_tchw
        return F.interpolate(
            frames_tchw,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )

    def _uniform_sample_or_pad(self, frames_tchw: torch.Tensor) -> torch.Tensor:
        T = frames_tchw.shape[0]
        if T >= self.cfg.num_frames:
            idx = torch.linspace(0, T - 1, self.cfg.num_frames).long()
            return frames_tchw[idx]
        pad = self.cfg.num_frames - T
        return torch.cat([frames_tchw, frames_tchw[-1:].expand(pad, -1, -1, -1)], dim=0)

    def _load_torchvision(self, path: Path) -> torch.Tensor:
        import torchvision
        frames, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")  # [T, H, W, C], uint8
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        return frames

    def _load_decord(self, path: Path) -> torch.Tensor:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(str(path))
        # Decode all frames (simple; optimize later)
        frames = vr.get_batch(list(range(len(vr))))  # [T, H, W, C], uint8 torch
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)          # [T, C, H, W]
        return frames

    def _load_video(self, path: Path) -> torch.Tensor:
        if self._decoder == "torchvision":
            return self._load_torchvision(path)
        if self._decoder == "decord":
            return self._load_decord(path)
        # random fallback
        return torch.rand(self.cfg.num_frames, self.cfg.channels, self.cfg.image_size, self.cfg.image_size)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if len(self.video_paths) == 0:
            frames = torch.rand(self.cfg.num_frames, self.cfg.channels, self.cfg.image_size, self.cfg.image_size)
            return {"frames": frames}

        path = self.video_paths[idx % len(self.video_paths)]
        try:
            frames = self._load_video(path)
        except Exception:
            frames = torch.rand(self.cfg.num_frames, self.cfg.channels, self.cfg.image_size, self.cfg.image_size)

        frames = self._resize_frames(frames)
        frames = self._uniform_sample_or_pad(frames)
        return {"frames": frames}


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        cfg: VJepaAutoV2Config,
        model: TriTemporalVJEPA,
        loader: DataLoader,
        device: torch.device,
        amp: bool = False,
    ):
        self.cfg = cfg
        self.device = device
        self.amp = amp and (device.type == "cuda")

        self.model = model.to(device)
        self.loader = loader
        self.step = 0

        params = list(self.model.online.parameters()) + list(self.model.predictor.parameters()) + list(self.model.outcome.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # Optional DDP wrapping (online + predictor + outcome get gradients)
        if is_distributed():
            self.model.online = torch.nn.parallel.DistributedDataParallel(self.model.online, device_ids=[device.index] if device.type == "cuda" else None)
            self.model.predictor = torch.nn.parallel.DistributedDataParallel(self.model.predictor, device_ids=[device.index] if device.type == "cuda" else None)
            self.model.outcome = torch.nn.parallel.DistributedDataParallel(self.model.outcome, device_ids=[device.index] if device.type == "cuda" else None)

        self.out_dir = Path(cfg.out_dir)
        if is_main_process():
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            with open(self.out_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(asdict(cfg), f, indent=2)

    def _unwrap(self, m: nn.Module) -> nn.Module:
        return m.module if hasattr(m, "module") else m

    @torch.no_grad()
    def _update_teacher(self) -> None:
        # Always update teacher from the non-DDP underlying module
        online = self._unwrap(self.model.online)
        target = self.model.target
        ema_update_(target, online, self.cfg.ema_momentum)

    def save_ckpt(self, tag: str) -> None:
        if not is_main_process():
            return
        ckpt = {
            "step": self.step,
            "cfg": asdict(self.cfg),
            "online": self._unwrap(self.model.online).state_dict(),
            "predictor": self._unwrap(self.model.predictor).state_dict(),
            "outcome": self._unwrap(self.model.outcome).state_dict(),
            "target": self.model.target.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        path = self.out_dir / "checkpoints" / f"ckpt_{tag}.pt"
        torch.save(ckpt, path)
        logger.info(f"saved checkpoint: {path}")

    def train(self, max_steps: int) -> None:
        self.model.train()

        it = iter(self.loader)
        t0 = time.time()

        for _ in range(max_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.loader)
                batch = next(it)

            lr = cosine_schedule(self.step, self.cfg.warmup_steps, self.cfg.total_steps, self.cfg.lr)
            for pg in self.opt.param_groups:
                pg["lr"] = lr

            frames = batch["frames"].to(self.device, non_blocking=True)  # [B,T,C,H,W] or [T,C,H,W]
            if frames.dim() == 4:
                frames = frames.unsqueeze(0)
            if frames.dim() != 5:
                raise ValueError(f"expected frames as [B,T,C,H,W], got shape {tuple(frames.shape)}")
            if frames.shape[2] != self.cfg.channels:
                raise ValueError(f"expected channels={self.cfg.channels}, got shape {tuple(frames.shape)}")

            video = frames  # [B,T,C,H,W]

            self.opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.amp):
                losses = self.model.forward_train(video)
                loss_total = losses["loss_total"]

            self.scaler.scale(loss_total).backward()

            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.opt.param_groups[0]["params"], self.cfg.grad_clip)

            self.scaler.step(self.opt)
            self.scaler.update()

            # EMA update for teacher
            self._update_teacher()

            if (self.step % self.cfg.log_every) == 0:
                dt = max(1e-6, time.time() - t0)
                steps_per_s = (self.cfg.log_every / dt) if self.step > 0 else 0.0
                t0 = time.time()
                if is_main_process():
                    logger.info(
                        f"step={self.step} lr={lr:.2e} "
                        f"total={loss_total.item():.4f} "
                        f"vjepa={losses['loss_vjepa'].item():.4f} "
                        f"soon={losses['loss_soon'].item():.4f} "
                        f"event={losses['loss_eventually'].item():.4f} "
                        f"steps_per_s={steps_per_s:.2f}"
                    )

            if self.cfg.ckpt_every > 0 and (self.step % self.cfg.ckpt_every) == 0 and self.step > 0:
                self.save_ckpt(tag=str(self.step))

            self.step += 1

        self.save_ckpt(tag="final")


# -----------------------------------------------------------------------------
# Distributed setup (optional)
# -----------------------------------------------------------------------------

def maybe_init_distributed(device: str) -> Tuple[torch.device, Optional[int]]:
    """
    Initializes torch.distributed if launched with torchrun.
    Returns (torch_device, local_rank_or_none).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        # Single process
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda", 0), None
        return torch.device("cpu"), None

    # Distributed
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed not available in this build")

    backend = "nccl" if (device == "cuda" and torch.cuda.is_available()) else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), local_rank
    return torch.device("cpu"), local_rank


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser("VJepa_AUTO_V2: tri-temporal V-JEPA scaffold")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--video_glob", type=str, default="*.mp4")
    p.add_argument("--amp", action="store_true", help="use CUDA AMP mixed precision")
    p.add_argument("--compile", action="store_true", help="torch.compile the model parts (PyTorch 2.x)")
    p.add_argument("--out_dir", type=str, default=None)

    # quick overrides
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    args = p.parse_args()

    cfg = VJepaAutoV2Config()
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    device, local_rank = maybe_init_distributed(args.device)
    seed_everything(cfg.seed + get_rank())

    if is_main_process():
        logger.info(f"device={device} world_size={get_world_size()}")

    ds = AutomotiveVideoDataset(args.data_root, cfg, split="train", video_glob=args.video_glob)

    sampler = None
    if is_distributed():
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True)
        shuffle = False
    else:
        shuffle = True

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    model = TriTemporalVJEPA(cfg)

    if args.compile and hasattr(torch, "compile"):
        # compile the trainable parts only (teacher is used in no_grad)
        model.online = torch.compile(model.online)
        model.predictor = torch.compile(model.predictor)
        model.outcome = torch.compile(model.outcome)

    if is_main_process():
        online_params = sum(p.numel() for p in model.online.parameters())
        pred_params = sum(p.numel() for p in model.predictor.parameters())
        out_params = sum(p.numel() for p in model.outcome.parameters())
        logger.info(f"online params: {online_params:,}")
        logger.info(f"predictor params: {pred_params:,}")
        logger.info(f"outcome params: {out_params:,}")

    trainer = Trainer(cfg, model, dl, device=device, amp=args.amp)
    trainer.train(args.steps)

    barrier()
    if is_main_process():
        logger.info("done")


if __name__ == "__main__":
    main()
