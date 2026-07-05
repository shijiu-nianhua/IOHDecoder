"""Clinical IOH baselines: HMF (IJCAI 2025) and CMA (AAAI 2023).

Adapted from the official references:

  - HMF: github.com/Mingyue-Cheng/HMF — Cheng et al., IJCAI 2025.
         Two-branch (trend / seasonal) decoder-only Transformer with patch
         embedding and symmetric instance normalization. Uses **MAP + SBP**
         (channels 0 and 1 of x_dyn under the MAP/SBP matched protocol). The autoregressive
         chunked rollout ("Dynamic Sequence Modeling") is replaced with a single
         pass: future positions are zero-padded into the buffer and predicted
         in one forward — we lose strict AR, but keep the decomposition + patch
         + transformer + symmetric-norm architecture. Loss is MSE on the MAP
         channel only.

  - CMA: Lu et al., "A Composite Multi-Attention Framework for Intraoperative
         Hypotension Early Warning", AAAI 2023 (no public code). Architecture
         per the paper: 4 modalities where each modality = (one vital sign +
         3 demographics broadcast along time), a single multi-head self-attention
         block with **cosine similarity** scoring (8 heads × 8 dim, d_model=64)
         applied independently per modality, then concatenated and fed through a
         lightweight CNN + dense head. Pure MAP regression — IOH event detection
         is a deterministic post-processing step over the forecast.

Both classes consume the FactorRAG batch dict and return (B, pred_len) — a MAP
forecast in normalised space, matching the protocol used by every other model
in `Traditional_models/`.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class _MovingAvgDecomp(nn.Module):
    """Autoformer-style series decomposition: return (seasonal, trend)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, C). Replicate-pad both sides to preserve length.
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_padded.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


# ===========================================================================
# HMF (IJCAI 2025): Hybrid Multi-Factor Network
# ===========================================================================


class _HMFBranch(nn.Module):
    """Single trend or seasonal branch of HMF: patch embed + Transformer + value head.

    Single-pass version (no AR rollout). The buffer [encoder, zero-future] is
    embedded once, processed by the Transformer, and projected back to per-timestep
    values via a learned upsample.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float,
        patch_kernel: int,
        actual_len: int,
    ):
        super().__init__()
        self.c_in = c_in
        self.patch_kernel = patch_kernel
        self.actual_len = actual_len
        assert actual_len % patch_kernel == 0
        self.n_patches = actual_len // patch_kernel

        # Patch embedding: 3-conv stack (1x1 channel mix -> strided patch conv -> 1x1)
        self.conv1 = nn.Conv1d(c_in, c_in, kernel_size=1)
        self.conv2 = nn.Conv1d(c_in, d_model, kernel_size=patch_kernel, stride=patch_kernel)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=1)
        # Learned positional embedding per patch (one row per patch index)
        self.pos_embed = nn.Embedding(self.n_patches, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder block (decoder-only-style: bidirectional, no causal mask)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Per-patch head: project d_model -> patch_kernel * c_in values
        # (= one chunk worth of timesteps × c_in channels)
        self.head = nn.Linear(d_model, patch_kernel * c_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, actual_len, c_in)
        B, L, C = x.shape
        assert L == self.actual_len and C == self.c_in
        # Patch embed
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.conv1(x)
        x = self.conv2(x)  # (B, d_model, n_patches)
        x = self.conv3(x)
        x = x.transpose(1, 2)  # (B, n_patches, d_model)
        # Add learned positional embedding
        pos_ids = torch.arange(self.n_patches, device=x.device)
        x = x + self.pos_embed(pos_ids).unsqueeze(0)
        x = self.dropout(x)
        # Transformer encoder
        x = self.encoder(x)  # (B, n_patches, d_model)
        # Project each patch token to (patch_kernel * c_in) timestep values
        out = self.head(x)  # (B, n_patches, patch_kernel * c_in)
        out = out.reshape(B, self.n_patches, self.patch_kernel, self.c_in)
        out = out.reshape(B, self.actual_len, self.c_in)
        return out


class HMFForecaster(nn.Module):
    """HMF: Hybrid Multi-Factor Network (Cheng et al., IJCAI 2025).

    Faithful adaptation of github.com/Mingyue-Cheng/HMF for the FactorRAG batch
    interface. Uses **MAP and SBP** (channels 0 and 1 of x_dyn under the
    matched protocol) as inputs. The autoregressive chunked rollout is replaced with a
    single-pass prediction over the zero-padded [encoder, future] buffer — the
    decomposition + patch + Transformer + symmetric-normalisation architecture
    is preserved.
    """

    MAP_CHANNEL = 0  # ART_MBP index in x_dyn under the matched protocol
    SBP_CHANNEL = 1  # ART_SBP index in x_dyn under the matched protocol

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        patch_kernel: int = 15,
        moving_avg_kernel: int = 25,
        **_kwargs,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.c_in = 2  # MAP + SBP
        self.patch_kernel = patch_kernel
        self.actual_len = history_len + pred_len
        # Both ends must align to patch_kernel
        assert self.actual_len % patch_kernel == 0, (
            f"history_len + pred_len ({self.actual_len}) must be a multiple of patch_kernel ({patch_kernel})"
        )

        self.decomp = _MovingAvgDecomp(moving_avg_kernel)
        self.season_branch = _HMFBranch(
            c_in=self.c_in,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            patch_kernel=patch_kernel,
            actual_len=self.actual_len,
        )
        self.trend_branch = _HMFBranch(
            c_in=self.c_in,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            patch_kernel=patch_kernel,
            actual_len=self.actual_len,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_dyn = batch["x_dyn"].float()
        # x_enc: (B, history_len, 2) — channels [MAP, SBP]
        x_enc = x_dyn[:, [self.MAP_CHANNEL, self.SBP_CHANNEL], :].transpose(1, 2).contiguous()

        # Symmetric instance norm: use encoder statistics (RevIN-like, no learnable affine)
        means = x_enc.mean(dim=1, keepdim=True).detach()
        stdev = x_enc.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        x_enc_norm = (x_enc - means) / stdev

        # Buffer = [encoder_norm, zero_future]
        B = x_enc.size(0)
        future_pad = torch.zeros(B, self.pred_len, self.c_in, device=x_enc.device, dtype=x_enc.dtype)
        buffer = torch.cat([x_enc_norm, future_pad], dim=1)  # (B, actual_len, 2)

        # Series decomposition (trend + seasonal) on the combined buffer
        seasonal, trend = self.decomp(buffer)

        # Two independent branches
        seasonal_out = self.season_branch(seasonal)
        trend_out = self.trend_branch(trend)

        # Recombine and inverse-normalise
        out = seasonal_out + trend_out  # (B, actual_len, 2)
        out = out * stdev + means

        # Return MAP channel (index 0 in our 2-channel local view) for the future window
        return out[:, -self.pred_len :, 0]


# ===========================================================================
# CMA (AAAI 2023): Composite Multi-Attention Framework
# ===========================================================================


class _CosineSimAttention(nn.Module):
    """Multi-head self-attention with cosine similarity scoring (CMA paper)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0  # cosine is already in [-1, 1]; an optional learnable scale could help

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        # Cosine similarity: normalise q and k per head
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # Scores: (B, H, T, T)
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # Weighted sum
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        out = out.reshape(B, T, D)
        return self.out_proj(out)


class _CMAModalityEncoder(nn.Module):
    """Per-modality encoder: project (T, 1+demographics) -> (T, d_model), PE + cosine attention."""

    def __init__(self, input_channels: int, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pe = _SinusoidalPE(d_model, max_len=2048)
        self.attn = _CosineSimAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_channels)
        h = self.input_proj(x)  # (B, T, d_model)
        h = h + self.pe(h.size(1))
        h = self.dropout(h)
        a = self.attn(h)
        return self.norm(h + a)


class CMAForecaster(nn.Module):
    """CMA: Composite Multi-Attention Framework (Lu et al., AAAI 2023).

    Architecture per the paper (no public code):
      * **4 modalities**: each modality = (one vital sign over time) ⊕ (3 demographics
        broadcast along time as additional channels).  We use:
          modality 0: MAP (ART_MBP, channel 0 of x_dyn)
          modality 1: SBP (ART_SBP, channel 1 of x_dyn)
        plus 3 demographics from x_stat[:, :3] (= age, sex, bmi).
      * Per modality: Linear(4, d_model=64) + sinusoidal PE + multi-head self-attention
        with **cosine similarity** (8 heads × 8 dim).
      * Composite head: concat modality features → CNN (Conv-ELU-Conv-ELU-MaxPool) →
        flatten → 3 dense layers with LayerNorm before the final dense → MAP forecast.
    """

    VITAL_INDICES = (0, 1)  # MAP, SBP under the matched protocol
    STATIC_INDICES = (0, 1, 2)  # age, sex, bmi (use first 3 of x_stat)

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        d_model: int = 64,
        n_heads: int = 8,
        dropout: float = 0.1,
        cnn_channels: int = 64,
        dense_dim: int = 128,
        n_dense: int = 3,
        vital_indices: tuple[int, ...] | list[int] | None = None,
        static_indices: tuple[int, ...] | list[int] | None = None,
        **_kwargs,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.vital_indices = tuple(int(index) for index in (self.VITAL_INDICES if vital_indices is None else vital_indices))
        self.static_indices = tuple(int(index) for index in (self.STATIC_INDICES if static_indices is None else static_indices))
        self.n_modalities = len(self.vital_indices)
        self.n_static = len(self.static_indices)
        self.d_model = d_model

        # Per-modality encoder (input = 1 vital + n_static demographics broadcast)
        self.modality_encoder = _CMAModalityEncoder(
            input_channels=1 + self.n_static,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # CNN composite head: concat modalities (n_modalities × d_model along channels),
        # then Conv1d along time. Kernels chosen modestly so the CNN can mix neighbours
        # without collapsing the time axis.
        concat_dim = self.n_modalities * d_model  # 4 × 64 = 256 default
        self.conv1 = nn.Conv1d(concat_dim, cnn_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # After Conv1+ELU+Conv2+ELU+MaxPool(2): time becomes history_len // 2
        pooled_len = history_len // 2

        layers: list[nn.Module] = []
        in_dim = cnn_channels * pooled_len
        for i in range(n_dense - 1):
            layers.append(nn.Linear(in_dim, dense_dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            in_dim = dense_dim
        self.dense_stack = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.head = nn.Linear(in_dim, pred_len)

    def _build_modalities(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # x_dyn: (B, C, T) — pick selected vitals
        x_dyn = batch["x_dyn"].float()
        vitals = x_dyn[:, list(self.vital_indices), :].transpose(1, 2)  # (B, T, n_modalities)
        # x_stat: (B, 4) — pick 3 demographics, broadcast along time
        B, T, _ = vitals.shape
        if self.n_static:
            statics = batch["x_stat"].float()[:, list(self.static_indices)]  # (B, n_static)
            statics_broadcast = statics.unsqueeze(1).expand(B, T, self.n_static)  # (B, T, n_static)
        else:
            statics_broadcast = x_dyn.new_zeros(B, T, 0)
        # For each modality, form (B, T, 1 + n_static)
        modality_inputs = []
        for m in range(self.n_modalities):
            vital_m = vitals[:, :, m : m + 1]  # (B, T, 1)
            modality_inputs.append(torch.cat([vital_m, statics_broadcast], dim=-1))
        # Stack: (B, n_modalities, T, 1 + n_static)
        return torch.stack(modality_inputs, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        modalities = self._build_modalities(batch)  # (B, n_modalities, T, 1 + n_static)
        B, M, T, C = modalities.shape
        # Encode each modality independently with shared encoder
        flat = modalities.reshape(B * M, T, C)
        encoded = self.modality_encoder(flat)  # (B*M, T, d_model)
        encoded = encoded.reshape(B, M, T, self.d_model)
        # Concat along channel dim: (B, T, n_modalities * d_model)
        encoded = encoded.permute(0, 2, 1, 3).reshape(B, T, M * self.d_model)
        # Conv1d expects (B, C, T)
        x = encoded.transpose(1, 2)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)  # (B, cnn_channels, T // 2)
        x = x.flatten(start_dim=1)  # (B, cnn_channels * (T // 2))
        x = self.dense_stack(x)
        x = self.layer_norm(x)
        return self.head(x)  # (B, pred_len)
