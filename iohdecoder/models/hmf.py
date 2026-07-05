"""Minimal HMF encoder/forecaster adapted from FlowRAG.

This keeps the local FlowRAG HMF protocol: MAP/SBP input, symmetric instance
normalization, trend/seasonal decomposition, patch encoders, and a single-pass
future prediction. It is a reproducible local baseline, not a claim of exact
equivalence to every detail of the original HMF paper.
"""

from __future__ import annotations

import torch
from torch import nn


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = torch.cat(
            [x[:, :1].repeat(1, left, 1), x, x[:, -1:].repeat(1, right, 1)],
            dim=1,
        )
        trend = self.pool(padded.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class HMFBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        dropout: float,
        patch_size: int,
        sequence_len: int,
    ):
        super().__init__()
        if sequence_len % patch_size:
            raise ValueError("sequence_len must be divisible by patch_size")
        self.channels = channels
        self.patch_size = patch_size
        self.sequence_len = sequence_len
        self.num_patches = sequence_len // patch_size
        self.patch_embed = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Conv1d(channels, d_model, kernel_size=patch_size, stride=patch_size),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.position = nn.Embedding(self.num_patches, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, patch_size * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        positions = torch.arange(self.num_patches, device=x.device)
        tokens = self.dropout(tokens + self.position(positions).unsqueeze(0))
        tokens = self.encoder(tokens)
        return self.head(tokens).reshape(batch_size, self.sequence_len, self.channels)


class HMFForecaster(nn.Module):
    MAP_CHANNEL = 0
    SBP_CHANNEL = 1

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        d_model: int = 64,
        d_ff: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        patch_size: int = 15,
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.channels = 2
        sequence_len = history_len + pred_len
        self.decomposition = MovingAverageDecomposition(moving_avg_kernel)
        branch_args = dict(
            channels=self.channels,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            patch_size=patch_size,
            sequence_len=sequence_len,
        )
        self.seasonal_branch = HMFBranch(**branch_args)
        self.trend_branch = HMFBranch(**branch_args)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history = batch["x_dyn"].float()[:, [self.MAP_CHANNEL, self.SBP_CHANNEL]].transpose(1, 2)
        mean = history.mean(dim=1, keepdim=True).detach()
        std = history.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = (history - mean) / std
        future_pad = normalized.new_zeros(normalized.shape[0], self.pred_len, self.channels)
        seasonal, trend = self.decomposition(torch.cat([normalized, future_pad], dim=1))
        prediction = self.seasonal_branch(seasonal) + self.trend_branch(trend)
        prediction = prediction * std + mean
        return prediction[:, -self.pred_len :, 0]
