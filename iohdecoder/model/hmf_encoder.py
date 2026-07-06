"""HMF-style historical encoder used by IOHDecoder.

The release keeps the HMF encoder components needed by the paper method:
MAP/SBP input, symmetric instance normalization, moving-average
trend/seasonal decomposition, patch embedding, and Transformer patch encoders.
The shared HMF forecasting head is intentionally omitted because IOHDecoder
uses the future-query decoder for prediction.
"""

from __future__ import annotations

import torch
from torch import nn


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = torch.cat(
            [x[:, :1].repeat(1, left, 1), x, x[:, -1:].repeat(1, right, 1)],
            dim=1,
        )
        trend = self.pool(padded.transpose(1, 2)).transpose(1, 2)
        return x - trend, trend


class HMFPatchEncoder(nn.Module):
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
            raise ValueError("sequence_len must be divisible by patch_size.")
        self.num_patches = int(sequence_len) // int(patch_size)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        positions = torch.arange(self.num_patches, device=x.device)
        tokens = self.dropout(tokens + self.position(positions).unsqueeze(0))
        return self.encoder(tokens)


class HMFEncoder(nn.Module):
    """Encode historical MAP/SBP into patch memory for future-query decoding."""

    def __init__(
        self,
        history_len: int = 450,
        channels: int = 2,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        patch_size: int = 15,
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.channels = int(channels)
        self.decomposition = MovingAverageDecomposition(moving_avg_kernel)
        branch_args = dict(
            channels=self.channels,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            patch_size=int(patch_size),
            sequence_len=self.history_len,
        )
        self.seasonal_encoder = HMFPatchEncoder(**branch_args)
        self.trend_encoder = HMFPatchEncoder(**branch_args)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time, channels].")
        if history.shape[1] != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {history.shape[1]}.")
        if history.shape[2] != self.channels:
            raise ValueError(f"Expected channels={self.channels}, got {history.shape[2]}.")

        mean = history.mean(dim=1, keepdim=True).detach()
        std = history.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5).detach()
        normalized = (history - mean) / std
        seasonal, trend = self.decomposition(normalized)
        memory = self.seasonal_encoder(seasonal) + self.trend_encoder(trend)
        return self.output_norm(memory)
