"""Crossformer encoder with the IOHDecoder future-query decoder.

This hybrid keeps Crossformer's DSW embedding and two-stage attention encoder,
but replaces the native hierarchical forecasting decoder with per-horizon
future queries. It is meant as a direct stress test: does the proposed decoder
and Event KL loss still help when the history encoder is already a strong
supervised time-series model?
"""

from __future__ import annotations

import math

import torch
from torch import nn

from iohdecoder.models.tsl_baselines import (
    ENC_IN_FULL,
    TARGET_CHANNEL_IDX,
    _CrossformerScaleBlock,
    _PatchEmbedding,
    _batch_to_enc,
)


class CrossformerFutureQueryIOHDecoder(nn.Module):
    """Crossformer memory encoder plus future-query cross-attention decoder."""

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        enc_in: int = ENC_IN_FULL,
        static_dim: int = 4,
        target_channel: int = TARGET_CHANNEL_IDX,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        seg_len: int = 12,
        win_size: int = 2,
        factor: int = 10,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        residual_prediction: bool = True,
        memory_scales: str = "final",
        **_kwargs,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if memory_scales not in {"final", "all"}:
            raise ValueError("memory_scales must be either 'final' or 'all'.")
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.static_dim = int(static_dim)
        self.target_channel = int(target_channel)
        self.seg_len = int(seg_len)
        self.e_layers = int(e_layers)
        self.residual_prediction = bool(residual_prediction)
        self.memory_scales = str(memory_scales)

        pad_in = (math.ceil(history_len / seg_len) * seg_len) - history_len
        in_seg_num = math.ceil(history_len / seg_len)
        seg_nums = [in_seg_num]
        for _layer in range(1, e_layers):
            seg_nums.append(math.ceil(seg_nums[-1] / win_size))

        self.enc_embed = _PatchEmbedding(d_model, seg_len, seg_len, padding=pad_in, dropout=0.0)
        self.enc_pos = nn.Parameter(torch.randn(1, enc_in, in_seg_num, d_model) * 0.02)
        self.pre_norm = nn.LayerNorm(d_model)
        self.scale_blocks = nn.ModuleList(
            [
                _CrossformerScaleBlock(
                    win_size=1 if layer == 0 else win_size,
                    seg_num=seg_nums[layer],
                    factor=factor,
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for layer in range(e_layers)
            ]
        )
        self.memory_norm = nn.LayerNorm(d_model)
        self.static_projection = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.future_query = nn.Embedding(pred_len, d_model)
        self.horizon_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.memory_to_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.register_buffer(
            "horizon_grid",
            torch.linspace(0.0, 1.0, pred_len).view(pred_len, 1),
            persistent=False,
        )

    def _encode_memory(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(batch)
        if x.shape[1] != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {x.shape[1]}.")
        if x.shape[2] != self.enc_in:
            raise ValueError(f"Expected enc_in={self.enc_in}, got {x.shape[2]}.")
        batch_size = x.size(0)
        z, _n_vars = self.enc_embed(x.transpose(1, 2))
        z = z.reshape(batch_size, self.enc_in, z.size(-2), z.size(-1))
        z = self.pre_norm(z + self.enc_pos[:, :, : z.size(2), :])

        memories = [z]
        for block in self.scale_blocks:
            z = block(z)
            memories.append(z)
        selected = memories if self.memory_scales == "all" else [memories[-1]]
        memory = torch.cat([item.reshape(batch_size, self.enc_in * item.size(2), item.size(3)) for item in selected], dim=1)
        return self.memory_norm(memory)

    def _future_queries(self, memory: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        positions = torch.arange(self.pred_len, device=memory.device)
        query = self.future_query(positions).unsqueeze(0).expand(batch_size, -1, -1)
        horizon = self.horizon_projection(self.horizon_grid.to(memory.device)).unsqueeze(0)
        context = self.memory_to_query(memory.mean(dim=1)).unsqueeze(1)
        static_context = self.static_projection(x_stat.float()).unsqueeze(1)
        return query + horizon + context + static_context

    def forward(self, batch: dict[str, torch.Tensor], return_dict: bool = False):
        memory = self._encode_memory(batch)
        decoded = self.decoder(self._future_queries(memory, batch["x_stat"]), memory)
        residual = self.output_head(decoded).squeeze(-1)
        if self.residual_prediction:
            anchor = batch["x_dyn"].float()[:, self.target_channel, -1].unsqueeze(1)
            prediction = anchor + residual
        else:
            prediction = residual
        if return_dict:
            return {
                "prediction": prediction,
                "memory": memory,
                "decoded_queries": decoded,
            }
        return prediction

