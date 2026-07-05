"""Faithful re-implementations of PatchTST / TimeMixer / Crossformer.

Architectures follow the Time-Series-Library reference
(github.com/thuml/Time-Series-Library, MIT License). Adapted to the FactorRAG
multi-modal batch:
  - x_dyn   (B, C, T)     dynamic vitals; in the matched protocol C=2
                          with channel 0 == ART_MBP and channel 1 == ART_SBP
  - x_pharma(B, 7, 450)   medication intensity
  - p_mask  (B, 7, 450)   medication observed mask
  - x_stat  (B, 4)        demographics (NOT consumed by TSL models — they are
                          pure multivariate forecasters; static features are
                          left for the clinical multi-modal baselines)

We expose three top-level forecaster classes whose forward() takes the batch
dict and returns (B, pred_len) — the predicted MBP trajectory in normalised
space. They are channel-mixing / channel-independent / segment-based multivariate
forecasters; we select the MBP output channel after the model.
"""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# Combined enc_in: 6 dyn + 7 pharma + 7 pharma_mask = 20 channels. Index 1 = MBP.
ENC_IN_FULL = 20
TARGET_CHANNEL_IDX = 1


def _batch_to_enc(
    batch: dict[str, torch.Tensor],
    *,
    dynamic_indices: tuple[int, ...] | list[int] | None = None,
    use_pharma_features: bool = True,
) -> torch.Tensor:
    """Flatten a project batch into the standard TSL [B, L, C] tensor.

    By default this keeps the original full multivariate protocol. Passing
    ``dynamic_indices=[1, 2]`` and ``use_pharma_features=False`` yields the
    MAP/SBP-only protocol used by the HMF-aligned comparison.
    """
    x_dyn = batch["x_dyn"].float()
    if dynamic_indices is not None:
        index = torch.as_tensor(tuple(int(i) for i in dynamic_indices), device=x_dyn.device, dtype=torch.long)
        x_dyn = x_dyn.index_select(1, index)
    if use_pharma_features:
        pharma = torch.tanh(batch["x_pharma"].float() / 90.0) * batch["p_mask"].float()
        x = torch.cat([x_dyn, pharma, batch["p_mask"].float()], dim=1)
    else:
        x = x_dyn
    return x.transpose(1, 2).contiguous()  # (B, L, 20)


# =============================================================================
# Generic layers
# =============================================================================


class _PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding, fixed (non-learnable)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


class _TokenEmbedding(nn.Module):
    """Conv1d(k=3, circular padding) used by DataEmbedding_wo_pos."""

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _DataEmbeddingNoPos(nn.Module):
    """TimeMixer-style embedding: token Conv + dropout (no positional, no mark)."""

    def __init__(self, c_in: int, d_model: int, dropout: float):
        super().__init__()
        self.token = _TokenEmbedding(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.token(x))


class _PatchEmbedding(nn.Module):
    """Channel-independent patch embedding used by PatchTST & Crossformer.

    Input: (B, C, L). Pads right by `padding` then unfolds with the given
    patch_len / stride and flattens (B, C) into the batch dim. Returns
    (B*C, patch_num, d_model) along with C so the caller can reshape.
    """

    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.pad = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        n_vars = x.shape[1]
        x = self.pad(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (B, C, patch_num, patch_len)
        patch_num = x.shape[2]
        x = x.reshape(x.shape[0] * n_vars, patch_num, self.patch_len)
        x = self.value_embedding(x) + self.position_embedding(patch_num)
        return self.dropout(x), n_vars


class _FullAttention(nn.Module):
    """Scaled dot-product attention with post-softmax dropout."""

    def __init__(self, attention_dropout: float = 0.0, scale: float | None = None):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, q, k, v) -> torch.Tensor:
        # q,k,v: (B, L, H, D)
        b, L, H, D = q.shape
        S = k.shape[1]
        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(D)
        scores = torch.einsum("blhd,bshd->bhls", q, k) * scale
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, v)
        return out.contiguous()


class _AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys: int | None = None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.q_proj = nn.Linear(d_model, n_heads * d_keys)
        self.k_proj = nn.Linear(d_model, n_heads * d_keys)
        self.v_proj = nn.Linear(d_model, n_heads * d_keys)
        self.out_proj = nn.Linear(n_heads * d_keys, d_model)
        self.inner = attention

    def forward(self, q, k, v) -> torch.Tensor:
        B, L, _ = q.shape
        S = k.shape[1]
        H = self.n_heads
        D = self.d_keys
        q = self.q_proj(q).view(B, L, H, D)
        k = self.k_proj(k).view(B, S, H, D)
        v = self.v_proj(v).view(B, S, H, D)
        out = self.inner(q, k, v)  # (B, L, H, D)
        return self.out_proj(out.reshape(B, L, H * D))


class _EncoderLayer(nn.Module):
    """Standard transformer encoder layer with Conv1d FFN (TSL style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.attn = _AttentionLayer(_FullAttention(dropout), d_model, n_heads)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(h))
        y = self.conv2(self.dropout(self.act(self.conv1(x.transpose(-1, 1))))).transpose(-1, 1)
        return self.norm2(x + self.dropout(y))


class _Encoder(nn.Module):
    """Stack of encoder layers + optional final norm wrapper.

    TSL's PatchTST wraps with `Transpose -> BatchNorm1d(d_model) -> Transpose`.
    We replicate that exactly to match the reference behaviour.
    """

    def __init__(self, layers: list[_EncoderLayer], d_model: int):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        # Transpose -> BN1d over d_model -> Transpose
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        return x


class _FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_len: int, dropout: float):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, d_model, patch_num)
        x = self.flatten(x)
        return self.dropout(self.linear(x))


# =============================================================================
# Series decomposition (Autoformer style, used by TimeMixer)
# =============================================================================


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C). Pad both sides by replication to keep length.
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        return self.avg(x_padded.transpose(1, 2)).transpose(1, 2)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        return x - trend, trend  # (season, trend)


# =============================================================================
# TimeMixer-specific: RevIN + Past Decomposable Mixing
# =============================================================================


class _RevIN(nn.Module):
    """Reversible instance normalization (TSL StandardNorm.Normalize)."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def _stats(self, x: torch.Tensor):
        reduce_dim = tuple(range(1, x.ndim - 1))
        mean = x.mean(dim=reduce_dim, keepdim=True).detach()
        var = x.var(dim=reduce_dim, keepdim=True, unbiased=False).detach()
        stdev = torch.sqrt(var + self.eps)
        return mean, stdev

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean, self._stdev = self._stats(x)
            out = (x - self._mean) / self._stdev
            if self.affine:
                out = out * self.affine_weight + self.affine_bias
            return out
        if mode == "denorm":
            out = x
            if self.affine:
                out = (out - self.affine_bias) / (self.affine_weight + 1e-8)
            return out * self._stdev + self._mean
        raise ValueError(f"_RevIN mode must be norm|denorm, got {mode}")


class _MultiScaleSeasonMixing(nn.Module):
    """Bottom-up (fine -> coarse) season mixing."""

    def __init__(self, lengths: list[int]):
        super().__init__()
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lengths[i], lengths[i + 1]),
                nn.GELU(),
                nn.Linear(lengths[i + 1], lengths[i + 1]),
            )
            for i in range(len(lengths) - 1)
        ])

    def forward(self, season_list: list[torch.Tensor]) -> list[torch.Tensor]:
        # season_list[i] shape (B, d_model, T_i)
        out_high = season_list[0]
        outs = [out_high]
        for i, down in enumerate(self.downs):
            out_low = season_list[i + 1]
            out_low = out_low + down(out_high)
            outs.append(out_low)
            out_high = out_low
        return outs


class _MultiScaleTrendMixing(nn.Module):
    """Top-down (coarse -> fine) trend mixing."""

    def __init__(self, lengths: list[int]):
        super().__init__()
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lengths[i + 1], lengths[i]),
                nn.GELU(),
                nn.Linear(lengths[i], lengths[i]),
            )
            for i in range(len(lengths) - 1)
        ])

    def forward(self, trend_list: list[torch.Tensor]) -> list[torch.Tensor]:
        # iterate from coarsest (last) to finest (first)
        rev = list(reversed(trend_list))
        ups_rev = list(reversed(self.ups))
        out_low = rev[0]
        outs = [out_low]
        for i, up in enumerate(ups_rev):
            out_high = rev[i + 1]
            out_high = out_high + up(out_low)
            outs.append(out_high)
            out_low = out_high
        return list(reversed(outs))


class _PastDecomposableMixing(nn.Module):
    def __init__(self, d_model: int, d_ff: int, lengths: list[int], decomp_kernel: int, dropout: float):
        super().__init__()
        self.decomp = _SeriesDecomp(decomp_kernel)
        self.season_mix = _MultiScaleSeasonMixing(lengths)
        self.trend_mix = _MultiScaleTrendMixing(lengths)
        self.cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in lengths])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        # x_list[i] shape (B, T_i, d_model)
        seasons = []
        trends = []
        for x in x_list:
            s, t = self.decomp(x)
            seasons.append(s.transpose(1, 2))  # (B, d_model, T_i)
            trends.append(t.transpose(1, 2))
        seasons_mixed = self.season_mix(seasons)
        trends_mixed = self.trend_mix(trends)
        outs = []
        for i, (s_m, t_m, original, norm) in enumerate(zip(seasons_mixed, trends_mixed, x_list, self.norms)):
            mixed = (s_m + t_m).transpose(1, 2)  # back to (B, T_i, d_model)
            out = original + self.dropout(self.cross_layer(mixed))
            outs.append(norm(out))
        return outs


# =============================================================================
# Crossformer-specific: SegMerging + TwoStageAttentionLayer + ScaleBlock + Decoder
# =============================================================================


class _SegMerging(nn.Module):
    def __init__(self, d_model: int, win_size: int):
        super().__init__()
        self.win_size = win_size
        self.norm = nn.LayerNorm(win_size * d_model)
        self.linear = nn.Linear(win_size * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, seg_num, d_model)
        B, C, S, D = x.shape
        w = self.win_size
        pad = (w - S % w) % w
        if pad > 0:
            x = torch.cat([x, x[:, :, -pad:, :]], dim=2)
            S = x.shape[2]
        x = x.reshape(B, C, S // w, w, D).reshape(B, C, S // w, w * D)
        return self.linear(self.norm(x))


class _TwoStageAttentionLayer(nn.Module):
    """Crossformer Two-Stage Attention: cross-time then cross-dimension via router."""

    def __init__(self, seg_num: int, factor: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.time_attn = _AttentionLayer(_FullAttention(dropout), d_model, n_heads)
        self.dim_sender = _AttentionLayer(_FullAttention(dropout), d_model, n_heads)
        self.dim_receiver = _AttentionLayer(_FullAttention(dropout), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.mlp1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.mlp2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, seg_num, d_model)
        B, C, S, D = x.shape
        # Stage 1: cross-time per variable
        t_in = x.reshape(B * C, S, D)
        t = self.norm1(t_in + self.dropout(self.time_attn(t_in, t_in, t_in)))
        t = self.norm2(t + self.dropout(self.mlp1(t)))
        # Stage 2: cross-dimension via router
        d_in = t.reshape(B, C, S, D).permute(0, 2, 1, 3).reshape(B * S, C, D)
        router = self.router[:S].unsqueeze(0).expand(B, S, -1, -1).reshape(B * S, -1, D)
        buffer = self.dim_sender(router, d_in, d_in)
        d_out = d_in + self.dropout(self.dim_receiver(d_in, buffer, buffer))
        d_out = self.norm3(d_out)
        d_out = self.norm4(d_out + self.dropout(self.mlp2(d_out)))
        return d_out.reshape(B, S, C, D).permute(0, 2, 1, 3).contiguous()


class _CrossformerScaleBlock(nn.Module):
    def __init__(self, win_size: int, seg_num: int, factor: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.merge = _SegMerging(d_model, win_size) if win_size > 1 else nn.Identity()
        self.tsa = _TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.merge, _SegMerging):
            x = self.merge(x)
        return self.tsa(x)


class _CrossformerDecoderLayer(nn.Module):
    def __init__(self, seg_num_dec: int, factor: int, d_model: int, n_heads: int, d_ff: int, dropout: float, seg_len: int):
        super().__init__()
        self.self_attn = _TwoStageAttentionLayer(seg_num_dec, factor, d_model, n_heads, d_ff, dropout)
        self.cross_attn = _AttentionLayer(_FullAttention(dropout), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x_dec: torch.Tensor, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_dec: (B, C, S_dec, D); x_enc: (B, C, S_enc, D)
        B, C, S_dec, D = x_dec.shape
        S_enc = x_enc.shape[2]
        x_dec = self.self_attn(x_dec)
        d_in = x_dec.reshape(B * C, S_dec, D)
        e_in = x_enc.reshape(B * C, S_enc, D)
        h = self.norm1(d_in + self.dropout(self.cross_attn(d_in, e_in, e_in)))
        h = self.norm2(h + self.dropout(self.mlp(h)))
        layer_pred = self.linear_pred(h)  # (B*C, S_dec, seg_len)
        layer_pred = layer_pred.reshape(B, C * S_dec, -1)
        return h.reshape(B, C, S_dec, D), layer_pred


# =============================================================================
# Top-level forecaster classes (FactorRAG batch -> (B, pred_len) MBP forecast)
# =============================================================================


class TSLPatchTSTForecaster(nn.Module):
    """Channel-independent PatchTST with inlined RevIN, port of thuml/PatchTST."""

    def __init__(
        self,
        history_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        **_kwargs,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.target_channel = target_channel
        self.embed = _PatchEmbedding(d_model, patch_len, stride, padding=stride, dropout=dropout)
        self.encoder = _Encoder(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)],
            d_model=d_model,
        )
        # patch_num = (L - patch_len) / stride + 2  (one extra from ReplicationPad on right)
        patch_num = (history_len - patch_len) // stride + 2
        self.head = _FlattenHead(enc_in, d_model * patch_num, pred_len, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(batch)  # (B, L, C)
        # Inlined RevIN
        mean = x.mean(dim=1, keepdim=True).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False).detach()
        stdev = torch.sqrt(var + 1e-5)
        x_norm = (x - mean) / stdev
        # Channel-independent patching
        z, n_vars = self.embed(x_norm.transpose(1, 2))  # (B*C, patch_num, d_model)
        z = self.encoder(z)  # (B*C, patch_num, d_model)
        B = x.size(0)
        z = z.reshape(B, n_vars, z.size(-2), z.size(-1)).permute(0, 1, 3, 2)  # (B, C, d_model, patch_num)
        z = self.head(z)  # (B, C, pred_len)
        z = z.permute(0, 2, 1)  # (B, pred_len, C)
        out = z * stdev[:, 0:1, :] + mean[:, 0:1, :]
        return out[:, :, self.target_channel]


class TSLTimeMixerForecaster(nn.Module):
    """TimeMixer with RevIN + Past Decomposable Mixing + Future Multipredictor Mixing.

    Channel-independent (each channel processed as own length-T sequence with shared d_model).
    """

    def __init__(
        self,
        history_len: int,
        pred_len: int,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 2,
        down_sampling_window: int = 2,
        down_sampling_layers: int = 3,
        decomp_kernel: int = 25,
        dropout: float = 0.1,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        **_kwargs,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.target_channel = target_channel
        self.down_window = down_sampling_window
        self.n_scales = down_sampling_layers + 1
        # Scale lengths (TSL uses integer floor division)
        self.lengths = [history_len // (down_sampling_window ** i) for i in range(self.n_scales)]
        self.down_pool = nn.AvgPool1d(kernel_size=down_sampling_window, stride=down_sampling_window)
        self.embed = _DataEmbeddingNoPos(c_in=1, d_model=d_model, dropout=dropout)
        self.normalize_layers = nn.ModuleList([_RevIN(enc_in, affine=True) for _ in range(self.n_scales)])
        self.pdm_blocks = nn.ModuleList([
            _PastDecomposableMixing(d_model, d_ff, self.lengths, decomp_kernel, dropout)
            for _ in range(e_layers)
        ])
        self.predict_layers = nn.ModuleList([nn.Linear(self.lengths[i], pred_len) for i in range(self.n_scales)])
        self.projection_layer = nn.Linear(d_model, 1)  # channel-independent → 1

    def _multi_scale_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: (B, L, C). Downsample along time using AvgPool1d.
        outs = [x]
        cur = x.transpose(1, 2)  # (B, C, L)
        for _ in range(self.n_scales - 1):
            cur = self.down_pool(cur)
            outs.append(cur.transpose(1, 2))
        # Truncate/match the precomputed lengths
        outs = [o[:, : self.lengths[i], :] for i, o in enumerate(outs)]
        return outs

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(batch)  # (B, L, C)
        B, L, C = x.shape
        x_list = self._multi_scale_inputs(x)
        # RevIN per scale
        x_list = [self.normalize_layers[i](x_list[i], mode="norm") for i in range(self.n_scales)]
        # Channel independence: (B, T_i, C) -> (B*C, T_i, 1)
        x_list = [xi.permute(0, 2, 1).reshape(B * C, xi.shape[1], 1) for xi in x_list]
        # Embed each scale to (B*C, T_i, d_model)
        h_list = [self.embed(xi) for xi in x_list]
        # PDM stack
        for block in self.pdm_blocks:
            h_list = block(h_list)
        # FMM: each scale -> (B*C, d_model, T_i) -> Linear to pred_len -> project to 1
        outs = []
        for i, h in enumerate(h_list):
            h_perm = h.transpose(1, 2)  # (B*C, d_model, T_i)
            pred = self.predict_layers[i](h_perm)  # (B*C, d_model, pred_len)
            pred = pred.transpose(1, 2)  # (B*C, pred_len, d_model)
            pred = self.projection_layer(pred)  # (B*C, pred_len, 1)
            pred = pred.squeeze(-1).reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)
            outs.append(pred)
        out = torch.stack(outs, dim=-1).sum(-1)  # (B, pred_len, C)
        # Denormalize using the finest-scale RevIN
        out = self.normalize_layers[0](out, mode="denorm")
        return out[:, :, self.target_channel]


class TSLCrossformerForecaster(nn.Module):
    """Crossformer with DSW embedding + Two-Stage Attention + hierarchical decoder."""

    def __init__(
        self,
        history_len: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        seg_len: int = 12,
        win_size: int = 2,
        factor: int = 10,
        dropout: float = 0.1,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        **_kwargs,
    ):
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.target_channel = target_channel
        self.seg_len = seg_len
        self.e_layers = e_layers

        pad_in = (math.ceil(history_len / seg_len) * seg_len) - history_len
        pad_out_len = math.ceil(pred_len / seg_len) * seg_len
        self.pad_out_len = pad_out_len
        in_seg_num = math.ceil(history_len / seg_len)
        seg_nums = [in_seg_num]
        for layer in range(1, e_layers):
            seg_nums.append(math.ceil(seg_nums[-1] / win_size))
        out_seg_num_dec = pad_out_len // seg_len

        self.enc_embed = _PatchEmbedding(d_model, seg_len, seg_len, padding=pad_in, dropout=0.0)
        self.enc_pos = nn.Parameter(torch.randn(1, enc_in, in_seg_num, d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        self.scale_blocks = nn.ModuleList([
            _CrossformerScaleBlock(
                win_size=1 if l == 0 else win_size,
                seg_num=seg_nums[l],
                factor=factor,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for l in range(e_layers)
        ])

        self.dec_pos = nn.Parameter(torch.randn(1, enc_in, out_seg_num_dec, d_model))
        self.decoder_layers = nn.ModuleList([
            _CrossformerDecoderLayer(
                seg_num_dec=out_seg_num_dec,
                factor=factor,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                seg_len=seg_len,
            )
            for _ in range(e_layers + 1)
        ])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(batch)  # (B, L, C)
        B = x.size(0)
        C = self.enc_in
        # DSW embedding (non-overlapping seg_len patches, right-padded)
        z, n_vars = self.enc_embed(x.transpose(1, 2))  # (B*C, in_seg_num, d_model)
        z = z.reshape(B, C, z.size(-2), z.size(-1))  # (B, C, in_seg_num, d_model)
        z = self.pre_norm(z + self.enc_pos[:, :, : z.size(2), :])

        enc_outs = [z]
        for block in self.scale_blocks:
            z = block(z)
            enc_outs.append(z)

        # Decoder: e_layers + 1 layers, one per encoder output (including pre-encoded)
        dec_in = self.dec_pos.expand(B, -1, -1, -1)
        layer_predicts = []
        x_dec = dec_in
        for i, layer in enumerate(self.decoder_layers):
            x_dec, layer_pred = layer(x_dec, enc_outs[i])
            layer_predicts.append(layer_pred)
        # Sum predictions across decoder layers
        total_pred = torch.stack(layer_predicts, dim=0).sum(dim=0)  # (B, C*S_dec, seg_len)
        # Rearrange to (B, pad_out_len, C)
        S_dec = self.pad_out_len // self.seg_len
        total_pred = total_pred.reshape(B, C, S_dec, self.seg_len).permute(0, 2, 3, 1).reshape(B, self.pad_out_len, C)
        out = total_pred[:, -self.pred_len :, :]
        return out[:, :, self.target_channel]
