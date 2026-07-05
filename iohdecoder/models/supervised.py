"""Supervised comparison models for the IOHDecoder study.

All models consume the project batch dictionary and return a normalized future
MAP trajectory with shape ``(B, pred_len)``. The module keeps the comparison
suite deliberately boring at the interface level: every baseline can be trained
by the same engine, with the same split, target normalization, event threshold,
and clinical metrics.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from iohdecoder.models.clinical_baselines import CMAForecaster
from iohdecoder.models.hmf import HMFForecaster
from iohdecoder.models.tsl_baselines import (
    ENC_IN_FULL,
    TARGET_CHANNEL_IDX,
    TSLCrossformerForecaster,
    TSLPatchTSTForecaster,
    TSLTimeMixerForecaster,
    _batch_to_enc,
)


SUPERVISED_BASELINE_TYPES = {
    "arima",
    "arima_forecaster",
    "lstm",
    "dlinear",
    "informer",
    "gru",
    "cma",
    "patchtst",
    "tsl_patchtst",
    "timemixer",
    "tsl_timemixer",
    "crossformer",
    "tsl_crossformer",
    "transformer",
    "vanilla_transformer",
    "itransformer",
    "iTransformer".lower(),
    "crosslinear",
}


class _ForecastHead(nn.Module):
    def __init__(self, in_dim: int, pred_len: int, dropout: float):
        super().__init__()
        hidden = max(in_dim // 2, pred_len)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FeatureProjector(nn.Module):
    def __init__(
        self,
        dynamic_dim: int,
        pharma_dim: int,
        static_dim: int,
        hidden_dim: int,
        dropout: float,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        use_static_context: bool = True,
    ):
        super().__init__()
        self.dynamic_indices = tuple(int(index) for index in dynamic_indices) if dynamic_indices is not None else None
        self.use_pharma_features = bool(use_pharma_features)
        self.use_static_context = bool(use_static_context)
        self.hidden_dim = int(hidden_dim)
        self.seq_dim = dynamic_dim + (pharma_dim + pharma_dim if self.use_pharma_features else 0)
        self.input_proj = nn.Sequential(
            nn.Linear(self.seq_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def sequence(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_dyn = batch["x_dyn"].float()
        if self.dynamic_indices is not None:
            index = torch.as_tensor(self.dynamic_indices, device=x_dyn.device, dtype=torch.long)
            x_dyn = x_dyn.index_select(1, index)
        if self.use_pharma_features:
            pharma = torch.tanh(batch["x_pharma"].float() / 90.0) * batch["p_mask"].float()
            x = torch.cat([x_dyn, pharma, batch["p_mask"].float()], dim=1)
        else:
            x = x_dyn
        return x.transpose(1, 2).contiguous()

    def projected_sequence(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.input_proj(self.sequence(batch))

    def static(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_static_context:
            return torch.zeros(batch["x_dyn"].shape[0], self.hidden_dim, device=batch["x_dyn"].device)
        return self.static_proj(batch["x_stat"].float())


class GRUForecaster(nn.Module):
    """Recurrent supervised baseline used in the previous FlowRAG comparisons."""

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        dynamic_dim: int = 6,
        pharma_dim: int = 7,
        static_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        use_static_context: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.features = _FeatureProjector(
            dynamic_dim,
            pharma_dim,
            static_dim,
            hidden_dim,
            dropout,
            dynamic_indices=dynamic_indices,
            use_pharma_features=use_pharma_features,
            use_static_context=use_static_context,
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _ForecastHead(hidden_dim * 2, pred_len, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = self.features.projected_sequence(batch)
        _, h = self.gru(seq)
        z = torch.cat([h[-1], self.features.static(batch)], dim=-1)
        return self.head(z)


class LSTMForecaster(nn.Module):
    """LSTM sequence baseline with the same multi-modal feature packing as GRU."""

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        dynamic_dim: int = 6,
        pharma_dim: int = 7,
        static_dim: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        num_layers: int = 2,
        bidirectional: bool = False,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        use_static_context: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.features = _FeatureProjector(
            dynamic_dim,
            pharma_dim,
            static_dim,
            hidden_dim,
            dropout,
            dynamic_indices=dynamic_indices,
            use_pharma_features=use_pharma_features,
            use_static_context=use_static_context,
        )
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.head = _ForecastHead(hidden_dim * directions + hidden_dim, pred_len, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq = self.features.projected_sequence(batch)
        output, _ = self.lstm(seq)
        z = torch.cat([output[:, -1], self.features.static(batch)], dim=-1)
        return self.head(z)


class ARIMAForecaster(nn.Module):
    """Trainable global ARIMA-style baseline.

    The environment does not include statsmodels, so this module implements a
    lightweight AR(p, d=1) forecaster with shared coefficients learned on the
    training split. It is a classical linear temporal sanity check: use recent
    differenced MAP values, recursively extrapolate future differences, and add
    them back to the latest MAP anchor.
    """

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        target_channel: int = TARGET_CHANNEL_IDX,
        ar_order: int = 12,
        dropout: float = 0.0,
        **_kwargs,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.target_channel = int(target_channel)
        self.ar_order = max(1, int(ar_order))
        self.dropout = nn.Dropout(dropout)
        self.raw_ar = nn.Parameter(torch.zeros(self.ar_order))
        self.drift = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        y = batch["x_dyn"].float()[:, self.target_channel, :]
        if y.size(1) != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {y.size(1)}.")
        diff = y[:, 1:] - y[:, :-1]
        if diff.size(1) < self.ar_order:
            pad = diff[:, :1].expand(-1, self.ar_order - diff.size(1))
            history = torch.cat([pad, diff], dim=1)
        else:
            history = diff[:, -self.ar_order :]
        coeff = torch.tanh(self.raw_ar)
        current = y[:, -1]
        preds = []
        for _step in range(self.pred_len):
            next_diff = (self.dropout(history) * coeff.view(1, -1)).sum(dim=1) + self.drift[0]
            current = current + self.scale[0] * next_diff
            preds.append(current)
            history = torch.cat([history[:, 1:], next_diff.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)


class _SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int) -> torch.Tensor:
        return self.pe[:, :length]


def _instance_norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=1, keepdim=True).detach()
    stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False).detach() + 1e-5)
    return (x - mean) / stdev, mean, stdev


def _target_denorm(pred: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor, target_channel: int) -> torch.Tensor:
    return pred * stdev[:, 0, target_channel : target_channel + 1] + mean[:, 0, target_channel : target_channel + 1]


class TransformerForecaster(nn.Module):
    """Vanilla time-token Transformer baseline.

    This is intentionally the point-error-oriented control from the manuscript:
    a standard temporal encoder that pools history tokens before producing the
    whole future MAP horizon.
    """

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = True,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.target_channel = target_channel
        self.use_revin = use_revin
        self.dynamic_indices = tuple(int(index) for index in dynamic_indices) if dynamic_indices is not None else None
        self.use_pharma_features = bool(use_pharma_features)
        self.input_proj = nn.Linear(enc_in, d_model)
        self.pos = _SinusoidalPosition(d_model, max_len=max(history_len + 1, 4096))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = _ForecastHead(d_model * 2, pred_len, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(
            batch,
            dynamic_indices=self.dynamic_indices,
            use_pharma_features=self.use_pharma_features,
        )
        if self.use_revin:
            x, mean, stdev = _instance_norm(x)
        else:
            mean = torch.zeros_like(x[:, :1, :])
            stdev = torch.ones_like(x[:, :1, :])
        h = self.input_proj(x) + self.pos(x.size(1)).to(x.device)
        h = self.encoder(h)
        z = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)
        pred = self.head(z)
        return _target_denorm(pred, mean, stdev, self.target_channel)


class ITransformerForecaster(nn.Module):
    """Inverted Transformer baseline with variables as attention tokens."""

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = True,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.target_channel = target_channel
        self.use_revin = use_revin
        self.dynamic_indices = tuple(int(index) for index in dynamic_indices) if dynamic_indices is not None else None
        self.use_pharma_features = bool(use_pharma_features)
        self.value_embedding = nn.Linear(history_len, d_model)
        self.variable_embedding = nn.Parameter(torch.randn(1, enc_in, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, pred_len),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(
            batch,
            dynamic_indices=self.dynamic_indices,
            use_pharma_features=self.use_pharma_features,
        )
        if self.use_revin:
            x, mean, stdev = _instance_norm(x)
        else:
            mean = torch.zeros_like(x[:, :1, :])
            stdev = torch.ones_like(x[:, :1, :])
        # iTransformer treats variables as tokens and full history as token content.
        tokens = self.value_embedding(x.transpose(1, 2)) + self.variable_embedding[:, : x.size(2), :]
        encoded = self.encoder(tokens)
        target_token = encoded[:, self.target_channel, :]
        pred = self.head(target_token)
        return _target_denorm(pred, mean, stdev, self.target_channel)


class CrossLinearForecaster(nn.Module):
    """CrossLinear-style non-attention multivariate forecasting baseline.

    Each variable first receives a shared temporal linear projection from history
    to horizon, followed by lightweight cross-variable mixing at every future
    step. This keeps the baseline intentionally efficient and non-attentional,
    matching the role CrossLinear plays in the manuscript comparisons.
    """

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        d_model: int = 128,
        dropout: float = 0.1,
        n_blocks: int = 2,
        use_revin: bool = True,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.target_channel = target_channel
        self.use_revin = use_revin
        self.dynamic_indices = tuple(int(index) for index in dynamic_indices) if dynamic_indices is not None else None
        self.use_pharma_features = bool(use_pharma_features)
        self.temporal = nn.Linear(history_len, pred_len)
        blocks: list[nn.Module] = []
        for _ in range(n_blocks):
            blocks.extend(
                [
                    nn.LayerNorm(enc_in),
                    nn.Linear(enc_in, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, enc_in),
                ]
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(
            batch,
            dynamic_indices=self.dynamic_indices,
            use_pharma_features=self.use_pharma_features,
        )
        if self.use_revin:
            x, mean, stdev = _instance_norm(x)
        else:
            mean = torch.zeros_like(x[:, :1, :])
            stdev = torch.ones_like(x[:, :1, :])
        h = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        for i in range(0, len(self.blocks), 5):
            residual = h
            h = self.blocks[i](h)
            h = self.blocks[i + 1](h)
            h = self.blocks[i + 2](h)
            h = self.blocks[i + 3](h)
            h = self.blocks[i + 4](h)
            h = h + residual
        pred = h[:, :, self.target_channel]
        return _target_denorm(pred, mean, stdev, self.target_channel)


class DLinearForecaster(nn.Module):
    """DLinear baseline with moving-average decomposition and linear heads."""

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        moving_avg_kernel: int = 25,
        individual: bool = False,
        use_revin: bool = True,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.target_channel = int(target_channel)
        self.moving_avg_kernel = int(moving_avg_kernel)
        self.individual = bool(individual)
        self.use_revin = bool(use_revin)
        self.dynamic_indices = tuple(int(index) for index in dynamic_indices) if dynamic_indices is not None else None
        self.use_pharma_features = bool(use_pharma_features)
        if self.individual:
            self.seasonal_linear = nn.ModuleList([nn.Linear(history_len, pred_len) for _ in range(enc_in)])
            self.trend_linear = nn.ModuleList([nn.Linear(history_len, pred_len) for _ in range(enc_in)])
        else:
            self.seasonal_linear = nn.Linear(history_len, pred_len)
            self.trend_linear = nn.Linear(history_len, pred_len)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.moving_avg_kernel - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, self.moving_avg_kernel - 1 - pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        return F.avg_pool1d(padded.transpose(1, 2), kernel_size=self.moving_avg_kernel, stride=1).transpose(1, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(
            batch,
            dynamic_indices=self.dynamic_indices,
            use_pharma_features=self.use_pharma_features,
        )
        if self.use_revin:
            x, mean, stdev = _instance_norm(x)
        else:
            mean = torch.zeros_like(x[:, :1, :])
            stdev = torch.ones_like(x[:, :1, :])
        trend = self._moving_average(x)
        seasonal = x - trend
        seasonal = seasonal.transpose(1, 2)
        trend = trend.transpose(1, 2)
        if self.individual:
            outputs = []
            for channel in range(self.enc_in):
                outputs.append(self.seasonal_linear[channel](seasonal[:, channel]) + self.trend_linear[channel](trend[:, channel]))
            pred_all = torch.stack(outputs, dim=-1)
        else:
            pred_all = self.seasonal_linear(seasonal) + self.trend_linear(trend)
            pred_all = pred_all.transpose(1, 2)
        pred = pred_all[:, :, self.target_channel]
        return _target_denorm(pred, mean, stdev, self.target_channel)


class InformerForecaster(nn.Module):
    """Informer-style long-sequence forecaster with temporal distillation.

    This keeps the baseline compact for the IOH setting: token embedding,
    sinusoidal positions, Transformer encoder blocks, and Conv1d distillation
    between encoder layers. It is not a TSFM; it is a supervised long-horizon
    forecasting baseline trained from scratch on the same split.
    """

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        enc_in: int = ENC_IN_FULL,
        target_channel: int = TARGET_CHANNEL_IDX,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = True,
        distil: bool = True,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_pharma_features: bool = True,
        **_kwargs,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.target_channel = int(target_channel)
        self.use_revin = bool(use_revin)
        self.distil = bool(distil)
        self.dynamic_indices = tuple(int(index) for index in dynamic_indices) if dynamic_indices is not None else None
        self.use_pharma_features = bool(use_pharma_features)
        self.value_embedding = nn.Conv1d(enc_in, d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        self.position = _SinusoidalPosition(d_model, max_len=max(history_len + 1, 4096))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(e_layers)
            ]
        )
        self.distill_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode="circular"),
                    nn.BatchNorm1d(d_model),
                    nn.ELU(),
                    nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                )
                for _ in range(max(0, e_layers - 1))
            ]
        )
        self.head = _ForecastHead(d_model * 2, pred_len, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = _batch_to_enc(
            batch,
            dynamic_indices=self.dynamic_indices,
            use_pharma_features=self.use_pharma_features,
        )
        if self.use_revin:
            x, mean, stdev = _instance_norm(x)
        else:
            mean = torch.zeros_like(x[:, :1, :])
            stdev = torch.ones_like(x[:, :1, :])
        h = self.value_embedding(x.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(h + self.position(h.size(1)).to(h.device))
        for index, layer in enumerate(self.layers):
            h = layer(h)
            if self.distil and index < len(self.distill_layers):
                h = self.distill_layers[index](h.transpose(1, 2)).transpose(1, 2)
        z = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)
        pred = self.head(z)
        return _target_denorm(pred, mean, stdev, self.target_channel)


def build_supervised_baseline(model_cfg: dict[str, Any]) -> nn.Module:
    model_type = str(model_cfg.get("type", "gru")).lower()
    common = dict(model_cfg)
    common.pop("type", None)

    if model_type in {"arima", "arima_forecaster"}:
        return ARIMAForecaster(**common)
    if model_type == "lstm":
        return LSTMForecaster(**common)
    if model_type == "gru":
        return GRUForecaster(**common)
    if model_type == "cma":
        return CMAForecaster(**common)
    if model_type in {"patchtst", "tsl_patchtst"}:
        return TSLPatchTSTForecaster(**common)
    if model_type in {"timemixer", "tsl_timemixer"}:
        return TSLTimeMixerForecaster(**common)
    if model_type in {"crossformer", "tsl_crossformer"}:
        return TSLCrossformerForecaster(**common)
    if model_type in {"transformer", "vanilla_transformer"}:
        return TransformerForecaster(**common)
    if model_type == "informer":
        return InformerForecaster(**common)
    if model_type == "itransformer":
        return ITransformerForecaster(**common)
    if model_type == "dlinear":
        return DLinearForecaster(**common)
    if model_type == "crosslinear":
        return CrossLinearForecaster(**common)
    if model_type in {"hmf", "hmf_forecaster"}:
        return HMFForecaster(**common)
    raise ValueError(f"Unsupported supervised baseline model.type: {model_type}")
