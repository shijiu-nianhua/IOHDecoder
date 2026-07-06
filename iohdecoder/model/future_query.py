"""HMF encoder plus future-query cross-attention decoder for IOH forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from iohdecoder.model.hmf_encoder import HMFEncoder


@dataclass(frozen=True)
class FutureQueryDecoderConfig:
    history_len: int = 450
    pred_len: int = 150
    dynamic_channels: int = 2
    static_dim: int = 4
    medicine_channels: int = 7
    map_channel: int = 0
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    patch_size: int = 15
    moving_avg_kernel: int = 25
    dropout: float = 0.1
    medication_decay_seconds: float = 180.0
    medication_clip_seconds: float = 900.0
    use_medication_mask: bool = True
    residual_prediction: bool = True
    dynamic_indices: tuple[int, ...] | None = None
    use_medication_features: bool = False
    use_static_context: bool = False


class FutureQueryIOHDecoder(nn.Module):
    """HMF historical encoder plus horizon-specific future-query decoder."""

    def __init__(
        self,
        history_len: int = 450,
        pred_len: int = 150,
        dynamic_channels: int = 2,
        static_dim: int = 4,
        medicine_channels: int = 7,
        map_channel: int = 0,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        patch_size: int = 15,
        moving_avg_kernel: int = 25,
        dropout: float = 0.1,
        medication_decay_seconds: float = 180.0,
        medication_clip_seconds: float = 900.0,
        use_medication_mask: bool = True,
        residual_prediction: bool = True,
        dynamic_indices: tuple[int, ...] | list[int] | None = None,
        use_medication_features: bool = False,
        use_static_context: bool = False,
    ) -> None:
        del medication_decay_seconds, medication_clip_seconds, use_medication_mask

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.dynamic_channels = int(dynamic_channels)
        self.static_dim = int(static_dim)
        self.medicine_channels = int(medicine_channels)
        self.map_channel = int(map_channel)
        self.residual_prediction = bool(residual_prediction)
        self.dynamic_indices = tuple(int(index) for index in dynamic_indices) if dynamic_indices is not None else None
        self.use_medication_features = bool(use_medication_features)
        self.use_static_context = bool(use_static_context)
        if self.use_medication_features:
            raise ValueError("The paper release uses the HMF MAP/SBP encoder and does not encode medication channels.")

        self.hmf_encoder = HMFEncoder(
            history_len=self.history_len,
            channels=self.dynamic_channels,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=encoder_layers,
            dropout=dropout,
            patch_size=patch_size,
            moving_avg_kernel=moving_avg_kernel,
        )
        self.static_projection = nn.Sequential(
            nn.Linear(self.static_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.future_query = nn.Embedding(self.pred_len, d_model)
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
            torch.linspace(0.0, 1.0, self.pred_len).view(self.pred_len, 1),
            persistent=False,
        )

    @classmethod
    def from_config(cls, cfg: FutureQueryDecoderConfig) -> "FutureQueryIOHDecoder":
        return cls(**cfg.__dict__)

    def _history_memory(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x_dyn = batch["x_dyn"].float()
        if x_dyn.shape[-1] != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {x_dyn.shape[-1]}.")
        if self.dynamic_indices is not None:
            index = torch.as_tensor(self.dynamic_indices, device=x_dyn.device, dtype=torch.long)
            x_dyn = x_dyn.index_select(1, index)
        if x_dyn.shape[1] != self.dynamic_channels:
            raise ValueError(f"Expected dynamic_channels={self.dynamic_channels}, got {x_dyn.shape[1]}.")
        memory = self.hmf_encoder(x_dyn.transpose(1, 2))
        if self.use_static_context:
            memory = memory + self.static_projection(batch["x_stat"].float()).unsqueeze(1)
        return memory

    def _future_queries(self, memory: torch.Tensor, x_stat: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]
        positions = torch.arange(self.pred_len, device=memory.device)
        query = self.future_query(positions).unsqueeze(0).expand(batch_size, -1, -1)
        horizon = self.horizon_projection(self.horizon_grid.to(memory.device)).unsqueeze(0)
        context = self.memory_to_query(memory.mean(dim=1)).unsqueeze(1)
        query = query + horizon + context
        if self.use_static_context:
            query = query + self.static_projection(x_stat.float()).unsqueeze(1)
        return query

    def forward(self, batch: dict[str, torch.Tensor], return_dict: bool = False):
        memory = self._history_memory(batch)
        decoded = self.decoder(self._future_queries(memory, batch["x_stat"]), memory)
        residual = self.output_head(decoded).squeeze(-1)

        if self.residual_prediction:
            anchor = batch["x_dyn"].float()[:, self.map_channel, -1].unsqueeze(1)
            prediction = anchor + residual
        else:
            prediction = residual

        if return_dict:
            output = {
                "prediction": prediction,
                "memory": memory,
                "decoded_queries": decoded,
            }
            return output
        return prediction
