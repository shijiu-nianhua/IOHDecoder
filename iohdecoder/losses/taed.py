"""Event-space TSFM distillation losses for IOH prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .soft_dtw_cuda import SoftDTW
except Exception:  # pragma: no cover - optional CUDA/numba dependency
    SoftDTW = None


@dataclass(frozen=True)
class TAEDConfig:
    threshold: float = 65.0
    duration_steps: int = 30
    point_temperature: float = 2.0
    softmin_beta: float = 12.0
    weight: float = 0.15
    margin: float = 0.0
    advantage_power: float = 1.0
    teacher_is_normalized: bool = False
    regression: str = "mse"
    mode: str = "event_kl"
    softdtw_weight: float = 0.05
    softdtw_gamma: float = 0.1
    softdtw_band: int = 15
    relation_lags: tuple[int, ...] = (1, 5, 15, 30)
    saliency_alpha: float = 2.0
    event_saliency_gamma: float = 1.0
    target_saliency_gamma: float = 1.0
    dynamic_regression: bool = False


class TeacherAdvantagedEventDistillationLoss(nn.Module):
    """Regression loss plus event-oriented TSFM supervision for IOH prediction."""

    def __init__(
        self,
        threshold: float = 65.0,
        duration_steps: int = 30,
        point_temperature: float = 2.0,
        softmin_beta: float = 12.0,
        weight: float = 0.15,
        margin: float = 0.0,
        advantage_power: float = 1.0,
        teacher_is_normalized: bool = False,
        regression: str = "mse",
        mode: str = "event_kl",
        softdtw_weight: float = 0.05,
        softdtw_gamma: float = 0.1,
        softdtw_band: int = 15,
        relation_lags: tuple[int, ...] | list[int] | str = (1, 5, 15, 30),
        saliency_alpha: float = 2.0,
        event_saliency_gamma: float = 1.0,
        target_saliency_gamma: float = 1.0,
        dynamic_regression: bool = False,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.duration_steps = max(1, int(duration_steps))
        self.point_temperature = max(float(point_temperature), 1e-3)
        self.softmin_beta = max(float(softmin_beta), 1e-3)
        self.weight = float(weight)
        self.margin = float(margin)
        self.advantage_power = max(float(advantage_power), 0.0)
        self.teacher_is_normalized = bool(teacher_is_normalized)
        self.regression = str(regression).lower()
        self.mode = str(mode).lower()
        self.softdtw_weight = max(float(softdtw_weight), 0.0)
        self.softdtw_gamma = max(float(softdtw_gamma), 1e-6)
        self.softdtw_band = int(softdtw_band)
        self.relation_lags = self._parse_relation_lags(relation_lags)
        self.saliency_alpha = max(float(saliency_alpha), 0.0)
        self.event_saliency_gamma = max(float(event_saliency_gamma), 0.0)
        self.target_saliency_gamma = max(float(target_saliency_gamma), 0.0)
        self.dynamic_regression = bool(dynamic_regression)
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-6)
        self._softdtw_cuda_loss: SoftDTW | None = None
        self._softdtw_cpu_loss: SoftDTW | None = None

    @staticmethod
    def _parse_relation_lags(values: tuple[int, ...] | list[int] | str) -> tuple[int, ...]:
        if isinstance(values, str):
            pieces = [piece.strip() for piece in values.split(",") if piece.strip()]
            parsed = [int(piece) for piece in pieces]
        else:
            parsed = [int(value) for value in values]
        return tuple(sorted({value for value in parsed if value > 0}))

    @classmethod
    def from_config(
        cls,
        cfg: TAEDConfig,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> "TeacherAdvantagedEventDistillationLoss":
        return cls(**cfg.__dict__, target_mean=target_mean, target_std=target_std)

    def _to_raw(self, values: torch.Tensor, *, normalized: bool) -> torch.Tensor:
        values = values.float()
        if normalized:
            return values * self.target_std + self.target_mean
        return values

    def _to_normalized(self, values: torch.Tensor, *, normalized: bool) -> torch.Tensor:
        values = values.float()
        if normalized:
            return values
        return (values - self.target_mean) / self.target_std

    def _regression_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.regression in {"none", "zero", "event_only", "event-only"}:
            return prediction.new_tensor(0.0)
        if self.regression == "mse":
            return F.mse_loss(prediction, target)
        if self.regression in {"mse_softdtw", "mse+softdtw", "mse_dtw"}:
            mse = F.mse_loss(prediction, target)
            dtw = self.soft_dtw_loss(prediction, target)
            return mse + self.softdtw_weight * dtw
        if self.regression == "softdtw":
            return self.soft_dtw_loss(prediction, target)
        if self.regression == "mae":
            return F.l1_loss(prediction, target)
        if self.regression in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(prediction, target)
        raise ValueError(f"Unsupported regression loss: {self.regression}")

    def soft_dtw_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Differentiable DTW on normalized MAP trajectories.

        CUDA execution uses the local copy of the Maghoumi soft-DTW kernel; the
        PyTorch wavefront implementation is retained as a dependency-free
        fallback.
        """
        x = prediction.float()
        y = target.float()
        if SoftDTW is not None:
            if x.is_cuda:
                if self._softdtw_cuda_loss is None:
                    self._softdtw_cuda_loss = SoftDTW(
                        use_cuda=True,
                        gamma=self.softdtw_gamma,
                        normalize=False,
                        bandwidth=self.softdtw_band if self.softdtw_band > 0 else None,
                    )
                return (self._softdtw_cuda_loss(x.unsqueeze(-1), y.unsqueeze(-1)) / float(x.shape[1])).mean()
            if self._softdtw_cpu_loss is None:
                self._softdtw_cpu_loss = SoftDTW(
                    use_cuda=False,
                    gamma=self.softdtw_gamma,
                    normalize=False,
                    bandwidth=self.softdtw_band if self.softdtw_band > 0 else None,
                )
            return (self._softdtw_cpu_loss(x.unsqueeze(-1), y.unsqueeze(-1)) / float(x.shape[1])).mean()

        batch_size, x_len = x.shape
        y_len = y.shape[1]
        distances = (x.unsqueeze(2) - y.unsqueeze(1)).pow(2)

        inf = torch.finfo(distances.dtype).max / 10.0
        dp = distances.new_full((batch_size, x_len + 1, y_len + 1), inf)
        dp[:, 0, 0] = 0.0

        band = self.softdtw_band
        if band <= 0:
            band = max(x_len, y_len)
        band = max(band, abs(x_len - y_len))
        gamma = self.softdtw_gamma

        for diag in range(2, x_len + y_len + 1):
            i_start = max(1, diag - y_len)
            i_end = min(x_len, diag - 1)
            if i_start > i_end:
                continue
            i_idx = torch.arange(i_start, i_end + 1, device=distances.device)
            j_idx = diag - i_idx
            keep = (j_idx >= 1) & (j_idx <= y_len) & ((i_idx - j_idx).abs() <= band)
            if not bool(keep.any()):
                continue
            i_idx = i_idx[keep]
            j_idx = j_idx[keep]

            prev = torch.stack(
                (
                    dp[:, i_idx - 1, j_idx],
                    dp[:, i_idx, j_idx - 1],
                    dp[:, i_idx - 1, j_idx - 1],
                ),
                dim=-1,
            )
            soft_min = -gamma * torch.logsumexp(-prev / gamma, dim=-1)
            dp[:, i_idx, j_idx] = distances[:, i_idx - 1, j_idx - 1] + soft_min

        return (dp[:, x_len, y_len] / float(max(x_len, y_len))).mean()

    @staticmethod
    def _normalize_saliency(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp_min(0.0)
        mean = values.mean(dim=1, keepdim=True)
        return values / mean.clamp_min(1e-6)

    @staticmethod
    def _point_change_strength(values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] <= 1:
            return torch.zeros_like(values)
        delta = (values[:, 1:] - values[:, :-1]).abs()
        left = torch.cat([delta[:, :1], delta], dim=1)
        right = torch.cat([delta, delta[:, -1:]], dim=1)
        return torch.maximum(left, right)

    def _point_saliency(self, target_raw: torch.Tensor, teacher_raw: torch.Tensor) -> torch.Tensor:
        target_risk = torch.sigmoid((self.threshold - target_raw.float()) / self.point_temperature)
        teacher_risk = torch.sigmoid((self.threshold - teacher_raw.float()) / self.point_temperature)

        target_saliency = self._point_change_strength(target_raw) + self.target_saliency_gamma * self._point_change_strength(
            target_risk
        )
        target_saliency = target_saliency + target_risk
        target_saliency = self._normalize_saliency(target_saliency)

        teacher_saliency = self._point_change_strength(teacher_raw) + self.event_saliency_gamma * self._point_change_strength(
            teacher_risk
        )
        teacher_saliency = teacher_saliency + teacher_risk
        teacher_saliency = self._normalize_saliency(teacher_saliency)

        saliency = 1.0 + target_saliency * (1.0 + self.saliency_alpha * teacher_saliency)
        return self._normalize_saliency(saliency).detach()

    def dynamic_regression_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        target_raw: torch.Tensor,
        teacher_raw: torch.Tensor,
    ) -> torch.Tensor:
        saliency = self._point_saliency(target_raw, teacher_raw)
        if self.regression == "mse":
            point_error = (prediction - target).pow(2)
        elif self.regression in {"mse_softdtw", "mse+softdtw", "mse_dtw", "softdtw"}:
            point_error = (prediction - target).pow(2)
        elif self.regression == "mae":
            point_error = (prediction - target).abs()
        elif self.regression in {"smooth_l1", "huber"}:
            point_error = F.smooth_l1_loss(prediction, target, reduction="none")
        else:
            raise ValueError(f"Unsupported regression loss: {self.regression}")
        return (saliency * point_error).sum() / saliency.sum().clamp_min(1e-6)

    def _relation_saliency(self, target_raw: torch.Tensor, teacher_raw: torch.Tensor, lag: int) -> torch.Tensor:
        target_delta = (target_raw[:, lag:] - target_raw[:, :-lag]).abs()
        target_risk = torch.sigmoid((self.threshold - target_raw.float()) / self.point_temperature)
        target_risk_delta = (target_risk[:, lag:] - target_risk[:, :-lag]).abs()
        target_saliency = target_delta + self.target_saliency_gamma * target_risk_delta
        target_saliency = self._normalize_saliency(target_saliency)

        teacher_delta = (teacher_raw[:, lag:] - teacher_raw[:, :-lag]).abs()
        teacher_risk = torch.sigmoid((self.threshold - teacher_raw.float()) / self.point_temperature)
        risk_delta = (teacher_risk[:, lag:] - teacher_risk[:, :-lag]).abs()
        teacher_saliency = teacher_delta + self.event_saliency_gamma * risk_delta
        teacher_saliency = self._normalize_saliency(teacher_saliency)

        saliency = target_saliency * (1.0 + self.saliency_alpha * teacher_saliency)
        return self._normalize_saliency(saliency).detach()

    def temporal_relation_loss(
        self,
        student_raw: torch.Tensor,
        target_raw: torch.Tensor,
        teacher_raw: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        horizon = int(student_raw.shape[1])
        for lag in self.relation_lags:
            if lag >= horizon:
                continue
            student_relation = student_raw[:, lag:] - student_raw[:, :-lag]
            target_relation = target_raw[:, lag:] - target_raw[:, :-lag]
            saliency = self._relation_saliency(target_raw, teacher_raw, lag)
            relation_error = F.smooth_l1_loss(student_relation, target_relation, reduction="none")
            losses.append((saliency * relation_error).sum() / saliency.sum().clamp_min(1e-6))
        if not losses:
            return student_raw.new_tensor(0.0)
        return torch.stack(losses).mean()

    def soft_event_trace(self, sequence_raw: torch.Tensor) -> torch.Tensor:
        point_risk = torch.sigmoid((self.threshold - sequence_raw.float()) / self.point_temperature)
        horizon = point_risk.shape[1]
        if self.duration_steps <= 1 or horizon < self.duration_steps:
            return point_risk.clamp(1e-5, 1.0 - 1e-5)

        windows = point_risk.unfold(dimension=1, size=self.duration_steps, step=1)
        log_mean_exp = torch.logsumexp(-self.softmin_beta * windows, dim=-1)
        log_mean_exp = log_mean_exp - torch.log(
            torch.tensor(float(self.duration_steps), device=sequence_raw.device, dtype=sequence_raw.dtype)
        )
        sustained_risk = -log_mean_exp / self.softmin_beta
        return sustained_risk.clamp(1e-5, 1.0 - 1e-5)

    @staticmethod
    def _bernoulli_kl(teacher_prob: torch.Tensor, student_prob: torch.Tensor) -> torch.Tensor:
        eps = 1e-5
        teacher_prob = teacher_prob.clamp(eps, 1.0 - eps)
        student_prob = student_prob.clamp(eps, 1.0 - eps)
        return teacher_prob * (teacher_prob.log() - student_prob.log()) + (1.0 - teacher_prob) * (
            (1.0 - teacher_prob).log() - (1.0 - student_prob).log()
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        teacher_prediction: torch.Tensor | None = None,
        model_output: torch.Tensor | dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        prediction = prediction.float()
        target = target.float()
        reg_loss = self._regression_loss(prediction, target)

        zero = prediction.new_tensor(0.0)
        output = {
            "loss": reg_loss,
            "regression_loss": reg_loss.detach(),
            "taed_loss": zero,
            "teacher_loss": zero,
            "relation_loss": zero,
            "advantage_mass": zero,
            "teacher_event_mae": zero,
            "student_event_mae": zero,
        }
        if teacher_prediction is None or self.weight <= 0.0:
            return output

        if teacher_prediction.shape[1] != prediction.shape[1]:
            teacher_prediction = teacher_prediction[:, : prediction.shape[1]]

        if self.mode in {"sequence_mse", "forecast_mse", "point_mse", "naive_sequence"}:
            teacher_target = self._to_normalized(teacher_prediction, normalized=self.teacher_is_normalized)
            teacher_loss = F.mse_loss(prediction, teacher_target.detach())
            total = reg_loss + self.weight * teacher_loss
            output.update(
                {
                    "loss": total,
                    "taed_loss": teacher_loss.detach(),
                    "teacher_loss": teacher_loss.detach(),
                }
            )
            return output

        student_raw = self._to_raw(prediction, normalized=True)
        target_raw = self._to_raw(target, normalized=True)
        teacher_raw = self._to_raw(teacher_prediction, normalized=self.teacher_is_normalized)

        if self.mode in {"selective_point_mse", "advantaged_point", "point_advantage"}:
            student_point_error = (student_raw - target_raw).abs()
            teacher_point_error = (teacher_raw - target_raw).abs()
            advantage = (student_point_error - teacher_point_error - self.margin).clamp_min(0.0).detach()
            if self.advantage_power != 1.0:
                advantage = advantage.pow(self.advantage_power)
            teacher_target = self._to_normalized(teacher_prediction, normalized=self.teacher_is_normalized)
            point_loss = (prediction - teacher_target.detach()).pow(2)
            selective_loss = (advantage * point_loss).sum() / advantage.sum().clamp_min(1e-6)
            total = reg_loss + self.weight * selective_loss
            output.update(
                {
                    "loss": total,
                    "taed_loss": selective_loss.detach(),
                    "teacher_loss": selective_loss.detach(),
                    "advantage_mass": advantage.mean().detach(),
                    "teacher_event_mae": teacher_point_error.mean().detach(),
                    "student_event_mae": student_point_error.mean().detach(),
                }
            )
            return output

        student_trace = self.soft_event_trace(student_raw)
        target_trace = self.soft_event_trace(target_raw)
        teacher_trace = self.soft_event_trace(teacher_raw)

        if self.mode in {"tarl", "temporal_relation", "teacher_guided_relation"}:
            if self.dynamic_regression:
                reg_loss = self.dynamic_regression_loss(prediction, target, target_raw, teacher_raw)
            relation_loss = self.temporal_relation_loss(student_raw, target_raw, teacher_raw)
            total = reg_loss + self.weight * relation_loss
            output.update(
                {
                    "loss": total,
                    "regression_loss": reg_loss.detach(),
                    "taed_loss": relation_loss.detach(),
                    "teacher_loss": relation_loss.detach(),
                    "relation_loss": relation_loss.detach(),
                }
            )
            return output

        if self.mode in {"event_kl", "event_kd", "event_no_selection"}:
            event_loss = self._bernoulli_kl(teacher_trace.detach(), student_trace).mean()
            total = reg_loss + self.weight * event_loss
            teacher_error = (teacher_trace - target_trace).abs()
            student_error = (student_trace - target_trace).abs()
            output.update(
                {
                    "loss": total,
                    "taed_loss": event_loss.detach(),
                    "teacher_loss": event_loss.detach(),
                    "teacher_event_mae": teacher_error.mean().detach(),
                    "student_event_mae": student_error.mean().detach(),
                }
            )
            return output

        if self.mode not in {"advantaged_event", "taed", "event_advantage"}:
            raise ValueError(f"Unsupported distillation mode: {self.mode}")

        student_error = (student_trace - target_trace).abs()
        teacher_error = (teacher_trace - target_trace).abs()
        advantage = (student_error - teacher_error - self.margin).clamp_min(0.0).detach()
        if self.advantage_power != 1.0:
            advantage = advantage.pow(self.advantage_power)

        kl = self._bernoulli_kl(teacher_trace.detach(), student_trace)
        denom = advantage.sum().clamp_min(1e-6)
        taed_loss = (advantage * kl).sum() / denom
        total = reg_loss + self.weight * taed_loss

        return {
            "loss": total,
            "regression_loss": reg_loss.detach(),
            "taed_loss": taed_loss.detach(),
            "teacher_loss": taed_loss.detach(),
            "advantage_mass": advantage.mean().detach(),
            "teacher_event_mae": teacher_error.mean().detach(),
            "student_event_mae": student_error.mean().detach(),
        }
