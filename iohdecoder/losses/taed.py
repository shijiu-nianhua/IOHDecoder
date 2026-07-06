"""MSE plus lower-tail event-space distillation for IOHDecoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TAEDConfig:
    threshold: float = 65.0
    duration_steps: int = 30
    point_temperature: float = 2.0
    softmin_beta: float = 12.0
    weight: float = 0.15
    teacher_is_normalized: bool = False
    regression: str = "mse"
    mode: str = "event_kl"


class TeacherAdvantagedEventDistillationLoss(nn.Module):
    """Regression loss with Chronos lower-tail Event-KL supervision.

    The kept paper path is:
      L = L_reg + lambda * KL(Pi_event(y_teacher_q10) || Pi_event(y_student)).

    The teacher trajectory is used only during training. Both teacher and
    student trajectories are projected into the same differentiable
    threshold-duration event space before the KL term is computed.
    """

    def __init__(
        self,
        threshold: float = 65.0,
        duration_steps: int = 30,
        point_temperature: float = 2.0,
        softmin_beta: float = 12.0,
        weight: float = 0.15,
        teacher_is_normalized: bool = False,
        regression: str = "mse",
        mode: str = "event_kl",
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.duration_steps = max(1, int(duration_steps))
        self.point_temperature = max(float(point_temperature), 1e-3)
        self.softmin_beta = max(float(softmin_beta), 1e-3)
        self.weight = float(weight)
        self.teacher_is_normalized = bool(teacher_is_normalized)
        self.regression = str(regression).lower()
        self.mode = str(mode).lower()
        self.target_mean = float(target_mean)
        self.target_std = max(float(target_std), 1e-6)

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

    def _regression_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.regression in {"none", "zero", "event_only", "event-only"}:
            return prediction.new_tensor(0.0)
        if self.regression == "mse":
            return F.mse_loss(prediction, target)
        if self.regression == "mae":
            return F.l1_loss(prediction, target)
        if self.regression in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(prediction, target)
        raise ValueError(f"Unsupported regression loss: {self.regression}")

    def soft_event_trace(self, sequence_raw: torch.Tensor) -> torch.Tensor:
        point_risk = torch.sigmoid((self.threshold - sequence_raw.float()) / self.point_temperature)
        horizon = point_risk.shape[1]
        if self.duration_steps <= 1 or horizon < self.duration_steps:
            return point_risk.clamp(1e-5, 1.0 - 1e-5)

        windows = point_risk.unfold(dimension=1, size=self.duration_steps, step=1)
        log_mean_exp = torch.logsumexp(-self.softmin_beta * windows, dim=-1)
        normalizer = torch.log(
            torch.tensor(float(self.duration_steps), device=sequence_raw.device, dtype=sequence_raw.dtype)
        )
        sustained_risk = -(log_mean_exp - normalizer) / self.softmin_beta
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
        del model_output

        prediction = prediction.float()
        target = target.float()
        reg_loss = self._regression_loss(prediction, target)

        zero = prediction.new_tensor(0.0)
        output = {
            "loss": reg_loss,
            "regression_loss": reg_loss.detach(),
            "taed_loss": zero,
            "teacher_loss": zero,
            "teacher_event_mae": zero,
            "student_event_mae": zero,
        }
        if teacher_prediction is None or self.weight <= 0.0:
            return output
        if self.mode not in {"event_kl", "event_kd", "event_no_selection"}:
            raise ValueError(f"Unsupported distillation mode in release: {self.mode}")

        teacher_prediction = teacher_prediction.float()
        event_horizon = min(prediction.shape[1], teacher_prediction.shape[1])
        prediction_for_event = prediction[:, :event_horizon]
        target_for_event = target[:, :event_horizon]
        teacher_for_event = teacher_prediction[:, :event_horizon]

        student_raw = self._to_raw(prediction_for_event, normalized=True)
        target_raw = self._to_raw(target_for_event, normalized=True)
        teacher_raw = self._to_raw(teacher_for_event, normalized=self.teacher_is_normalized)

        student_trace = self.soft_event_trace(student_raw)
        target_trace = self.soft_event_trace(target_raw)
        teacher_trace = self.soft_event_trace(teacher_raw)

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
