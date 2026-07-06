"""Training and evaluation engine for IOHDecoder experiments."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from iohdecoder.data import get_ioh_loader
from iohdecoder.losses import TeacherAdvantagedEventDistillationLoss
from iohdecoder.metrics import ClinicalEvaluator
from iohdecoder.model import FutureQueryIOHDecoder
from iohdecoder.teachers import SplitTeacherCache
from iohdecoder.utils import load_yaml_config, project_abs_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    return torch.device(name if name.startswith("cuda") and torch.cuda.is_available() else "cpu")


def extract_prediction(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["prediction"]
    return output


def build_future_query_decoder(config: dict[str, Any]) -> FutureQueryIOHDecoder:
    model = config.get("model", {})
    dynamic_indices = model.get("dynamic_indices")
    return FutureQueryIOHDecoder(
        history_len=int(model.get("history_len", 450)),
        pred_len=int(model.get("pred_len", 150)),
        dynamic_channels=int(model.get("dynamic_channels", 6)),
        static_dim=int(model.get("static_dim", 4)),
        medicine_channels=int(model.get("medicine_channels", 7)),
        map_channel=int(model.get("map_channel", 1)),
        d_model=int(model.get("d_model", 128)),
        d_ff=int(model.get("d_ff", 256)),
        n_heads=int(model.get("n_heads", 8)),
        encoder_layers=int(model.get("encoder_layers", 2)),
        decoder_layers=int(model.get("decoder_layers", 2)),
        patch_size=int(model.get("patch_size", model.get("patch_kernel", 15))),
        moving_avg_kernel=int(model.get("moving_avg_kernel", 25)),
        dropout=float(model.get("dropout", 0.1)),
        medication_decay_seconds=float(model.get("medication_decay_seconds", 180.0)),
        medication_clip_seconds=float(model.get("medication_clip_seconds", 900.0)),
        use_medication_mask=bool(model.get("use_medication_mask", True)),
        residual_prediction=bool(model.get("residual_prediction", True)),
        dynamic_indices=dynamic_indices,
        use_medication_features=bool(model.get("use_medication_features", True)),
        use_static_context=bool(model.get("use_static_context", True)),
    )


def build_model(config: dict[str, Any]) -> nn.Module:
    model_type = str(config.get("model", {}).get("type", "future_query_decoder")).lower()
    if model_type in {"future_query", "future_query_decoder", "iohdecoder"}:
        return build_future_query_decoder(config)
    raise ValueError(f"Unsupported model.type: {model_type}. This release keeps only the IOHDecoder method.")


def prepare_config(config_path: str | Path) -> tuple[dict[str, Any], torch.device, str, Path]:
    config = load_yaml_config(config_path)
    set_seed(int(config.get("runtime", {}).get("seed", 42)))
    device = resolve_device(str(config.get("runtime", {}).get("device", "cuda:0")))
    paths = config.get("paths", {})
    data_path = project_abs_path(PROJECT_ROOT, str(paths["data_config"]))
    result_dir = Path(project_abs_path(PROJECT_ROOT, str(paths["results_dir"])))
    result_dir.mkdir(parents=True, exist_ok=True)
    return config, device, data_path, result_dir


def prepare(config_path: str | Path):
    config, device, data_path, result_dir = prepare_config(config_path)
    model = build_model(config).to(device)
    return config, device, data_path, result_dir, model


def make_loader(config: dict[str, Any], data_path: str, split: str):
    train = config.get("train", {})
    data = config.get("data", {})
    evaluation = config.get("eval", {})
    batch_size = (
        int(evaluation.get("batch_size", train.get("batch_size", 256)))
        if split == "test"
        else int(train.get("batch_size", 256))
    )
    return get_ioh_loader(
        data_path,
        split=split,
        batch_size=batch_size,
        normalize=bool(data.get("normalize", True)),
        num_workers=data.get("num_workers", train.get("num_workers", 0)),
        persistent_workers=bool(data.get("persistent_workers", True)),
        prefetch_factor=int(data.get("prefetch_factor", 2)),
        verbose=bool(data.get("verbose", False)),
        shuffle_train=bool(data.get("shuffle_train", True)),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_objective(
    config: dict[str, Any],
    *,
    target_mean: float,
    target_std: float,
) -> TeacherAdvantagedEventDistillationLoss:
    loss_cfg = config.get("loss", {})
    event_cfg = config.get("event", {})
    return TeacherAdvantagedEventDistillationLoss(
        threshold=float(event_cfg.get("threshold", loss_cfg.get("threshold", 65.0))),
        duration_steps=int(loss_cfg.get("duration_steps", event_cfg.get("duration_steps", 30))),
        point_temperature=float(loss_cfg.get("point_temperature", 2.0)),
        softmin_beta=float(loss_cfg.get("softmin_beta", 12.0)),
        weight=float(loss_cfg.get("weight", 0.0)),
        teacher_is_normalized=bool(loss_cfg.get("teacher_is_normalized", False)),
        regression=str(loss_cfg.get("regression", "mse")),
        mode=str(loss_cfg.get("mode", "event_kl")),
        target_mean=target_mean,
        target_std=target_std,
    )


def build_teacher_cache(
    config: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> SplitTeacherCache:
    teacher_cfg = config.get("teacher", {})
    root = teacher_cfg.get("cache_dir")
    if root:
        root = project_abs_path(PROJECT_ROOT, str(root))
    return SplitTeacherCache(
        root=root,
        pred_filename=str(teacher_cfg.get("pred_filename", "pred_q10.npy")),
        splits=splits,
        mmap_mode=teacher_cfg.get("mmap_mode", "r"),
    )


def _teacher_for_batch(
    cache: SplitTeacherCache,
    split: str,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor | None:
    if "index" not in batch:
        return None
    return cache.get(split, batch["index"], device=device, dtype=torch.float32)


def _meter_to_float(meter: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in meter.items() if values}


@torch.no_grad()
def evaluate_mse(model: nn.Module, loader, device: torch.device, max_batches: int = -1) -> float:
    model.eval()
    losses = []
    for index, batch in enumerate(tqdm(loader, desc="val", ncols=100)):
        if max_batches > 0 and index >= max_batches:
            break
        batch = move_batch(batch, device)
        prediction = extract_prediction(model(batch))
        losses.append(float(nn.functional.mse_loss(prediction, batch["y_true"].float()).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _save_checkpoint(path: Path, model: nn.Module, config: dict[str, Any], epoch: int, best_val_mse: float) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "best_val_mse": float(best_val_mse),
            "config": config,
        },
        path,
    )


def _load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def train_experiment(config_path: str | Path) -> Path:
    config, device, data_path, result_dir = prepare_config(config_path)
    train_config = config.get("train", {})
    train_loader = make_loader(config, data_path, "train")
    val_loader = make_loader(config, data_path, "val")
    model = build_model(config).to(device)
    objective = build_objective(
        config,
        target_mean=float(train_loader.dataset.target_mean),
        target_std=float(train_loader.dataset.target_std),
    )
    teacher_cfg = config.get("teacher", {})
    loss_weight = float(config.get("loss", {}).get("weight", 0.0))
    use_teacher = loss_weight > 0.0 or bool(teacher_cfg.get("required", False))
    teacher_cache = build_teacher_cache(config, splits=("train", "val")) if use_teacher else SplitTeacherCache(root=None)
    if bool(teacher_cfg.get("required", False)):
        teacher_cache.require("train")
        teacher_cache.require("val")
    if teacher_cache.summary():
        print(f"teacher_cache={teacher_cache.summary()}", flush=True)
    elif loss_weight > 0.0:
        print("teacher_cache=missing; TAED term will be inactive until cache is available.", flush=True)

    checkpoint = result_dir / "best_model.pt"
    with open(result_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
        betas=tuple(train_config.get("betas", [0.9, 0.99])),
    )
    epochs = int(train_config.get("epochs", 120))
    warmup = int(train_config.get("warmup_epochs", 0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup),
        eta_min=float(train_config.get("min_lr", 2e-5)),
    )
    best = float("inf")
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        if warmup and epoch <= warmup:
            learning_rate = float(train_config.get("learning_rate", 1e-3)) * epoch / warmup
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

        model.train()
        meters: dict[str, list[float]] = defaultdict(list)
        for index, batch in enumerate(tqdm(train_loader, desc=f"train {epoch}", ncols=100)):
            if int(train_config.get("max_batches", -1)) > 0 and index >= int(train_config["max_batches"]):
                break
            batch = move_batch(batch, device)
            model_output = model(batch)
            prediction = extract_prediction(model_output)
            teacher_prediction = _teacher_for_batch(teacher_cache, "train", batch, device) if use_teacher else None
            loss_dict = objective(prediction, batch["y_true"].float(), teacher_prediction, model_output=model_output)
            loss = loss_dict["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(train_config.get("grad_clip", 1.0)))
            optimizer.step()
            for key, value in loss_dict.items():
                meters[key].append(float(value.detach().cpu()))

        val_mse = evaluate_mse(model, val_loader, device, int(train_config.get("val_max_batches", -1)))
        if epoch > warmup:
            scheduler.step()
        meter_summary = _meter_to_float(meters)
        metric_blob = " ".join(f"{key}={value:.6f}" for key, value in meter_summary.items())
        print(f"epoch={epoch:03d} {metric_blob} val_mse={val_mse:.6f}", flush=True)
        if val_mse < best:
            best = val_mse
            bad_epochs = 0
            _save_checkpoint(checkpoint, model, config, epoch, best)
        else:
            bad_epochs += 1
            if epoch >= int(train_config.get("min_epochs", 30)) and bad_epochs >= int(train_config.get("patience", 30)):
                break
    return checkpoint


@torch.no_grad()
def evaluate_experiment(config_path: str | Path) -> dict[str, float]:
    config, device, data_path, result_dir, model = prepare(config_path)
    checkpoint = result_dir / "best_model.pt"
    _load_checkpoint(checkpoint, model, device)
    loader = make_loader(config, data_path, str(config.get("eval", {}).get("split", "test")))
    mean = float(loader.dataset.target_mean)
    std = max(float(loader.dataset.target_std), 1e-6)
    prediction = config.get("prediction", {})
    clip_min = float(prediction.get("clip_min", 20.0))
    clip_max = float(prediction.get("clip_max", 220.0))
    predictions, truths = [], []

    model.eval()
    for batch in tqdm(loader, desc="test", ncols=100):
        output = extract_prediction(model(move_batch(batch, device))).cpu().numpy()
        truth = batch["y_true"].numpy()
        predictions.append(np.clip(output * std + mean, clip_min, clip_max))
        truths.append(np.clip(truth * std + mean, clip_min, clip_max))

    predictions = np.concatenate(predictions).astype(np.float32)
    truths = np.concatenate(truths).astype(np.float32)
    prediction_dir = result_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    np.save(prediction_dir / "pred_mean.npy", predictions)
    np.save(prediction_dir / "ground_truth.npy", truths)

    event = config.get("event", {})
    evaluator = ClinicalEvaluator(
        threshold=float(event.get("threshold", 65.0)),
        duration_steps=int(event.get("duration_steps", 30)),
        prob_threshold=float(event.get("prob_threshold", 0.5)),
    )
    metrics = evaluator.calculate_metrics(list(predictions), None, list(truths))
    with open(result_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump({key: float(value) for key, value in metrics.items()}, handle, indent=2)
    return metrics
