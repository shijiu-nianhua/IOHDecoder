import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from iohdecoder.utils.io import load_yaml_config, resolve_dataset_cache_paths

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class IOHDataset(Dataset):
    def __init__(self, pkl_path, config_path, normalize=True, verbose=True, progress_file=None):
        self.config = load_yaml_config(config_path)
        self.normalize = bool(normalize)
        self.verbose = bool(verbose)
        self.progress_file = progress_file if progress_file is not None else sys.stdout

        data_params = self.config["data_params"]
        feature_defs = self.config["feature_definitions"]
        column_names = self.config["column_names"]

        self.history_len = int(data_params["history_len"])
        self.sample_interval_seconds = float(data_params.get("sample_interval_seconds", 2.0))
        self.dynamic_features = feature_defs["dynamic_features"]
        self.static_features = feature_defs["static_features"]
        self.medicine_features = feature_defs["medicine_ST_features"]
        self.target_col = column_names["target_col"]
        self.main_seq_col = column_names.get("main_seq_col", "Solar8000/ART_MBP")

        _, stats_path = resolve_dataset_cache_paths(Path(PROJECT_ROOT), config_path)
        self.norm_stats = self._load_stats(stats_path)

        dyn_stats = self.norm_stats.get("dynamic", {})
        tgt_stats = dyn_stats.get(self.target_col)
        if not tgt_stats:
            tgt_stats = dyn_stats.get(self.main_seq_col, {"mean": 0.0, "std": 1.0})

        self.target_mean = float(tgt_stats["mean"])
        self.target_std = float(tgt_stats["std"])
        if self.target_std < 1e-6:
            self.target_std = 1.0

        if self.verbose:
            print(f"🔧 [Dataset] Target Normalization Info: Mean={self.target_mean:.2f}, Std={self.target_std:.2f}")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"❌ 找不到数据文件: {pkl_path}")

        self.df = pd.read_pickle(pkl_path)
        self._build_cached_arrays()
        self.df = None
        self._print_length_adjust_summary()

    def _load_stats(self, path):
        if os.path.exists(path):
            return load_yaml_config(path)
        if self.verbose:
            print(f"⚠️ 警告: 统计文件未找到 {path}，将跳过归一化 (使用原始值)。")
        return {"dynamic": {}, "static": {}}

    @staticmethod
    def _parse_serialized_value(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val

    def _record_length_adjustment(self, scalar_mode, original_len, expected_len):
        if not hasattr(self, "_length_adjustments"):
            self._length_adjustments = {}
        key = (str(scalar_mode), int(original_len), int(expected_len))
        self._length_adjustments[key] = self._length_adjustments.get(key, 0) + 1

    def _resize_sequence_array(self, arr, expected_len, scalar_mode="repeat"):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        current_len = int(arr.shape[0])
        expected_len = int(expected_len)
        if current_len == expected_len:
            return arr

        self._record_length_adjustment(scalar_mode, current_len, expected_len)

        if current_len == 0:
            return np.zeros(expected_len, dtype=np.float32)

        if current_len > expected_len:
            # Exact-ratio downsampling: use block mean so 450->30 and 150->10 preserve window averages.
            if current_len % expected_len == 0:
                factor = current_len // expected_len
                return arr.reshape(expected_len, factor).mean(axis=1, dtype=np.float32)

            src_idx = np.arange(current_len, dtype=np.float32)
            dst_idx = np.linspace(0.0, float(current_len - 1), expected_len, dtype=np.float32)
            return np.interp(dst_idx, src_idx, arr).astype(np.float32, copy=False)

        pad_len = expected_len - current_len
        if scalar_mode == "pharma" and current_len > 0:
            step = np.float32(self.sample_interval_seconds)
            start = np.float32(arr[-1]) + step
            pad = start + np.arange(pad_len, dtype=np.float32) * step
        else:
            pad_value = np.float32(arr[-1])
            pad = np.full(pad_len, pad_value, dtype=np.float32)
        return np.concatenate([arr, pad], axis=0).astype(np.float32, copy=False)

    def _print_length_adjust_summary(self):
        if not self.verbose or not hasattr(self, "_length_adjustments") or not self._length_adjustments:
            return
        print("⚠️ [Dataset] Sequence length mismatches detected and auto-adjusted:")
        for (scalar_mode, original_len, expected_len), count in sorted(self._length_adjustments.items()):
            print(
                f"   - mode={scalar_mode}, original_len={original_len}, "
                f"expected_len={expected_len}, count={count}"
            )

    def _to_fixed_length_array(self, val, expected_len, scalar_mode="repeat"):
        val = self._parse_serialized_value(val)
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 0:
            scalar = float(arr.item())
            if scalar_mode == "pharma":
                if scalar == float("-inf"):
                    return np.full(expected_len, float("-inf"), dtype=np.float32)
                step = np.float32(self.sample_interval_seconds)
                return np.float32(scalar) + np.arange(expected_len, dtype=np.float32) * step
            if scalar_mode == "repeat":
                return np.full(expected_len, scalar, dtype=np.float32)
            raise ValueError(f"Unsupported scalar_mode: {scalar_mode}")

        return self._resize_sequence_array(arr, expected_len, scalar_mode=scalar_mode)

    def _build_dynamic_cache(self):
        dyn_stats = self.norm_stats.get("dynamic", {})
        dyn_blocks = []
        for col in tqdm(
            self.dynamic_features,
            desc="Parse dynamic features",
            leave=False,
            disable=not self.verbose,
            file=self.progress_file,
        ):
            col_values = self.df[col].tolist()
            col_arr = np.stack(
                [
                    self._to_fixed_length_array(val, self.history_len, scalar_mode="repeat")
                    for val in tqdm(
                        col_values,
                        desc=f"{col.split('/')[-1]}",
                        leave=False,
                        disable=not self.verbose,
                        file=self.progress_file,
                    )
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            col_stat = dyn_stats.get(col, {"mean": 0.0, "std": 1.0})
            mean = float(col_stat["mean"])
            std = float(col_stat["std"])
            if std < 1e-6:
                std = 1.0
            if self.normalize:
                col_arr = (col_arr - mean) / std
            dyn_blocks.append(col_arr)
        self.x_dyn = np.stack(dyn_blocks, axis=1).astype(np.float32, copy=False)

    def _build_static_cache(self):
        stat_arr = self.df[self.static_features].to_numpy(dtype=np.float32, copy=True)
        if self.normalize:
            means = []
            stds = []
            static_stats = self.norm_stats.get("static", {})
            for col in self.static_features:
                col_stat = static_stats.get(col, {"mean": 0.0, "std": 1.0})
                means.append(float(col_stat["mean"]))
                std = float(col_stat["std"])
                stds.append(1.0 if std < 1e-6 else std)
            stat_arr = (stat_arr - np.asarray(means, dtype=np.float32)) / np.asarray(stds, dtype=np.float32)
        self.x_stat = stat_arr.astype(np.float32, copy=False)

    def _build_pharma_cache(self):
        pharma_blocks = []
        for col in tqdm(
            self.medicine_features,
            desc="Parse medicine features",
            leave=False,
            disable=not self.verbose,
            file=self.progress_file,
        ):
            col_values = self.df[col].tolist()
            col_arr = np.stack(
                [
                    self._to_fixed_length_array(val, self.history_len, scalar_mode="pharma")
                    for val in tqdm(
                        col_values,
                        desc=f"{col.split('/')[-1]}",
                        leave=False,
                        disable=not self.verbose,
                        file=self.progress_file,
                    )
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            pharma_blocks.append(col_arr)

        pharma_raw = np.stack(pharma_blocks, axis=1).astype(np.float32, copy=False)
        valid_mask = pharma_raw != float("-inf")
        self._event_flags = np.any(
            valid_mask & (pharma_raw >= 0.0) & (pharma_raw <= 90.0),
            axis=(1, 2),
        ).astype(np.int64)
        self.p_mask = valid_mask.astype(np.float32, copy=False)
        self.x_pharma = np.where(valid_mask, pharma_raw, 0.0).astype(np.float32, copy=False)

    def _build_target_cache(self):
        target_values = self.df[self.target_col].tolist()
        target_arr = np.stack(
            [
                self._to_fixed_length_array(val, self.config["data_params"]["future_len"], scalar_mode="repeat")
                for val in tqdm(
                    target_values,
                    desc="target_value",
                    leave=False,
                    disable=not self.verbose,
                    file=self.progress_file,
                )
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        if self.normalize:
            target_arr = (target_arr - self.target_mean) / self.target_std
        self.y_true = target_arr.astype(np.float32, copy=False)

    def _build_cached_arrays(self):
        if self.verbose:
            print(f"⚙️ [Dataset] Pre-parsing cached arrays from {len(self.df)} samples...")
        with tqdm(
            total=4,
            desc="Dataset cache build",
            leave=False,
            disable=not self.verbose,
            file=self.progress_file,
        ) as pbar:
            self._build_dynamic_cache()
            pbar.update(1)
            self._build_static_cache()
            pbar.update(1)
            self._build_pharma_cache()
            pbar.update(1)
            self._build_target_cache()
            pbar.update(1)
        self.num_samples = int(self.y_true.shape[0])

    @property
    def event_flags(self):
        return self._event_flags

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "x_dyn": torch.from_numpy(self.x_dyn[idx]),
            "x_stat": torch.from_numpy(self.x_stat[idx]),
            "x_pharma": torch.from_numpy(self.x_pharma[idx]),
            "p_mask": torch.from_numpy(self.p_mask[idx]),
            "y_true": torch.from_numpy(self.y_true[idx]),
        }

def _resolve_num_workers(num_workers, cpu_util_target=0.9, reserve_cpu_cores=1):
    """
    Resolve num_workers from user input.

    Supported values:
    - int >= 0: use as-is
    - int < 0: auto by cpu_util_target (default 90%)
    - str:
        - "auto", "auto90", "90", "90%": target ~90% CPU
        - "full", "max", "100", "100%": target ~100% CPU
        - numeric string: parsed as int
    """
    cpu_count = max(1, int(os.cpu_count() or 1))

    if isinstance(num_workers, str):
        mode = num_workers.strip().lower()
        if mode in {"auto", "auto90", "90", "90%"}:
            target_workers = int(round(cpu_count * float(cpu_util_target)))
            target_workers = max(1, target_workers - int(max(0, reserve_cpu_cores)))
            return min(target_workers, cpu_count)
        if mode in {"full", "max", "100", "100%"}:
            return cpu_count
        try:
            num_workers = int(mode)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported num_workers mode: {num_workers}. "
                "Use int, 'auto90', or 'full'."
            ) from exc

    num_workers = int(num_workers)
    if num_workers < 0:
        target_workers = int(round(cpu_count * float(cpu_util_target)))
        target_workers = max(1, target_workers - int(max(0, reserve_cpu_cores)))
        return min(target_workers, cpu_count)
    return num_workers


def _create_ioh_loader(
    config_path,
    mode,
    batch_size,
    normalize,
    resolved_workers,
    persistent_workers,
    prefetch_factor,
    verbose,
    shuffle_train=True,
):
    cache_dir, _ = resolve_dataset_cache_paths(Path(PROJECT_ROOT), config_path)
    pkl_name = f"{mode}_data_final_with_stats.pkl"
    pkl_path = os.path.join(cache_dir, pkl_name)

    if verbose:
        print(f"📦 正在准备 {mode} 数据集: {pkl_path}")
    dataset = IOHDataset(pkl_path, config_path, normalize=normalize, verbose=verbose)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train" and bool(shuffle_train)),
        num_workers=int(resolved_workers),
        pin_memory=True,
    )
    if int(resolved_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_kwargs)


def get_ioh_loaders(
    config_path,
    batch_size=32,
    normalize=True,
    num_workers=0,
    persistent_workers=True,
    prefetch_factor=2,
    cpu_util_target=0.9,
    reserve_cpu_cores=1,
    verbose=True,
    shuffle_train=True,
):
    resolved_workers = _resolve_num_workers(
        num_workers=num_workers,
        cpu_util_target=cpu_util_target,
        reserve_cpu_cores=reserve_cpu_cores,
    )
    if verbose:
        print(
            f"🧵 DataLoader workers: requested={num_workers}, resolved={resolved_workers}, "
            f"cpu_count={os.cpu_count()}"
        )

    return (
        _create_ioh_loader(
            config_path,
            "train",
            batch_size,
            normalize,
            resolved_workers,
            persistent_workers,
            prefetch_factor,
            verbose,
            shuffle_train,
        ),
        _create_ioh_loader(
            config_path,
            "val",
            batch_size,
            normalize,
            resolved_workers,
            persistent_workers,
            prefetch_factor,
            verbose,
            shuffle_train,
        ),
        _create_ioh_loader(
            config_path,
            "test",
            batch_size,
            normalize,
            resolved_workers,
            persistent_workers,
            prefetch_factor,
            verbose,
            shuffle_train,
        ),
    )


def get_ioh_loader(
    config_path,
    split="test",
    batch_size=32,
    normalize=True,
    num_workers=0,
    persistent_workers=True,
    prefetch_factor=2,
    cpu_util_target=0.9,
    reserve_cpu_cores=1,
    verbose=True,
    shuffle_train=True,
):
    split = str(split).lower()
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}. Expected one of train/val/test.")

    resolved_workers = _resolve_num_workers(
        num_workers=num_workers,
        cpu_util_target=cpu_util_target,
        reserve_cpu_cores=reserve_cpu_cores,
    )
    if verbose:
        print(
            f"🧵 DataLoader workers: requested={num_workers}, resolved={resolved_workers}, "
            f"cpu_count={os.cpu_count()}"
        )
    return _create_ioh_loader(
        config_path=config_path,
        mode=split,
        batch_size=batch_size,
        normalize=normalize,
        resolved_workers=resolved_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        verbose=verbose,
        shuffle_train=shuffle_train,
    )
