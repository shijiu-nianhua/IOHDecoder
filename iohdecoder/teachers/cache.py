"""Teacher prediction cache utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class SplitTeacherCache:
    """Load split-aligned teacher prediction arrays from disk."""

    def __init__(
        self,
        root: str | Path | None,
        pred_filename: str = "pred_q10.npy",
        splits: tuple[str, ...] = ("train", "val", "test"),
        mmap_mode: str | None = "r",
    ) -> None:
        self.root = Path(root).expanduser() if root else None
        self.pred_filename = pred_filename
        self.arrays: dict[str, np.ndarray] = {}
        self.index_maps: dict[str, dict[int, int] | None] = {}
        if self.root is None:
            return
        for split in splits:
            path = self._resolve_split_path(split)
            if path is not None and path.exists():
                self.arrays[split] = np.load(path, mmap_mode=mmap_mode)
                self.index_maps[split] = self._load_index_map(split, path)

    def _resolve_split_path(self, split: str) -> Path | None:
        if self.root is None:
            return None
        candidates = [
            self.root / split / self.pred_filename,
            self.root / f"{split}_{self.pred_filename}",
            self.root / f"{split}_pred_q10.npy",
            self.root / f"{split}_pred_mean.npy",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _load_metadata(split_dir: Path) -> dict:
        meta_path = split_dir / "metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_index_map(self, split: str, pred_path: Path) -> dict[int, int] | None:
        split_dir = pred_path.parent
        index_path = split_dir / "indices.npy"
        metadata = self._load_metadata(split_dir)
        if bool(metadata.get("aligned_by_dataset_index", False)):
            return None
        if not index_path.exists():
            return None

        indices = np.asarray(np.load(index_path), dtype=np.int64).reshape(-1)
        num_rows = int(self.arrays[split].shape[0])
        if len(indices) != num_rows:
            raise ValueError(
                f"Teacher cache split={split!r} has {num_rows} prediction rows "
                f"but {len(indices)} indices in {index_path}."
            )
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Teacher cache split={split!r} contains duplicate dataset indices in {index_path}.")
        return {int(dataset_index): row_index for row_index, dataset_index in enumerate(indices.tolist())}

    def available(self, split: str) -> bool:
        return split in self.arrays

    def require(self, split: str) -> None:
        if not self.available(split):
            root = "<unset>" if self.root is None else str(self.root)
            raise FileNotFoundError(f"Teacher cache for split={split!r} was not found under {root}.")

    def get(
        self,
        split: str,
        indices: torch.Tensor,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        if split not in self.arrays:
            return None
        idx = indices.detach().cpu().numpy().astype(np.int64)
        index_map = self.index_maps.get(split)
        if index_map is not None:
            flat_idx = idx.reshape(-1)
            try:
                row_idx = np.asarray([index_map[int(item)] for item in flat_idx], dtype=np.int64).reshape(idx.shape)
            except KeyError as exc:
                raise KeyError(
                    f"Teacher cache split={split!r} does not contain dataset index {int(exc.args[0])}."
                ) from exc
            idx = row_idx
        values = np.asarray(self.arrays[split][idx], dtype=np.float32)
        return torch.as_tensor(values, device=device, dtype=dtype)

    def summary(self) -> dict[str, tuple[int, ...]]:
        return {split: tuple(array.shape) for split, array in self.arrays.items()}
