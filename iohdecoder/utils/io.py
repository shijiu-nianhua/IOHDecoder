from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_abs_path(project_root: Path, path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((project_root / path).resolve())


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_dataset_cache_paths(project_root: Path, data_config_path: str | Path) -> tuple[str, str]:
    config_path = Path(data_config_path)
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    config = load_yaml_config(config_path)
    paths = config.get("paths", {}) or {}
    return (
        project_abs_path(project_root, str(paths["preprocessed_data_path"])),
        project_abs_path(project_root, str(paths["normalization_stats_path"])),
    )

