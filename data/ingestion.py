from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".npz", ".npy", ".pt", ".pth"}
COMMON_TIME_COLUMNS = ("TIME", "time", "Time", "BTJD", "BJD")
COMMON_FLUX_COLUMNS = ("PDCSAP_FLUX", "SAP_FLUX", "flux", "Flux", "FLUX")
COMMON_QUALITY_COLUMNS = ("QUALITY", "quality", "Quality")


def discover_supported_files(path: str | Path) -> list[Path]:
    """Return all supported dataset files from a file or directory path."""

    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported dataset file extension: {root.suffix}")
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No supported dataset files found under: {root}")
    return sorted(files)


def load_supported_file(path: str | Path) -> Any:
    """Load a supported raw or preprocessed dataset file by extension."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".npz":
        return np.load(path, allow_pickle=True)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported dataset file extension: {suffix}")


def detect_column(columns, configured: str | None, candidates: tuple[str, ...], role: str) -> str | None:
    """Resolve a configured or commonly used astronomical column name."""

    columns = list(columns)
    if configured:
        if configured in columns:
            return configured
        raise ValueError(f"Configured {role} column '{configured}' was not found. Available columns: {columns}")
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def dataframe_to_light_curves(df: pd.DataFrame, config: dict[str, Any], source_id: str) -> list[dict[str, Any]]:
    """Convert a tabular astronomical file into one or more raw light-curve records."""

    data_cfg = config["data"]
    time_col = detect_column(df.columns, data_cfg.get("time_column"), COMMON_TIME_COLUMNS, "time")
    flux_col = detect_column(df.columns, data_cfg.get("flux_column"), COMMON_FLUX_COLUMNS, "flux")
    quality_col = detect_column(df.columns, data_cfg.get("quality_column"), COMMON_QUALITY_COLUMNS, "quality")
    label_col = data_cfg.get("label_column", "label") if data_cfg.get("label_column", "label") in df.columns else None
    sample_id_col = data_cfg.get("sample_id_column") if data_cfg.get("sample_id_column") in df.columns else None

    if flux_col is None:
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {label_col, quality_col}]
        if not numeric:
            raise ValueError(f"Could not detect a flux column in {source_id}. Configure data.flux_column.")
        flux_col = numeric[-1]

    groups = [(source_id, df)] if sample_id_col is None else list(df.groupby(sample_id_col, sort=False))
    records = []
    for group_id, group in groups:
        label = None
        if label_col:
            labels = group[label_col].dropna().unique()
            if len(labels) > 0:
                label = int(labels[0])
        records.append(
            {
                "sample_id": str(group_id),
                "source": source_id,
                "time": group[time_col].to_numpy() if time_col else np.arange(len(group)),
                "flux": group[flux_col].to_numpy(),
                "quality": group[quality_col].to_numpy() if quality_col else None,
                "label": label,
            }
        )
    return records


def array_to_light_curves(obj: Any, config: dict[str, Any], source_id: str) -> list[dict[str, Any]]:
    """Convert array-like files into raw light-curve records."""

    data_cfg = config["data"]
    label = _label_from_mapping(obj, data_cfg.get("label_column", "label"))
    if isinstance(obj, np.lib.npyio.NpzFile):
        x_key = data_cfg.get("x_key", "x") if data_cfg.get("x_key", "x") in obj.files else obj.files[0]
        arr = obj[x_key]
        labels = obj[data_cfg.get("y_key", "y")] if data_cfg.get("y_key", "y") in obj.files else None
        return _records_from_array(arr, labels, source_id)
    if isinstance(obj, dict):
        if "x" in obj:
            return _records_from_array(obj["x"], obj.get("y"), source_id)
        if "flux" in obj:
            return [
                {
                    "sample_id": source_id,
                    "source": source_id,
                    "time": np.asarray(obj.get("time", np.arange(len(obj["flux"])))),
                    "flux": np.asarray(obj["flux"]),
                    "quality": obj.get("quality"),
                    "label": label,
                }
            ]
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return _records_from_array(obj[0], obj[1], source_id)
    return _records_from_array(obj, None, source_id)


def load_raw_light_curves_from_path(config: dict[str, Any], path: str | Path) -> list[dict[str, Any]]:
    """Load raw light-curve records from one file or directory."""

    records = []
    for file_path in discover_supported_files(path):
        obj = load_supported_file(file_path)
        if isinstance(obj, pd.DataFrame):
            records.extend(dataframe_to_light_curves(obj, config, str(file_path)))
        else:
            records.extend(array_to_light_curves(obj, config, str(file_path)))
    return records


def load_raw_light_curves(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load all configured raw dataset files into raw light-curve records."""

    path = config["data"].get("dataset_path")
    if not path:
        raise ValueError("data.dataset_path is required for raw dataset mode.")
    return load_raw_light_curves_from_path(config, path)


def _records_from_array(arr: Any, labels: Any, source_id: str) -> list[dict[str, Any]]:
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=object if getattr(arr, "dtype", None) == object else None)
    labels_arr = np.asarray(labels) if labels is not None else None
    if arr.ndim == 1 or arr.dtype == object:
        samples = list(arr) if arr.dtype == object else [arr]
    elif arr.ndim == 2:
        samples = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 3:
        samples = [arr[i].squeeze() for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Unsupported array shape for {source_id}: {arr.shape}")

    records = []
    for idx, flux in enumerate(samples):
        label = None if labels_arr is None else int(labels_arr[idx])
        sample_id = source_id if len(samples) == 1 else f"{source_id}::{idx}"
        flux = np.asarray(flux, dtype=np.float32).squeeze()
        records.append(
            {
                "sample_id": sample_id,
                "source": source_id,
                "time": np.arange(flux.shape[-1]),
                "flux": flux,
                "quality": None,
                "label": label,
            }
        )
    return records


def _label_from_mapping(obj: Any, label_column: str) -> int | None:
    if isinstance(obj, dict) and label_column in obj:
        value = obj[label_column]
        if np.asarray(value).size == 1:
            return int(np.asarray(value).reshape(-1)[0])
    return None
