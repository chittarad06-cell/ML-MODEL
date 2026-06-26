from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.ingestion import load_raw_light_curves, load_raw_light_curves_from_path
from data.preprocessing import ProcessedLightCurve, LightCurvePreprocessor
from data.splitting import create_or_load_splits, ensure_labels, split_train_into_train_val


@dataclass
class DatasetSpec:
    format: str
    path: str | None = None
    x_path: str | None = None
    y_path: str | None = None


class LightCurveDataset(Dataset):
    """Dataset wrapper for preprocessed light-curve arrays."""

    def __init__(self, x: Any, y: Any):
        self.x = [torch.as_tensor(sample, dtype=torch.float32) for sample in x] if _is_ragged(x) else torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y length mismatch: {len(self.x)} != {len(self.y)}")

    @classmethod
    def from_config(cls, spec: dict[str, Any], label_column: str = "label") -> "LightCurveDataset":
        fmt = spec["format"].lower()
        if fmt == "csv":
            df = pd.read_csv(spec["path"])
            y = df[label_column].to_numpy()
            x = df.drop(columns=[label_column]).to_numpy()
        elif fmt == "npz":
            data = np.load(spec["path"], allow_pickle=True)
            x = data[spec.get("x_key", "x")]
            y = data[spec.get("y_key", "y")]
        elif fmt == "npy":
            x = np.load(spec["x_path"], allow_pickle=True)
            y = np.load(spec["y_path"], allow_pickle=True)
        elif fmt in {"pt", "pth", "torch"}:
            obj = torch.load(spec["path"], map_location="cpu")
            if isinstance(obj, dict):
                x, y = obj["x"], obj["y"]
            else:
                x, y = obj
        else:
            raise ValueError(f"Unsupported dataset format: {fmt}")
        return cls(x, y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = self.x[idx] if isinstance(self.x, list) else self.x[idx]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 2 and x.shape[0] > x.shape[1]:
            x = x.transpose(0, 1)
        return x, self.y[idx], x.shape[-1]

    @classmethod
    def from_processed(cls, samples: list[ProcessedLightCurve]) -> "LightCurveDataset":
        x = [sample.x for sample in samples]
        y = [sample.y for sample in samples]
        return cls(x, y)


def collate_light_curves(batch: list[tuple[torch.Tensor, torch.Tensor, int]]) -> dict[str, torch.Tensor]:
    xs, ys, lengths = zip(*batch)
    max_len = max(lengths)
    channels = xs[0].shape[0]
    padded = torch.zeros(len(xs), channels, max_len, dtype=torch.float32)
    mask = torch.ones(len(xs), max_len, dtype=torch.bool)
    for i, x in enumerate(xs):
        seq_len = x.shape[-1]
        padded[i, :, :seq_len] = x
        mask[i, :seq_len] = False
    return {
        "x": padded,
        "y": torch.stack([y.reshape(()) for y in ys]).float(),
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
        "padding_mask": mask,
    }


def create_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    if data_cfg.get("raw_train_path") or data_cfg.get("raw_test_path"):
        return _create_raw_split_dataloaders(config)
    if data_cfg.get("dataset_path"):
        return _create_raw_dataloaders(config)

    label_column = data_cfg.get("label_column", "label")
    loaders = {}
    for split in ("train", "val", "test"):
        if split not in data_cfg or data_cfg[split] is None:
            continue
        dataset = LightCurveDataset.from_config(data_cfg[split], label_column)
        loaders[split] = DataLoader(
            dataset,
            batch_size=data_cfg.get("batch_size", 32),
            shuffle=split == "train",
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=data_cfg.get("pin_memory", True),
            collate_fn=collate_light_curves,
        )
    return loaders


def _processed_from_path(config: dict[str, Any], path: str) -> list[ProcessedLightCurve]:
    data_cfg = config["data"]
    raw_records = load_raw_light_curves_from_path(config, path)
    processed = LightCurvePreprocessor(config).transform_records(raw_records)
    return ensure_labels(processed, data_cfg.get("labels_path"), data_cfg.get("label_column", "label"))


def _create_raw_split_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    if not data_cfg.get("raw_train_path") or not data_cfg.get("raw_test_path"):
        raise ValueError("Both data.raw_train_path and data.raw_test_path are required for raw split mode.")

    train_samples = _processed_from_path(config, data_cfg["raw_train_path"])
    test_samples = _processed_from_path(config, data_cfg["raw_test_path"])
    if data_cfg.get("raw_val_path"):
        val_samples = _processed_from_path(config, data_cfg["raw_val_path"])
    else:
        train_samples, val_samples = split_train_into_train_val(train_samples, config)

    return _loaders_from_splits({"train": train_samples, "val": val_samples, "test": test_samples}, config)


def _create_raw_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    raw_records = load_raw_light_curves(config)
    processed = LightCurvePreprocessor(config).transform_records(raw_records)
    processed = ensure_labels(processed, data_cfg.get("labels_path"), data_cfg.get("label_column", "label"))
    splits = create_or_load_splits(processed, config)
    return _loaders_from_splits(splits, config)


def _loaders_from_splits(splits: dict[str, list[ProcessedLightCurve]], config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config["data"]

    loaders = {}
    for split, samples in splits.items():
        dataset = LightCurveDataset.from_processed(samples)
        loaders[split] = DataLoader(
            dataset,
            batch_size=data_cfg.get("batch_size", 32),
            shuffle=split == "train",
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=data_cfg.get("pin_memory", True),
            collate_fn=collate_light_curves,
        )
    return loaders


def _is_ragged(x: Any) -> bool:
    if isinstance(x, np.ndarray) and x.dtype == object:
        return True
    if isinstance(x, (list, tuple)) and x:
        try:
            first = len(x[0])
            return any(len(item) != first for item in x)
        except TypeError:
            return False
    return False
