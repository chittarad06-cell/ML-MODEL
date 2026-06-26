from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


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
