from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from data.preprocessing import ProcessedLightCurve


def ensure_labels(samples: list[ProcessedLightCurve], label_path: str | None = None, label_column: str = "label") -> list[ProcessedLightCurve]:
    """Attach external labels if provided and fail clearly when labels are missing."""

    if label_path:
        labels = _read_label_file(label_path, label_column)
        for sample in samples:
            key_options = [sample.sample_id, sample.source, sample.sample_id.split("::seq")[0]]
            for key in key_options:
                if key in labels:
                    sample.y = int(labels[key])
                    break

    missing = [s.sample_id for s in samples if s.y is None]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            "Supervised training requires labels, but labels were missing for "
            f"{len(missing)} sample(s), including: {preview}. Provide a label column in the dataset "
            "or configure data.labels_path with sample_id,label rows."
        )
    return samples


def create_or_load_splits(samples: list[ProcessedLightCurve], config: dict[str, Any]) -> dict[str, list[ProcessedLightCurve]]:
    split_cfg = config["data"].get("splits", {})
    split_path = Path(split_cfg.get("cache_path", "outputs/splits/splits.json"))
    if split_cfg.get("reuse_cached", True) and split_path.exists():
        return _apply_cached_splits(samples, split_path)

    ratios = split_cfg.get("ratios", {"train": 0.7, "val": 0.15, "test": 0.15})
    train_ratio, val_ratio, test_ratio = ratios["train"], ratios["val"], ratios["test"]
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios}")

    seed = config.get("seed", 42)
    labels = np.asarray([s.y for s in samples])
    indices = np.arange(len(samples))
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_idx, temp_idx = train_test_split(indices, train_size=train_ratio, random_state=seed, stratify=stratify)
    temp_labels = labels[temp_idx]
    rel_val = val_ratio / (val_ratio + test_ratio)
    temp_stratify = temp_labels if len(np.unique(temp_labels)) > 1 else None
    val_idx, test_idx = train_test_split(temp_idx, train_size=rel_val, random_state=seed, stratify=temp_stratify)

    splits = {
        "train": [samples[i] for i in train_idx],
        "val": [samples[i] for i in val_idx],
        "test": [samples[i] for i in test_idx],
    }
    _save_split_cache(splits, split_path)
    return splits


def split_train_into_train_val(samples: list[ProcessedLightCurve], config: dict[str, Any]) -> tuple[list[ProcessedLightCurve], list[ProcessedLightCurve]]:
    """Create validation data from an already provided training split."""

    split_cfg = config["data"].get("splits", {})
    val_ratio = float(split_cfg.get("val_from_train_ratio", split_cfg.get("ratios", {}).get("val", 0.15)))
    if val_ratio <= 0 or val_ratio >= 1:
        raise ValueError(f"val_from_train_ratio must be between 0 and 1, got {val_ratio}")

    labels = np.asarray([s.y for s in samples])
    indices = np.arange(len(samples))
    stratify = labels if len(np.unique(labels)) > 1 else None
    train_idx, val_idx = train_test_split(indices, test_size=val_ratio, random_state=config.get("seed", 42), stratify=stratify)
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


def _read_label_file(path: str, label_column: str) -> dict[str, int]:
    import pandas as pd

    df = pd.read_csv(path) if Path(path).suffix.lower() == ".csv" else pd.read_parquet(path)
    id_col = "sample_id" if "sample_id" in df.columns else df.columns[0]
    if label_column not in df.columns:
        raise ValueError(f"Label file must contain '{label_column}' column. Available: {list(df.columns)}")
    return {str(row[id_col]): int(row[label_column]) for _, row in df.iterrows()}


def _save_split_cache(splits: dict[str, list[ProcessedLightCurve]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {split: [sample.sample_id for sample in values] for split, values in splits.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_cached_splits(samples: list[ProcessedLightCurve], path: Path) -> dict[str, list[ProcessedLightCurve]]:
    by_id = {s.sample_id: s for s in samples}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {split: [by_id[sid] for sid in ids if sid in by_id] for split, ids in payload.items()}
