from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ProcessedLightCurve:
    sample_id: str
    x: np.ndarray
    y: int | None
    time: np.ndarray
    source: str


class LightCurvePreprocessor:
    """Reusable preprocessing pipeline for raw astronomical light curves."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.pre_cfg = config["data"].get("preprocessing", {})
        self.sequence_length = int(config["data"].get("sequence_length", 2048))
        self.stride = int(config["data"].get("sequence_stride", self.sequence_length))

    def transform_records(self, records: list[dict[str, Any]]) -> list[ProcessedLightCurve]:
        processed = []
        for record in records:
            cleaned = self._clean_record(record)
            if cleaned is None:
                continue
            for seq_idx, (time, flux) in enumerate(self._make_sequences(cleaned["time"], cleaned["flux"])):
                processed.append(
                    ProcessedLightCurve(
                        sample_id=f"{record['sample_id']}::seq{seq_idx}",
                        x=flux.astype(np.float32)[None, :],
                        y=record.get("label"),
                        time=time.astype(np.float32),
                        source=record["source"],
                    )
                )
        return processed

    def _clean_record(self, record: dict[str, Any]) -> dict[str, np.ndarray] | None:
        time = np.asarray(record.get("time", np.arange(len(record["flux"]))), dtype=np.float64).reshape(-1)
        flux = np.asarray(record["flux"], dtype=np.float64).reshape(-1)
        n = min(len(time), len(flux))
        time, flux = time[:n], flux[:n]
        mask = np.isfinite(time) & np.isfinite(flux)

        quality = record.get("quality")
        if self.pre_cfg.get("remove_bad_quality", True) and quality is not None:
            q = np.asarray(quality).reshape(-1)[:n]
            mask &= q == self.pre_cfg.get("good_quality_value", 0)

        time, flux = time[mask], flux[mask]
        if len(flux) < self.pre_cfg.get("min_points", 16):
            return None

        order = np.argsort(time)
        time, flux = time[order], flux[order]
        _, unique_idx = np.unique(time, return_index=True)
        unique_idx = np.sort(unique_idx)
        time, flux = time[unique_idx], flux[unique_idx]

        flux = self._clip_outliers(flux)
        flux = self._normalize(flux)
        return {"time": time, "flux": flux}

    def _clip_outliers(self, flux: np.ndarray) -> np.ndarray:
        clip_cfg = self.pre_cfg.get("outlier_clip", {})
        if not clip_cfg.get("enabled", True):
            return flux
        sigma = float(clip_cfg.get("sigma", 5.0))
        median = np.nanmedian(flux)
        mad = np.nanmedian(np.abs(flux - median))
        scale = 1.4826 * mad if mad > 0 else np.nanstd(flux)
        if not np.isfinite(scale) or scale == 0:
            return flux
        return np.clip(flux, median - sigma * scale, median + sigma * scale)

    def _normalize(self, flux: np.ndarray) -> np.ndarray:
        method = self.pre_cfg.get("normalization", "standard").lower()
        if method == "standard":
            return (flux - flux.mean()) / (flux.std() + 1e-8)
        if method == "robust":
            median = np.median(flux)
            iqr = np.percentile(flux, 75) - np.percentile(flux, 25)
            return (flux - median) / (iqr + 1e-8)
        if method == "minmax":
            return (flux - flux.min()) / (flux.max() - flux.min() + 1e-8)
        if method in {"none", "null"}:
            return flux
        raise ValueError(f"Unsupported normalization method: {method}")

    def _make_sequences(self, time: np.ndarray, flux: np.ndarray):
        mode = self.pre_cfg.get("sequence_mode", "pad_or_window").lower()
        if len(flux) <= self.sequence_length:
            if self.pre_cfg.get("pad_short_sequences", True):
                pad = self.sequence_length - len(flux)
                strategy = self.pre_cfg.get("padding_strategy", "zero").lower()
                pad_value = 0.0 if strategy == "zero" else float(np.nanmedian(flux))
                flux = np.pad(flux, (0, pad), constant_values=pad_value)
                time = np.pad(time, (0, pad), constant_values=time[-1] if len(time) else 0)
            yield time[: self.sequence_length], flux[: self.sequence_length]
            return

        if mode in {"first", "truncate"}:
            yield time[: self.sequence_length], flux[: self.sequence_length]
            return

        for start in range(0, len(flux) - self.sequence_length + 1, self.stride):
            end = start + self.sequence_length
            yield time[start:end], flux[start:end]
