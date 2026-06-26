from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from models import build_model
from utils.checkpointing import load_checkpoint
from utils.config import load_config
from utils.explainability import gradient_saliency


class ExoplanetPredictor:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "auto"):
        self.config = load_config(config_path)
        self.device = _resolve_device(device)
        self.model = build_model(self.config).to(self.device)
        load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.eval()

    def predict_file(self, input_path: str) -> dict:
        arr = _load_input(input_path)
        return self.predict_array(arr)

    def predict_array(self, light_curve, mc_samples: int | None = None) -> dict:
        x = _prepare_array(light_curve).to(self.device)
        mask = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.bool, device=self.device)
        cfg = self.config["inference"]
        mc_samples = mc_samples or cfg.get("mc_dropout_samples", 30)
        start = time.perf_counter()
        if cfg.get("tta", {}).get("enabled", False):
            probs = tta_mc_dropout_predict(
                self.model,
                x,
                mask,
                mc_samples=mc_samples,
                tta_samples=cfg["tta"].get("samples", 8),
                noise_std=cfg["tta"].get("noise_std", 0.005),
            )
        else:
            probs = mc_dropout_predict(self.model, x, mask, mc_samples)
        elapsed_ms = (time.perf_counter() - start) * 1000
        prob = float(probs.mean())
        uncertainty = float(probs.std())
        confidence = float(max(prob, 1 - prob) * (1 - min(uncertainty * 2, 1)))
        threshold = cfg.get("threshold", 0.5)
        explanation = gradient_saliency(self.model, x, mask, top_k=cfg.get("top_k_explanation_points", 50))
        transit_region = _predicted_transit_region(explanation["top_indices"])
        return {
            "prediction": "Planet" if prob >= threshold else "No Planet",
            "probability": prob,
            "confidence": confidence,
            "reliability": _reliability(confidence, uncertainty),
            "uncertainty": uncertainty,
            "model_used": self.config["model"]["name"],
            "saliency_values": explanation["saliency"],
            "most_important_time_indices": explanation["top_indices"],
            "predicted_transit_region": transit_region,
            "explanation": explanation,
            "inference_time_ms": elapsed_ms,
        }


class EnsemblePredictor:
    def __init__(self, predictors: Iterable[ExoplanetPredictor]):
        self.predictors = list(predictors)

    def predict_array(self, light_curve) -> dict:
        results = [p.predict_array(light_curve) for p in self.predictors]
        probabilities = np.asarray([r["probability"] for r in results])
        confidence = np.asarray([r["confidence"] for r in results])
        base = results[0]
        base["probability"] = float(probabilities.mean())
        base["uncertainty"] = float(probabilities.std())
        base["confidence"] = float(confidence.mean())
        base["prediction"] = "Planet" if base["probability"] >= self.predictors[0].config["inference"].get("threshold", 0.5) else "No Planet"
        base["reliability"] = _reliability(base["confidence"], base["uncertainty"])
        base["ensemble_members"] = len(results)
        return base


def mc_dropout_predict(model, x, mask, samples: int) -> np.ndarray:
    model.train()
    probs = []
    with torch.no_grad():
        for _ in range(samples):
            probs.append(torch.sigmoid(model(x, mask)).detach().cpu().numpy())
    model.eval()
    return np.asarray(probs).reshape(samples, -1)


def tta_mc_dropout_predict(model, x, mask, mc_samples: int, tta_samples: int, noise_std: float) -> np.ndarray:
    probs = []
    for _ in range(tta_samples):
        augmented = x + torch.randn_like(x) * noise_std
        probs.append(mc_dropout_predict(model, augmented, mask, mc_samples))
    return np.concatenate(probs, axis=0)


def _prepare_array(light_curve) -> torch.Tensor:
    arr = torch.as_tensor(light_curve, dtype=torch.float32)
    if arr.ndim == 1:
        arr = arr.unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 2:
        arr = arr.unsqueeze(0)
    return arr


def _load_input(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        return data["x"] if "x" in data else data[data.files[0]]
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        return obj["x"] if isinstance(obj, dict) else obj
    return np.loadtxt(path, delimiter=",")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _reliability(confidence: float, uncertainty: float) -> str:
    if confidence >= 0.8 and uncertainty <= 0.1:
        return "High"
    if confidence >= 0.65 and uncertainty <= 0.2:
        return "Medium"
    return "Low"


def _predicted_transit_region(indices: list[int]) -> dict[str, int] | None:
    if not indices:
        return None
    ordered = sorted(int(i) for i in indices)
    clusters = [[ordered[0]]]
    for idx in ordered[1:]:
        if idx - clusters[-1][-1] <= 3:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    best = max(clusters, key=len)
    return {"start_index": best[0], "end_index": best[-1], "center_index": int(round(sum(best) / len(best)))}
