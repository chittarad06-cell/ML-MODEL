from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from evaluation.metrics import compute_binary_metrics
from utils.model_info import count_parameters, model_size_mb


@torch.no_grad()
def evaluate_model(model, loader, config: dict, device: torch.device) -> dict:
    model.eval()
    y_true, y_prob, timings = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["padding_mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) / x.size(0))
        y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        y_true.extend(batch["y"].numpy().tolist())

    threshold = config["evaluation"].get("threshold", 0.5)
    metrics = compute_binary_metrics(y_true, y_prob, threshold)
    metrics["avg_inference_time_ms"] = float(np.mean(timings) * 1000)
    metrics["model_size_mb"] = model_size_mb(model)
    metrics["parameters"] = count_parameters(model)

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_curves(np.asarray(y_true), np.asarray(y_prob), output_dir)
    return metrics


def _plot_curves(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path) -> None:
    if len(np.unique(y_true)) < 2:
        return
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close()
