from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import create_dataloaders
from evaluation import evaluate_model
from models import build_model
from training import Trainer
from utils.config import load_config
from utils.seeding import seed_everything


ARCHITECTURES = ("cnn1d", "cnn_lstm", "cnn_transformer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare all supported exoplanet models.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/benchmark/leaderboard.json")
    args = parser.parse_args()

    base_config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for model_name in ARCHITECTURES:
        config = copy.deepcopy(base_config)
        config["model"]["name"] = model_name
        config["training"]["checkpoint_dir"] = f"outputs/checkpoints/{model_name}"
        config["training"]["tensorboard_dir"] = f"outputs/tensorboard/{model_name}"
        config["evaluation"]["output_dir"] = f"outputs/evaluation/{model_name}"

        seed_everything(config.get("seed", 42))
        loaders = create_dataloaders(config)
        model = build_model(config)
        start = time.perf_counter()
        Trainer(model, loaders, config, device).fit()
        training_time = time.perf_counter() - start
        metrics = evaluate_model(model, loaders["test"], config, device)
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "training_time_sec": training_time,
                "inference_time_ms": metrics["avg_inference_time_ms"],
                "model_size_mb": metrics["model_size_mb"],
                "parameters": metrics["parameters"],
            }
        )

    leaderboard = sorted(rows, key=lambda r: (r["f1"], r["roc_auc"], r["pr_auc"]), reverse=True)
    payload = {"best_model": leaderboard[0]["model"], "leaderboard": leaderboard}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
