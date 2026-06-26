from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data import create_dataloaders
from models import build_model
from training import Trainer
from utils.config import load_config
from utils.seeding import seed_everything


SEARCH_SPACE = {
    "model.dropout": [0.15, 0.25, 0.35],
    "training.learning_rate": [1e-4, 3e-4],
    "training.weight_decay": [1e-3, 1e-2],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Small practical grid search for hackathon experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-runs", type=int, default=6)
    parser.add_argument("--output", default="experiments/tuning_results.json")
    args = parser.parse_args()

    base_config = load_config(args.config)
    keys = list(SEARCH_SPACE)
    values = list(SEARCH_SPACE.values())
    results = []
    for run_id, combo in enumerate(itertools.product(*values), start=1):
        if run_id > args.max_runs:
            break
        config = copy.deepcopy(base_config)
        for key, value in zip(keys, combo):
            _set_nested(config, key, value)
        config["training"]["checkpoint_dir"] = f"outputs/checkpoints/tune_run_{run_id}"
        config["training"]["tensorboard_dir"] = f"outputs/tensorboard/tune_run_{run_id}"
        seed_everything(config.get("seed", 42))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaders = create_dataloaders(config)
        model = build_model(config)
        result = Trainer(model, loaders, config, device).fit()
        results.append({"run_id": run_id, "params": dict(zip(keys, combo)), "best_val_loss": result["best_val_loss"]})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(sorted(results, key=lambda x: x["best_val_loss"]), indent=2), encoding="utf-8")
    print(f"Wrote tuning results to {args.output}")


def _set_nested(config: dict, dotted_key: str, value) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


if __name__ == "__main__":
    main()
