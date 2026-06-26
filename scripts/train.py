from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data import create_dataloaders
from models import build_model
from training import Trainer
from utils.config import load_config
from utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["cnn1d", "cnn_lstm", "cnn_transformer"])
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model:
        config["model"]["name"] = args.model
    seed_everything(config.get("seed", 42))
    device_name = config.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    loaders = create_dataloaders(config)
    model = build_model(config)
    result = Trainer(model, loaders, config, device).fit()
    print(result)


if __name__ == "__main__":
    main()
