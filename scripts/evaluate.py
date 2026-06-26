from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from data import create_dataloaders
from evaluation import evaluate_model
from models import build_model
from utils.checkpointing import load_checkpoint
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device_name = config.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    loaders = create_dataloaders(config)
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    print(evaluate_model(model, loaders["test"], config, device))


if __name__ == "__main__":
    main()
