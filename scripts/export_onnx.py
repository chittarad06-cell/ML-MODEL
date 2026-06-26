from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models import build_model
from utils.checkpointing import load_checkpoint
from utils.config import load_config
from utils.onnx_export import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/model.onnx")
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    export_onnx(model, args.output, args.sequence_length, config["model"].get("input_channels", 1))
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
