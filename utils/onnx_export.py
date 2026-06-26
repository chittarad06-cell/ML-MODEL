from __future__ import annotations

from pathlib import Path

import torch


def export_onnx(model, output_path: str, sequence_length: int, input_channels: int = 1, opset: int = 17) -> None:
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(1, input_channels, sequence_length, device=device)
    mask = torch.zeros(1, sequence_length, dtype=torch.bool, device=device)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (x, mask),
        output_path,
        input_names=["light_curve", "padding_mask"],
        output_names=["logit"],
        dynamic_axes={"light_curve": {0: "batch", 2: "sequence"}, "padding_mask": {0: "batch", 1: "sequence"}, "logit": {0: "batch"}},
        opset_version=opset,
    )
