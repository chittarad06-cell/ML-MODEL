from __future__ import annotations

import tempfile
from pathlib import Path

import torch


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        torch.save(model.state_dict(), path)
        return path.stat().st_size / (1024 * 1024)
    finally:
        path.unlink(missing_ok=True)
