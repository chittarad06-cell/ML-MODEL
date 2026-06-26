from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, pos_weight: float | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", torch.tensor(pos_weight) if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config["training"]["loss"]
    name = loss_cfg.get("name", "focal").lower()
    pos_weight = loss_cfg.get("pos_weight")
    if name == "focal":
        return FocalLossWithLogits(loss_cfg.get("alpha", 0.75), loss_cfg.get("gamma", 2.0), pos_weight)
    if name in {"bce", "weighted_bce"}:
        weight = torch.tensor(pos_weight) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=weight)
    raise ValueError(f"Unsupported loss: {name}")
