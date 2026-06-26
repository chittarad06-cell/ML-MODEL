from __future__ import annotations

import torch


def gradient_saliency(model, x: torch.Tensor, padding_mask: torch.Tensor | None = None, top_k: int = 50) -> dict:
    was_training = model.training
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    logit = model(x, padding_mask)
    prob = torch.sigmoid(logit).mean()
    model.zero_grad(set_to_none=True)
    prob.backward()
    saliency = x.grad.detach().abs().mean(dim=1).squeeze(0)
    if padding_mask is not None:
        saliency = saliency.masked_fill(padding_mask.squeeze(0), 0)
    saliency = saliency / saliency.max().clamp_min(1e-8)
    values, indices = torch.topk(saliency, k=min(top_k, saliency.numel()))
    if was_training:
        model.train()
    return {
        "saliency": saliency.cpu().numpy().tolist(),
        "top_indices": indices.cpu().numpy().tolist(),
        "top_scores": values.cpu().numpy().tolist(),
    }
