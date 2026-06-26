from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation.metrics import compute_binary_metrics
from training.losses import build_loss
from utils.checkpointing import save_checkpoint


class Trainer:
    def __init__(self, model: nn.Module, loaders: dict, config: dict[str, Any], device: torch.device):
        self.model = model.to(device)
        self.loaders = loaders
        self.config = config
        self.device = device
        self.loss_fn = build_loss(config).to(device)
        train_cfg = config["training"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("mixed_precision", True) and device.type == "cuda")
        self.writer = SummaryWriter(train_cfg["tensorboard_dir"])
        self.history: list[dict[str, float]] = []

    def fit(self) -> dict[str, Any]:
        train_cfg = self.config["training"]
        best_val = float("inf")
        patience_left = train_cfg["early_stopping_patience"]
        ckpt_dir = Path(train_cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, train_cfg["epochs"] + 1):
            train_loss = self._run_epoch("train", epoch)
            val_loss, val_metrics = self.validate(epoch)
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.history.append(row)
            self._log_history(row)

            if val_loss < best_val:
                best_val = val_loss
                patience_left = train_cfg["early_stopping_patience"]
                save_checkpoint(ckpt_dir / "best_model.pt", self.model, self.optimizer, self.config, epoch, val_loss)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        save_checkpoint(ckpt_dir / "last_model.pt", self.model, self.optimizer, self.config, self.history[-1]["epoch"], self.history[-1]["val_loss"])
        history_path = ckpt_dir / "history.json"
        history_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        self.writer.close()
        return {"best_val_loss": best_val, "history": self.history, "checkpoint_dir": str(ckpt_dir)}

    def validate(self, epoch: int = 0) -> tuple[float, dict[str, float]]:
        loss, logits, targets = self._evaluate_loader(self.loaders["val"], "val")
        metrics = compute_binary_metrics(targets, torch.sigmoid(logits).numpy())
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{key}", value, epoch)
        return loss, {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    def _run_epoch(self, split: str, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        loader = self.loaders[split]
        for batch in tqdm(loader, desc=f"epoch {epoch} {split}", leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            mask = batch["padding_mask"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(x, mask)
                loss = self.loss_fn(logits, y)
            self.scaler.scale(loss).backward()
            if self.config["training"].get("grad_clip_norm"):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["grad_clip_norm"])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(loader.dataset)
        self.writer.add_scalar(f"{split}/loss", avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def _evaluate_loader(self, loader, split: str):
        self.model.eval()
        losses, logits_all, targets_all = [], [], []
        for batch in tqdm(loader, desc=split, leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            mask = batch["padding_mask"].to(self.device)
            logits = self.model(x, mask)
            loss = self.loss_fn(logits, y)
            losses.append(loss.item() * x.size(0))
            logits_all.append(logits.cpu())
            targets_all.append(y.cpu())
        return sum(losses) / len(loader.dataset), torch.cat(logits_all), torch.cat(targets_all).numpy()

    def _build_scheduler(self):
        name = self.config["training"].get("scheduler", "cosine").lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config["training"]["epochs"])
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=3)
        if name in {"none", "null"}:
            return None
        raise ValueError(f"Unsupported scheduler: {name}")

    def _log_history(self, row: dict[str, float]) -> None:
        for key, value in row.items():
            if key != "epoch":
                self.writer.add_scalar(f"history/{key}", value, row["epoch"])
