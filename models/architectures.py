from __future__ import annotations

import math

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        x = self.act(self.net(x) + residual)
        return self.pool(x)


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int, channels: list[int], kernel_size: int, dropout: float):
        super().__init__()
        blocks = []
        in_channels = input_channels
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels, kernel_size, dropout))
            in_channels = out_channels
        self.net = nn.Sequential(*blocks)
        self.output_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1DClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        self.features = CNNFeatureExtractor(cfg["input_channels"], cfg["cnn_channels"], cfg["kernel_size"], cfg["dropout"])
        hidden = self.features.output_channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)


class CNNLSTMClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        self.features = CNNFeatureExtractor(cfg["input_channels"], cfg["cnn_channels"], cfg["kernel_size"], cfg["dropout"])
        self.lstm = nn.LSTM(
            input_size=self.features.output_channels,
            hidden_size=cfg["lstm_hidden"],
            num_layers=cfg["lstm_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=cfg["dropout"] if cfg["lstm_layers"] > 1 else 0.0,
        )
        hidden = cfg["lstm_hidden"] * 2
        self.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.features(x).transpose(1, 2)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 20000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CNNTransformerClassifier(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        self.features = CNNFeatureExtractor(cfg["input_channels"], cfg["cnn_channels"], cfg["kernel_size"], cfg["dropout"])
        self.proj = nn.Linear(self.features.output_channels, cfg["transformer_dim"])
        self.positional = PositionalEncoding(cfg["transformer_dim"])
        layer = nn.TransformerEncoderLayer(
            d_model=cfg["transformer_dim"],
            nhead=cfg["transformer_heads"],
            dim_feedforward=cfg["transformer_ff_dim"],
            dropout=cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg["transformer_layers"])
        self.norm = nn.LayerNorm(cfg["transformer_dim"])
        self.classifier = nn.Sequential(
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["transformer_dim"], cfg["transformer_dim"] // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["transformer_dim"] // 2, 1),
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.features(x).transpose(1, 2)
        x = self.positional(self.proj(x))
        reduced_mask = _downsample_mask(padding_mask, x.shape[1]) if padding_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=reduced_mask)
        if reduced_mask is not None:
            valid = (~reduced_mask).unsqueeze(-1).float()
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        else:
            pooled = x.mean(dim=1)
        return self.classifier(self.norm(pooled)).squeeze(-1)


def _downsample_mask(mask: torch.Tensor | None, target_len: int) -> torch.Tensor | None:
    if mask is None:
        return None
    idx = torch.linspace(0, mask.shape[1] - 1, target_len, device=mask.device).long()
    return mask.index_select(1, idx)
