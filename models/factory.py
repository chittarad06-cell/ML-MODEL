from __future__ import annotations

from .architectures import CNN1DClassifier, CNNLSTMClassifier, CNNTransformerClassifier


def build_model(config: dict):
    name = config["model"]["name"].lower()
    if name == "cnn1d":
        return CNN1DClassifier(config)
    if name == "cnn_lstm":
        return CNNLSTMClassifier(config)
    if name == "cnn_transformer":
        return CNNTransformerClassifier(config)
    raise ValueError(f"Unknown model architecture: {name}")
