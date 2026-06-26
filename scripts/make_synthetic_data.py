from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_split(samples: int, seq_len: int, positive_rate: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, noise, size=(samples, seq_len)).astype("float32")
    y = (rng.random(samples) < positive_rate).astype("float32")
    time = np.arange(seq_len)
    for i, label in enumerate(y):
        if label == 1:
            center = rng.integers(seq_len // 5, seq_len - seq_len // 5)
            width = rng.uniform(4, 12)
            depth = rng.uniform(0.8, 1.8)
            transit = -depth * np.exp(-0.5 * ((time - center) / width) ** 2)
            x[i] += transit.astype("float32")
        x[i] = (x[i] - x[i].mean()) / (x[i].std() + 1e-6)
    return x[:, None, :], y


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny synthetic light-curve data for framework smoke tests.")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--val", type=int, default=64)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--positive-rate", type=float, default=0.15)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset, (split, count) in enumerate({"train": args.train, "val": args.val, "test": args.test}.items()):
        x, y = make_split(count, args.seq_len, args.positive_rate, args.noise, args.seed + offset)
        np.savez(out_dir / f"{split}.npz", x=x, y=y)
        print(f"wrote {out_dir / f'{split}.npz'}: x={x.shape}, positives={int(y.sum())}/{len(y)}")


if __name__ == "__main__":
    main()
