from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import ExoplanetPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    predictor = ExoplanetPredictor(args.config, args.checkpoint)
    print(json.dumps(predictor.predict_file(args.input), indent=2))


if __name__ == "__main__":
    main()
