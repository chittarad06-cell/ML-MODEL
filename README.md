# AI-Enabled Exoplanet Detection from Noisy Astronomical Light Curves

Research-quality, dataset-agnostic PyTorch framework for binary exoplanet transit detection from preprocessed astronomical light curves such as Kepler or TESS.

This repository focuses only on the model pipeline. Preprocessing can be handled by another teammate and plugged in through CSV, NPY, NPZ, or Torch tensor files.

## What Is Built

- Modular data loader for preprocessed train/validation/test datasets
- Three benchmark architectures:
  - `cnn1d`: 1D CNN baseline
  - `cnn_lstm`: CNN feature extractor plus LSTM
  - `cnn_transformer`: CNN plus Transformer Encoder final model
- Weighted BCE or Focal Loss for class imbalance
- AdamW optimizer, LR scheduler, early stopping, checkpoints, reproducible seeds
- Mixed precision training when CUDA is available
- Monte Carlo Dropout confidence estimation
- Time-series explainability through gradient saliency maps
- Evaluation metrics, curves, confusion matrix, model size, inference timing
- Standalone dashboard-friendly inference API
- Optional ensemble inference, test-time augmentation, TensorBoard logging, ONNX export

## Project Structure

```text
data/          Dataset adapters and batch collation
models/        CNN, CNN+LSTM, CNN+Transformer, model factory
training/      Losses, trainer, checkpointing
evaluation/    Metrics, plots, model evaluation
inference/     Single-sample and ensemble inference pipeline
utils/         Config, seeds, logging, explainability, ONNX export
configs/       YAML experiment configs
notebooks/     Placeholder for experiments
scripts/       CLI entry points
outputs/       Generated checkpoints, metrics, plots
experiments/   Hyperparameter tuning results and experiment notes
```

## Dataset Contract

Your teammate should provide preprocessed light curves and labels in one of these forms:

### Option 1: CSV

Each row is one light curve. Feature columns contain flux/time-series values. The label column defaults to `label`.

```csv
flux_0,flux_1,flux_2,...,label
0.01,0.03,-0.02,...,1
0.00,-0.01,0.01,...,0
```

### Option 2: NPZ

```python
np.savez("train.npz", x=X_train, y=y_train)
```

`x` must be shaped `[samples, sequence_length]` or `[samples, channels, sequence_length]`. `y` must contain binary labels.

### Option 3: NPY Pair

```text
train_x.npy
train_y.npy
```

Use the config keys `x_path` and `y_path`.

### Option 4: Torch

A `.pt` or `.pth` file may contain either:

```python
{"x": tensor_or_array, "y": tensor_or_array}
```

or a tuple/list:

```python
(x, y)
```

Variable sequence lengths are supported through padding and masks for the Transformer model.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Edit [configs/default.yaml](configs/default.yaml) with your dataset paths, then train:

```bash
python scripts/train.py --config configs/default.yaml --model cnn_transformer
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Run single-light-curve inference:

```bash
python scripts/predict.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt --input path/to/light_curve.npy
```

Create a tiny synthetic dataset for a smoke test or meeting demo:

```bash
python scripts/make_synthetic_data.py
python scripts/train.py --config configs/default.yaml --model cnn_transformer
```

Run practical hyperparameter tuning:

```bash
python scripts/tune.py --config configs/default.yaml --max-runs 6
```

Export ONNX:

```bash
python scripts/export_onnx.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt --output outputs/model.onnx
```

## Configuration

Architecture is selected by YAML or CLI:

```yaml
model:
  name: cnn_transformer
```

CLI overrides YAML:

```bash
python scripts/train.py --config configs/default.yaml --model cnn_lstm
```

## Inference Output

The dashboard-facing pipeline returns:

```json
{
  "prediction": "Planet",
  "probability": 0.91,
  "confidence": 0.84,
  "reliability": "High",
  "uncertainty": 0.07,
  "explanation": {
    "saliency": [0.01, 0.04, 0.90],
    "top_indices": [241, 242, 243]
  },
  "inference_time_ms": 8.5
}
```

## Evaluation Outputs

The evaluator generates:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Precision-Recall AUC
- Confusion matrix
- ROC curve
- Precision-recall curve
- Training and validation history
- Inference time
- Model size

## Meeting Summary

You can explain it like this:

> I built a modular PyTorch framework for exoplanet transit detection from preprocessed astronomical light curves. It is dataset-agnostic, so Kepler, TESS, or any compatible processed time-series dataset can be plugged in through config files. The framework benchmarks three models: a baseline 1D CNN, a CNN+LSTM, and a preferred CNN+Transformer model. The final model uses CNN layers for local transit-shape extraction and Transformer attention for long-range temporal context. It also includes class-imbalance handling, early stopping, mixed precision, checkpointing, explainable saliency outputs, MC-dropout confidence estimation, evaluation metrics and curves, and a standalone inference pipeline that can be directly connected to the dashboard.

## Notes for Teammates

- Preprocessing team: only needs to produce train/val/test arrays and labels following the dataset contract above.
- Dashboard team: can call `ExoplanetPredictor.predict_array(...)` from [inference/predictor.py](inference/predictor.py).
- Report team: can use generated metrics and plots from `outputs/evaluation/`.
