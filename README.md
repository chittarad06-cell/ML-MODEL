# AI-Enabled Exoplanet Detection from Noisy Astronomical Light Curves

Dataset-agnostic PyTorch framework for the ISRO Bharatiya Antariksh Hackathon problem statement: detecting exoplanet transit signals from noisy astronomical light curves.

The repository now supports the complete model-side pipeline:

Raw TESS/Kepler-style files -> ingestion -> cleaning -> normalization -> sequence generation -> automatic train/validation/test split -> model training -> benchmark -> confidence estimation -> explainability -> dashboard-ready inference.

## What This Project Provides

- Raw dataset ingestion from one file or a directory of files
- Supported formats: `.parquet`, `.csv`, `.npz`, `.npy`, `.pt`, `.pth`
- Automatic file type detection from extension
- Common astronomy column detection:
  - `TIME`
  - `PDCSAP_FLUX`
  - `SAP_FLUX`
  - `QUALITY`
- YAML overrides for time, flux, quality, label, and sample ID columns
- Internal preprocessing:
  - NaN and infinity removal
  - duplicate observation removal
  - time sorting
  - optional bad-quality filtering
  - flux normalization
  - outlier clipping
  - fixed-length sequence generation
  - padding and masking
- Automatic 70/15/15 train/validation/test split by default
- Reproducible split caching
- Stratified split when labels are available
- Three benchmark architectures:
  - `cnn1d`
  - `cnn_lstm`
  - `cnn_transformer`
- Focal Loss or weighted BCE for rare exoplanet class imbalance
- AdamW, LR scheduling, early stopping, checkpoints
- Mixed precision training on CUDA
- TensorBoard logging
- Monte Carlo Dropout uncertainty
- Gradient saliency explainability
- Standalone dashboard-compatible inference API
- Benchmark leaderboard across all architectures
- Optional ONNX export

## Project Structure

```text
configs/       YAML configuration
data/          Ingestion, preprocessing, splitting, dataset wrappers
models/        CNN, CNN+LSTM, CNN+Transformer
training/      Losses, trainer, checkpoints
evaluation/    Metrics, plots, model evaluation
inference/     Dashboard-ready prediction API
utils/         Config, seeds, explainability, ONNX, model info
scripts/       Train/evaluate/predict/benchmark commands
experiments/   Hyperparameter search outputs
notebooks/     Optional analysis notebooks
outputs/       Generated splits, checkpoints, metrics, plots
```

## Installation

```bash
pip install -r requirements.txt
```

For Parquet support, make sure your environment has either `pyarrow` or `fastparquet`. `pyarrow` is recommended:

```bash
pip install pyarrow
```

## Dataset Modes

### 1. Raw Dataset Mode: One File Or Directory

Use this when you have one raw file or a directory of many TESS/Kepler-like files.

In [configs/default.yaml](configs/default.yaml):

```yaml
data:
  dataset_path: data/raw
  dataset_format: auto
  directory_mode: auto
  time_column: null
  flux_column: null
  quality_column: null
  label_column: label
```

The loader automatically finds every supported file under `data/raw`.

### 2. Raw Train/Test Split Mode

Use this when the dataset is already separated into train and test folders or files.

```yaml
data:
  dataset_path: null
  raw_train_path: data/train
  raw_val_path: null
  raw_test_path: data/test
  label_column: label
  time_column: TIME
  flux_column: PDCSAP_FLUX
  quality_column: QUALITY
```

If `raw_val_path` is `null`, the framework preserves the test set and automatically creates validation data from the training set using `data.splits.val_from_train_ratio`.

### 3. Backward-Compatible Pre-Split Mode

If you already have prepared train/validation/test files, leave `dataset_path: null` and configure:

```yaml
data:
  dataset_path: null
  train:
    format: npz
    path: data/train.npz
  val:
    format: npz
    path: data/val.npz
  test:
    format: npz
    path: data/test.npz
```

## Supported Formats

### Parquet

```yaml
data:
  dataset_path: data/tess_light_curves.parquet
  flux_column: PDCSAP_FLUX
  time_column: TIME
  quality_column: QUALITY
  label_column: label
```

The code uses `pandas.read_parquet()`.

### CSV

CSV files can contain one light curve or many light curves. If many samples are stored in one table, provide `sample_id_column`.

```yaml
data:
  dataset_path: data/light_curves.csv
  sample_id_column: object_id
  time_column: TIME
  flux_column: PDCSAP_FLUX
  label_column: label
```

### NPZ

```python
np.savez("dataset.npz", x=X, y=y)
```

### NPY

`.npy` may contain one light curve or a stack of light curves. If labels are not inside the file, provide `labels_path`.

### Torch

`.pt` or `.pth` may contain:

```python
{"x": x_tensor, "y": y_tensor}
```

or:

```python
(x_tensor, y_tensor)
```

## Label Handling

Supervised training requires labels.

Labels can come from:

- a label column inside CSV/Parquet files
- `y` inside NPZ/Torch files
- a separate label file configured with `data.labels_path`

Example external label CSV:

```csv
sample_id,label
target_001,1
target_002,0
```

If labels are missing, training stops with a clear error explaining where labels should be supplied.

## Preprocessing Pipeline

Configured in YAML:

```yaml
data:
  sequence_length: 2048
  sequence_stride: 2048
  preprocessing:
    remove_bad_quality: true
    good_quality_value: 0
    min_points: 16
    normalization: standard
    outlier_clip:
      enabled: true
      sigma: 5.0
    sequence_mode: pad_or_window
    pad_short_sequences: true
    padding_strategy: zero
```

Available normalization modes:

- `standard`
- `robust`
- `minmax`
- `none`

## Automatic Splitting

Default split is 70/15/15:

```yaml
data:
  splits:
    ratios:
      train: 0.7
      val: 0.15
      test: 0.15
    val_from_train_ratio: 0.15
    cache_path: outputs/splits/splits.json
    reuse_cached: true
```

Splits are made at sample/sequence level, stratified when both classes are present, and cached for reproducibility.

If your raw data already has train/test folders, the framework does not mix the test set into training. It only splits the training data into train/validation when no validation folder is supplied.

## Training

Train the preferred model:

```bash
python scripts/train.py --config configs/default.yaml --model cnn_transformer
```

Train another architecture:

```bash
python scripts/train.py --config configs/default.yaml --model cnn_lstm
```

## Benchmark All Models

Train and compare CNN, CNN+LSTM, and CNN+Transformer:

```bash
python scripts/benchmark.py --config configs/default.yaml
```

Output:

```text
outputs/benchmark/leaderboard.json
```

Leaderboard metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Training time
- Inference time
- Model size

The script automatically identifies the best model.

## Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Evaluation outputs:

```text
outputs/evaluation/metrics.json
outputs/evaluation/roc_curve.png
outputs/evaluation/precision_recall_curve.png
```

## Inference

```bash
python scripts/predict.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt --input path/to/light_curve.npy
```

Inference returns:

```json
{
  "prediction": "Planet",
  "probability": 0.91,
  "confidence": 0.84,
  "uncertainty": 0.07,
  "reliability": "High",
  "model_used": "cnn_transformer",
  "inference_time_ms": 8.5,
  "saliency_values": [0.01, 0.04, 0.90],
  "most_important_time_indices": [241, 242, 243],
  "predicted_transit_region": {
    "start_index": 241,
    "end_index": 243,
    "center_index": 242
  }
}
```

## Dashboard Integration

The dashboard can call:

```python
from inference import ExoplanetPredictor

predictor = ExoplanetPredictor("configs/default.yaml", "outputs/checkpoints/best_model.pt")
result = predictor.predict_array(preprocessed_or_raw_flux_array)
```

Use:

- `prediction`
- `probability`
- `confidence`
- `uncertainty`
- `reliability`
- `saliency_values`
- `most_important_time_indices`
- `predicted_transit_region`

## Demo Without Real Data

Create a tiny synthetic dataset for smoke testing:

```bash
python scripts/make_synthetic_data.py
python scripts/train.py --config configs/default.yaml --model cnn_transformer
```

This is only for framework testing, not scientific validation.

## ONNX Export

```bash
python scripts/export_onnx.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt --output outputs/model.onnx
```
