# Implementation Report: Raw Dataset AI Model Pipeline

## Overview

The repository was upgraded from a preprocessed-split-only model framework into a complete raw astronomical light-curve modeling pipeline for the ISRO Bharatiya Antariksh Hackathon.

The existing modular PyTorch design was preserved. The update extends the data layer, benchmark workflow, configuration, documentation, and inference output while keeping the original CNN, CNN+LSTM, CNN+Transformer, training, evaluation, saliency, MC-dropout, and dashboard API functionality.

## Features Implemented

- Raw dataset ingestion from one file or a directory
- Automatic file extension detection
- Support for:
  - Apache Parquet
  - CSV
  - NPZ
  - NPY
  - Torch `.pt` and `.pth`
- Parquet loading through `pandas.read_parquet()`
- Recursive directory loading for many TESS/Kepler-like files
- Automatic astronomy column detection:
  - `TIME`
  - `PDCSAP_FLUX`
  - `SAP_FLUX`
  - `QUALITY`
- YAML overrides for:
  - time column
  - flux column
  - quality column
  - label column
  - sample ID column
- Internal preprocessing:
  - remove NaN values
  - remove infinite values
  - remove duplicates
  - sort by time
  - optional quality filtering
  - flux normalization
  - outlier clipping
  - fixed-length sequence generation
  - padding support
  - masking support through existing collate function
- Automatic train/validation/test splitting:
  - default 70/15/15
  - configurable ratios
  - reproducible seed
  - stratified splitting when possible
  - split cache saved to `outputs/splits/splits.json`
- Existing train/test dataset support:
  - `data.raw_train_path`
  - `data.raw_test_path`
  - optional `data.raw_val_path`
  - preserves test split
  - creates validation data from train when validation is not supplied
- Label handling:
  - labels inside tabular files
  - labels inside NPZ/Torch files
  - external label CSV/Parquet through `data.labels_path`
  - clear error when supervised labels are missing
- Benchmark workflow:
  - trains CNN, CNN+LSTM, and CNN+Transformer on the same data
  - evaluates each model
  - creates `outputs/benchmark/leaderboard.json`
  - identifies best model
- Inference API improvements:
  - prediction
  - probability
  - confidence
  - uncertainty
  - reliability
  - model used
  - inference time
  - saliency values
  - most important time indices
  - predicted transit region
- README rewritten for raw dataset workflow
- Requirements updated with `pyarrow` for Parquet support

## Files Modified

- `README.md`
- `requirements.txt`
- `configs/default.yaml`
- `data/__init__.py`
- `data/datasets.py`
- `inference/predictor.py`

## New Modules Added

- `data/ingestion.py`
  - file discovery
  - format detection
  - Parquet/CSV/NPZ/NPY/Torch loading
  - astronomy column detection
  - raw record conversion
- `data/preprocessing.py`
  - reusable raw light-curve cleaning and sequence generation
- `data/splitting.py`
  - label validation
  - external label support
  - reproducible split creation and caching
- `scripts/benchmark.py`
  - architecture leaderboard generation
- `IMPLEMENTATION_REPORT.md`
  - implementation summary for team/project review

## Configuration Changes

`configs/default.yaml` now supports raw dataset mode:

- `data.dataset_path`
- `data.dataset_format`
- `data.directory_mode`
- `data.raw_train_path`
- `data.raw_val_path`
- `data.raw_test_path`
- `data.labels_path`
- `data.sample_id_column`
- `data.time_column`
- `data.flux_column`
- `data.quality_column`
- `data.sequence_length`
- `data.sequence_stride`
- `data.preprocessing`
- `data.splits`

Backward compatibility is preserved. If `data.dataset_path` is `null`, the previous `train`, `val`, and `test` dataset configuration is used.

## Benchmark Additions

Run:

```bash
python scripts/benchmark.py --config configs/default.yaml
```

Metrics included:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Training time
- Inference time
- Model size
- Parameter count

## Assumptions Made

- A directory of TESS/Kepler files usually contains one light curve per file.
- If train/test data is already separated, the test split should remain untouched and validation can be derived from training data.
- A single CSV/Parquet file can represent multiple samples if `sample_id_column` is configured.
- If no time column is present, observation index is used as time.
- If no flux column is detected, the last numeric non-label column is used as a fallback.
- Supervised training cannot continue without labels.
- Sequence windows generated from one original light curve inherit the same label.

## Validation Performed

- Syntax compilation was run across all Python modules.
- Full training/evaluation execution depends on PyTorch being installed in the active Python environment.

## Remaining TODO Items

- Add FITS file ingestion if the final dataset arrives in native astronomy FITS format.
- Add unit tests with small fixture files for Parquet, CSV, NPZ, NPY, and Torch formats.
- Add more advanced transit-region postprocessing using actual time values, not only sequence indices.
- Add optional class-weight auto-computation from training labels.

## Suggested Future Improvements

- Integrate `astropy` for FITS and astronomy metadata handling.
- Add Bayesian calibration metrics for confidence reliability.
- Add Grad-CAM-style temporal attribution in addition to gradient saliency.
- Add k-fold cross-validation for stronger scientific reporting.
- Add model registry metadata for dashboard deployment.
