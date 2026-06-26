# PR: Modular AI Framework for Exoplanet Detection from Light Curves

## Summary

This PR adds a complete dataset-agnostic PyTorch framework for ISRO Bharatiya Antariksh Hackathon exoplanet detection from preprocessed astronomical light curves.

The framework is designed so the preprocessing teammate can provide train/validation/test arrays, while the dashboard teammate can call a standalone inference API.

## Key Features

- Modular data loading for CSV, NPZ, NPY, and Torch files
- Variable-length sequence support through padding masks
- Three selectable model architectures:
  - 1D CNN baseline
  - CNN + LSTM
  - CNN + Transformer Encoder
- Preferred hybrid model:
  - CNN extracts local transit signatures
  - Transformer Encoder learns long-range temporal context
  - Fully connected classifier performs binary prediction
- Residual CNN blocks, batch normalization, dropout
- AdamW optimizer, LR scheduler, early stopping, checkpointing
- Mixed precision training when CUDA is available
- Focal Loss or weighted BCE for class imbalance
- Monte Carlo Dropout confidence estimation
- Test-time augmentation support
- Gradient saliency explainability for dashboard visualization
- Evaluation metrics and plots:
  - Accuracy
  - Precision
  - Recall
  - F1
  - ROC-AUC
  - PR-AUC
  - Confusion matrix
  - ROC curve
  - Precision-recall curve
  - Inference speed
  - Model size
- TensorBoard logging
- ONNX export
- Basic grid-search hyperparameter tuning
- Synthetic dataset generator for quick smoke tests and demos

## Main Commands

```bash
pip install -r requirements.txt
python scripts/make_synthetic_data.py
python scripts/train.py --config configs/default.yaml --model cnn_transformer
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt
python scripts/predict.py --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt --input data/test_sample.npy
```

## Integration Points

- Preprocessing team should output compatible `train.npz`, `val.npz`, and `test.npz` files with `x` and `y` arrays.
- Dashboard team can use `ExoplanetPredictor` from `inference/predictor.py`.
- Scientific validation team can use `outputs/evaluation/metrics.json` and generated curve plots.

## Verification Performed

- Python syntax compilation passed for all project modules.
- Full training smoke test was not run because the bundled local Python runtime does not currently include PyTorch.
