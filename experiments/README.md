# Experiments

This folder is for hyperparameter search outputs and comparison notes.

Run a small grid search after dataset paths are configured:

```bash
python scripts/tune.py --config configs/default.yaml --max-runs 6
```

Results are written to `experiments/tuning_results.json`.
