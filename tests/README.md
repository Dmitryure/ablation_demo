# Tests

This folder owns repo tests and local-only test video assets.

## Automated tests

Run the current automated test suite from the repo root:

```bash
venv/bin/python -m pytest tests
```

## Local video assets

Use these ignored folders for tiny manual sanity checks:

```text
tests/overfit_videos/
  real/
  fake/
tests/predict_videos/
  real/
  fake/
```

`overfit_videos` is for 5-10 labeled clips the model should memorize during a tiny overfit run.
`predict_videos` is for 5-10 labeled clips used to inspect predictions from that overfitted model.


## Tiny overfit

Run the fake/real task-model sanity check from the repo root:

```bash
venv/bin/python scripts/run_tiny_overfit.py --device cuda
```

The script trains a binary head on top of fused pipeline output, using `tests/overfit_videos`.
It predicts on `tests/predict_videos` and writes ignored artifacts to `tests/overfit_runs`:

- `best.pt`
- `metrics.csv`
- `train_accuracy.png`
- `predictions.csv`
- `summary.json`

To compare modality subsets on the same tiny run:

```bash
venv/bin/python scripts/run_tiny_overfit.py --device cuda --modality-permutations singletons-plus-all
```

This writes per-subset runs under `tests/overfit_runs/modality_permutations` and an aggregate
summary/plot:

- `tests/overfit_runs/modality_permutations_summary.json`
- `tests/overfit_runs/modality_permutations_train_accuracy.png`
