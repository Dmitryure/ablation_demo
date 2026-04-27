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

Video files in those folders are intentionally not tracked by git.
