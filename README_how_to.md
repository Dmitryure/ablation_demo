# Local Setup, Training, and FPS Inference

This project trains a multimodal fake-video detector and then runs single-video inference with FPS timing.

This guide is local-only. It assumes dataset, final feature caches, and packed training shards already exist.

## 1. Repository

Use this repo root:

```bash
cd /home/comp/ablation_task
```

All commands below assume current working directory is repo root.

## 2. Python Environment

Use repo-local `venv`.

Create it if missing:

```bash
python3.12 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

If `venv` already exists, only install/update requirements when needed:

```bash
venv/bin/pip install -r requirements.txt
```

## 3. Local Data Assumptions

Training config expects dataset here:

```text
/mnt/d/final_dataset
```

Expected video layout is handled by project dataset code. For the current training config, the script reads real/fake examples from this dataset root and then uses packed feature shards for training.

## 4. Model Assets

Raw-video inference needs model/extractor assets referenced by config:

```text
checkpoints/mvit_v2_s-ae3be167.pth # used for rgb
models/face_landmarker_v2_with_blendshapes.task # eye_gaze, face_mesh
models/depth-anything-v2-small-hf # depth
```

Training from packed feature shards does not recompute raw features, but inference on a new video does. So keep these assets present for `scripts/check_model_fps.py`.

## 5. Feature Cache

Assumed existing final cache roots:

```text
/mnt/d/final_cache/v1
/mnt/d/final_cache/v2
```

Current important cache contents:

- `v1`: non-face-cropped cache. Includes `rgb`, `depth`, `eye_gaze`, `face_mesh`, `fau`, `fft`, `stft`, and `rppg`.
- `v2`: face-cropped cache. Includes `rgb`, `depth`, `eye_gaze`, `face_mesh`, `fau`, `fft`, `stft`. It intentionally has no native `rppg`.
- current `v1/rppg` training-spec path:

## 6. Training Shards

Training should use packed shards, not loose `.pt` cache files (loading shards are much faster than each example's `.pt`).

Existing local shard dirs:

```text
shards/v1_no_fau_mixed_v2
shards/v2_no_fau_with_v1_rppg_mixed_v2
```

Each shard dir contains:

```text
index.json
shards/shard_000000.pt
shards/shard_000001.pt
...
```

## 7. Current Canonical Training Config

Use generator multitask training:

```bash
venv/bin/python scripts/run_generator_multitask_training.py \
  --config runs/configs/night_sweep/09_seed0_lr3e4_gen0p15_warm2_e26.yaml
```

That config sets:

```text
dataset_root: /mnt/d/final_dataset
sharded_cache_dir: shards/v2_no_fau_with_v1_rppg_mixed_v2
output_dir: runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26
modalities: rgb, eye_gaze, face_mesh, depth
epochs: 26
lr: 0.0003
generator_loss_weight: 0.15
binary_warmup_epochs: 2
device: cuda
```

After training, output dir should contain:

```text
best.pt
run_config.json
summary.json
predictions.csv
metrics.csv
generator_metrics.csv
```

Current known best checkpoint:

```text
runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt
```

## 8. Run FPS Prediction After Training

The FPS script takes both video path and model checkpoint path as positional args:

```bash
venv/bin/python scripts/check_model_fps.py \
  /path/to/video.mp4 \
  runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt \
  --device cuda
```

For more stable timing, use warmup and repeats:

```bash
venv/bin/python scripts/check_model_fps.py \
  /path/to/video.mp4 \
  runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt \
  --device cuda \
  --warmup-runs 1 \
  --repeat 3
```

For full machine-readable output:

```bash
venv/bin/python scripts/check_model_fps.py \
  /path/to/video.mp4 \
  runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt \
  --device cuda \
  --json
```

The checkpoint directory must contain `summary.json`, and `run_config.json` must exist in that directory or an ancestor directory. This is true for normal training outputs.

## 9. FPS Output Meaning

Example:

```text
prediction=fake fake_probability=0.944426
generator=ltx2 generator_probability=0.812345
fps=112.27 end_to_end_seconds=7.3842 forward_seconds=5.8287
model=generator_multitask device=cuda score=0.916961 score_key=best_checkpoint_score
checkpoint=runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt
summary=runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/summary.json
```

Meaning:

- `prediction`: final binary label using `--threshold` default `0.5`.
- `fake_probability`: model probability for fake class.
- `generator`: top fake-generator group, only for generator multitask checkpoints.
- `fps`: original video frame count divided by end-to-end processing seconds.
- `end_to_end_seconds`: decode/sample/preprocess plus model forward time.
- `forward_seconds`: timed forward pass after video batch is prepared.
- `score`: training summary score for the checkpoint run.

## 10. How Cache Is Built

Normal workflow:

```text
raw videos
-> precompute per-modality feature cache
-> pack intersecting cached examples into read-only shards
-> train from shards
```

Feature cache is produced by:

```bash
venv/bin/python scripts/precompute_feature_cache.py \
  --dataset-root /mnt/d/final_dataset \
  --cache-dir /mnt/d/final_cache/v2 \
  --config configs/train_final_cache_v2.yaml \
  --modalities rgb eye_gaze face_mesh depth fft stft \
  --device cuda \
  --skip-failures \
  --video-decode-mode scan
```

That command is illustrative. Building full cache is long-running and should be done intentionally.

Pack cache into shards with:

```bash
venv/bin/python scripts/build_readonly_cache_shards.py \
  --config runs/configs/night_sweep/09_seed0_lr3e4_gen0p15_warm2_e26.yaml \
  --cache-dir /mnt/d/final_cache/v2 \
  --output-dir shards/v2_no_fau_with_v1_rppg_mixed_v2 \
  --modalities rgb eye_gaze face_mesh depth \
  --expected-manifest-rows 13768
```

If mixing `v2` non-rPPG cache with `v1` rPPG cache for another config, use per-modality overrides:

```bash
venv/bin/python scripts/build_readonly_cache_shards.py \
  --config <config.yaml> \
  --cache-dir /mnt/d/final_cache/v2 \
  --modality-cache-dir rppg=/mnt/d/final_cache/v1 \
  --modality-manifest-dir rppg=/mnt/d/final_cache/v1 \
  --output-dir <new_shard_dir> \
  --expected-manifest-rows 13768
```

Do not reuse an old shard dir for a config with different modalities, frame counts, image size, or crop settings. Build a new shard dir.

## 11. Common Commands

Train canonical model:

```bash
venv/bin/python scripts/run_generator_multitask_training.py \
  --config runs/configs/night_sweep/09_seed0_lr3e4_gen0p15_warm2_e26.yaml
```

Predict one video and measure FPS:

```bash
venv/bin/python scripts/check_model_fps.py \
  /path/to/video.mp4 \
  runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt \
  --device cuda
```

Predict with JSON output:

```bash
venv/bin/python scripts/check_model_fps.py \
  /path/to/video.mp4 \
  runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt \
  --device cuda \
  --json
```

Inspect training summary:

```bash
cat runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/summary.json
```
