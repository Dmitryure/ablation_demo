# How To Run Training

Create a config:

```bash
cd /home/comp/ablation_task
cp configs/train_final_cache_v2.yaml runs/configs/my_run.yaml
```

Example `runs/configs/my_run.yaml` values to change:

```yaml
training:
  run:
    dataset_root: /path/to/dataset
    cache_dir: /path/to/dataset/feature_cache
    sharded_cache_dir: null
    output_dir: runs/my_run
    modalities:
      - rgb
      - eye_gaze
      - face_mesh
      - depth
    batch_size: 8
    epochs: 30
    lr: 0.001
    device: cuda
```

Keep FF++ and Celeb-DF data separate at:

```text
/path/to/celeb_ffpp_dataset
```

Run training:

```bash
venv/bin/python scripts/run_iterative_cached_ablation.py --config runs/configs/my_run.yaml
```

Outputs go to `runs/my_run`.
