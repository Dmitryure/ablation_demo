# Best Run Metrics

Run: `seed0_lr3e4_gen0p15_warm2_e26`

| Metric | Value |
|---|---:|
| `run` | seed0_lr3e4_gen0p15_warm2_e26 |
| `checkpoint` | runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26/best.pt |
| `score` | 0.916961 |
| `modalities` | rgb, eye_gaze, face_mesh, depth |
| `sharded_cache_dir` | shards/v2_no_fau_with_v1_rppg_mixed_v2 |
| `val_balanced_accuracy` | 0.913697 |
| `test_balanced_accuracy` | 0.945034 |
| `test_accuracy` | 0.944928 |
| `test_fake_recall` | 0.923077 |
| `test_real_specificity` | 0.966990 |
| `test_fake_precision` | 0.965795 |
| `test_f1` | 0.943953 |
| `test_macro_generator_recall` | 0.854535 |
| `test_worst_generator_recall` | 0.692308 |
| `test_macro_binary_recall_by_generator` | 0.920492 |
| `calibration_known_generator_precision` | 0.996183 |
| `calibration_known_generator_coverage` | 0.595455 |
| `calibration_unknown_or_low_confidence_rate` | 0.427027 |
| `holdout_fake_recall` | 0.959965 |
| `holdout_false_negative_rate` | 0.040035 |
| `holdout_known_generator_precision` | 0.992916 |
| `holdout_known_generator_coverage` | 0.743635 |
| `holdout_low_confidence_or_unknown_rate` | 0.216330 |

## Test Generator Metrics

| Generator group | Count | Binary recall | Generator recall |
|---|---:|---:|---:|
| `dlc` | 91 | 0.978022 | 0.692308 |
| `liveavatar` | 101 | 0.900990 | 0.851485 |
| `ltx2` | 129 | 0.984496 | 0.976744 |
| `ovi` | 46 | 0.934783 | 0.934783 |
| `sadtalker` | 66 | 0.954545 | 0.878788 |
| `unknown_or_other` | 87 | 0.770115 | 0.793103 |

## Extra Fake Holdout Recall

| Generator group | Fake recall |
|---|---:|
| `dlc` | 0.886578 |
| `liveavatar` | 0.932416 |
| `ltx2` | 0.984249 |
| `sadtalker` | 0.984334 |
