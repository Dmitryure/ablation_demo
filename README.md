# Ablation Task

## Preprocessing Flow: Input to Fused Token Bank

This document describes only the preprocessing and fusion path inside the model.
It does not cover dataset construction or training.

### High-Level Flow

```text
raw batch
-> extractors
-> modality features
-> branches
-> modality tokens
-> token bank
-> fusion transformer
-> fused tokens + CLS summary
```

### 1. Raw Model Input

Assume the model already receives a batch with the raw inputs needed by modality extractors:

- `video`: clip tensor shaped `[B, 3, T, H, W]`
- `video_rgb_frames`: Python list of raw RGB frames for each clip

The preprocessing path begins in `ClipRealFakePredictor.forward()`, which calls `fuse()`, then `prepare_features()`.

Relevant code:

- [prediction.py](/home/comp/ablation_task/prediction.py:325)

### 2. Enabled Modalities

The model first resolves which modalities are active from YAML or CLI override.
Only enabled modalities participate in extraction, tokenization, and fusion.

Current registry branches live here:

- [registry.py](/home/comp/ablation_task/registry.py:31)

Current config example:

- [configs/registry_fusion.yaml](/home/comp/ablation_task/configs/registry_fusion.yaml:2)

### 3. Extractors Build Modality Features

`prepare_features()` loops over enabled modalities.
For each modality:

- if precomputed required keys already exist in the batch, reuse them
- otherwise run the modality extractor

Relevant code:

- [prediction.py](/home/comp/ablation_task/prediction.py:325)

Extractors do not all behave the same way:

- `rgb`, `fau`, `rppg`: extractor wraps an encoder/backbone
- `eye_gaze`, `face_mesh`: extractor computes features directly with MediaPipe

#### RGB

`RGBExtractor` reads `video_rgb_frames`, converts frames to tensors, normalizes them for MViT, and runs `RGBEncoder`.
The result is `rgb_features` with shape `[B, N_rgb, F]`.

Relevant code:

- [extractors/rgb.py](/home/comp/ablation_task/extractors/rgb.py:49)
- [encoders/video_backbones/mvit.py](/home/comp/ablation_task/encoders/video_backbones/mvit.py:102)

#### FAU

`FAUExtractor` reads `video`, reshapes the clip into a frame batch, and runs `FAUEncoder`.
The main result is `fau_features` with shape `[B, T, num_au, F]`.

Relevant code:

- [extractors/fau.py](/home/comp/ablation_task/extractors/fau.py:325)

#### rPPG

`RPPGExtractor` reads `video` and runs `RPPGEncoder`.
It returns:

- `rppg_waveform`
- `rppg_features` with shape `[B, T, F]`

Relevant code:

- [extractors/rppg.py](/home/comp/ablation_task/extractors/rppg.py:383)
- [encoders/rppg.py](/home/comp/ablation_task/encoders/rppg.py:28)

#### Eye Gaze

`EyeGazeExtractor` reads `video_rgb_frames` and uses MediaPipe face blendshapes to compute 8 eye-gaze values per frame.
It returns `eye_gaze` with shape `[B, T, 8]`.

Relevant code:

- [extractors/eye_gaze.py](/home/comp/ablation_task/extractors/eye_gaze.py:161)

#### Face Mesh

`FaceMeshExtractor` reads `video_rgb_frames` and uses MediaPipe landmarks to compute face contour points.
It returns `face_mesh` with shape `[B, T, num_points, 3]`.

Relevant code:

- [extractors/face_mesh.py](/home/comp/ablation_task/extractors/face_mesh.py:275)

### 4. Branches Convert Features Into Common Tokens

After extraction, each modality branch converts its modality-specific features into a common token format:

- `tokens`: shape `[B, N, dim]`
- `time_ids`: shape `[N]`

This is the first point where all modalities become compatible with the shared fusion module.

Key idea:

- `dim` is the common token width
- `N` can differ by modality

#### RGB Branch

Projects `rgb_features` into `[B, N_rgb, dim]`.

- [branches/rgb.py](/home/comp/ablation_task/branches/rgb.py:21)

#### Eye Gaze Branch

Projects per-frame gaze features into `[B, T, dim]`.

- [branches/eye_gaze.py](/home/comp/ablation_task/branches/eye_gaze.py:59)

#### rPPG Branch

Projects temporal rPPG features into `[B, T, dim]`.

- [branches/rppg.py](/home/comp/ablation_task/branches/rppg.py:181)

#### FAU Branch

Projects `[B, T, num_au, F]` and reshapes it into `[B, T * num_au, dim]`.
This creates one token per AU per frame.

- [branches/fau.py](/home/comp/ablation_task/branches/fau.py:133)

#### Face Mesh Branch

Projects `[B, T, num_points, 3]` and reshapes it into `[B, T * num_points, dim]`.
This creates one token per point per frame.

- [branches/face_mesh.py](/home/comp/ablation_task/branches/face_mesh.py:94)

### 5. Token Bank Assembly

Once each branch has emitted tokens, the model assembles a single token bank.
This happens in `prepare_token_bank()`.

The function:

- concatenates all modality tokens across the token dimension
- concatenates all `time_ids`
- builds `modality_ids`
- normalizes modality weights
- builds:
  - `tokens`: unweighted token bank
  - `weighted_tokens`: token bank scaled by modality weight

Relevant code:

- [fusion.py](/home/comp/ablation_task/fusion.py:76)

This is the token bank:

- shape: `[B, total_tokens, dim]`
- all enabled modality tokens live in one shared sequence
- tokens still keep identity through `time_ids` and `modality_ids`

Example mental model:

```text
RGB tokens
+ FAU tokens
+ rPPG tokens
+ Eye Gaze tokens
+ Face Mesh tokens
= token bank
```

### 6. Fusion Transformer

The token bank then goes into `TokenBankFusion`.

Fusion does the following:

1. starts from `weighted_tokens`
2. adds time embeddings using `time_ids`
3. adds modality embeddings using `modality_ids`
4. prepends one learned `CLS` token
5. sends the full sequence through a Transformer encoder

Relevant code:

- [fusion.py](/home/comp/ablation_task/fusion.py:166)

So the transformer input looks like:

```text
[CLS] [tok_1] [tok_2] ... [tok_N]
```

After transformer mixing:

- every token may attend to tokens from other modalities
- `CLS` becomes the global summary token for the whole clip

### 7. Fusion Output

The fusion path returns `FusionOutput`, which includes:

- `tokens`: original token bank before transformer mixing
- `fused_tokens`: token sequence after transformer mixing, including `CLS`
- `cls_token`: first token after transformer, shape `[B, dim]`
- `fused`: same tensor as `cls_token`
- `time_ids`
- `modality_ids`
- `modality_names`

Relevant code:

- [fusion.py](/home/comp/ablation_task/fusion.py:12)
- [prediction.py](/home/comp/ablation_task/prediction.py:228)

### Terminology Summary

- `features`: extractor output, still modality-specific
- `tokens`: branch output, already projected to common width `dim`
- `token bank`: concatenated modality tokens before transformer fusion
- `fused_tokens`: token sequence after transformer mixing
- `cls_token`: global fused summary vector for the clip

### One-Line Summary

The preprocessing path is:

```text
raw batch -> extractors -> modality features -> branches -> token bank -> fusion transformer -> fused token space + CLS summary
```
