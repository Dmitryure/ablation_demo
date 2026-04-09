from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from extractors import build_extractors
from registry import CURRENT_MODALITIES, build_registry


DEFAULT_CONFIG = Path("/home/comp/ablation_task/configs/registry_fusion.yaml")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def load_video_inputs(
    path: Path,
    num_frames: int,
    image_size: int = 224,
) -> dict[str, object]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        raise RuntimeError(f"Video has only {total_frames} frames, need at least {num_frames}")

    indices = torch.linspace(0, total_frames - 1, steps=num_frames).round().to(dtype=torch.int64).tolist()
    clip_frames = []
    rgb_frames = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame {frame_index} from {path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frames.append(frame)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD
        clip_frames.append(frame_tensor)

    cap.release()
    return {
        "video": torch.stack(clip_frames, dim=1).unsqueeze(0),  # [1, 3, N, H, W]
        "video_rgb_frames": rgb_frames,
    }


def extract_selected_modalities(
    extractors: Mapping[str, object],
    raw_batch: Mapping[str, object],
    enabled_modalities: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    feature_batch: dict[str, torch.Tensor] = {}
    debug: dict[str, object] = {}
    for name in enabled_modalities:
        extracted = extractors[name].extract(raw_batch)
        feature_batch.update(extracted)
        debug[name] = {key: tuple(value.shape) for key, value in extracted.items() if isinstance(value, torch.Tensor)}
    return feature_batch, debug


def fuse_selected_modalities(
    registry: nn.ModuleDict,
    batch: dict[str, torch.Tensor],
    enabled_modalities: Sequence[str],
) -> tuple[torch.Tensor, dict[str, object]]:
    outputs = []
    debug = {}
    for name in enabled_modalities:
        output = registry[name].encode(batch)
        outputs.append(output)
        debug[name] = {
            "token_shape": tuple(output.tokens.shape),
            "time_ids_shape": tuple(output.time_ids.shape),
        }

    tokens = torch.cat([output.tokens for output in outputs], dim=1)
    fused = tokens.mean(dim=1)
    debug["fused_token_shape"] = tuple(tokens.shape)
    debug["fused_tensor_shape"] = tuple(fused.shape)
    return fused, debug


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer in the YAML config.")
    return value


def require_path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"`{key}` must be a non-empty string path in the YAML config.")
    return Path(value)


def require_modalities(config: Mapping[str, Any]) -> tuple[str, ...]:
    value = config.get("modalities")
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError("`modalities` must be a non-empty YAML list of strings.")
    return tuple(item.strip() for item in value if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a plug-and-play registry fusion smoke test from YAML.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    video_path = require_path(config, "video")
    enabled_modalities = require_modalities(config)
    frames = require_int(config, "frames")
    image_size = require_int(config, "image_size")
    dim = require_int(config, "dim")
    seed = require_int(config, "seed")

    unsupported = [m for m in enabled_modalities if m not in {"eye_gaze", "fau", "rppg"}]
    if unsupported:
        raise ValueError(
            "This smoke script currently supports only eye_gaze, fau, and rppg, "
            f"got unsupported modalities: {unsupported}"
        )

    torch.manual_seed(seed)

    extractor_result = build_extractors(config, modalities=enabled_modalities)
    registry = build_registry(dim=dim)
    registry.eval()
    raw_batch = load_video_inputs(video_path, num_frames=frames, image_size=image_size)
    try:
        batch, feature_debug = extract_selected_modalities(
            extractor_result.extractors,
            raw_batch,
            enabled_modalities,
        )
        with torch.inference_mode():
            fused, debug = fuse_selected_modalities(registry, batch, enabled_modalities)
    finally:
        for extractor in extractor_result.extractors.values():
            extractor.close()

    print("config_path:", str(args.config))
    print("video_path:", str(video_path))
    print("available_modalities:", CURRENT_MODALITIES)
    print("selected_modalities:", enabled_modalities)
    print("video_tensor_shape:", tuple(raw_batch["video"].shape))
    for name in enabled_modalities:
        for key, shape in feature_debug[name].items():
            print(f"{key}_shape:", shape)
    for warning in extractor_result.warnings:
        print("warning:", warning)
    for name in enabled_modalities:
        print(f"{name}_token_shape:", debug[name]["token_shape"])
    print("fused_token_shape:", debug["fused_token_shape"])
    print("fused_tensor_shape:", debug["fused_tensor_shape"])
    print("fused_tensor:")
    print(fused)


if __name__ == "__main__":
    main()
