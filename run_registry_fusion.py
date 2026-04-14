from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from extractors import build_extractors
from fusion import FusionOutput, TokenBankFusion, prepare_token_bank
from registry import CURRENT_MODALITIES, MODALITY_TO_ID, build_registry


DEFAULT_CONFIG = Path("/home/comp/ablation_task/configs/registry_fusion.yaml")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

@dataclass(frozen=True)
class FusionConfig:
    type: str
    num_layers: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    max_time_steps: int
    checkpoint_path: Path | None


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
    modality_weights: Mapping[str, float],
    fusion_module: TokenBankFusion,
) -> tuple[FusionOutput, dict[str, object]]:
    outputs_by_name = {}
    debug: dict[str, object] = {}
    for name in enabled_modalities:
        output = registry[name].encode(batch)
        outputs_by_name[name] = output
        debug[name] = {
            "token_shape": tuple(output.tokens.shape),
            "time_ids_shape": tuple(output.time_ids.shape),
        }

    token_bank = prepare_token_bank(
        outputs_by_name=outputs_by_name,
        enabled_modalities=enabled_modalities,
        modality_to_id=MODALITY_TO_ID,
        modality_weights=modality_weights,
    )
    cls_token, fused_tokens = fusion_module(
        tokens=token_bank.weighted_tokens,
        time_ids=token_bank.time_ids,
        modality_ids=token_bank.modality_ids,
    )

    debug["raw_modality_weights"] = {
        name: float(modality_weights[name]) for name in enabled_modalities
    }
    debug["normalized_modality_weights"] = token_bank.normalized_weights
    debug["token_bank_shape"] = tuple(token_bank.tokens.shape)
    debug["token_time_ids_shape"] = tuple(token_bank.time_ids.shape)
    debug["token_modality_ids_shape"] = tuple(token_bank.modality_ids.shape)
    debug["modality_token_counts"] = {
        name: int(outputs_by_name[name].tokens.shape[1]) for name in enabled_modalities
    }
    debug["cls_token_shape"] = tuple(cls_token.shape)
    debug["fused_tokens_shape"] = tuple(fused_tokens.shape)
    debug["fused_tensor_shape"] = tuple(cls_token.shape)
    return FusionOutput(
        fused=cls_token,
        tokens=token_bank.tokens,
        time_ids=token_bank.time_ids,
        modality_ids=token_bank.modality_ids,
        modality_names=token_bank.modality_names,
        cls_token=cls_token,
        fused_tokens=fused_tokens,
    ), debug


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


def require_modality_weights(
    config: Mapping[str, Any],
    enabled_modalities: Sequence[str],
) -> dict[str, float]:
    value = config.get("modality_weights")
    if value is None:
        return {name: 1.0 for name in enabled_modalities}
    if not isinstance(value, Mapping):
        raise ValueError("`modality_weights` must be a YAML mapping when provided.")

    weights: dict[str, float] = {}
    for name in enabled_modalities:
        raw_weight = value.get(name, 1.0)
        if not isinstance(raw_weight, (int, float)):
            raise ValueError(f"`modality_weights.{name}` must be a number.")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError(f"`modality_weights.{name}` must be non-negative.")
        weights[name] = weight

    if not any(weight > 0.0 for weight in weights.values()):
        raise ValueError("At least one enabled modality weight must be greater than zero.")

    extra_keys = [key for key in value.keys() if key not in enabled_modalities]
    if extra_keys:
        raise ValueError(
            "Found weights for modalities that are not enabled: "
            f"{sorted(str(key) for key in extra_keys)}"
        )

    return weights


def validate_selected_modalities(
    enabled_modalities: Sequence[str],
    supported_modalities: Sequence[str] = CURRENT_MODALITIES,
) -> None:
    unsupported = [name for name in enabled_modalities if name not in supported_modalities]
    if unsupported:
        raise ValueError(
            "This smoke script currently supports only rgb, eye_gaze, fau, and rppg, "
            f"got unsupported modalities: {unsupported}"
        )


def require_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` must be a number in the YAML config.")
    return float(value)


def optional_path(config: Mapping[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string path or null in the YAML config.")
    return Path(value)


def require_fusion_config(config: Mapping[str, Any]) -> FusionConfig:
    value = config.get("fusion")
    if not isinstance(value, Mapping):
        raise ValueError("`fusion` must be a YAML mapping.")

    fusion_type = value.get("type")
    if fusion_type != "token_transformer":
        raise ValueError("`fusion.type` must be `token_transformer`.")

    return FusionConfig(
        type=fusion_type,
        num_layers=require_int(value, "num_layers"),
        num_heads=require_int(value, "num_heads"),
        mlp_ratio=require_float(value, "mlp_ratio"),
        dropout=require_float(value, "dropout"),
        max_time_steps=require_int(value, "max_time_steps"),
        checkpoint_path=optional_path(value, "checkpoint_path"),
    )


def build_fusion_module(dim: int, fusion_config: FusionConfig) -> TokenBankFusion:
    return TokenBankFusion(
        dim=dim,
        num_layers=fusion_config.num_layers,
        num_heads=fusion_config.num_heads,
        mlp_ratio=fusion_config.mlp_ratio,
        dropout=fusion_config.dropout,
        max_time_steps=fusion_config.max_time_steps,
        num_modalities=len(MODALITY_TO_ID),
    )


def load_fusion_checkpoint(fusion_module: TokenBankFusion, checkpoint_path: Path | None) -> bool:
    if checkpoint_path is None:
        return False
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint does not exist: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Fusion checkpoint must be a state_dict mapping or contain `state_dict`.")
    fusion_module.load_state_dict(state)
    return True


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
    fusion_config = require_fusion_config(config)

    validate_selected_modalities(enabled_modalities)
    modality_weights = require_modality_weights(config, enabled_modalities)

    torch.manual_seed(seed)

    extractor_result = build_extractors(config, modalities=enabled_modalities)
    registry = build_registry(dim=dim)
    fusion_module = build_fusion_module(dim=dim, fusion_config=fusion_config)
    fusion_checkpoint_loaded = load_fusion_checkpoint(
        fusion_module=fusion_module,
        checkpoint_path=fusion_config.checkpoint_path,
    )
    registry.eval()
    fusion_module.eval()
    raw_batch = load_video_inputs(video_path, num_frames=frames, image_size=image_size)
    try:
        batch, feature_debug = extract_selected_modalities(
            extractor_result.extractors,
            raw_batch,
            enabled_modalities,
        )
        with torch.inference_mode():
            fusion_output, debug = fuse_selected_modalities(
                registry,
                batch,
                enabled_modalities,
                modality_weights,
                fusion_module,
            )
    finally:
        for extractor in extractor_result.extractors.values():
            extractor.close()

    print("config_path:", str(args.config))
    print("video_path:", str(video_path))
    print("available_modalities:", CURRENT_MODALITIES)
    print("selected_modalities:", enabled_modalities)
    print("fusion_type:", fusion_config.type)
    print("fusion_checkpoint_loaded:", fusion_checkpoint_loaded)
    print("modality_weights:", debug["raw_modality_weights"])
    print("normalized_modality_weights:", debug["normalized_modality_weights"])
    print("video_tensor_shape:", tuple(raw_batch["video"].shape))
    for name in enabled_modalities:
        for key, shape in feature_debug[name].items():
            print(f"{key}_shape:", shape)
    for warning in extractor_result.warnings:
        print("warning:", warning)
    for name in enabled_modalities:
        print(f"{name}_token_shape:", debug[name]["token_shape"])
    print("token_bank_shape:", debug["token_bank_shape"])
    print("token_time_ids_shape:", debug["token_time_ids_shape"])
    print("token_modality_ids_shape:", debug["token_modality_ids_shape"])
    print("modality_token_counts:", debug["modality_token_counts"])
    print("cls_token_shape:", debug["cls_token_shape"])
    print("fused_tokens_shape:", debug["fused_tokens_shape"])
    print("fused_tensor_shape:", debug["fused_tensor_shape"])
    print("token_time_ids:")
    print(fusion_output.time_ids)
    print("token_modality_ids:")
    print(fusion_output.modality_ids)
    print("cls_token:")
    print(fusion_output.cls_token)
    print("fused_tensor:")
    print(fusion_output.fused)


if __name__ == "__main__":
    main()
