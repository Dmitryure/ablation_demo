from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from encoders.depth import DEFAULT_DEPTH_MODEL_ID
from extractors.base import FeatureExtractor, module_device


def _load_depth_processor(model_id_or_path: str):
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ImportError(
            "Depth modality requires `transformers`. Install project requirements before "
            "building DepthExtractor."
        ) from exc
    return AutoImageProcessor.from_pretrained(model_id_or_path)


def _validate_frame(frame_rgb: np.ndarray) -> None:
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(
            "`video_rgb_frames` must contain RGB numpy arrays shaped [H, W, 3], "
            f"got {type(frame_rgb)}"
        )


def _is_clip_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and isinstance(value[0], np.ndarray)


def _is_clip_batch(value: object) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and _is_clip_sequence(value[0])


def _normalize_clips(frames_rgb: object) -> Sequence[Sequence[np.ndarray]]:
    if _is_clip_batch(frames_rgb):
        return frames_rgb
    if _is_clip_sequence(frames_rgb):
        return [frames_rgb]
    raise ValueError(
        "`video_rgb_frames` must be either a clip `[N][H,W,3]` or a batch `[B][N][H,W,3]`."
    )


class DepthExtractor(FeatureExtractor):
    name = "depth"

    def __init__(
        self,
        encoder: nn.Module,
        processor: Any | None = None,
        model_id_or_path: str = DEFAULT_DEPTH_MODEL_ID,
    ):
        self.encoder = encoder
        self.processor = processor if processor is not None else _load_depth_processor(model_id_or_path)
        self.model_id_or_path = model_id_or_path

    def required_keys(self) -> Tuple[str, ...]:
        return ("video_rgb_frames",)

    def extract(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        clips_rgb = _normalize_clips(batch["video_rgb_frames"])

        flat_frames: list[np.ndarray] = []
        clip_lengths: list[int] = []
        for clip_frames_rgb in clips_rgb:
            clip_lengths.append(len(clip_frames_rgb))
            for frame_rgb in clip_frames_rgb:
                _validate_frame(frame_rgb)
                flat_frames.append(frame_rgb)

        if len(set(clip_lengths)) != 1:
            raise ValueError("All clips in `video_rgb_frames` batch must have the same frame count.")

        processed = self.processor(images=flat_frames, return_tensors="pt")
        pixel_values = processed["pixel_values"]
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise ValueError(
                "Depth image processor must return `pixel_values` shaped [B*T, 3, H, W], "
                f"got {tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else type(pixel_values)}"
            )

        pixel_values = pixel_values.to(module_device(self.encoder))
        flat_features = self.encoder(pixel_values)
        if flat_features.ndim != 2:
            raise ValueError(
                "Depth encoder must return pooled frame features shaped [B*T, feature_dim], "
                f"got {tuple(flat_features.shape)}"
            )

        batch_size = len(clip_lengths)
        frames = clip_lengths[0]
        depth_features = flat_features.reshape(batch_size, frames, -1)
        return {"depth_features": depth_features}
