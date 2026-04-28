from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from extractors.base import FeatureExtractor, module_device

MViT_MEAN = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(3, 1, 1)
MViT_STD = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(3, 1, 1)


def _validate_frame(frame_rgb: np.ndarray) -> None:
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(
            "`video_rgb_frames` must contain RGB numpy arrays shaped [H, W, 3], "
            f"got {type(frame_rgb)}"
        )


def _frame_to_tensor(frame_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    frame_tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1).float() / 255.0
    return (frame_tensor - MViT_MEAN) / MViT_STD


def _is_clip_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and isinstance(value[0], np.ndarray)


def _is_clip_batch(value: object) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and _is_clip_sequence(value[0])


class RGBExtractor(FeatureExtractor):
    name = "rgb"

    def __init__(self, encoder: nn.Module, image_size: int = 224):
        self.encoder = encoder
        self.image_size = image_size

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def extract(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        frames_rgb = batch["video_rgb_frames"]
        if _is_clip_batch(frames_rgb):
            clips_rgb = frames_rgb
        elif _is_clip_sequence(frames_rgb):
            clips_rgb = [frames_rgb]
        else:
            raise ValueError(
                "`video_rgb_frames` must be either a clip `[N][H,W,3]` or a batch `[B][N][H,W,3]`."
            )

        batch_clips = []
        for clip_frames_rgb in clips_rgb:
            clip_frames = []
            for frame_rgb in clip_frames_rgb:
                _validate_frame(frame_rgb)
                clip_frames.append(_frame_to_tensor(frame_rgb, image_size=self.image_size))
            batch_clips.append(torch.stack(clip_frames, dim=1))

        clip = torch.stack(batch_clips, dim=0)
        clip = clip.to(module_device(self.encoder))
        rgb_features = self.encoder(clip)
        if rgb_features.ndim != 3:
            raise ValueError(
                "RGB encoder must return temporal clip features shaped [B, T, feature_dim], "
                f"got {tuple(rgb_features.shape)}"
            )

        return {"rgb_features": rgb_features}
