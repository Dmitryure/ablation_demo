from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from extractors.base import FeatureExtractor


class RGBExtractor(FeatureExtractor):
    name = "rgb"

    def __init__(self, encoder: nn.Module):
        self.encoder = encoder

    def required_keys(self) -> Tuple[str, ...]:
        return ("video",)

    def extract(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        video = batch["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, 3, N, H, W], got {tuple(video.shape)}")

        batch_size, channels, num_frames, height, width = video.shape
        # reshape [B, C, N, H, W] to [B, N, C, H, W]
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        rgb_features = self.encoder(frame_batch)
        if rgb_features.ndim != 2:
            raise ValueError(
                "RGB encoder must return pooled frame features shaped [B*N, feature_dim], "
                f"got {tuple(rgb_features.shape)}"
            )

        return {
            "rgb_features": rgb_features.view(batch_size, num_frames, rgb_features.shape[-1]),
        }
