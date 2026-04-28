from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from extractors.base import FeatureExtractor, module_device


class RPPGExtractor(FeatureExtractor):
    name = "rppg"

    def __init__(self, encoder: nn.Module):
        self.encoder = encoder

    def required_keys(self) -> tuple[str, ...]:
        return ("video",)

    def extract(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        video = batch["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, 3, N, H, W], got {tuple(video.shape)}")

        encoded = self.encoder(video.to(module_device(self.encoder)))
        if not isinstance(encoded, tuple) or len(encoded) < 2:
            raise ValueError("RPPG encoder must return `(waveform, temporal_features)`")

        waveform, temporal_features = encoded[:2]
        if temporal_features.ndim != 3:
            raise ValueError(
                "RPPG temporal features must have shape [B, N, feature_dim], "
                f"got {tuple(temporal_features.shape)}"
            )

        return {
            "rppg_waveform": waveform,
            "rppg_features": temporal_features,
        }
