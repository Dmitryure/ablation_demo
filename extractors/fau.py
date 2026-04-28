from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from extractors.base import FeatureExtractor, module_device


class FAUExtractor(FeatureExtractor):
    name = "fau"

    def __init__(self, encoder: nn.Module):
        self.encoder = encoder

    def required_keys(self) -> tuple[str, ...]:
        return ("video",)

    def extract(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        video = batch["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, 3, N, H, W], got {tuple(video.shape)}")

        batch_size, channels, num_frames, height, width = video.shape
        frame_batch = (
            video.to(module_device(self.encoder))
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size * num_frames, channels, height, width)
        )
        encoded = self.encoder(frame_batch)

        if isinstance(encoded, tuple):
            fau_features = encoded[0]
            au_logits = encoded[1] if len(encoded) > 1 else None
            au_edge_logits = encoded[2] if len(encoded) > 2 else None
        else:
            fau_features = encoded
            au_logits = None
            au_edge_logits = None

        if fau_features.ndim != 3 or fau_features.shape[0] != batch_size * num_frames:
            raise ValueError(
                "FAU encoder must return per-frame features shaped [B*N, num_au, feature_dim], "
                f"got {tuple(fau_features.shape)}"
            )

        feature_batch: dict[str, Any] = {
            "fau_features": fau_features.view(
                batch_size, num_frames, fau_features.shape[1], fau_features.shape[2]
            ),
        }
        if isinstance(au_logits, torch.Tensor):
            feature_batch["fau_au_logits"] = au_logits.view(
                batch_size, num_frames, au_logits.shape[1]
            )
        if isinstance(au_edge_logits, torch.Tensor):
            feature_batch["fau_au_edge_logits"] = au_edge_logits.view(
                batch_size,
                num_frames,
                au_edge_logits.shape[1],
                au_edge_logits.shape[2],
            )
        return feature_batch
