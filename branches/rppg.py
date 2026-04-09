from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput


class RPPGBranch(ModalityBranch):
    name = "rppg"

    def __init__(self, encoder: nn.Module, dim: int):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.LazyLinear(dim)

    def required_keys(self) -> Tuple[str, ...]:
        return ("video",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        video = batch["video"]
        if video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, 3, N, H, W], got {tuple(video.shape)}")

        encoded = self.encoder(video)
        if not isinstance(encoded, tuple) or len(encoded) < 2:
            raise ValueError("RPPG encoder must return `(waveform, temporal_features)`")

        waveform, temporal_features = encoded[:2]
        if temporal_features.ndim != 3:
            raise ValueError(
                "RPPG temporal features must have shape [B, N, feature_dim], "
                f"got {tuple(temporal_features.shape)}"
            )

        tokens = self.proj(temporal_features)
        num_frames = temporal_features.shape[1]
        time_ids = torch.arange(num_frames, device=video.device)
        debug = {
            "input_shape": tuple(video.shape),
            "waveform_shape": tuple(waveform.shape),
            "feature_shape": tuple(temporal_features.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "waveform": waveform.detach(),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
