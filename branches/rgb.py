from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput


class RGBBranch(ModalityBranch):
    name = "rgb"

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.LazyLinear(dim)

    def required_keys(self) -> Tuple[str, ...]:
        return ("rgb_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        rgb_features = batch["rgb_features"]
        if rgb_features.ndim != 3:
            raise ValueError(
                "RGB features must have shape [B, N, feature_dim], "
                f"got {tuple(rgb_features.shape)}"
            )

        tokens = self.proj(rgb_features)
        num_frames = rgb_features.shape[1]
        time_ids = torch.arange(num_frames, device=rgb_features.device)
        debug = {
            "input_shape": tuple(rgb_features.shape),
            "feature_shape": tuple(rgb_features.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
