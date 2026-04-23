from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput
from branches.compression import DEFAULT_SLOT_COUNTS, TemporalLatentQueryPooling, validate_positive_int


class RGBBranch(ModalityBranch):
    name = "rgb"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["rgb"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "rgb.slot_count")
        self.proj = nn.LazyLinear(dim)
        self.pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count)

    def required_keys(self) -> Tuple[str, ...]:
        return ("rgb_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        rgb_features = batch["rgb_features"]
        if rgb_features.ndim != 3:
            raise ValueError(
                "RGB features must have shape [B, N, feature_dim], "
                f"got {tuple(rgb_features.shape)}"
            )

        projected_tokens = self.proj(rgb_features)
        tokens = self.pool(projected_tokens)
        num_frames = rgb_features.shape[1]
        time_ids = torch.arange(self.slot_count, device=rgb_features.device)
        debug = {
            "input_shape": tuple(rgb_features.shape),
            "feature_shape": tuple(rgb_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "slot_count": self.slot_count,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
