from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput
from branches.compression import (
    DEFAULT_SLOT_COUNTS,
    TemporalLatentQueryPooling,
    validate_positive_int,
)


class DepthBranch(ModalityBranch):
    name = "depth"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["depth"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "depth.slot_count")
        self.proj = nn.LazyLinear(dim)
        self.pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count)

    def required_keys(self) -> tuple[str, ...]:
        return ("depth_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        depth_features = batch["depth_features"]
        if depth_features.ndim != 3:
            raise ValueError(
                "Depth features must have shape [B, N, feature_dim], "
                f"got {tuple(depth_features.shape)}"
            )

        projected_tokens = self.proj(depth_features)
        tokens = self.pool(projected_tokens)
        num_frames = depth_features.shape[1]
        time_ids = torch.arange(self.slot_count, device=depth_features.device)
        debug = {
            "input_shape": tuple(depth_features.shape),
            "feature_shape": tuple(depth_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "slot_count": self.slot_count,
            "time_ids": tuple(time_ids.tolist()),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
