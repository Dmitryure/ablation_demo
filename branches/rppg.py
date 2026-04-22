from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput
from branches.compression import (
    DEFAULT_OUTPUT_TOKENS,
    TemporalLatentQueryPooling,
    validate_positive_int,
)


class RPPGBranch(ModalityBranch):
    name = "rppg"

    def __init__(self, dim: int, output_tokens_per_clip: int = DEFAULT_OUTPUT_TOKENS["rppg"]):
        super().__init__()
        self.output_tokens_per_clip = validate_positive_int(
            output_tokens_per_clip,
            "rppg.output_tokens_per_clip",
        )
        self.proj = nn.LazyLinear(dim)
        self.pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.output_tokens_per_clip)

    def required_keys(self) -> Tuple[str, ...]:
        return ("rppg_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        temporal_features = batch["rppg_features"]
        if temporal_features.ndim != 3:
            raise ValueError(
                "RPPG features must have shape [B, N, feature_dim], "
                f"got {tuple(temporal_features.shape)}"
            )

        waveform = batch.get("rppg_waveform")
        projected_tokens = self.proj(temporal_features)
        tokens = self.pool(projected_tokens)
        num_frames = temporal_features.shape[1]
        time_ids = torch.arange(self.output_tokens_per_clip, device=temporal_features.device)
        debug = {
            "input_shape": tuple(temporal_features.shape),
            "waveform_shape": tuple(waveform.shape) if isinstance(waveform, torch.Tensor) else None,
            "feature_shape": tuple(temporal_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "output_tokens_per_clip": self.output_tokens_per_clip,
            "time_ids": tuple(time_ids.tolist()),
            "waveform": waveform.detach() if isinstance(waveform, torch.Tensor) else None,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
