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


class FFTBranch(ModalityBranch):
    name = "fft"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["fft"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "fft.slot_count")
        self.proj = nn.LazyLinear(dim)
        self.pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count)

    def required_keys(self) -> tuple[str, ...]:
        return ("fft_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        fft_features = batch["fft_features"]
        if fft_features.ndim != 3:
            raise ValueError(
                "FFT features must have shape [B, N, num_bins], "
                f"got {tuple(fft_features.shape)}"
            )

        projected_tokens = self.proj(fft_features)
        tokens = self.pool(projected_tokens)
        num_frames = fft_features.shape[1]
        time_ids = torch.arange(self.slot_count, device=fft_features.device)
        debug = {
            "input_shape": tuple(fft_features.shape),
            "feature_shape": tuple(fft_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "slot_count": self.slot_count,
            "time_ids": tuple(time_ids.tolist()),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)