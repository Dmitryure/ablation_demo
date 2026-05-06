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


class STFTBranch(ModalityBranch):
    name = "stft"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["stft"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "stft.slot_count")
        self.proj = nn.LazyLinear(dim)
        self.pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count)

    def required_keys(self) -> tuple[str, ...]:
        return ("stft_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        stft_features = batch["stft_features"]
        if stft_features.ndim != 3:
            raise ValueError(
                "STFT features must have shape [B, num_windows, num_freq_bins], "
                f"got {tuple(stft_features.shape)}"
            )

        projected_tokens = self.proj(stft_features)
        tokens = self.pool(projected_tokens)
        num_windows = stft_features.shape[1]
        time_ids = torch.arange(self.slot_count, device=stft_features.device)
        debug = {
            "input_shape": tuple(stft_features.shape),
            "feature_shape": tuple(stft_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_windows": num_windows,
            "slot_count": self.slot_count,
            "time_ids": tuple(time_ids.tolist()),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
