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


class RPPGBranch(ModalityBranch):
    name = "rppg"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["rppg"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "rppg.slot_count")
        if self.slot_count < 2:
            raise ValueError("`rppg.slot_count` must be at least 2.")
        self.proj = nn.LazyLinear(dim)
        self.signal_proj = nn.LazyLinear(dim)
        self.pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count - 1)

    def required_keys(self) -> tuple[str, ...]:
        return ("rppg_features", "rppg_signal_features")

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        temporal_features = batch["rppg_features"]
        signal_features = batch["rppg_signal_features"]
        if temporal_features.ndim != 3:
            raise ValueError(
                "RPPG features must have shape [B, N, feature_dim], "
                f"got {tuple(temporal_features.shape)}"
            )
        if signal_features.ndim != 2:
            raise ValueError(
                "RPPG signal features must have shape [B, feature_dim], "
                f"got {tuple(signal_features.shape)}"
            )
        if signal_features.shape[0] != temporal_features.shape[0]:
            raise ValueError("RPPG signal features batch size must match temporal features.")
        signal_features = signal_features.to(
            device=temporal_features.device,
            dtype=temporal_features.dtype,
        )

        waveform = batch.get("rppg_waveform")
        projected_tokens = self.proj(temporal_features)
        temporal_tokens = self.pool(projected_tokens)
        signal_token = self.signal_proj(signal_features).unsqueeze(1)
        tokens = torch.cat([signal_token, temporal_tokens], dim=1)
        num_frames = temporal_features.shape[1]
        time_ids = torch.arange(self.slot_count, device=temporal_features.device)
        debug = {
            "input_shape": tuple(temporal_features.shape),
            "signal_feature_shape": tuple(signal_features.shape),
            "waveform_shape": tuple(waveform.shape) if isinstance(waveform, torch.Tensor) else None,
            "feature_shape": tuple(temporal_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "signal_token_shape": tuple(signal_token.shape),
            "temporal_token_shape": tuple(temporal_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "slot_count": self.slot_count,
            "time_ids": tuple(time_ids.tolist()),
            "waveform": waveform.detach() if isinstance(waveform, torch.Tensor) else None,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
