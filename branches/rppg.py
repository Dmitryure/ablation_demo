from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput


class RPPGBranch(ModalityBranch):
    name = "rppg"

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.LazyLinear(dim)

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
        tokens = self.proj(temporal_features)
        num_frames = temporal_features.shape[1]
        time_ids = torch.arange(num_frames, device=temporal_features.device)
        debug = {
            "input_shape": tuple(temporal_features.shape),
            "waveform_shape": tuple(waveform.shape) if isinstance(waveform, torch.Tensor) else None,
            "feature_shape": tuple(temporal_features.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "waveform": waveform.detach() if isinstance(waveform, torch.Tensor) else None,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
