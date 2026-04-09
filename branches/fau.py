from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput


class FAUBranch(ModalityBranch):
    name = "fau"

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.LazyLinear(dim)

    def required_keys(self) -> Tuple[str, ...]:
        return ("fau_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        fau_features = batch["fau_features"]
        if fau_features.ndim != 4:
            raise ValueError(
                "FAU features must have shape [B, N, num_au, feature_dim], "
                f"got {tuple(fau_features.shape)}"
            )

        au_logits = batch.get("fau_au_logits")
        au_edge_logits = batch.get("fau_au_edge_logits")
        batch_size, num_frames, num_au, _ = fau_features.shape
        tokens = self.proj(fau_features)
        tokens = tokens.reshape(batch_size, num_frames * num_au, -1)
        time_ids = torch.arange(num_frames, device=fau_features.device).repeat_interleave(num_au)

        debug = {
            "input_shape": tuple(fau_features.shape),
            "feature_shape": tuple(fau_features.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "num_au": num_au,
            "au_logits": au_logits.detach() if isinstance(au_logits, torch.Tensor) else None,
            "au_edge_logits": (
                au_edge_logits.detach() if isinstance(au_edge_logits, torch.Tensor) else None
            ),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
