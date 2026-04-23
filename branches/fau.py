from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput
from branches.compression import (
    DEFAULT_SLOT_COUNTS,
    LatentQueryPooling,
    TemporalLatentQueryPooling,
    validate_positive_int,
)


FRAME_QUERY_TOKENS = 2


class FAUBranch(ModalityBranch):
    name = "fau"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["fau"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "fau.slot_count")
        self.proj = nn.LazyLinear(dim)
        self.frame_pool = LatentQueryPooling(dim=dim, output_tokens=FRAME_QUERY_TOKENS)
        self.clip_pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count)

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
        projected_tokens = self.proj(fau_features)
        frame_tokens = self.frame_pool(projected_tokens.reshape(batch_size * num_frames, num_au, -1))
        clip_tokens = frame_tokens.reshape(batch_size, num_frames * FRAME_QUERY_TOKENS, -1)
        tokens = self.clip_pool(clip_tokens)
        time_ids = torch.arange(self.slot_count, device=fau_features.device)

        debug = {
            "input_shape": tuple(fau_features.shape),
            "feature_shape": tuple(fau_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "frame_token_shape": tuple(frame_tokens.shape),
            "clip_token_shape": tuple(clip_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "num_au": num_au,
            "frame_query_tokens": FRAME_QUERY_TOKENS,
            "slot_count": self.slot_count,
            "au_logits": au_logits.detach() if isinstance(au_logits, torch.Tensor) else None,
            "au_edge_logits": (
                au_edge_logits.detach() if isinstance(au_edge_logits, torch.Tensor) else None
            ),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
