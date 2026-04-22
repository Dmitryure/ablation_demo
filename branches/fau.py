from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput
from branches.compression import DEFAULT_OUTPUT_TOKENS, LatentQueryPooling, validate_positive_int


class FAUBranch(ModalityBranch):
    name = "fau"

    def __init__(self, dim: int, output_tokens_per_frame: int = DEFAULT_OUTPUT_TOKENS["fau"]):
        super().__init__()
        self.output_tokens_per_frame = validate_positive_int(
            output_tokens_per_frame,
            "fau.output_tokens_per_frame",
        )
        self.proj = nn.LazyLinear(dim)
        self.pool = LatentQueryPooling(dim=dim, output_tokens=self.output_tokens_per_frame)

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
        pooled_tokens = self.pool(projected_tokens.reshape(batch_size * num_frames, num_au, -1))
        tokens = pooled_tokens.reshape(batch_size, num_frames * self.output_tokens_per_frame, -1)
        time_ids = torch.arange(num_frames, device=fau_features.device).repeat_interleave(
            self.output_tokens_per_frame
        )

        debug = {
            "input_shape": tuple(fau_features.shape),
            "feature_shape": tuple(fau_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "num_au": num_au,
            "output_tokens_per_frame": self.output_tokens_per_frame,
            "au_logits": au_logits.detach() if isinstance(au_logits, torch.Tensor) else None,
            "au_edge_logits": (
                au_edge_logits.detach() if isinstance(au_edge_logits, torch.Tensor) else None
            ),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
