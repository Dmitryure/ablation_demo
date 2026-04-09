from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput


class FAUBranch(ModalityBranch):
    name = "fau"

    def __init__(self, encoder: nn.Module, dim: int):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.LazyLinear(dim)

    def required_keys(self) -> Tuple[str, ...]:
        return ("video",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        video = batch["video"]
        if video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, 3, N, H, W], got {tuple(video.shape)}")

        batch_size, channels, num_frames, height, width = video.shape
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        encoded = self.encoder(frame_batch)

        if isinstance(encoded, tuple):
            fau_features = encoded[0]
            au_logits = encoded[1] if len(encoded) > 1 else None
            au_edge_logits = encoded[2] if len(encoded) > 2 else None
        else:
            fau_features = encoded
            au_logits = None
            au_edge_logits = None

        if fau_features.ndim != 3 or fau_features.shape[0] != batch_size * num_frames:
            raise ValueError(
                "FAU encoder must return per-frame features shaped [B*N, num_au, feature_dim], "
                f"got {tuple(fau_features.shape)}"
            )

        tokens = self.proj(fau_features)
        num_au = tokens.shape[1]
        tokens = tokens.view(batch_size, num_frames, num_au, -1).reshape(
            batch_size, num_frames * num_au, -1
        )
        time_ids = torch.arange(num_frames, device=video.device).repeat_interleave(num_au)

        debug = {
            "input_shape": tuple(video.shape),
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
