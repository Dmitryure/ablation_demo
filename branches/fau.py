from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput
from branches.compression import (
    DEFAULT_SLOT_COUNTS,
    TemporalLatentQueryPooling,
    TemporalPositionEncoding,
    validate_positive_int,
)


def reshape_to_au_tracks(projected_tokens: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, num_au, dim = projected_tokens.shape
    return projected_tokens.permute(0, 2, 1, 3).reshape(batch_size * num_au, num_frames, dim)


def reshape_to_clip_tokens(
    au_track_tokens: torch.Tensor,
    batch_size: int,
    num_frames: int,
    num_au: int,
) -> torch.Tensor:
    return (
        au_track_tokens.reshape(batch_size, num_au, num_frames, -1)
        .permute(0, 2, 1, 3)
        .reshape(batch_size, num_frames * num_au, -1)
    )


class FAUBranch(ModalityBranch):
    name = "fau"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["fau"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "fau.slot_count")
        self.proj = nn.LazyLinear(dim)
        self.temporal_position_encoding = TemporalPositionEncoding(dim)
        self.clip_pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count)

    def required_keys(self) -> tuple[str, ...]:
        return ("fau_features",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        fau_features = batch["fau_features"]
        if fau_features.ndim != 4:
            raise ValueError(
                "FAU features must have shape [B, N, num_au, feature_dim], "
                f"got {tuple(fau_features.shape)}"
            )

        batch_size, num_frames, num_au, _ = fau_features.shape
        projected_tokens = self.proj(fau_features)
        au_track_tokens = reshape_to_au_tracks(projected_tokens)
        temporal_encoding = self.temporal_position_encoding(
            num_tokens=num_frames,
            device=fau_features.device,
            dtype=projected_tokens.dtype,
        )
        au_track_tokens = au_track_tokens + temporal_encoding.unsqueeze(0)
        clip_tokens = reshape_to_clip_tokens(au_track_tokens, batch_size, num_frames, num_au)
        tokens = self.clip_pool(clip_tokens)
        time_ids = torch.arange(self.slot_count, device=fau_features.device)

        debug = {
            "input_shape": tuple(fau_features.shape),
            "feature_shape": tuple(fau_features.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "au_track_token_shape": tuple(au_track_tokens.shape),
            "temporal_encoding_shape": tuple(temporal_encoding.shape),
            "clip_token_shape": tuple(clip_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "num_au": num_au,
            "slot_count": self.slot_count,
            "ignored_optional_keys": tuple(
                key
                for key in ("fau_au_logits", "fau_au_edge_logits")
                if isinstance(batch.get(key), torch.Tensor)
            ),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
