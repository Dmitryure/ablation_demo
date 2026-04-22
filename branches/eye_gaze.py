from __future__ import annotations

from typing import Mapping, Tuple

import torch

from branches.base import ModalityBranch, ModalityOutput, mlp
from branches.compression import (
    DEFAULT_OUTPUT_TOKENS,
    TemporalLatentQueryPooling,
    validate_positive_int,
)


class EyeGazeBranch(ModalityBranch):
    name = "eye_gaze"

    def __init__(self, dim: int, output_tokens_per_clip: int = DEFAULT_OUTPUT_TOKENS["eye_gaze"]):
        super().__init__()
        self.output_tokens_per_clip = validate_positive_int(
            output_tokens_per_clip,
            "eye_gaze.output_tokens_per_clip",
        )
        self.proj = mlp(8, dim, dim)
        self.pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.output_tokens_per_clip)

    def required_keys(self) -> Tuple[str, ...]:
        return ("eye_gaze",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        eye_gaze = batch["eye_gaze"]
        if eye_gaze.ndim != 3 or eye_gaze.shape[-1] != 8:
            raise ValueError(f"`eye_gaze` must have shape [B, N, 8], got {tuple(eye_gaze.shape)}")

        projected_tokens = self.proj(eye_gaze)
        tokens = self.pool(projected_tokens)
        num_frames = eye_gaze.shape[1]
        time_ids = torch.arange(self.output_tokens_per_clip, device=eye_gaze.device)
        debug = {
            "input_shape": tuple(eye_gaze.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "output_tokens_per_clip": self.output_tokens_per_clip,
            "time_ids": tuple(time_ids.tolist()),
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
