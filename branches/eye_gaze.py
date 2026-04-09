from __future__ import annotations

from typing import Mapping, Tuple

import torch

from branches.base import ModalityBranch, ModalityOutput, mlp


class EyeGazeBranch(ModalityBranch):
    name = "eye_gaze"

    def __init__(self, dim: int):
        super().__init__()
        self.proj = mlp(8, dim, dim)

    def required_keys(self) -> Tuple[str, ...]:
        return ("eye_gaze",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        eye_gaze = batch["eye_gaze"]
        if eye_gaze.ndim != 3 or eye_gaze.shape[-1] != 8:
            raise ValueError(f"`eye_gaze` must have shape [B, N, 8], got {tuple(eye_gaze.shape)}")

        tokens = self.proj(eye_gaze)
        num_frames = eye_gaze.shape[1]
        time_ids = torch.arange(num_frames, device=eye_gaze.device)
        debug = {
            "input_shape": tuple(eye_gaze.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
