from __future__ import annotations

from typing import Mapping, Tuple

import torch
import torch.nn as nn

from branches.base import ModalityBranch, ModalityOutput, mlp


class FaceMeshBranch(ModalityBranch):
    name = "face_mesh"

    def __init__(self, dim: int):
        super().__init__()
        self.proj = mlp(3, dim, dim)

    def required_keys(self) -> Tuple[str, ...]:
        return ("face_mesh",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        face_mesh = batch["face_mesh"]
        if face_mesh.ndim != 4 or face_mesh.shape[-1] != 3:
            raise ValueError(
                "`face_mesh` must have shape [B, N, num_points, 3], "
                f"got {tuple(face_mesh.shape)}"
            )

        batch_size, num_frames, num_points, _ = face_mesh.shape
        tokens = self.proj(face_mesh).reshape(batch_size, num_frames * num_points, -1)
        time_ids = torch.arange(num_frames, device=face_mesh.device).repeat_interleave(num_points)
        debug = {
            "input_shape": tuple(face_mesh.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "num_points": num_points,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
