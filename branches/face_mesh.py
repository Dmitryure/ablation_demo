from __future__ import annotations

from typing import Mapping, Tuple

import torch

from branches.base import ModalityBranch, ModalityOutput, mlp
from branches.compression import DEFAULT_OUTPUT_TOKENS, LatentQueryPooling, validate_positive_int


class FaceMeshBranch(ModalityBranch):
    name = "face_mesh"

    def __init__(self, dim: int, output_tokens_per_frame: int = DEFAULT_OUTPUT_TOKENS["face_mesh"]):
        super().__init__()
        self.output_tokens_per_frame = validate_positive_int(
            output_tokens_per_frame,
            "face_mesh.output_tokens_per_frame",
        )
        self.proj = mlp(3, dim, dim)
        self.pool = LatentQueryPooling(dim=dim, output_tokens=self.output_tokens_per_frame)

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
        projected_tokens = self.proj(face_mesh)
        pooled_tokens = self.pool(projected_tokens.reshape(batch_size * num_frames, num_points, -1))
        tokens = pooled_tokens.reshape(batch_size, num_frames * self.output_tokens_per_frame, -1)
        time_ids = torch.arange(num_frames, device=face_mesh.device).repeat_interleave(
            self.output_tokens_per_frame
        )
        debug = {
            "input_shape": tuple(face_mesh.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "num_points": num_points,
            "output_tokens_per_frame": self.output_tokens_per_frame,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
