from __future__ import annotations

from collections.abc import Mapping

import torch

from branches.base import ModalityBranch, ModalityOutput, mlp
from branches.compression import (
    DEFAULT_SLOT_COUNTS,
    LatentQueryPooling,
    TemporalLatentQueryPooling,
    validate_positive_int,
)

POINT_QUERY_TOKENS = 1


class FaceMeshBranch(ModalityBranch):
    name = "face_mesh"

    def __init__(self, dim: int, slot_count: int = DEFAULT_SLOT_COUNTS["face_mesh"]):
        super().__init__()
        self.slot_count = validate_positive_int(slot_count, "face_mesh.slot_count")
        self.proj = mlp(3, dim, dim)
        self.point_pool = LatentQueryPooling(dim=dim, output_tokens=POINT_QUERY_TOKENS)
        self.clip_pool = TemporalLatentQueryPooling(dim=dim, output_tokens=self.slot_count)

    def required_keys(self) -> tuple[str, ...]:
        return ("face_mesh",)

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        face_mesh = batch["face_mesh"]
        if face_mesh.ndim != 4 or face_mesh.shape[-1] != 3:
            raise ValueError(
                f"`face_mesh` must have shape [B, N, num_points, 3], got {tuple(face_mesh.shape)}"
            )

        batch_size, num_frames, num_points, _ = face_mesh.shape
        projected_tokens = self.proj(face_mesh)
        frame_tokens = self.point_pool(
            projected_tokens.reshape(batch_size * num_frames, num_points, -1)
        )
        clip_tokens = frame_tokens.reshape(batch_size, num_frames * POINT_QUERY_TOKENS, -1)
        tokens = self.clip_pool(clip_tokens)
        time_ids = torch.arange(self.slot_count, device=face_mesh.device)
        debug = {
            "input_shape": tuple(face_mesh.shape),
            "projected_token_shape": tuple(projected_tokens.shape),
            "frame_token_shape": tuple(frame_tokens.shape),
            "clip_token_shape": tuple(clip_tokens.shape),
            "token_shape": tuple(tokens.shape),
            "token_count": tokens.shape[1],
            "num_frames": num_frames,
            "num_points": num_points,
            "point_query_tokens": POINT_QUERY_TOKENS,
            "slot_count": self.slot_count,
        }
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug=debug)
