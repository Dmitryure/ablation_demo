from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from encoders.checkpoints import CheckpointLoadResult, load_checkpoint


def _require_torchvision_mvit() -> object:
    try:
        from torchvision.models.video import mvit_v2_s
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RGB MViT backbone needs `torchvision`. Install repo requirements in the local venv."
        ) from exc
    return mvit_v2_s


def _extract_token_states(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    x = model.conv_proj(x)
    x = x.flatten(2).transpose(1, 2)
    x = model.pos_encoding(x)

    thw = (model.pos_encoding.temporal_size,) + model.pos_encoding.spatial_size
    for block in model.blocks:
        x, thw = block(x, thw)
    x = model.norm(x)
    return x, thw


def _pool_spatial_tokens(
    token_states: torch.Tensor,
    thw: tuple[int, int, int],
) -> torch.Tensor:
    if token_states.ndim != 3:
        raise ValueError(
            "MViT token states must have shape [B, N, feature_dim], "
            f"got {tuple(token_states.shape)}"
        )

    temporal_size, height_tokens, width_tokens = (int(size) for size in thw)
    batch_size, token_count_with_cls, feature_dim = token_states.shape
    patch_token_count = token_count_with_cls - 1
    expected_patch_token_count = temporal_size * height_tokens * width_tokens
    if patch_token_count != expected_patch_token_count:
        raise ValueError(
            "MViT token grid mismatch: "
            f"{patch_token_count} patch tokens vs expected {expected_patch_token_count} from thw={thw}"
        )

    patch_tokens = token_states[:, 1:, :]
    patch_tokens = patch_tokens.reshape(
        batch_size,
        temporal_size,
        height_tokens * width_tokens,
        feature_dim,
    )
    return patch_tokens.mean(dim=2)


class MViTV2SBackbone(nn.Module):
    def __init__(
        self,
        temporal_size: int = 16,
        spatial_size: int = 224,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if temporal_size != 16:
            raise ValueError("`mvit_v2_s` in this repo currently supports only 16-frame clips.")
        if spatial_size != 224:
            raise ValueError("`mvit_v2_s` in this repo currently supports only image_size=224.")

        mvit_v2_s = _require_torchvision_mvit()
        self.temporal_size = temporal_size
        self.spatial_size = spatial_size
        self.model = mvit_v2_s(weights=None)
        self.feature_dim = self.model.head[-1].in_features
        self.checkpoint_result: CheckpointLoadResult | None = None
        if checkpoint_path is not None:
            self.checkpoint_result = self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path: str | Path) -> CheckpointLoadResult:
        self.checkpoint_result = load_checkpoint(
            self.model,
            checkpoint_path,
            prefixes=(
                "rgb_encoder.backbone.model.",
                "encoder.backbone.model.",
                "backbone.model.",
                "rgb_encoder.",
                "encoder.",
                "backbone.",
                "model.",
            ),
        )
        return self.checkpoint_result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"MViT input must have shape [B, 3, T, H, W], got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"MViT input must have 3 channels, got {x.shape[1]}")
        if x.shape[2] != self.temporal_size:
            raise ValueError(
                f"MViT temporal size mismatch: expected {self.temporal_size}, got {x.shape[2]}"
            )
        if x.shape[3] != self.spatial_size or x.shape[4] != self.spatial_size:
            raise ValueError(
                "MViT spatial size mismatch: "
                f"expected {(self.spatial_size, self.spatial_size)}, got {(x.shape[3], x.shape[4])}"
            )

        token_states, thw = _extract_token_states(self.model, x)
        temporal_tokens = _pool_spatial_tokens(token_states, thw)
        expected_temporal_tokens = math.ceil(self.temporal_size / 2)
        if temporal_tokens.shape[1] != expected_temporal_tokens:
            raise ValueError(
                "Unexpected MViT temporal token count: "
                f"expected {expected_temporal_tokens}, got {temporal_tokens.shape[1]}"
            )
        return temporal_tokens
