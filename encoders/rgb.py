from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from encoders.checkpoints import CheckpointLoadResult
from encoders.video_backbones import MViTV2SBackbone


class RGBEncoder(nn.Module):
    def __init__(
        self,
        frames: int = 16,
        image_size: int = 224,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = "mvit_v2_s"
        self.frames = frames
        self.image_size = image_size
        self.backbone = MViTV2SBackbone(
            temporal_size=frames,
            spatial_size=image_size,
            checkpoint_path=checkpoint_path,
        )
        self.feature_dim = self.backbone.feature_dim
        self.checkpoint_result: CheckpointLoadResult | None = None
        if self.backbone.checkpoint_result is not None:
            self.checkpoint_result = self.backbone.checkpoint_result

    def load_pretrained(self, checkpoint_path: str | Path) -> CheckpointLoadResult:
        self.checkpoint_result = self.backbone.load_pretrained(checkpoint_path)
        return self.checkpoint_result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim != 3:
            raise ValueError(
                "RGB backbone must return temporal features shaped [B, T, feature_dim], "
                f"got {tuple(features.shape)}"
            )
        return features
