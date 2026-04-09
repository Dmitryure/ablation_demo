from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from encoders.checkpoints import CheckpointLoadResult, load_checkpoint
from encoders.image_backbones.resnet import (
    resnet101,
    resnet152,
    resnet18,
    resnet34,
    resnet50,
)


RGB_BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


class RGBEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        checkpoint_path: str | Path | None = None,
    ):
        super().__init__()
        if backbone not in RGB_BACKBONES:
            raise ValueError(
                "Unsupported RGB backbone. Expected one of "
                "`resnet18`, `resnet34`, `resnet50`, `resnet101`, or `resnet152`."
            )
        self.backbone_name = backbone
        self.backbone = RGB_BACKBONES[backbone](pretrained=False)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = None
        self.checkpoint_result: CheckpointLoadResult | None = None
        if checkpoint_path is not None:
            self.checkpoint_result = self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path: str | Path) -> CheckpointLoadResult:
        self.checkpoint_result = load_checkpoint(
            self.backbone,
            checkpoint_path,
            prefixes=("rgb_encoder.", "encoder.", "backbone.", "model."),
        )
        return self.checkpoint_result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim != 3:
            raise ValueError(
                "RGB backbone must return spatial features shaped [B, spatial_tokens, feature_dim], "
                f"got {tuple(features.shape)}"
            )
        return features.mean(dim=1)
