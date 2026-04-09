from __future__ import annotations

from pathlib import Path

import torch

from encoders.checkpoints import CheckpointLoadResult, load_checkpoint
from encoders.physnet import PhysNetPaddingEncoderDecoderMax


class RPPGEncoder(PhysNetPaddingEncoderDecoderMax):
    def __init__(self, frames: int = 16, checkpoint_path: str | Path | None = None):
        super().__init__(frames=frames)
        self.frames = frames
        self.out_channels = self.ConvBlock10.in_channels
        self.checkpoint_result: CheckpointLoadResult | None = None
        if checkpoint_path is not None:
            self.checkpoint_result = self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path: str | Path) -> CheckpointLoadResult:
        self.checkpoint_result = load_checkpoint(
            self,
            checkpoint_path,
            prefixes=("rppg_encoder.", "phys_encoder.", "encoder.", "model."),
        )
        return self.checkpoint_result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, _, _ = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x_visual6464 = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x_visual6464)
        x = self.ConvBlock4(x)
        x_visual3232 = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x_visual3232)
        x = self.ConvBlock6(x)
        x_visual1616 = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x_visual1616)
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)
        pooled = self.poolspa(x)
        x = self.ConvBlock10(pooled)

        waveform = x.view(-1, num_frames)
        temporal_features = pooled.squeeze(-1).squeeze(-1).transpose(1, 2)
        return waveform, temporal_features
