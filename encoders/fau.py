from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from encoders.checkpoints import CheckpointLoadResult, load_checkpoint
from encoders.megraphau.mefl import MEFARG


class FAUEncoder(MEFARG):
    def __init__(
        self,
        num_classes: int = 12,
        backbone: str = "swin_transformer_tiny",
        checkpoint_path: str | Path | None = None,
    ):
        super().__init__(num_classes=num_classes, backbone=backbone)
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.checkpoint_result: CheckpointLoadResult | None = None
        if checkpoint_path is not None:
            self.checkpoint_result = self.load_pretrained(checkpoint_path)

    def load_pretrained(self, checkpoint_path: str | Path) -> CheckpointLoadResult:
        self.checkpoint_result = load_checkpoint(
            self,
            checkpoint_path,
            prefixes=("fau_encoder.", "au_encoder.", "encoder.", "model."),
        )
        return self.checkpoint_result

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        x = self.global_linear(x)

        f_u = []
        for layer in self.head.class_linears:
            f_u.append(layer(x).unsqueeze(1))
        f_u = torch.cat(f_u, dim=1)
        f_v = f_u.mean(dim=-2)

        f_e = self.head.edge_extractor(f_u, x)
        f_e = f_e.mean(dim=-2)
        f_v, f_e = self.head.gnn(f_v, f_e)

        batch_size, num_classes, channels = f_v.shape
        sc = self.head.relu(self.head.sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_v, p=2, dim=-1)
        cl = (cl * sc.view(1, num_classes, channels)).sum(dim=-1)
        cl_edge = self.head.edge_fc(f_e)
        return f_v, cl, cl_edge
