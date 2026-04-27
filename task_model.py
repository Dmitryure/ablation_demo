from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from fusion import FusionOutput
from pipeline import ClipFusionPipeline


@dataclass(frozen=True)
class BinaryClassificationOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    fusion: FusionOutput
    labels: torch.Tensor | None = None
    paths: tuple[str, ...] = ()


class BinaryFusionHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("`dim` must be positive.")
        self.projection = nn.Linear(dim, 1)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim != 2:
            raise ValueError(f"`fused` must have shape [B, dim], got {tuple(fused.shape)}")
        return self.projection(fused)


def _optional_labels(batch: Mapping[str, Any]) -> torch.Tensor | None:
    label = batch.get("label")
    return label if isinstance(label, torch.Tensor) else None


def _optional_paths(batch: Mapping[str, Any]) -> tuple[str, ...]:
    paths = batch.get("path")
    if isinstance(paths, str):
        return (paths,)
    if isinstance(paths, (list, tuple)) and all(isinstance(path, str) for path in paths):
        return tuple(paths)
    return ()


class BinaryFusionClassifier(nn.Module):
    def __init__(self, pipeline: ClipFusionPipeline, head: BinaryFusionHead) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.head = head

    def forward(self, batch: Mapping[str, Any]) -> BinaryClassificationOutput:
        fusion = self.pipeline(batch)
        logits = self.head(fusion.fused)
        return BinaryClassificationOutput(
            logits=logits,
            probabilities=torch.sigmoid(logits),
            fusion=fusion,
            labels=_optional_labels(batch),
            paths=_optional_paths(batch),
        )


def build_binary_fusion_classifier(pipeline: ClipFusionPipeline, dim: int) -> BinaryFusionClassifier:
    return BinaryFusionClassifier(
        pipeline=pipeline,
        head=BinaryFusionHead(dim=dim),
    )
