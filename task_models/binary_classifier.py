from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from fusion import FusionOutput
from pipeline import ClipFusionPipeline
from task_models.heads import BinaryFusionHead, build_binary_head, detach_diagnostics
from task_models.types import BinaryHeadResult, HeadDiagnostics


@dataclass(frozen=True)
class BinaryClassificationOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    fusion: FusionOutput
    labels: torch.Tensor | None = None
    paths: tuple[str, ...] = ()
    diagnostics: HeadDiagnostics = field(default_factory=dict)


class BinaryFusionClassifier(nn.Module):
    def __init__(self, pipeline: ClipFusionPipeline, head: nn.Module) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.head = head

    def forward(self, batch: Mapping[str, Any]) -> BinaryClassificationOutput:
        fusion = self.pipeline(batch)
        head_output = (
            self.head(fusion.fused)
            if isinstance(self.head, BinaryFusionHead)
            else self.head(fusion)
        )
        head_result = _head_result(head_output, detach=not self.training)
        return BinaryClassificationOutput(
            logits=head_result.logits,
            probabilities=torch.sigmoid(head_result.logits),
            fusion=fusion,
            labels=_optional_labels(batch),
            paths=_optional_paths(batch),
            diagnostics=head_result.diagnostics,
        )


def build_binary_fusion_classifier(
    pipeline: ClipFusionPipeline,
    dim: int,
    head_config: Mapping[str, Any] | None = None,
    modality_names: Sequence[str] | None = None,
) -> BinaryFusionClassifier:
    return BinaryFusionClassifier(
        pipeline=pipeline,
        head=build_binary_head(
            dim=dim,
            head_config=head_config,
            modality_names=modality_names,
        ),
    )


def _head_result(output: torch.Tensor | BinaryHeadResult, detach: bool) -> BinaryHeadResult:
    if isinstance(output, BinaryHeadResult):
        diagnostics = detach_diagnostics(output.diagnostics) if detach else output.diagnostics
        return BinaryHeadResult(
            logits=output.logits,
            diagnostics=diagnostics,
        )
    return BinaryHeadResult(logits=output)


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
