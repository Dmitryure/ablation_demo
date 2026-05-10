from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from fusion import FusionOutput
from pipeline import ClipFusionPipeline
from task_models.heads import (
    _build_gated_attention,
    _build_modality_masks,
    _masked_softmax,
    _pool_by_modality,
    _split_fused_tokens,
)


@dataclass(frozen=True)
class GeneratorMultitaskOutput:
    binary_logits: torch.Tensor
    generator_logits: torch.Tensor
    binary_probabilities: torch.Tensor
    generator_probabilities: torch.Tensor
    fusion: FusionOutput
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


class MultitaskModalityGatedMILHead(nn.Module):
    def __init__(self, dim: int, num_generators: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if dim <= 0 or num_generators <= 0 or hidden_dim <= 0:
            raise ValueError("Head dimensions must be positive.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("`dropout` must be in [0.0, 1.0).")
        self.attention = _build_gated_attention(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.binary_expert = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.generator_expert = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_generators),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.binary_cls_projection = nn.Linear(dim, 1)
        self.generator_cls_projection = nn.Linear(dim, num_generators)

    def forward(
        self, fusion: FusionOutput
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        cls_token, tokens = _split_fused_tokens(fusion)
        modality_masks = _build_modality_masks(
            token_mask=fusion.token_mask,
            modality_ids=fusion.modality_ids,
            modality_names=fusion.modality_names,
        )
        attention_scores = self.attention(tokens).squeeze(-1)
        modality_pools, token_attention = _pool_by_modality(
            tokens=tokens,
            scores=attention_scores,
            modality_masks=modality_masks,
        )
        modality_valid_mask = modality_masks.any(dim=1)
        gate_context = torch.cat(
            [cls_token.unsqueeze(1).expand(-1, modality_pools.shape[1], -1), modality_pools],
            dim=2,
        )
        gate_weights = _masked_softmax(self.gate(gate_context).squeeze(-1), modality_valid_mask)
        binary_expert_logits = self.binary_expert(modality_pools).squeeze(-1)
        generator_expert_logits = self.generator_expert(modality_pools)
        binary_logits = self.binary_cls_projection(cls_token) + (
            gate_weights * binary_expert_logits
        ).sum(dim=1, keepdim=True)
        generator_logits = self.generator_cls_projection(cls_token) + (
            gate_weights.unsqueeze(-1) * generator_expert_logits
        ).sum(dim=1)
        return (
            binary_logits,
            generator_logits,
            {
                "modality_gate_weights": gate_weights,
                "binary_modality_expert_logits": binary_expert_logits,
                "generator_modality_expert_logits": generator_expert_logits,
                "modality_valid_mask": modality_valid_mask,
                "token_attention_weights": token_attention,
            },
        )


class GeneratorMultitaskClassifier(nn.Module):
    def __init__(
        self,
        pipeline: ClipFusionPipeline,
        dim: int,
        num_generators: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.head = MultitaskModalityGatedMILHead(
            dim=dim,
            num_generators=num_generators,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch: Mapping[str, Any]) -> GeneratorMultitaskOutput:
        fusion = self.pipeline(batch)
        binary_logits, generator_logits, diagnostics = self.head(fusion)
        return GeneratorMultitaskOutput(
            binary_logits=binary_logits,
            generator_logits=generator_logits,
            binary_probabilities=torch.sigmoid(binary_logits),
            generator_probabilities=torch.softmax(generator_logits, dim=-1),
            fusion=fusion,
            diagnostics=diagnostics,
        )


def build_generator_multitask_classifier(
    pipeline: ClipFusionPipeline,
    dim: int,
    num_generators: int,
    head_config: Mapping[str, Any] | None = None,
) -> GeneratorMultitaskClassifier:
    config = head_config or {}
    head_type = str(config.get("type", "modality_gated_mil"))
    if head_type != "modality_gated_mil":
        raise ValueError(f"Unsupported generator multitask head type: {head_type}")
    hidden_dim = int(config.get("hidden_dim", dim * 2))
    dropout = float(config.get("dropout", 0.0))
    return GeneratorMultitaskClassifier(
        pipeline=pipeline,
        dim=dim,
        num_generators=num_generators,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
