from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from fusion import FusionOutput
from task_models.types import BinaryHeadResult, HeadDiagnostics


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


class CLSMLPBinaryHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        _validate_head_dims(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fusion: FusionOutput) -> torch.Tensor:
        _validate_fusion_dim(fusion.fused, "`fusion.fused`")
        return self.classifier(fusion.fused)


class AttentionMILBinaryHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        _validate_head_dims(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.attention = _build_gated_attention(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fusion: FusionOutput) -> BinaryHeadResult:
        cls_token, tokens = _split_fused_tokens(fusion)
        weights = _masked_softmax(self.attention(tokens).squeeze(-1), fusion.token_mask)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
        logits = self.classifier(torch.cat([cls_token, pooled], dim=1))
        return BinaryHeadResult(
            logits=logits,
            diagnostics={"token_attention_weights": weights},
        )


class ModalityGatedMILBinaryHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        _validate_head_dims(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.attention = _build_gated_attention(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
        self.expert = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.cls_projection = nn.Linear(dim, 1)

    def forward(self, fusion: FusionOutput) -> BinaryHeadResult:
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
        expert_logits = self.expert(modality_pools).squeeze(-1)
        gate_context = torch.cat(
            [cls_token.unsqueeze(1).expand(-1, modality_pools.shape[1], -1), modality_pools],
            dim=2,
        )
        gate_weights = _masked_softmax(self.gate(gate_context).squeeze(-1), modality_valid_mask)
        mixed_logit = (gate_weights * expert_logits).sum(dim=1, keepdim=True)
        logits = self.cls_projection(cls_token) + mixed_logit
        return BinaryHeadResult(
            logits=logits,
            diagnostics={
                "modality_gate_weights": gate_weights,
                "modality_expert_logits": expert_logits,
                "modality_valid_mask": modality_valid_mask,
                "token_attention_weights": token_attention,
            },
        )


def build_binary_head(
    dim: int,
    head_config: Mapping[str, Any] | None = None,
    modality_names: Sequence[str] | None = None,
) -> nn.Module:
    del modality_names
    config = head_config or {}
    head_type = str(config.get("type", "cls_linear"))
    hidden_dim = int(config.get("hidden_dim", dim * 2))
    dropout = float(config.get("dropout", 0.0))
    if head_type == "cls_linear":
        return BinaryFusionHead(dim=dim)
    if head_type == "cls_mlp":
        return CLSMLPBinaryHead(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    if head_type == "attention_mil":
        return AttentionMILBinaryHead(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    if head_type == "modality_gated_mil":
        return ModalityGatedMILBinaryHead(dim=dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unsupported binary head type: {head_type}")


def _validate_head_dims(dim: int, hidden_dim: int, dropout: float) -> None:
    if dim <= 0:
        raise ValueError("`dim` must be positive.")
    if hidden_dim <= 0:
        raise ValueError("`hidden_dim` must be positive.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("`dropout` must be in [0.0, 1.0).")


def _validate_fusion_dim(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [B, dim], got {tuple(tensor.shape)}")


def _split_fused_tokens(fusion: FusionOutput) -> tuple[torch.Tensor, torch.Tensor]:
    if fusion.fused_tokens.ndim != 3:
        raise ValueError(
            f"`fusion.fused_tokens` must have shape [B, N + 1, dim], "
            f"got {tuple(fusion.fused_tokens.shape)}"
        )
    if fusion.fused_tokens.shape[1] != fusion.token_mask.shape[0] + 1:
        raise ValueError("`fusion.fused_tokens` length must match token_mask plus CLS token.")
    return fusion.fused_tokens[:, 0, :], fusion.fused_tokens[:, 1:, :]


def _build_gated_attention(dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"`scores` must have shape [B, N], got {tuple(scores.shape)}")
    if mask.ndim != 1 or mask.shape[0] != scores.shape[1]:
        raise ValueError("`mask` must have shape [N] and match scores length.")
    valid_mask = mask.to(device=scores.device, dtype=torch.bool).unsqueeze(0)
    has_valid = valid_mask.any(dim=1, keepdim=True)
    masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
    max_scores = masked_scores.max(dim=1, keepdim=True).values
    max_scores = torch.where(has_valid, max_scores, torch.zeros_like(max_scores))
    shifted = masked_scores - max_scores
    exp_scores = torch.exp(shifted) * valid_mask.to(dtype=scores.dtype)
    return exp_scores / exp_scores.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(scores.dtype).tiny
    )


def _build_modality_masks(
    token_mask: torch.Tensor,
    modality_ids: torch.Tensor,
    modality_names: Sequence[str],
) -> torch.Tensor:
    if token_mask.ndim != 1:
        raise ValueError(f"`token_mask` must have shape [N], got {tuple(token_mask.shape)}")
    if modality_ids.ndim != 1 or modality_ids.shape[0] != token_mask.shape[0]:
        raise ValueError("`modality_ids` must have shape [N] and match token_mask length.")
    modality_id_order = _modality_id_order(modality_ids, len(modality_names))
    masks = [
        (modality_ids == modality_id).to(dtype=torch.bool) & token_mask.to(dtype=torch.bool)
        for modality_id in modality_id_order
    ]
    if not masks:
        return torch.zeros((0, token_mask.shape[0]), dtype=torch.bool, device=token_mask.device)
    return torch.stack(masks, dim=0)


def _modality_id_order(modality_ids: torch.Tensor, expected_count: int) -> list[int]:
    ordered_ids: list[int] = []
    seen: set[int] = set()
    for modality_id in modality_ids.detach().cpu().tolist():
        int_id = int(modality_id)
        if int_id not in seen:
            seen.add(int_id)
            ordered_ids.append(int_id)
        if len(ordered_ids) == expected_count:
            break
    return ordered_ids


def _pool_by_modality(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    modality_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pools: list[torch.Tensor] = []
    token_attention = torch.zeros_like(scores)
    for modality_mask in modality_masks:
        weights = _masked_softmax(scores, modality_mask)
        pools.append(torch.bmm(weights.unsqueeze(1), tokens).squeeze(1))
        token_attention = token_attention + weights
    if not pools:
        batch_size, _, dim = tokens.shape
        return tokens.new_zeros((batch_size, 0, dim)), token_attention
    return torch.stack(pools, dim=1), token_attention


def detach_diagnostics(diagnostics: Mapping[str, torch.Tensor]) -> HeadDiagnostics:
    return {key: value.detach() for key, value in diagnostics.items()}
