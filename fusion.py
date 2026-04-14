from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from branches import ModalityOutput


@dataclass(frozen=True)
class FusionOutput:
    fused: torch.Tensor
    tokens: torch.Tensor
    time_ids: torch.Tensor
    modality_ids: torch.Tensor
    modality_names: tuple[str, ...]
    cls_token: torch.Tensor
    fused_tokens: torch.Tensor


@dataclass(frozen=True)
class TokenBankBatch:
    tokens: torch.Tensor
    weighted_tokens: torch.Tensor
    time_ids: torch.Tensor
    modality_ids: torch.Tensor
    modality_names: tuple[str, ...]
    normalized_weights: dict[str, float]


def validate_modality_output(name: str, output: ModalityOutput) -> None:
    if output.tokens.ndim != 3:
        raise ValueError(
            f"Modality `{name}` tokens must have shape [B, N, dim], got {tuple(output.tokens.shape)}"
        )
    if output.time_ids.ndim != 1:
        raise ValueError(
            f"Modality `{name}` time_ids must have shape [N], got {tuple(output.time_ids.shape)}"
        )
    if output.tokens.shape[1] != output.time_ids.shape[0]:
        raise ValueError(
            f"Modality `{name}` token/time mismatch: "
            f"{output.tokens.shape[1]} tokens vs {output.time_ids.shape[0]} time ids"
        )


def build_modality_id_tensor(
    modality_name: str,
    token_count: int,
    modality_to_id: Mapping[str, int],
    device: torch.device,
) -> torch.Tensor:
    if modality_name not in modality_to_id:
        raise KeyError(f"Missing stable modality id for `{modality_name}`.")
    return torch.full(
        (token_count,),
        modality_to_id[modality_name],
        dtype=torch.long,
        device=device,
    )


def normalize_modality_weights(
    modality_names: Sequence[str],
    modality_weights: Mapping[str, float],
) -> dict[str, float]:
    raw_weights = {name: float(modality_weights[name]) for name in modality_names}
    total = sum(raw_weights.values())
    if total <= 0.0:
        raise ValueError("At least one modality weight must be greater than zero.")
    return {name: raw_weights[name] / total for name in modality_names}


def prepare_token_bank(
    outputs_by_name: Mapping[str, ModalityOutput],
    enabled_modalities: Sequence[str],
    modality_to_id: Mapping[str, int],
    modality_weights: Mapping[str, float],
) -> TokenBankBatch:
    tokens_by_modality: list[torch.Tensor] = []
    weighted_tokens_by_modality: list[torch.Tensor] = []
    time_ids_by_modality: list[torch.Tensor] = []
    modality_ids_by_modality: list[torch.Tensor] = []
    normalized_weights = normalize_modality_weights(enabled_modalities, modality_weights)

    for name in enabled_modalities:
        output = outputs_by_name[name]
        validate_modality_output(name, output)

        tokens = output.tokens
        time_ids = output.time_ids.to(device=tokens.device, dtype=torch.long)
        tokens_by_modality.append(tokens)
        weighted_tokens_by_modality.append(tokens * normalized_weights[name])
        time_ids_by_modality.append(time_ids)
        modality_ids_by_modality.append(
            build_modality_id_tensor(
                modality_name=name,
                token_count=tokens.shape[1],
                modality_to_id=modality_to_id,
                device=tokens.device,
            )
        )

    return TokenBankBatch(
        tokens=torch.cat(tokens_by_modality, dim=1),
        weighted_tokens=torch.cat(weighted_tokens_by_modality, dim=1),
        time_ids=torch.cat(time_ids_by_modality, dim=0),
        modality_ids=torch.cat(modality_ids_by_modality, dim=0),
        modality_names=tuple(enabled_modalities),
        normalized_weights=normalized_weights,
    )


class TokenBankFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        max_time_steps: int,
        num_modalities: int,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("`dim` must be positive.")
        if num_layers <= 0:
            raise ValueError("`num_layers` must be positive.")
        if num_heads <= 0:
            raise ValueError("`num_heads` must be positive.")
        if dim % num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        if mlp_ratio <= 0.0:
            raise ValueError("`mlp_ratio` must be positive.")
        if max_time_steps <= 0:
            raise ValueError("`max_time_steps` must be positive.")
        if num_modalities <= 0:
            raise ValueError("`num_modalities` must be positive.")

        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("`mlp_ratio` produced a non-positive hidden dimension.")

        self.dim = dim
        self.max_time_steps = max_time_steps
        self.num_modalities = num_modalities
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_embedding = nn.Embedding(max_time_steps, dim)
        self.modality_embedding = nn.Embedding(num_modalities, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(dim)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        time_ids: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must have shape [B, N, dim], got {tuple(tokens.shape)}")
        if time_ids.ndim != 1:
            raise ValueError(f"`time_ids` must have shape [N], got {tuple(time_ids.shape)}")
        if modality_ids.ndim != 1:
            raise ValueError(
                f"`modality_ids` must have shape [N], got {tuple(modality_ids.shape)}"
            )
        if tokens.shape[1] != time_ids.shape[0] or tokens.shape[1] != modality_ids.shape[0]:
            raise ValueError("Token bank length must match time_ids and modality_ids length.")
        if tokens.shape[2] != self.dim:
            raise ValueError(f"Token dim mismatch: expected {self.dim}, got {tokens.shape[2]}")
        if time_ids.numel() and int(time_ids.max().item()) >= self.max_time_steps:
            raise ValueError(
                f"time_ids exceed configured max_time_steps={self.max_time_steps}: "
                f"max time_id is {int(time_ids.max().item())}"
            )
        if time_ids.numel() and int(time_ids.min().item()) < 0:
            raise ValueError("time_ids must be non-negative.")
        if modality_ids.numel() and int(modality_ids.max().item()) >= self.num_modalities:
            raise ValueError(
                f"modality_ids exceed configured num_modalities={self.num_modalities}: "
                f"max modality_id is {int(modality_ids.max().item())}"
            )
        if modality_ids.numel() and int(modality_ids.min().item()) < 0:
            raise ValueError("modality_ids must be non-negative.")

        token_states = tokens
        token_states = token_states + self.time_embedding(time_ids).unsqueeze(0)
        token_states = token_states + self.modality_embedding(modality_ids).unsqueeze(0)

        batch_size = token_states.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        fused_tokens = torch.cat([cls_tokens, token_states], dim=1)
        fused_tokens = self.encoder(fused_tokens)
        fused_tokens = self.output_norm(fused_tokens)
        cls_token = fused_tokens[:, 0, :]
        return cls_token, fused_tokens
