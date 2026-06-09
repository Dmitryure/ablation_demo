from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from frame_config import resolve_modality_frame_count, validate_frame_count

OUTPUT_TOKEN_CONFIG_KEYS = {
    "rgb": "slot_count",
    "fau": "slot_count",
    "rppg": "slot_count",
    "eye_gaze": "slot_count",
    "face_mesh": "slot_count",
    "depth": "slot_count",
    "fft": "slot_count",
    "stft": "slot_count",
}

DEFAULT_SLOT_COUNTS = {
    "rgb": 8,
    "fau": 32,
    "rppg": 4,
    "eye_gaze": 4,
    "face_mesh": 16,
    "depth": 4,
    "fft": 4,
    "stft": 4,
}


def validate_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{field_name}` must be a positive integer.")
    return value


def resolve_slot_count(config: Mapping[str, Any] | None, modality_name: str) -> int:
    if modality_name not in DEFAULT_SLOT_COUNTS:
        raise KeyError(f"Unsupported slot-count config lookup for modality `{modality_name}`.")

    default_value = DEFAULT_SLOT_COUNTS[modality_name]
    if config is None:
        return default_value

    section = config.get(modality_name, {})
    if section is None:
        return default_value
    if not isinstance(section, Mapping):
        raise ValueError(f"`{modality_name}` must be a YAML mapping when provided.")

    field_name = OUTPUT_TOKEN_CONFIG_KEYS[modality_name]
    value = section.get(field_name)
    if value is None:
        return default_value
    return validate_positive_int(value, f"{modality_name}.{field_name}")


def require_modality_frames(config: Mapping[str, Any]) -> int:
    frames = config.get("frames")
    if isinstance(frames, Mapping):
        return validate_frame_count(frames.get("default"), "frames.default")
    return validate_frame_count(frames, "frames")


def require_frame_count(config: Mapping[str, Any], modality_name: str) -> int:
    return resolve_modality_frame_count(config, modality_name)


def required_fusion_time_steps(config: Mapping[str, Any], modality_name: str) -> int | None:
    if modality_name not in DEFAULT_SLOT_COUNTS:
        return None
    return resolve_slot_count(config, modality_name)


def validate_branch_token_config(
    config: Mapping[str, Any] | None,
    modalities: Sequence[str] = DEFAULT_SLOT_COUNTS.keys(),
    fusion_max_time_steps: int | None = None,
) -> None:
    if config is None:
        return

    for modality_name in modalities:
        if modality_name in DEFAULT_SLOT_COUNTS:
            resolve_slot_count(config, modality_name)

    if fusion_max_time_steps is None:
        return

    max_time_steps = validate_positive_int(fusion_max_time_steps, "fusion.max_time_steps")
    for modality_name in modalities:
        required_steps = required_fusion_time_steps(config, modality_name)
        if required_steps is not None and required_steps > max_time_steps:
            raise ValueError(
                f"`fusion.max_time_steps`={max_time_steps} is too small for modality "
                f"`{modality_name}`; requires at least {required_steps}."
            )


class TemporalPositionEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        validate_positive_int(dim, "dim")
        self.dim = dim
        self.position_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        num_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(num_tokens, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.dim)
        )
        positional_encoding = torch.zeros(num_tokens, self.dim, device=device, dtype=dtype)
        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        positional_encoding[:, 1::2] = torch.cos(
            positions * div_term[: positional_encoding[:, 1::2].shape[1]]
        )
        position_scale = self.position_scale.to(device=device, dtype=dtype)
        return position_scale * positional_encoding


class LatentQueryPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        output_tokens: int,
        num_layers: int = 2,
        num_heads: int | None = None,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        validate_positive_int(dim, "dim")
        self.dim = dim
        self.output_tokens = validate_positive_int(output_tokens, "output_tokens")
        self.num_layers = validate_positive_int(num_layers, "num_layers")
        if mlp_ratio <= 0.0:
            raise ValueError("`mlp_ratio` must be positive.")
        self.num_heads = resolve_attention_heads(dim, num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.output_norm = nn.LayerNorm(dim)
        self.latent_queries = nn.Parameter(torch.empty(self.output_tokens, dim))
        self.blocks = nn.ModuleList(
            [
                LatentQueryPoolingBlock(
                    dim=dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                )
                for _ in range(self.num_layers)
            ]
        )
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must have shape [B, N, dim], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"Token dim mismatch: expected {self.dim}, got {tokens.shape[-1]}")
        if attention_bias is not None and attention_bias.shape != (
            tokens.shape[0],
            self.output_tokens,
            tokens.shape[1],
        ):
            raise ValueError(
                "`attention_bias` must have shape "
                f"[B, {self.output_tokens}, {tokens.shape[1]}], got {tuple(attention_bias.shape)}"
            )

        pooled_tokens = self.latent_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            pooled_tokens = block(
                slots=pooled_tokens,
                tokens=tokens,
                attention_bias=attention_bias,
            )
        return self.output_norm(pooled_tokens)


def resolve_attention_heads(dim: int, num_heads: int | None) -> int:
    if num_heads is not None:
        heads = validate_positive_int(num_heads, "num_heads")
        if dim % heads != 0:
            raise ValueError(f"`dim`={dim} must be divisible by `num_heads`={heads}.")
        return heads

    for candidate in (8, 4, 2, 1):
        if dim % candidate == 0:
            return candidate
    return 1


def expand_attention_bias(
    attention_bias: torch.Tensor | None,
    num_heads: int,
) -> torch.Tensor | None:
    if attention_bias is None:
        return None
    return attention_bias.repeat_interleave(num_heads, dim=0)


class LatentQueryPoolingBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("`mlp_ratio` produced a non-positive hidden dimension.")

        self.num_heads = validate_positive_int(num_heads, "num_heads")
        self.slot_cross_norm = nn.LayerNorm(dim)
        self.input_cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.slot_self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(
        self,
        slots: torch.Tensor,
        tokens: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized_slots = self.slot_cross_norm(slots)
        normalized_tokens = self.input_cross_norm(tokens)
        cross_attention_output, _ = self.cross_attention(
            query=normalized_slots,
            key=normalized_tokens,
            value=normalized_tokens,
            attn_mask=expand_attention_bias(attention_bias, self.num_heads),
            need_weights=False,
        )
        slots = slots + cross_attention_output

        normalized_slots = self.slot_self_norm(slots)
        self_attention_output, _ = self.self_attention(
            query=normalized_slots,
            key=normalized_slots,
            value=normalized_slots,
            need_weights=False,
        )
        slots = slots + self_attention_output
        return slots + self.ffn(self.ffn_norm(slots))


class TemporalLatentQueryPooling(nn.Module):
    def __init__(self, dim: int, output_tokens: int) -> None:
        super().__init__()
        self.position_encoding = TemporalPositionEncoding(dim)
        self.pool = LatentQueryPooling(dim=dim, output_tokens=output_tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must have shape [B, N, dim], got {tuple(tokens.shape)}")

        position_tokens = self.position_encoding(
            num_tokens=tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        queries = self.pool.latent_queries.to(device=tokens.device, dtype=tokens.dtype)
        position_aware_tokens = tokens + position_tokens.unsqueeze(0)
        attention_bias = torch.matmul(queries, position_tokens.transpose(0, 1)) / math.sqrt(
            self.pool.dim
        )
        attention_bias = attention_bias.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        return self.pool(position_aware_tokens, attention_bias=attention_bias)
