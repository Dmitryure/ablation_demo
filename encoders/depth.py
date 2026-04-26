from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


DEFAULT_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_DEPTH_FEATURE_DIM = 384


def _load_depth_anything_model(model_id_or_path: str) -> nn.Module:
    try:
        from transformers import AutoModelForDepthEstimation
    except ImportError as exc:
        raise ImportError(
            "Depth modality requires `transformers`. Install project requirements before "
            "building DepthAnythingEncoder."
        ) from exc
    return AutoModelForDepthEstimation.from_pretrained(model_id_or_path)


def _last_hidden_state(outputs: Any) -> torch.Tensor:
    hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise ValueError("DepthAnything model did not return hidden_states.")
    final_hidden = hidden_states[-1]
    if not isinstance(final_hidden, torch.Tensor):
        raise ValueError("DepthAnything final hidden state must be a torch.Tensor.")
    return final_hidden


def _pool_hidden_map(hidden: torch.Tensor, feature_dim: int | None) -> torch.Tensor:
    if hidden.ndim == 4:
        if feature_dim is None or hidden.shape[1] == feature_dim:
            return hidden.mean(dim=(2, 3))
        if hidden.shape[-1] == feature_dim:
            return hidden.mean(dim=(1, 2))
        return hidden.mean(dim=(2, 3))
    if hidden.ndim == 3:
        return hidden.mean(dim=1)
    if hidden.ndim == 2:
        return hidden
    raise ValueError(
        "DepthAnything final hidden state must be [B, C, H, W], [B, N, C], or [B, C], "
        f"got {tuple(hidden.shape)}"
    )


class DepthAnythingEncoder(nn.Module):
    def __init__(
        self,
        model_id_or_path: str = DEFAULT_DEPTH_MODEL_ID,
        feature_dim: int = DEFAULT_DEPTH_FEATURE_DIM,
        model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model_id_or_path = model_id_or_path
        self.feature_dim = feature_dim
        self.model = model if model is not None else _load_depth_anything_model(model_id_or_path)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(
                "`pixel_values` must have shape [B*T, 3, H, W], "
                f"got {tuple(pixel_values.shape)}"
            )

        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        features = _pool_hidden_map(_last_hidden_state(outputs), self.feature_dim)
        if features.ndim != 2:
            raise ValueError(
                "DepthAnything pooled features must have shape [B*T, feature_dim], "
                f"got {tuple(features.shape)}"
            )
        if self.feature_dim is not None and features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"DepthAnything feature dim mismatch: expected {self.feature_dim}, "
                f"got {features.shape[-1]}"
            )
        return features
