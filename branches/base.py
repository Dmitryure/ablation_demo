from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class ModalityOutput:
    tokens: torch.Tensor
    time_ids: torch.Tensor
    debug: Dict[str, Any]


class ModalityBranch(nn.Module):
    name: str

    def required_keys(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def encode(self, batch: Mapping[str, torch.Tensor]) -> ModalityOutput:
        raise NotImplementedError
