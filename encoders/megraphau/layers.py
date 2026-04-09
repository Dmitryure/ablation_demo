from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def to_2tuple(value: int | Iterable[int]) -> tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        items = tuple(value)
        if len(items) != 2:
            raise ValueError(f"Expected a 2-item iterable, got {items}")
        return int(items[0]), int(items[1])
    return int(value), int(value)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
