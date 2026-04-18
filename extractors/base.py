from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn


class FeatureExtractor:
    name: str

    def required_keys(self) -> Tuple[str, ...]:
        raise NotImplementedError

    def extract(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None


def module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")
