from __future__ import annotations

from dataclasses import dataclass, field

import torch

HeadDiagnostics = dict[str, torch.Tensor]


@dataclass(frozen=True)
class BinaryHeadResult:
    logits: torch.Tensor
    diagnostics: HeadDiagnostics = field(default_factory=dict)
