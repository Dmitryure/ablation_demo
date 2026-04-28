from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CheckpointLoadResult:
    path: Path
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
            return payload["state_dict"]
        return payload
    raise TypeError("Checkpoint must be a state dict or contain a `state_dict` mapping.")


def _strip_prefixes(
    state_dict: Mapping[str, torch.Tensor],
    prefixes: Iterable[str],
) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        cleaned[normalized] = value
    return cleaned


def load_checkpoint(
    module: nn.Module,
    checkpoint_path: str | Path,
    prefixes: Iterable[str] = (),
) -> CheckpointLoadResult:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(payload)
    cleaned_state_dict = _strip_prefixes(state_dict, ("module.", *prefixes))
    missing_keys, unexpected_keys = module.load_state_dict(cleaned_state_dict, strict=False)
    return CheckpointLoadResult(
        path=path,
        missing_keys=tuple(missing_keys),
        unexpected_keys=tuple(unexpected_keys),
    )
