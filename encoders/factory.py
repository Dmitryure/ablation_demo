from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch.nn as nn

from encoders.fau import FAUEncoder
from encoders.rppg import RPPGEncoder


@dataclass(frozen=True)
class EncoderFactoryResult:
    fau_encoder: nn.Module | None
    rppg_encoder: nn.Module | None
    warnings: tuple[str, ...]


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"`{key}` must be a YAML mapping.")
    return value


def _require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer.")
    return value


def _require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{key}` must be a non-empty string.")
    return value.strip()


def _optional_path(config: Mapping[str, Any], key: str) -> Path | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return Path(stripped)
    raise ValueError(f"`{key}` must be a string path or null.")


def build_local_encoders(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> EncoderFactoryResult:
    enabled = set(modalities or ("fau", "rppg"))
    frames = _require_int(config, "frames")

    fau_config = _require_mapping(config, "fau")
    rppg_config = _require_mapping(config, "rppg")

    fau_checkpoint_path = _optional_path(fau_config, "checkpoint_path")
    rppg_checkpoint_path = _optional_path(rppg_config, "checkpoint_path")

    warnings: list[str] = []
    if "fau" in enabled and fau_checkpoint_path is None:
        warnings.append("FAU checkpoint_path omitted; building encoder without pretrained weights.")
    if "rppg" in enabled and rppg_checkpoint_path is None:
        warnings.append("rPPG checkpoint_path omitted; building encoder without pretrained weights.")

    fau_encoder = None
    rppg_encoder = None
    if "fau" in enabled:
        fau_encoder = FAUEncoder(
            backbone=_require_str(fau_config, "backbone"),
            num_classes=_require_int(fau_config, "num_classes"),
            checkpoint_path=fau_checkpoint_path,
        )
    if "rppg" in enabled:
        rppg_encoder = RPPGEncoder(
            frames=frames,
            checkpoint_path=rppg_checkpoint_path,
        )
    return EncoderFactoryResult(
        fau_encoder=fau_encoder,
        rppg_encoder=rppg_encoder,
        warnings=tuple(warnings),
    )
