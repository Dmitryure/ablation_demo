from __future__ import annotations

from typing import Dict, Tuple

import torch.nn as nn

from branches import EyeGazeBranch, FAUBranch, ModalityBranch, RPPGBranch


FULL_MODALITIES: Tuple[str, ...] = (
    "rgb",
    "metadata",
    "depth",
    "face_mesh",
    "rppg",
    "eye_gaze",
    "fau",
    "text",
    "manipulation_mask",
)

CURRENT_MODALITIES: Tuple[str, ...] = ("eye_gaze", "fau", "rppg")
PENDING_MODALITIES: Tuple[str, ...] = tuple(
    modality for modality in FULL_MODALITIES if modality not in CURRENT_MODALITIES
)
SUPPORTED_FRAME_COUNTS: Tuple[int, ...] = (16, 32, 64)


def build_registry(
    dim: int,
    fau_encoder: nn.Module,
    rppg_encoder: nn.Module,
) -> nn.ModuleDict:
    return nn.ModuleDict(
        {
            "eye_gaze": EyeGazeBranch(dim=dim),
            "fau": FAUBranch(encoder=fau_encoder, dim=dim),
            "rppg": RPPGBranch(encoder=rppg_encoder, dim=dim),
        }
    )


def validate_registry(registry: nn.ModuleDict) -> None:
    missing = [name for name in CURRENT_MODALITIES if name not in registry]
    if missing:
        raise ValueError(f"Registry is missing required modalities: {missing}")

    for name in CURRENT_MODALITIES:
        branch = registry[name]
        if not isinstance(branch, ModalityBranch):
            raise TypeError(f"Registry entry `{name}` must inherit from ModalityBranch")


def registry_required_keys(registry: nn.ModuleDict, modalities: Tuple[str, ...]) -> Dict[str, Tuple[str, ...]]:
    return {name: registry[name].required_keys() for name in modalities}
