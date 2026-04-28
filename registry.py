from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from branches import (
    DepthBranch,
    EyeGazeBranch,
    FaceMeshBranch,
    FAUBranch,
    ModalityBranch,
    RGBBranch,
    RPPGBranch,
)
from branches.compression import resolve_slot_count, validate_branch_token_config

FULL_MODALITIES: tuple[str, ...] = (
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

MODALITY_TO_ID = {name: index for index, name in enumerate(FULL_MODALITIES)}

FIXED_SLOT_MODALITIES: tuple[str, ...] = ("rgb", "fau", "rppg", "eye_gaze", "face_mesh", "depth")
CURRENT_MODALITIES: tuple[str, ...] = FIXED_SLOT_MODALITIES
PENDING_MODALITIES: tuple[str, ...] = tuple(
    modality for modality in FULL_MODALITIES if modality not in CURRENT_MODALITIES
)
SUPPORTED_FRAME_COUNTS: tuple[int, ...] = (16, 32, 64)


def build_registry(dim: int, config: Mapping[str, Any] | None = None) -> nn.ModuleDict:
    validate_branch_token_config(config, modalities=CURRENT_MODALITIES)
    return nn.ModuleDict(
        {
            "rgb": RGBBranch(dim=dim, slot_count=resolve_slot_count(config, "rgb")),
            "fau": FAUBranch(dim=dim, slot_count=resolve_slot_count(config, "fau")),
            "rppg": RPPGBranch(dim=dim, slot_count=resolve_slot_count(config, "rppg")),
            "eye_gaze": EyeGazeBranch(dim=dim, slot_count=resolve_slot_count(config, "eye_gaze")),
            "face_mesh": FaceMeshBranch(
                dim=dim, slot_count=resolve_slot_count(config, "face_mesh")
            ),
            "depth": DepthBranch(dim=dim, slot_count=resolve_slot_count(config, "depth")),
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


def registry_required_keys(
    registry: nn.ModuleDict, modalities: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    return {name: registry[name].required_keys() for name in modalities}


def registry_slot_counts(
    registry: nn.ModuleDict,
    modalities: tuple[str, ...] = FIXED_SLOT_MODALITIES,
) -> dict[str, int]:
    return {name: int(registry[name].slot_count) for name in modalities}
