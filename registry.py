from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch.nn as nn

from branches import EyeGazeBranch, FAUBranch, FaceMeshBranch, ModalityBranch, RGBBranch, RPPGBranch
from branches.compression import resolve_output_token_count, validate_branch_token_config


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

MODALITY_TO_ID = {name: index for index, name in enumerate(FULL_MODALITIES)}

CURRENT_MODALITIES: Tuple[str, ...] = ("rgb", "eye_gaze", "face_mesh", "fau", "rppg")
PENDING_MODALITIES: Tuple[str, ...] = tuple(
    modality for modality in FULL_MODALITIES if modality not in CURRENT_MODALITIES
)
SUPPORTED_FRAME_COUNTS: Tuple[int, ...] = (16, 32, 64)


def build_registry(dim: int, config: Mapping[str, Any] | None = None) -> nn.ModuleDict:
    validate_branch_token_config(config, modalities=CURRENT_MODALITIES)
    return nn.ModuleDict(
        {
            "rgb": RGBBranch(dim=dim),
            "eye_gaze": EyeGazeBranch(
                dim=dim,
                output_tokens_per_clip=resolve_output_token_count(config, "eye_gaze"),
            ),
            "face_mesh": FaceMeshBranch(
                dim=dim,
                output_tokens_per_frame=resolve_output_token_count(config, "face_mesh"),
            ),
            "fau": FAUBranch(
                dim=dim,
                output_tokens_per_frame=resolve_output_token_count(config, "fau"),
            ),
            "rppg": RPPGBranch(
                dim=dim,
                output_tokens_per_clip=resolve_output_token_count(config, "rppg"),
            ),
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
