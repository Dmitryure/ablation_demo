from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from encoders.depth import DEFAULT_DEPTH_MODEL_ID
from encoders import build_local_encoders
from extractors.base import FeatureExtractor
from extractors.depth import DepthExtractor
from extractors.eye_gaze import build_eye_gaze_extractor
from extractors.face_mesh import build_face_mesh_extractor
from extractors.fau import FAUExtractor
from extractors.rgb import RGBExtractor
from extractors.rppg import RPPGExtractor


@dataclass(frozen=True)
class ExtractorFactoryResult:
    extractors: dict[str, FeatureExtractor]
    warnings: tuple[str, ...]


def _build_extractors_from_encoder_result(
    config: Mapping[str, Any],
    enabled: Sequence[str],
    encoder_result,
) -> dict[str, FeatureExtractor]:
    enabled_set = set(enabled)
    extractors: dict[str, FeatureExtractor] = {}

    if "rgb" in enabled_set:
        if encoder_result.rgb_encoder is None:
            raise RuntimeError("RGB encoder was not built for the selected modalities.")
        extractors["rgb"] = RGBExtractor(
            encoder_result.rgb_encoder,
            image_size=int(config.get("image_size", 224)),
        )
    if "eye_gaze" in enabled_set:
        extractors["eye_gaze"] = build_eye_gaze_extractor(config)
    if "face_mesh" in enabled_set:
        extractors["face_mesh"] = build_face_mesh_extractor(config)
    if "fau" in enabled_set:
        if encoder_result.fau_encoder is None:
            raise RuntimeError("FAU encoder was not built for the selected modalities.")
        extractors["fau"] = FAUExtractor(encoder_result.fau_encoder)
    if "rppg" in enabled_set:
        if encoder_result.rppg_encoder is None:
            raise RuntimeError("rPPG encoder was not built for the selected modalities.")
        extractors["rppg"] = RPPGExtractor(encoder_result.rppg_encoder)
    if "depth" in enabled_set:
        if encoder_result.depth_encoder is None:
            raise RuntimeError("Depth encoder was not built for the selected modalities.")
        depth_config = config.get("depth", {})
        if not isinstance(depth_config, Mapping):
            raise ValueError("`depth` must be a YAML mapping.")
        model_id_or_path = depth_config.get("model_id_or_path", DEFAULT_DEPTH_MODEL_ID)
        if not isinstance(model_id_or_path, str) or not model_id_or_path.strip():
            raise ValueError("`depth.model_id_or_path` must be a non-empty string.")
        extractors["depth"] = DepthExtractor(
            encoder_result.depth_encoder,
            model_id_or_path=model_id_or_path.strip(),
        )

    return extractors


def build_extractors(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> ExtractorFactoryResult:
    enabled = tuple(modalities or ("rgb", "eye_gaze", "face_mesh", "fau", "rppg", "depth"))
    encoder_result = build_local_encoders(config, modalities=enabled)
    return ExtractorFactoryResult(
        extractors=_build_extractors_from_encoder_result(
            config=config,
            enabled=enabled,
            encoder_result=encoder_result,
        ),
        warnings=encoder_result.warnings,
    )


def build_extractors_from_encoders(
    config: Mapping[str, Any],
    encoder_result,
    modalities: Sequence[str] | None = None,
) -> ExtractorFactoryResult:
    enabled = tuple(modalities or ("rgb", "eye_gaze", "face_mesh", "fau", "rppg", "depth"))
    return ExtractorFactoryResult(
        extractors=_build_extractors_from_encoder_result(
            config=config,
            enabled=enabled,
            encoder_result=encoder_result,
        ),
        warnings=encoder_result.warnings,
    )
