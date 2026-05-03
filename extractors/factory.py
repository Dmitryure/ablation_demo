from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from encoders import build_local_encoders
from encoders.depth import DEFAULT_DEPTH_MODEL_ID
from extractors.base import FeatureExtractor
from extractors.depth import DepthExtractor
from extractors.eye_gaze import build_eye_gaze_extractor
from extractors.face_mesh import build_face_mesh_extractor
from extractors.fau import FAUExtractor
from extractors.fft import FFTExtractor
from extractors.rgb import RGBExtractor
from extractors.rppg import RPPGExtractor
from extractors.stft import STFTExtractor


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
    if "fft" in enabled_set:
        fft_config = config.get("fft", {})
        if not isinstance(fft_config, Mapping):
            raise ValueError("`fft` must be a YAML mapping.")
        num_bins = fft_config.get("num_bins", 32)
        if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins <= 0:
            raise ValueError("`fft.num_bins` must be a positive integer.")
        extractors["fft"] = FFTExtractor(
            image_size=int(config.get("image_size", 224)),
            num_bins=num_bins,
        )
    if "stft" in enabled_set:
        stft_config = config.get("stft", {})
        if not isinstance(stft_config, Mapping):
            raise ValueError("`stft` must be a YAML mapping.")
        n_fft = stft_config.get("n_fft", 8)
        if not isinstance(n_fft, int) or isinstance(n_fft, bool) or n_fft <= 1:
            raise ValueError("`stft.n_fft` must be an integer greater than 1.")
        hop_length = stft_config.get("hop_length", None)
        if hop_length is not None and (
            not isinstance(hop_length, int) or isinstance(hop_length, bool) or hop_length <= 0
        ):
            raise ValueError("`stft.hop_length` must be a positive integer if set.")
        extractors["stft"] = STFTExtractor(n_fft=n_fft, hop_length=hop_length)

    return extractors


def build_extractors(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> ExtractorFactoryResult:
    enabled = tuple(modalities or ("rgb", "eye_gaze", "face_mesh", "fau", "rppg", "depth", "fft", "stft"))
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
    enabled = tuple(modalities or ("rgb", "eye_gaze", "face_mesh", "fau", "rppg", "depth", "fft", "stft"))
    return ExtractorFactoryResult(
        extractors=_build_extractors_from_encoder_result(
            config=config,
            enabled=enabled,
            encoder_result=encoder_result,
        ),
        warnings=encoder_result.warnings,
    )