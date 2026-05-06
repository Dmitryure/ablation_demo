from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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


def _require_encoder(encoder: Any, message: str) -> Any:
    if encoder is None:
        raise RuntimeError(message)
    return encoder


def _config_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"`{key}` must be a YAML mapping.")
    return value


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        if minimum == 2:
            raise ValueError(f"`{name}` must be an integer greater than 1.")
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"`{name}` must be a boolean.")
    return value


def _build_rgb_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    return RGBExtractor(
        _require_encoder(
            encoder_result.rgb_encoder,
            "RGB encoder was not built for the selected modalities.",
        ),
        image_size=int(config.get("image_size", 224)),
    )


def _build_eye_gaze_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del encoder_result
    return build_eye_gaze_extractor(config)


def _build_face_mesh_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del encoder_result
    return build_face_mesh_extractor(config)


def _build_fau_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del config
    return FAUExtractor(
        _require_encoder(
            encoder_result.fau_encoder,
            "FAU encoder was not built for the selected modalities.",
        )
    )


def _build_rppg_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del config
    return RPPGExtractor(
        _require_encoder(
            encoder_result.rppg_encoder,
            "rPPG encoder was not built for the selected modalities.",
        )
    )


def _build_depth_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    depth_config = _config_mapping(config, "depth")
    model_id_or_path = depth_config.get("model_id_or_path", DEFAULT_DEPTH_MODEL_ID)
    if not isinstance(model_id_or_path, str) or not model_id_or_path.strip():
        raise ValueError("`depth.model_id_or_path` must be a non-empty string.")
    return DepthExtractor(
        _require_encoder(
            encoder_result.depth_encoder,
            "Depth encoder was not built for the selected modalities.",
        ),
        model_id_or_path=model_id_or_path.strip(),
    )


def _build_fft_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del encoder_result
    fft_config = _config_mapping(config, "fft")
    return FFTExtractor(
        image_size=int(config.get("image_size", 224)),
        num_bins=_positive_int(fft_config.get("num_bins", 32), "fft.num_bins"),
    )


def _build_stft_extractor(config: Mapping[str, Any], encoder_result: Any) -> FeatureExtractor:
    del encoder_result
    stft_config = _config_mapping(config, "stft")
    return STFTExtractor(
        n_fft=_positive_int(stft_config.get("n_fft", 8), "stft.n_fft", minimum=2),
        hop_length=_optional_positive_int(stft_config.get("hop_length"), "stft.hop_length"),
        grid_size=_positive_int(stft_config.get("grid_size", 4), "stft.grid_size"),
        include_chrominance=_bool_value(
            stft_config.get("include_chrominance", True),
            "stft.include_chrominance",
        ),
    )


_EXTRACTOR_BUILDERS: dict[str, Callable[[Mapping[str, Any], Any], FeatureExtractor]] = {
    "rgb": _build_rgb_extractor,
    "eye_gaze": _build_eye_gaze_extractor,
    "face_mesh": _build_face_mesh_extractor,
    "fau": _build_fau_extractor,
    "rppg": _build_rppg_extractor,
    "depth": _build_depth_extractor,
    "fft": _build_fft_extractor,
    "stft": _build_stft_extractor,
}


def _build_extractors_from_encoder_result(
    config: Mapping[str, Any],
    enabled: Sequence[str],
    encoder_result,
) -> dict[str, FeatureExtractor]:
    extractors: dict[str, FeatureExtractor] = {}
    for modality in enabled:
        builder = _EXTRACTOR_BUILDERS.get(modality)
        if builder is not None:
            extractors[modality] = builder(config, encoder_result)
    return extractors


def build_extractors(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None = None,
) -> ExtractorFactoryResult:
    enabled = tuple(
        modalities or ("rgb", "eye_gaze", "face_mesh", "fau", "rppg", "depth", "fft", "stft")
    )
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
    enabled = tuple(
        modalities or ("rgb", "eye_gaze", "face_mesh", "fau", "rppg", "depth", "fft", "stft")
    )
    return ExtractorFactoryResult(
        extractors=_build_extractors_from_encoder_result(
            config=config,
            enabled=enabled,
            encoder_result=encoder_result,
        ),
        warnings=encoder_result.warnings,
    )
