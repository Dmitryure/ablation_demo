from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FACE_CROP_VARIANT_VERSION = 2
FACE_CROP_STATUS_DETECTED = "detected"
FACE_CROP_STATUS_FALLBACK = "fallback_full_frame"
FACE_CROP_STATUS_DISABLED = "disabled"
FACE_CROP_DEFAULT_BACKEND = "opencv_haar"
FACE_CROP_DEFAULT_DETECTION_FREQUENCY = 16
FACE_CROP_DEFAULT_LARGE_BOX_COEF = 1.5
FACE_CROP_DEFAULT_FALLBACK = "full_frame"
RPPG_CACHE_VARIANT = "rppg_v2_contiguous128_facecrop_haar_diffnorm128"
EYE_GAZE_RICH_FEATURE_VARIANT = "rich_v1"
EYE_GAZE_RICH_CACHE_VARIANT = "eye_gaze_rich_v1"


def _variant_part(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip().lower()).strip("_")


def _variant_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def config_section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"`{key}` must be a YAML mapping when provided.")
    return value


def default_face_crop_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    value = config.get("face_crop", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("`face_crop` must be a YAML mapping when provided.")
    return value


def merged_face_crop_config(
    global_config: Mapping[str, Any] | None,
    modality_config: Mapping[str, Any] | None,
    default_enabled: bool = False,
) -> dict[str, Any]:
    global_face_crop = default_face_crop_config(global_config)
    modality_face_crop = {}
    if modality_config is not None:
        value = modality_config.get("face_crop", {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError("`face_crop` must be a YAML mapping when provided.")
        modality_face_crop = value
    merged = {**global_face_crop, **modality_face_crop}
    return {
        "enabled": bool(merged.get("enabled", default_enabled)),
        "backend": str(merged.get("backend", FACE_CROP_DEFAULT_BACKEND)),
        "detection_frequency": int(
            merged.get("detection_frequency", FACE_CROP_DEFAULT_DETECTION_FREQUENCY)
        ),
        "large_box_coef": float(merged.get("large_box_coef", FACE_CROP_DEFAULT_LARGE_BOX_COEF)),
        "fallback": str(merged.get("fallback", FACE_CROP_DEFAULT_FALLBACK)),
    }


def face_crop_cache_variant(
    global_config: Mapping[str, Any] | None,
    modality_config: Mapping[str, Any] | None,
    default_enabled: bool = False,
) -> str | None:
    crop = merged_face_crop_config(
        global_config=global_config,
        modality_config=modality_config,
        default_enabled=default_enabled,
    )
    if not crop["enabled"]:
        return None
    return (
        f"facecrop_v{FACE_CROP_VARIANT_VERSION}_{_variant_part(crop['backend'])}"
        f"_df{int(crop['detection_frequency'])}"
        f"_box{_variant_float(float(crop['large_box_coef']))}"
        f"_fallback_{_variant_part(crop['fallback'])}"
    )


def modality_cache_variant(config: Mapping[str, Any], modality: str) -> str | None:
    modality_config = config_section(config, modality)
    if modality == "rppg":
        return RPPG_CACHE_VARIANT
    if modality == "eye_gaze":
        feature_variant = modality_config.get("feature_variant")
        if feature_variant is not None and str(feature_variant) != EYE_GAZE_RICH_FEATURE_VARIANT:
            raise ValueError(
                f"Unsupported eye_gaze.feature_variant: {feature_variant!r}. "
                f"Expected {EYE_GAZE_RICH_FEATURE_VARIANT!r} or omit it for legacy features."
            )
        crop_variant = face_crop_cache_variant(
            global_config=config,
            modality_config=modality_config,
            default_enabled=False,
        )
        if feature_variant is None:
            return crop_variant
        if crop_variant is None:
            return EYE_GAZE_RICH_CACHE_VARIANT
        return f"{EYE_GAZE_RICH_CACHE_VARIANT}_{crop_variant}"
    return face_crop_cache_variant(
        global_config=config,
        modality_config=modality_config,
        default_enabled=False,
    )
