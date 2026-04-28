from __future__ import annotations

from typing import Any, Mapping, Sequence


FRAME_CONFIG_KEY = "frames"


def validate_frame_count(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{field_name}` must be a positive integer.")
    return value


def _modality_section(config: Mapping[str, Any], modality_name: str) -> Mapping[str, Any]:
    section = config.get(modality_name, {})
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError(f"`{modality_name}` must be a YAML mapping when provided.")
    return section


def _top_level_default(config: Mapping[str, Any]) -> int | None:
    value = config.get(FRAME_CONFIG_KEY)
    if value is None:
        return None
    if isinstance(value, Mapping):
        default = value.get("default")
        if default is None:
            return None
        return validate_frame_count(default, "frames.default")
    return validate_frame_count(value, "frames")


def resolve_modality_frame_count(config: Mapping[str, Any], modality_name: str) -> int:
    section = _modality_section(config, modality_name)
    section_value = section.get(FRAME_CONFIG_KEY)
    if section_value is not None:
        return validate_frame_count(section_value, f"{modality_name}.frames")

    value = config.get(FRAME_CONFIG_KEY)
    if isinstance(value, Mapping):
        modality_value = value.get(modality_name)
        if modality_value is not None:
            return validate_frame_count(modality_value, f"frames.{modality_name}")

    default = _top_level_default(config)
    if default is not None:
        return default

    raise ValueError(
        f"Missing frame count for modality `{modality_name}`. "
        "Set `<modality>.frames`, `frames.<modality>`, or `frames.default`."
    )


def resolve_modality_frame_counts(
    config: Mapping[str, Any],
    modalities: Sequence[str],
) -> dict[str, int]:
    return {
        modality_name: resolve_modality_frame_count(config, modality_name)
        for modality_name in modalities
    }


def describe_frame_counts(frame_counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in frame_counts.items())
