from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FRAME_SAMPLING_KEY = "frame_sampling"
FRAME_SAMPLING_LINSPACE = "linspace"
FRAME_SAMPLING_CENTER_STRIDE = "center_stride"
SHORT_VIDEO_ERROR = "error"
SHORT_VIDEO_REPEAT_PAD = "repeat_pad"


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{field_name}` must be a positive integer.")
    return value


def _sampling_section(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is None:
        return {}
    value = config.get(FRAME_SAMPLING_KEY, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("`frame_sampling` must be a YAML mapping when provided.")
    return value


def resolve_frame_sampling_config(
    global_config: Mapping[str, Any] | None,
    modality_config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    merged = {**_sampling_section(global_config), **_sampling_section(modality_config)}
    if not merged:
        return None

    name = str(merged.get("name", merged.get("type", FRAME_SAMPLING_LINSPACE)))
    if name == FRAME_SAMPLING_LINSPACE:
        return None
    if name != FRAME_SAMPLING_CENTER_STRIDE:
        raise ValueError(f"Unsupported frame_sampling.name: {name!r}")

    output_frames = _positive_int(merged.get("output_frames"), "frame_sampling.output_frames")
    source_window_frames = _positive_int(
        merged.get("source_window_frames"),
        "frame_sampling.source_window_frames",
    )
    stride = _positive_int(merged.get("stride"), "frame_sampling.stride")
    short_video = str(merged.get("short_video", SHORT_VIDEO_REPEAT_PAD))
    if short_video not in (SHORT_VIDEO_ERROR, SHORT_VIDEO_REPEAT_PAD):
        raise ValueError(
            "`frame_sampling.short_video` must be "
            f"{SHORT_VIDEO_ERROR!r} or {SHORT_VIDEO_REPEAT_PAD!r}."
        )
    minimum_window = (output_frames - 1) * stride + 1
    if source_window_frames < minimum_window:
        raise ValueError(
            "`frame_sampling.source_window_frames` is too small for "
            f"output_frames={output_frames} and stride={stride}; need at least {minimum_window}."
        )
    return {
        "name": name,
        "output_frames": output_frames,
        "source_window_frames": source_window_frames,
        "stride": stride,
        "short_video": short_video,
    }


def frame_sampling_cache_variant(
    global_config: Mapping[str, Any] | None,
    modality_config: Mapping[str, Any] | None = None,
) -> str | None:
    config = resolve_frame_sampling_config(global_config, modality_config)
    if config is None:
        return None
    suffix = "repeatpad" if config["short_video"] == SHORT_VIDEO_REPEAT_PAD else "error"
    return (
        f"center{config['source_window_frames']}"
        f"_stride{config['stride']}"
        f"_frames{config['output_frames']}"
        f"_{suffix}"
    )


def center_stride_frame_indices(
    total_frames: int,
    num_frames: int,
    frame_sampling: Mapping[str, Any],
) -> list[int]:
    if total_frames <= 0:
        raise RuntimeError("Video has no frames.")
    output_frames = int(frame_sampling["output_frames"])
    if num_frames != output_frames:
        raise ValueError(
            f"Configured frame count {num_frames} does not match "
            f"frame_sampling.output_frames={output_frames}."
        )

    source_window_frames = int(frame_sampling["source_window_frames"])
    stride = int(frame_sampling["stride"])
    span = min(total_frames, source_window_frames)
    start = max(0, (total_frames - span) // 2)
    indices = [start + offset for offset in range(0, span, stride)]
    indices = indices[:num_frames]
    if len(indices) >= num_frames:
        return indices

    short_video = str(frame_sampling["short_video"])
    if short_video == SHORT_VIDEO_ERROR:
        raise RuntimeError(
            f"Video yields only {len(indices)} sampled frames, need {num_frames}."
        )
    if not indices:
        raise RuntimeError("Video has no sampled frames.")
    return [*indices, *([indices[-1]] * (num_frames - len(indices)))]


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    frame_sampling: Mapping[str, Any] | None = None,
) -> list[int]:
    _positive_int(num_frames, "num_frames")
    if frame_sampling is None:
        if total_frames < num_frames:
            raise RuntimeError(f"Video has only {total_frames} frames, need at least {num_frames}")
        if num_frames == 1:
            return [0]
        step = (total_frames - 1) / (num_frames - 1)
        return [round(index * step) for index in range(num_frames)]
    if str(frame_sampling.get("name")) == FRAME_SAMPLING_CENTER_STRIDE:
        return center_stride_frame_indices(total_frames, num_frames, frame_sampling)
    raise ValueError(f"Unsupported frame_sampling.name: {frame_sampling.get('name')!r}")
