from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch

from extractors.base import FeatureExtractor


def _validate_frame(frame_rgb: np.ndarray) -> None:
    if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(
            "`video_rgb_frames` must contain RGB numpy arrays shaped [H, W, 3], "
            f"got {type(frame_rgb)}"
        )


def _is_clip_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and isinstance(value[0], np.ndarray)


def _is_clip_batch(value: object) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and _is_clip_sequence(value[0])


def _normalize_clips(frames_rgb: object) -> list:
    if _is_clip_batch(frames_rgb):
        return list(frames_rgb)
    if _is_clip_sequence(frames_rgb):
        return [frames_rgb]
    raise ValueError(
        "`video_rgb_frames` must be either a clip `[N][H,W,3]` or a batch `[B][N][H,W,3]`."
    )


def _build_radial_bin_indices(image_size: int, num_bins: int) -> torch.Tensor:
    centre = image_size // 2
    coordinates = torch.arange(image_size, dtype=torch.float32) - centre
    ys = coordinates.unsqueeze(1)
    xs = coordinates.unsqueeze(0)
    radius = torch.sqrt(ys * ys + xs * xs)
    max_radius = float(radius.max().item())
    edges = torch.linspace(0.0, max_radius + 1e-6, num_bins + 1)
    indices = torch.bucketize(radius, edges, right=False) - 1
    return indices.clamp(0, num_bins - 1).to(torch.int64)


class FFTExtractor(FeatureExtractor):
    name = "fft"

    def __init__(self, image_size: int = 128, num_bins: int = 32) -> None:
        if image_size <= 0:
            raise ValueError("`image_size` must be positive.")
        if num_bins <= 0:
            raise ValueError("`num_bins` must be positive.")
        self.image_size = int(image_size)
        self.num_bins = int(num_bins)
        self._bin_indices = _build_radial_bin_indices(self.image_size, self.num_bins)
        self._bin_counts = (
            torch.bincount(self._bin_indices.flatten(), minlength=self.num_bins)
            .clamp(min=1)
            .to(torch.float32)
        )

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def _compute_frame_spectrum(self, frame_rgb: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if gray.shape != (self.image_size, self.image_size):
            gray = cv2.resize(
                gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
        tensor = torch.from_numpy(np.ascontiguousarray(gray)).to(torch.float32) / 255.0
        spectrum = torch.fft.fftshift(torch.fft.fft2(tensor))
        magnitude = torch.log1p(spectrum.abs())
        bins = torch.zeros(self.num_bins, dtype=torch.float32)
        bins.scatter_add_(0, self._bin_indices.flatten(), magnitude.flatten())
        return bins / self._bin_counts

    def extract(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        clips_rgb = _normalize_clips(batch["video_rgb_frames"])
        per_clip: list[torch.Tensor] = []
        for clip in clips_rgb:
            per_frame: list[torch.Tensor] = []
            for frame in clip:
                _validate_frame(frame)
                per_frame.append(self._compute_frame_spectrum(frame))
            per_clip.append(torch.stack(per_frame, dim=0))
        return {"fft_features": torch.stack(per_clip, dim=0)}

    def close(self) -> None:
        return None
