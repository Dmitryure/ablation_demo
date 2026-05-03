from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from extractors.base import FeatureExtractor


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


def _frame_to_regional_signals(frame_rgb: np.ndarray, grid_size: int) -> np.ndarray:
    frame_float = frame_rgb.astype(np.float32) / 255.0
    height, width, _ = frame_float.shape
    if height < grid_size or width < grid_size:
        raise ValueError(
            f"Frame is too small for {grid_size}x{grid_size} grid: got {height}x{width}."
        )
    trimmed_h = (height // grid_size) * grid_size
    trimmed_w = (width // grid_size) * grid_size
    frame_float = frame_float[:trimmed_h, :trimmed_w, :]

    luminance = (
        0.299 * frame_float[..., 0]
        + 0.587 * frame_float[..., 1]
        + 0.114 * frame_float[..., 2]
    )
    chrominance = frame_float[..., 0] - frame_float[..., 2]  # R - B

    cell_h = trimmed_h // grid_size
    cell_w = trimmed_w // grid_size
    luminance_grid = luminance.reshape(grid_size, cell_h, grid_size, cell_w).mean(axis=(1, 3))
    chrominance_grid = chrominance.reshape(grid_size, cell_h, grid_size, cell_w).mean(axis=(1, 3))

    return np.concatenate([luminance_grid.flatten(), chrominance_grid.flatten()]).astype(np.float32)


class STFTExtractor(FeatureExtractor):

    name = "stft"

    def __init__(
        self,
        n_fft: int = 8,
        hop_length: int | None = None,
        grid_size: int = 4,
        include_chrominance: bool = True,
    ) -> None:
        if n_fft <= 1:
            raise ValueError("`n_fft` must be greater than 1.")
        if grid_size <= 0:
            raise ValueError("`grid_size` must be a positive integer.")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else max(1, self.n_fft // 4)
        if self.hop_length <= 0:
            raise ValueError("`hop_length` must be positive.")
        self.grid_size = int(grid_size)
        self.include_chrominance = bool(include_chrominance)
        self.num_signals = (2 if self.include_chrominance else 1) * self.grid_size * self.grid_size
        self._window = torch.hann_window(self.n_fft)

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def _compute_clip_stft(self, clip: list[np.ndarray]) -> torch.Tensor:
        signals_per_frame: list[np.ndarray] = []
        for frame in clip:
            stats = _frame_to_regional_signals(frame, self.grid_size)
            if not self.include_chrominance:
                stats = stats[: self.grid_size * self.grid_size]
            signals_per_frame.append(stats)
        # signals: [num_signals, T] — torch.stft expects each row to be
        # one independent time series.
        signals = torch.from_numpy(np.stack(signals_per_frame, axis=1))

        if signals.shape[1] < self.n_fft:
            raise ValueError(
                f"STFT needs at least n_fft={self.n_fft} frames, got {signals.shape[1]}. "
                "Lower n_fft or sample more frames."
            )
        spectrum = torch.stft(
            signals,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.log1p(spectrum.abs()) 
        magnitude = magnitude.permute(2, 0, 1).reshape(magnitude.shape[2], -1)
        return magnitude.contiguous()

    def extract(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        clips_rgb = _normalize_clips(batch["video_rgb_frames"])
        per_clip = [self._compute_clip_stft(clip) for clip in clips_rgb]
        return {"stft_features": torch.stack(per_clip, dim=0)}

    def close(self) -> None:
        return None