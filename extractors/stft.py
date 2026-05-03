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


def _temporal_signal_from_clip(clip: list[np.ndarray]) -> torch.Tensor:
    intensities = [float(frame.astype(np.float32).mean()) / 255.0 for frame in clip]
    return torch.tensor(intensities, dtype=torch.float32)


class STFTExtractor(FeatureExtractor):
    name = "stft"

    def __init__(self, n_fft: int = 8, hop_length: int | None = None) -> None:
        if n_fft <= 1:
            raise ValueError("`n_fft` must be greater than 1.")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else max(1, self.n_fft // 4)
        if self.hop_length <= 0:
            raise ValueError("`hop_length` must be positive.")
        # Hann window — standard for STFT, reduces spectral leakage at window edges.
        self._window = torch.hann_window(self.n_fft)

    def required_keys(self) -> tuple[str, ...]:
        return ("video_rgb_frames",)

    def _compute_clip_stft(self, clip: list[np.ndarray]) -> torch.Tensor:
        signal = _temporal_signal_from_clip(clip)
        if signal.numel() < self.n_fft:
            raise ValueError(
                f"STFT needs at least n_fft={self.n_fft} frames, "
                f"got {signal.numel()}. Lower n_fft or sample more frames."
            )
        spectrum = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self._window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.log1p(spectrum.abs())
        return magnitude.transpose(0, 1) 

    def extract(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        clips_rgb = _normalize_clips(batch["video_rgb_frames"])
        per_clip = [self._compute_clip_stft(clip) for clip in clips_rgb]
        return {"stft_features": torch.stack(per_clip, dim=0)}

    def close(self) -> None:
        return None