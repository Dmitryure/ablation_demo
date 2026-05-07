from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from extractors.base import FeatureExtractor, module_device

RPPG_SIGNAL_FEATURE_DIM = 6
RPPG_DEFAULT_FPS = 30.0
RPPG_HR_BAND = (0.7, 3.0)


def z_normalize_waveform(waveform: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"`waveform` must have shape [B, T], got {tuple(waveform.shape)}")
    centered = waveform - waveform.mean(dim=1, keepdim=True)
    std = centered.std(dim=1, keepdim=True, correction=0)
    return torch.where(std > eps, centered / (std + eps), torch.zeros_like(centered))


def _fps_for_batch(fps: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if fps is None:
        return torch.full((batch_size,), RPPG_DEFAULT_FPS, dtype=torch.float32, device=device)
    if isinstance(fps, torch.Tensor):
        values = fps.detach().to(device=device, dtype=torch.float32).flatten()
    else:
        values = torch.as_tensor(fps, dtype=torch.float32, device=device).flatten()
    if values.numel() == 1:
        values = values.expand(batch_size)
    if values.numel() != batch_size:
        raise ValueError(f"`video_fps` must have {batch_size} values, got {values.numel()}")
    return torch.where(values > 0.0, values, torch.full_like(values, RPPG_DEFAULT_FPS))


def _spectral_features_for_waveform(waveform: torch.Tensor, fps: float) -> torch.Tensor:
    num_frames = int(waveform.numel())
    if num_frames < 2 or fps <= 0.0:
        return torch.zeros(RPPG_SIGNAL_FEATURE_DIM, dtype=waveform.dtype, device=waveform.device)

    spectrum = torch.fft.rfft(waveform)
    power = spectrum.abs().pow(2)
    freqs = torch.fft.rfftfreq(num_frames, d=1.0 / fps).to(waveform.device)
    non_dc = freqs > 0.0
    total_power = power[non_dc].sum()
    hr_mask = (freqs >= RPPG_HR_BAND[0]) & (freqs <= RPPG_HR_BAND[1])
    hr_power = power[hr_mask]
    hr_freqs = freqs[hr_mask]
    raw_std = waveform.std(correction=0)
    if hr_power.numel() == 0 or float(hr_power.sum()) <= 1e-12 or float(total_power) <= 1e-12:
        return torch.stack(
            [
                torch.tensor(0.0, dtype=waveform.dtype, device=waveform.device),
                torch.tensor(0.0, dtype=waveform.dtype, device=waveform.device),
                torch.tensor(0.0, dtype=waveform.dtype, device=waveform.device),
                torch.tensor(0.0, dtype=waveform.dtype, device=waveform.device),
                raw_std,
                _autocorrelation_peak(waveform, fps),
            ]
        )

    peak_index = int(torch.argmax(hr_power).item())
    dominant_freq = hr_freqs[peak_index]
    hr_power_sum = hr_power.sum()
    probabilities = hr_power / (hr_power_sum + 1e-12)
    entropy = -(probabilities * torch.log(probabilities + 1e-12)).sum()
    if hr_power.numel() > 1:
        entropy = entropy / torch.log(
            torch.tensor(float(hr_power.numel()), dtype=waveform.dtype, device=waveform.device)
        )
    return torch.stack(
        [
            dominant_freq.to(dtype=waveform.dtype),
            (hr_power_sum / (total_power + 1e-12)).to(dtype=waveform.dtype),
            (hr_power[peak_index] / (hr_power_sum + 1e-12)).to(dtype=waveform.dtype),
            entropy.to(dtype=waveform.dtype),
            raw_std,
            _autocorrelation_peak(waveform, fps),
        ]
    )


def _autocorrelation_peak(waveform: torch.Tensor, fps: float) -> torch.Tensor:
    centered = waveform - waveform.mean()
    denom = centered.pow(2).sum()
    if int(waveform.numel()) < 3 or float(denom) <= 1e-12 or fps <= 0.0:
        return torch.tensor(0.0, dtype=waveform.dtype, device=waveform.device)
    min_lag = max(1, int(round(fps / RPPG_HR_BAND[1])))
    max_lag = min(int(waveform.numel()) - 1, int(round(fps / RPPG_HR_BAND[0])))
    if min_lag > max_lag:
        return torch.tensor(0.0, dtype=waveform.dtype, device=waveform.device)
    values = [
        (centered[:-lag] * centered[lag:]).sum() / (denom + 1e-12)
        for lag in range(min_lag, max_lag + 1)
    ]
    return torch.stack(values).max()


def compute_rppg_signal_features(waveform: torch.Tensor, fps: Any = None) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"`waveform` must have shape [B, T], got {tuple(waveform.shape)}")
    fps_values = _fps_for_batch(fps, waveform.shape[0], waveform.device)
    return torch.stack(
        [
            _spectral_features_for_waveform(waveform[index], float(fps_values[index].item()))
            for index in range(waveform.shape[0])
        ],
        dim=0,
    )


class RPPGExtractor(FeatureExtractor):
    name = "rppg"

    def __init__(self, encoder: nn.Module):
        self.encoder = encoder

    def required_keys(self) -> tuple[str, ...]:
        return ("video",)

    def extract(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        video = batch["video"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(f"`video` must have shape [B, 3, N, H, W], got {tuple(video.shape)}")

        encoded = self.encoder(video.to(module_device(self.encoder)))
        if not isinstance(encoded, tuple) or len(encoded) < 2:
            raise ValueError("RPPG encoder must return `(waveform, temporal_features)`")

        waveform, temporal_features = encoded[:2]
        if waveform.ndim != 2:
            raise ValueError(f"RPPG waveform must have shape [B, N], got {tuple(waveform.shape)}")
        if temporal_features.ndim != 3:
            raise ValueError(
                "RPPG temporal features must have shape [B, N, feature_dim], "
                f"got {tuple(temporal_features.shape)}"
            )

        waveform = waveform.to(dtype=torch.float32)
        signal_features = compute_rppg_signal_features(waveform.detach(), batch.get("video_fps"))
        normalized_waveform = z_normalize_waveform(waveform)
        return {
            "rppg_waveform": normalized_waveform,
            "rppg_features": temporal_features,
            "rppg_signal_features": signal_features,
        }
