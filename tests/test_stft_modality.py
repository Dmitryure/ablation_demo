from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branches.stft import STFTBranch
from extractors.stft import STFTExtractor
from fusion import TokenBankFusion, prepare_token_bank
from registry import FIXED_SLOT_MODALITIES, MODALITY_TO_ID


def make_fake_clip_batch(batch_size: int = 2, num_frames: int = 16, image_size: int = 32):
    np_rng = np.random.default_rng(0)
    clips = []
    for clip_idx in range(batch_size):
        is_fake = clip_idx >= batch_size // 2
        frames = []
        for frame_idx in range(num_frames):
            if is_fake:
                base = 64 if frame_idx % 2 == 0 else 192
            else:
                base = 80 + (120 * frame_idx // num_frames)
            jitter = np_rng.integers(-8, 9, (image_size, image_size, 3), dtype=np.int16)
            frame = np.clip(base + jitter, 0, 255).astype(np.uint8)
            frames.append(frame)
        clips.append(frames)
    return clips


def check_extractor():
    print("== 1. extractor ==")
    extractor = STFTExtractor(n_fft=8, hop_length=1, grid_size=4, include_chrominance=True)
    clips = make_fake_clip_batch(batch_size=2, num_frames=16, image_size=32)
    out = extractor.extract({"video_rgb_frames": clips})
    stft_features = out["stft_features"]
    assert stft_features.ndim == 3, f"got {stft_features.shape}"
    assert stft_features.shape[0] == 2, f"expected batch=2, got {stft_features.shape}"
    expected_feature_dim = 32 * 5  # num_signals * num_freq_bins
    assert stft_features.shape[2] == expected_feature_dim, (
        f"expected feature_dim={expected_feature_dim}, got {stft_features.shape[2]}"
    )
    assert torch.isfinite(stft_features).all(), "non-finite values in stft_features"
    real_spectrum = stft_features[0].mean(dim=0)
    fake_spectrum = stft_features[1].mean(dim=0)
    spectral_distance = (real_spectrum - fake_spectrum).abs().mean().item()
    assert spectral_distance > 0.01, f"real and fake spectra too similar: {spectral_distance}"
    print(f"  stft_features.shape = {tuple(stft_features.shape)}")
    print(f"  real-vs-fake mean abs spectral distance = {spectral_distance:.4f}")
    print("  OK\n")
    return stft_features


def check_branch(stft_features: torch.Tensor) -> torch.Tensor:
    print("== 2. branch ==")
    branch = STFTBranch(dim=16, slot_count=4)
    output = branch.encode({"stft_features": stft_features})
    assert output.tokens.shape == (2, 4, 16), f"got {output.tokens.shape}"
    assert output.time_ids.shape == (4,), f"got {output.time_ids.shape}"
    print(f"  tokens.shape = {tuple(output.tokens.shape)}")
    print(f"  time_ids = {output.time_ids.tolist()}")
    print("  OK\n")
    return output


def check_fusion_integration(stft_features: torch.Tensor) -> None:
    print("== 3. fusion integration ==")
    dim = 16
    branches = nn.ModuleDict()
    slot_counts = {
        "rgb": 8,
        "fau": 32,
        "rppg": 16,
        "eye_gaze": 4,
        "face_mesh": 16,
        "depth": 4,
        "fft": 4,
        "stft": 4,
    }
    for name, slot_count in slot_counts.items():
        if name == "stft":
            branches[name] = STFTBranch(dim=dim, slot_count=slot_count)
        else:

            class StubBranch(nn.Module):
                def __init__(self, slot_count: int):
                    super().__init__()
                    self.slot_count = slot_count

            branches[name] = StubBranch(slot_count)

    stft_output = branches["stft"].encode({"stft_features": stft_features})
    token_bank = prepare_token_bank(
        outputs_by_name={"stft": stft_output},
        enabled_modalities=("stft",),
        modality_to_id=MODALITY_TO_ID,
        fixed_slot_modalities=FIXED_SLOT_MODALITIES,
        slot_counts=slot_counts,
    )
    expected_total = sum(slot_counts[name] for name in FIXED_SLOT_MODALITIES)
    assert token_bank.tokens.shape == (2, expected_total, dim), token_bank.tokens.shape
    stft_position = list(FIXED_SLOT_MODALITIES).index("stft")
    stft_offset = sum(slot_counts[name] for name in FIXED_SLOT_MODALITIES[:stft_position])
    assert token_bank.token_mask[stft_offset : stft_offset + 4].all()
    other_mask = torch.cat(
        [
            token_bank.token_mask[:stft_offset],
            token_bank.token_mask[stft_offset + 4 :],
        ]
    )
    assert (~other_mask).all()
    print(f"  total tokens = {token_bank.tokens.shape[1]} (expected {expected_total})")
    print(
        f"  stft slot range = [{stft_offset}, {stft_offset + 4}); only those have token_mask=True"
    )

    fusion = TokenBankFusion(
        dim=dim,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        max_time_steps=64,
        num_modalities=len(MODALITY_TO_ID),
    )
    cls, _fused = fusion(
        tokens=token_bank.tokens,
        token_mask=token_bank.token_mask,
        time_ids=token_bank.time_ids,
        modality_ids=token_bank.modality_ids,
    )
    assert cls.shape == (2, dim)
    assert torch.isfinite(cls).all()

    head = nn.Linear(dim, 1)
    optimizer = torch.optim.AdamW(
        list(branches["stft"].parameters()) + list(fusion.parameters()) + list(head.parameters()),
        lr=3e-3,
    )
    labels = torch.tensor([[0.0], [1.0]])
    initial_loss = None
    for _step in range(50):
        optimizer.zero_grad()
        stft_out = branches["stft"].encode({"stft_features": stft_features})
        bank = prepare_token_bank(
            outputs_by_name={"stft": stft_out},
            enabled_modalities=("stft",),
            modality_to_id=MODALITY_TO_ID,
            fixed_slot_modalities=FIXED_SLOT_MODALITIES,
            slot_counts=slot_counts,
        )
        cls_step, _ = fusion(
            tokens=bank.tokens,
            token_mask=bank.token_mask,
            time_ids=bank.time_ids,
            modality_ids=bank.modality_ids,
        )
        logits = head(cls_step)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
    print(f"  loss: {initial_loss:.4f} -> {loss.item():.4f} (50 steps)")
    assert loss.item() < initial_loss * 0.5, "loss did not decrease"
    print("  OK\n")


def main() -> int:
    stft_features = check_extractor()
    check_branch(stft_features)
    check_fusion_integration(stft_features)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
