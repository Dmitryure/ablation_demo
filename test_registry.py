import unittest

import torch
import torch.nn as nn

from branches import ModalityBranch, ModalityOutput
from encoders import FAUEncoder, RPPGEncoder
from registry import CURRENT_MODALITIES, build_registry, registry_required_keys, validate_registry


class DummyFAUEncoder(nn.Module):
    def __init__(self, num_au: int = 12, feature_dim: int = 20):
        super().__init__()
        self.num_au = num_au
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor):
        batch_frames = x.shape[0]
        features = torch.randn(batch_frames, self.num_au, self.feature_dim, device=x.device)
        logits = torch.randn(batch_frames, self.num_au, device=x.device)
        edge_logits = torch.randn(batch_frames, self.num_au, 3, device=x.device)
        return features, logits, edge_logits


class DummyRPPGEncoder(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor):
        batch, _, num_frames, _, _ = x.shape
        waveform = torch.randn(batch, num_frames, device=x.device)
        features = torch.randn(batch, num_frames, self.feature_dim, device=x.device)
        return waveform, features


class RegistryTest(unittest.TestCase):
    def test_registry_contains_current_modalities(self):
        registry = build_registry(
            dim=32,
            fau_encoder=DummyFAUEncoder(),
            rppg_encoder=DummyRPPGEncoder(),
        )

        self.assertEqual(tuple(registry.keys()), CURRENT_MODALITIES)
        validate_registry(registry)

    def test_registry_entries_inherit_common_branch_interface(self):
        registry = build_registry(
            dim=32,
            fau_encoder=DummyFAUEncoder(),
            rppg_encoder=DummyRPPGEncoder(),
        )

        for name in CURRENT_MODALITIES:
            self.assertIsInstance(registry[name], ModalityBranch)

    def test_registry_required_keys_for_video_modalities(self):
        registry = build_registry(
            dim=32,
            fau_encoder=DummyFAUEncoder(),
            rppg_encoder=DummyRPPGEncoder(),
        )

        keys = registry_required_keys(registry, ("fau", "rppg"))
        self.assertEqual(keys["fau"], ("video",))
        self.assertEqual(keys["rppg"], ("video",))

    def test_local_wrappers_instantiate(self):
        fau_encoder = FAUEncoder(backbone="swin_transformer_tiny", num_classes=4)
        rppg_encoder = RPPGEncoder(frames=4)

        self.assertIsInstance(fau_encoder, nn.Module)
        self.assertIsInstance(rppg_encoder, nn.Module)

    def test_local_fau_branch_output_contract(self):
        registry = build_registry(
            dim=16,
            fau_encoder=FAUEncoder(backbone="swin_transformer_tiny", num_classes=4),
            rppg_encoder=DummyRPPGEncoder(),
        )
        video = torch.randn(1, 3, 1, 224, 224)

        output = registry["fau"].encode({"video": video})

        self.assertEqual(tuple(output.tokens.shape), (1, 4, 16))
        self.assertEqual(tuple(output.time_ids.shape), (4,))

    def test_local_rppg_branch_output_contract(self):
        registry = build_registry(
            dim=16,
            fau_encoder=DummyFAUEncoder(),
            rppg_encoder=RPPGEncoder(frames=4),
        )
        video = torch.randn(1, 3, 4, 32, 32)

        output = registry["rppg"].encode({"video": video})

        self.assertEqual(tuple(output.tokens.shape), (1, 4, 16))
        self.assertEqual(tuple(output.time_ids.shape), (4,))


if __name__ == "__main__":
    unittest.main()
