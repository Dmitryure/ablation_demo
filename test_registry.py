import unittest

import numpy as np
import torch
import torch.nn as nn

from branches import ModalityBranch, ModalityOutput
from encoders import FAUEncoder, RPPGEncoder
from extractors import EYE_GAZE_COLUMNS, EyeGazeExtractor, FAUExtractor, RPPGExtractor
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
        registry = build_registry(dim=32)

        self.assertEqual(tuple(registry.keys()), CURRENT_MODALITIES)
        validate_registry(registry)

    def test_registry_entries_inherit_common_branch_interface(self):
        registry = build_registry(dim=32)

        for name in CURRENT_MODALITIES:
            self.assertIsInstance(registry[name], ModalityBranch)

    def test_registry_required_keys_for_video_modalities(self):
        registry = build_registry(dim=32)

        keys = registry_required_keys(registry, ("fau", "rppg"))
        self.assertEqual(keys["fau"], ("fau_features",))
        self.assertEqual(keys["rppg"], ("rppg_features",))

    def test_local_wrappers_instantiate(self):
        fau_encoder = FAUEncoder(backbone="swin_transformer_tiny", num_classes=4)
        rppg_encoder = RPPGEncoder(frames=4)

        self.assertIsInstance(fau_encoder, nn.Module)
        self.assertIsInstance(rppg_encoder, nn.Module)

    def test_fau_extractor_output_contract(self):
        extractor = FAUExtractor(DummyFAUEncoder(num_au=4, feature_dim=20))
        video = torch.randn(1, 3, 2, 224, 224)

        output = extractor.extract({"video": video})

        self.assertEqual(tuple(output["fau_features"].shape), (1, 2, 4, 20))
        self.assertEqual(tuple(output["fau_au_logits"].shape), (1, 2, 4))

    def test_local_fau_branch_output_contract(self):
        registry = build_registry(dim=16)
        fau_features = torch.randn(1, 2, 4, 20)

        output = registry["fau"].encode({"fau_features": fau_features})

        self.assertEqual(tuple(output.tokens.shape), (1, 8, 16))
        self.assertEqual(tuple(output.time_ids.shape), (8,))

    def test_rppg_extractor_output_contract(self):
        extractor = RPPGExtractor(DummyRPPGEncoder(feature_dim=12))
        video = torch.randn(1, 3, 4, 32, 32)

        output = extractor.extract({"video": video})

        self.assertEqual(tuple(output["rppg_features"].shape), (1, 4, 12))
        self.assertEqual(tuple(output["rppg_waveform"].shape), (1, 4))

    def test_local_rppg_branch_output_contract(self):
        registry = build_registry(dim=16)
        temporal_features = torch.randn(1, 4, 12)

        output = registry["rppg"].encode({"rppg_features": temporal_features})

        self.assertEqual(tuple(output.tokens.shape), (1, 4, 16))
        self.assertEqual(tuple(output.time_ids.shape), (4,))

    def test_eye_gaze_extractor_tensor_shape_with_fake_detector(self):
        def fake_detect_features(_: np.ndarray):
            return {name: index / 10.0 for index, name in enumerate(EYE_GAZE_COLUMNS, start=1)}

        extractor = EyeGazeExtractor(detect_features_fn=fake_detect_features)
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

        tensor = extractor.extract_tensor(frames)

        self.assertEqual(tuple(tensor.shape), (3, 8))
        self.assertAlmostEqual(float(tensor[0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(tensor[0, -1]), 0.8, places=6)

    def test_eye_gaze_branch_output_contract(self):
        registry = build_registry(dim=16)
        eye_gaze = torch.randn(1, 4, 8)

        output = registry["eye_gaze"].encode({"eye_gaze": eye_gaze})

        self.assertEqual(tuple(output.tokens.shape), (1, 4, 16))
        self.assertEqual(tuple(output.time_ids.shape), (4,))


if __name__ == "__main__":
    unittest.main()
