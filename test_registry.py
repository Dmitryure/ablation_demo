import unittest

import numpy as np
import torch
import torch.nn as nn

from branches import ModalityBranch, ModalityOutput
from branches.compression import DEFAULT_OUTPUT_TOKENS, validate_branch_token_config
from encoders import FAUEncoder, RGBEncoder, RPPGEncoder, build_local_encoders
from extractors import (
    EYE_GAZE_COLUMNS,
    FACE_MESH_CONTOUR_INDICES,
    EyeGazeExtractor,
    FAUExtractor,
    FaceMeshExtractor,
    RGBExtractor,
    RPPGExtractor,
)
from fusion import FusionOutput, TokenBankFusion
from registry import CURRENT_MODALITIES, MODALITY_TO_ID, build_registry, registry_required_keys, validate_registry
from run_registry_fusion import (
    build_fusion_module,
    fuse_selected_modalities,
    require_fusion_config,
    validate_selected_modalities,
)


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


class DummyRGBClipEncoder(nn.Module):
    def __init__(self, token_count: int = 8, feature_dim: int = 96):
        super().__init__()
        self.token_count = token_count
        self.feature_dim = feature_dim
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        batch = x.shape[0]
        return torch.randn(batch, self.token_count, self.feature_dim, device=x.device)


class DummyMetadataBranch(ModalityBranch):
    name = "metadata"

    def required_keys(self):
        return ("metadata_tokens",)

    def encode(self, batch):
        tokens = batch["metadata_tokens"]
        time_ids = torch.zeros(tokens.shape[1], dtype=torch.long, device=tokens.device)
        return ModalityOutput(tokens=tokens, time_ids=time_ids, debug={"token_shape": tuple(tokens.shape)})


def fake_eye_gaze_detector(_: np.ndarray):
    return {name: index / 10.0 for index, name in enumerate(EYE_GAZE_COLUMNS, start=1)}


def fake_face_mesh_detector(_: np.ndarray):
    points = np.zeros((len(FACE_MESH_CONTOUR_INDICES), 3), dtype=np.float32)
    for index in range(points.shape[0]):
        points[index] = (index / 100.0, index / 200.0, -index / 300.0)
    return points


class RegistryTest(unittest.TestCase):
    def build_test_fusion(self, dim: int = 16, max_time_steps: int = 8) -> TokenBankFusion:
        num_heads = 4 if dim % 4 == 0 else 2 if dim % 2 == 0 else 1
        return TokenBankFusion(
            dim=dim,
            num_layers=2,
            num_heads=num_heads,
            mlp_ratio=2.0,
            dropout=0.0,
            max_time_steps=max_time_steps,
            num_modalities=len(MODALITY_TO_ID),
        )

    def test_registry_contains_current_modalities(self):
        registry = build_registry(dim=32)

        self.assertEqual(tuple(registry.keys()), CURRENT_MODALITIES)
        validate_registry(registry)

    def test_registry_entries_inherit_common_branch_interface(self):
        registry = build_registry(dim=32)

        for name in CURRENT_MODALITIES:
            self.assertIsInstance(registry[name], ModalityBranch)

    def test_build_registry_without_config_uses_default_token_budgets(self):
        registry = build_registry(dim=32)

        self.assertEqual(
            registry["eye_gaze"].output_tokens_per_clip,
            DEFAULT_OUTPUT_TOKENS["eye_gaze"],
        )
        self.assertEqual(
            registry["rppg"].output_tokens_per_clip,
            DEFAULT_OUTPUT_TOKENS["rppg"],
        )
        self.assertEqual(
            registry["face_mesh"].output_tokens_per_frame,
            DEFAULT_OUTPUT_TOKENS["face_mesh"],
        )
        self.assertEqual(
            registry["fau"].output_tokens_per_frame,
            DEFAULT_OUTPUT_TOKENS["fau"],
        )

    def test_registry_required_keys_for_video_modalities(self):
        registry = build_registry(dim=32)

        keys = registry_required_keys(registry, ("rgb", "face_mesh", "fau", "rppg"))
        self.assertEqual(keys["rgb"], ("rgb_features",))
        self.assertEqual(keys["face_mesh"], ("face_mesh",))
        self.assertEqual(keys["fau"], ("fau_features",))
        self.assertEqual(keys["rppg"], ("rppg_features",))

    def test_local_wrappers_instantiate(self):
        rgb_encoder = RGBEncoder(frames=16, image_size=224)
        fau_encoder = FAUEncoder(backbone="swin_transformer_tiny", num_classes=4)
        rppg_encoder = RPPGEncoder(frames=4)

        self.assertIsInstance(rgb_encoder, nn.Module)
        self.assertIsInstance(fau_encoder, nn.Module)
        self.assertIsInstance(rppg_encoder, nn.Module)

    def test_rgb_encoder_output_contract(self):
        encoder = RGBEncoder(frames=16, image_size=224)
        video = torch.randn(1, 3, 16, 224, 224)

        with torch.no_grad():
            output = encoder(video)

        self.assertEqual(tuple(output.shape), (1, 8, 768))

    def test_rgb_extractor_output_contract(self):
        encoder = DummyRGBClipEncoder(token_count=8, feature_dim=96)
        extractor = RGBExtractor(encoder, image_size=224)
        frames = [np.full((12, 10, 3), 128, dtype=np.uint8) for _ in range(16)]

        output = extractor.extract({"video_rgb_frames": frames})

        self.assertEqual(tuple(output["rgb_features"].shape), (1, 8, 96))
        self.assertIsNotNone(encoder.last_input)
        self.assertEqual(tuple(encoder.last_input.shape), (1, 3, 16, 224, 224))
        self.assertAlmostEqual(
            float(encoder.last_input[0, 0, 0, 0, 0]),
            (128 / 255.0 - 0.45) / 0.225,
            places=5,
        )

    def test_rgb_branch_output_contract(self):
        registry = build_registry(dim=16)
        rgb_features = torch.randn(1, 8, 768)

        output = registry["rgb"].encode({"rgb_features": rgb_features})

        self.assertEqual(tuple(output.tokens.shape), (1, 8, 16))
        self.assertEqual(tuple(output.time_ids.shape), (8,))

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

        self.assertEqual(tuple(output.tokens.shape), (1, 4, 16))
        self.assertEqual(tuple(output.time_ids.shape), (4,))
        self.assertTrue(torch.equal(output.time_ids, torch.tensor([0, 0, 1, 1])))

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
        self.assertTrue(torch.equal(output.time_ids, torch.tensor([0, 1, 2, 3])))

    def test_rppg_temporal_pooling_changes_when_frame_order_changes(self):
        registry = build_registry(dim=16)
        temporal_features = torch.arange(1, 1 + 6 * 12, dtype=torch.float32).reshape(1, 6, 12)
        reversed_features = torch.flip(temporal_features, dims=(1,))

        output = registry["rppg"].encode({"rppg_features": temporal_features})
        reversed_output = registry["rppg"].encode({"rppg_features": reversed_features})

        self.assertFalse(torch.allclose(output.tokens, reversed_output.tokens))

    def test_eye_gaze_extractor_tensor_shape_with_fake_detector(self):
        extractor = EyeGazeExtractor(detect_features_fn=fake_eye_gaze_detector)
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
        self.assertTrue(torch.equal(output.time_ids, torch.tensor([0, 1, 2, 3])))

    def test_eye_gaze_temporal_pooling_changes_when_frame_order_changes(self):
        registry = build_registry(dim=16)
        eye_gaze = torch.arange(1, 1 + 6 * 8, dtype=torch.float32).reshape(1, 6, 8)
        reversed_eye_gaze = torch.flip(eye_gaze, dims=(1,))

        output = registry["eye_gaze"].encode({"eye_gaze": eye_gaze})
        reversed_output = registry["eye_gaze"].encode({"eye_gaze": reversed_eye_gaze})

        self.assertFalse(torch.allclose(output.tokens, reversed_output.tokens))

    def test_face_mesh_extractor_tensor_shape_with_fake_detector(self):
        extractor = FaceMeshExtractor(detect_landmarks_fn=fake_face_mesh_detector)
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

        tensor = extractor.extract_tensor(frames)

        self.assertEqual(tuple(tensor.shape), (3, len(FACE_MESH_CONTOUR_INDICES), 3))
        self.assertAlmostEqual(float(tensor[0, 1, 0]), 0.01, places=6)
        self.assertAlmostEqual(float(tensor[0, 1, 2]), -1.0 / 300.0, places=6)

    def test_face_mesh_branch_output_contract(self):
        registry = build_registry(dim=16)
        face_mesh = torch.randn(1, 4, len(FACE_MESH_CONTOUR_INDICES), 3)

        output = registry["face_mesh"].encode({"face_mesh": face_mesh})

        self.assertEqual(tuple(output.tokens.shape), (1, 4, 16))
        self.assertEqual(tuple(output.time_ids.shape), (4,))
        self.assertTrue(torch.equal(output.time_ids, torch.tensor([0, 1, 2, 3])))

    def test_build_registry_config_overrides_change_token_counts(self):
        registry = build_registry(
            dim=16,
            config={
                "eye_gaze": {"output_tokens_per_clip": 3},
                "rppg": {"output_tokens_per_clip": 2},
                "face_mesh": {"output_tokens_per_frame": 2},
                "fau": {"output_tokens_per_frame": 3},
            },
        )

        eye_gaze_output = registry["eye_gaze"].encode({"eye_gaze": torch.randn(1, 16, 8)})
        rppg_output = registry["rppg"].encode({"rppg_features": torch.randn(1, 16, 12)})
        face_mesh_output = registry["face_mesh"].encode(
            {"face_mesh": torch.randn(1, 4, len(FACE_MESH_CONTOUR_INDICES), 3)}
        )
        fau_output = registry["fau"].encode({"fau_features": torch.randn(1, 4, 5, 20)})

        self.assertEqual(tuple(eye_gaze_output.tokens.shape), (1, 3, 16))
        self.assertEqual(tuple(rppg_output.tokens.shape), (1, 2, 16))
        self.assertEqual(tuple(face_mesh_output.tokens.shape), (1, 8, 16))
        self.assertEqual(tuple(fau_output.tokens.shape), (1, 12, 16))
        self.assertTrue(torch.equal(face_mesh_output.time_ids, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])))
        self.assertTrue(torch.equal(fau_output.time_ids, torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])))

    def test_build_registry_rejects_non_positive_token_budgets(self):
        with self.assertRaisesRegex(ValueError, "rppg.output_tokens_per_clip"):
            build_registry(dim=16, config={"rppg": {"output_tokens_per_clip": 0}})

    def test_validate_branch_token_config_rejects_fusion_time_step_mismatch(self):
        config = {
            "frames": 16,
            "fusion": {"max_time_steps": 3},
            "eye_gaze": {"output_tokens_per_clip": 4},
        }

        with self.assertRaisesRegex(ValueError, "fusion.max_time_steps"):
            validate_branch_token_config(
                config,
                modalities=("eye_gaze",),
                fusion_max_time_steps=config["fusion"]["max_time_steps"],
            )

    def test_fuse_selected_modalities_returns_token_bank_metadata(self):
        registry = build_registry(dim=16)
        fusion_module = self.build_test_fusion(dim=16, max_time_steps=16)
        batch = {
            "rgb_features": torch.randn(1, 8, 768),
            "rppg_features": torch.randn(1, 16, 12),
        }

        fusion_output, debug = fuse_selected_modalities(
            registry,
            batch,
            ("rgb", "rppg"),
            {"rgb": 1.0, "rppg": 2.0},
            fusion_module,
        )

        self.assertIsInstance(fusion_output, FusionOutput)
        self.assertEqual(tuple(fusion_output.tokens.shape), (1, 12, 16))
        self.assertEqual(tuple(fusion_output.fused.shape), (1, 16))
        self.assertEqual(tuple(fusion_output.cls_token.shape), (1, 16))
        self.assertEqual(tuple(fusion_output.fused_tokens.shape), (1, 13, 16))
        self.assertEqual(tuple(fusion_output.time_ids.shape), (12,))
        self.assertEqual(tuple(fusion_output.modality_ids.shape), (12,))
        self.assertEqual(fusion_output.modality_names, ("rgb", "rppg"))
        self.assertTrue(
            torch.equal(
                fusion_output.time_ids,
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]),
            )
        )
        self.assertTrue(torch.equal(fusion_output.modality_ids, torch.tensor([0] * 8 + [4] * 4)))
        self.assertEqual(debug["token_bank_shape"], (1, 12, 16))
        self.assertEqual(debug["modality_token_counts"], {"rgb": 8, "rppg": 4})
        self.assertEqual(debug["cls_token_shape"], (1, 16))
        self.assertEqual(debug["fused_tokens_shape"], (1, 13, 16))

    def test_fuse_selected_modalities_uses_stable_subset_modality_ids(self):
        registry = build_registry(dim=8)
        fusion_module = self.build_test_fusion(dim=8, max_time_steps=4)
        batch = {
            "eye_gaze": torch.randn(1, 2, 8),
            "rppg_features": torch.randn(1, 2, 12),
        }

        fusion_output, _ = fuse_selected_modalities(
            registry,
            batch,
            ("eye_gaze", "rppg"),
            {"eye_gaze": 1.0, "rppg": 1.0},
            fusion_module,
        )

        expected = torch.tensor(
            [
                MODALITY_TO_ID["eye_gaze"],
                MODALITY_TO_ID["eye_gaze"],
                MODALITY_TO_ID["eye_gaze"],
                MODALITY_TO_ID["eye_gaze"],
                MODALITY_TO_ID["rppg"],
                MODALITY_TO_ID["rppg"],
                MODALITY_TO_ID["rppg"],
                MODALITY_TO_ID["rppg"],
            ]
        )
        self.assertTrue(torch.equal(fusion_output.modality_ids, expected))

    def test_fuse_selected_modalities_supports_mixed_token_layouts(self):
        registry = build_registry(dim=12)
        fusion_module = self.build_test_fusion(dim=12, max_time_steps=8)
        batch = {
            "rgb_features": torch.randn(1, 8, 768),
            "fau_features": torch.randn(1, 3, 2, 20),
        }

        fusion_output, debug = fuse_selected_modalities(
            registry,
            batch,
            ("rgb", "fau"),
            {"rgb": 1.0, "fau": 1.0},
            fusion_module,
        )

        self.assertEqual(tuple(fusion_output.tokens.shape), (1, 14, 12))
        self.assertEqual(tuple(fusion_output.fused_tokens.shape), (1, 15, 12))
        self.assertEqual(debug["modality_token_counts"], {"rgb": 8, "fau": 6})
        self.assertTrue(
            torch.equal(
                fusion_output.time_ids,
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 1, 1, 2, 2]),
            )
        )

    def test_face_mesh_tokens_repeat_time_id_per_contour_point(self):
        registry = build_registry(dim=12)
        fusion_module = self.build_test_fusion(dim=12, max_time_steps=3)
        batch = {
            "face_mesh": torch.randn(1, 3, len(FACE_MESH_CONTOUR_INDICES), 3),
        }

        fusion_output, debug = fuse_selected_modalities(
            registry,
            batch,
            ("face_mesh",),
            {"face_mesh": 1.0},
            fusion_module,
        )

        self.assertEqual(
            tuple(fusion_output.tokens.shape),
            (1, 3, 12),
        )
        self.assertTrue(
            torch.equal(
                fusion_output.time_ids,
                torch.arange(3),
            )
        )
        self.assertEqual(debug["modality_token_counts"], {"face_mesh": 3})

    def test_build_local_encoders_requires_rgb_checkpoint(self):
        config = {
            "frames": 16,
            "image_size": 224,
            "rgb": {"checkpoint_path": None},
            "fau": {"backbone": "swin_transformer_tiny", "num_classes": 12, "checkpoint_path": None},
            "rppg": {"checkpoint_path": None},
        }

        with self.assertRaisesRegex(ValueError, "RGB checkpoint_path is required"):
            build_local_encoders(config, modalities=("rgb",))

    def test_future_non_temporal_modality_can_use_time_zero_tokens(self):
        registry = build_registry(dim=10)
        registry["metadata"] = DummyMetadataBranch()
        fusion_module = self.build_test_fusion(dim=10, max_time_steps=4)
        batch = {"metadata_tokens": torch.randn(1, 3, 10)}

        fusion_output, debug = fuse_selected_modalities(
            registry,
            batch,
            ("metadata",),
            {"metadata": 1.0},
            fusion_module,
        )

        self.assertEqual(tuple(fusion_output.tokens.shape), (1, 3, 10))
        self.assertTrue(torch.equal(fusion_output.time_ids, torch.zeros(3, dtype=torch.long)))
        self.assertTrue(
            torch.equal(
                fusion_output.modality_ids,
                torch.full((3,), MODALITY_TO_ID["metadata"], dtype=torch.long),
            )
        )
        self.assertEqual(debug["fused_tokens_shape"], (1, 4, 10))

    def test_missing_fusion_checkpoint_keeps_random_init_smoke_path_valid(self):
        config = {
            "fusion": {
                "type": "token_transformer",
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "max_time_steps": 8,
                "checkpoint_path": None,
            }
        }

        fusion_config = require_fusion_config(config)
        fusion_module = build_fusion_module(dim=16, fusion_config=fusion_config)

        self.assertIsInstance(fusion_module, TokenBankFusion)

    def test_require_fusion_config_rejects_malformed_values(self):
        with self.assertRaisesRegex(ValueError, "`fusion.type` must be `token_transformer`."):
            require_fusion_config({"fusion": {"type": "mean_pool"}})

    def test_validate_selected_modalities_rejects_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "unsupported modalities"):
            validate_selected_modalities(("rgb", "metadata"))


if __name__ == "__main__":
    unittest.main()
