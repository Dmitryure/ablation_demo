import unittest

import numpy as np
import torch
import torch.nn as nn

from branches import ModalityBranch, ModalityOutput
from branches.compression import DEFAULT_SLOT_COUNTS, validate_branch_token_config
from encoders import FAUEncoder, RGBEncoder, RPPGEncoder, build_local_encoders
from extractors import (
    EYE_GAZE_COLUMNS,
    FACE_MESH_CONTOUR_INDICES,
    DepthExtractor,
    EyeGazeExtractor,
    FaceMeshExtractor,
    FAUExtractor,
    RGBExtractor,
    RPPGExtractor,
)
from frame_config import resolve_modality_frame_count, resolve_modality_frame_counts
from fusion import FusionOutput, TokenBankFusion
from pipeline import build_fusion_from_config, fuse_selected_modalities
from registry import (
    CURRENT_MODALITIES,
    FIXED_SLOT_MODALITIES,
    MODALITY_TO_ID,
    build_registry,
    registry_required_keys,
    validate_registry,
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


class DummyDepthEncoder(nn.Module):
    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.feature_dim = feature_dim
        self.last_input: torch.Tensor | None = None

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.last_input = pixel_values.detach().clone()
        batch_frames = pixel_values.shape[0]
        return torch.randn(batch_frames, self.feature_dim, device=pixel_values.device)


class DummyDepthProcessor:
    def __call__(self, images, return_tensors: str):
        if return_tensors != "pt":
            raise ValueError("DummyDepthProcessor only supports return_tensors='pt'.")
        pixel_values = torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
                for image in images
            ],
            dim=0,
        )
        return {"pixel_values": pixel_values}


class DummyMetadataBranch(ModalityBranch):
    name = "metadata"

    def required_keys(self):
        return ("metadata_tokens",)

    def encode(self, batch):
        tokens = batch["metadata_tokens"]
        time_ids = torch.zeros(tokens.shape[1], dtype=torch.long, device=tokens.device)
        return ModalityOutput(
            tokens=tokens, time_ids=time_ids, debug={"token_shape": tuple(tokens.shape)}
        )


def fake_eye_gaze_detector(_: np.ndarray):
    return {name: index / 10.0 for index, name in enumerate(EYE_GAZE_COLUMNS, start=1)}


def fake_face_mesh_detector(_: np.ndarray):
    points = np.zeros((len(FACE_MESH_CONTOUR_INDICES), 3), dtype=np.float32)
    for index in range(points.shape[0]):
        points[index] = (index / 100.0, index / 200.0, -index / 300.0)
    return points


class RegistryTest(unittest.TestCase):
    def build_test_fusion(self, dim: int = 16, max_time_steps: int = 32) -> TokenBankFusion:
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

        self.assertEqual(registry["rgb"].slot_count, DEFAULT_SLOT_COUNTS["rgb"])
        self.assertEqual(registry["eye_gaze"].slot_count, DEFAULT_SLOT_COUNTS["eye_gaze"])
        self.assertEqual(registry["rppg"].slot_count, DEFAULT_SLOT_COUNTS["rppg"])
        self.assertEqual(registry["face_mesh"].slot_count, DEFAULT_SLOT_COUNTS["face_mesh"])
        self.assertEqual(registry["fau"].slot_count, DEFAULT_SLOT_COUNTS["fau"])
        self.assertEqual(registry["depth"].slot_count, DEFAULT_SLOT_COUNTS["depth"])

    def test_registry_required_keys_for_video_modalities(self):
        registry = build_registry(dim=32)

        keys = registry_required_keys(registry, ("rgb", "face_mesh", "fau", "rppg", "depth"))
        self.assertEqual(keys["rgb"], ("rgb_features",))
        self.assertEqual(keys["face_mesh"], ("face_mesh",))
        self.assertEqual(keys["fau"], ("fau_features",))
        self.assertEqual(keys["rppg"], ("rppg_features",))
        self.assertEqual(keys["depth"], ("depth_features",))

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

    def test_depth_extractor_output_contract(self):
        encoder = DummyDepthEncoder(feature_dim=384)
        extractor = DepthExtractor(
            encoder,
            processor=DummyDepthProcessor(),
            model_id_or_path="unused",
        )
        frames = [np.full((12, 10, 3), 128, dtype=np.uint8) for _ in range(16)]

        output = extractor.extract({"video_rgb_frames": frames})

        self.assertEqual(tuple(output["depth_features"].shape), (1, 16, 384))
        self.assertIsNotNone(encoder.last_input)
        self.assertEqual(tuple(encoder.last_input.shape), (16, 3, 12, 10))

    def test_depth_extractor_output_contract_for_batched_clips(self):
        encoder = DummyDepthEncoder(feature_dim=384)
        extractor = DepthExtractor(
            encoder,
            processor=DummyDepthProcessor(),
            model_id_or_path="unused",
        )
        clips = [
            [np.full((6, 5, 3), 64 + frame_index, dtype=np.uint8) for frame_index in range(4)],
            [np.full((6, 5, 3), 96 + frame_index, dtype=np.uint8) for frame_index in range(4)],
        ]

        output = extractor.extract({"video_rgb_frames": clips})

        self.assertEqual(tuple(output["depth_features"].shape), (2, 4, 384))
        self.assertIsNotNone(encoder.last_input)
        self.assertEqual(tuple(encoder.last_input.shape), (8, 3, 6, 5))

    def test_depth_branch_output_contract(self):
        registry = build_registry(dim=16)
        depth_features = torch.randn(1, 16, 384)

        output = registry["depth"].encode({"depth_features": depth_features})

        self.assertEqual(tuple(output.tokens.shape), (1, 4, 16))
        self.assertEqual(tuple(output.time_ids.shape), (4,))
        self.assertTrue(torch.equal(output.time_ids, torch.arange(4)))

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

        self.assertEqual(tuple(output.tokens.shape), (1, 32, 16))
        self.assertEqual(tuple(output.time_ids.shape), (32,))
        self.assertTrue(torch.equal(output.time_ids, torch.arange(32)))

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

        self.assertEqual(tuple(output.tokens.shape), (1, 16, 16))
        self.assertEqual(tuple(output.time_ids.shape), (16,))
        self.assertTrue(torch.equal(output.time_ids, torch.arange(16)))

    def test_build_registry_config_overrides_change_token_counts(self):
        registry = build_registry(
            dim=16,
            config={
                "rgb": {"slot_count": 6},
                "eye_gaze": {"slot_count": 3},
                "rppg": {"slot_count": 2},
                "face_mesh": {"slot_count": 7},
                "fau": {"slot_count": 5},
                "depth": {"slot_count": 4},
            },
        )

        rgb_output = registry["rgb"].encode({"rgb_features": torch.randn(1, 8, 768)})
        eye_gaze_output = registry["eye_gaze"].encode({"eye_gaze": torch.randn(1, 16, 8)})
        rppg_output = registry["rppg"].encode({"rppg_features": torch.randn(1, 16, 12)})
        face_mesh_output = registry["face_mesh"].encode(
            {"face_mesh": torch.randn(1, 4, len(FACE_MESH_CONTOUR_INDICES), 3)}
        )
        fau_output = registry["fau"].encode({"fau_features": torch.randn(1, 4, 5, 20)})
        depth_output = registry["depth"].encode({"depth_features": torch.randn(1, 4, 384)})

        self.assertEqual(tuple(rgb_output.tokens.shape), (1, 6, 16))
        self.assertEqual(tuple(eye_gaze_output.tokens.shape), (1, 3, 16))
        self.assertEqual(tuple(rppg_output.tokens.shape), (1, 2, 16))
        self.assertEqual(tuple(face_mesh_output.tokens.shape), (1, 7, 16))
        self.assertEqual(tuple(fau_output.tokens.shape), (1, 5, 16))
        self.assertEqual(tuple(depth_output.tokens.shape), (1, 4, 16))
        self.assertTrue(torch.equal(face_mesh_output.time_ids, torch.arange(7)))
        self.assertTrue(torch.equal(fau_output.time_ids, torch.arange(5)))
        self.assertTrue(torch.equal(depth_output.time_ids, torch.arange(4)))

    def test_build_registry_rejects_non_positive_token_budgets(self):
        with self.assertRaisesRegex(ValueError, "rppg.slot_count"):
            build_registry(dim=16, config={"rppg": {"slot_count": 0}})

    def test_resolve_modality_frame_counts_supports_section_and_default_values(self):
        config = {
            "frames": {"default": 16, "rppg": 32},
            "rgb": {"frames": 8},
            "rppg": {},
        }

        self.assertEqual(resolve_modality_frame_count(config, "rgb"), 8)
        self.assertEqual(resolve_modality_frame_count(config, "rppg"), 32)
        self.assertEqual(resolve_modality_frame_count(config, "face_mesh"), 16)
        self.assertEqual(
            resolve_modality_frame_counts(config, ("rgb", "rppg")),
            {"rgb": 8, "rppg": 32},
        )

    def test_validate_branch_token_config_rejects_fusion_time_step_mismatch(self):
        config = {
            "frames": 16,
            "fusion": {"max_time_steps": 31},
        }

        with self.assertRaisesRegex(ValueError, "fusion.max_time_steps"):
            validate_branch_token_config(
                config,
                modalities=("fau",),
                fusion_max_time_steps=config["fusion"]["max_time_steps"],
            )

    def test_fuse_selected_modalities_returns_token_bank_metadata(self):
        registry = build_registry(dim=16)
        fusion_module = self.build_test_fusion(dim=16)
        batch = {
            "rgb_features": torch.randn(1, 8, 768),
            "rppg_features": torch.randn(1, 16, 12),
        }

        fusion_output = fuse_selected_modalities(
            registry,
            batch,
            ("rgb", "rppg"),
            fusion_module,
        )

        self.assertIsInstance(fusion_output, FusionOutput)
        self.assertEqual(tuple(fusion_output.tokens.shape), (1, 72, 16))
        self.assertEqual(tuple(fusion_output.fused.shape), (1, 16))
        self.assertEqual(tuple(fusion_output.cls_token.shape), (1, 16))
        self.assertEqual(tuple(fusion_output.fused_tokens.shape), (1, 73, 16))
        self.assertEqual(tuple(fusion_output.token_mask.shape), (72,))
        self.assertEqual(tuple(fusion_output.time_ids.shape), (72,))
        self.assertEqual(tuple(fusion_output.modality_ids.shape), (72,))
        self.assertEqual(fusion_output.modality_names, FIXED_SLOT_MODALITIES)
        self.assertTrue(torch.equal(fusion_output.time_ids[:8], torch.arange(8)))
        self.assertTrue(torch.equal(fusion_output.time_ids[8:40], torch.arange(32)))
        self.assertTrue(torch.equal(fusion_output.time_ids[40:44], torch.arange(4)))
        self.assertTrue(torch.equal(fusion_output.time_ids[44:48], torch.arange(4)))
        self.assertTrue(torch.equal(fusion_output.time_ids[48:64], torch.arange(16)))
        self.assertTrue(torch.equal(fusion_output.time_ids[64:68], torch.arange(4)))
        self.assertTrue(torch.equal(fusion_output.time_ids[68:72], torch.arange(4)))
        self.assertTrue(
            torch.equal(
                fusion_output.token_mask,
                torch.tensor(
                    [True] * 8
                    + [False] * 32
                    + [True] * 4
                    + [False] * 4
                    + [False] * 16
                    + [False] * 4
                    + [False] * 4
                ),
            )
        )
        self.assertTrue(torch.equal(fusion_output.modality_ids[:8], torch.tensor([0] * 8)))
        self.assertTrue(
            torch.equal(
                fusion_output.modality_ids[40:44],
                torch.tensor([MODALITY_TO_ID["rppg"]] * 4),
            )
        )
        self.assertTrue(torch.count_nonzero(fusion_output.tokens[:, 8:40]).item() == 0)

    def test_fuse_selected_modalities_uses_stable_subset_modality_ids(self):
        registry = build_registry(dim=8)
        fusion_module = self.build_test_fusion(dim=8)
        batch = {
            "eye_gaze": torch.randn(1, 2, 8),
            "rppg_features": torch.randn(1, 2, 12),
        }

        fusion_output = fuse_selected_modalities(
            registry,
            batch,
            ("eye_gaze", "rppg"),
            fusion_module,
        )

        self.assertTrue(
            torch.equal(
                fusion_output.modality_ids[40:44],
                torch.tensor([MODALITY_TO_ID["rppg"]] * 4),
            )
        )
        self.assertTrue(
            torch.equal(
                fusion_output.modality_ids[44:48],
                torch.tensor([MODALITY_TO_ID["eye_gaze"]] * 4),
            )
        )
        self.assertTrue(
            torch.equal(
                fusion_output.token_mask,
                torch.tensor(
                    [False] * 8
                    + [False] * 32
                    + [True] * 4
                    + [True] * 4
                    + [False] * 16
                    + [False] * 4
                    + [False] * 4
                ),
            )
        )

    def test_fuse_selected_modalities_supports_mixed_token_layouts(self):
        registry = build_registry(dim=12)
        fusion_module = self.build_test_fusion(dim=12)
        batch = {
            "rgb_features": torch.randn(1, 8, 768),
            "fau_features": torch.randn(1, 3, 2, 20),
        }

        fusion_output = fuse_selected_modalities(
            registry,
            batch,
            ("rgb", "fau"),
            fusion_module,
        )

        self.assertEqual(tuple(fusion_output.tokens.shape), (1, 72, 12))
        self.assertEqual(tuple(fusion_output.fused_tokens.shape), (1, 73, 12))
        self.assertTrue(
            torch.equal(
                fusion_output.token_mask,
                torch.tensor(
                    [True] * 8
                    + [True] * 32
                    + [False] * 4
                    + [False] * 4
                    + [False] * 16
                    + [False] * 4
                    + [False] * 4
                ),
            )
        )

    def test_depth_only_path_uses_final_bank_slots(self):
        registry = build_registry(dim=12)
        fusion_module = self.build_test_fusion(dim=12)
        batch = {"depth_features": torch.randn(1, 16, 384)}

        fusion_output = fuse_selected_modalities(
            registry,
            batch,
            ("depth",),
            fusion_module,
        )

        self.assertEqual(tuple(fusion_output.tokens.shape), (1, 72, 12))
        self.assertEqual(tuple(fusion_output.fused_tokens.shape), (1, 73, 12))
        self.assertEqual(int(fusion_output.token_mask.sum().item()), 4)
        self.assertTrue(
            torch.equal(
                fusion_output.token_mask,
                torch.tensor([False] * 64 + [True] * 4 + [False] * 4),
            )
        )
        self.assertTrue(
            torch.equal(
                fusion_output.modality_ids[64:68],
                torch.tensor([MODALITY_TO_ID["depth"]] * 4),
            )
        )

    def test_rgb_depth_path_keeps_depth_final_slots(self):
        registry = build_registry(dim=12)
        fusion_module = self.build_test_fusion(dim=12)
        batch = {
            "rgb_features": torch.randn(1, 8, 768),
            "depth_features": torch.randn(1, 16, 384),
        }

        fusion_output = fuse_selected_modalities(
            registry,
            batch,
            ("rgb", "depth"),
            fusion_module,
        )

        self.assertEqual(int(fusion_output.token_mask.sum().item()), 12)
        self.assertTrue(torch.equal(fusion_output.token_mask[:8], torch.tensor([True] * 8)))
        self.assertTrue(torch.equal(fusion_output.token_mask[8:64], torch.tensor([False] * 56)))
        self.assertTrue(torch.equal(fusion_output.token_mask[64:68], torch.tensor([True] * 4)))
        self.assertTrue(torch.equal(fusion_output.token_mask[68:72], torch.tensor([False] * 4)))

    def test_face_mesh_branch_emits_fixed_time_ids(self):
        registry = build_registry(dim=12)
        output = registry["face_mesh"].encode(
            {"face_mesh": torch.randn(1, 3, len(FACE_MESH_CONTOUR_INDICES), 3)}
        )

        self.assertEqual(
            tuple(output.tokens.shape),
            (1, 16, 12),
        )
        self.assertTrue(torch.equal(output.time_ids, torch.arange(16)))

    def test_build_local_encoders_requires_rgb_checkpoint(self):
        config = {
            "frames": 16,
            "image_size": 224,
            "rgb": {"checkpoint_path": None},
            "fau": {
                "backbone": "swin_transformer_tiny",
                "num_classes": 12,
                "checkpoint_path": None,
            },
            "rppg": {"checkpoint_path": None},
        }

        with self.assertRaisesRegex(ValueError, "RGB checkpoint_path is required"):
            build_local_encoders(config, modalities=("rgb",))

    def test_masked_disabled_slots_do_not_change_cls_token(self):
        fusion_module = self.build_test_fusion(dim=10)
        time_ids = torch.cat(
            [
                torch.arange(8),
                torch.arange(32),
                torch.arange(4),
                torch.arange(4),
                torch.arange(16),
                torch.arange(4),
                torch.arange(4),
            ]
        )
        modality_ids = torch.tensor(
            [MODALITY_TO_ID["rgb"]] * 8
            + [MODALITY_TO_ID["fau"]] * 32
            + [MODALITY_TO_ID["rppg"]] * 4
            + [MODALITY_TO_ID["eye_gaze"]] * 4
            + [MODALITY_TO_ID["face_mesh"]] * 16
            + [MODALITY_TO_ID["depth"]] * 4
            + [MODALITY_TO_ID["fft"]] * 4
        )
        token_mask = torch.tensor(
            [True] * 8
            + [False] * 32
            + [True] * 4
            + [False] * 4
            + [False] * 16
            + [False] * 4
            + [False] * 4
        )
        tokens = torch.randn(1, 72, 10)
        perturbed_tokens = tokens.clone()
        perturbed_tokens[:, ~token_mask, :] = (
            torch.randn_like(perturbed_tokens[:, ~token_mask, :]) * 50.0
        )

        cls_token, _ = fusion_module(tokens, token_mask, time_ids, modality_ids)
        perturbed_cls_token, _ = fusion_module(
            perturbed_tokens,
            token_mask,
            time_ids,
            modality_ids,
        )

        self.assertTrue(torch.allclose(cls_token, perturbed_cls_token, atol=1e-6, rtol=1e-5))

    def test_missing_fusion_checkpoint_keeps_random_init_smoke_path_valid(self):
        config = {
            "dim": 16,
            "fusion": {
                "type": "token_transformer",
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "max_time_steps": 32,
                "checkpoint_path": None,
            },
        }

        fusion_module = build_fusion_from_config(config)

        self.assertIsInstance(fusion_module, TokenBankFusion)


if __name__ == "__main__":
    unittest.main()