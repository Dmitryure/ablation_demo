from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from encoders.factory import EncoderFactoryResult
from extractors.depth import DepthExtractor
from extractors.eye_gaze import EYE_GAZE_COLUMNS, EyeGazeExtractor
from extractors.face_mesh import FACE_MESH_CONTOUR_INDICES, FaceMeshExtractor
from extractors.factory import ExtractorFactoryResult
from extractors.fau import FAUExtractor
from extractors.rgb import RGBExtractor
from extractors.rppg import RPPGExtractor
from fusion import TokenBankFusion
from pipeline import ClipFusionPipeline, build_fusion_pipeline, resolve_model_device
from registry import FIXED_SLOT_MODALITIES, MODALITY_TO_ID, build_registry


def fake_eye_gaze_detector(_: np.ndarray) -> dict[str, float]:
    return {name: float(index) / 10.0 for index, name in enumerate(EYE_GAZE_COLUMNS, start=1)}


def fake_face_mesh_detector(_: np.ndarray) -> np.ndarray:
    points = np.zeros((len(FACE_MESH_CONTOUR_INDICES), 3), dtype=np.float32)
    for index in range(points.shape[0]):
        points[index] = (index / 100.0, index / 200.0, -index / 300.0)
    return points


class ParameterizedDummyRGBEncoder(nn.Module):
    def __init__(self, token_count: int = 8, feature_dim: int = 12):
        super().__init__()
        self.token_count = token_count
        self.feature_dim = feature_dim
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        token_positions = torch.linspace(-1.0, 1.0, steps=self.token_count, device=x.device)
        features = token_positions.view(1, self.token_count, 1).repeat(batch, 1, self.feature_dim)
        return features * self.scale


class ParameterizedDummyFAUEncoder(nn.Module):
    def __init__(self, num_au: int = 12, feature_dim: int = 10):
        super().__init__()
        self.num_au = num_au
        self.feature_dim = feature_dim
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_frames = x.shape[0]
        base = torch.linspace(-0.5, 0.5, steps=self.num_au * self.feature_dim, device=x.device)
        features = base.view(1, self.num_au, self.feature_dim).repeat(batch_frames, 1, 1)
        features = features * self.scale
        logits = torch.ones(batch_frames, self.num_au, device=x.device) * self.scale
        edge_logits = torch.ones(batch_frames, self.num_au, 3, device=x.device) * self.scale
        return features, logits, edge_logits


class ParameterizedDummyRPPGEncoder(nn.Module):
    def __init__(self, feature_dim: int = 9):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, num_frames, _, _ = x.shape
        waveform = torch.linspace(-1.0, 1.0, steps=num_frames, device=x.device).view(1, num_frames)
        waveform = waveform.repeat(batch, 1) * self.scale
        features = waveform.unsqueeze(-1).repeat(1, 1, self.feature_dim)
        return waveform, features


class ParameterizedDummyDepthEncoder(nn.Module):
    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.feature_dim = feature_dim
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_frames = pixel_values.shape[0]
        base = torch.linspace(-0.25, 0.25, steps=self.feature_dim, device=pixel_values.device)
        return base.view(1, self.feature_dim).repeat(batch_frames, 1) * self.scale


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


def build_test_pipeline(
    dim: int = 16,
    enabled_modalities: tuple[str, ...] = ("rgb", "fau", "rppg", "eye_gaze", "face_mesh", "depth"),
) -> ClipFusionPipeline:
    rgb_encoder = ParameterizedDummyRGBEncoder()
    fau_encoder = ParameterizedDummyFAUEncoder()
    rppg_encoder = ParameterizedDummyRPPGEncoder()
    depth_encoder = ParameterizedDummyDepthEncoder()
    extractors = {
        "rgb": RGBExtractor(rgb_encoder, image_size=32),
        "fau": FAUExtractor(fau_encoder),
        "rppg": RPPGExtractor(rppg_encoder),
        "eye_gaze": EyeGazeExtractor(detect_features_fn=fake_eye_gaze_detector),
        "face_mesh": FaceMeshExtractor(detect_landmarks_fn=fake_face_mesh_detector),
        "depth": DepthExtractor(
            depth_encoder,
            processor=DummyDepthProcessor(),
            model_id_or_path="unused",
        ),
    }
    return ClipFusionPipeline(
        registry=build_registry(dim=dim),
        fusion_module=TokenBankFusion(
            dim=dim,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_time_steps=32,
            num_modalities=len(MODALITY_TO_ID),
        ),
        enabled_modalities=enabled_modalities,
        extractors={name: extractors[name] for name in enabled_modalities},
        encoder_modules=nn.ModuleDict(
            {
                "rgb": rgb_encoder,
                "fau": fau_encoder,
                "rppg": rppg_encoder,
                "depth": depth_encoder,
            }
        ),
    )


def build_raw_batch(num_frames: int = 16) -> dict[str, object]:
    frames = [np.full((16, 16, 3), 64 + frame_index, dtype=np.uint8) for frame_index in range(num_frames)]
    return {
        "video": torch.randn(1, 3, num_frames, 16, 16),
        "video_rgb_frames": frames,
    }


class PipelineTest(unittest.TestCase):
    def test_clip_pipeline_returns_fusion_output_and_full_token_layout(self):
        pipeline = build_test_pipeline()

        output = pipeline(build_raw_batch())

        self.assertEqual(tuple(output.tokens.shape), (1, 68, 16))
        self.assertEqual(tuple(output.token_mask.shape), (68,))
        self.assertEqual(tuple(output.fused.shape), (1, 16))
        self.assertEqual(tuple(output.cls_token.shape), (1, 16))
        self.assertEqual(tuple(output.fused_tokens.shape), (1, 69, 16))
        self.assertEqual(output.modality_names, FIXED_SLOT_MODALITIES)

    def test_prepare_features_reuses_precomputed_modalities(self):
        pipeline = build_test_pipeline(enabled_modalities=("fau", "rppg"))
        batch = {
            "fau_features": torch.randn(1, 16, 12, 10),
            "fau_au_logits": torch.randn(1, 16, 12),
            "fau_au_edge_logits": torch.randn(1, 16, 12, 3),
            "rppg_features": torch.randn(1, 16, 9),
            "rppg_waveform": torch.randn(1, 16),
        }

        features = pipeline.prepare_features(batch)

        self.assertIs(features["fau_features"], batch["fau_features"])
        self.assertIs(features["fau_au_logits"], batch["fau_au_logits"])
        self.assertIs(features["fau_au_edge_logits"], batch["fau_au_edge_logits"])
        self.assertIs(features["rppg_features"], batch["rppg_features"])
        self.assertIs(features["rppg_waveform"], batch["rppg_waveform"])
        self.assertEqual(pipeline.last_feature_timings, {"fau": 0.0, "rppg": 0.0})

    def test_modality_subset_keeps_stable_ids_and_expected_token_count(self):
        pipeline = build_test_pipeline(enabled_modalities=("face_mesh", "rppg"))

        output = pipeline(build_raw_batch())

        self.assertEqual(tuple(output.tokens.shape), (1, 68, 16))
        self.assertEqual(tuple(output.fused_tokens.shape), (1, 69, 16))
        self.assertEqual(int(output.token_mask.sum().item()), 20)
        self.assertTrue(torch.count_nonzero(output.tokens[:, :40]).item() == 0)
        self.assertTrue(torch.count_nonzero(output.tokens[:, 64:]).item() == 0)
        self.assertTrue(
            torch.equal(
                output.modality_ids,
                torch.tensor(
                    [MODALITY_TO_ID["rgb"]] * 8
                    + [MODALITY_TO_ID["fau"]] * 32
                    + [MODALITY_TO_ID["rppg"]] * 4
                    + [MODALITY_TO_ID["eye_gaze"]] * 4
                    + [MODALITY_TO_ID["face_mesh"]] * 16
                    + [MODALITY_TO_ID["depth"]] * 4
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                output.token_mask,
                torch.tensor(
                    [False] * 8
                    + [False] * 32
                    + [True] * 4
                    + [False] * 4
                    + [True] * 16
                    + [False] * 4
                ),
            )
        )

    def test_resolve_model_device_accepts_cpu(self):
        self.assertEqual(str(resolve_model_device({"device": "cpu"})), "cpu")

    def test_build_fusion_pipeline_moves_modules_to_requested_cpu_device(self):
        config = {
            "device": "cpu",
            "modalities": ["face_mesh"],
            "frames": 16,
            "image_size": 224,
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
            "rgb": {
                "checkpoint_path": "/tmp/unused-rgb-checkpoint.pth",
            },
            "fau": {
                "backbone": "swin_transformer_tiny",
                "num_classes": 12,
                "checkpoint_path": None,
            },
            "rppg": {
                "checkpoint_path": None,
            },
            "eye_gaze": {},
            "face_mesh": {"slot_count": 12},
        }

        encoder_result = EncoderFactoryResult(
            depth_encoder=None,
            fau_encoder=None,
            rgb_encoder=None,
            rppg_encoder=None,
            warnings=(),
        )
        extractors_result = ExtractorFactoryResult(
            extractors={"face_mesh": FaceMeshExtractor(detect_landmarks_fn=fake_face_mesh_detector)},
            warnings=(),
        )
        with patch("pipeline.build_local_encoders", return_value=encoder_result), patch(
            "pipeline.build_extractors_from_encoders", return_value=extractors_result
        ):
            build_result = build_fusion_pipeline(config, modalities=("face_mesh",))

        self.assertEqual(str(build_result.device), "cpu")
        self.assertEqual(str(next(build_result.pipeline.parameters()).device), "cpu")
        self.assertEqual(build_result.pipeline.registry["face_mesh"].slot_count, 12)
        build_result.pipeline.close()

    def test_build_fusion_pipeline_rejects_fusion_max_time_step_mismatch(self):
        config = {
            "device": "cpu",
            "modalities": ["rppg"],
            "frames": 16,
            "image_size": 224,
            "dim": 16,
            "fusion": {
                "type": "token_transformer",
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "max_time_steps": 31,
                "checkpoint_path": None,
            },
            "rgb": {
                "checkpoint_path": "/tmp/unused-rgb-checkpoint.pth",
            },
            "fau": {
                "backbone": "swin_transformer_tiny",
                "num_classes": 12,
                "checkpoint_path": None,
            },
            "rppg": {
                "checkpoint_path": None,
            },
            "eye_gaze": {},
            "face_mesh": {},
        }

        with self.assertRaisesRegex(ValueError, "fusion.max_time_steps"):
            build_fusion_pipeline(config, modalities=("rppg",))


if __name__ == "__main__":
    unittest.main()
