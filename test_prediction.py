from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from extractors.eye_gaze import EYE_GAZE_COLUMNS, EyeGazeExtractor
from extractors.face_mesh import FACE_MESH_CONTOUR_INDICES, FaceMeshExtractor
from extractors.factory import ExtractorFactoryResult
from extractors.fau import FAUExtractor
from extractors.rgb import RGBExtractor
from extractors.rppg import RPPGExtractor
from fusion import TokenBankFusion
from prediction import (
    ClassifierConfig,
    ClipRealFakePredictor,
    TrainingConfig,
    VideoRealFakeHead,
    build_binary_classification_loss,
    build_prediction_model,
    resolve_model_device,
)
from registry import MODALITY_TO_ID, build_registry
from encoders.factory import EncoderFactoryResult


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


def build_test_predictor(
    dim: int = 16,
    enabled_modalities: tuple[str, ...] = ("rgb", "fau", "rppg", "eye_gaze", "face_mesh"),
    freeze_encoders: bool = True,
) -> ClipRealFakePredictor:
    rgb_encoder = ParameterizedDummyRGBEncoder()
    fau_encoder = ParameterizedDummyFAUEncoder()
    rppg_encoder = ParameterizedDummyRPPGEncoder()
    extractors = {
        "rgb": RGBExtractor(rgb_encoder, image_size=32),
        "fau": FAUExtractor(fau_encoder),
        "rppg": RPPGExtractor(rppg_encoder),
        "eye_gaze": EyeGazeExtractor(detect_features_fn=fake_eye_gaze_detector),
        "face_mesh": FaceMeshExtractor(detect_landmarks_fn=fake_face_mesh_detector),
    }
    model = ClipRealFakePredictor(
        registry=build_registry(dim=dim),
        fusion_module=TokenBankFusion(
            dim=dim,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            max_time_steps=16,
            num_modalities=len(MODALITY_TO_ID),
        ),
        classifier_head=VideoRealFakeHead(input_dim=dim, hidden_dim=dim, dropout=0.1),
        enabled_modalities=enabled_modalities,
        modality_weights={name: 1.0 for name in enabled_modalities},
        extractors={name: extractors[name] for name in enabled_modalities},
        encoder_modules=nn.ModuleDict(
            {
                "rgb": rgb_encoder,
                "fau": fau_encoder,
                "rppg": rppg_encoder,
            }
        ),
    )
    if freeze_encoders:
        model.freeze_encoder_parameters()
    return model


def build_raw_batch(num_frames: int = 16) -> dict[str, object]:
    frames = [np.full((16, 16, 3), 64 + frame_index, dtype=np.uint8) for frame_index in range(num_frames)]
    return {
        "video": torch.randn(1, 3, num_frames, 16, 16),
        "video_rgb_frames": frames,
    }


class PredictionModelTest(unittest.TestCase):
    def test_clip_predictor_returns_logits_probs_and_full_token_layout(self):
        model = build_test_predictor()

        output = model(build_raw_batch())

        self.assertEqual(tuple(output.logits.shape), (1, 1))
        self.assertEqual(tuple(output.probs.shape), (1, 1))
        self.assertTrue(torch.all(output.probs >= 0.0))
        self.assertTrue(torch.all(output.probs <= 1.0))
        self.assertEqual(tuple(output.fusion_output.tokens.shape), (1, 64, 16))
        self.assertEqual(tuple(output.fusion_output.fused_tokens.shape), (1, 65, 16))

    def test_gradient_flow_skips_frozen_encoders_but_updates_fusion_and_head(self):
        model = build_test_predictor(freeze_encoders=True)
        criterion = build_binary_classification_loss(
            TrainingConfig(
                freeze_encoders=True,
                lr_head=1e-3,
                lr_fusion=1e-4,
                pos_weight=1.0,
            )
        )

        output = model(build_raw_batch())
        loss = criterion(output.logits, torch.ones_like(output.logits))
        loss.backward()

        self.assertIsNotNone(model.classifier_head.layers[0].weight.grad)
        self.assertIsNotNone(model.fusion_module.cls_token.grad)
        self.assertIsNotNone(model.registry["rgb"].proj.weight.grad)
        self.assertIsNotNone(model.registry["fau"].proj.weight.grad)
        self.assertIsNotNone(model.registry["rppg"].proj.weight.grad)
        self.assertIsNotNone(model.registry["eye_gaze"].proj[0].weight.grad)
        self.assertIsNotNone(model.registry["face_mesh"].proj[0].weight.grad)
        self.assertIsNone(model.encoder_modules["rgb"].scale.grad)
        self.assertIsNone(model.encoder_modules["fau"].scale.grad)
        self.assertIsNone(model.encoder_modules["rppg"].scale.grad)

    def test_modality_subset_keeps_stable_ids_and_expected_token_count(self):
        model = build_test_predictor(enabled_modalities=("face_mesh", "rppg"))

        output = model(build_raw_batch())

        self.assertEqual(tuple(output.fusion_output.tokens.shape), (1, 20, 16))
        self.assertEqual(tuple(output.fusion_output.fused_tokens.shape), (1, 21, 16))
        self.assertTrue(
            torch.equal(
                output.fusion_output.modality_ids,
                torch.tensor(
                    [MODALITY_TO_ID["face_mesh"]] * 16 + [MODALITY_TO_ID["rppg"]] * 4
                ),
            )
        )

    def test_tiny_overfit_on_precomputed_clip_features(self):
        torch.manual_seed(0)
        model = build_test_predictor(freeze_encoders=True)
        training_config = TrainingConfig(
            freeze_encoders=True,
            lr_head=5e-2,
            lr_fusion=5e-2,
            pos_weight=1.0,
        )
        optimizer = model.build_optimizer(training_config, weight_decay=0.0)
        criterion = build_binary_classification_loss(training_config)

        labels = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
        polarity = labels * 2.0 - 1.0
        batch = {
            "rgb_features": polarity.view(8, 1, 1).repeat(1, 8, 12),
            "fau_features": polarity.view(8, 1, 1, 1).repeat(1, 16, 12, 10),
            "rppg_features": polarity.view(8, 1, 1).repeat(1, 16, 9),
            "eye_gaze": polarity.view(8, 1, 1).repeat(1, 16, 8),
            "face_mesh": polarity.view(8, 1, 1, 1).repeat(1, 16, len(FACE_MESH_CONTOUR_INDICES), 3),
        }

        model.train()
        for _ in range(120):
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(batch)
        predictions = (output.probs >= 0.5).float()
        self.assertTrue(torch.equal(predictions, labels))
        self.assertLess(float(criterion(output.logits, labels).item()), 0.05)

    def test_classifier_config_dataclass_keeps_requested_shape(self):
        config = ClassifierConfig(hidden_dim=32, dropout=0.2)
        head = VideoRealFakeHead(input_dim=16, hidden_dim=config.hidden_dim, dropout=config.dropout)

        logits = head(torch.randn(3, 16))

        self.assertEqual(tuple(logits.shape), (3, 1))

    def test_resolve_model_device_accepts_cpu(self):
        self.assertEqual(str(resolve_model_device({"device": "cpu"})), "cpu")

    def test_build_prediction_model_moves_modules_to_requested_cpu_device(self):
        config = {
            "device": "cpu",
            "modalities": ["face_mesh"],
            "frames": 16,
            "image_size": 224,
            "dim": 16,
            "modality_weights": {"face_mesh": 1.0},
            "fusion": {
                "type": "token_transformer",
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "max_time_steps": 16,
                "checkpoint_path": None,
            },
            "classifier": {
                "hidden_dim": 16,
                "dropout": 0.1,
            },
            "training": {
                "freeze_encoders": True,
                "lr_head": 1e-3,
                "lr_fusion": 1e-4,
                "pos_weight": 1.0,
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
            "face_mesh": {"output_tokens_per_frame": 2},
        }

        encoder_result = EncoderFactoryResult(
            fau_encoder=None,
            rgb_encoder=None,
            rppg_encoder=None,
            warnings=(),
        )
        extractors_result = ExtractorFactoryResult(
            extractors={"face_mesh": FaceMeshExtractor(detect_landmarks_fn=fake_face_mesh_detector)},
            warnings=(),
        )
        with patch("prediction.build_local_encoders", return_value=encoder_result), patch(
            "prediction.build_extractors_from_encoders", return_value=extractors_result
        ):
            build_result = build_prediction_model(config, modalities=("face_mesh",))

        self.assertEqual(str(build_result.device), "cpu")
        self.assertEqual(str(next(build_result.model.parameters()).device), "cpu")
        self.assertEqual(build_result.model.registry["face_mesh"].output_tokens_per_frame, 2)
        build_result.model.close()

    def test_build_prediction_model_rejects_fusion_max_time_step_mismatch(self):
        config = {
            "device": "cpu",
            "modalities": ["rppg"],
            "frames": 16,
            "image_size": 224,
            "dim": 16,
            "modality_weights": {"rppg": 1.0},
            "fusion": {
                "type": "token_transformer",
                "num_layers": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "max_time_steps": 3,
                "checkpoint_path": None,
            },
            "classifier": {
                "hidden_dim": 16,
                "dropout": 0.1,
            },
            "training": {
                "freeze_encoders": True,
                "lr_head": 1e-3,
                "lr_fusion": 1e-4,
                "pos_weight": 1.0,
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
                "output_tokens_per_clip": 4,
            },
            "eye_gaze": {},
            "face_mesh": {},
        }

        with self.assertRaisesRegex(ValueError, "fusion.max_time_steps"):
            build_prediction_model(config, modalities=("rppg",))


if __name__ == "__main__":
    unittest.main()
