from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

from fusion import FusionOutput
from prediction import PredictionBuildResult, PredictionOutput, TrainingConfig
from test_prediction import build_test_predictor
from training import RunConfig, load_training_result, train_model, validate_model


def build_test_config() -> dict[str, Any]:
    return {
        "device": "cpu",
        "modalities": ["rgb", "fau", "rppg", "eye_gaze"],
        "frames": 16,
        "image_size": 224,
        "dim": 16,
        "modality_weights": {"rgb": 1.0, "fau": 1.0, "rppg": 1.0, "eye_gaze": 1.0},
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
            "lr_head": 5e-2,
            "lr_fusion": 5e-2,
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
        "seed": 0,
    }


class PrecomputedFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, labels: torch.Tensor):
        polarity = labels * 2.0 - 1.0
        self.items = [
            {
                "rgb_features": polarity[index].view(1, 1).repeat(8, 12),
                "fau_features": polarity[index].view(1, 1, 1).repeat(16, 12, 10),
                "rppg_features": polarity[index].view(1, 1).repeat(16, 9),
                "eye_gaze": polarity[index].view(1, 1).repeat(16, 8),
                "label": labels[index].view(1),
            }
            for index in range(labels.shape[0])
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate_precomputed_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = values
    return batch


def build_precomputed_loaders() -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_labels = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    val_labels = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    train_loader = DataLoader(
        PrecomputedFeatureDataset(train_labels),
        batch_size=4,
        shuffle=True,
        collate_fn=collate_precomputed_batch,
    )
    val_loader = DataLoader(
        PrecomputedFeatureDataset(val_labels),
        batch_size=4,
        shuffle=False,
        collate_fn=collate_precomputed_batch,
    )
    return train_loader, val_loader


def build_prediction_result(model=None) -> tuple[PredictionBuildResult, Any]:
    torch.manual_seed(0)
    predictor = model or build_test_predictor(freeze_encoders=True)
    result = PredictionBuildResult(
        model=predictor,
        classifier_config=None,  # type: ignore[arg-type]
        training_config=TrainingConfig(
            freeze_encoders=True,
            lr_head=5e-2,
            lr_fusion=5e-2,
            pos_weight=1.0,
        ),
        device=torch.device("cpu"),
        warnings=(),
    )
    return result, predictor


@dataclass
class FixedLogitSample:
    logits: float
    label: float


class FixedLogitDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[FixedLogitSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "fixed_logits": torch.tensor([sample.logits], dtype=torch.float32),
            "label": torch.tensor([sample.label], dtype=torch.float32),
        }


def collate_fixed_logits(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "fixed_logits": torch.stack([item["fixed_logits"] for item in items], dim=0),
        "label": torch.stack([item["label"] for item in items], dim=0),
    }


class FixedLogitModel(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> PredictionOutput:
        logits = batch["fixed_logits"]
        batch_size = logits.shape[0]
        fusion_output = FusionOutput(
            fused=torch.zeros(batch_size, 1),
            tokens=torch.zeros(batch_size, 1, 1),
            time_ids=torch.zeros(1, dtype=torch.long),
            modality_ids=torch.zeros(1, dtype=torch.long),
            modality_names=("fixed",),
            cls_token=torch.zeros(batch_size, 1),
            fused_tokens=torch.zeros(batch_size, 2, 1),
        )
        return PredictionOutput(
            logits=logits,
            probs=torch.sigmoid(logits),
            fusion_output=fusion_output,
        )


class TrainingTest(unittest.TestCase):
    def test_tiny_overfit_writes_checkpoints_and_reaches_near_zero_loss(self):
        config = build_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                epochs=120,
                batch_size=4,
                num_workers=0,
                weight_decay=0.0,
                output_dir=Path(tmpdir),
                save_every_epoch=True,
            )
            build_result, _ = build_prediction_result()

            with patch("training._build_dataloaders", return_value=build_precomputed_loaders()), patch(
                "training.build_prediction_model",
                return_value=build_result,
            ):
                result = train_model(config=config, manifest_path="/tmp/unused.csv", run_config=run_config)

            self.assertTrue(result.best_checkpoint_path.exists())
            self.assertTrue(result.last_checkpoint_path.exists())
            self.assertTrue((Path(tmpdir) / "epoch_0120.pt").exists())
            self.assertLess(result.history[-1]["train"]["loss"], 0.05)
            self.assertLess(result.history[-1]["val"]["loss"], 0.05)

    def test_validate_model_returns_expected_metrics_and_single_class_auc_none(self):
        criterion = torch.nn.BCEWithLogitsLoss()
        model = FixedLogitModel()
        loader = DataLoader(
            FixedLogitDataset(
                [
                    FixedLogitSample(logits=-2.0, label=0.0),
                    FixedLogitSample(logits=-1.0, label=0.0),
                    FixedLogitSample(logits=1.0, label=1.0),
                    FixedLogitSample(logits=2.0, label=1.0),
                ]
            ),
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fixed_logits,
        )

        metrics = validate_model(model=model, dataloader=loader, criterion=criterion, device=torch.device("cpu"))

        self.assertAlmostEqual(metrics.accuracy, 1.0, places=6)
        self.assertAlmostEqual(metrics.precision, 1.0, places=6)
        self.assertAlmostEqual(metrics.recall, 1.0, places=6)
        self.assertAlmostEqual(metrics.f1, 1.0, places=6)
        self.assertIsNotNone(metrics.roc_auc)
        self.assertAlmostEqual(metrics.roc_auc or 0.0, 1.0, places=6)

        single_class_loader = DataLoader(
            FixedLogitDataset(
                [
                    FixedLogitSample(logits=-2.0, label=0.0),
                    FixedLogitSample(logits=-1.0, label=0.0),
                ]
            ),
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fixed_logits,
        )
        single_class_metrics = validate_model(
            model=model,
            dataloader=single_class_loader,
            criterion=criterion,
            device=torch.device("cpu"),
        )
        self.assertIsNone(single_class_metrics.roc_auc)

    def test_checkpoint_round_trip_restores_same_logits(self):
        config = build_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = RunConfig(
                epochs=8,
                batch_size=4,
                num_workers=0,
                weight_decay=0.0,
                output_dir=Path(tmpdir),
            )
            build_result, trained_model = build_prediction_result()
            loaders = build_precomputed_loaders()

            with patch("training._build_dataloaders", return_value=loaders), patch(
                "training.build_prediction_model",
                return_value=build_result,
            ):
                result = train_model(config=config, manifest_path="/tmp/unused.csv", run_config=run_config)

            reloaded_build_result, _ = build_prediction_result()
            batch = next(iter(build_precomputed_loaders()[1]))
            expected_logits = trained_model(batch).logits.detach()
            with patch("training.build_prediction_model", return_value=reloaded_build_result):
                loaded = load_training_result(result.last_checkpoint_path)
            actual_logits = loaded.model(batch).logits.detach()

            self.assertTrue(torch.allclose(actual_logits, expected_logits, atol=1e-6))
            self.assertEqual(loaded.enabled_modalities, trained_model.enabled_modalities)
            loaded.model.close()

    def test_resume_continues_history_and_epoch_counter(self):
        config = build_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            first_run = RunConfig(
                epochs=1,
                batch_size=4,
                num_workers=0,
                weight_decay=0.0,
                output_dir=output_dir,
            )
            initial_build_result, _ = build_prediction_result()
            with patch("training._build_dataloaders", return_value=build_precomputed_loaders()), patch(
                "training.build_prediction_model",
                return_value=initial_build_result,
            ):
                initial_result = train_model(config=config, manifest_path="/tmp/unused.csv", run_config=first_run)

            resumed_run = RunConfig(
                epochs=2,
                batch_size=4,
                num_workers=0,
                weight_decay=0.0,
                output_dir=output_dir,
                resume_from=initial_result.last_checkpoint_path,
            )
            resumed_build_result, _ = build_prediction_result()
            with patch("training._build_dataloaders", return_value=build_precomputed_loaders()), patch(
                "training.build_prediction_model",
                return_value=resumed_build_result,
            ):
                resumed_result = train_model(config=config, manifest_path="/tmp/unused.csv", run_config=resumed_run)

            self.assertEqual(len(resumed_result.history), 2)
            self.assertEqual(resumed_result.history[0]["epoch"], 1)
            self.assertEqual(resumed_result.history[1]["epoch"], 2)

    def test_train_model_requires_non_empty_train_and_val_splits(self):
        config = build_test_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.csv"
            manifest_path.write_text(
                "path,label,class_name,source_id,split,identity_id\n"
                "/tmp/example.mp4,0,real,clipA,train,\n",
                encoding="utf-8",
            )
            run_config = RunConfig(
                epochs=1,
                batch_size=2,
                num_workers=0,
                weight_decay=0.0,
                output_dir=Path(tmpdir) / "out",
            )

            with self.assertRaisesRegex(ValueError, "Manifest must contain at least one `val` example."):
                train_model(config=config, manifest_path=manifest_path, run_config=run_config)


if __name__ == "__main__":
    unittest.main()
