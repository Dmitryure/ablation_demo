from __future__ import annotations

from pathlib import Path
import unittest

import torch
import torch.nn as nn
from tempfile import TemporaryDirectory

from dataset import VideoExample
from fusion import FusionOutput
from scripts.run_tiny_overfit import (
    PredictionRow,
    build_feature_loader,
    build_modality_sets,
    find_label_name_overlaps,
    format_miss_line,
    miss_rate_percent,
    modality_set_name,
    predict_rows,
    resolve_base_modalities,
    should_log_epoch,
    summarize_seen_predict_rows,
    summarize_unseen_predict_rows,
    train_one_epoch,
    write_metrics,
    write_modality_accuracy_plot,
    write_train_accuracy_plot,
)
from task_model import BinaryFusionClassifier, BinaryFusionHead


def build_fusion_output(fused: torch.Tensor) -> FusionOutput:
    batch, dim = fused.shape
    tokens = torch.zeros(batch, 1, dim, device=fused.device)
    token_ids = torch.zeros(1, dtype=torch.long, device=fused.device)
    return FusionOutput(
        fused=fused,
        tokens=tokens,
        token_mask=torch.ones(1, dtype=torch.bool, device=fused.device),
        time_ids=token_ids,
        modality_ids=token_ids,
        modality_names=("dummy",),
        cls_token=fused,
        fused_tokens=torch.cat([fused.unsqueeze(1), tokens], dim=1),
    )


class FeaturePipeline(nn.Module):
    def forward(self, batch):
        return build_fusion_output(batch["feature"])


def build_feature_items() -> list[dict[str, object]]:
    return [
        {
            "feature": torch.tensor([-2.0]),
            "label": torch.tensor([0.0]),
            "path": "real_a.mp4",
            "class_name": "real",
            "split": "train",
        },
        {
            "feature": torch.tensor([-1.5]),
            "label": torch.tensor([0.0]),
            "path": "real_b.mp4",
            "class_name": "real",
            "split": "train",
        },
        {
            "feature": torch.tensor([1.5]),
            "label": torch.tensor([1.0]),
            "path": "fake_a.mp4",
            "class_name": "fake",
            "split": "train",
        },
        {
            "feature": torch.tensor([2.0]),
            "label": torch.tensor([1.0]),
            "path": "fake_b.mp4",
            "class_name": "fake",
            "split": "train",
        },
    ]


class TinyOverfitTest(unittest.TestCase):
    def test_resolve_base_modalities_uses_request_or_config_default(self):
        config = {"modalities": ["rgb", "fau"]}

        self.assertEqual(resolve_base_modalities(config, None), ("rgb", "fau"))
        self.assertEqual(resolve_base_modalities(config, ["face_mesh"]), ("face_mesh",))

    def test_build_modality_sets_supports_singletons_plus_all(self):
        sets = build_modality_sets(("rgb", "fau", "rppg"), mode="singletons-plus-all")

        self.assertEqual(sets, [("rgb",), ("fau",), ("rppg",), ("rgb", "fau", "rppg")])

    def test_build_modality_sets_supports_all_non_empty_combinations(self):
        sets = build_modality_sets(("rgb", "fau"), mode="all")

        self.assertEqual(sets, [("rgb",), ("fau",), ("rgb", "fau")])
        self.assertEqual(modality_set_name(("rgb", "fau")), "rgb__fau")

    def test_should_log_epoch_logs_tens_and_last_epoch(self):
        self.assertFalse(should_log_epoch(9, 50))
        self.assertTrue(should_log_epoch(10, 50))
        self.assertTrue(should_log_epoch(7, 7))

    def test_find_label_name_overlaps_matches_class_and_filename(self):
        train_examples = [
            VideoExample(Path("tests/overfit_videos/real/a.mp4"), 0, "real", "a", "train"),
            VideoExample(Path("tests/overfit_videos/fake/a.mp4"), 1, "fake", "a", "train"),
        ]
        predict_examples = [
            VideoExample(Path("tests/predict_videos/real/a.mp4"), 0, "real", "a", "test"),
            VideoExample(Path("tests/predict_videos/fake/b.mp4"), 1, "fake", "b", "test"),
        ]

        overlaps = find_label_name_overlaps(train_examples, predict_examples)

        self.assertEqual(overlaps, [("real", "a.mp4")])

    def test_train_loop_can_overfit_dummy_fused_features(self):
        torch.manual_seed(0)
        classifier = BinaryFusionClassifier(
            pipeline=FeaturePipeline(),
            head=BinaryFusionHead(dim=1),
        )
        items = build_feature_items()
        loader = build_feature_loader(items, batch_size=4, shuffle=False)
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.2)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        first_loss, _ = train_one_epoch(classifier, loader, optimizer, loss_fn)
        last_loss = first_loss
        for _ in range(30):
            last_loss, _ = train_one_epoch(classifier, loader, optimizer, loss_fn)
        accuracy, rows = predict_rows(classifier, build_feature_loader(items, batch_size=4, shuffle=False))

        self.assertLess(last_loss, first_loss)
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(len(rows), 4)
        self.assertEqual({row.prediction for row in rows}, {0, 1})

    def test_summarize_seen_predict_rows_counts_only_seen_test_misses(self):
        rows = [
            PredictionRow(
                path="train_real.mp4",
                class_name="real",
                label=0,
                prediction=0,
                probability=0.1,
                split="train",
                seen_in_train=True,
            ),
            PredictionRow(
                path="seen_fake.mp4",
                class_name="fake",
                label=1,
                prediction=1,
                probability=0.9,
                split="test",
                seen_in_train=True,
            ),
            PredictionRow(
                path="seen_real.mp4",
                class_name="real",
                label=0,
                prediction=1,
                probability=0.8,
                split="test",
                seen_in_train=True,
            ),
            PredictionRow(
                path="new_real.mp4",
                class_name="real",
                label=0,
                prediction=1,
                probability=0.8,
                split="test",
                seen_in_train=False,
            ),
        ]

        summary = summarize_seen_predict_rows(rows)

        self.assertEqual(summary["predict_seen_count"], 2)
        self.assertEqual(summary["predict_seen_misses"], 1)
        self.assertEqual(summary["predict_seen_accuracy"], 0.5)

    def test_summarize_unseen_predict_rows_counts_only_unseen_test_misses(self):
        rows = [
            PredictionRow(
                path="train_real.mp4",
                class_name="real",
                label=0,
                prediction=0,
                probability=0.1,
                split="train",
                seen_in_train=True,
            ),
            PredictionRow(
                path="seen_fake.mp4",
                class_name="fake",
                label=1,
                prediction=0,
                probability=0.4,
                split="test",
                seen_in_train=True,
            ),
            PredictionRow(
                path="new_real.mp4",
                class_name="real",
                label=0,
                prediction=1,
                probability=0.8,
                split="test",
                seen_in_train=False,
            ),
            PredictionRow(
                path="new_fake.mp4",
                class_name="fake",
                label=1,
                prediction=1,
                probability=0.9,
                split="test",
                seen_in_train=False,
            ),
        ]

        summary = summarize_unseen_predict_rows(rows)

        self.assertEqual(summary["predict_unseen_count"], 2)
        self.assertEqual(summary["predict_unseen_misses"], 1)
        self.assertEqual(summary["predict_unseen_accuracy"], 0.5)

    def test_miss_rate_percent_and_formatting(self):
        self.assertEqual(miss_rate_percent(1, 4), 25.0)
        self.assertIsNone(miss_rate_percent(0, 0))
        self.assertEqual(format_miss_line("predict_seen_misses", 1, 4), "predict_seen_misses=1 of 4 (25.00%)")
        self.assertEqual(format_miss_line("predict_unseen_misses", 0, 0), "predict_unseen_misses=0 of 0 (n/a)")

    def test_write_train_accuracy_plot_writes_png(self):
        rows = [
            {"epoch": 1, "train_accuracy": "0.50000000"},
            {"epoch": 2, "train_accuracy": "1.00000000"},
        ]
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_accuracy.png"

            wrote = write_train_accuracy_plot(path, rows, title="test")

            self.assertTrue(wrote)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_write_modality_accuracy_plot_writes_subplots_png(self):
        rows = [
            {"epoch": 1, "train_loss": "0.70000000", "train_accuracy": "0.50000000"},
            {"epoch": 2, "train_loss": "0.10000000", "train_accuracy": "1.00000000"},
        ]
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rgb_metrics = root / "rgb" / "metrics.csv"
            fau_metrics = root / "fau" / "metrics.csv"
            write_metrics(rgb_metrics, rows)
            write_metrics(fau_metrics, rows)
            path = root / "all_modalities.png"

            wrote = write_modality_accuracy_plot(
                path,
                [
                    {"metrics_csv": str(rgb_metrics), "modalities": ["rgb"]},
                    {"metrics_csv": str(fau_metrics), "modalities": ["fau"]},
                ],
            )

            self.assertTrue(wrote)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
