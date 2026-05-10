from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

import torch

from dataset import VideoExample
from scripts.run_iterative_cached_ablation import (
    BalancedTrainBatchSampler,
    PredictionRow,
    build_train_balance_summary,
    cap_fake_generators,
    fake_generator_counts,
    fake_generator_sample_weights,
    parse_fake_generator_loss_weights,
    prediction_generator_metrics,
    train_pos_weight,
)


def example(class_name: str, index: int, generator_id: str | None = None) -> VideoExample:
    return VideoExample(
        path=Path(f"/dataset/{class_name}/{generator_id or 'real'}_{index}.mp4"),
        label=0 if class_name == "real" else 1,
        class_name=class_name,
        source_id=f"{class_name}_{index}",
        split="train",
        generator_id="real" if class_name == "real" else generator_id,
    )


def flatten_batches(batches: list[list[int]]) -> list[int]:
    return [index for batch in batches for index in batch]


class TrainBalanceTest(unittest.TestCase):
    def test_class_balanced_batches_balance_imbalanced_epoch(self):
        examples = [
            *[example("real", index) for index in range(3)],
            *[example("fake", index, "gen_a") for index in range(9)],
        ]
        sampler = BalancedTrainBatchSampler(
            examples=examples,
            batch_size=4,
            mode="class_balanced_batches",
            seed=0,
        )

        sampled = flatten_batches(list(iter(sampler)))
        counts = Counter(examples[index].class_name for index in sampled)

        self.assertEqual(counts, {"real": 9, "fake": 9})
        self.assertEqual(len(sampler), 5)

    def test_generator_balanced_batches_balance_fake_generators(self):
        examples = [
            *[example("real", index) for index in range(6)],
            *[example("fake", index, "gen_a") for index in range(4)],
            *[example("fake", index, "gen_b") for index in range(1)],
            *[example("fake", index, "gen_c") for index in range(1)],
        ]
        sampler = BalancedTrainBatchSampler(
            examples=examples,
            batch_size=4,
            mode="generator_balanced_batches",
            seed=0,
        )

        sampled = flatten_batches(list(iter(sampler)))
        fake_generator_counts = Counter(
            examples[index].generator_id for index in sampled if examples[index].label == 1
        )

        self.assertEqual(fake_generator_counts, {"gen_a": 2, "gen_b": 2, "gen_c": 2})

    def test_balanced_sampler_is_seeded_and_epoch_order_changes(self):
        examples = [
            *[example("real", index) for index in range(6)],
            *[example("fake", index, "gen_a") for index in range(6)],
        ]
        sampler = BalancedTrainBatchSampler(
            examples=examples,
            batch_size=4,
            mode="class_balanced_batches",
            seed=7,
        )
        repeat_sampler = BalancedTrainBatchSampler(
            examples=examples,
            batch_size=4,
            mode="class_balanced_batches",
            seed=7,
        )

        first_epoch = list(iter(sampler))
        second_epoch = list(iter(sampler))
        repeat_first_epoch = list(iter(repeat_sampler))

        self.assertEqual(first_epoch, repeat_first_epoch)
        self.assertNotEqual(first_epoch, second_epoch)

    def test_class_weighted_loss_pos_weight_uses_real_over_fake(self):
        examples = [
            *[example("real", index) for index in range(3)],
            *[example("fake", index, "gen_a") for index in range(9)],
        ]

        summary = build_train_balance_summary(
            examples=examples,
            mode="class_weighted_loss",
            batch_size=4,
            seed=0,
        )

        self.assertAlmostEqual(train_pos_weight(examples), 3 / 9)
        self.assertAlmostEqual(summary["loss"]["pos_weight"], 3 / 9)

    def test_generator_balanced_batches_require_fake_generator_ids(self):
        examples = [
            example("real", 0),
            example("fake", 0, None),
        ]

        with self.assertRaises(ValueError):
            BalancedTrainBatchSampler(
                examples=examples,
                batch_size=2,
                mode="generator_balanced_batches",
                seed=0,
            )

    def test_fake_generator_cap_keeps_reals_and_caps_large_generators(self):
        examples = [
            *[example("real", index) for index in range(3)],
            *[example("fake", index, "gen_a") for index in range(10)],
            *[example("fake", index, "gen_b") for index in range(4)],
            *[example("fake", index, "gen_c") for index in range(2)],
        ]

        capped, summary = cap_fake_generators(
            examples,
            cap_multiplier=1.0,
            seed=0,
        )

        self.assertEqual(sum(item.class_name == "real" for item in capped), 3)
        self.assertEqual(fake_generator_counts(capped), {"gen_a": 4, "gen_b": 4, "gen_c": 2})
        self.assertEqual(summary["cap"], 4)
        self.assertEqual(summary["dropped_fake_count"], 6)

    def test_fake_generator_cap_exemptions_keep_selected_generators_uncapped(self):
        examples = [
            *[example("real", index) for index in range(3)],
            *[example("fake", index, "gen_a") for index in range(10)],
            *[example("fake", index, "gen_b") for index in range(4)],
            *[example("fake", index, "gen_c") for index in range(2)],
        ]

        capped, summary = cap_fake_generators(
            examples,
            cap_multiplier=1.0,
            seed=0,
            cap_exemptions=("gen_a",),
        )

        self.assertEqual(fake_generator_counts(capped), {"gen_a": 10, "gen_b": 4, "gen_c": 2})
        self.assertEqual(summary["exemptions"], ["gen_a"])
        self.assertEqual(summary["dropped_fake_count"], 0)

    def test_fake_generator_cap_disabled_returns_original_counts(self):
        examples = [
            example("real", 0),
            example("fake", 0, "gen_a"),
            example("fake", 1, "gen_a"),
        ]

        capped, summary = cap_fake_generators(
            examples,
            cap_multiplier=None,
            seed=0,
        )

        self.assertEqual(capped, examples)
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["dropped_fake_count"], 0)

    def test_fake_generator_loss_weights_parse_and_apply_to_fake_only(self):
        weights = parse_fake_generator_loss_weights(["dlc=1.5", "visomaster=2.0"])
        batch = {
            "class_name": ["real", "fake", "fake", "fake"],
            "generator_id": ["real", "dlc", "ltx2", "visomaster"],
        }

        sample_weights = fake_generator_sample_weights(
            batch,
            weights,
            device=torch.device("cpu"),
        )

        self.assertEqual(weights, {"dlc": 1.5, "visomaster": 2.0})
        self.assertEqual(sample_weights.view(-1).tolist(), [1.0, 1.5, 1.0, 2.0])

    def test_prediction_generator_metrics_group_by_split_class_and_generator(self):
        rows = [
            PredictionRow("/real/a.mp4", "real", "real", 0, 0.1, 0, "val"),
            PredictionRow("/fake/dlc/a.mp4", "fake", "dlc", 1, 0.8, 1, "val"),
            PredictionRow("/fake/dlc/b.mp4", "fake", "dlc", 1, 0.2, 0, "val"),
        ]

        metrics = prediction_generator_metrics(rows)
        dlc = next(row for row in metrics if row["generator_id"] == "dlc")

        self.assertEqual(dlc["count"], 2)
        self.assertEqual(dlc["false_negative"], 1)
        self.assertEqual(dlc["true_positive"], 1)
        self.assertEqual(dlc["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
