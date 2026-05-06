from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dataset import VideoExample
from feature_cache import (
    CachedFeatureDataset,
    build_feature_cache_spec,
    build_feature_cache_specs,
    collate_cached_feature_batch,
    feature_cache_item_exists,
    feature_cache_item_path,
    feature_cache_spec_id,
    load_feature_cache_item,
    write_feature_cache_item,
)
from scripts.run_iterative_cached_ablation import (
    BinaryMetrics,
    EpochEvalResult,
    EpochTrainResult,
    binary_metrics_from_counts,
    build_balanced_train_order,
    checkpoint_metric_value,
    is_metric_improvement,
    metrics_row_values,
    rebalance_eval_examples,
    resolve_round_targets,
    resolve_warm_start_checkpoint,
    select_balanced_subset,
    split_balanced_total_examples,
)


def build_example(
    path: Path,
    label: int = 1,
    class_name: str = "fake",
    source_id: str = "clipA",
    split: str = "train",
    identity_id: str | None = "generatorA",
) -> VideoExample:
    return VideoExample(
        path=path,
        label=label,
        class_name=class_name,
        source_id=source_id,
        split=split,
        identity_id=identity_id,
    )


class FeatureCacheTest(unittest.TestCase):
    def test_spec_ignores_fusion_head_and_slot_count_changes(self):
        config = {
            "image_size": 224,
            "frames": {"default": 16},
            "fusion": {"num_layers": 2},
            "head": {"type": "cls_mlp"},
            "rgb": {"checkpoint_path": "/tmp/rgb.pt", "slot_count": 8},
        }
        changed = {
            **config,
            "fusion": {"num_layers": 8},
            "head": {"type": "attention_mil"},
            "rgb": {"checkpoint_path": "/tmp/rgb.pt", "slot_count": 16},
        }

        spec = build_feature_cache_spec(config, "rgb")
        changed_spec = build_feature_cache_spec(changed, "rgb")

        self.assertEqual(feature_cache_spec_id(spec), feature_cache_spec_id(changed_spec))

    def test_spec_changes_when_extractor_config_frame_or_modality_changes(self):
        config = {
            "image_size": 224,
            "frames": {"default": 16},
            "rgb": {"checkpoint_path": "/tmp/rgb-a.pt"},
            "fau": {"checkpoint_path": "/tmp/fau.pt"},
        }

        rgb_spec = build_feature_cache_spec(config, "rgb")
        checkpoint_spec = build_feature_cache_spec(
            {**config, "rgb": {"checkpoint_path": "/tmp/rgb-b.pt"}},
            "rgb",
        )
        frame_spec = build_feature_cache_spec({**config, "frames": {"default": 32}}, "rgb")
        modality_spec = build_feature_cache_spec(config, "fau")

        self.assertNotEqual(feature_cache_spec_id(rgb_spec), feature_cache_spec_id(checkpoint_spec))
        self.assertNotEqual(feature_cache_spec_id(rgb_spec), feature_cache_spec_id(frame_spec))
        self.assertNotEqual(feature_cache_spec_id(rgb_spec), feature_cache_spec_id(modality_spec))

    def test_spectral_modalities_are_cacheable(self):
        config = {
            "image_size": 224,
            "frames": {"default": 16},
            "fft": {"num_bins": 32},
            "stft": {"n_fft": 8},
        }

        specs = build_feature_cache_specs(config, ("fft", "stft"))

        self.assertEqual(specs["fft"].modality, "fft")
        self.assertEqual(specs["stft"].modality, "stft")

    def test_cache_path_is_stable_and_collision_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "a" / "same.mp4"
            second = root / "b" / "same.mp4"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            spec = build_feature_cache_spec({"frames": {"default": 4}, "rgb": {}}, "rgb")

            first_path = feature_cache_item_path(
                root / "cache", build_example(first), spec, dataset_root=root
            )
            second_path = feature_cache_item_path(
                root / "cache", build_example(second), spec, dataset_root=root
            )

            self.assertEqual(
                first_path,
                feature_cache_item_path(
                    root / "cache", build_example(first), spec, dataset_root=root
                ),
            )
            self.assertNotEqual(first_path, second_path)
            self.assertEqual(
                first_path.parent,
                root / "cache" / "rgb" / "frames_4" / feature_cache_spec_id(spec),
            )
            self.assertTrue(first_path.name.startswith("same-"))

    def test_cache_path_ignores_dataset_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "fake" / "clip.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            spec = build_feature_cache_spec({"frames": {"default": 4}, "rgb": {}}, "rgb")

            base_path = feature_cache_item_path(
                root / "cache",
                build_example(video, label=1, class_name="fake", split="train"),
                spec,
                dataset_root=root,
            )
            changed_path = feature_cache_item_path(
                root / "cache",
                build_example(video, label=0, class_name="real", split="val"),
                spec,
                dataset_root=root,
            )

            self.assertEqual(base_path, changed_path)

    def test_write_load_and_stale_invalidation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "fake" / "clip.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            example = build_example(video)
            spec = build_feature_cache_spec({"frames": {"default": 4}, "rgb": {}}, "rgb")

            write_feature_cache_item(
                root / "cache",
                example,
                spec,
                {"rgb_features": torch.ones(2, 3)},
                dataset_root=root,
            )
            payload = torch.load(
                feature_cache_item_path(root / "cache", example, spec, dataset_root=root),
                map_location="cpu",
                weights_only=False,
            )
            loaded = load_feature_cache_item(root / "cache", example, spec, dataset_root=root)
            changed_metadata_loaded = load_feature_cache_item(
                root / "cache",
                build_example(video, label=0, class_name="real", split="val"),
                spec,
                dataset_root=root,
            )

            self.assertIn("video", payload)
            self.assertNotIn("example", payload)
            self.assertEqual(payload["version"], 2)
            self.assertEqual(payload["video"]["relative_path"], "fake/clip.mp4")
            self.assertNotIn("label", payload["video"])
            self.assertNotIn("split", payload["video"])
            self.assertIsNotNone(loaded)
            self.assertTrue(torch.equal(loaded["rgb_features"], torch.ones(2, 3)))
            self.assertIsNotNone(changed_metadata_loaded)
            self.assertTrue(feature_cache_item_exists(root / "cache", example, spec, root))

            video.write_bytes(b"changed")

            self.assertIsNone(load_feature_cache_item(root / "cache", example, spec, root))

    def test_cache_exists_uses_metadata_sidecar_without_loading_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "fake" / "clip.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            example = build_example(video)
            spec = build_feature_cache_spec({"frames": {"default": 4}, "rgb": {}}, "rgb")
            cache_path = write_feature_cache_item(
                root / "cache",
                example,
                spec,
                {"rgb_features": torch.ones(2, 3)},
                dataset_root=root,
            )

            self.assertTrue(cache_path.with_suffix(".json").exists())
            with patch("feature_cache.torch.load", side_effect=AssertionError("loaded tensor")):
                self.assertTrue(feature_cache_item_exists(root / "cache", example, spec, root))

    def test_cached_feature_dataset_strict_loads_and_collates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "fake" / "clip.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            example = build_example(video)
            specs = build_feature_cache_specs({"frames": {"default": 4}, "rgb": {}}, ("rgb",))
            write_feature_cache_item(
                root / "cache",
                example,
                specs["rgb"],
                {"rgb_features": torch.ones(2, 3)},
                dataset_root=root,
            )

            dataset = CachedFeatureDataset(
                [example],
                root / "cache",
                specs,
                ("rgb",),
                dataset_root=root,
            )
            batch = collate_cached_feature_batch([dataset[0], dataset[0]])

            self.assertEqual(tuple(batch["rgb_features"].shape), (2, 2, 3))
            self.assertEqual(tuple(batch["label"].shape), (2, 1))
            self.assertEqual(batch["split"], ["train", "train"])

    def test_cached_feature_dataset_strict_errors_on_missing_modality(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "fake" / "clip.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            specs = build_feature_cache_specs(
                {"frames": {"default": 4}, "rgb": {}, "fau": {}},
                ("rgb", "fau"),
            )

            dataset = CachedFeatureDataset(
                [build_example(video)],
                root / "cache",
                specs,
                ("rgb", "fau"),
                dataset_root=root,
            )

            with self.assertRaises(FileNotFoundError):
                _ = dataset[0]

    def test_iterative_selectors_are_nested_and_balanced(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples: list[VideoExample] = []
            for index in range(20):
                path = root / "real" / f"real{index}.mp4"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"video")
                examples.append(
                    build_example(
                        path,
                        label=0,
                        class_name="real",
                        source_id=f"r{index}",
                        identity_id=None,
                    )
                )
            for index in range(20):
                generator = f"g{index % 4}"
                path = root / "fake" / generator / f"fake{index}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
                examples.append(build_example(path, source_id=f"f{index}", identity_id=generator))

            ordered = build_balanced_train_order(examples, seed=0)
            first = ordered[:10]
            second = ordered[:20]
            selected = select_balanced_subset(examples, target_count=12, seed=0)
            limited = select_balanced_subset(examples[:23], target_count=20, seed=0)

            self.assertEqual(first, second[:10])
            self.assertEqual(len(selected), 12)
            self.assertEqual(len(limited), 6)
            self.assertEqual(sum(item.class_name == "real" for item in first), 5)
            self.assertEqual(sum(item.class_name == "fake" for item in first), 5)
            self.assertEqual(sum(item.class_name == "real" for item in limited), 3)
            self.assertEqual(sum(item.class_name == "fake" for item in limited), 3)
            self.assertEqual(resolve_round_targets(1200, "tiny"), [40, 100, 200, 500, 1200])
            self.assertEqual(
                resolve_round_targets(1200, "tiny", explicit_targets=(1000, 200, 500)),
                [200, 500, 1000],
            )

            with self.assertRaises(ValueError):
                resolve_round_targets(100, "tiny", explicit_targets=(200,))

    def test_iterative_selectors_prefer_cached_scores_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples: list[VideoExample] = []
            for index in range(3):
                examples.append(
                    build_example(
                        root / f"real-{index}.mp4",
                        label=0,
                        class_name="real",
                        source_id=f"real-{index}",
                        identity_id=None,
                    )
                )
                examples.append(
                    build_example(
                        root / f"fake-{index}.mp4",
                        label=1,
                        class_name="fake",
                        source_id=f"fake-{index}",
                        identity_id=f"gen-{index}",
                    )
                )
            scores = {
                str(root / "real-0.mp4"): 0,
                str(root / "real-1.mp4"): 7,
                str(root / "real-2.mp4"): 3,
                str(root / "fake-0.mp4"): 0,
                str(root / "fake-1.mp4"): 7,
                str(root / "fake-2.mp4"): 3,
            }

            selected = select_balanced_subset(
                examples,
                target_count=4,
                seed=0,
                cache_score_by_path=scores,
            )
            ordered = build_balanced_train_order(
                examples,
                seed=0,
                cache_score_by_path=scores,
            )

            self.assertEqual(
                {str(example.path) for example in selected},
                {
                    str(root / "real-1.mp4"),
                    str(root / "real-2.mp4"),
                    str(root / "fake-1.mp4"),
                    str(root / "fake-2.mp4"),
                },
            )
            self.assertEqual([example.class_name for example in selected], ["real", "fake"] * 2)
            self.assertGreaterEqual(scores[str(ordered[0].path)], 3)
            self.assertGreaterEqual(scores[str(ordered[1].path)], 3)

    def test_balanced_total_selection_forms_own_splits_from_cached_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples: list[VideoExample] = []
            for index in range(10):
                examples.append(
                    build_example(
                        root / f"real-{index}.mp4",
                        label=0,
                        class_name="real",
                        source_id=f"real-{index}",
                        split="original",
                        identity_id=None,
                    )
                )
                examples.append(
                    build_example(
                        root / f"fake-{index}.mp4",
                        label=1,
                        class_name="fake",
                        source_id=f"fake-{index}",
                        split="original",
                        identity_id=f"gen-{index % 3}",
                    )
                )
            scores = {str(root / f"real-{index}.mp4"): 7 if index < 5 else 0 for index in range(10)}
            scores.update(
                {str(root / f"fake-{index}.mp4"): 7 if index < 5 else 0 for index in range(10)}
            )

            train, val, test = split_balanced_total_examples(
                examples,
                balanced_total=10,
                train_ratio=0.6,
                val_ratio=0.2,
                seed=0,
                cache_score_by_path=scores,
            )

            selected = [*train, *val, *test]
            self.assertEqual(len(selected), 10)
            self.assertEqual(len(train), 6)
            self.assertEqual(len(val), 2)
            self.assertEqual(len(test), 2)
            self.assertEqual({example.split for example in train}, {"train"})
            self.assertEqual({example.split for example in val}, {"val"})
            self.assertEqual({example.split for example in test}, {"test"})
            self.assertEqual(sum(example.class_name == "real" for example in selected), 5)
            self.assertEqual(sum(example.class_name == "fake" for example in selected), 5)
            self.assertTrue(all(scores[str(example.path)] == 7 for example in selected))

    def test_warm_start_checkpoint_requires_enabled_previous_round(self):
        previous = {"best_checkpoint": "/tmp/run/train_00200/rgb/best.pt"}

        self.assertIsNone(resolve_warm_start_checkpoint(None, enabled=True))
        self.assertIsNone(resolve_warm_start_checkpoint(previous, enabled=False))
        self.assertEqual(
            resolve_warm_start_checkpoint(previous, enabled=True),
            Path("/tmp/run/train_00200/rgb/best.pt"),
        )

        with self.assertRaises(ValueError):
            resolve_warm_start_checkpoint({}, enabled=True)

    def test_checkpoint_metric_prefers_validation_by_default(self):
        train = EpochTrainResult(
            loss=0.2,
            accuracy=0.9,
            elapsed_seconds=1.0,
            metrics=BinaryMetrics(balanced_accuracy=0.85, f1=0.8),
        )
        val = EpochEvalResult(
            loss=0.6,
            accuracy=0.7,
            elapsed_seconds=1.0,
            metrics=BinaryMetrics(balanced_accuracy=0.65, f1=0.6),
        )

        self.assertEqual(checkpoint_metric_value("val_accuracy", train, val), 0.7)
        self.assertEqual(checkpoint_metric_value("val_balanced_accuracy", train, val), 0.65)
        self.assertEqual(checkpoint_metric_value("val_f1", train, val), 0.6)
        self.assertEqual(checkpoint_metric_value("train_balanced_accuracy", train, val), 0.85)
        self.assertEqual(checkpoint_metric_value("train_f1", train, val), 0.8)
        self.assertTrue(is_metric_improvement("val_accuracy", 0.71, 0.7, 0.0))
        self.assertFalse(is_metric_improvement("val_accuracy", 0.70, 0.7, 0.0))
        self.assertTrue(is_metric_improvement("val_loss", 0.59, 0.6, 0.0))
        self.assertFalse(is_metric_improvement("val_loss", 0.61, 0.6, 0.0))

    def test_binary_metrics_from_counts_reports_false_positive_and_false_negative(self):
        metrics = binary_metrics_from_counts(tp=3, tn=5, fp=2, fn=1)
        row = metrics_row_values("val", metrics)

        self.assertEqual(metrics.false_positive, 2)
        self.assertEqual(metrics.false_negative, 1)
        self.assertAlmostEqual(metrics.precision, 0.6)
        self.assertAlmostEqual(metrics.recall, 0.75)
        self.assertAlmostEqual(metrics.specificity, 5 / 7)
        self.assertEqual(row["val_false_positive"], 2)
        self.assertEqual(row["val_false_negative"], 1)
        self.assertEqual(row["val_f1"], f"{metrics.f1:.8f}")

    def test_rebalance_eval_examples_removes_class_skew_after_cache_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            examples: list[VideoExample] = []
            for index in range(2):
                path = root / "real" / f"real{index}.mp4"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"video")
                examples.append(build_example(path, label=0, class_name="real", identity_id=None))
            for index in range(5):
                path = root / "fake" / f"fake{index}.mp4"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"video")
                examples.append(build_example(path, label=1, class_name="fake"))

            balanced = rebalance_eval_examples(examples, target_count=7, seed=0, label="test")

            self.assertEqual(len(balanced), 4)
            self.assertEqual(sum(item.class_name == "real" for item in balanced), 2)
            self.assertEqual(sum(item.class_name == "fake" for item in balanced), 2)


if __name__ == "__main__":
    unittest.main()
