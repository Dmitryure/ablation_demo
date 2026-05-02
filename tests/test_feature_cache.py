from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
    build_balanced_train_order,
    resolve_round_targets,
    select_balanced_subset,
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
            loaded = load_feature_cache_item(root / "cache", example, spec, dataset_root=root)

            self.assertIsNotNone(loaded)
            self.assertTrue(torch.equal(loaded["rgb_features"], torch.ones(2, 3)))
            self.assertTrue(feature_cache_item_exists(root / "cache", example, spec, root))

            video.write_bytes(b"changed")

            self.assertIsNone(load_feature_cache_item(root / "cache", example, spec, root))

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


if __name__ == "__main__":
    unittest.main()
