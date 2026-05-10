from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dataset import VideoExample, load_video_clip_for_example
from feature_cache import (
    CachedFeatureDataset,
    ShardedCachedFeatureDataset,
    ShardedFeatureBatchSampler,
    ShardedFeatureRef,
    build_feature_cache_specs,
    cache_example_key,
    collate_cached_feature_batch,
    feature_cache_item_exists,
    feature_cache_item_path,
    load_feature_cache_item,
    write_feature_cache_item,
    write_feature_cache_manifest,
)


def build_example(
    path: Path,
    class_name: str,
    metadata_filename: str,
    label: int | None = None,
    generator_id: str | None = None,
) -> VideoExample:
    resolved_label = (0 if class_name == "real" else 1) if label is None else label
    return VideoExample(
        path=path,
        label=resolved_label,
        class_name=class_name,
        source_id=path.stem,
        split="train",
        metadata_filename=metadata_filename,
        identity_id=generator_id,
        generator_id=generator_id,
    )


class MinimalFeatureCacheTest(unittest.TestCase):
    def test_real_cache_path_uses_class_and_meta_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "videos" / "real" / "clip.mp4"
            example = build_example(video, "real", "clip.mp4", generator_id="real")
            spec = build_feature_cache_specs({"frames": {"default": 16}, "rgb": {}}, ("rgb",))[
                "rgb"
            ]

            path = feature_cache_item_path(root / "cache", example, spec, dataset_root=root)

            self.assertEqual(path, root / "cache" / "rgb" / "frames_16" / "real" / "clip.mp4.pt")

    def test_schema_v2_sharded_dataset_uses_explicit_key_locations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_dir = root / "shards"
            shard_dir.mkdir()
            real = build_example(root / "videos" / "real" / "r1.mp4", "real", "r1.mp4")
            fake = build_example(root / "videos" / "fake" / "gen" / "f1.mp4", "fake", "gen/f1.mp4")
            payload = {
                "version": 2,
                "examples": [
                    {
                        "class_name": "fake",
                        "metadata_filename": "gen/f1.mp4",
                    },
                    {
                        "class_name": "real",
                        "metadata_filename": "r1.mp4",
                    },
                ],
                "features": {"rgb_features": torch.tensor([[10.0], [20.0]])},
            }
            torch.save(payload, shard_dir / "shard_000000.pt")
            index = {
                "schema_version": 2,
                "shard_size": 2,
                "example_count": 2,
                "example_locations": {
                    cache_example_key(fake, root): {
                        "shard_index": 0,
                        "row_index": 0,
                        "class_name": "fake",
                        "filename": "gen/f1.mp4",
                        "label": 1,
                    },
                    cache_example_key(real, root): {
                        "shard_index": 0,
                        "row_index": 1,
                        "class_name": "real",
                        "filename": "r1.mp4",
                        "label": 0,
                    },
                },
                "shards": [{"path": "shards/shard_000000.pt", "example_count": 2}],
            }
            (root / "index.json").write_text(__import__("json").dumps(index), encoding="utf-8")

            dataset = ShardedCachedFeatureDataset(
                root,
                all_examples=[real, fake],
                selected_examples=[real, fake],
                dataset_root=root,
            )

            self.assertTrue(torch.equal(dataset[0]["rgb_features"], torch.tensor([20.0])))
            self.assertTrue(torch.equal(dataset[1]["rgb_features"], torch.tensor([10.0])))

    def test_legacy_sharded_dataset_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example = build_example(root / "videos" / "real" / "r1.mp4", "real", "r1.mp4")
            index = {
                "version": 1,
                "shard_size": 1,
                "example_locations": {
                    cache_example_key(example, root): {"shard_index": 0, "row_index": 0}
                },
                "shards": [{"path": "shard_000000.pt", "example_count": 1}],
            }
            (root / "index.json").write_text(__import__("json").dumps(index), encoding="utf-8")

            with self.assertRaises(ValueError):
                ShardedCachedFeatureDataset(
                    root,
                    all_examples=[example],
                    selected_examples=[example],
                    dataset_root=root,
                )

    def test_mixed_shard_sampler_rejects_class_blocked_training_groups(self):
        real = build_example(Path("/data/real/r.mp4"), "real", "r.mp4")
        fake = build_example(Path("/data/fake/gen/f.mp4"), "fake", "gen/f.mp4")
        refs = [
            *[ShardedFeatureRef(fake, shard_index=0, row_index=index) for index in range(8)],
            *[ShardedFeatureRef(real, shard_index=1, row_index=index) for index in range(8)],
        ]

        with self.assertRaises(ValueError):
            ShardedFeatureBatchSampler(
                refs=refs,
                batch_size=4,
                shuffle=True,
                batch_strategy="mixed_shard_local",
            )

    def test_mixed_shard_sampler_interleaves_labels_within_shard(self):
        real = build_example(Path("/data/real/r.mp4"), "real", "r.mp4")
        fake = build_example(Path("/data/fake/gen/f.mp4"), "fake", "gen/f.mp4")
        refs = [
            *[ShardedFeatureRef(real, shard_index=0, row_index=index) for index in range(4)],
            *[ShardedFeatureRef(fake, shard_index=0, row_index=index + 4) for index in range(4)],
        ]

        sampler = ShardedFeatureBatchSampler(
            refs=refs,
            batch_size=4,
            shuffle=False,
            batch_strategy="mixed_shard_local",
        )

        batches = list(sampler)
        self.assertTrue(batches)
        for batch in batches:
            labels = {refs[index].example.label for index in batch}
            self.assertEqual(labels, {0, 1})

    def test_fake_cache_path_preserves_generator_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "videos" / "fake" / "liveavatar" / "men____000001.mp4"
            example = build_example(
                video,
                "fake",
                "liveavatar/men____000001.mp4",
                generator_id="liveavatar",
            )
            spec = build_feature_cache_specs({"frames": {"default": 16}, "rgb": {}}, ("rgb",))[
                "rgb"
            ]

            path = feature_cache_item_path(root / "cache", example, spec, dataset_root=root)

            self.assertEqual(
                path,
                root
                / "cache"
                / "rgb"
                / "frames_16"
                / "fake"
                / "liveavatar"
                / "men____000001.mp4.pt",
            )

    def test_duplicate_real_fake_basenames_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec = build_feature_cache_specs({"frames": {"default": 16}, "rgb": {}}, ("rgb",))[
                "rgb"
            ]
            real = build_example(
                root / "videos" / "real" / "same.mp4",
                "real",
                "same.mp4",
                generator_id="real",
            )
            fake = build_example(
                root / "videos" / "fake" / "dlc" / "same.mp4",
                "fake",
                "dlc/same.mp4",
                generator_id="dlc",
            )

            real_path = feature_cache_item_path(root / "cache", real, spec, dataset_root=root)
            fake_path = feature_cache_item_path(root / "cache", fake, spec, dataset_root=root)

            self.assertNotEqual(real_path, fake_path)
            self.assertEqual(real_path.name, "same.mp4.pt")
            self.assertEqual(fake_path.name, "same.mp4.pt")

    def test_write_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example = build_example(
                root / "videos" / "fake" / "liveavatar" / "men____000001.mp4",
                "fake",
                "liveavatar/men____000001.mp4",
                generator_id="liveavatar",
            )
            spec = build_feature_cache_specs({"frames": {"default": 16}, "rgb": {}}, ("rgb",))[
                "rgb"
            ]

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

    def test_rppg_cache_variant_invalidates_old_payload_but_not_rgb_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            example = build_example(
                root / "videos" / "real" / "clip.mp4",
                "real",
                "clip.mp4",
                generator_id="real",
            )
            specs = build_feature_cache_specs(
                {
                    "frames": {"default": 16},
                    "image_size": 224,
                    "rgb": {},
                    "rppg": {"frames": 128, "image_size": 128},
                },
                ("rgb", "rppg"),
            )
            write_feature_cache_item(
                cache_dir,
                example,
                specs["rgb"],
                {"rgb_features": torch.ones(2, 3)},
                dataset_root=root,
            )
            rppg_path = feature_cache_item_path(
                cache_dir, example, specs["rppg"], dataset_root=root
            )
            rppg_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 3,
                    "class_name": "real",
                    "filename": "clip.mp4",
                    "modality": "rppg",
                    "frame_count": 128,
                    "features": {
                        "rppg_features": torch.ones(128, 8),
                        "rppg_waveform": torch.ones(128),
                    },
                },
                rppg_path,
            )

            self.assertTrue(feature_cache_item_exists(cache_dir, example, specs["rgb"], root))
            self.assertFalse(feature_cache_item_exists(cache_dir, example, specs["rppg"], root))

    def test_cropped_rgb_cache_variant_does_not_reuse_full_frame_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            example = build_example(
                root / "videos" / "real" / "clip.mp4",
                "real",
                "clip.mp4",
                generator_id="real",
            )
            legacy_spec = build_feature_cache_specs(
                {"frames": {"default": 16}, "rgb": {}},
                ("rgb",),
            )["rgb"]
            cropped_spec = build_feature_cache_specs(
                {
                    "frames": {"default": 16},
                    "image_size": 224,
                    "face_crop": {"enabled": True},
                    "rgb": {},
                },
                ("rgb",),
            )["rgb"]

            write_feature_cache_item(
                cache_dir,
                example,
                legacy_spec,
                {"rgb_features": torch.ones(2, 3)},
                dataset_root=root,
            )

            self.assertIsNone(legacy_spec.cache_variant)
            self.assertIsNotNone(cropped_spec.cache_variant)
            self.assertTrue(feature_cache_item_exists(cache_dir, example, legacy_spec, root))
            self.assertFalse(feature_cache_item_exists(cache_dir, example, cropped_spec, root))

    def test_disabled_face_crop_keeps_legacy_rgb_cache_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example = build_example(
                root / "videos" / "real" / "clip.mp4",
                "real",
                "clip.mp4",
                generator_id="real",
            )
            legacy_spec = build_feature_cache_specs(
                {"frames": {"default": 16}, "rgb": {}},
                ("rgb",),
            )["rgb"]
            disabled_spec = build_feature_cache_specs(
                {
                    "frames": {"default": 16},
                    "face_crop": {"enabled": True},
                    "rgb": {"face_crop": {"enabled": False}},
                },
                ("rgb",),
            )["rgb"]

            self.assertIsNone(disabled_spec.cache_variant)
            self.assertEqual(
                feature_cache_item_path(root / "cache", example, legacy_spec, dataset_root=root),
                feature_cache_item_path(root / "cache", example, disabled_spec, dataset_root=root),
            )

    def test_rich_eye_gaze_cache_variant_does_not_reuse_legacy_eye_gaze_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example = build_example(
                root / "videos" / "real" / "clip.mp4",
                "real",
                "clip.mp4",
                generator_id="real",
            )
            legacy_spec = build_feature_cache_specs(
                {
                    "frames": {"default": 32},
                    "image_size": 224,
                    "face_crop": {"enabled": True},
                    "eye_gaze": {},
                },
                ("eye_gaze",),
            )["eye_gaze"]
            rich_spec = build_feature_cache_specs(
                {
                    "frames": {"default": 32},
                    "image_size": 224,
                    "face_crop": {"enabled": True},
                    "eye_gaze": {"feature_variant": "rich_v1", "feature_dim": 32},
                },
                ("eye_gaze",),
            )["eye_gaze"]

            self.assertIsNotNone(legacy_spec.cache_variant)
            self.assertIsNotNone(rich_spec.cache_variant)
            self.assertIn("eye_gaze_rich_v1", rich_spec.cache_variant)
            self.assertNotEqual(legacy_spec.cache_variant, rich_spec.cache_variant)
            self.assertNotEqual(
                feature_cache_item_path(root / "cache", example, legacy_spec, dataset_root=root),
                feature_cache_item_path(root / "cache", example, rich_spec, dataset_root=root),
            )

    def test_rppg_cache_roundtrip_requires_signal_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            example = build_example(
                root / "videos" / "real" / "clip.mp4",
                "real",
                "clip.mp4",
                generator_id="real",
            )
            spec = build_feature_cache_specs(
                {"frames": {"default": 16}, "rppg": {"frames": 128, "image_size": 128}},
                ("rppg",),
            )["rppg"]
            item = {
                "rppg_features": torch.ones(128, 8),
                "rppg_waveform": torch.zeros(128),
                "rppg_signal_features": torch.arange(6, dtype=torch.float32),
            }

            write_feature_cache_item(cache_dir, example, spec, item, dataset_root=root)
            loaded = load_feature_cache_item(cache_dir, example, spec, dataset_root=root)

            self.assertIsNotNone(loaded)
            self.assertTrue(
                torch.equal(loaded["rppg_signal_features"], item["rppg_signal_features"])
            )

    def test_manifest_includes_generator_status_and_cache_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            spec = build_feature_cache_specs({"frames": {"default": 16}, "rgb": {}}, ("rgb",))[
                "rgb"
            ]
            cached = build_example(
                root / "videos" / "fake" / "liveavatar" / "cached.mp4",
                "fake",
                "liveavatar/cached.mp4",
                generator_id="liveavatar",
            )
            failed = build_example(
                root / "videos" / "fake" / "dlc" / "failed.mp4",
                "fake",
                "dlc/failed.mp4",
                generator_id="dlc",
            )
            write_feature_cache_item(
                cache_dir,
                cached,
                spec,
                {"rgb_features": torch.ones(2, 3)},
                dataset_root=root,
            )

            manifest = write_feature_cache_manifest(
                cache_dir,
                [cached, failed],
                spec,
                dataset_root=root,
                errors_by_path={str(failed.path): "decode failed"},
            )

            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["status"], "cached")
            self.assertEqual(rows[0]["generator_id"], "liveavatar")
            self.assertTrue(rows[0]["cache_path"].endswith("fake/liveavatar/cached.mp4.pt"))
            self.assertEqual(rows[1]["status"], "failed")
            self.assertEqual(rows[1]["error"], "decode failed")

    def test_cached_dataset_strict_missing_and_collate_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            specs = build_feature_cache_specs(
                {"frames": {"default": 16}, "rgb": {}, "fau": {}},
                ("rgb", "fau"),
            )
            example = build_example(
                root / "videos" / "fake" / "liveavatar" / "clip.mp4",
                "fake",
                "liveavatar/clip.mp4",
                generator_id="liveavatar",
            )

            missing_dataset = CachedFeatureDataset(
                [example],
                cache_dir,
                specs,
                ("rgb",),
                dataset_root=root,
            )
            with self.assertRaises(FileNotFoundError):
                _ = missing_dataset[0]

            write_feature_cache_item(
                cache_dir,
                example,
                specs["rgb"],
                {"rgb_features": torch.ones(2, 3)},
                dataset_root=root,
            )
            dataset = CachedFeatureDataset(
                [example],
                cache_dir,
                specs,
                ("rgb",),
                dataset_root=root,
            )
            batch = collate_cached_feature_batch([dataset[0], dataset[0]])

            self.assertEqual(tuple(batch["rgb_features"].shape), (2, 2, 3))
            self.assertEqual(batch["metadata_filename"], ["liveavatar/clip.mp4"] * 2)

    def test_decoded_clip_cache_reuses_saved_clip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            example = build_example(
                root / "videos" / "fake" / "liveavatar" / "clip.mp4",
                "fake",
                "liveavatar/clip.mp4",
                generator_id="liveavatar",
            )
            clip = {
                "video": torch.ones(3, 4, 8, 8),
                "video_rgb_frames": [np.ones((8, 8, 3), dtype=np.uint8)],
            }

            with patch("dataset.load_video_clip", return_value=clip) as load_video:
                first = load_video_clip_for_example(
                    example,
                    num_frames=4,
                    image_size=8,
                    clip_cache_dir=root / "cache" / "_clips",
                    dataset_root=root,
                )
                second = load_video_clip_for_example(
                    example,
                    num_frames=4,
                    image_size=8,
                    clip_cache_dir=root / "cache" / "_clips",
                    dataset_root=root,
                )

            self.assertEqual(load_video.call_count, 1)
            self.assertTrue(torch.equal(first["video"], second["video"]))


if __name__ == "__main__":
    unittest.main()
