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
    build_feature_cache_specs,
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
