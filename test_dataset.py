from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from dataset import (
    LabeledVideoDataset,
    VideoExample,
    build_real_fake_examples,
    collate_labeled_video_batch,
    load_dataset_manifest,
    summarize_examples,
    write_dataset_manifest,
)
from extractors.eye_gaze import EYE_GAZE_COLUMNS, EyeGazeExtractor
from extractors.rgb import RGBExtractor


def fake_eye_gaze_detector(_: np.ndarray) -> dict[str, float]:
    return {name: float(index) / 10.0 for index, name in enumerate(EYE_GAZE_COLUMNS, start=1)}


class DummyRGBEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.ones(batch, 8, 12, device=x.device)


class DatasetTest(unittest.TestCase):
    def test_build_examples_write_and_reload_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_dir = root / "real"
            fake_dir = root / "fake"
            (fake_dir / "f1").mkdir(parents=True)
            real_dir.mkdir(parents=True)

            for name in (
                "abcdEFGhijk_00_00_01-00_00_31part1_f.mp4",
                "abcdEFGhijk_00_00_31-00_01_01part2_f.mp4",
                "zzzzEFGhijk_00_00_01-00_00_31part1_m.mp4",
                "qqqqEFGhijk_00_00_01-00_00_31part1_f.mp4",
            ):
                (real_dir / name).touch()
            for name in (
                "clipA_clip_1_0_20_swapped.mp4",
                "clipA_clip_2_20_20_swapped.mp4",
                "clipB_clip_1_0_20_swapped.mp4",
                "clipC_clip_1_0_20_swapped.mp4",
            ):
                (fake_dir / "f1" / name).touch()

            examples = build_real_fake_examples(real_dir=real_dir, fake_dir=fake_dir, seed=0)
            manifest_path = write_dataset_manifest(examples, root / "manifest.csv")
            loaded = load_dataset_manifest(manifest_path)
            summary = summarize_examples(loaded)

            self.assertEqual(len(loaded), 8)
            self.assertEqual(sum(split["total"] for split in summary.values()), 8)
            self.assertEqual({example.class_name for example in loaded}, {"real", "fake"})
            self.assertEqual({example.split for example in loaded}, {"train", "val", "test"})

    def test_collate_labeled_video_batch_stacks_raw_video_and_labels(self):
        frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        batch = collate_labeled_video_batch(
            [
                {
                    "video": torch.randn(3, 3, 8, 8),
                    "video_rgb_frames": frames,
                    "label": torch.tensor([0.0]),
                    "path": "a.mp4",
                    "source_id": "a",
                    "split": "train",
                    "class_name": "real",
                    "identity_id": None,
                },
                {
                    "video": torch.randn(3, 3, 8, 8),
                    "video_rgb_frames": frames,
                    "label": torch.tensor([1.0]),
                    "path": "b.mp4",
                    "source_id": "b",
                    "split": "train",
                    "class_name": "fake",
                    "identity_id": "f1",
                },
            ]
        )

        self.assertEqual(tuple(batch["video"].shape), (2, 3, 3, 8, 8))
        self.assertEqual(tuple(batch["label"].shape), (2, 1))
        self.assertEqual(len(batch["video_rgb_frames"]), 2)

    def test_rgb_extractor_supports_batched_clip_sequences(self):
        extractor = RGBExtractor(DummyRGBEncoder(), image_size=8)
        clip = [np.zeros((6, 6, 3), dtype=np.uint8) for _ in range(4)]

        output = extractor.extract({"video_rgb_frames": [clip, clip]})

        self.assertEqual(tuple(output["rgb_features"].shape), (2, 8, 12))

    def test_eye_gaze_extractor_supports_batched_clip_sequences(self):
        extractor = EyeGazeExtractor(detect_features_fn=fake_eye_gaze_detector)
        clip = [np.zeros((6, 6, 3), dtype=np.uint8) for _ in range(4)]

        output = extractor.extract({"video_rgb_frames": [clip, clip]})

        self.assertEqual(tuple(output["eye_gaze"].shape), (2, 4, 8))

    def test_dataset_can_use_stubbed_video_loader(self):
        dataset = LabeledVideoDataset(
            examples=[
                VideoExample(
                    path=Path("/tmp/example.mp4"),
                    label=1,
                    class_name="fake",
                    source_id="clipA",
                    split="train",
                    identity_id="f1",
                )
            ],
            num_frames=4,
            image_size=8,
        )
        stub_clip = {
            "video": torch.randn(3, 4, 8, 8),
            "video_rgb_frames": [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)],
        }

        with patch("dataset.load_video_clip", return_value=stub_clip):
            sample = dataset[0]

        self.assertEqual(tuple(sample["video"].shape), (3, 4, 8, 8))
        self.assertEqual(tuple(sample["label"].shape), (1,))
        self.assertEqual(sample["identity_id"], "f1")


if __name__ == "__main__":
    unittest.main()
