from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dataset import (
    LabeledVideoDataset,
    VideoExample,
    build_labeled_folder_examples,
    build_metadata_real_fake_examples,
    build_real_fake_examples,
    collate_labeled_video_batch,
    format_split_audit,
    infer_metadata_source_id,
    load_dataset_manifest,
    load_video_metadata,
    split_metadata_examples,
    summarize_examples,
    summarize_split_audit,
    write_dataset_manifest,
)
from extractors.eye_gaze import EYE_GAZE_COLUMNS, EyeGazeExtractor
from extractors.face_mesh import FACE_MESH_CONTOUR_INDICES, FaceMeshExtractor
from extractors.rgb import RGBExtractor


def fake_eye_gaze_detector(_: np.ndarray) -> dict[str, float]:
    return {name: float(index) / 10.0 for index, name in enumerate(EYE_GAZE_COLUMNS, start=1)}


def fake_face_mesh_detector(_: np.ndarray) -> np.ndarray:
    points = np.zeros((len(FACE_MESH_CONTOUR_INDICES), 3), dtype=np.float32)
    for index in range(points.shape[0]):
        points[index] = (index / 100.0, index / 200.0, -index / 300.0)
    return points


class DummyRGBEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return torch.ones(batch, 8, 12, device=x.device)


class DatasetTest(unittest.TestCase):
    def write_meta(
        self,
        path: Path,
        rows: list[dict[str, str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("filename", "age", "gender", "ethnicity", "emotion", "aus_summary"),
            )
            writer.writeheader()
            for row in rows:
                payload = {
                    "filename": row["filename"],
                    "age": row.get("age", "30"),
                    "gender": row.get("gender", "Male"),
                    "ethnicity": row.get("ethnicity", "White"),
                    "emotion": row.get("emotion", "Neutral"),
                    "aus_summary": row.get("aus_summary", ""),
                }
                writer.writerow(payload)

    def test_build_labeled_folder_examples_reads_flat_real_fake_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_dir = root / "real"
            fake_dir = root / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()
            (real_dir / "a.MP4").touch()
            (real_dir / "ignored.txt").touch()
            (fake_dir / "b.webm").touch()

            examples = build_labeled_folder_examples(root, split="train")

            self.assertEqual(len(examples), 2)
            self.assertEqual([example.class_name for example in examples], ["real", "fake"])
            self.assertEqual([example.label for example in examples], [0, 1])
            self.assertEqual({example.split for example in examples}, {"train"})

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

    def test_metadata_loader_rejects_duplicate_filenames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "meta.csv"
            self.write_meta(
                meta_path,
                [
                    {"filename": "a.mp4"},
                    {"filename": "a.mp4"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "duplicate filename"):
                load_video_metadata(meta_path)

    def test_metadata_source_id_parsing_handles_real_fake_and_synthetic_names(self):
        cases = {
            "dutkin_files____6KSteG3p0hQ_00_16_30-00_16_45_m.mp4": (
                "source_video:6KSteG3p0hQ",
                "source_video",
            ),
            "dlc/dutkin_asian____6KSteG3p0hQ_00_16_30-00_16_45_m.mp4": (
                "source_video:6KSteG3p0hQ",
                "source_video",
            ),
            "facefusion/AF100____730aR4nMZm8_00_24_24-00_32_1part10_f_swapped.mp4": (
                "source_video:730aR4nMZm8",
                "source_video",
            ),
            "visomaster/zenin_clips____zkF-bYRDLMg_clip_1_0_20.mp4": (
                "clip:zkF-bYRDLMg",
                "clip",
            ),
            "sadtalker/sadtalker____2025_12_17_12.50.43.mp4": (
                "synthetic_file:sadtalker/sadtalker____2025_12_17_12.50.43.mp4",
                "synthetic_file",
            ),
            "ltx2/ltx2_common1____0.mp4": (
                "synthetic_file:ltx2/ltx2_common1____0.mp4",
                "synthetic_file",
            ),
            "dreamidv/DreamID-V____ComfyUI_00001_.mp4": (
                "synthetic_file:dreamidv/DreamID-V____ComfyUI_00001_.mp4",
                "synthetic_file",
            ),
            "background_aug/background_augmentations____000000_background_augmented.mp4": (
                "synthetic_file:background_aug/background_augmentations____000000_background_augmented.mp4",
                "synthetic_file",
            ),
        }

        for filename, expected in cases.items():
            class_name = "real" if "/" not in filename else "fake"
            self.assertEqual(infer_metadata_source_id(filename, class_name), expected)

    def test_metadata_examples_validate_file_coverage_and_split_with_audit_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_dir = root / "real"
            fake_dir = root / "fake"
            real_dir.mkdir(parents=True)
            for generator in ("gen_a", "gen_b", "gen_c"):
                (fake_dir / generator).mkdir(parents=True)

            real_rows = []
            for index in range(12):
                filename = f"realSrc{index:02d}_00_00_00-00_00_10.mp4"
                (real_dir / filename).write_bytes(b"video")
                real_rows.append(
                    {
                        "filename": filename,
                        "age": str(20 + index),
                        "gender": "Male" if index % 2 == 0 else "Female",
                        "ethnicity": "White" if index % 3 else "Asian",
                        "emotion": "Neutral" if index % 2 == 0 else "Happiness",
                    }
                )
            fake_rows = []
            for generator in ("gen_a", "gen_b", "gen_c"):
                for index in range(6):
                    filename = f"{generator}/{generator}____{index:03d}.mp4"
                    (fake_dir / filename).write_bytes(b"video")
                    fake_rows.append(
                        {
                            "filename": filename,
                            "age": str(30 + index),
                            "gender": "Male" if index % 2 == 0 else "Female",
                            "ethnicity": "Black" if index % 2 == 0 else "White",
                            "emotion": "Sadness" if index % 2 == 0 else "Surprise",
                        }
                    )
            self.write_meta(real_dir / "meta.csv", real_rows)
            self.write_meta(fake_dir / "meta.csv", fake_rows)

            examples = build_metadata_real_fake_examples(real_dir, fake_dir)
            split_examples = split_metadata_examples(
                examples,
                eval_real_count=3,
                eval_fake_count=3,
                seed=7,
            )
            summary = summarize_examples(split_examples)
            audit = summarize_split_audit(split_examples)
            manifest_path = write_dataset_manifest(split_examples, root / "manifest.csv")
            loaded = load_dataset_manifest(manifest_path)

            self.assertEqual(summary["val"]["real"], 3)
            self.assertEqual(summary["test"]["real"], 3)
            self.assertGreaterEqual(summary["val"]["fake"], 3)
            self.assertGreaterEqual(summary["test"]["fake"], 3)
            self.assertEqual(audit["source_leakage_count"], 0)
            self.assertIn("gen_a", audit["fake_generators"]["val"])
            self.assertIn("gen_a", audit["fake_generators"]["test"])
            self.assertTrue(
                any(line.startswith("split_audit") for line in format_split_audit(split_examples))
            )
            self.assertEqual(
                [example.generator_id for example in loaded if example.class_name == "fake"],
                [
                    example.generator_id
                    for example in split_examples
                    if example.class_name == "fake"
                ],
            )

    def test_metadata_builder_rejects_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            real_dir = root / "real"
            fake_dir = root / "fake"
            (fake_dir / "gen").mkdir(parents=True)
            real_dir.mkdir(parents=True)
            self.write_meta(real_dir / "meta.csv", [{"filename": "missing.mp4"}])
            self.write_meta(fake_dir / "meta.csv", [{"filename": "gen/missing.mp4"}])

            with self.assertRaisesRegex(FileNotFoundError, "Metadata coverage mismatch"):
                build_metadata_real_fake_examples(real_dir, fake_dir)

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

    def test_face_mesh_extractor_supports_batched_clip_sequences(self):
        extractor = FaceMeshExtractor(detect_landmarks_fn=fake_face_mesh_detector)
        clip = [np.zeros((6, 6, 3), dtype=np.uint8) for _ in range(4)]

        output = extractor.extract({"video_rgb_frames": [clip, clip]})

        self.assertEqual(
            tuple(output["face_mesh"].shape),
            (2, 4, len(FACE_MESH_CONTOUR_INDICES), 3),
        )

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

    def test_dataset_can_load_modality_specific_frame_counts(self):
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
            num_frames={"rgb": 4, "rppg": 6},
            image_size=8,
        )

        def fake_load_video_clip(path, num_frames, image_size, decode_mode="seek"):
            del path, decode_mode
            return {
                "video": torch.full((3, num_frames, image_size, image_size), float(num_frames)),
                "video_rgb_frames": [
                    np.full((image_size, image_size, 3), num_frames, dtype=np.uint8)
                    for _ in range(num_frames)
                ],
            }

        with patch("dataset.load_video_clip", side_effect=fake_load_video_clip):
            sample = dataset[0]
            batch = collate_labeled_video_batch([sample, sample])

        self.assertEqual(tuple(sample["video_by_modality"]["rgb"].shape), (3, 4, 8, 8))
        self.assertEqual(tuple(sample["video_by_modality"]["rppg"].shape), (3, 6, 8, 8))
        self.assertEqual(tuple(batch["video_by_modality"]["rgb"].shape), (2, 3, 4, 8, 8))
        self.assertEqual(tuple(batch["video_by_modality"]["rppg"].shape), (2, 3, 6, 8, 8))
        self.assertEqual(len(batch["video_rgb_frames_by_modality"]["rgb"][0]), 4)
        self.assertEqual(len(batch["video_rgb_frames_by_modality"]["rppg"][0]), 6)


if __name__ == "__main__":
    unittest.main()
