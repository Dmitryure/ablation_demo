from __future__ import annotations

import unittest
from pathlib import Path

from dataset import VideoExample
from scripts.precompute_feature_cache import (
    select_balanced_total_examples,
    select_examples_by_split,
    select_precompute_examples,
)
from scripts.run_iterative_cached_ablation import extraction_modality_groups


def build_example(index: int, split: str, class_name: str) -> VideoExample:
    label = 0 if class_name == "real" else 1
    return VideoExample(
        path=Path(f"/tmp/{split}/{class_name}/{index}.mp4"),
        label=label,
        class_name=class_name,
        source_id=f"{class_name}-{index}",
        split=split,
    )


class PrecomputeFeatureCacheSelectionTest(unittest.TestCase):
    def test_balanced_limit_per_split_caps_by_smaller_class(self):
        examples: list[VideoExample] = []
        for index in range(8):
            examples.append(build_example(index, "train", "real"))
        for index in range(20):
            examples.append(build_example(index, "train", "fake"))
        for index in range(2):
            examples.append(build_example(index, "test", "real"))
        for index in range(10):
            examples.append(build_example(index, "test", "fake"))

        selected = select_examples_by_split(
            examples,
            splits=("train", "test"),
            limit_per_split=None,
            limit_total=None,
            balanced_limit_per_split=10,
            seed=0,
        )

        self.assertEqual(len(selected), 14)
        for split, expected_per_class in (("train", 5), ("test", 2)):
            split_examples = [example for example in selected if example.split == split]
            self.assertEqual(
                sum(example.class_name == "real" for example in split_examples),
                expected_per_class,
            )
            self.assertEqual(
                sum(example.class_name == "fake" for example in split_examples),
                expected_per_class,
            )

    def test_balanced_limit_rejects_other_limits(self):
        with self.assertRaises(ValueError):
            select_examples_by_split(
                [],
                splits=("train",),
                limit_per_split=None,
                limit_total=100,
                balanced_limit_per_split=10,
                seed=0,
            )

    def test_balanced_total_selects_equal_classes_across_whole_pool(self):
        examples: list[VideoExample] = []
        for index in range(8):
            examples.append(build_example(index, "train", "real"))
        for index in range(8, 12):
            examples.append(build_example(index, "val", "real"))
        for index in range(20):
            examples.append(build_example(index, "test", "fake"))

        selected = select_balanced_total_examples(examples, balanced_total=10, seed=0)

        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(example.class_name == "real" for example in selected), 5)
        self.assertEqual(sum(example.class_name == "fake" for example in selected), 5)

    def test_balanced_total_rejects_odd_or_too_large_counts(self):
        examples = [build_example(index, "train", "real") for index in range(2)]
        examples.extend(build_example(index, "train", "fake") for index in range(2))

        with self.assertRaises(ValueError):
            select_balanced_total_examples(examples, balanced_total=3, seed=0)
        with self.assertRaises(ValueError):
            select_balanced_total_examples(examples, balanced_total=6, seed=0)

    def test_select_precompute_examples_defaults_to_all_splits(self):
        examples = [
            build_example(0, "train", "real"),
            build_example(1, "val", "fake"),
            build_example(2, "test", "fake"),
        ]

        selected = select_precompute_examples(
            examples,
            splits=("train", "val", "test"),
            limit_per_split=None,
            limit_total=None,
            balanced_limit_per_split=None,
            balanced_total=None,
            seed=0,
        )

        self.assertEqual(selected, examples)

    def test_select_precompute_examples_rejects_balanced_total_with_other_limits(self):
        with self.assertRaises(ValueError):
            select_precompute_examples(
                [],
                splits=("train",),
                limit_per_split=1,
                limit_total=None,
                balanced_limit_per_split=None,
                balanced_total=10,
                seed=0,
            )

    def test_extraction_groups_can_isolate_modalities(self):
        class Spec:
            frame_count = 32
            image_size = 224
            cache_variant = None
            frame_sampling_variant = None

        specs = {modality: Spec() for modality in ("rppg", "depth", "fft")}

        grouped = extraction_modality_groups(
            ("rppg", "depth", "fft"),
            specs,
            group_by_modality=True,
        )

        self.assertEqual(
            grouped,
            [
                (32, 224, None, None, ("rppg",)),
                (32, 224, None, None, ("depth",)),
                (32, 224, None, None, ("fft",)),
            ],
        )

    def test_extraction_groups_separate_cache_variants(self):
        class Spec:
            def __init__(
                self,
                cache_variant: str | None,
                frame_sampling_variant: str | None = None,
            ):
                self.frame_count = 32
                self.image_size = 224
                self.cache_variant = cache_variant
                self.frame_sampling_variant = frame_sampling_variant

        specs = {
            "rgb": Spec("facecrop_v2_opencv_haar"),
            "depth": Spec("facecrop_v2_opencv_haar"),
            "fft": Spec(None),
        }

        grouped = extraction_modality_groups(
            ("rgb", "depth", "fft"),
            specs,
            group_by_modality=False,
        )

        self.assertEqual(
            grouped,
            [
                (32, 224, "facecrop_v2_opencv_haar", None, ("rgb", "depth")),
                (32, 224, None, None, ("fft",)),
            ],
        )

    def test_extraction_groups_separate_frame_sampling_variants(self):
        class Spec:
            def __init__(self, frame_sampling_variant: str | None):
                self.frame_count = 96
                self.image_size = 224
                self.cache_variant = "facecrop_v2_opencv_haar"
                self.frame_sampling_variant = frame_sampling_variant

        specs = {
            "rgb": Spec("center192_stride2_frames96_repeatpad"),
            "depth": Spec("center192_stride2_frames96_repeatpad"),
            "face_mesh": Spec(None),
        }

        grouped = extraction_modality_groups(
            ("rgb", "depth", "face_mesh"),
            specs,
            group_by_modality=False,
        )

        self.assertEqual(
            grouped,
            [
                (
                    96,
                    224,
                    "facecrop_v2_opencv_haar",
                    "center192_stride2_frames96_repeatpad",
                    ("rgb", "depth"),
                ),
                (96, 224, "facecrop_v2_opencv_haar", None, ("face_mesh",)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
