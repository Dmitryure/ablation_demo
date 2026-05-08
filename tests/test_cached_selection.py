from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from dataset import VideoExample
from feature_cache import (
    FEATURE_CACHE_MANIFEST_COLUMNS,
    FeatureCacheSpec,
    feature_cache_manifest_path,
)
from scripts.run_iterative_cached_ablation import select_fully_cached_examples
from scripts.run_iterative_cached_ablation import count_cached_and_skipped
from scripts.run_iterative_cached_ablation import count_missing_cache
from scripts.run_iterative_cached_ablation import missing_examples_for_modality
from scripts.run_iterative_cached_ablation import read_cached_manifest_keys


def build_example(class_name: str, filename: str) -> VideoExample:
    return VideoExample(
        path=Path(f"/dataset/videos/{class_name}/{filename}"),
        label=0 if class_name == "real" else 1,
        class_name=class_name,
        source_id=filename,
        split="train",
        metadata_filename=filename,
    )


def write_manifest(
    cache_dir: Path,
    spec: FeatureCacheSpec,
    rows: list[tuple[str, str, str]],
) -> None:
    path = feature_cache_manifest_path(cache_dir, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_CACHE_MANIFEST_COLUMNS)
        writer.writeheader()
        for class_name, filename, status in rows:
            writer.writerow(
                {
                    "class_name": class_name,
                    "filename": filename,
                    "generator_id": "real" if class_name == "real" else "fake_gen",
                    "source_path": f"/dataset/videos/{class_name}/{filename}",
                    "cache_path": f"{class_name}/{filename}.pt",
                    "modality": spec.modality,
                    "frame_count": spec.frame_count,
                    "status": status,
                    "error": "",
                }
            )


class CachedSelectionTest(unittest.TestCase):
    def test_select_fully_cached_examples_uses_manifest_intersection(self):
        examples = [
            build_example("real", "a.mp4"),
            build_example("real", "b.mp4"),
            build_example("fake", "gen/c.mp4"),
        ]
        rgb_spec = FeatureCacheSpec(
            version=3,
            modality="rgb",
            frame_count=16,
            image_size=224,
            extractor_config={},
        )
        fau_spec = FeatureCacheSpec(
            version=3,
            modality="fau",
            frame_count=64,
            image_size=224,
            extractor_config={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            write_manifest(
                cache_dir,
                rgb_spec,
                [
                    ("real", "a.mp4", "cached"),
                    ("real", "b.mp4", "failed"),
                    ("fake", "gen/c.mp4", "cached"),
                ],
            )
            write_manifest(
                cache_dir,
                fau_spec,
                [
                    ("real", "a.mp4", "cached"),
                    ("real", "b.mp4", "failed"),
                    ("fake", "gen/c.mp4", "failed"),
                ],
            )

            selected, summary = select_fully_cached_examples(
                examples=examples,
                cache_dir=cache_dir,
                specs={"rgb": rgb_spec, "fau": fau_spec},
                modalities=("rgb", "fau"),
                dataset_root=Path("/dataset"),
            )

        self.assertEqual([example.metadata_filename for example in selected], ["a.mp4"])
        self.assertEqual(summary["fully_cached_examples"], 1)
        self.assertEqual(summary["manifest_backed_modalities"], 2)

    def test_cache_audit_helpers_use_manifest(self):
        examples = [
            build_example("real", "a.mp4"),
            build_example("real", "b.mp4"),
            build_example("fake", "gen/c.mp4"),
        ]
        spec = FeatureCacheSpec(
            version=3,
            modality="rgb",
            frame_count=16,
            image_size=224,
            extractor_config={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            write_manifest(
                cache_dir,
                spec,
                [
                    ("real", "a.mp4", "cached"),
                    ("real", "b.mp4", "failed"),
                    ("fake", "gen/c.mp4", "cached"),
                ],
            )

            missing = missing_examples_for_modality(
                examples=examples,
                cache_dir=cache_dir,
                spec=spec,
                dataset_root=Path("/dataset"),
                overwrite=False,
                skip_failure_keys=set(),
            )
            missing_counts = count_missing_cache(
                examples=examples,
                cache_dir=cache_dir,
                specs={"rgb": spec},
                modalities=("rgb",),
                dataset_root=Path("/dataset"),
            )
            cached, skipped = count_cached_and_skipped(
                examples=examples,
                cache_dir=cache_dir,
                spec=spec,
                dataset_root=Path("/dataset"),
                skip_failure_keys=set(),
            )

        self.assertEqual([example.metadata_filename for example in missing], ["b.mp4"])
        self.assertEqual(missing_counts, {"rgb": 1})
        self.assertEqual((cached, skipped), (2, 0))

    def test_manifest_reader_rejects_too_small_subset_manifest(self):
        spec = FeatureCacheSpec(
            version=3,
            modality="rgb",
            frame_count=16,
            image_size=224,
            extractor_config={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            write_manifest(cache_dir, spec, [("real", "a.mp4", "cached")])

            keys = read_cached_manifest_keys(cache_dir, spec, minimum_rows=2)

        self.assertIsNone(keys)


if __name__ == "__main__":
    unittest.main()
