from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from dataset import VideoExample
from feature_cache import (
    FEATURE_CACHE_MANIFEST_COLUMNS,
    FeatureCacheSpec,
    feature_cache_manifest_path,
)
from scripts.build_readonly_cache_shards import (
    default_shard_size,
    load_manifest_backed_examples_from_cache_dirs,
    shard_example_locations,
    stratified_shard_order,
)
from scripts.validate_readonly_cache_shards import validate_readonly_cache_shards


def example(class_name: str, filename: str) -> VideoExample:
    return VideoExample(
        path=Path("/videos") / class_name / filename,
        label=0 if class_name == "real" else 1,
        class_name=class_name,
        source_id=Path(filename).stem,
        split="train",
        metadata_filename=filename,
    )


def rgb_spec() -> FeatureCacheSpec:
    return FeatureCacheSpec(
        version=3,
        modality="rgb",
        frame_count=16,
        image_size=224,
        extractor_config={},
    )


def write_manifest(
    root: Path,
    spec: FeatureCacheSpec,
    rows: list[tuple[str, str, str]],
) -> None:
    path = feature_cache_manifest_path(root, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_CACHE_MANIFEST_COLUMNS)
        writer.writeheader()
        for class_name, filename, status in rows:
            writer.writerow(
                {
                    "class_name": class_name,
                    "filename": filename,
                    "generator_id": "real" if class_name == "real" else filename.split("/", 1)[0],
                    "source_path": f"/videos/{class_name}/{filename}",
                    "cache_path": f"/cache/{class_name}/{filename}.pt",
                    "modality": spec.modality,
                    "frame_count": spec.frame_count,
                    "status": status,
                    "error": "",
                }
            )


class ReadOnlyCacheShardTest(unittest.TestCase):
    def test_default_shard_size_is_smaller_with_fau(self):
        self.assertEqual(default_shard_size(("rgb", "fau")), 128)
        self.assertEqual(default_shard_size(("rgb", "rppg")), 512)

    def test_stratified_order_avoids_single_label_shards(self):
        examples = [
            *[example("real", f"r{index}.mp4") for index in range(12)],
            *[example("fake", f"gen/f{index}.mp4") for index in range(12)],
        ]

        ordered = stratified_shard_order(examples, shard_size=6, seed=0)
        shards = [ordered[index : index + 6] for index in range(0, len(ordered), 6)]

        self.assertEqual(len(ordered), len(examples))
        for shard in shards:
            self.assertEqual({item.class_name for item in shard}, {"real", "fake"})

    def test_v2_locations_include_label_and_filename(self):
        examples = [example("real", "r.mp4"), example("fake", "gen/f.mp4")]

        locations = shard_example_locations(examples, shard_size=1, dataset_root=Path("/videos"))

        self.assertEqual(locations["real/r.mp4"]["label"], 0)
        self.assertEqual(locations["fake/gen/f.mp4"]["filename"], "gen/f.mp4")

    def test_manifest_override_controls_builder_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            manifest_dir = root / "shadow"
            spec = rgb_spec()
            write_manifest(
                cache_dir,
                spec,
                [
                    ("real", "r1.mp4", "cached"),
                    ("fake", "gen/f1.mp4", "cached"),
                ],
            )
            write_manifest(
                manifest_dir,
                spec,
                [
                    ("real", "r1.mp4", "cached"),
                    ("fake", "gen/f1.mp4", "failed"),
                ],
            )

            examples, summary = load_manifest_backed_examples_from_cache_dirs(
                cache_dirs={"rgb": cache_dir},
                manifest_dirs={"rgb": manifest_dir},
                specs={"rgb": spec},
                modalities=("rgb",),
                expected_rows=2,
            )

        self.assertEqual([item.metadata_filename for item in examples], ["r1.mp4"])
        self.assertEqual(summary["cache_dirs_by_modality"]["rgb"], str(cache_dir))
        self.assertEqual(summary["manifest_dirs_by_modality"]["rgb"], str(manifest_dir))

    def test_validator_detects_row_mapping_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "shards").mkdir()
            torch.save(
                {
                    "examples": [
                        {"class_name": "real", "metadata_filename": "wrong.mp4"},
                    ],
                    "features": {"rgb_features": torch.ones(1, 2)},
                },
                root / "shards" / "shard_000000.pt",
            )
            index = {
                "schema_version": 2,
                "shard_size": 1,
                "example_count": 1,
                "example_locations": {
                    "real/r.mp4": {
                        "shard_index": 0,
                        "row_index": 0,
                        "class_name": "real",
                        "filename": "r.mp4",
                        "label": 0,
                    }
                },
                "shards": [
                    {
                        "path": "shards/shard_000000.pt",
                        "example_count": 1,
                        "class_counts": {"real": 1},
                    }
                ],
            }
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_readonly_cache_shards(root, check_payloads=True)

    def test_validator_rejects_legacy_index_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index = {
                "version": 1,
                "shard_size": 1,
                "example_count": 1,
                "example_locations": {"real/r.mp4": {"shard_index": 0, "row_index": 0}},
                "shards": [],
            }
            (root / "index.json").write_text(json.dumps(index), encoding="utf-8")

            with self.assertRaises(ValueError):
                validate_readonly_cache_shards(root, check_payloads=False)


if __name__ == "__main__":
    unittest.main()
