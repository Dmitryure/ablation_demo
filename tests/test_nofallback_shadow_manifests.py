from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feature_cache import (
    FEATURE_CACHE_MANIFEST_COLUMNS,
    FeatureCacheSpec,
    feature_cache_manifest_path,
)
from scripts.build_nofallback_shadow_manifests import build_shadow_manifests


def spec(modality: str, frames: int) -> FeatureCacheSpec:
    return FeatureCacheSpec(
        version=3,
        modality=modality,
        frame_count=frames,
        image_size=224,
        extractor_config={},
        cache_variant="facecrop_v2_opencv_haar_df16_box1p5_fallback_full_frame",
    )


def write_manifest(
    root: Path,
    cache_spec: FeatureCacheSpec,
    rows: list[tuple[str, str, str]],
) -> Path:
    path = feature_cache_manifest_path(root, cache_spec)
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
                    "source_path": f"/stale/{class_name}/{filename}",
                    "cache_path": f"/stale/cache/{class_name}/{filename}.pt",
                    "modality": cache_spec.modality,
                    "frame_count": cache_spec.frame_count,
                    "status": status,
                    "error": "",
                }
            )
    return path


def read_statuses(path: Path) -> dict[str, tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            f"{row['class_name']}/{row['filename']}": (row["status"], row["error"])
            for row in reader
        }


class NoFallbackShadowManifestTest(unittest.TestCase):
    def test_shadow_manifest_marks_fallback_rows_failed_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            output_dir = root / "shadow"
            dataset_root = root / "dataset"
            specs = {
                "rgb": spec("rgb", 16),
                "depth": spec("depth", 32),
                "eye_gaze": spec("eye_gaze", 32),
            }
            rows = [
                ("real", "r1.mp4", "cached"),
                ("fake", "gen/f1.mp4", "cached"),
                ("real", "r2.mp4", "failed"),
            ]
            source_paths = {
                modality: write_manifest(cache_dir, cache_spec, rows)
                for modality, cache_spec in specs.items()
            }
            source_before = {
                modality: path.read_text(encoding="utf-8")
                for modality, path in source_paths.items()
            }
            config = {
                "frames": {"default": 32},
                "image_size": 224,
                "face_crop": {
                    "enabled": True,
                    "backend": "opencv_haar",
                    "detection_frequency": 16,
                    "large_box_coef": 1.5,
                    "fallback": "full_frame",
                },
                "rgb": {"frames": 16},
            }

            def fake_detect(
                rows_by_key,
                group,
                dataset_root,
                decode_mode,
                progress_every,
                skip_failures,
            ):
                fallback = {"real/r1.mp4"} if group.frame_count == 16 else {"fake/gen/f1.mp4"}
                failures = {"real/r2.mp4"} if group.frame_count == 16 else set()
                return fallback, failures, {
                    "checked": len(rows_by_key),
                    "fallback": len(fallback),
                    "decode_failures": len(failures),
                    "detected": len(rows_by_key) - len(fallback) - len(failures),
                    "elapsed_seconds": 0.0,
                }

            with patch(
                "scripts.build_nofallback_shadow_manifests.detect_fallback_keys",
                side_effect=fake_detect,
            ):
                summary = build_shadow_manifests(
                    config=config,
                    cache_dir=cache_dir,
                    output_dir=output_dir,
                    dataset_root=dataset_root,
                    modalities=("rgb", "depth", "eye_gaze"),
                    apply_to_modalities=None,
                    expected_rows=3,
                    decode_mode="scan",
                    progress_every=0,
                    skip_failures=True,
                )

            rgb_statuses = read_statuses(feature_cache_manifest_path(output_dir, specs["rgb"]))
            depth_statuses = read_statuses(feature_cache_manifest_path(output_dir, specs["depth"]))
            eye_statuses = read_statuses(feature_cache_manifest_path(output_dir, specs["eye_gaze"]))

            self.assertEqual(rgb_statuses["real/r1.mp4"], ("failed", "face_crop_not_detected"))
            self.assertEqual(rgb_statuses["fake/gen/f1.mp4"], ("cached", ""))
            self.assertEqual(depth_statuses["fake/gen/f1.mp4"], ("failed", "face_crop_not_detected"))
            self.assertEqual(eye_statuses["fake/gen/f1.mp4"], ("failed", "face_crop_not_detected"))
            self.assertEqual(summary["manifests"]["depth"]["marked_failed"], 1)
            self.assertEqual(summary["manifests"]["eye_gaze"]["marked_failed"], 1)
            for modality, path in source_paths.items():
                self.assertEqual(path.read_text(encoding="utf-8"), source_before[modality])

    def test_rgb_fallback_keys_can_be_applied_to_other_modalities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "cache"
            output_dir = root / "shadow"
            dataset_root = root / "dataset"
            specs = {
                "rgb": spec("rgb", 16),
                "depth": spec("depth", 32),
            }
            rows = [
                ("real", "r1.mp4", "cached"),
                ("fake", "gen/f1.mp4", "cached"),
            ]
            for cache_spec in specs.values():
                write_manifest(cache_dir, cache_spec, rows)
            config = {
                "frames": {"default": 32},
                "image_size": 224,
                "face_crop": {
                    "enabled": True,
                    "backend": "opencv_haar",
                    "detection_frequency": 16,
                    "large_box_coef": 1.5,
                    "fallback": "full_frame",
                },
                "rgb": {"frames": 16},
            }

            def fake_detect(
                rows_by_key,
                group,
                dataset_root,
                decode_mode,
                progress_every,
                skip_failures,
            ):
                fallback = {"fake/gen/f1.mp4"}
                return fallback, set(), {
                    "checked": len(rows_by_key),
                    "fallback": len(fallback),
                    "decode_failures": 0,
                    "detected": len(rows_by_key) - len(fallback),
                    "elapsed_seconds": 0.0,
                }

            with patch(
                "scripts.build_nofallback_shadow_manifests.detect_fallback_keys",
                side_effect=fake_detect,
            ):
                build_shadow_manifests(
                    config=config,
                    cache_dir=cache_dir,
                    output_dir=output_dir,
                    dataset_root=dataset_root,
                    modalities=("rgb",),
                    apply_to_modalities=("rgb", "depth"),
                    expected_rows=2,
                    decode_mode="scan",
                    progress_every=0,
                    skip_failures=False,
                )

            rgb_statuses = read_statuses(feature_cache_manifest_path(output_dir, specs["rgb"]))
            depth_statuses = read_statuses(feature_cache_manifest_path(output_dir, specs["depth"]))
            self.assertEqual(rgb_statuses["fake/gen/f1.mp4"], ("failed", "face_crop_not_detected"))
            self.assertEqual(depth_statuses["fake/gen/f1.mp4"], ("failed", "face_crop_not_detected"))


if __name__ == "__main__":
    unittest.main()
