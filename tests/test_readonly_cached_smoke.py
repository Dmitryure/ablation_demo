from __future__ import annotations

import csv
import tempfile
import time
import unittest
from pathlib import Path

from feature_cache import (
    FEATURE_CACHE_MANIFEST_COLUMNS,
    FeatureCacheSpec,
    feature_cache_manifest_path,
)
from scripts.run_readonly_cached_smoke import (
    assert_cache_stats_unchanged,
    cache_stat_snapshot,
    import_matplotlib_pyplot,
    load_manifest_backed_examples,
    load_manifest_backed_examples_from_cache_dirs,
    reject_missing_selected_cache,
    reject_output_inside_inputs,
    select_readonly_splits,
    selected_cache_paths,
    write_core_plots,
)


def rgb_spec() -> FeatureCacheSpec:
    return FeatureCacheSpec(
        version=3,
        modality="rgb",
        frame_count=16,
        image_size=224,
        extractor_config={},
    )


def fau_spec() -> FeatureCacheSpec:
    return FeatureCacheSpec(
        version=3,
        modality="fau",
        frame_count=64,
        image_size=224,
        extractor_config={},
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
                    "generator_id": "real" if class_name == "real" else filename.split("/", 1)[0],
                    "source_path": f"/stale/videos/{class_name}/{filename}",
                    "cache_path": f"/stale/cache/{class_name}/{filename}.pt",
                    "modality": spec.modality,
                    "frame_count": spec.frame_count,
                    "status": status,
                    "error": "",
                }
            )


class ReadOnlyCachedSmokeTest(unittest.TestCase):
    def test_manifest_only_selection_ignores_stale_cache_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            specs = {"rgb": rgb_spec(), "fau": fau_spec()}
            rows = [
                ("real", "r1.mp4", "cached"),
                ("real", "r2.mp4", "cached"),
                ("real", "r3.mp4", "cached"),
                ("real", "r4.mp4", "cached"),
                ("fake", "gen/f1.mp4", "cached"),
                ("fake", "gen/f2.mp4", "cached"),
                ("fake", "gen/f3.mp4", "cached"),
                ("fake", "gen/f4.mp4", "cached"),
            ]
            write_manifest(cache_dir, specs["rgb"], rows)
            write_manifest(cache_dir, specs["fau"], rows)

            examples, summary = load_manifest_backed_examples(
                cache_dir=cache_dir,
                specs=specs,
                modalities=("rgb", "fau"),
                expected_rows=8,
            )
            train, val, test = select_readonly_splits(
                examples,
                balanced_total=8,
                train_ratio=0.5,
                val_ratio=0.25,
                seed=0,
            )

        self.assertEqual(summary["intersection_cached"], 8)
        self.assertEqual(len(train) + len(val) + len(test), 8)
        self.assertTrue(all(example.metadata_filename for example in [*train, *val, *test]))

    def test_full_cache_selection_keeps_all_intersected_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            specs = {"rgb": rgb_spec(), "fau": fau_spec()}
            rows = [
                *[("real", f"r{index}.mp4", "cached") for index in range(5)],
                *[("fake", f"gen/f{index}.mp4", "cached") for index in range(5)],
            ]
            write_manifest(cache_dir, specs["rgb"], rows)
            write_manifest(cache_dir, specs["fau"], rows)

            examples, _ = load_manifest_backed_examples(
                cache_dir=cache_dir,
                specs=specs,
                modalities=("rgb", "fau"),
                expected_rows=10,
            )
            train, val, test = select_readonly_splits(
                examples,
                balanced_total=None,
                train_ratio=0.6,
                val_ratio=0.2,
                seed=0,
            )

        self.assertEqual(len(train), 6)
        self.assertEqual(len(val), 2)
        self.assertEqual(len(test), 2)
        self.assertEqual(len(train) + len(val) + len(test), 10)

    def test_manifest_dir_override_controls_selection(self):
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

        self.assertEqual([example.metadata_filename for example in examples], ["r1.mp4"])
        self.assertEqual(summary["cache_dirs_by_modality"]["rgb"], str(cache_dir))
        self.assertEqual(summary["manifest_dirs_by_modality"]["rgb"], str(manifest_dir))

    def test_smoke_selection_still_uses_balanced_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            spec = rgb_spec()
            rows = [
                *[("real", f"r{index}.mp4", "cached") for index in range(6)],
                *[("fake", f"gen/f{index}.mp4", "cached") for index in range(6)],
            ]
            write_manifest(cache_dir, spec, rows)
            examples, _ = load_manifest_backed_examples(
                cache_dir=cache_dir,
                specs={"rgb": spec},
                modalities=("rgb",),
                expected_rows=12,
            )
            train, val, test = select_readonly_splits(
                examples,
                balanced_total=8,
                train_ratio=0.5,
                val_ratio=0.25,
                seed=0,
            )

        self.assertEqual(len(train) + len(val) + len(test), 8)
        self.assertEqual({example.split for example in train}, {"train"})
        self.assertEqual({example.split for example in val}, {"val"})
        self.assertEqual({example.split for example in test}, {"test"})

    def test_output_guard_rejects_cache_or_dataset_subdirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(ValueError):
                reject_output_inside_inputs(root / "cache" / "out", root / "cache", root / "data")
            with self.assertRaises(ValueError):
                reject_output_inside_inputs(root / "data" / "out", root / "cache", root / "data")
            reject_output_inside_inputs(root / "runs", root / "cache", root / "data")

    def test_missing_selected_pt_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            spec = rgb_spec()
            write_manifest(cache_dir, spec, [("real", "r1.mp4", "cached")])
            examples, _ = load_manifest_backed_examples(
                cache_dir=cache_dir,
                specs={"rgb": spec},
                modalities=("rgb",),
                expected_rows=1,
            )
            paths = selected_cache_paths(examples, cache_dir, {"rgb": spec}, ("rgb",))

            with self.assertRaises(FileNotFoundError):
                reject_missing_selected_cache(paths)

    def test_readonly_audit_detects_changed_cache_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.csv"
            path.write_text("a\n", encoding="utf-8")
            before = cache_stat_snapshot([path])
            time.sleep(0.001)
            path.write_text("changed\n", encoding="utf-8")
            after = cache_stat_snapshot([path])

            with self.assertRaises(RuntimeError):
                assert_cache_stats_unchanged(before, after)

    def test_core_plots_generate_pngs_from_training_csvs(self):
        plt, import_error = import_matplotlib_pyplot()
        if plt is None:
            self.skipTest(f"matplotlib unavailable: {import_error}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            training_dir = root / "train"
            training_dir.mkdir()
            with (training_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "train_accuracy",
                        "train_f1",
                        "val_loss",
                        "val_accuracy",
                        "val_f1",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "epoch": "1",
                        "train_loss": "0.7",
                        "train_accuracy": "0.5",
                        "train_f1": "0.5",
                        "val_loss": "0.8",
                        "val_accuracy": "0.4",
                        "val_f1": "0.4",
                    }
                )
                writer.writerow(
                    {
                        "epoch": "2",
                        "train_loss": "0.5",
                        "train_accuracy": "0.8",
                        "train_f1": "0.8",
                        "val_loss": "0.6",
                        "val_accuracy": "0.7",
                        "val_f1": "0.7",
                    }
                )
            with (training_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["path", "class_name", "label", "prediction", "probability", "split"],
                )
                writer.writeheader()
                for split in ("train", "val", "test"):
                    writer.writerow(
                        {
                            "path": f"{split}/real.mp4",
                            "class_name": "real",
                            "label": "0",
                            "prediction": "0",
                            "probability": "0.1",
                            "split": split,
                        }
                    )
                    writer.writerow(
                        {
                            "path": f"{split}/fake.mp4",
                            "class_name": "fake",
                            "label": "1",
                            "prediction": "1",
                            "probability": "0.9",
                            "split": split,
                        }
                    )

            summary = write_core_plots(training_dir, root)

            for plot in summary["plots"].values():
                self.assertTrue(plot["produced"], plot)
                self.assertTrue(Path(plot["path"]).is_file(), plot)
            self.assertTrue((root / "plots" / "plots_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
