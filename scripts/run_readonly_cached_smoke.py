from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoExample, write_dataset_manifest
from feature_cache import (
    FeatureCacheSpec,
    build_feature_cache_specs,
    feature_cache_item_path,
    feature_cache_manifest_path,
    feature_cache_spec_id,
)
from pipeline import load_pipeline_yaml
from scripts.run_iterative_cached_ablation import (
    RUN_ARG_DEFAULTS,
    class_counts,
    feature_cache_manifest_summary,
    run_training_round,
    training_run_section,
    video_metadata_summary,
    write_json,
)
from scripts.validate_readonly_cache_shards import validate_readonly_cache_shards

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_smoke_cache_v1_rgb_1k.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "readonly_smoke_runs"
DEFAULT_EXPECTED_MANIFEST_ROWS = 13768
DEFAULT_PROGRESS_EVERY = 5000
V1_ALL_MODALITIES = frozenset(
    ("rgb", "fau", "rppg", "eye_gaze", "face_mesh", "depth", "fft", "stft")
)
V1_ALL_EXPECTED_INTERSECTION = 13743
MANIFEST_COLUMNS = {
    "class_name",
    "filename",
    "generator_id",
    "source_path",
    "status",
}


@dataclass(frozen=True)
class ManifestEntry:
    key: str
    example: VideoExample


@dataclass(frozen=True)
class CacheFileStat:
    path: str
    exists: bool
    size: int | None
    mtime_ns: int | None


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train from immutable cached feature manifests without cache writes."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--sharded-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--modality-cache-dir",
        action="append",
        default=[],
        help="Per-modality cache override, e.g. rppg=/mnt/d/final_cache/v1.",
    )
    parser.add_argument(
        "--modality-manifest-dir",
        action="append",
        default=[],
        help="Per-modality manifest override, e.g. rppg=/mnt/d/final_cache/v1.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument(
        "--balanced-total",
        default=None,
        help="Even integer for a balanced subset, or `full` to use all cached examples.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--expected-manifest-rows", type=int, default=DEFAULT_EXPECTED_MANIFEST_ROWS
    )
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument(
        "--readonly-audit",
        action="store_true",
        help="Opt in to before/after cache stat comparison for manifests and selected .pt files.",
    )
    parser.add_argument(
        "--check-cache-files",
        action="store_true",
        help="Opt in to checking every selected .pt path exists before model build.",
    )
    return parser.parse_args()


def parse_balanced_total(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        parsed = value
    else:
        text = str(value).strip().lower()
        if text in {"", "full", "all", "none", "null"}:
            return None
        parsed = int(text)
    if parsed <= 0 or parsed % 2 != 0:
        raise ValueError("`balanced_total` must be a positive even integer or `full`.")
    return parsed


def resolve_run_args(config: Mapping[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    run_config = training_run_section(config)
    values = dict(RUN_ARG_DEFAULTS)
    for key, value in run_config.items():
        if key in values:
            values[key] = value

    values["manifest_dir"] = run_config.get("manifest_dir")
    values["modality_manifest_dirs"] = run_config.get("modality_manifest_dirs") or {}

    for key in (
        "dataset_root",
        "cache_dir",
        "manifest_dir",
        "sharded_cache_dir",
        "output_dir",
        "clip_cache_dir",
    ):
        if values.get(key) is not None:
            values[key] = Path(values[key])

    for key in ("modalities", "round_targets", "occlusion_splits"):
        if values.get(key) is not None and not isinstance(values[key], tuple):
            values[key] = list(values[key])

    if cli_args.cache_dir is not None:
        values["cache_dir"] = cli_args.cache_dir
    if cli_args.manifest_dir is not None:
        values["manifest_dir"] = cli_args.manifest_dir
    if cli_args.sharded_cache_dir is not None:
        values["sharded_cache_dir"] = cli_args.sharded_cache_dir
    if cli_args.output_dir is not None:
        values["output_dir"] = cli_args.output_dir
    if cli_args.modalities is not None:
        values["modalities"] = list(cli_args.modalities)
    if cli_args.balanced_total is not None:
        values["balanced_total"] = parse_balanced_total(cli_args.balanced_total)
    if cli_args.epochs is not None:
        values["epochs"] = cli_args.epochs
    if cli_args.seed is not None:
        values["seed"] = cli_args.seed

    if values["cache_dir"] is None:
        raise ValueError("Read-only smoke requires `training.run.cache_dir` or `--cache-dir`.")
    values["balanced_total"] = parse_balanced_total(values.get("balanced_total"))
    if values["modalities"] is None:
        configured = config.get("modalities")
        if not isinstance(configured, list):
            raise ValueError("Config `modalities` must be a YAML list when no override is given.")
        values["modalities"] = list(configured)
    values["output_dir"] = Path(values["output_dir"] or DEFAULT_OUTPUT_DIR)

    return argparse.Namespace(**values)


def parse_modality_cache_dir_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODALITY=PATH for --modality-cache-dir, got {value!r}.")
        modality, path = value.split("=", 1)
        modality = modality.strip()
        if not modality:
            raise ValueError(f"Empty modality in --modality-cache-dir {value!r}.")
        overrides[modality] = Path(path.strip())
    return overrides


def parse_modality_manifest_dir_overrides(values: list[str]) -> dict[str, Path]:
    return parse_modality_cache_dir_overrides(values)


def config_modality_cache_dirs(config: Mapping[str, Any]) -> dict[str, Path]:
    run_config = training_run_section(config)
    value = run_config.get("modality_cache_dirs") or {}
    if not isinstance(value, Mapping):
        raise ValueError("`training.run.modality_cache_dirs` must be a YAML mapping.")
    return {str(modality): Path(path) for modality, path in value.items()}


def config_modality_manifest_dirs(config: Mapping[str, Any]) -> dict[str, Path]:
    run_config = training_run_section(config)
    value = run_config.get("modality_manifest_dirs") or {}
    if not isinstance(value, Mapping):
        raise ValueError("`training.run.modality_manifest_dirs` must be a YAML mapping.")
    return {str(modality): Path(path) for modality, path in value.items()}


def resolve_cache_dirs_by_modality(
    config: Mapping[str, Any],
    base_cache_dir: Path,
    modalities: Sequence[str],
    cli_overrides: list[str],
) -> dict[str, Path]:
    overrides = {
        **config_modality_cache_dirs(config),
        **parse_modality_cache_dir_overrides(cli_overrides),
    }
    return {modality: overrides.get(modality, base_cache_dir) for modality in modalities}


def resolve_manifest_dirs_by_modality(
    config: Mapping[str, Any],
    base_cache_dir: Path,
    base_manifest_dir: Path | None,
    modalities: Sequence[str],
    cli_overrides: list[str],
) -> dict[str, Path]:
    base_dir = base_manifest_dir or base_cache_dir
    overrides = {
        **config_modality_manifest_dirs(config),
        **parse_modality_manifest_dir_overrides(cli_overrides),
    }
    return {modality: overrides.get(modality, base_dir) for modality in modalities}


def _resolve_for_guard(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def reject_output_inside_inputs(output_dir: Path, cache_dir: Path, dataset_root: Path) -> None:
    output = _resolve_for_guard(output_dir)
    cache = _resolve_for_guard(cache_dir)
    dataset = _resolve_for_guard(dataset_root)
    if is_relative_to(output, cache):
        raise ValueError(f"Refusing output_dir inside cache_dir: {output_dir}")
    if is_relative_to(output, dataset):
        raise ValueError(f"Refusing output_dir inside dataset_root: {output_dir}")


def manifest_key(class_name: str, filename: str) -> str:
    return f"{class_name}/{filename}"


def label_for_class(class_name: str) -> int:
    if class_name == "real":
        return 0
    if class_name == "fake":
        return 1
    raise ValueError(f"Unsupported class_name in cache manifest: {class_name!r}")


def source_id_from_filename(class_name: str, filename: str, generator_id: str) -> str:
    if class_name == "fake" and generator_id:
        return generator_id
    return Path(filename).stem


def example_from_manifest_row(row: Mapping[str, str]) -> VideoExample:
    class_name = str(row["class_name"]).strip()
    filename = str(row["filename"]).strip()
    generator_id = str(row.get("generator_id", "")).strip()
    source_path = str(row.get("source_path", "")).strip()
    if not filename:
        raise ValueError("Cache manifest row has empty filename.")
    path = (
        Path(source_path)
        if source_path
        else Path("/readonly_cache_manifest") / class_name / filename
    )
    return VideoExample(
        path=path,
        label=label_for_class(class_name),
        class_name=class_name,
        source_id=source_id_from_filename(class_name, filename, generator_id),
        split="train",
        metadata_filename=filename,
        identity_id=generator_id or None,
        generator_id=generator_id or ("real" if class_name == "real" else None),
        source_id_kind="cache_manifest",
    )


def read_cached_manifest_entries(
    manifest_dir: Path,
    spec: FeatureCacheSpec,
    expected_rows: int,
    label: str | None = None,
) -> dict[str, ManifestEntry]:
    path = feature_cache_manifest_path(manifest_dir, spec)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {path}")

    started = time.monotonic()
    if label is not None:
        log(f"manifest read start: modality={label} path={path}")
    entries: dict[str, ManifestEntry] = {}
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not MANIFEST_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"Malformed cache manifest columns: {path}")
        for row in reader:
            row_count += 1
            if str(row.get("status", "")).strip() != "cached":
                continue
            example = example_from_manifest_row(row)
            key = manifest_key(example.class_name, example.metadata_filename or "")
            entries[key] = ManifestEntry(key=key, example=example)

    if row_count != expected_rows:
        raise ValueError(f"Cache manifest must have {expected_rows} rows, got {row_count}: {path}")
    if label is not None:
        elapsed = time.monotonic() - started
        log(
            f"manifest read done: modality={label} rows={row_count} "
            f"cached={len(entries)} elapsed={elapsed:.1f}s"
        )
    return entries


def load_manifest_backed_examples(
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    expected_rows: int,
) -> tuple[list[VideoExample], dict[str, Any]]:
    entries_by_modality: dict[str, dict[str, ManifestEntry]] = {}
    for modality in modalities:
        entries_by_modality[modality] = read_cached_manifest_entries(
            cache_dir,
            specs[modality],
            expected_rows,
            label=modality,
        )
    common_keys = set.intersection(*(set(entries) for entries in entries_by_modality.values()))
    if not common_keys:
        raise ValueError("No examples are cached for all requested modalities.")

    first_modality = modalities[0]
    examples = [entries_by_modality[first_modality][key].example for key in sorted(common_keys)]
    summary = {
        "manifest_rows_expected": expected_rows,
        "modalities": list(modalities),
        "cached_by_modality": {
            modality: len(entries) for modality, entries in sorted(entries_by_modality.items())
        },
        "intersection_cached": len(examples),
        "class_counts": class_counts(examples),
    }
    return examples, summary


def load_manifest_backed_examples_from_cache_dirs(
    cache_dirs: Mapping[str, Path],
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    expected_rows: int,
    manifest_dirs: Mapping[str, Path] | None = None,
) -> tuple[list[VideoExample], dict[str, Any]]:
    entries_by_modality: dict[str, dict[str, ManifestEntry]] = {}
    resolved_manifest_dirs = manifest_dirs or cache_dirs
    for modality in modalities:
        entries_by_modality[modality] = read_cached_manifest_entries(
            resolved_manifest_dirs[modality],
            specs[modality],
            expected_rows,
            label=modality,
        )
    common_keys = set.intersection(*(set(entries) for entries in entries_by_modality.values()))
    if not common_keys:
        raise ValueError("No examples are cached for all requested modalities.")

    first_modality = modalities[0]
    examples = [entries_by_modality[first_modality][key].example for key in sorted(common_keys)]
    summary = {
        "manifest_rows_expected": expected_rows,
        "modalities": list(modalities),
        "cached_by_modality": {
            modality: len(entries) for modality, entries in sorted(entries_by_modality.items())
        },
        "cache_dirs_by_modality": {modality: str(cache_dirs[modality]) for modality in modalities},
        "manifest_dirs_by_modality": {
            modality: str(resolved_manifest_dirs[modality]) for modality in modalities
        },
        "intersection_cached": len(examples),
        "class_counts": class_counts(examples),
    }
    return examples, summary


def expected_intersection_count(cache_dir: Path, modalities: Sequence[str]) -> int | None:
    if cache_dir.name == "v1" and set(modalities) == V1_ALL_MODALITIES:
        return V1_ALL_EXPECTED_INTERSECTION
    return None


def reject_unexpected_intersection(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    modalities: Sequence[str],
) -> None:
    expected = expected_intersection_count(cache_dir, modalities)
    if expected is not None and len(examples) != expected:
        raise ValueError(
            f"Expected v1 all-modality cache intersection of {expected}, got {len(examples)}."
        )


def select_readonly_splits(
    examples: Sequence[VideoExample],
    balanced_total: int | None,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("`train_ratio` must be in (0.0, 1.0).")
    if val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("`val_ratio` must be in (0.0, 1.0) and leave room for test.")

    real = [example for example in examples if example.class_name == "real"]
    fake = [example for example in examples if example.class_name == "fake"]
    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(fake)

    if balanced_total is not None:
        per_class = balanced_total // 2
        if len(real) < per_class or len(fake) < per_class:
            raise ValueError(
                f"`balanced_total` exceeds cached class balance: real={len(real)} fake={len(fake)}"
            )
        real = real[:per_class]
        fake = fake[:per_class]

    train_real, val_real, test_real = split_class_examples(real, train_ratio, val_ratio)
    train_fake, val_fake, test_fake = split_class_examples(fake, train_ratio, val_ratio)
    train = [
        *[replace(example, split="train") for example in train_real],
        *[replace(example, split="train") for example in train_fake],
    ]
    val = [
        *[replace(example, split="val") for example in val_real],
        *[replace(example, split="val") for example in val_fake],
    ]
    test = [
        *[replace(example, split="test") for example in test_real],
        *[replace(example, split="test") for example in test_fake],
    ]
    if not train or not val or not test:
        raise ValueError("Selected cache split produced an empty train, val, or test split.")
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def split_class_examples(
    examples: Sequence[VideoExample],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    train_count = int(len(examples) * train_ratio)
    val_count = int(len(examples) * val_ratio)
    train = list(examples[:train_count])
    val = list(examples[train_count : train_count + val_count])
    test = list(examples[train_count + val_count :])
    return train, val, test


def selected_cache_paths(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    for example in examples:
        for modality in modalities:
            paths.append(feature_cache_item_path(cache_dir, example, specs[modality], None))
    return paths


def selected_cache_paths_by_modality(
    examples: Sequence[VideoExample],
    cache_dirs: Mapping[str, Path],
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> list[Path]:
    paths: list[Path] = []
    for example in examples:
        for modality in modalities:
            paths.append(
                feature_cache_item_path(cache_dirs[modality], example, specs[modality], None)
            )
    return paths


def reject_missing_selected_cache(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        sample = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"Selected cached examples have {len(missing)} missing .pt files. Sample:\n{sample}"
        )


def reject_missing_selected_cache_with_progress(
    paths: Sequence[Path],
    progress_every: int,
) -> None:
    started = time.monotonic()
    missing: list[Path] = []
    total = len(paths)
    log(f"cache file check start: files={total}")
    for index, path in enumerate(paths, start=1):
        if not path.is_file():
            missing.append(path)
        if progress_every > 0 and index % progress_every == 0:
            elapsed = time.monotonic() - started
            log(
                f"cache file check: checked={index}/{total} "
                f"missing={len(missing)} elapsed={elapsed:.1f}s"
            )
    if missing:
        sample = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"Selected cached examples have {len(missing)} missing .pt files. Sample:\n{sample}"
        )
    elapsed = time.monotonic() - started
    log(f"cache file check done: checked={total} missing=0 elapsed={elapsed:.1f}s")


def file_stat(path: Path) -> CacheFileStat:
    if not path.exists():
        return CacheFileStat(path=str(path), exists=False, size=None, mtime_ns=None)
    stat = path.stat()
    return CacheFileStat(
        path=str(path),
        exists=True,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def cache_stat_snapshot(paths: Sequence[Path]) -> dict[str, CacheFileStat]:
    return {str(path): file_stat(path) for path in paths}


def cache_stat_snapshot_with_progress(
    paths: Sequence[Path],
    label: str,
    progress_every: int,
) -> dict[str, CacheFileStat]:
    started = time.monotonic()
    total = len(paths)
    stats: dict[str, CacheFileStat] = {}
    log(f"{label} start: files={total}")
    for index, path in enumerate(paths, start=1):
        stats[str(path)] = file_stat(path)
        if progress_every > 0 and index % progress_every == 0:
            elapsed = time.monotonic() - started
            log(f"{label}: checked={index}/{total} elapsed={elapsed:.1f}s")
    elapsed = time.monotonic() - started
    log(f"{label} done: checked={total} elapsed={elapsed:.1f}s")
    return stats


def assert_cache_stats_unchanged(
    before: Mapping[str, CacheFileStat],
    after: Mapping[str, CacheFileStat],
) -> None:
    changed = [path for path, before_stat in before.items() if before_stat != after.get(path)]
    if changed:
        sample = "\n".join(changed[:10])
        raise RuntimeError(f"Read-only cache audit failed; changed files:\n{sample}")


def stats_payload(stats: Mapping[str, CacheFileStat]) -> dict[str, Any]:
    return {
        "count": len(stats),
        "exists": sum(1 for stat in stats.values() if stat.exists),
        "missing": sum(1 for stat in stats.values() if not stat.exists),
    }


def run_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    keys = [*RUN_ARG_DEFAULTS, "manifest_dir", "modality_manifest_dirs"]
    return {key: getattr(args, key) for key in keys if hasattr(args, key)}


def sharded_loader_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    training = config.get("training", {})
    if training is None:
        training = {}
    if not isinstance(training, Mapping):
        raise ValueError("Config `training` must be a mapping when provided.")
    sharded = training.get("sharded_loader", {})
    if sharded is None:
        sharded = {}
    if not isinstance(sharded, Mapping):
        raise ValueError("Config `training.sharded_loader` must be a mapping when provided.")
    return sharded


def bool_config(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError("Boolean sharded loader config fields must be true or false.")


def feature_cache_manifest_summary_by_modality(
    cache_dirs: Mapping[str, Path],
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> dict[str, Any]:
    return {
        modality: feature_cache_manifest_summary(
            cache_dirs[modality],
            {modality: specs[modality]},
            (modality,),
        )[modality]
        for modality in modalities
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def import_matplotlib_pyplot() -> tuple[Any | None, str | None]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional environment package.
        return None, str(exc)
    return plt, None


def split_prediction_rows(rows: Sequence[Mapping[str, str]], split: str) -> list[Mapping[str, str]]:
    return [row for row in rows if row.get("split") == split]


def float_field(row: Mapping[str, str], field: str) -> float:
    return float(row[field])


def int_field(row: Mapping[str, str], field: str) -> int:
    return int(float(row[field]))


def binary_confusion_counts(rows: Sequence[Mapping[str, str]]) -> list[list[int]]:
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        label = int_field(row, "label")
        prediction = int_field(row, "prediction")
        if label in (0, 1) and prediction in (0, 1):
            matrix[label][prediction] += 1
    return matrix


def confusion_metrics_from_matrix(matrix: Sequence[Sequence[int]]) -> dict[str, float | int]:
    true_negative = int(matrix[0][0])
    false_positive = int(matrix[0][1])
    false_negative = int(matrix[1][0])
    true_positive = int(matrix[1][1])
    real_total = true_negative + false_positive
    fake_total = true_positive + false_negative
    total = real_total + fake_total
    real_recall = true_negative / real_total if real_total else 0.0
    fake_recall = true_positive / fake_total if fake_total else 0.0
    balanced_accuracy = (real_recall + fake_recall) / 2.0
    accuracy = (true_positive + true_negative) / total if total else 0.0
    predicted_real = true_negative + false_negative
    predicted_fake = true_positive + false_positive
    return {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "real_total": real_total,
        "fake_total": fake_total,
        "predicted_real": predicted_real,
        "predicted_fake": predicted_fake,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "false_positive_rate": 1.0 - real_recall if real_total else 0.0,
        "false_negative_rate": 1.0 - fake_recall if fake_total else 0.0,
        "predicted_fake_rate": predicted_fake / total if total else 0.0,
    }


def prediction_metric_summary(
    prediction_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for split in ("train", "val", "test"):
        matrix = binary_confusion_counts(split_prediction_rows(prediction_rows, split))
        summary[split] = confusion_metrics_from_matrix(matrix)
    return summary


def binary_curve_points(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    pairs = sorted(
        ((float_field(row, "probability"), int_field(row, "label")) for row in rows),
        reverse=True,
    )
    positive = sum(1 for _, label in pairs if label == 1)
    negative = len(pairs) - positive
    if positive == 0 or negative == 0:
        return [], [], [], []

    tp = 0
    fp = 0
    roc_fpr = [0.0]
    roc_tpr = [0.0]
    pr_recall = [0.0]
    pr_precision = [1.0]
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        roc_fpr.append(fp / negative)
        roc_tpr.append(tp / positive)
        pr_recall.append(tp / positive)
        pr_precision.append(tp / max(tp + fp, 1))
    return roc_fpr, roc_tpr, pr_recall, pr_precision


def plot_learning_curves(plt: Any, metrics_csv: Path, output_path: Path) -> str | None:
    rows = read_csv_rows(metrics_csv)
    if not rows:
        return "metrics.csv has no rows"

    epochs = [int_field(row, "epoch") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [float_field(row, "train_loss") for row in rows], label="train loss")
    axes[0].plot(epochs, [float_field(row, "val_loss") for row in rows], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, [float_field(row, "train_accuracy") for row in rows], label="train acc")
    axes[1].plot(epochs, [float_field(row, "val_accuracy") for row in rows], label="val acc")
    if "train_f1" in rows[0] and "val_f1" in rows[0]:
        axes[1].plot(epochs, [float_field(row, "train_f1") for row in rows], label="train f1")
        axes[1].plot(epochs, [float_field(row, "val_f1") for row in rows], label="val f1")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return None


def plot_confusion_matrices(
    plt: Any,
    prediction_rows: Sequence[Mapping[str, str]],
    output_path: Path,
) -> str | None:
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for axis, split in zip(axes, splits):
        rows = split_prediction_rows(prediction_rows, split)
        matrix = binary_confusion_counts(rows)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(split)
        axis.set_xticks([0, 1], ["pred real", "pred fake"])
        axis.set_yticks([0, 1], ["real", "fake"])
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                axis.text(column_index, row_index, str(value), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return None


def plot_class_balance_metrics(
    plt: Any,
    prediction_rows: Sequence[Mapping[str, str]],
    output_path: Path,
) -> str | None:
    summary = prediction_metric_summary(prediction_rows)
    splits = ["train", "val", "test"]
    if not any(summary[split]["real_total"] or summary[split]["fake_total"] for split in splits):
        return "predictions.csv has no class rows"

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for axis, split in zip(axes, splits):
        item = summary[split]
        real_total = int(item["real_total"])
        fake_total = int(item["fake_total"])
        real_correct = float(item["real_recall"])
        real_error = float(item["false_positive_rate"])
        fake_correct = float(item["fake_recall"])
        fake_error = float(item["false_negative_rate"])

        axis.bar([0], [real_correct], color="#4c78a8", label="correct")
        axis.bar([0], [real_error], bottom=[real_correct], color="#e45756", label="error")
        axis.bar([1], [fake_correct], color="#4c78a8")
        axis.bar([1], [fake_error], bottom=[fake_correct], color="#e45756")

        axis.text(
            0,
            min(real_correct / 2.0, 0.95),
            f"{real_correct * 100:.1f}%\n{int(item['true_negative'])}/{real_total}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
        axis.text(
            1,
            min(fake_correct / 2.0, 0.95),
            f"{fake_correct * 100:.1f}%\n{int(item['true_positive'])}/{fake_total}",
            ha="center",
            va="center",
            color="white",
            fontsize=9,
        )
        if real_error >= 0.035:
            axis.text(
                0,
                real_correct + real_error / 2.0,
                f"FP {real_error * 100:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
        if fake_error >= 0.035:
            axis.text(
                1,
                fake_correct + fake_error / 2.0,
                f"FN {fake_error * 100:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
        axis.set_title(
            f"{split}\nBA {float(item['balanced_accuracy']) * 100:.1f}% | "
            f"pred fake {float(item['predicted_fake_rate']) * 100:.1f}%"
        )
        axis.set_xticks([0, 1], ["actual real", "actual fake"])
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("share of each actual class")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return None


def plot_roc_pr_curves(
    plt: Any,
    prediction_rows: Sequence[Mapping[str, str]],
    output_path: Path,
) -> str | None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plotted = False
    for split in ("train", "val", "test"):
        rows = split_prediction_rows(prediction_rows, split)
        fpr, tpr, recall, precision = binary_curve_points(rows)
        if not fpr:
            continue
        axes[0].plot(fpr, tpr, label=split)
        axes[1].plot(recall, precision, label=split)
        plotted = True
    if not plotted:
        plt.close(fig)
        return "predictions.csv lacks both classes for ROC/PR"
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[1].set_title("Precision / Recall")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    for axis in axes:
        axis.legend()
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return None


def plot_probability_histograms(
    plt: Any,
    prediction_rows: Sequence[Mapping[str, str]],
    output_path: Path,
) -> str | None:
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    plotted = False
    for axis, split in zip(axes, splits):
        rows = split_prediction_rows(prediction_rows, split)
        real = [float_field(row, "probability") for row in rows if int_field(row, "label") == 0]
        fake = [float_field(row, "probability") for row in rows if int_field(row, "label") == 1]
        if real:
            axis.hist(real, bins=20, alpha=0.65, label="real")
            plotted = True
        if fake:
            axis.hist(fake, bins=20, alpha=0.65, label="fake")
            plotted = True
        axis.set_title(split)
        axis.set_xlim(0, 1)
        axis.set_xlabel("fake probability")
        axis.legend()
    axes[0].set_ylabel("count")
    if not plotted:
        plt.close(fig)
        return "predictions.csv has no probability rows"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return None


def plot_record(path: Path, reason: str | None) -> dict[str, Any]:
    return {
        "path": str(path),
        "produced": reason is None,
        "skipped_reason": reason,
    }


def write_core_plots(training_output_dir: Path, run_output_dir: Path) -> dict[str, Any]:
    log(f"plots start: source={training_output_dir}")
    plots_dir = run_output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary_path = plots_dir / "plots_summary.json"
    plot_summary: dict[str, Any] = {
        "plots_dir": str(plots_dir),
        "metrics_csv": str(training_output_dir / "metrics.csv"),
        "predictions_csv": str(training_output_dir / "predictions.csv"),
        "plots": {},
    }
    plt, import_error = import_matplotlib_pyplot()
    if plt is None:
        log(f"plots skipped: matplotlib import failed: {import_error}")
        for name in (
            "learning_curves",
            "confusion_matrices",
            "class_balance_metrics",
            "roc_pr_curves",
            "probability_histograms",
        ):
            plot_summary["plots"][name] = plot_record(plots_dir / f"{name}.png", import_error)
        write_json(summary_path, plot_summary)
        return plot_summary

    metrics_csv = training_output_dir / "metrics.csv"
    predictions_csv = training_output_dir / "predictions.csv"
    prediction_rows = read_csv_rows(predictions_csv) if predictions_csv.is_file() else []
    learning_curves_path = plots_dir / "learning_curves.png"
    confusion_path = plots_dir / "confusion_matrices.png"
    class_balance_path = plots_dir / "class_balance_metrics.png"
    roc_pr_path = plots_dir / "roc_pr_curves.png"
    histograms_path = plots_dir / "probability_histograms.png"
    if prediction_rows:
        write_json(
            plots_dir / "prediction_metrics_summary.json",
            prediction_metric_summary(prediction_rows),
        )

    try:
        reason = (
            plot_learning_curves(plt, metrics_csv, learning_curves_path)
            if metrics_csv.is_file()
            else "missing metrics.csv"
        )
    except Exception as exc:  # Plotting must not fail training.
        reason = str(exc)
    plot_summary["plots"]["learning_curves"] = plot_record(learning_curves_path, reason)

    try:
        reason = (
            plot_confusion_matrices(plt, prediction_rows, confusion_path)
            if prediction_rows
            else "missing predictions.csv"
        )
    except Exception as exc:
        reason = str(exc)
    plot_summary["plots"]["confusion_matrices"] = plot_record(confusion_path, reason)

    try:
        reason = (
            plot_class_balance_metrics(plt, prediction_rows, class_balance_path)
            if prediction_rows
            else "missing predictions.csv"
        )
    except Exception as exc:
        reason = str(exc)
    plot_summary["plots"]["class_balance_metrics"] = plot_record(class_balance_path, reason)

    try:
        reason = (
            plot_roc_pr_curves(plt, prediction_rows, roc_pr_path)
            if prediction_rows
            else "missing predictions.csv"
        )
    except Exception as exc:
        reason = str(exc)
    plot_summary["plots"]["roc_pr_curves"] = plot_record(roc_pr_path, reason)

    try:
        reason = (
            plot_probability_histograms(plt, prediction_rows, histograms_path)
            if prediction_rows
            else "missing predictions.csv"
        )
    except Exception as exc:
        reason = str(exc)
    plot_summary["plots"]["probability_histograms"] = plot_record(histograms_path, reason)

    write_json(summary_path, plot_summary)
    produced = sum(1 for item in plot_summary["plots"].values() if item["produced"])
    log(f"plots done: produced={produced}/{len(plot_summary['plots'])} summary={summary_path}")
    return plot_summary


def main() -> None:
    cli_args = parse_args()
    log(f"readonly cache run start: config={cli_args.config}")
    config = load_pipeline_yaml(cli_args.config)
    args = resolve_run_args(config, cli_args)
    if args.device is not None:
        config["device"] = args.device

    cache_dir = Path(args.cache_dir)
    dataset_root = Path(args.dataset_root)
    output_base = Path(args.output_dir)
    sharded_cache_dir = None if args.sharded_cache_dir is None else Path(args.sharded_cache_dir)
    modalities = tuple(args.modalities)
    cache_dirs = resolve_cache_dirs_by_modality(
        config,
        cache_dir,
        modalities,
        cli_args.modality_cache_dir,
    )
    manifest_dirs = resolve_manifest_dirs_by_modality(
        config,
        cache_dir,
        args.manifest_dir,
        modalities,
        cli_args.modality_manifest_dir,
    )
    for modality_cache_dir in set(cache_dirs.values()):
        reject_output_inside_inputs(output_base, modality_cache_dir, dataset_root)
    for modality_manifest_dir in set(manifest_dirs.values()):
        reject_output_inside_inputs(output_base, modality_manifest_dir, dataset_root)
    log(
        f"readonly cache run resolved: cache_dir={cache_dir} "
        f"sharded_cache_dir={sharded_cache_dir} dataset_root={dataset_root} output_base={output_base}"
    )

    log(
        "cache dirs by modality: "
        + ", ".join(f"{modality}={cache_dirs[modality]}" for modality in modalities)
    )
    log(
        "manifest dirs by modality: "
        + ", ".join(f"{modality}={manifest_dirs[modality]}" for modality in modalities)
    )
    log(
        f"selection start: modalities={','.join(modalities)} "
        f"balanced_total={args.balanced_total if args.balanced_total is not None else 'full'} "
        f"train_ratio={args.train_ratio} val_ratio={args.val_ratio}"
    )
    specs = build_feature_cache_specs(config, modalities)
    examples, manifest_summary = load_manifest_backed_examples_from_cache_dirs(
        cache_dirs=cache_dirs,
        manifest_dirs=manifest_dirs,
        specs=specs,
        modalities=modalities,
        expected_rows=cli_args.expected_manifest_rows,
    )
    reject_unexpected_intersection(examples, cache_dir, modalities)
    log(f"selection intersection: cached={len(examples)} class_counts={class_counts(examples)}")
    train_examples, val_examples, test_examples = select_readonly_splits(
        examples=examples,
        balanced_total=args.balanced_total,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    log(
        f"split done: train={len(train_examples)} {class_counts(train_examples)} "
        f"val={len(val_examples)} {class_counts(val_examples)} "
        f"test={len(test_examples)} {class_counts(test_examples)}"
    )
    selected_examples = [*train_examples, *val_examples, *test_examples]
    shard_validation_summary = None
    if sharded_cache_dir is not None:
        sharded_config = sharded_loader_section(config)
        validate_before_train = bool_config(
            sharded_config.get("validate_before_train"),
            True,
        )
        allow_legacy_shards = bool_config(
            sharded_config.get("allow_legacy_shards"),
            False,
        )
        if validate_before_train:
            log(f"sharded cache validation start: {sharded_cache_dir}")
            shard_validation_summary = validate_readonly_cache_shards(
                sharded_cache_dir=sharded_cache_dir,
                dataset_root=dataset_root,
                selected_examples=selected_examples,
                allow_legacy_shards=allow_legacy_shards,
                check_payloads=False,
            )
            log(
                "sharded cache validation done: "
                f"schema_version={shard_validation_summary['schema_version']} "
                f"row_mapping_errors={shard_validation_summary['row_mapping_errors']} "
                f"duplicate_keys={shard_validation_summary['duplicate_keys']} "
                f"single_label_shards={shard_validation_summary['single_label_shards']}"
            )
        else:
            log("sharded cache validation skipped: validate_before_train=false")
    selected_paths = selected_cache_paths_by_modality(
        selected_examples, cache_dirs, specs, modalities
    )
    if cli_args.check_cache_files:
        reject_missing_selected_cache_with_progress(selected_paths, cli_args.progress_every)
    else:
        log(f"cache file check skipped: files={len(selected_paths)} opt_in=--check-cache-files")

    manifest_paths = [
        feature_cache_manifest_path(manifest_dirs[modality], specs[modality])
        for modality in modalities
    ]
    audit_paths = [*manifest_paths, *selected_paths]
    before_stats = (
        cache_stat_snapshot_with_progress(
            audit_paths,
            "readonly audit before",
            cli_args.progress_every,
        )
        if cli_args.readonly_audit
        else {}
    )
    if not cli_args.readonly_audit:
        log(f"readonly audit skipped: files={len(audit_paths)} opt_in=--readonly-audit")

    output_dir = output_base / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    reject_output_inside_inputs(output_dir, cache_dir, dataset_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    log(f"output prepared: {output_dir}")
    write_dataset_manifest(selected_examples, output_dir / "manifest.csv")
    log("run metadata write start")
    write_json(
        output_dir / "run_config.json",
        {
            "mode": "readonly_cached_training",
            "config_path": str(cli_args.config),
            "resolved_training_run": run_config_payload(args),
            "cache_dir": str(cache_dir),
            "cache_dirs_by_modality": {
                modality: str(cache_dirs[modality]) for modality in modalities
            },
            "manifest_dirs_by_modality": {
                modality: str(manifest_dirs[modality]) for modality in modalities
            },
            "sharded_cache_dir": None if sharded_cache_dir is None else str(sharded_cache_dir),
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "modalities": list(modalities),
            "spec_ids": {modality: feature_cache_spec_id(spec) for modality, spec in specs.items()},
            "balanced_total": args.balanced_total,
            "train_counts": class_counts(train_examples),
            "val_counts": class_counts(val_examples),
            "test_counts": class_counts(test_examples),
            "manifest_selection": manifest_summary,
            "cache_manifest_summary": feature_cache_manifest_summary_by_modality(
                manifest_dirs,
                specs,
                modalities,
            ),
            "dataset_metadata": video_metadata_summary(selected_examples),
            "readonly_audit_before": stats_payload(before_stats) if before_stats else None,
            "sharded_cache_validation": shard_validation_summary,
        },
    )
    log(f"run metadata write done: {output_dir / 'run_config.json'}")

    training_output_dir = output_dir / f"train_{len(train_examples):05d}" / "_".join(modalities)
    log(f"training start: output_dir={training_output_dir}")
    summary = run_training_round(
        args=args,
        config=config,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        dataset_root=dataset_root,
        output_dir=training_output_dir,
        warm_start_checkpoint=None,
        sharded_cache_dir=sharded_cache_dir,
        all_cache_examples=examples,
    )
    log(f"training done: output_dir={training_output_dir}")
    plots_summary = write_core_plots(training_output_dir, output_dir)

    audit_summary = None
    if before_stats:
        after_stats = cache_stat_snapshot_with_progress(
            audit_paths,
            "readonly audit after",
            cli_args.progress_every,
        )
        assert_cache_stats_unchanged(before_stats, after_stats)
        audit_summary = {
            "before": stats_payload(before_stats),
            "after": stats_payload(after_stats),
        }
        write_json(output_dir / "readonly_audit.json", audit_summary)
        log(f"readonly audit passed: {output_dir / 'readonly_audit.json'}")
    write_json(
        output_dir / "summary.json",
        {
            "training_summary": summary,
            "plots_summary": plots_summary,
            "readonly_audit": audit_summary,
            "sharded_cache_validation": shard_validation_summary,
        },
    )
    log(f"readonly cache run done: summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
