from __future__ import annotations

import argparse
import csv
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

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_smoke_cache_v1_rgb_1k.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "readonly_smoke_runs"
DEFAULT_EXPECTED_MANIFEST_ROWS = 13768
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small smoke run from read-only cached feature manifests."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--balanced-total", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--expected-manifest-rows", type=int, default=DEFAULT_EXPECTED_MANIFEST_ROWS)
    parser.add_argument(
        "--no-readonly-audit",
        action="store_true",
        help="Skip before/after cache stat comparison. Cache is still never written by this script.",
    )
    return parser.parse_args()


def resolve_run_args(config: Mapping[str, Any], cli_args: argparse.Namespace) -> argparse.Namespace:
    run_config = training_run_section(config)
    values = dict(RUN_ARG_DEFAULTS)
    for key, value in run_config.items():
        if key in values:
            values[key] = value

    for key in ("dataset_root", "cache_dir", "output_dir", "clip_cache_dir"):
        if values.get(key) is not None:
            values[key] = Path(values[key])

    for key in ("modalities", "round_targets", "occlusion_splits"):
        if values.get(key) is not None and not isinstance(values[key], tuple):
            values[key] = list(values[key])

    if cli_args.cache_dir is not None:
        values["cache_dir"] = cli_args.cache_dir
    if cli_args.output_dir is not None:
        values["output_dir"] = cli_args.output_dir
    if cli_args.modalities is not None:
        values["modalities"] = list(cli_args.modalities)
    if cli_args.balanced_total is not None:
        values["balanced_total"] = cli_args.balanced_total
    if cli_args.epochs is not None:
        values["epochs"] = cli_args.epochs
    if cli_args.seed is not None:
        values["seed"] = cli_args.seed

    if values["cache_dir"] is None:
        raise ValueError("Read-only smoke requires `training.run.cache_dir` or `--cache-dir`.")
    if values["balanced_total"] is None:
        values["balanced_total"] = 1000
    if values["modalities"] is None:
        configured = config.get("modalities")
        if not isinstance(configured, list):
            raise ValueError("Config `modalities` must be a YAML list when no override is given.")
        values["modalities"] = list(configured)
    values["output_dir"] = cli_args.output_dir or DEFAULT_OUTPUT_DIR

    return argparse.Namespace(**values)


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
    cache_dir: Path,
    spec: FeatureCacheSpec,
    expected_rows: int,
) -> dict[str, ManifestEntry]:
    path = feature_cache_manifest_path(cache_dir, spec)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache manifest: {path}")

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
        raise ValueError(
            f"Cache manifest must have {expected_rows} rows, got {row_count}: {path}"
        )
    return entries


def load_manifest_backed_examples(
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    expected_rows: int,
) -> tuple[list[VideoExample], dict[str, Any]]:
    entries_by_modality = {
        modality: read_cached_manifest_entries(cache_dir, specs[modality], expected_rows)
        for modality in modalities
    }
    common_keys = set.intersection(*(set(entries) for entries in entries_by_modality.values()))
    if not common_keys:
        raise ValueError("No examples are cached for all requested modalities.")

    first_modality = modalities[0]
    examples = [
        entries_by_modality[first_modality][key].example
        for key in sorted(common_keys)
    ]
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


def select_readonly_splits(
    examples: Sequence[VideoExample],
    balanced_total: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    if balanced_total <= 0 or balanced_total % 2 != 0:
        raise ValueError("`balanced_total` must be a positive even integer.")
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("`train_ratio` must be in (0.0, 1.0).")
    if val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("`val_ratio` must be in (0.0, 1.0) and leave room for test.")

    real = [example for example in examples if example.class_name == "real"]
    fake = [example for example in examples if example.class_name == "fake"]
    per_class = balanced_total // 2
    if len(real) < per_class or len(fake) < per_class:
        raise ValueError(
            f"`balanced_total` exceeds cached class balance: real={len(real)} fake={len(fake)}"
        )

    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(fake)
    real = real[:per_class]
    fake = fake[:per_class]
    train_per_class = int(per_class * train_ratio)
    val_per_class = int(per_class * val_ratio)
    if train_per_class <= 0 or val_per_class <= 0:
        raise ValueError("`balanced_total` too small for non-empty train/val splits.")

    train = [
        *[replace(example, split="train") for example in real[:train_per_class]],
        *[replace(example, split="train") for example in fake[:train_per_class]],
    ]
    val_start = train_per_class
    val_end = train_per_class + val_per_class
    val = [
        *[replace(example, split="val") for example in real[val_start:val_end]],
        *[replace(example, split="val") for example in fake[val_start:val_end]],
    ]
    test = [
        *[replace(example, split="test") for example in real[val_end:]],
        *[replace(example, split="test") for example in fake[val_end:]],
    ]
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
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


def reject_missing_selected_cache(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        sample = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"Selected cached examples have {len(missing)} missing .pt files. Sample:\n{sample}"
        )


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


def assert_cache_stats_unchanged(
    before: Mapping[str, CacheFileStat],
    after: Mapping[str, CacheFileStat],
) -> None:
    changed = [
        path
        for path, before_stat in before.items()
        if before_stat != after.get(path)
    ]
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
    return {key: getattr(args, key) for key in RUN_ARG_DEFAULTS if hasattr(args, key)}


def main() -> None:
    cli_args = parse_args()
    config = load_pipeline_yaml(cli_args.config)
    args = resolve_run_args(config, cli_args)
    if args.device is not None:
        config["device"] = args.device

    cache_dir = Path(args.cache_dir)
    dataset_root = Path(args.dataset_root)
    output_base = Path(args.output_dir)
    reject_output_inside_inputs(output_base, cache_dir, dataset_root)

    modalities = tuple(args.modalities)
    specs = build_feature_cache_specs(config, modalities)
    examples, manifest_summary = load_manifest_backed_examples(
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        expected_rows=cli_args.expected_manifest_rows,
    )
    train_examples, val_examples, test_examples = select_readonly_splits(
        examples=examples,
        balanced_total=int(args.balanced_total),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    selected_examples = [*train_examples, *val_examples, *test_examples]
    selected_paths = selected_cache_paths(selected_examples, cache_dir, specs, modalities)
    reject_missing_selected_cache(selected_paths)

    manifest_paths = [
        feature_cache_manifest_path(cache_dir, specs[modality]) for modality in modalities
    ]
    audit_paths = [*manifest_paths, *selected_paths]
    before_stats = {} if cli_args.no_readonly_audit else cache_stat_snapshot(audit_paths)

    output_dir = output_base / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    reject_output_inside_inputs(output_dir, cache_dir, dataset_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_dataset_manifest(selected_examples, output_dir / "manifest.csv")
    write_json(
        output_dir / "run_config.json",
        {
            "mode": "readonly_cached_smoke",
            "config_path": str(cli_args.config),
            "resolved_training_run": run_config_payload(args),
            "cache_dir": str(cache_dir),
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "modalities": list(modalities),
            "spec_ids": {modality: feature_cache_spec_id(spec) for modality, spec in specs.items()},
            "balanced_total": args.balanced_total,
            "train_counts": class_counts(train_examples),
            "val_counts": class_counts(val_examples),
            "test_counts": class_counts(test_examples),
            "manifest_selection": manifest_summary,
            "cache_manifest_summary": feature_cache_manifest_summary(cache_dir, specs, modalities),
            "dataset_metadata": video_metadata_summary(selected_examples),
            "readonly_audit_before": stats_payload(before_stats) if before_stats else None,
        },
    )

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
        output_dir=output_dir / f"train_{len(train_examples):05d}" / "_".join(modalities),
        warm_start_checkpoint=None,
    )
    write_json(output_dir / "summary.json", {"training_summary": summary})

    if before_stats:
        after_stats = cache_stat_snapshot(audit_paths)
        assert_cache_stats_unchanged(before_stats, after_stats)
        write_json(
            output_dir / "readonly_audit.json",
            {
                "before": stats_payload(before_stats),
                "after": stats_payload(after_stats),
            },
        )


if __name__ == "__main__":
    main()
