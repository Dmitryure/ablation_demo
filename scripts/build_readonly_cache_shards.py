from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoExample
from feature_cache import (
    FeatureCacheSpec,
    build_feature_cache_specs,
    cache_example_key,
    feature_cache_item_path,
    feature_cache_manifest_path,
    feature_cache_spec_id,
    load_feature_cache_item,
    metadata_filename_for_example,
)
from pipeline import load_pipeline_yaml
from scripts.run_iterative_cached_ablation import (
    class_counts,
    training_run_section,
    write_json,
)
from scripts.run_readonly_cached_smoke import (
    DEFAULT_EXPECTED_MANIFEST_ROWS,
    ManifestEntry,
    read_cached_manifest_entries,
    reject_output_inside_inputs,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_final_cache_v1.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "shards" / "v1_all_mixed_v2"
DEFAULT_PROGRESS_EVERY = 1
DEFAULT_WORKERS = 8
SHARD_VERSION = 2
DEFAULT_ORDER_STRATEGY = "stratified_shards"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build read-only packed training shards from an existing feature cache."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=None)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--order-strategy",
        choices=("stratified_shards", "sorted"),
        default=DEFAULT_ORDER_STRATEGY,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--expected-manifest-rows", type=int, default=DEFAULT_EXPECTED_MANIFEST_ROWS
    )
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    return parser.parse_args()


def resolve_cache_dir(config: dict[str, Any], cli_cache_dir: Path | None) -> Path:
    if cli_cache_dir is not None:
        return cli_cache_dir
    run_config = training_run_section(config)
    cache_dir = run_config.get("cache_dir")
    if cache_dir is None:
        raise ValueError("Shard build requires `training.run.cache_dir` or `--cache-dir`.")
    return Path(cache_dir)


def resolve_manifest_dir(config: dict[str, Any], cli_manifest_dir: Path | None) -> Path | None:
    if cli_manifest_dir is not None:
        return cli_manifest_dir
    run_config = training_run_section(config)
    manifest_dir = run_config.get("manifest_dir")
    return None if manifest_dir is None else Path(manifest_dir)


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


def config_modality_cache_dirs(config: dict[str, Any]) -> dict[str, Path]:
    run_config = training_run_section(config)
    value = run_config.get("modality_cache_dirs") or {}
    if not isinstance(value, dict):
        raise ValueError("`training.run.modality_cache_dirs` must be a mapping.")
    return {str(modality): Path(path) for modality, path in value.items()}


def config_modality_manifest_dirs(config: dict[str, Any]) -> dict[str, Path]:
    run_config = training_run_section(config)
    value = run_config.get("modality_manifest_dirs") or {}
    if not isinstance(value, dict):
        raise ValueError("`training.run.modality_manifest_dirs` must be a mapping.")
    return {str(modality): Path(path) for modality, path in value.items()}


def resolve_cache_dirs_by_modality(
    config: dict[str, Any],
    base_cache_dir: Path,
    modalities: tuple[str, ...],
    cli_overrides: list[str],
) -> dict[str, Path]:
    overrides = {
        **config_modality_cache_dirs(config),
        **parse_modality_cache_dir_overrides(cli_overrides),
    }
    return {modality: overrides.get(modality, base_cache_dir) for modality in modalities}


def resolve_manifest_dirs_by_modality(
    config: dict[str, Any],
    base_cache_dir: Path,
    base_manifest_dir: Path | None,
    modalities: tuple[str, ...],
    cli_overrides: list[str],
) -> dict[str, Path]:
    base_dir = base_manifest_dir or base_cache_dir
    overrides = {
        **config_modality_manifest_dirs(config),
        **parse_modality_manifest_dir_overrides(cli_overrides),
    }
    return {modality: overrides.get(modality, base_dir) for modality in modalities}


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    run_config = training_run_section(config)
    return Path(run_config.get("dataset_root", "/mnt/d/final_dataset"))


def resolve_modalities(config: dict[str, Any], cli_modalities: list[str] | None) -> tuple[str, ...]:
    if cli_modalities is not None:
        return tuple(cli_modalities)
    run_config = training_run_section(config)
    modalities = run_config.get("modalities") or config.get("modalities")
    if not isinstance(modalities, list):
        raise ValueError("Config must provide a modality list.")
    return tuple(str(modality) for modality in modalities)


def reject_existing_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Shard output directory already exists and is not empty: {output_dir}"
        )


def source_manifest_stats(
    manifest_dirs: dict[str, Path],
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        path = feature_cache_manifest_path(manifest_dirs[modality], specs[modality])
        stat = path.stat()
        stats[modality] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return stats


def load_manifest_backed_examples_from_cache_dirs(
    cache_dirs: dict[str, Path],
    manifest_dirs: dict[str, Path],
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
    expected_rows: int,
) -> tuple[list[VideoExample], dict[str, Any]]:
    entries_by_modality: dict[str, dict[str, ManifestEntry]] = {}
    for modality in modalities:
        entries_by_modality[modality] = read_cached_manifest_entries(
            manifest_dirs[modality],
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
            modality: str(manifest_dirs[modality]) for modality in modalities
        },
        "intersection_cached": len(examples),
        "class_counts": class_counts(examples),
    }
    return examples, summary


def chunk_examples(
    examples: list[VideoExample],
    shard_size: int,
) -> list[list[VideoExample]]:
    if shard_size <= 0:
        raise ValueError("`shard_size` must be positive.")
    return [examples[index : index + shard_size] for index in range(0, len(examples), shard_size)]


def default_shard_size(modalities: tuple[str, ...]) -> int:
    return 128 if "fau" in modalities else 512


def stratified_shard_order(
    examples: list[VideoExample],
    shard_size: int,
    seed: int,
) -> list[VideoExample]:
    if shard_size <= 0:
        raise ValueError("`shard_size` must be positive.")
    buckets: dict[str, list[VideoExample]] = {}
    for example in examples:
        buckets.setdefault(example.class_name, []).append(example)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    ordered: list[VideoExample] = []
    active_classes = sorted(buckets)
    while any(buckets[class_name] for class_name in active_classes):
        remaining_total = sum(len(buckets[class_name]) for class_name in active_classes)
        capacity = min(shard_size, remaining_total)
        allocation = stratified_shard_allocation(
            {class_name: len(buckets[class_name]) for class_name in active_classes},
            capacity,
        )
        shard_examples: list[VideoExample] = []
        for class_name in active_classes:
            count = allocation.get(class_name, 0)
            shard_examples.extend(buckets[class_name][:count])
            del buckets[class_name][:count]
        rng.shuffle(shard_examples)
        ordered.extend(shard_examples)
    return ordered


def stratified_shard_allocation(
    remaining_by_class: dict[str, int],
    capacity: int,
) -> dict[str, int]:
    if capacity <= 0:
        return {}
    active = {class_name: count for class_name, count in remaining_by_class.items() if count > 0}
    total = sum(active.values())
    if total <= 0:
        return {}
    raw = {class_name: (count * capacity) / total for class_name, count in active.items()}
    allocation = {
        class_name: min(active[class_name], int(raw[class_name])) for class_name in active
    }
    remainder = capacity - sum(allocation.values())
    by_fraction = sorted(
        active,
        key=lambda class_name: (raw[class_name] - int(raw[class_name]), active[class_name]),
        reverse=True,
    )
    while remainder > 0:
        changed = False
        for class_name in by_fraction:
            if allocation[class_name] >= active[class_name]:
                continue
            allocation[class_name] += 1
            remainder -= 1
            changed = True
            if remainder == 0:
                break
        if not changed:
            break
    if capacity >= len(active):
        for class_name in sorted(active):
            if allocation[class_name] > 0:
                continue
            donor = max(allocation, key=lambda name: allocation[name])
            if allocation[donor] <= 1:
                continue
            allocation[donor] -= 1
            allocation[class_name] += 1
    return allocation


def example_payload(example: VideoExample) -> dict[str, Any]:
    return {
        "path": str(example.path),
        "label": example.label,
        "class_name": example.class_name,
        "source_id": example.source_id,
        "split": example.split,
        "metadata_filename": example.metadata_filename,
        "identity_id": example.identity_id,
        "generator_id": example.generator_id,
        "source_id_kind": example.source_id_kind,
        "age_bin": example.age_bin,
        "gender": example.gender,
        "ethnicity": example.ethnicity,
        "emotion": example.emotion,
    }


def load_example_features(
    example: VideoExample,
    cache_dirs: dict[str, Path],
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
    dataset_root: Path,
) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for modality in modalities:
        spec = specs[modality]
        item = load_feature_cache_item(
            cache_dirs[modality],
            example,
            spec,
            dataset_root=dataset_root,
        )
        if item is None:
            path = feature_cache_item_path(
                cache_dirs[modality],
                example,
                spec,
                dataset_root=dataset_root,
            )
            raise FileNotFoundError(f"Missing or invalid cached feature: {path}")
        features.update(item)
    return features


def stack_shard_features(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("Cannot build an empty shard.")
    keys = sorted(items[0])
    stacked: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in items]
        stacked[key] = torch.stack(values, dim=0)
    return stacked


def tensor_summary(features: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(features.items())
    }


def load_shard_feature_items(
    shard_examples: list[VideoExample],
    cache_dirs: dict[str, Path],
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
    dataset_root: Path,
    workers: int,
) -> list[dict[str, torch.Tensor]]:
    loader = partial(
        load_example_features,
        cache_dirs=cache_dirs,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
    )
    if workers <= 1:
        return [loader(example) for example in shard_examples]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(loader, shard_examples))


def write_shard_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        raise FileExistsError(f"Temporary shard already exists: {tmp_path}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def build_shard_payload(
    shard_examples: list[VideoExample],
    cache_dirs: dict[str, Path],
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
    dataset_root: Path,
    shard_index: int,
    workers: int,
) -> dict[str, Any]:
    feature_items = load_shard_feature_items(
        shard_examples=shard_examples,
        cache_dirs=cache_dirs,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
        workers=workers,
    )
    features = stack_shard_features(feature_items)
    return {
        "version": SHARD_VERSION,
        "shard_index": shard_index,
        "modalities": list(modalities),
        "examples": [example_payload(example) for example in shard_examples],
        "class_counts": class_counts(shard_examples),
        "feature_summary": tensor_summary(features),
        "features": features,
    }


def build_shards(
    examples: list[VideoExample],
    cache_dirs: dict[str, Path],
    output_dir: Path,
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
    dataset_root: Path,
    shard_size: int,
    workers: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    shards_dir = output_dir / "shards"
    chunks = chunk_examples(examples, shard_size)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for shard_index, shard_examples in enumerate(chunks):
        shard_name = f"shard_{shard_index:06d}.pt"
        shard_path = shards_dir / shard_name
        if shard_path.exists():
            raise FileExistsError(f"Shard already exists: {shard_path}")
        payload = build_shard_payload(
            shard_examples=shard_examples,
            cache_dirs=cache_dirs,
            specs=specs,
            modalities=modalities,
            dataset_root=dataset_root,
            shard_index=shard_index,
            workers=workers,
        )
        write_shard_atomic(shard_path, payload)
        stat = shard_path.stat()
        first = shard_examples[0]
        last = shard_examples[-1]
        records.append(
            {
                "path": str(shard_path.relative_to(output_dir)),
                "example_count": len(shard_examples),
                "size": stat.st_size,
                "first_key": f"{first.class_name}/{metadata_filename_for_example(first, dataset_root)}",
                "last_key": f"{last.class_name}/{metadata_filename_for_example(last, dataset_root)}",
                "class_counts": class_counts(shard_examples),
                "feature_summary": payload["feature_summary"],
            }
        )
        if progress_every > 0 and (
            len(records) == len(chunks) or len(records) % progress_every == 0
        ):
            elapsed = time.monotonic() - started
            done = sum(int(record["example_count"]) for record in records)
            rate = 0.0 if elapsed <= 0.0 else done / elapsed
            log(
                f"shard build: shards={len(records)}/{len(chunks)} workers={workers} "
                f"examples={done}/{len(examples)} elapsed={elapsed:.1f}s examples_per_s={rate:.2f}"
            )
    return records


def shard_example_locations(
    examples: list[VideoExample],
    shard_size: int,
    dataset_root: Path,
) -> dict[str, dict[str, Any]]:
    locations: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        filename = metadata_filename_for_example(example, dataset_root)
        locations[cache_example_key(example, dataset_root)] = {
            "shard_index": index // shard_size,
            "row_index": index % shard_size,
            "class_name": example.class_name,
            "filename": filename,
            "label": int(example.label),
        }
    return locations


def apply_order_strategy(
    examples: list[VideoExample],
    shard_size: int,
    seed: int,
    order_strategy: str,
) -> list[VideoExample]:
    if order_strategy == "sorted":
        return list(examples)
    if order_strategy == "stratified_shards":
        return stratified_shard_order(examples, shard_size, seed)
    raise ValueError(f"Unsupported shard order strategy: {order_strategy}")


def main() -> None:
    args = parse_args()
    config = load_pipeline_yaml(args.config)
    cache_dir = resolve_cache_dir(config, args.cache_dir)
    manifest_dir = resolve_manifest_dir(config, args.manifest_dir)
    dataset_root = resolve_dataset_root(config)
    output_dir = args.output_dir
    modalities = resolve_modalities(config, args.modalities)
    shard_size = args.shard_size if args.shard_size is not None else default_shard_size(modalities)
    specs = build_feature_cache_specs(config, modalities)
    cache_dirs = resolve_cache_dirs_by_modality(
        config,
        cache_dir,
        modalities,
        args.modality_cache_dir,
    )
    manifest_dirs = resolve_manifest_dirs_by_modality(
        config,
        cache_dir,
        manifest_dir,
        modalities,
        args.modality_manifest_dir,
    )

    for modality_cache_dir in set(cache_dirs.values()):
        reject_output_inside_inputs(output_dir, modality_cache_dir, dataset_root)
    for modality_manifest_dir in set(manifest_dirs.values()):
        reject_output_inside_inputs(output_dir, modality_manifest_dir, dataset_root)
    reject_existing_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"shard build start: config={args.config} cache_dir={cache_dir} "
        f"manifest_dir={manifest_dir or cache_dir} "
        f"output_dir={output_dir} modalities={','.join(modalities)} "
        f"shard_size={shard_size} workers={args.workers} "
        f"order_strategy={args.order_strategy} seed={args.seed}"
    )
    log(
        "cache dirs by modality: "
        + ", ".join(f"{modality}={cache_dirs[modality]}" for modality in modalities)
    )
    log(
        "manifest dirs by modality: "
        + ", ".join(f"{modality}={manifest_dirs[modality]}" for modality in modalities)
    )
    examples, manifest_summary = load_manifest_backed_examples_from_cache_dirs(
        cache_dirs=cache_dirs,
        manifest_dirs=manifest_dirs,
        specs=specs,
        modalities=modalities,
        expected_rows=args.expected_manifest_rows,
    )
    if args.limit is not None:
        examples = examples[: args.limit]
        log(f"limit applied: examples={len(examples)}")
    examples = apply_order_strategy(
        examples=examples,
        shard_size=shard_size,
        seed=args.seed,
        order_strategy=args.order_strategy,
    )
    log(f"shard source ready: examples={len(examples)} class_counts={class_counts(examples)}")

    shard_records = build_shards(
        examples=examples,
        cache_dirs=cache_dirs,
        output_dir=output_dir,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
        shard_size=shard_size,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index = {
        "version": SHARD_VERSION,
        "schema_version": SHARD_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": str(args.config),
        "cache_dir": str(cache_dir),
        "cache_dirs_by_modality": {modality: str(cache_dirs[modality]) for modality in modalities},
        "manifest_dirs_by_modality": {
            modality: str(manifest_dirs[modality]) for modality in modalities
        },
        "dataset_root": str(dataset_root),
        "modalities": list(modalities),
        "spec_ids": {modality: feature_cache_spec_id(spec) for modality, spec in specs.items()},
        "specs": {modality: asdict(spec) for modality, spec in specs.items()},
        "source_manifests": source_manifest_stats(manifest_dirs, specs, modalities),
        "manifest_selection": manifest_summary,
        "class_counts": class_counts(examples),
        "example_count": len(examples),
        "example_locations": shard_example_locations(examples, shard_size, dataset_root),
        "order_strategy": args.order_strategy,
        "seed": args.seed,
        "shard_size": shard_size,
        "workers": args.workers,
        "shard_count": len(shard_records),
        "shards": shard_records,
    }
    write_json(output_dir / "index.json", index)
    log(f"shard build done: index={output_dir / 'index.json'} shards={len(shard_records)}")


if __name__ == "__main__":
    main()
