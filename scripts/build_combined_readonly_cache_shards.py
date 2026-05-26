from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoExample, write_dataset_manifest
from feature_cache import (
    MODALITY_FEATURE_KEYS,
    SHARDED_CACHE_SCHEMA_VERSION,
    FeatureCacheSpec,
    build_feature_cache_specs,
    cache_example_key,
    feature_cache_manifest_path,
    feature_cache_spec_id,
)
from pipeline import load_pipeline_yaml
from scripts.build_readonly_cache_shards import (
    apply_order_strategy,
    default_shard_size,
    example_payload,
    reject_existing_output,
    shard_example_locations,
    stack_shard_features,
    tensor_summary,
    write_shard_atomic,
)
from scripts.run_iterative_cached_ablation import class_counts, training_run_section, write_json

DEFAULT_CONFIG = (
    PROJECT_ROOT / "runs" / "configs" / "night_sweep" / "09_seed0_lr3e4_gen0p15_warm2_e26.yaml"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "shards" / "final_v2_ffpp_celebdf_domain_70_15_15"
DEFAULT_SOURCES = (
    ("final", Path("/mnt/d/final_cache/v2")),
    ("ffpp", Path("/mnt/d/ffpp_c23_prediction_cache/v1")),
    ("celebdf", Path("/mnt/d/celebdf_prediction_cache/v1")),
)
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_ORDER_STRATEGY = "stratified_shards"
DEFAULT_WORKERS = 8
DEFAULT_PROGRESS_EVERY = 1


@dataclass(frozen=True)
class CacheManifestRow:
    class_name: str
    filename: str
    generator_id: str
    source_path: Path
    cache_path: Path
    manifest_cache_path: Path
    status: str


@dataclass(frozen=True)
class CombinedRecord:
    source_name: str
    original_key: str
    shard_key: str
    example: VideoExample
    cache_paths: dict[str, Path]
    original_generator_id: str


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mixed read-only shards from final v2, FF++ C23, and CelebDF caches."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source cache root as NAME=PATH. Defaults to final, ffpp, celebdf roots.",
    )
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument(
        "--order-strategy",
        choices=("stratified_shards", "sorted"),
        default=DEFAULT_ORDER_STRATEGY,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read manifests and print summary without writing shards.",
    )
    return parser.parse_args()


def parse_sources(values: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    if not values:
        return DEFAULT_SOURCES
    sources: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH for --source, got {value!r}.")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty source name in --source {value!r}.")
        if name in seen:
            raise ValueError(f"Duplicate source name: {name}")
        seen.add(name)
        sources.append((name, Path(path.strip())))
    return tuple(sources)


def resolve_modalities(
    config: Mapping[str, Any], cli_modalities: Sequence[str] | None
) -> tuple[str, ...]:
    if cli_modalities is not None:
        return tuple(str(modality) for modality in cli_modalities)
    run = training_run_section(config)
    modalities = run.get("modalities") or config.get("modalities")
    if not isinstance(modalities, list):
        raise ValueError("Config must provide a modality list.")
    return tuple(str(modality) for modality in modalities)


def validate_split_ratios(train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("`--train-ratio` must be in (0.0, 1.0).")
    if val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("`--val-ratio` must be positive and leave room for test split.")


def manifest_key(class_name: str, filename: str) -> str:
    return f"{class_name}/{filename}"


def read_cached_manifest_rows(
    manifest_path: Path,
) -> tuple[dict[str, CacheManifestRow], dict[str, int]]:
    rows: dict[str, CacheManifestRow] = {}
    status_counts: Counter[str] = Counter()
    row_count = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"class_name", "filename", "generator_id", "source_path", "cache_path", "status"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Malformed cache manifest columns: {manifest_path}")
        for row in reader:
            row_count += 1
            status = str(row.get("status", "")).strip()
            status_counts[status] += 1
            if status != "cached":
                continue
            class_name = str(row["class_name"]).strip()
            filename = str(row["filename"]).strip()
            manifest_cache_path = Path(str(row.get("cache_path", "")).strip())
            local_cache_path = manifest_path.parent / class_name / f"{filename}.pt"
            cache_path = manifest_cache_path if manifest_cache_path.is_file() else local_cache_path
            key = manifest_key(class_name, filename)
            rows[key] = CacheManifestRow(
                class_name=class_name,
                filename=filename,
                generator_id=str(row.get("generator_id", "")).strip(),
                source_path=Path(str(row.get("source_path", "")).strip()),
                cache_path=cache_path,
                manifest_cache_path=manifest_cache_path,
                status=status,
            )
    return rows, {"rows": row_count, **dict(sorted(status_counts.items()))}


def read_source_modality_rows(
    source_name: str,
    cache_root: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> tuple[dict[str, dict[str, CacheManifestRow]], dict[str, Any]]:
    rows_by_modality: dict[str, dict[str, CacheManifestRow]] = {}
    stats: dict[str, Any] = {}
    for modality in modalities:
        path = feature_cache_manifest_path(cache_root, specs[modality])
        if not path.is_file():
            raise FileNotFoundError(f"Missing {source_name}/{modality} manifest: {path}")
        log(f"manifest read start: source={source_name} modality={modality} path={path}")
        rows, counts = read_cached_manifest_rows(path)
        rows_by_modality[modality] = rows
        stat = path.stat()
        stats[modality] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "row_counts": counts,
            "cached": len(rows),
        }
        log(
            f"manifest read done: source={source_name} modality={modality} "
            f"rows={counts['rows']} cached={len(rows)}"
        )
    return rows_by_modality, stats


def normalized_generator_id(source_name: str, class_name: str, raw_generator_id: str) -> str:
    if class_name == "real":
        return "real"
    if source_name in {"ffpp", "celebdf"}:
        return source_name
    return raw_generator_id or "unknown_or_other"


def namespaced_filename(source_name: str, filename: str) -> str:
    return f"{source_name}/{filename}"


def combined_source_id(source_name: str, class_name: str, filename: str) -> str:
    return f"{source_name}:{class_name}:{filename}"


def combined_record_from_rows(
    source_name: str,
    key: str,
    rows_by_modality: Mapping[str, Mapping[str, CacheManifestRow]],
    modalities: Sequence[str],
) -> CombinedRecord:
    first = rows_by_modality[modalities[0]][key]
    filename = namespaced_filename(source_name, first.filename)
    generator_id = normalized_generator_id(source_name, first.class_name, first.generator_id)
    example = VideoExample(
        path=first.source_path,
        label=0 if first.class_name == "real" else 1,
        class_name=first.class_name,
        source_id=combined_source_id(source_name, first.class_name, first.filename),
        split="train",
        metadata_filename=filename,
        identity_id=generator_id,
        generator_id=generator_id,
        source_id_kind="combined_cache_manifest",
    )
    return CombinedRecord(
        source_name=source_name,
        original_key=key,
        shard_key=cache_example_key(example),
        example=example,
        cache_paths={
            modality: rows_by_modality[modality][key].cache_path for modality in modalities
        },
        original_generator_id=first.generator_id,
    )


def load_source_records(
    source_name: str,
    cache_root: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> tuple[list[CombinedRecord], dict[str, Any]]:
    rows_by_modality, stats = read_source_modality_rows(
        source_name=source_name,
        cache_root=cache_root,
        specs=specs,
        modalities=modalities,
    )
    common_keys = set.intersection(*(set(rows) for rows in rows_by_modality.values()))
    records = [
        combined_record_from_rows(source_name, key, rows_by_modality, modalities)
        for key in sorted(common_keys)
    ]
    return records, {
        "cache_root": str(cache_root),
        "modalities": stats,
        "cached_by_modality": {
            modality: len(rows_by_modality[modality]) for modality in modalities
        },
        "intersection_cached": len(records),
    }


def split_counts(total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    raw = {split: total * ratio for split, ratio in ratios.items()}
    counts = {split: int(value) for split, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda split: (raw[split] - counts[split], split), reverse=True)
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def split_records(
    records: Sequence[CombinedRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[CombinedRecord]:
    buckets: dict[str, list[CombinedRecord]] = {}
    for record in records:
        buckets.setdefault(record.example.class_name, []).append(record)

    rng = random.Random(seed)
    split_by_key: dict[str, str] = {}
    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        counts = split_counts(len(bucket), train_ratio, val_ratio)
        cursor = 0
        for split in ("train", "val", "test"):
            for record in bucket[cursor : cursor + counts[split]]:
                split_by_key[record.shard_key] = split
            cursor += counts[split]

    result: list[CombinedRecord] = []
    for record in records:
        split = split_by_key[record.shard_key]
        result.append(
            CombinedRecord(
                source_name=record.source_name,
                original_key=record.original_key,
                shard_key=record.shard_key,
                example=VideoExample(
                    path=record.example.path,
                    label=record.example.label,
                    class_name=record.example.class_name,
                    source_id=record.example.source_id,
                    split=split,
                    metadata_filename=record.example.metadata_filename,
                    identity_id=record.example.identity_id,
                    generator_id=record.example.generator_id,
                    source_id_kind=record.example.source_id_kind,
                    age_bin=record.example.age_bin,
                    gender=record.example.gender,
                    ethnicity=record.example.ethnicity,
                    emotion=record.example.emotion,
                ),
                cache_paths=record.cache_paths,
                original_generator_id=record.original_generator_id,
            )
        )
    return result


def validate_unique_records(records: Sequence[CombinedRecord]) -> None:
    counts = Counter(record.shard_key for record in records)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate shard keys detected: {duplicates[:5]}")


def generator_counts(examples: Sequence[VideoExample]) -> dict[str, int]:
    counts = Counter(
        example.generator_id or "unknown" for example in examples if example.class_name == "fake"
    )
    return dict(sorted(counts.items()))


def split_summary(examples: Sequence[VideoExample]) -> dict[str, Any]:
    return {
        split: {
            "total": len(split_examples),
            "class_counts": class_counts(split_examples),
            "fake_generator_counts": generator_counts(split_examples),
        }
        for split in ("train", "val", "test")
        for split_examples in ([example for example in examples if example.split == split],)
    }


def source_summary(records: Sequence[CombinedRecord]) -> dict[str, Any]:
    return {
        source: {
            "total": len(source_records),
            "class_counts": class_counts([record.example for record in source_records]),
            "fake_generator_counts": generator_counts(
                [record.example for record in source_records]
            ),
        }
        for source in sorted({record.source_name for record in records})
        for source_records in ([record for record in records if record.source_name == source],)
    }


def dry_summary(
    records: Sequence[CombinedRecord],
    source_manifests: Mapping[str, Any],
    modalities: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    shard_size: int,
) -> dict[str, Any]:
    examples = [record.example for record in records]
    return {
        "modalities": list(modalities),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "shard_size": shard_size,
        "example_count": len(records),
        "class_counts": class_counts(examples),
        "fake_generator_counts": generator_counts(examples),
        "split_summary": split_summary(examples),
        "source_summary": source_summary(records),
        "source_manifests": source_manifests,
        "duplicate_shard_keys": len(records) - len({record.shard_key for record in records}),
    }


def load_feature_item_from_path(path: Path, spec: FeatureCacheSpec) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached feature file: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed cached feature payload: {path}")
    features = payload.get("features")
    if not isinstance(features, Mapping):
        raise ValueError(f"Cached feature payload missing features: {path}")
    result = {
        str(key): normalize_feature_tensor(value)
        for key, value in features.items()
        if isinstance(value, torch.Tensor)
    }
    required = MODALITY_FEATURE_KEYS[spec.modality]
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"Cached feature payload missing {missing}: {path}")
    return result


def normalize_feature_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dim() >= 3 and value.shape[0] == 1:
        return value.squeeze(0)
    return value


def load_record_features(
    record: CombinedRecord,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for modality in modalities:
        features.update(load_feature_item_from_path(record.cache_paths[modality], specs[modality]))
    return features


def load_shard_feature_items(
    shard_records: Sequence[CombinedRecord],
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    workers: int,
) -> list[dict[str, torch.Tensor]]:
    loader = partial(load_record_features, specs=specs, modalities=modalities)
    if workers <= 1:
        return [loader(record) for record in shard_records]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(loader, shard_records))


def build_shard_payload(
    shard_records: Sequence[CombinedRecord],
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    shard_index: int,
    workers: int,
) -> dict[str, Any]:
    feature_items = load_shard_feature_items(
        shard_records=shard_records,
        specs=specs,
        modalities=modalities,
        workers=workers,
    )
    features = stack_shard_features(feature_items)
    examples = [record.example for record in shard_records]
    return {
        "version": SHARDED_CACHE_SCHEMA_VERSION,
        "schema_version": SHARDED_CACHE_SCHEMA_VERSION,
        "shard_index": shard_index,
        "modalities": list(modalities),
        "examples": [example_payload(example) for example in examples],
        "class_counts": class_counts(examples),
        "feature_summary": tensor_summary(features),
        "features": features,
    }


def build_shards(
    records: Sequence[CombinedRecord],
    output_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    shard_size: int,
    workers: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    shards_dir = output_dir / "shards"
    shard_records: list[dict[str, Any]] = []
    started = time.monotonic()
    chunks = [records[index : index + shard_size] for index in range(0, len(records), shard_size)]
    for shard_index, chunk in enumerate(chunks):
        shard_name = f"shard_{shard_index:06d}.pt"
        shard_path = shards_dir / shard_name
        if shard_path.exists():
            raise FileExistsError(f"Shard already exists: {shard_path}")
        payload = build_shard_payload(
            shard_records=chunk,
            specs=specs,
            modalities=modalities,
            shard_index=shard_index,
            workers=workers,
        )
        write_shard_atomic(shard_path, payload)
        stat = shard_path.stat()
        examples = [record.example for record in chunk]
        shard_records.append(
            {
                "path": str(shard_path.relative_to(output_dir)),
                "example_count": len(chunk),
                "size": stat.st_size,
                "first_key": chunk[0].shard_key,
                "last_key": chunk[-1].shard_key,
                "class_counts": class_counts(examples),
                "feature_summary": payload["feature_summary"],
            }
        )
        if progress_every > 0 and (
            len(shard_records) == len(chunks) or len(shard_records) % progress_every == 0
        ):
            elapsed = time.monotonic() - started
            done = sum(int(record["example_count"]) for record in shard_records)
            rate = 0.0 if elapsed <= 0.0 else done / elapsed
            log(
                f"shard build: shards={len(shard_records)}/{len(chunks)} "
                f"examples={done}/{len(records)} elapsed={elapsed:.1f}s examples_per_s={rate:.2f}"
            )
    return shard_records


def main() -> None:
    args = parse_args()
    validate_split_ratios(args.train_ratio, args.val_ratio)
    config = load_pipeline_yaml(args.config)
    sources = parse_sources(args.source)
    modalities = resolve_modalities(config, args.modalities)
    shard_size = args.shard_size if args.shard_size is not None else default_shard_size(modalities)
    specs = build_feature_cache_specs(config, modalities)

    records: list[CombinedRecord] = []
    source_manifests: dict[str, Any] = {}
    for source_name, cache_root in sources:
        source_records, manifest_summary = load_source_records(
            source_name=source_name,
            cache_root=cache_root,
            specs=specs,
            modalities=modalities,
        )
        records.extend(source_records)
        source_manifests[source_name] = manifest_summary

    records = split_records(
        records=records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    validate_unique_records(records)
    summary = dry_summary(
        records=records,
        source_manifests=source_manifests,
        modalities=modalities,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        shard_size=shard_size,
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return

    output_dir = args.output_dir
    reject_existing_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = [record.example for record in records]
    write_dataset_manifest(examples, output_dir / "manifest.csv")
    records_by_key = {record.shard_key: record for record in records}
    ordered_examples = apply_order_strategy(
        examples=examples,
        shard_size=shard_size,
        seed=args.seed,
        order_strategy=args.order_strategy,
    )
    ordered_records = [records_by_key[cache_example_key(example)] for example in ordered_examples]
    shard_records = build_shards(
        records=ordered_records,
        output_dir=output_dir,
        specs=specs,
        modalities=modalities,
        shard_size=shard_size,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    index = {
        "version": SHARDED_CACHE_SCHEMA_VERSION,
        "schema_version": SHARDED_CACHE_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_path": str(args.config),
        "dataset_manifest": str(output_dir / "manifest.csv"),
        "source_cache_roots": {name: str(path) for name, path in sources},
        "modalities": list(modalities),
        "spec_ids": {modality: feature_cache_spec_id(specs[modality]) for modality in modalities},
        "specs": {modality: asdict(specs[modality]) for modality in modalities},
        "source_manifests": source_manifests,
        "manifest_selection": summary,
        "class_counts": class_counts(ordered_examples),
        "fake_generator_counts": generator_counts(ordered_examples),
        "split_summary": split_summary(ordered_examples),
        "example_count": len(ordered_records),
        "example_locations": shard_example_locations(ordered_examples, shard_size, None),
        "order_strategy": args.order_strategy,
        "seed": args.seed,
        "shard_size": shard_size,
        "workers": args.workers,
        "shard_count": len(shard_records),
        "shards": shard_records,
    }
    write_json(output_dir / "index.json", index)
    log(f"combined shard build done: output_dir={output_dir} shards={len(shard_records)}")


if __name__ == "__main__":
    main()
