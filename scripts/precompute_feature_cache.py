from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import build_real_fake_examples, format_split_audit, summarize_examples
from feature_cache import (
    build_feature_cache_specs,
    feature_cache_item_exists,
    feature_cache_spec_id,
)
from scripts.run_iterative_cached_ablation import (
    DEFAULT_CONFIG,
    DEFAULT_DATASET_ROOT,
    build_config,
    ensure_feature_cache,
    read_failure_keys,
    resolve_base_modalities,
    resolve_video_root,
    write_json,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "feature_cache_runs"
VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute cached features for configured modalities over dataset splits."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS))
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override config image_size. Creates a separate cache spec.",
    )
    parser.add_argument(
        "--modality-frames",
        nargs="*",
        default=None,
        help="Override per-modality frame counts, e.g. fau=32 rgb=16. Creates separate cache specs.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--balanced-total",
        type=int,
        default=None,
        help=(
            "Select exactly this many videos from the whole dataset with equal real/fake "
            "counts, ignoring split boundaries. Example: 1000 means 500 real and 500 fake."
        ),
    )
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument(
        "--balanced-limit-per-split",
        type=int,
        default=None,
        help=(
            "Select up to this many videos per split with equal real/fake counts. "
            "Each split is capped by its smaller class."
        ),
    )
    parser.add_argument(
        "--limit-total",
        type=int,
        default=None,
        help="Limit total selected videos after split filtering using seeded shuffle.",
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument(
        "--skip-cache-audits",
        action="store_true",
        help="Skip pre/post missing-cache scans and skipped-files CSV generation.",
    )
    parser.add_argument(
        "--assume-missing-cache",
        action="store_true",
        help=(
            "Do not scan for existing cache items before extraction. Fastest for a fresh cache dir; "
            "existing cache files for selected examples can be overwritten."
        ),
    )
    parser.add_argument(
        "--modality-grouping",
        choices=("modality", "clip-spec"),
        default="modality",
        help=(
            "Use 'modality' for one extraction pipeline per modality, or 'clip-spec' "
            "to group modalities with the same frame count and image size so videos are decoded once."
        ),
    )
    parser.add_argument(
        "--cache-format",
        choices=("files", "shards"),
        default="files",
        help="Write one file per video, or write indexed shard files.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=256,
        help="Videos per shard when --cache-format shards is used.",
    )
    parser.add_argument(
        "--video-decode-mode",
        choices=("seek", "scan"),
        default="seek",
        help="Use random frame seeks or sequential video scan when sampling frames.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Use plain periodic logs instead of tqdm progress bars.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_device(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` requested, but CUDA is not available.")


def parse_modality_frames(values: Sequence[str] | None) -> dict[str, int]:
    if not values:
        return {}
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"`--modality-frames` must use modality=count, got {value!r}.")
        modality, raw_count = value.split("=", 1)
        modality = modality.strip()
        if not modality:
            raise ValueError(f"`--modality-frames` has empty modality in {value!r}.")
        frame_count = int(raw_count)
        if frame_count <= 0:
            raise ValueError(f"`--modality-frames` must be positive, got {value!r}.")
        parsed[modality] = frame_count
    return parsed


def apply_cache_variant_overrides(
    config: dict[str, Any],
    image_size: int | None,
    modality_frames: Mapping[str, int],
) -> dict[str, Any]:
    updated = dict(config)
    if image_size is not None:
        if image_size <= 0:
            raise ValueError("`--image-size` must be positive.")
        updated["image_size"] = image_size
    for modality, frame_count in modality_frames.items():
        section = updated.get(modality, {})
        if section is None:
            section = {}
        if not isinstance(section, dict):
            raise ValueError(f"`{modality}` must be a mapping to override frames.")
        updated[modality] = {**section, "frames": frame_count}
    return updated


def select_examples_by_split(
    examples: list[Any],
    splits: tuple[str, ...],
    limit_per_split: int | None,
    limit_total: int | None,
    balanced_limit_per_split: int | None,
    seed: int,
) -> list[Any]:
    if balanced_limit_per_split is not None:
        if balanced_limit_per_split <= 1:
            raise ValueError("`--balanced-limit-per-split` must be greater than 1.")
        if limit_per_split is not None or limit_total is not None:
            raise ValueError(
                "`--balanced-limit-per-split` cannot be combined with "
                "`--limit-per-split` or `--limit-total`."
            )
        per_class_target = balanced_limit_per_split // 2
        selected: list[Any] = []
        for split_index, split in enumerate(splits):
            split_examples = [example for example in examples if example.split == split]
            real_examples = [example for example in split_examples if example.class_name == "real"]
            fake_examples = [example for example in split_examples if example.class_name == "fake"]
            rng = random.Random(seed + split_index)
            rng.shuffle(real_examples)
            rng.shuffle(fake_examples)
            per_class_count = min(per_class_target, len(real_examples), len(fake_examples))
            split_selected = real_examples[:per_class_count] + fake_examples[:per_class_count]
            rng.shuffle(split_selected)
            selected.extend(split_selected)
        return selected

    selected: list[Any] = []
    for split in splits:
        split_examples = [example for example in examples if example.split == split]
        if limit_per_split is not None:
            split_examples = split_examples[:limit_per_split]
        selected.extend(split_examples)
    if limit_total is not None:
        if limit_total <= 0:
            raise ValueError("`--limit-total` must be positive.")
        rng = random.Random(seed)
        rng.shuffle(selected)
        selected = selected[:limit_total]
    return selected


def select_balanced_total_examples(
    examples: Sequence[Any],
    balanced_total: int,
    seed: int,
) -> list[Any]:
    if balanced_total <= 0:
        raise ValueError("`--balanced-total` must be positive.")
    if balanced_total % 2 != 0:
        raise ValueError("`--balanced-total` must be even for equal real/fake selection.")

    per_class = balanced_total // 2
    real_examples = [example for example in examples if example.class_name == "real"]
    fake_examples = [example for example in examples if example.class_name == "fake"]
    if per_class > len(real_examples) or per_class > len(fake_examples):
        max_total = 2 * min(len(real_examples), len(fake_examples))
        raise ValueError(
            f"`--balanced-total {balanced_total}` exceeds available balanced total "
            f"{max_total} (real={len(real_examples)}, fake={len(fake_examples)})."
        )

    rng = random.Random(seed)
    rng.shuffle(real_examples)
    rng.shuffle(fake_examples)
    selected = real_examples[:per_class] + fake_examples[:per_class]
    rng.shuffle(selected)
    return selected


def select_precompute_examples(
    examples: Sequence[Any],
    splits: tuple[str, ...],
    limit_per_split: int | None,
    limit_total: int | None,
    balanced_limit_per_split: int | None,
    balanced_total: int | None,
    seed: int,
) -> list[Any]:
    if balanced_total is not None:
        if (
            limit_per_split is not None
            or limit_total is not None
            or balanced_limit_per_split is not None
        ):
            raise ValueError(
                "`--balanced-total` cannot be combined with `--limit-per-split`, "
                "`--limit-total`, or `--balanced-limit-per-split`."
            )
        return select_balanced_total_examples(examples, balanced_total, seed)

    return select_examples_by_split(
        examples=list(examples),
        splits=splits,
        limit_per_split=limit_per_split,
        limit_total=limit_total,
        balanced_limit_per_split=balanced_limit_per_split,
        seed=seed,
    )


def class_counts(examples: list[Any]) -> dict[str, int]:
    counts = {"real": 0, "fake": 0}
    for example in examples:
        counts[example.class_name] += 1
    return counts


def write_skipped_files_csv(
    path: Path,
    examples: Sequence[Any],
    cache_dir: Path,
    specs: Mapping[str, Any],
    modalities: Sequence[str],
    dataset_root: Path,
) -> int:
    failure_keys = read_failure_keys(cache_dir)
    rows: list[dict[str, Any]] = []
    for example in examples:
        for modality in modalities:
            spec = specs[modality]
            if feature_cache_item_exists(cache_dir, example, spec, dataset_root):
                continue
            spec_id = feature_cache_spec_id(spec)
            failure_key = (spec_id, modality, str(example.path))
            rows.append(
                {
                    "split": example.split,
                    "class_name": example.class_name,
                    "label": example.label,
                    "modality": modality,
                    "spec_id": spec_id,
                    "path": str(example.path),
                    "source_id": example.source_id,
                    "identity_id": example.identity_id or "",
                    "skipped_failure_log": "1" if failure_key in failure_keys else "0",
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "split",
            "class_name",
            "label",
            "modality",
            "spec_id",
            "path",
            "source_id",
            "identity_id",
            "skipped_failure_log",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def count_missing_cache_with_progress(
    examples: Sequence[Any],
    cache_dir: Path,
    specs: Mapping[str, Any],
    modalities: Sequence[str],
    dataset_root: Path,
    progress_bar: bool,
    progress_every: int,
) -> dict[str, int]:
    progress_bar = progress_bar and sys.stderr.isatty()
    missing: dict[str, int] = {}
    progress_interval = max(1, progress_every)
    for modality in modalities:
        spec = specs[modality]
        iterator = examples
        if progress_bar:
            iterator = tqdm(
                examples,
                desc=f"count missing/{modality}",
                unit="video",
                dynamic_ncols=True,
                leave=False,
            )
        missing_count = 0
        for index, example in enumerate(iterator, start=1):
            if not feature_cache_item_exists(cache_dir, example, spec, dataset_root):
                missing_count += 1
            if not progress_bar and (index == len(examples) or index % progress_interval == 0):
                print(
                    f"count missing: modality={modality} "
                    f"checked={index}/{len(examples)} missing={missing_count}",
                    flush=True,
                )
        missing[modality] = missing_count
    return missing


def main() -> None:
    args = parse_args()
    require_device(args.device)
    config = build_config(args.config, args.device)
    modality_frame_overrides = parse_modality_frames(args.modality_frames)
    config = apply_cache_variant_overrides(
        config=config,
        image_size=args.image_size,
        modality_frames=modality_frame_overrides,
    )
    dataset_root = args.dataset_root
    video_root = resolve_video_root(dataset_root)
    cache_dir = args.cache_dir or (dataset_root / "feature_cache")
    modalities = resolve_base_modalities(config, args.modalities)
    specs = build_feature_cache_specs(config, modalities)
    output_dir = args.output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    examples = build_real_fake_examples(
        real_dir=video_root / "real",
        fake_dir=video_root / "fake",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    selected_splits = tuple(args.splits)
    selected_examples = select_precompute_examples(
        examples=examples,
        splits=selected_splits,
        limit_per_split=args.limit_per_split,
        limit_total=args.limit_total,
        balanced_limit_per_split=args.balanced_limit_per_split,
        balanced_total=args.balanced_total,
        seed=args.seed,
    )
    print(f"output_dir={output_dir}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"video_root={video_root}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"device={config['device']}", flush=True)
    print(f"modalities={','.join(modalities)}", flush=True)
    selection_mode = (
        f"balanced_total_{args.balanced_total}"
        if args.balanced_total is not None
        else "split_filter"
    )
    print(f"selection_mode={selection_mode}", flush=True)
    print(f"splits={','.join(selected_splits)}", flush=True)
    print(f"dataset_total={len(examples)} summary={summarize_examples(examples)}", flush=True)
    for line in format_split_audit(examples):
        print(line, flush=True)
    print(
        f"selected={len(selected_examples)} counts={class_counts(selected_examples)}",
        flush=True,
    )
    print(f"selected_summary={summarize_examples(selected_examples)}", flush=True)
    if args.skip_cache_audits:
        missing_before = None
        print("skipping missing_before cache audit", flush=True)
    else:
        print("counting missing cache entries...", flush=True)
        missing_before = count_missing_cache_with_progress(
            selected_examples,
            cache_dir,
            specs,
            modalities,
            dataset_root,
            progress_bar=not args.no_progress_bar,
            progress_every=args.progress_every,
        )
        print(f"missing_before={missing_before}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        {
            "dataset_root": str(dataset_root),
            "cache_dir": str(cache_dir),
            "config": str(args.config),
            "modalities": list(modalities),
            "splits": list(selected_splits),
            "seed": args.seed,
            "image_size_override": args.image_size,
            "modality_frame_overrides": modality_frame_overrides,
            "limit_per_split": args.limit_per_split,
            "limit_total": args.limit_total,
            "balanced_total": args.balanced_total,
            "balanced_limit_per_split": args.balanced_limit_per_split,
            "overwrite_cache": args.overwrite_cache,
            "skip_failures": args.skip_failures,
            "skip_cache_audits": args.skip_cache_audits,
            "assume_missing_cache": args.assume_missing_cache,
            "modality_grouping": args.modality_grouping,
            "cache_format": args.cache_format,
            "shard_size": args.shard_size,
            "video_decode_mode": args.video_decode_mode,
            "progress_bar": not args.no_progress_bar,
            "spec_ids": {modality: feature_cache_spec_id(spec) for modality, spec in specs.items()},
            "dataset_summary": summarize_examples(examples),
            "selected_count": len(selected_examples),
            "selected_counts": class_counts(selected_examples),
            "selected_summary": summarize_examples(selected_examples),
            "missing_before": missing_before,
        },
    )

    if args.dry_run:
        print(f"wrote: {output_dir / 'run_config.json'}", flush=True)
        return

    progress = ensure_feature_cache(
        examples=selected_examples,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        config=config,
        dataset_root=dataset_root,
        extract_batch_size=args.extract_batch_size,
        overwrite=args.overwrite_cache,
        skip_failures=args.skip_failures,
        progress_every=args.progress_every,
        label="precompute",
        progress_bar=not args.no_progress_bar,
        group_by_modality=args.modality_grouping == "modality",
        assume_missing_cache=args.assume_missing_cache,
        cache_format=args.cache_format,
        shard_size=args.shard_size,
        video_decode_mode=args.video_decode_mode,
    )
    if args.skip_cache_audits:
        missing_after = None
        skipped_count = None
        print("skipping missing_after cache audit", flush=True)
    else:
        missing_after = count_missing_cache_with_progress(
            selected_examples,
            cache_dir,
            specs,
            modalities,
            dataset_root,
            progress_bar=not args.no_progress_bar,
            progress_every=args.progress_every,
        )
        skipped_count = write_skipped_files_csv(
            output_dir / "skipped_files.csv",
            examples=selected_examples,
            cache_dir=cache_dir,
            specs=specs,
            modalities=modalities,
            dataset_root=dataset_root,
        )
    write_json(output_dir / "cache_progress.json", progress)
    write_json(
        output_dir / "summary.json",
        {
            "progress": progress,
            "missing_before": missing_before,
            "missing_after": missing_after,
            "skipped_files_csv": str(output_dir / "skipped_files.csv"),
            "skipped_file_rows": skipped_count,
        },
    )
    print(f"missing_after={missing_after}", flush=True)
    print(f"skipped_file_rows={skipped_count}", flush=True)
    print(f"wrote: {output_dir / 'cache_progress.json'}", flush=True)
    if not args.skip_cache_audits:
        print(f"wrote: {output_dir / 'skipped_files.csv'}", flush=True)
    print(f"wrote: {output_dir / 'summary.json'}", flush=True)
    print(f"wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
