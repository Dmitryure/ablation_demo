from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoExample
from feature_cache import (
    MODALITY_FEATURE_KEYS,
    CachedFeatureDataset,
    FeatureCacheSpec,
    build_feature_cache_specs,
    collate_cached_feature_batch,
    feature_cache_spec_dir,
)
from pipeline import build_fusion_pipeline
from scripts.run_generator_multitask_training import (
    predict,
    write_json,
)
from scripts.run_iterative_cached_ablation import (
    resolve_cached_loader_config,
)
from task_models.generator_multitask_classifier import build_generator_multitask_classifier
from training_metrics import (
    binary_metrics,
    generator_metrics,
    prediction_summary,
    write_predictions,
)
from training_targets import GeneratorTargetSpec, real_fake_counts

CLASS_FILTERS = ("real", "fake", "all")
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs/night_sweep/seed0_lr3e4_gen0p15_warm2_e26"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs/external_cache_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a generator-multitask checkpoint on prediction feature caches."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--class-filter", choices=CLASS_FILTERS, default="real")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def checkpoint_path(run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    path = run_dir / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint; pass --checkpoint explicitly: {path}")
    return path


def target_from_run_config(run_config: dict[str, Any]) -> GeneratorTargetSpec:
    raw = run_config.get("target")
    if not isinstance(raw, dict):
        raise ValueError("Run config is missing `target`.")
    generator_names = raw.get("generator_names")
    if not isinstance(generator_names, list) or not generator_names:
        raise ValueError("Run config target is missing `generator_names`.")
    raw_to_group = raw.get("raw_to_group", {})
    if isinstance(raw_to_group, dict):
        raw_to_group_items = tuple(sorted((str(k), str(v)) for k, v in raw_to_group.items()))
    elif isinstance(raw_to_group, list):
        raw_to_group_items = tuple((str(k), str(v)) for k, v in raw_to_group)
    else:
        raise ValueError("Run config target `raw_to_group` must be a mapping or list.")
    return GeneratorTargetSpec(
        generator_names=tuple(str(name) for name in generator_names),
        raw_to_group=raw_to_group_items,
        unknown_group_name=str(raw.get("unknown_group_name", "unknown_or_other")),
    )


def cache_key_from_path(path: Path, spec_dir: Path) -> tuple[str, str]:
    relative = path.relative_to(spec_dir)
    parts = relative.parts
    if len(parts) < 2:
        raise ValueError(f"Unexpected cache item path: {path}")
    class_name = parts[0]
    if class_name not in {"real", "fake"}:
        raise ValueError(f"Unexpected cache class folder in {path}")
    filename = PurePosixPath(*parts[1:]).as_posix()
    if not filename.endswith(".pt"):
        raise ValueError(f"Unexpected cache item suffix: {path}")
    return class_name, filename[: -len(".pt")]


def cached_keys_for_spec(cache_dir: Path, spec: FeatureCacheSpec) -> set[tuple[str, str]]:
    spec_dir = feature_cache_spec_dir(cache_dir, spec)
    if not spec_dir.is_dir():
        return set()
    return {cache_key_from_path(path, spec_dir) for path in sorted(spec_dir.rglob("*.pt"))}


def generator_id_for_key(class_name: str, filename: str) -> str:
    if class_name == "real":
        return "real"
    parts = PurePosixPath(filename).parts
    return parts[0] if parts else "unknown"


def source_id_for_filename(filename: str) -> str:
    return PurePosixPath(filename).name.removesuffix(".mp4")


def examples_from_cache_dir(
    cache_dir: Path,
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
    class_filter: str,
) -> list[VideoExample]:
    key_sets = [cached_keys_for_spec(cache_dir, specs[modality]) for modality in modalities]
    if not key_sets:
        return []
    keys = set.intersection(*key_sets)
    if class_filter != "all":
        keys = {key for key in keys if key[0] == class_filter}
    examples = [
        VideoExample(
            path=cache_dir / class_name / filename,
            label=1 if class_name == "fake" else 0,
            class_name=class_name,
            source_id=source_id_for_filename(filename),
            split=cache_dir.name,
            metadata_filename=filename,
            generator_id=generator_id_for_key(class_name, filename),
            source_id_kind="prediction_feature_cache",
        )
        for class_name, filename in sorted(keys)
    ]
    if not examples:
        raise ValueError(f"No complete cached examples found in {cache_dir}")
    return examples


def prediction_cache_feature_keys(modalities: tuple[str, ...]) -> set[str]:
    return {key for modality in modalities for key in MODALITY_FEATURE_KEYS.get(modality, ())}


def collate_prediction_feature_batch(
    items: list[dict[str, Any]],
    feature_keys: set[str],
) -> dict[str, Any]:
    batch = collate_cached_feature_batch(items)
    for key in feature_keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == 1:
            batch[key] = value.squeeze(1)
    return batch


def build_prediction_cache_loader(
    examples: list[VideoExample],
    cache_dir: Path,
    specs: dict[str, FeatureCacheSpec],
    modalities: tuple[str, ...],
    batch_size: int,
    config: dict[str, Any],
) -> DataLoader[dict[str, Any]]:
    loader_config = resolve_cached_loader_config(config)
    dataset = CachedFeatureDataset(
        examples=examples,
        cache_dir=cache_dir,
        spec_by_modality=specs,
        modalities=modalities,
        strict=True,
        dataset_root=cache_dir,
    )
    feature_keys = prediction_cache_feature_keys(modalities)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": lambda items: collate_prediction_feature_batch(items, feature_keys),
        "num_workers": loader_config.num_workers,
        "pin_memory": loader_config.pin_memory,
    }
    if loader_config.num_workers > 0:
        loader_kwargs["persistent_workers"] = loader_config.persistent_workers
        if loader_config.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = loader_config.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def load_model(
    config: dict[str, Any],
    checkpoint: Path,
    target: GeneratorTargetSpec,
    modalities: tuple[str, ...],
    device_override: str | None,
) -> Any:
    if device_override is not None:
        config = {**config, "device": device_override}
    build_result = build_fusion_pipeline(config=config, modalities=modalities)
    model = build_generator_multitask_classifier(
        build_result.pipeline,
        dim=int(config["dim"]),
        num_generators=target.num_generators,
        head_config=config.get("head"),
    ).to(build_result.device)
    state = torch.load(checkpoint, map_location=build_result.device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def safe_name(paths: list[Path], class_filter: str) -> str:
    names = "_".join(path.parent.name for path in paths)
    return f"{names}_{class_filter}" if names else class_filter


def binary_auc(records: Sequence[Any]) -> float | None:
    labels = [int(record.binary_label) for record in records]
    scores = [float(record.binary_probability) for record in records]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        average_rank = (index + 1 + next_index) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:next_index])
        index = next_index

    return (rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def format_optional_metric(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    run_config = load_json(args.run_dir / "run_config.json")
    config_path = Path(str(run_config["config_path"]))
    config = load_yaml(config_path)
    modalities = tuple(str(modality) for modality in run_config["modalities"])
    specs = build_feature_cache_specs(config, modalities)
    target = target_from_run_config(run_config)
    checkpoint = checkpoint_path(args.run_dir, args.checkpoint)

    examples_by_cache = {
        str(cache_dir): examples_from_cache_dir(
            cache_dir,
            specs=specs,
            modalities=modalities,
            class_filter=args.class_filter,
        )
        for cache_dir in args.cache_dir
    }
    examples = [example for group in examples_by_cache.values() for example in group]
    if not examples:
        raise ValueError("No external cached examples selected.")

    model = load_model(
        config=config,
        checkpoint=checkpoint,
        target=target,
        modalities=modalities,
        device_override=args.device,
    )
    records = []
    for cache_dir in args.cache_dir:
        cache_examples = examples_by_cache[str(cache_dir)]
        loader = build_prediction_cache_loader(
            examples=cache_examples,
            cache_dir=cache_dir,
            specs=specs,
            modalities=modalities,
            batch_size=args.batch_size,
            config=config,
        )
        records.extend(predict(model, loader, target=target, split=cache_dir.parent.name))

    output_dir = args.output_dir / safe_name(args.cache_dir, args.class_filter)
    metrics = binary_metrics(records)
    auc = binary_auc(records)
    summary = prediction_summary(records)
    gen_metrics = generator_metrics(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.csv", records)
    write_json(
        output_dir / "summary.json",
        {
            "run_dir": str(args.run_dir),
            "checkpoint": str(checkpoint),
            "cache_dirs": [str(path) for path in args.cache_dir],
            "class_filter": args.class_filter,
            "modalities": list(modalities),
            "count": len(records),
            "class_counts": real_fake_counts(examples),
            "cache_counts": {
                cache_dir: real_fake_counts(cache_examples)
                for cache_dir, cache_examples in examples_by_cache.items()
            },
            "binary_metrics": asdict(metrics),
            "binary_auc": auc,
            "prediction_summary": summary,
            "generator_metrics": gen_metrics,
            "predictions_csv": str(output_dir / "predictions.csv"),
        },
    )
    model.pipeline.close()
    print(
        f"external cache eval: count={len(records)} "
        f"accuracy={metrics.accuracy:.4f} "
        f"balanced_accuracy={metrics.balanced_accuracy:.4f} "
        f"f1={metrics.f1:.4f} "
        f"auc={format_optional_metric(auc)} "
        f"precision={metrics.precision:.4f} "
        f"recall={metrics.recall:.4f} "
        f"specificity={metrics.specificity:.4f} "
        f"false_positive={metrics.false_positive} "
        f"false_negative={metrics.false_negative} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
