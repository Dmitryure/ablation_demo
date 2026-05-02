from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (
    LabeledVideoDataset,
    VideoExample,
    build_real_fake_examples,
    collate_labeled_video_batch,
    summarize_examples,
    write_dataset_manifest,
)
from feature_cache import (
    CachedFeatureDataset,
    FeatureCacheSpec,
    build_feature_cache_specs,
    collate_cached_feature_batch,
    feature_cache_item_exists,
    feature_cache_spec_id,
    split_feature_batch,
    write_feature_cache_item,
)
from pipeline import build_fusion_pipeline, load_pipeline_yaml
from task_models import BinaryFusionClassifier, build_binary_fusion_classifier

DEFAULT_DATASET_ROOT = Path("/mnt/d/final_dataset")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "iterative_ablation_runs"
FAST_LADDER = (200, 500, 1000, 2000, 4000)
TINY_LADDER = (40, 100, 200, 500)
LARGE_LADDER = (1000, 2500, 5000)
HEAD_TYPES = ("cls_linear", "cls_mlp", "attention_mil", "modality_gated_mil")


@dataclass(frozen=True)
class PredictionRow:
    path: str
    class_name: str
    label: int
    probability: float
    prediction: int
    split: str


@dataclass(frozen=True)
class EpochTrainResult:
    loss: float
    accuracy: float
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iteratively grow per-modality feature cache and train cached ablations."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument(
        "--modality-permutations",
        choices=("none", "singletons", "singletons-plus-all", "all"),
        default="none",
    )
    parser.add_argument(
        "--round-ladder",
        choices=("fast", "tiny", "large"),
        default="fast",
    )
    parser.add_argument(
        "--round-targets",
        type=int,
        nargs="+",
        default=None,
        help="Explicit train video counts. Overrides --round-ladder and does not append full train set.",
    )
    parser.add_argument("--eval-count-per-split", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--modality-lr",
        nargs="*",
        default=None,
        help="Optional branch LR overrides, e.g. rgb=0.0003 fau=0.0001.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--head-type", choices=HEAD_TYPES, default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-failures", action="store_true")
    parser.add_argument("--sanity-count", type=int, default=300)
    parser.add_argument("--no-sanity-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(config_path: Path, device: str | None) -> dict[str, Any]:
    config = load_pipeline_yaml(config_path)
    if device is not None:
        config["device"] = device
    return config


def resolve_video_root(dataset_root: Path) -> Path:
    if (dataset_root / "real").is_dir() and (dataset_root / "fake").is_dir():
        return dataset_root
    videos_root = dataset_root / "videos"
    if (videos_root / "real").is_dir() and (videos_root / "fake").is_dir():
        return videos_root
    raise FileNotFoundError(f"Expected real/fake folders under {dataset_root} or {videos_root}.")


def resolve_base_modalities(
    config: Mapping[str, Any],
    modalities: Sequence[str] | None,
) -> tuple[str, ...]:
    if modalities is not None:
        return tuple(modalities)
    configured = config.get("modalities")
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise ValueError("Config `modalities` must be a list of strings.")
    return tuple(item for item in configured if item)


def build_modality_sets(
    base_modalities: Sequence[str],
    mode: str,
) -> list[tuple[str, ...]]:
    base = tuple(base_modalities)
    if mode == "none":
        return [base]
    singletons = [(modality,) for modality in base]
    if mode == "singletons":
        return singletons
    if mode == "singletons-plus-all":
        return [*singletons, base]
    if mode == "all":
        return [
            tuple(combo)
            for size in range(1, len(base) + 1)
            for combo in itertools.combinations(base, size)
        ]
    raise ValueError(f"Unknown modality permutation mode: {mode}")


def class_counts(examples: Sequence[VideoExample]) -> dict[str, int]:
    counts = {"real": 0, "fake": 0}
    for example in examples:
        counts[example.class_name] += 1
    return counts


def _shuffled(items: Sequence[VideoExample], seed: int) -> list[VideoExample]:
    result = list(items)
    random.Random(seed).shuffle(result)
    return result


def _balanced_fake_order(fake_examples: Sequence[VideoExample], seed: int) -> list[VideoExample]:
    by_identity: dict[str, list[VideoExample]] = defaultdict(list)
    for example in fake_examples:
        by_identity[example.identity_id or "unknown"].append(example)
    rng = random.Random(seed)
    for examples in by_identity.values():
        rng.shuffle(examples)
    identities = sorted(by_identity)
    rng.shuffle(identities)

    ordered: list[VideoExample] = []
    while identities:
        next_identities: list[str] = []
        for identity in identities:
            examples = by_identity[identity]
            if examples:
                ordered.append(examples.pop())
            if examples:
                next_identities.append(identity)
        identities = next_identities
    return ordered


def build_balanced_train_order(
    examples: Sequence[VideoExample],
    seed: int,
) -> list[VideoExample]:
    real = _shuffled([example for example in examples if example.class_name == "real"], seed)
    fake = _balanced_fake_order(
        [example for example in examples if example.class_name == "fake"],
        seed + 1,
    )
    ordered: list[VideoExample] = []
    real_index = 0
    fake_index = 0
    while real_index < len(real) or fake_index < len(fake):
        if real_index < len(real):
            ordered.append(real[real_index])
            real_index += 1
        if fake_index < len(fake):
            ordered.append(fake[fake_index])
            fake_index += 1
    return ordered


def select_balanced_subset(
    examples: Sequence[VideoExample],
    target_count: int,
    seed: int,
) -> list[VideoExample]:
    real = _shuffled([example for example in examples if example.class_name == "real"], seed)
    fake = _balanced_fake_order(
        [example for example in examples if example.class_name == "fake"],
        seed + 1,
    )
    per_class = min(target_count // 2, len(real), len(fake))
    selected: list[VideoExample] = []
    for index in range(per_class):
        selected.append(real[index])
        selected.append(fake[index])
    return selected


def split_examples(
    examples: Sequence[VideoExample],
    eval_count_per_split: int,
    seed: int,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    train = [example for example in examples if example.split == "train"]
    val = select_balanced_subset(
        [example for example in examples if example.split == "val"],
        eval_count_per_split,
        seed + 11,
    )
    test = select_balanced_subset(
        [example for example in examples if example.split == "test"],
        eval_count_per_split,
        seed + 23,
    )
    return train, val, test


def resolve_round_targets(
    train_count: int,
    ladder: str,
    explicit_targets: Sequence[int] | None = None,
) -> list[int]:
    if explicit_targets is not None:
        targets = sorted(set(explicit_targets))
        if any(target <= 0 for target in targets):
            raise ValueError("`--round-targets` values must be positive.")
        if any(target > train_count for target in targets):
            raise ValueError(
                f"`--round-targets` cannot exceed available train videos ({train_count})."
            )
        return targets
    base = {
        "fast": FAST_LADDER,
        "tiny": TINY_LADDER,
        "large": LARGE_LADDER,
    }[ladder]
    targets = [target for target in base if target < train_count]
    targets.append(train_count)
    return list(dict.fromkeys(targets))


def model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    return torch.device("cpu")


def move_tensor_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def freeze_encoder_modules(model: BinaryFusionClassifier) -> None:
    for parameter in model.pipeline.encoder_modules.parameters():
        parameter.requires_grad_(False)


def build_head_config(config: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    raw_config = config.get("head")
    if raw_config is None:
        head_config: dict[str, Any] = {}
    elif isinstance(raw_config, Mapping):
        head_config = dict(raw_config)
    else:
        raise ValueError("Config `head` must be a mapping when provided.")
    if args.head_type is not None:
        head_config["type"] = args.head_type
    if args.head_hidden_dim is not None:
        head_config["hidden_dim"] = args.head_hidden_dim
    if args.head_dropout is not None:
        head_config["dropout"] = args.head_dropout
    return head_config or None


def parse_modality_lrs(values: Sequence[str] | None) -> dict[str, float]:
    if not values:
        return {}
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"`--modality-lr` must use modality=value, got {value!r}.")
        modality, raw_lr = value.split("=", 1)
        modality = modality.strip()
        if not modality:
            raise ValueError(f"`--modality-lr` has empty modality in {value!r}.")
        lr = float(raw_lr)
        if lr <= 0.0:
            raise ValueError(f"`--modality-lr` must be positive, got {value!r}.")
        parsed[modality] = lr
    return parsed


def build_optimizer(
    model: BinaryFusionClassifier,
    base_lr: float,
    modality_lrs: Mapping[str, float],
) -> torch.optim.Optimizer:
    grouped_parameter_ids: set[int] = set()
    parameter_groups: list[dict[str, Any]] = []
    for modality, lr in sorted(modality_lrs.items()):
        if modality not in model.pipeline.registry:
            raise ValueError(f"Cannot set LR for unknown modality branch: {modality}")
        parameters = [
            parameter
            for parameter in model.pipeline.registry[modality].parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            continue
        grouped_parameter_ids.update(id(parameter) for parameter in parameters)
        parameter_groups.append({"params": parameters, "lr": lr})

    remaining_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in grouped_parameter_ids
    ]
    if remaining_parameters:
        parameter_groups.insert(0, {"params": remaining_parameters, "lr": base_lr})
    return torch.optim.AdamW(parameter_groups)


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)
    return float((predictions == labels).to(dtype=torch.float32).mean().item())


def train_one_epoch(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
) -> EpochTrainResult:
    device = model_device(model)
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    start = time.perf_counter()
    for batch in loader:
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(move_tensor_batch_to_device(batch, device))
        loss = loss_fn(output.logits, labels)
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.item()) * count
        total_correct += binary_accuracy(output.logits.detach(), labels) * count
        total_count += count
    elapsed = time.perf_counter() - start
    return EpochTrainResult(
        loss=total_loss / total_count,
        accuracy=total_correct / total_count,
        elapsed_seconds=elapsed,
    )


def predict_rows(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
) -> tuple[float, list[PredictionRow]]:
    device = model_device(model)
    correct = 0
    total = 0
    rows: list[PredictionRow] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(move_tensor_batch_to_device(batch, device))
            probabilities = output.probabilities.detach().cpu().view(-1)
            labels = batch["label"].view(-1).to(dtype=torch.long)
            predictions = (probabilities >= 0.5).to(dtype=torch.long)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            for index, probability in enumerate(probabilities.tolist()):
                rows.append(
                    PredictionRow(
                        path=batch["path"][index],
                        class_name=batch["class_name"][index],
                        label=int(labels[index].item()),
                        probability=float(probability),
                        prediction=int(predictions[index].item()),
                        split=batch["split"][index],
                    )
                )
    return correct / total, rows


def build_cached_loader(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    batch_size: int,
    shuffle: bool,
    dataset_root: Path,
) -> DataLoader[dict[str, Any]]:
    dataset = CachedFeatureDataset(
        examples=examples,
        cache_dir=cache_dir,
        spec_by_modality=specs,
        modalities=modalities,
        strict=True,
        dataset_root=dataset_root,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_cached_feature_batch,
    )


def failure_log_path(cache_dir: Path) -> Path:
    return cache_dir / "feature_cache_failures.csv"


def read_failure_keys(cache_dir: Path) -> set[tuple[str, str, str]]:
    path = failure_log_path(cache_dir)
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {(row["spec_id"], row["modality"], row["path"]) for row in csv.DictReader(handle)}


def append_failure_rows(cache_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path = failure_log_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        fieldnames = ("spec_id", "modality", "path", "error")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def missing_examples_for_modality(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    spec: FeatureCacheSpec,
    dataset_root: Path,
    overwrite: bool,
    skip_failure_keys: set[tuple[str, str, str]],
) -> list[VideoExample]:
    spec_id = feature_cache_spec_id(spec)
    modality_cache_root = cache_dir / spec_id / spec.modality
    if not overwrite and not modality_cache_root.exists():
        return [
            example
            for example in examples
            if (spec_id, spec.modality, str(example.path)) not in skip_failure_keys
        ]
    missing: list[VideoExample] = []
    for example in examples:
        failure_key = (spec_id, spec.modality, str(example.path))
        if failure_key in skip_failure_keys:
            continue
        if overwrite or not feature_cache_item_exists(
            cache_dir,
            example,
            spec,
            dataset_root=dataset_root,
        ):
            missing.append(example)
    return missing


def count_missing_cache(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
) -> dict[str, int]:
    missing: dict[str, int] = {}
    for modality in modalities:
        spec = specs[modality]
        modality_cache_root = cache_dir / feature_cache_spec_id(spec) / spec.modality
        if not modality_cache_root.exists():
            missing[modality] = len(examples)
            continue
        missing[modality] = sum(
            not feature_cache_item_exists(cache_dir, example, spec, dataset_root)
            for example in examples
        )
    return missing


def count_cached_and_skipped(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    spec: FeatureCacheSpec,
    dataset_root: Path,
    skip_failure_keys: set[tuple[str, str, str]],
) -> tuple[int, int]:
    spec_id = feature_cache_spec_id(spec)
    cached = 0
    skipped_failed = 0
    for example in examples:
        if feature_cache_item_exists(cache_dir, example, spec, dataset_root):
            cached += 1
        elif (spec_id, spec.modality, str(example.path)) in skip_failure_keys:
            skipped_failed += 1
    return cached, skipped_failed


def missing_modalities_for_example(
    example: VideoExample,
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
) -> tuple[str, ...]:
    return tuple(
        modality
        for modality in modalities
        if not feature_cache_item_exists(cache_dir, example, specs[modality], dataset_root)
    )


def filter_examples_with_cache(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    label: str,
) -> list[VideoExample]:
    kept: list[VideoExample] = []
    dropped: list[tuple[VideoExample, tuple[str, ...]]] = []
    for example in examples:
        missing = missing_modalities_for_example(
            example,
            cache_dir,
            specs,
            modalities,
            dataset_root,
        )
        if missing:
            dropped.append((example, missing))
        else:
            kept.append(example)

    if dropped:
        by_modality: dict[str, int] = defaultdict(int)
        for _, missing in dropped:
            for modality in missing:
                by_modality[modality] += 1
        print(
            f"cache filter {label}: kept={len(kept)} dropped={len(dropped)} "
            f"missing={dict(sorted(by_modality.items()))}",
            flush=True,
        )
        for example, missing in dropped[:5]:
            print(
                f"cache filter {label}: dropped path={example.path} missing={','.join(missing)}",
                flush=True,
            )
        if len(dropped) > 5:
            print(
                f"cache filter {label}: dropped_more={len(dropped) - 5}",
                flush=True,
            )
    else:
        print(f"cache filter {label}: kept={len(kept)} dropped=0", flush=True)
    return kept


def ensure_feature_cache(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    config: Mapping[str, Any],
    dataset_root: Path,
    extract_batch_size: int,
    overwrite: bool,
    skip_failures: bool,
    progress_every: int,
    label: str,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    skip_failure_keys = read_failure_keys(cache_dir) if skip_failures else set()
    progress: dict[str, Any] = {}
    progress_interval = max(1, progress_every)
    if extract_batch_size != 1:
        print(
            "warning: extraction runs one video at a time so failed videos can be skipped; "
            f"--extract-batch-size={extract_batch_size} ignored.",
            flush=True,
        )
    for modality in modalities:
        spec = specs[modality]
        missing = missing_examples_for_modality(
            examples=examples,
            cache_dir=cache_dir,
            spec=spec,
            dataset_root=dataset_root,
            overwrite=overwrite,
            skip_failure_keys=skip_failure_keys,
        )
        progress[modality] = {
            "requested": len(examples),
            "missing_before": len(missing),
            "cached_before": 0,
            "skipped_failed": 0,
            "written": 0,
            "failed": 0,
        }
        cached_before, skipped_failed = count_cached_and_skipped(
            examples=examples,
            cache_dir=cache_dir,
            spec=spec,
            dataset_root=dataset_root,
            skip_failure_keys=skip_failure_keys,
        )
        progress[modality]["cached_before"] = cached_before
        progress[modality]["skipped_failed"] = skipped_failed
        print(
            f"cache {label}: modality={modality} requested={len(examples)} "
            f"cached={cached_before} skipped_failed={skipped_failed} missing={len(missing)}",
            flush=True,
        )
        if not missing:
            continue
        start = time.perf_counter()
        build_result = build_fusion_pipeline(config=config, modalities=(modality,))
        failure_rows: list[dict[str, Any]] = []
        build_result.pipeline.eval()
        with torch.no_grad():
            for index, example in enumerate(missing, start=1):
                try:
                    dataset = LabeledVideoDataset(
                        examples=[example],
                        num_frames={modality: spec.frame_count},
                        image_size=spec.image_size,
                    )
                    raw_batch = collate_labeled_video_batch([dataset[0]])
                    feature_batch = build_result.pipeline.prepare_features(raw_batch)
                    for item in split_feature_batch(feature_batch, raw_batch):
                        write_feature_cache_item(
                            cache_dir=cache_dir,
                            example=example,
                            spec=spec,
                            item=item,
                            dataset_root=dataset_root,
                        )
                        progress[modality]["written"] += 1
                except Exception as exc:
                    print(
                        f"cache {label}: modality={modality} failed "
                        f"{index}/{len(missing)} path={example.path} error={exc}",
                        flush=True,
                    )
                    failure_rows.append(
                        {
                            "spec_id": feature_cache_spec_id(spec),
                            "modality": modality,
                            "path": str(example.path),
                            "error": str(exc),
                        }
                    )
                    progress[modality]["failed"] += 1
                if index == len(missing) or index % progress_interval == 0:
                    elapsed = time.perf_counter() - start
                    done = progress[modality]["written"] + progress[modality]["failed"]
                    rate = 0.0 if elapsed <= 0.0 else done / elapsed
                    print(
                        f"cache {label}: modality={modality} "
                        f"done={done}/{len(missing)} "
                        f"written={progress[modality]['written']} "
                        f"failed={progress[modality]['failed']} "
                        f"elapsed={elapsed:.1f}s videos_per_s={rate:.2f}",
                        flush=True,
                    )
        build_result.pipeline.close()
        append_failure_rows(cache_dir, failure_rows)
        if build_result.device.type == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - start
        print(
            f"cache {label}: modality={modality} complete "
            f"written={progress[modality]['written']} "
            f"failed={progress[modality]['failed']} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return progress


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "train_loss", "train_accuracy", "train_elapsed_seconds"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_predictions(path: Path, rows: Sequence[PredictionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "class_name", "label", "prediction", "probability", "split"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "class_name": row.class_name,
                    "label": row.label,
                    "prediction": row.prediction,
                    "probability": f"{row.probability:.8f}",
                    "split": row.split,
                }
            )


def run_training_round(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    train_examples: Sequence[VideoExample],
    val_examples: Sequence[VideoExample],
    test_examples: Sequence[VideoExample],
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not train_examples:
        raise ValueError("No cached train examples available for this round.")
    if not val_examples:
        raise ValueError("No cached validation examples available for this round.")
    if not test_examples:
        raise ValueError("No cached test examples available for this round.")

    build_result = build_fusion_pipeline(config=config, modalities=modalities)
    model = build_binary_fusion_classifier(
        build_result.pipeline,
        dim=int(config["dim"]),
        head_config=build_head_config(config, args),
    )
    freeze_encoder_modules(model)
    model = model.to(build_result.device)
    modality_lrs = parse_modality_lrs(args.modality_lr)
    optimizer = build_optimizer(model, base_lr=args.lr, modality_lrs=modality_lrs)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loader = build_cached_loader(
        train_examples,
        cache_dir,
        specs,
        modalities,
        args.batch_size,
        shuffle=True,
        dataset_root=dataset_root,
    )
    val_loader = build_cached_loader(
        val_examples,
        cache_dir,
        specs,
        modalities,
        args.batch_size,
        shuffle=False,
        dataset_root=dataset_root,
    )
    test_loader = build_cached_loader(
        test_examples,
        cache_dir,
        specs,
        modalities,
        args.batch_size,
        shuffle=False,
        dataset_root=dataset_root,
    )

    metrics: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_path = output_dir / "best.pt"
    print(
        "training: "
        f"modalities={','.join(modalities)} epochs={args.epochs} "
        f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"batch_size={args.batch_size} device={build_result.device} "
        f"lr={args.lr} modality_lrs={modality_lrs}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        result = train_one_epoch(model, train_loader, optimizer, loss_fn)
        row = {
            "epoch": epoch,
            "train_loss": f"{result.loss:.8f}",
            "train_accuracy": f"{result.accuracy:.8f}",
            "train_elapsed_seconds": f"{result.elapsed_seconds:.6f}",
        }
        metrics.append(row)
        if result.accuracy > best_accuracy:
            best_accuracy = result.accuracy
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={result.loss:.6f} "
            f"train_accuracy={result.accuracy:.4f} "
            f"elapsed={result.elapsed_seconds:.3f}s",
            flush=True,
        )

    best_state = torch.load(best_path, map_location=build_result.device, weights_only=False)
    model.load_state_dict(best_state)
    print(f"loaded best checkpoint for eval: {best_path}", flush=True)
    train_accuracy, train_rows = predict_rows(model, train_loader)
    val_accuracy, val_rows = predict_rows(model, val_loader)
    test_accuracy, test_rows = predict_rows(model, test_loader)
    print(
        f"eval: train_accuracy={train_accuracy:.4f} "
        f"val_accuracy={val_accuracy:.4f} test_accuracy={test_accuracy:.4f}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(output_dir / "metrics.csv", metrics)
    write_predictions(output_dir / "predictions.csv", [*train_rows, *val_rows, *test_rows])
    summary = {
        "modalities": list(modalities),
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "test_count": len(test_examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "modality_lrs": modality_lrs,
        "batch_size": args.batch_size,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "best_checkpoint": str(best_path),
    }
    write_json(output_dir / "summary.json", summary)
    build_result.pipeline.close()
    print(f"wrote: {output_dir}", flush=True)
    return summary


def select_sanity_examples(
    examples: Sequence[VideoExample],
    excluded_examples: Sequence[VideoExample],
    target_count: int,
    seed: int,
) -> list[VideoExample]:
    excluded_paths = {str(example.path) for example in excluded_examples}
    pool = [example for example in examples if str(example.path) not in excluded_paths]
    return select_balanced_subset(pool, target_count=target_count, seed=seed)


def run_sanity_check(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    dataset_root: Path,
    sanity_examples: Sequence[VideoExample],
    summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    sanity_results: list[dict[str, Any]] = []
    for summary in summaries:
        modalities = tuple(str(modality) for modality in summary["modalities"])
        name = f"train_{int(summary['round_target']):05d}_{modality_set_name(modalities)}"
        cached_examples = filter_examples_with_cache(
            sanity_examples,
            cache_dir,
            specs,
            modalities,
            dataset_root,
            label=f"sanity/{name}",
        )
        if not cached_examples:
            raise ValueError(f"No cached sanity examples available for {name}.")

        build_result = build_fusion_pipeline(config=config, modalities=modalities)
        model = build_binary_fusion_classifier(
            build_result.pipeline,
            dim=int(config["dim"]),
            head_config=build_head_config(config, args),
        )
        state = torch.load(
            summary["best_checkpoint"],
            map_location=build_result.device,
            weights_only=False,
        )
        model.load_state_dict(state)
        model = model.to(build_result.device)
        loader = build_cached_loader(
            cached_examples,
            cache_dir,
            specs,
            modalities,
            args.batch_size,
            shuffle=False,
            dataset_root=dataset_root,
        )
        accuracy, rows = predict_rows(model, loader)
        run_dir = output_dir / "sanity_check" / name
        write_predictions(run_dir / "predictions.csv", rows)
        result = {
            "round_target": int(summary["round_target"]),
            "modalities": list(modalities),
            "best_checkpoint": str(summary["best_checkpoint"]),
            "sanity_count": len(cached_examples),
            "sanity_accuracy": accuracy,
            "predictions_csv": str(run_dir / "predictions.csv"),
        }
        write_json(run_dir / "summary.json", result)
        sanity_results.append(result)
        build_result.pipeline.close()
        print(
            f"sanity: round={summary['round_target']} modalities={modality_set_name(modalities)} "
            f"count={len(cached_examples)} accuracy={accuracy:.4f}",
            flush=True,
        )
        print(f"wrote: {run_dir}", flush=True)
    return sanity_results


def write_dry_run(
    examples: Sequence[VideoExample],
    train_examples: Sequence[VideoExample],
    val_examples: Sequence[VideoExample],
    test_examples: Sequence[VideoExample],
    round_targets: Sequence[int],
    missing: Mapping[str, int],
) -> None:
    print(f"dataset_total={len(examples)}")
    print(f"summary={summarize_examples(examples)}")
    print(f"train_pool={len(train_examples)} counts={class_counts(train_examples)}")
    print(f"val_fixed={len(val_examples)} counts={class_counts(val_examples)}")
    print(f"test_fixed={len(test_examples)} counts={class_counts(test_examples)}")
    print(f"round_targets={','.join(str(target) for target in round_targets)}")
    print(f"missing_cache={dict(missing)}")


def modality_set_name(modalities: Sequence[str]) -> str:
    return "plus".join(modalities)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = build_config(args.config, args.device)
    dataset_root = args.dataset_root
    video_root = resolve_video_root(dataset_root)
    cache_dir = args.cache_dir or (dataset_root / "feature_cache")
    output_dir = args.output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    base_modalities = resolve_base_modalities(config, args.modalities)
    modality_sets = build_modality_sets(base_modalities, args.modality_permutations)
    specs = build_feature_cache_specs(config, base_modalities)

    examples = build_real_fake_examples(
        real_dir=video_root / "real",
        fake_dir=video_root / "fake",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_pool, val_examples, test_examples = split_examples(
        examples,
        eval_count_per_split=args.eval_count_per_split,
        seed=args.seed,
    )
    train_order = build_balanced_train_order(train_pool, args.seed + 101)
    round_targets = resolve_round_targets(
        len(train_order),
        args.round_ladder,
        explicit_targets=args.round_targets,
    )
    max_train_examples = train_order[: round_targets[-1]]
    fixed_examples = [*max_train_examples, *val_examples, *test_examples]

    print(f"output_dir={output_dir}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"video_root={video_root}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"modalities={','.join(base_modalities)}", flush=True)
    print(
        f"modality_sets={','.join(modality_set_name(item) for item in modality_sets)}", flush=True
    )
    print(f"dataset_total={len(examples)} summary={summarize_examples(examples)}", flush=True)
    print(f"train_pool={len(train_pool)} counts={class_counts(train_pool)}", flush=True)
    print(f"val_fixed={len(val_examples)} counts={class_counts(val_examples)}", flush=True)
    print(f"test_fixed={len(test_examples)} counts={class_counts(test_examples)}", flush=True)
    print(f"round_targets={','.join(str(target) for target in round_targets)}", flush=True)

    if args.dry_run:
        write_dry_run(
            examples,
            train_pool,
            val_examples,
            test_examples,
            round_targets,
            count_missing_cache(fixed_examples, cache_dir, specs, base_modalities, dataset_root),
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_manifest(examples, output_dir / "manifest.csv")
    write_json(
        output_dir / "run_config.json",
        {
            "dataset_root": str(dataset_root),
            "cache_dir": str(cache_dir),
            "base_modalities": list(base_modalities),
            "modality_sets": [list(item) for item in modality_sets],
            "round_targets": list(round_targets),
            "round_targets_source": "explicit"
            if args.round_targets is not None
            else args.round_ladder,
            "eval_count_per_split": args.eval_count_per_split,
            "seed": args.seed,
            "spec_ids": {modality: feature_cache_spec_id(spec) for modality, spec in specs.items()},
        },
    )

    eval_progress = ensure_feature_cache(
        examples=[*val_examples, *test_examples],
        cache_dir=cache_dir,
        specs=specs,
        modalities=base_modalities,
        config=config,
        dataset_root=dataset_root,
        extract_batch_size=args.extract_batch_size,
        overwrite=args.overwrite_cache,
        skip_failures=args.skip_failures,
        progress_every=args.progress_every,
        label="eval",
    )
    write_json(output_dir / "eval_cache_progress.json", eval_progress)
    print(f"wrote: {output_dir / 'eval_cache_progress.json'}", flush=True)

    summaries: list[dict[str, Any]] = []
    previous_by_modality_set: dict[str, dict[str, Any]] = {}
    for target in round_targets:
        train_examples = train_order[:target]
        round_dir = output_dir / f"train_{target:05d}"
        print(
            f"round start: train_videos={target} "
            f"counts={class_counts(train_examples)} output_dir={round_dir}",
            flush=True,
        )
        cache_progress = ensure_feature_cache(
            examples=train_examples,
            cache_dir=cache_dir,
            specs=specs,
            modalities=base_modalities,
            config=config,
            dataset_root=dataset_root,
            extract_batch_size=args.extract_batch_size,
            overwrite=args.overwrite_cache,
            skip_failures=args.skip_failures,
            progress_every=args.progress_every,
            label=f"train_{target}",
        )
        write_json(round_dir / "cache_progress.json", cache_progress)
        print(f"wrote: {round_dir / 'cache_progress.json'}", flush=True)

        for modalities in modality_sets:
            name = modality_set_name(modalities)
            print(f"round={target} modalities={name}", flush=True)
            cached_train_examples = filter_examples_with_cache(
                train_examples,
                cache_dir,
                specs,
                modalities,
                dataset_root,
                label=f"train_{target}/{name}",
            )
            cached_val_examples = filter_examples_with_cache(
                val_examples,
                cache_dir,
                specs,
                modalities,
                dataset_root,
                label=f"val/{name}",
            )
            cached_test_examples = filter_examples_with_cache(
                test_examples,
                cache_dir,
                specs,
                modalities,
                dataset_root,
                label=f"test/{name}",
            )
            summary = run_training_round(
                args=args,
                config=config,
                cache_dir=cache_dir,
                specs=specs,
                modalities=modalities,
                train_examples=cached_train_examples,
                val_examples=cached_val_examples,
                test_examples=cached_test_examples,
                dataset_root=dataset_root,
                output_dir=round_dir / name,
            )
            previous = previous_by_modality_set.get(name)
            summary["previous_val_accuracy"] = (
                None if previous is None else previous["val_accuracy"]
            )
            summary["val_accuracy_delta"] = (
                None if previous is None else summary["val_accuracy"] - previous["val_accuracy"]
            )
            summary["round_target"] = target
            summary["output_dir"] = str(round_dir / name)
            summaries.append(summary)
            previous_by_modality_set[name] = summary

    sanity_results: list[dict[str, Any]] = []
    if not args.no_sanity_check and args.sanity_count > 0:
        sanity_examples = select_sanity_examples(
            examples=examples,
            excluded_examples=[*max_train_examples, *val_examples, *test_examples],
            target_count=args.sanity_count,
            seed=args.seed + 303,
        )
        print(
            f"sanity setup: requested={args.sanity_count} selected={len(sanity_examples)} "
            f"counts={class_counts(sanity_examples)}",
            flush=True,
        )
        sanity_progress = ensure_feature_cache(
            examples=sanity_examples,
            cache_dir=cache_dir,
            specs=specs,
            modalities=base_modalities,
            config=config,
            dataset_root=dataset_root,
            extract_batch_size=args.extract_batch_size,
            overwrite=args.overwrite_cache,
            skip_failures=args.skip_failures,
            progress_every=args.progress_every,
            label="sanity",
        )
        write_json(output_dir / "sanity_cache_progress.json", sanity_progress)
        print(f"wrote: {output_dir / 'sanity_cache_progress.json'}", flush=True)
        sanity_results = run_sanity_check(
            args=args,
            config=config,
            cache_dir=cache_dir,
            specs=specs,
            dataset_root=dataset_root,
            sanity_examples=sanity_examples,
            summaries=summaries,
            output_dir=output_dir,
        )
    write_json(output_dir / "summary.json", {"rounds": summaries, "sanity": sanity_results})
    print(f"wrote: {output_dir / 'summary.json'}", flush=True)
    print(f"wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
