from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import VideoExample, build_real_fake_examples, format_split_audit, summarize_examples
from feature_cache import (
    MODALITY_FEATURE_KEYS,
    FeatureCacheSpec,
    build_feature_cache_specs,
    feature_cache_spec_dir,
    load_feature_cache_item,
)
from pipeline import load_pipeline_yaml
from scripts.run_iterative_cached_ablation import (
    build_modality_sets,
    resolve_base_modalities,
    resolve_video_root,
)

DEFAULT_DATASET_ROOT = Path("/mnt/d/final_dataset")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cached_feature_probes"
POOL_MODES = ("mean", "mean-std", "flatten")


@dataclass(frozen=True)
class ProbeDataset:
    features: torch.Tensor
    labels: torch.Tensor
    examples: tuple[VideoExample, ...]
    missing_count: int
    feature_dim: int


@dataclass(frozen=True)
class ProbeMetrics:
    loss: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float
    matthews_corrcoef: float
    auc: float | None
    count: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    label_counts: dict[str, int]
    prediction_counts: dict[str, int]
    probability_mean: float
    probability_min: float
    probability_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear probes on cached modality features.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument(
        "--modality-permutations",
        choices=("none", "singletons", "singletons-plus-all", "all"),
        default="singletons",
    )
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--eval-count-per-split", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--pool", choices=POOL_MODES, default="mean-std")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` requested, but CUDA is not available.")
    return torch.device(name)


def class_counts(examples: Sequence[VideoExample]) -> dict[str, int]:
    counts = {"real": 0, "fake": 0}
    for example in examples:
        counts[example.class_name] += 1
    return counts


def split_examples(
    examples: Sequence[VideoExample],
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    train_pool = [example for example in examples if example.split == "train"]
    val_pool = [example for example in examples if example.split == "val"]
    test_pool = [example for example in examples if example.split == "test"]
    return train_pool, val_pool, test_pool


def primary_feature_key(modality: str) -> str:
    if modality not in MODALITY_FEATURE_KEYS:
        raise ValueError(f"Unsupported modality: {modality}")
    return MODALITY_FEATURE_KEYS[modality][0]


def pooled_feature_vector(tensor: torch.Tensor, pool: str) -> torch.Tensor:
    feature = tensor.detach().cpu().to(dtype=torch.float32)
    if torch.isnan(feature).any():
        feature = torch.nan_to_num(feature)
    if pool == "flatten":
        return feature.reshape(-1)
    if feature.ndim == 0:
        feature = feature.view(1)
    if feature.ndim == 1:
        if pool == "mean":
            return feature
        return torch.cat([feature, torch.zeros_like(feature)])
    flattened = feature.reshape(-1, feature.shape[-1])
    mean = flattened.mean(dim=0)
    if pool == "mean":
        return mean
    std = flattened.std(dim=0, unbiased=False)
    return torch.cat([mean, std])


def load_feature_vector(
    example: VideoExample,
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    pool: str,
) -> torch.Tensor | None:
    vectors: list[torch.Tensor] = []
    for modality in modalities:
        item = load_feature_cache_item(
            cache_dir,
            example,
            specs[modality],
            dataset_root=dataset_root,
        )
        if item is None:
            return None
        key = primary_feature_key(modality)
        value = item.get(key)
        if not isinstance(value, torch.Tensor):
            return None
        vectors.append(pooled_feature_vector(value, pool=pool))
    return torch.cat(vectors)


def load_probe_dataset(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    pool: str,
) -> ProbeDataset:
    rows: list[torch.Tensor] = []
    labels: list[float] = []
    kept_examples: list[VideoExample] = []
    missing_count = 0
    feature_dim: int | None = None
    for example in examples:
        vector = load_feature_vector(
            example=example,
            cache_dir=cache_dir,
            specs=specs,
            modalities=modalities,
            dataset_root=dataset_root,
            pool=pool,
        )
        if vector is None:
            missing_count += 1
            continue
        if feature_dim is None:
            feature_dim = int(vector.numel())
        if int(vector.numel()) != feature_dim:
            raise ValueError(
                f"Feature dim mismatch for {example.path}: {vector.numel()} != {feature_dim}"
            )
        rows.append(vector)
        labels.append(float(example.label))
        kept_examples.append(example)
    if not rows or feature_dim is None:
        raise ValueError(f"No cached examples available for modalities={','.join(modalities)}")
    return ProbeDataset(
        features=torch.stack(rows, dim=0),
        labels=torch.tensor(labels, dtype=torch.float32).view(-1, 1),
        examples=tuple(kept_examples),
        missing_count=missing_count,
        feature_dim=feature_dim,
    )


def shuffled_class_examples(
    examples: Sequence[VideoExample],
    class_name: str,
    seed: int,
) -> list[VideoExample]:
    selected = [example for example in examples if example.class_name == class_name]
    random.Random(seed).shuffle(selected)
    return selected


def load_cached_class_rows(
    examples: Sequence[VideoExample],
    target_count: int,
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    pool: str,
) -> tuple[list[torch.Tensor], list[float], list[VideoExample], int]:
    rows: list[torch.Tensor] = []
    labels: list[float] = []
    kept_examples: list[VideoExample] = []
    missing_count = 0
    for example in examples:
        if len(rows) >= target_count:
            break
        vector = load_feature_vector(
            example=example,
            cache_dir=cache_dir,
            specs=specs,
            modalities=modalities,
            dataset_root=dataset_root,
            pool=pool,
        )
        if vector is None:
            missing_count += 1
            continue
        rows.append(vector)
        labels.append(float(example.label))
        kept_examples.append(example)
    return rows, labels, kept_examples, missing_count


def load_balanced_probe_dataset(
    examples: Sequence[VideoExample],
    target_count: int,
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    pool: str,
    seed: int,
) -> ProbeDataset:
    per_class_target = max(1, target_count // 2)
    real_examples = shuffled_class_examples(examples, "real", seed=seed)
    fake_examples = shuffled_class_examples(examples, "fake", seed=seed + 1)
    real_rows, real_labels, real_kept, real_missing = load_cached_class_rows(
        real_examples,
        per_class_target,
        cache_dir,
        specs,
        modalities,
        dataset_root,
        pool,
    )
    fake_rows, fake_labels, fake_kept, fake_missing = load_cached_class_rows(
        fake_examples,
        per_class_target,
        cache_dir,
        specs,
        modalities,
        dataset_root,
        pool,
    )
    per_class = min(len(real_rows), len(fake_rows))
    if per_class == 0:
        raise ValueError(f"No balanced cached examples for modalities={','.join(modalities)}")
    rows = [*real_rows[:per_class], *fake_rows[:per_class]]
    labels = [*real_labels[:per_class], *fake_labels[:per_class]]
    kept_examples = [*real_kept[:per_class], *fake_kept[:per_class]]
    order = list(range(len(rows)))
    random.Random(seed + 2).shuffle(order)
    rows = [rows[index] for index in order]
    labels = [labels[index] for index in order]
    kept_examples = [kept_examples[index] for index in order]
    feature_dim = int(rows[0].numel())
    for example, row in zip(kept_examples, rows, strict=True):
        if int(row.numel()) != feature_dim:
            raise ValueError(
                f"Feature dim mismatch for {example.path}: {row.numel()} != {feature_dim}"
            )
    return ProbeDataset(
        features=torch.stack(rows, dim=0),
        labels=torch.tensor(labels, dtype=torch.float32).view(-1, 1),
        examples=tuple(kept_examples),
        missing_count=real_missing + fake_missing,
        feature_dim=feature_dim,
    )


def standardize_splits(
    train: ProbeDataset,
    val: ProbeDataset,
    test: ProbeDataset,
) -> tuple[ProbeDataset, ProbeDataset, ProbeDataset]:
    mean = train.features.mean(dim=0, keepdim=True)
    std = train.features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (
        replace_features(train, (train.features - mean) / std),
        replace_features(val, (val.features - mean) / std),
        replace_features(test, (test.features - mean) / std),
    )


def replace_features(dataset: ProbeDataset, features: torch.Tensor) -> ProbeDataset:
    return ProbeDataset(
        features=features,
        labels=dataset.labels,
        examples=dataset.examples,
        missing_count=dataset.missing_count,
        feature_dim=dataset.feature_dim,
    )


def train_linear_probe(
    train: ProbeDataset,
    feature_dim: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> torch.nn.Linear:
    torch.manual_seed(seed)
    model = torch.nn.Linear(feature_dim, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataset = TensorDataset(train.features, train.labels)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()
    return model


def predict_probabilities(
    model: torch.nn.Module,
    dataset: ProbeDataset,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(dataset.features), batch_size=batch_size, shuffle=False)
    probabilities: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (features,) in loader:
            logits = model(features.to(device))
            probabilities.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(probabilities, dim=0).view(-1)


def binary_auc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    label_values = [int(value) for value in labels.view(-1).tolist()]
    score_values = [float(value) for value in scores.view(-1).tolist()]
    positive_count = sum(label_values)
    negative_count = len(label_values) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    pairs = sorted(zip(score_values, label_values, strict=True), key=lambda item: item[0])
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


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def matthews_corrcoef(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return safe_divide(float(tp * tn - fp * fn), denominator)


def evaluate_predictions(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> ProbeMetrics:
    labels_int = labels.view(-1).to(dtype=torch.long)
    predictions = (probabilities >= 0.5).to(dtype=torch.long)
    loss = torch.nn.functional.binary_cross_entropy(
        probabilities.clamp(1e-7, 1.0 - 1e-7).view(-1, 1),
        labels,
    )
    correct = (predictions == labels_int).to(dtype=torch.float32)
    real_mask = labels_int == 0
    fake_mask = labels_int == 1
    true_positive = int(((predictions == 1) & fake_mask).sum().item())
    true_negative = int(((predictions == 0) & real_mask).sum().item())
    false_positive = int(((predictions == 1) & real_mask).sum().item())
    false_negative = int(((predictions == 0) & fake_mask).sum().item())
    precision = safe_divide(float(true_positive), float(true_positive + false_positive))
    recall = safe_divide(float(true_positive), float(true_positive + false_negative))
    specificity = safe_divide(float(true_negative), float(true_negative + false_positive))
    negative_predictive_value = safe_divide(
        float(true_negative), float(true_negative + false_negative)
    )
    false_positive_rate = safe_divide(float(false_positive), float(false_positive + true_negative))
    false_negative_rate = safe_divide(float(false_negative), float(false_negative + true_positive))
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    label_counts = Counter(str(int(value)) for value in labels_int.tolist())
    prediction_counts = Counter(str(int(value)) for value in predictions.tolist())
    return ProbeMetrics(
        loss=float(loss.item()),
        accuracy=float(correct.mean().item()),
        balanced_accuracy=(specificity + recall) / 2.0,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        negative_predictive_value=negative_predictive_value,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        matthews_corrcoef=matthews_corrcoef(
            true_positive, true_negative, false_positive, false_negative
        ),
        auc=binary_auc(labels_int, probabilities),
        count=int(labels_int.numel()),
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        label_counts=dict(sorted(label_counts.items())),
        prediction_counts=dict(sorted(prediction_counts.items())),
        probability_mean=float(probabilities.mean().item()),
        probability_min=float(probabilities.min().item()),
        probability_max=float(probabilities.max().item()),
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def write_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "modalities",
            "split",
            "count",
            "loss",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "negative_predictive_value",
            "false_positive_rate",
            "false_negative_rate",
            "matthews_corrcoef",
            "auc",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "label_counts",
            "prediction_counts",
            "probability_mean",
            "probability_min",
            "probability_max",
            "feature_dim",
            "missing_count",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_skipped(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("modalities", "reason")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_predictions(
    path: Path,
    modality_name: str,
    split: str,
    dataset: ProbeDataset,
    probabilities: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "modalities",
            "split",
            "path",
            "class_name",
            "label",
            "prediction",
            "probability",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        predictions = (probabilities >= 0.5).to(dtype=torch.long)
        for example, probability, prediction in zip(
            dataset.examples,
            probabilities.tolist(),
            predictions.tolist(),
            strict=True,
        ):
            writer.writerow(
                {
                    "modalities": modality_name,
                    "split": split,
                    "path": str(example.path),
                    "class_name": example.class_name,
                    "label": int(example.label),
                    "prediction": int(prediction),
                    "probability": f"{float(probability):.8f}",
                }
            )


def metric_row(
    modality_name: str,
    split: str,
    metrics: ProbeMetrics,
    feature_dim: int,
    missing_count: int,
) -> dict[str, Any]:
    return {
        "modalities": modality_name,
        "split": split,
        "count": metrics.count,
        "loss": f"{metrics.loss:.8f}",
        "accuracy": f"{metrics.accuracy:.8f}",
        "balanced_accuracy": f"{metrics.balanced_accuracy:.8f}",
        "precision": f"{metrics.precision:.8f}",
        "recall": f"{metrics.recall:.8f}",
        "f1": f"{metrics.f1:.8f}",
        "specificity": f"{metrics.specificity:.8f}",
        "negative_predictive_value": f"{metrics.negative_predictive_value:.8f}",
        "false_positive_rate": f"{metrics.false_positive_rate:.8f}",
        "false_negative_rate": f"{metrics.false_negative_rate:.8f}",
        "matthews_corrcoef": f"{metrics.matthews_corrcoef:.8f}",
        "auc": "" if metrics.auc is None else f"{metrics.auc:.8f}",
        "true_positive": metrics.true_positive,
        "true_negative": metrics.true_negative,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "label_counts": json.dumps(metrics.label_counts, sort_keys=True),
        "prediction_counts": json.dumps(metrics.prediction_counts, sort_keys=True),
        "probability_mean": f"{metrics.probability_mean:.8f}",
        "probability_min": f"{metrics.probability_min:.8f}",
        "probability_max": f"{metrics.probability_max:.8f}",
        "feature_dim": feature_dim,
        "missing_count": missing_count,
    }


def modality_set_name(modalities: Sequence[str]) -> str:
    return "plus".join(modalities)


def run_probe_for_modalities(
    modalities: Sequence[str],
    train_pool: Sequence[VideoExample],
    val_pool: Sequence[VideoExample],
    test_pool: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    dataset_root: Path,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    name = modality_set_name(modalities)
    missing_cache_roots = [
        modality
        for modality in modalities
        if not feature_cache_spec_dir(cache_dir, specs[modality]).exists()
    ]
    if missing_cache_roots:
        raise ValueError(
            "No cache directory for modalities="
            f"{','.join(missing_cache_roots)}. Generate cache first or pass --modalities."
        )
    train = load_balanced_probe_dataset(
        train_pool,
        target_count=args.train_count,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
        pool=args.pool,
        seed=args.seed + 101,
    )
    val = load_balanced_probe_dataset(
        val_pool,
        target_count=args.eval_count_per_split,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
        pool=args.pool,
        seed=args.seed + 201,
    )
    test = load_balanced_probe_dataset(
        test_pool,
        target_count=args.eval_count_per_split,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
        pool=args.pool,
        seed=args.seed + 301,
    )
    train, val, test = standardize_splits(train, val, test)
    model = train_linear_probe(
        train=train,
        feature_dim=train.feature_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    probabilities_by_split = {
        "train": predict_probabilities(model, train, device, args.batch_size),
        "val": predict_probabilities(model, val, device, args.batch_size),
        "test": predict_probabilities(model, test, device, args.batch_size),
    }
    datasets_by_split = {"train": train, "val": val, "test": test}
    metrics_by_split = {
        split: evaluate_predictions(datasets_by_split[split].labels, probabilities)
        for split, probabilities in probabilities_by_split.items()
    }
    for split, probabilities in probabilities_by_split.items():
        write_predictions(
            output_dir / "predictions.csv",
            modality_name=name,
            split=split,
            dataset=datasets_by_split[split],
            probabilities=probabilities,
        )
    return {
        "modalities": list(modalities),
        "pool": args.pool,
        "feature_dim": train.feature_dim,
        "missing_counts": {
            "train": train.missing_count,
            "val": val.missing_count,
            "test": test.missing_count,
        },
        "metrics": {split: asdict(metrics) for split, metrics in metrics_by_split.items()},
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = require_device(args.device)
    config = load_pipeline_yaml(args.config)
    dataset_root = args.dataset_root
    video_root = resolve_video_root(dataset_root)
    cache_dir = args.cache_dir or (dataset_root / "feature_cache")
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
    train_pool, val_pool, test_pool = split_examples(examples)
    output_dir = args.output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"output_dir={output_dir}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"cache_dir={cache_dir}", flush=True)
    print(f"device={device}", flush=True)
    print(f"modalities={','.join(base_modalities)}", flush=True)
    print(
        f"modality_sets={','.join(modality_set_name(item) for item in modality_sets)}", flush=True
    )
    print(f"dataset_total={len(examples)} summary={summarize_examples(examples)}", flush=True)
    for line in format_split_audit(examples):
        print(line, flush=True)
    print(
        f"train_pool={len(train_pool)} counts={class_counts(train_pool)} target={args.train_count}",
        flush=True,
    )
    print(
        f"val_pool={len(val_pool)} counts={class_counts(val_pool)} "
        f"target={args.eval_count_per_split}",
        flush=True,
    )
    print(
        f"test_pool={len(test_pool)} counts={class_counts(test_pool)} "
        f"target={args.eval_count_per_split}",
        flush=True,
    )
    if args.dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_config.json",
        {
            "args": vars(args),
            "base_modalities": list(base_modalities),
            "modality_sets": [list(item) for item in modality_sets],
            "dataset_summary": summarize_examples(examples),
            "train_pool_counts": class_counts(train_pool),
            "val_pool_counts": class_counts(val_pool),
            "test_pool_counts": class_counts(test_pool),
        },
    )
    summaries: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for modalities in modality_sets:
        name = modality_set_name(modalities)
        print(f"probe start: modalities={name}", flush=True)
        try:
            summary = run_probe_for_modalities(
                modalities=modalities,
                train_pool=train_pool,
                val_pool=val_pool,
                test_pool=test_pool,
                cache_dir=cache_dir,
                specs=specs,
                dataset_root=dataset_root,
                args=args,
                device=device,
                output_dir=output_dir,
            )
        except ValueError as exc:
            reason = str(exc)
            print(f"probe skipped: modalities={name} reason={reason}", flush=True)
            skipped = {"modalities": list(modalities), "skipped": True, "reason": reason}
            summaries.append(skipped)
            skipped_rows.append({"modalities": name, "reason": reason})
            write_json(output_dir / "summary.json", {"probes": summaries})
            write_skipped(output_dir / "skipped.csv", skipped_rows)
            continue
        summaries.append(summary)
        for split, metrics_payload in summary["metrics"].items():
            metrics = ProbeMetrics(**metrics_payload)
            metric_rows.append(
                metric_row(
                    modality_name=name,
                    split=split,
                    metrics=metrics,
                    feature_dim=int(summary["feature_dim"]),
                    missing_count=int(summary["missing_counts"][split]),
                )
            )
        val_metrics = summary["metrics"]["val"]
        test_metrics = summary["metrics"]["test"]
        print(
            f"probe done: modalities={name} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_fp={val_metrics['false_positive']} val_fn={val_metrics['false_negative']} "
            f"val_auc={val_metrics['auc']} "
            f"test_acc={test_metrics['accuracy']:.4f} test_f1={test_metrics['f1']:.4f} "
            f"test_fp={test_metrics['false_positive']} test_fn={test_metrics['false_negative']} "
            f"test_auc={test_metrics['auc']}",
            flush=True,
        )
        write_metrics(output_dir / "metrics.csv", metric_rows)
        write_json(output_dir / "summary.json", {"probes": summaries})
    write_metrics(output_dir / "metrics.csv", metric_rows)
    write_skipped(output_dir / "skipped.csv", skipped_rows)
    write_json(output_dir / "summary.json", {"probes": summaries})
    print(f"wrote: {output_dir / 'metrics.csv'}", flush=True)
    print(f"wrote: {output_dir / 'summary.json'}", flush=True)
    print(f"wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
