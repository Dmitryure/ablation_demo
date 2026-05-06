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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (
    LabeledVideoDataset,
    VideoExample,
    build_real_fake_examples,
    collate_labeled_video_batch,
    format_split_audit,
    summarize_examples,
    summarize_split_audit,
    write_dataset_manifest,
)
from feature_cache import (
    CachedFeatureDataset,
    FeatureCacheSpec,
    build_feature_cache_specs,
    collate_cached_feature_batch,
    feature_cache_item_exists,
    feature_cache_spec_dir,
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
CHECKPOINT_METRICS = (
    "val_accuracy",
    "val_balanced_accuracy",
    "val_f1",
    "val_loss",
    "train_accuracy",
    "train_balanced_accuracy",
    "train_f1",
    "train_loss",
)


@dataclass(frozen=True)
class PredictionRow:
    path: str
    class_name: str
    label: int
    probability: float
    prediction: int
    split: str


@dataclass(frozen=True)
class DiagnosticRow:
    path: str
    class_name: str
    label: int
    probability: float
    prediction: int
    split: str
    generator_id: str
    modality_name: str
    modality_gate_weight: float
    modality_expert_logit: float
    modality_mixed_logit_contribution: float
    token_attention_sum: float | None


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    specificity: float = 0.0
    negative_predictive_value: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    matthews_corrcoef: float = 0.0
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0


@dataclass(frozen=True)
class EpochTrainResult:
    loss: float
    accuracy: float
    elapsed_seconds: float
    metrics: BinaryMetrics = field(default_factory=BinaryMetrics)


@dataclass(frozen=True)
class EpochEvalResult:
    loss: float
    accuracy: float
    elapsed_seconds: float
    metrics: BinaryMetrics = field(default_factory=BinaryMetrics)


@dataclass(frozen=True)
class CachedLoaderConfig:
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None


@dataclass(frozen=True)
class ModalityDropoutConfig:
    default_probability: float = 0.0
    modality_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRegularizationConfig:
    modality_dropout: ModalityDropoutConfig = field(default_factory=ModalityDropoutConfig)
    gate_entropy_weight: float = 0.0


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
    parser.add_argument(
        "--balanced-total",
        type=int,
        default=None,
        help=(
            "Select this many videos across the whole dataset with equal real/fake counts, "
            "then derive train/val/test splits from that selected set."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--modality-lr",
        nargs="*",
        default=None,
        help="Optional branch LR overrides, e.g. rgb=0.0003 fau=0.0001.",
    )
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=None,
        help="Training-only probability for dropping each enabled modality from a batch.",
    )
    parser.add_argument(
        "--depth-dropout",
        type=float,
        default=None,
        help="Training-only depth dropout probability. Overrides --modality-dropout for depth.",
    )
    parser.add_argument(
        "--gate-entropy-weight",
        type=float,
        default=None,
        help="Weight for gated-head entropy regularization. Positive values discourage gate collapse.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--head-type", choices=HEAD_TYPES, default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument(
        "--checkpoint-metric",
        choices=CHECKPOINT_METRICS,
        default="val_accuracy",
        help="Metric used for best.pt and early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without checkpoint-metric improvement. 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum checkpoint-metric improvement required to reset early stopping.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument(
        "--prefer-cached-selection",
        action="store_true",
        help=(
            "Prefer examples with existing valid cached features when selecting train/val/test "
            "examples. Class balance and split boundaries are still preserved."
        ),
    )
    parser.add_argument(
        "--warm-start-rounds",
        action="store_true",
        help="Initialize each train-count round from the previous round checkpoint for the same modality set.",
    )
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


def _optional_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"`{field_name}` must be a boolean.")
    return value


def _optional_nonnegative_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`{field_name}` must be a non-negative integer.")
    return value


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{field_name}` must be a positive integer.")
    return value


def _optional_probability(
    value: Any, field_name: str, default: float | None = None
) -> float | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"`{field_name}` must be a probability in [0.0, 1.0).")
    probability = float(value)
    if probability < 0.0 or probability >= 1.0:
        raise ValueError(f"`{field_name}` must be in [0.0, 1.0), got {probability}.")
    return probability


def _optional_nonnegative_float(value: Any, field_name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError(f"`{field_name}` must be a non-negative number.")
    return float(value)


def resolve_cached_loader_config(config: Mapping[str, Any]) -> CachedLoaderConfig:
    training = config.get("training", {})
    if training is None:
        training = {}
    if not isinstance(training, Mapping):
        raise ValueError("Config `training` must be a mapping when provided.")
    loader = training.get("cached_loader", {})
    if loader is None:
        loader = {}
    if not isinstance(loader, Mapping):
        raise ValueError("Config `training.cached_loader` must be a mapping when provided.")

    num_workers = _optional_nonnegative_int(
        loader.get("num_workers"),
        "training.cached_loader.num_workers",
        0,
    )
    pin_memory = _optional_bool(
        loader.get("pin_memory"),
        "training.cached_loader.pin_memory",
        False,
    )
    persistent_workers = _optional_bool(
        loader.get("persistent_workers"),
        "training.cached_loader.persistent_workers",
        False,
    )
    prefetch_factor = _optional_positive_int(
        loader.get("prefetch_factor"),
        "training.cached_loader.prefetch_factor",
    )
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    return CachedLoaderConfig(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def resolve_training_regularization_config(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> TrainingRegularizationConfig:
    training = config.get("training", {})
    if training is None:
        training = {}
    if not isinstance(training, Mapping):
        raise ValueError("Config `training` must be a mapping when provided.")
    dropout = training.get("modality_dropout", {})
    if dropout is None:
        dropout = {}
    if not isinstance(dropout, Mapping):
        raise ValueError("Config `training.modality_dropout` must be a mapping when provided.")

    default_probability = _optional_probability(
        dropout.get("default_probability"),
        "training.modality_dropout.default_probability",
        0.0,
    )
    modality_probabilities_raw = dropout.get("modality_probabilities", {})
    if modality_probabilities_raw is None:
        modality_probabilities_raw = {}
    if not isinstance(modality_probabilities_raw, Mapping):
        raise ValueError(
            "Config `training.modality_dropout.modality_probabilities` must be a mapping."
        )
    modality_probabilities = {
        str(modality): float(
            _optional_probability(
                probability,
                f"training.modality_dropout.modality_probabilities.{modality}",
                0.0,
            )
        )
        for modality, probability in modality_probabilities_raw.items()
    }
    if args.modality_dropout is not None:
        default_probability = float(
            _optional_probability(args.modality_dropout, "--modality-dropout", 0.0)
        )
    if args.depth_dropout is not None:
        modality_probabilities["depth"] = float(
            _optional_probability(args.depth_dropout, "--depth-dropout", 0.0)
        )

    gate_entropy_weight = _optional_nonnegative_float(
        training.get("gate_entropy_weight"),
        "training.gate_entropy_weight",
        0.0,
    )
    if args.gate_entropy_weight is not None:
        gate_entropy_weight = _optional_nonnegative_float(
            args.gate_entropy_weight,
            "--gate-entropy-weight",
            0.0,
        )

    return TrainingRegularizationConfig(
        modality_dropout=ModalityDropoutConfig(
            default_probability=float(default_probability or 0.0),
            modality_probabilities=modality_probabilities,
        ),
        gate_entropy_weight=gate_entropy_weight,
    )


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


def video_metadata_summary(examples: Sequence[VideoExample]) -> dict[str, Any]:
    return summarize_split_audit(examples)


def _random_rank_by_path(items: Sequence[VideoExample], seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {str(example.path): rng.random() for example in items}


def _random_rank_by_value(items: Sequence[str], seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {item: rng.random() for item in items}


def _shuffled(
    items: Sequence[VideoExample],
    seed: int,
    cache_score_by_path: Mapping[str, int] | None = None,
) -> list[VideoExample]:
    result = list(items)
    if cache_score_by_path is None:
        random.Random(seed).shuffle(result)
        return result
    ranks = _random_rank_by_path(result, seed)
    result.sort(
        key=lambda example: (
            -cache_score_by_path.get(str(example.path), 0),
            ranks[str(example.path)],
        )
    )
    return result


def _balanced_fake_order(
    fake_examples: Sequence[VideoExample],
    seed: int,
    cache_score_by_path: Mapping[str, int] | None = None,
) -> list[VideoExample]:
    by_identity: dict[str, list[VideoExample]] = defaultdict(list)
    for example in fake_examples:
        by_identity[example.identity_id or "unknown"].append(example)
    identity_seed_offsets = {identity: index for index, identity in enumerate(sorted(by_identity))}
    for identity, examples in by_identity.items():
        by_identity[identity] = _shuffled(
            examples,
            seed + identity_seed_offsets[identity] + 1,
            cache_score_by_path=cache_score_by_path,
        )
    identities = sorted(by_identity)
    if cache_score_by_path is None:
        random.Random(seed).shuffle(identities)
    else:
        ranks = _random_rank_by_value(identities, seed)
        ordered: list[VideoExample] = []
        active = list(identities)
        while active:
            active.sort(
                key=lambda identity: (
                    -cache_score_by_path.get(str(by_identity[identity][0].path), 0),
                    ranks[identity],
                )
            )
            identity = active[0]
            ordered.append(by_identity[identity].pop(0))
            if not by_identity[identity]:
                active.pop(0)
        return ordered

    ordered: list[VideoExample] = []
    while identities:
        next_identities: list[str] = []
        for identity in identities:
            examples = by_identity[identity]
            if examples:
                ordered.append(examples.pop(0))
            if examples:
                next_identities.append(identity)
        identities = next_identities
    return ordered


def build_balanced_train_order(
    examples: Sequence[VideoExample],
    seed: int,
    cache_score_by_path: Mapping[str, int] | None = None,
) -> list[VideoExample]:
    real = _shuffled(
        [example for example in examples if example.class_name == "real"],
        seed,
        cache_score_by_path=cache_score_by_path,
    )
    fake = _balanced_fake_order(
        [example for example in examples if example.class_name == "fake"],
        seed + 1,
        cache_score_by_path=cache_score_by_path,
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
    cache_score_by_path: Mapping[str, int] | None = None,
) -> list[VideoExample]:
    real = _shuffled(
        [example for example in examples if example.class_name == "real"],
        seed,
        cache_score_by_path=cache_score_by_path,
    )
    fake = _balanced_fake_order(
        [example for example in examples if example.class_name == "fake"],
        seed + 1,
        cache_score_by_path=cache_score_by_path,
    )
    per_class = min(target_count // 2, len(real), len(fake))
    selected: list[VideoExample] = []
    for index in range(per_class):
        selected.append(real[index])
        selected.append(fake[index])
    return selected


def _with_split(example: VideoExample, split: str) -> VideoExample:
    return VideoExample(
        path=example.path,
        label=example.label,
        class_name=example.class_name,
        source_id=example.source_id,
        split=split,
        identity_id=example.identity_id,
        generator_id=example.generator_id,
        source_id_kind=example.source_id_kind,
        age_bin=example.age_bin,
        gender=example.gender,
        ethnicity=example.ethnicity,
        emotion=example.emotion,
    )


def _split_class_examples(
    examples: Sequence[VideoExample],
    train_count: int,
    val_count: int,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    train = [_with_split(example, "train") for example in examples[:train_count]]
    val = [
        _with_split(example, "val") for example in examples[train_count : train_count + val_count]
    ]
    test = [_with_split(example, "test") for example in examples[train_count + val_count :]]
    return train, val, test


def split_balanced_total_examples(
    examples: Sequence[VideoExample],
    balanced_total: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    cache_score_by_path: Mapping[str, int] | None = None,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    if balanced_total <= 0:
        raise ValueError("`--balanced-total` must be positive.")
    if balanced_total % 2 != 0:
        raise ValueError("`--balanced-total` must be even for equal real/fake selection.")
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("`--train-ratio` must be in (0.0, 1.0).")
    if val_ratio <= 0.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("`--val-ratio` must be in (0.0, 1.0) and leave room for test.")

    per_class = balanced_total // 2
    real = _shuffled(
        [example for example in examples if example.class_name == "real"],
        seed,
        cache_score_by_path=cache_score_by_path,
    )
    fake = _balanced_fake_order(
        [example for example in examples if example.class_name == "fake"],
        seed + 1,
        cache_score_by_path=cache_score_by_path,
    )
    if len(real) < per_class or len(fake) < per_class:
        max_total = 2 * min(len(real), len(fake))
        raise ValueError(
            f"`--balanced-total {balanced_total}` exceeds available balanced total "
            f"{max_total} (real={len(real)}, fake={len(fake)})."
        )

    real = real[:per_class]
    fake = fake[:per_class]
    train_per_class = int(per_class * train_ratio)
    val_per_class = int(per_class * val_ratio)
    if train_per_class <= 0 or val_per_class <= 0:
        raise ValueError("`--balanced-total` is too small for non-empty train/val splits.")

    real_train, real_val, real_test = _split_class_examples(real, train_per_class, val_per_class)
    fake_train, fake_val, fake_test = _split_class_examples(fake, train_per_class, val_per_class)
    return (
        build_balanced_train_order([*real_train, *fake_train], seed + 11),
        select_balanced_subset([*real_val, *fake_val], len(real_val) + len(fake_val), seed + 23),
        select_balanced_subset(
            [*real_test, *fake_test],
            len(real_test) + len(fake_test),
            seed + 29,
        ),
    )


def split_examples(
    examples: Sequence[VideoExample],
    eval_count_per_split: int,
    seed: int,
    cache_score_by_path: Mapping[str, int] | None = None,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    train = [example for example in examples if example.split == "train"]
    val = select_balanced_subset(
        [example for example in examples if example.split == "val"],
        eval_count_per_split,
        seed + 11,
        cache_score_by_path=cache_score_by_path,
    )
    test = select_balanced_subset(
        [example for example in examples if example.split == "test"],
        eval_count_per_split,
        seed + 23,
        cache_score_by_path=cache_score_by_path,
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


def resolve_warm_start_checkpoint(
    previous_summary: Mapping[str, Any] | None,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    if previous_summary is None:
        return None
    checkpoint = previous_summary.get("best_checkpoint")
    if checkpoint is None:
        raise ValueError("Previous round summary is missing `best_checkpoint`.")
    return Path(str(checkpoint))


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
    weight_decay: float,
) -> torch.optim.Optimizer:
    if weight_decay < 0.0:
        raise ValueError("`--weight-decay` must be non-negative.")
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
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


def binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.float32)
    return float((predictions == labels).to(dtype=torch.float32).mean().item())


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def matthews_corrcoef(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return safe_divide(float(tp * tn - fp * fn), denominator)


def binary_metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> BinaryMetrics:
    total = tp + tn + fp + fn
    precision = safe_divide(float(tp), float(tp + fp))
    recall = safe_divide(float(tp), float(tp + fn))
    specificity = safe_divide(float(tn), float(tn + fp))
    negative_predictive_value = safe_divide(float(tn), float(tn + fn))
    false_positive_rate = safe_divide(float(fp), float(fp + tn))
    false_negative_rate = safe_divide(float(fn), float(fn + tp))
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return BinaryMetrics(
        accuracy=safe_divide(float(tp + tn), float(total)),
        balanced_accuracy=(specificity + recall) / 2.0,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        negative_predictive_value=negative_predictive_value,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        matthews_corrcoef=matthews_corrcoef(tp, tn, fp, fn),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def binary_confusion_from_predictions(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int, int, int]:
    labels_int = labels.view(-1).to(dtype=torch.long)
    predictions_int = predictions.view(-1).to(dtype=torch.long)
    fake_mask = labels_int == 1
    real_mask = labels_int == 0
    true_positive = int(((predictions_int == 1) & fake_mask).sum().item())
    true_negative = int(((predictions_int == 0) & real_mask).sum().item())
    false_positive = int(((predictions_int == 1) & real_mask).sum().item())
    false_negative = int(((predictions_int == 0) & fake_mask).sum().item())
    return true_positive, true_negative, false_positive, false_negative


def binary_confusion_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[int, int, int, int]:
    predictions = (torch.sigmoid(logits.detach()) >= 0.5).to(dtype=torch.long)
    return binary_confusion_from_predictions(predictions, labels)


def metrics_row_fields(prefix: str) -> tuple[str, ...]:
    return (
        f"{prefix}_balanced_accuracy",
        f"{prefix}_precision",
        f"{prefix}_recall",
        f"{prefix}_f1",
        f"{prefix}_specificity",
        f"{prefix}_negative_predictive_value",
        f"{prefix}_false_positive_rate",
        f"{prefix}_false_negative_rate",
        f"{prefix}_matthews_corrcoef",
        f"{prefix}_true_positive",
        f"{prefix}_true_negative",
        f"{prefix}_false_positive",
        f"{prefix}_false_negative",
    )


def metrics_row_values(prefix: str, metrics: BinaryMetrics) -> dict[str, Any]:
    return {
        f"{prefix}_balanced_accuracy": f"{metrics.balanced_accuracy:.8f}",
        f"{prefix}_precision": f"{metrics.precision:.8f}",
        f"{prefix}_recall": f"{metrics.recall:.8f}",
        f"{prefix}_f1": f"{metrics.f1:.8f}",
        f"{prefix}_specificity": f"{metrics.specificity:.8f}",
        f"{prefix}_negative_predictive_value": f"{metrics.negative_predictive_value:.8f}",
        f"{prefix}_false_positive_rate": f"{metrics.false_positive_rate:.8f}",
        f"{prefix}_false_negative_rate": f"{metrics.false_negative_rate:.8f}",
        f"{prefix}_matthews_corrcoef": f"{metrics.matthews_corrcoef:.8f}",
        f"{prefix}_true_positive": metrics.true_positive,
        f"{prefix}_true_negative": metrics.true_negative,
        f"{prefix}_false_positive": metrics.false_positive,
        f"{prefix}_false_negative": metrics.false_negative,
    }


def modality_dropout_probability(modality: str, config: ModalityDropoutConfig) -> float:
    return float(config.modality_probabilities.get(modality, config.default_probability))


def sample_dropped_modalities(
    modalities: Sequence[str],
    config: ModalityDropoutConfig,
) -> tuple[str, ...]:
    if not modalities:
        return ()
    dropped = [
        modality
        for modality in modalities
        if modality_dropout_probability(modality, config) > 0.0
        and float(torch.rand(()).item()) < modality_dropout_probability(modality, config)
    ]
    if len(dropped) >= len(modalities):
        keep_index = int(torch.randint(len(modalities), (1,)).item())
        dropped = [modality for index, modality in enumerate(modalities) if index != keep_index]
    return tuple(dropped)


def apply_training_modality_dropout(
    batch: Mapping[str, Any],
    modalities: Sequence[str],
    config: ModalityDropoutConfig,
) -> dict[str, Any]:
    dropped = sample_dropped_modalities(modalities, config)
    if not dropped:
        return dict(batch)
    return {**batch, "dropped_modalities": dropped}


def gate_entropy_regularization(
    output: Any,
    weight: float,
) -> torch.Tensor | None:
    if weight <= 0.0:
        return None
    gate_weights = output.diagnostics.get("modality_gate_weights")
    valid_mask = output.diagnostics.get("modality_valid_mask")
    if not isinstance(gate_weights, torch.Tensor) or not isinstance(valid_mask, torch.Tensor):
        return None
    if gate_weights.ndim != 2 or valid_mask.ndim != 1:
        return None
    valid_mask = valid_mask.to(device=gate_weights.device, dtype=torch.bool)
    valid_count = int(valid_mask.sum().item())
    if valid_count <= 1:
        return None
    valid_weights = gate_weights * valid_mask.to(dtype=gate_weights.dtype).unsqueeze(0)
    entropy = -(valid_weights * valid_weights.clamp_min(torch.finfo(gate_weights.dtype).tiny).log())
    entropy = entropy.sum(dim=1)
    normalized_entropy = entropy / torch.log(gate_weights.new_tensor(float(valid_count))).clamp_min(
        torch.finfo(gate_weights.dtype).tiny
    )
    return -float(weight) * normalized_entropy.mean()


def train_one_epoch(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    regularization_config: TrainingRegularizationConfig | None = None,
) -> EpochTrainResult:
    device = model_device(model)
    model.train()
    resolved_regularization = regularization_config or TrainingRegularizationConfig()
    total_loss = 0.0
    total_count = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    start = time.perf_counter()
    for batch in loader:
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        training_batch = apply_training_modality_dropout(
            batch,
            model.pipeline.enabled_modalities,
            resolved_regularization.modality_dropout,
        )
        output = model(move_tensor_batch_to_device(training_batch, device))
        loss = loss_fn(output.logits, labels)
        entropy_loss = gate_entropy_regularization(
            output,
            resolved_regularization.gate_entropy_weight,
        )
        if entropy_loss is not None:
            loss = loss + entropy_loss
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        tp, tn, fp, fn = binary_confusion_from_logits(output.logits, labels)
        total_loss += float(loss.item()) * count
        total_count += count
        true_positive += tp
        true_negative += tn
        false_positive += fp
        false_negative += fn
    elapsed = time.perf_counter() - start
    metrics = binary_metrics_from_counts(
        true_positive, true_negative, false_positive, false_negative
    )
    return EpochTrainResult(
        loss=total_loss / total_count,
        accuracy=metrics.accuracy,
        elapsed_seconds=elapsed,
        metrics=metrics,
    )


def evaluate_loss_accuracy(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    loss_fn: torch.nn.Module,
) -> EpochEvalResult:
    device = model_device(model)
    model.eval()
    total_loss = 0.0
    total_count = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            output = model(move_tensor_batch_to_device(batch, device))
            loss = loss_fn(output.logits, labels)
            count = int(labels.numel())
            tp, tn, fp, fn = binary_confusion_from_logits(output.logits, labels)
            total_loss += float(loss.item()) * count
            total_count += count
            true_positive += tp
            true_negative += tn
            false_positive += fp
            false_negative += fn
    elapsed = time.perf_counter() - start
    metrics = binary_metrics_from_counts(
        true_positive, true_negative, false_positive, false_negative
    )
    return EpochEvalResult(
        loss=total_loss / total_count,
        accuracy=metrics.accuracy,
        elapsed_seconds=elapsed,
        metrics=metrics,
    )


def metric_is_loss(metric_name: str) -> bool:
    return metric_name.endswith("_loss")


def checkpoint_metric_value(
    metric_name: str,
    train_result: EpochTrainResult,
    val_result: EpochEvalResult,
) -> float:
    if metric_name == "train_accuracy":
        return train_result.accuracy
    if metric_name == "train_balanced_accuracy":
        return train_result.metrics.balanced_accuracy
    if metric_name == "train_f1":
        return train_result.metrics.f1
    if metric_name == "train_loss":
        return train_result.loss
    if metric_name == "val_accuracy":
        return val_result.accuracy
    if metric_name == "val_balanced_accuracy":
        return val_result.metrics.balanced_accuracy
    if metric_name == "val_f1":
        return val_result.metrics.f1
    if metric_name == "val_loss":
        return val_result.loss
    raise ValueError(f"Unsupported checkpoint metric: {metric_name}")


def is_metric_improvement(
    metric_name: str,
    value: float,
    best_value: float | None,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if metric_is_loss(metric_name):
        return value < best_value - min_delta
    return value > best_value + min_delta


def predict_rows(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    diagnostic_rows: list[DiagnosticRow] | None = None,
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
            if diagnostic_rows is not None:
                append_diagnostic_rows(
                    diagnostic_rows=diagnostic_rows,
                    output=output,
                    batch=batch,
                    labels=labels,
                    probabilities=probabilities,
                    predictions=predictions,
                )
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


def append_diagnostic_rows(
    diagnostic_rows: list[DiagnosticRow],
    output: Any,
    batch: Mapping[str, Any],
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    gate_weights = output.diagnostics.get("modality_gate_weights")
    expert_logits = output.diagnostics.get("modality_expert_logits")
    if not isinstance(gate_weights, torch.Tensor) or not isinstance(expert_logits, torch.Tensor):
        return
    gate_weights = gate_weights.detach().cpu()
    expert_logits = expert_logits.detach().cpu()
    if gate_weights.ndim != 2 or expert_logits.shape != gate_weights.shape:
        return

    modality_names = tuple(str(name) for name in output.fusion.modality_names)
    if gate_weights.shape[1] > len(modality_names):
        return
    token_attention_sums = diagnostic_token_attention_sums(
        output=output,
        modality_count=gate_weights.shape[1],
    )
    paths = tuple(str(path) for path in batch["path"])
    class_names = tuple(str(class_name) for class_name in batch["class_name"])
    splits = tuple(str(split) for split in batch["split"])
    labels_list = [int(value) for value in labels.view(-1).tolist()]
    predictions_list = [int(value) for value in predictions.view(-1).tolist()]
    probabilities_list = [float(value) for value in probabilities.view(-1).tolist()]

    for batch_index, path in enumerate(paths):
        generator_id = infer_prediction_generator(path, class_names[batch_index])
        for modality_index in range(gate_weights.shape[1]):
            gate_weight = float(gate_weights[batch_index, modality_index].item())
            expert_logit = float(expert_logits[batch_index, modality_index].item())
            token_attention_sum = (
                None
                if token_attention_sums is None
                else float(token_attention_sums[batch_index, modality_index].item())
            )
            diagnostic_rows.append(
                DiagnosticRow(
                    path=path,
                    class_name=class_names[batch_index],
                    label=labels_list[batch_index],
                    probability=probabilities_list[batch_index],
                    prediction=predictions_list[batch_index],
                    split=splits[batch_index],
                    generator_id=generator_id,
                    modality_name=modality_names[modality_index],
                    modality_gate_weight=gate_weight,
                    modality_expert_logit=expert_logit,
                    modality_mixed_logit_contribution=gate_weight * expert_logit,
                    token_attention_sum=token_attention_sum,
                )
            )


def diagnostic_token_attention_sums(
    output: Any,
    modality_count: int,
) -> torch.Tensor | None:
    token_attention = output.diagnostics.get("token_attention_weights")
    if not isinstance(token_attention, torch.Tensor) or token_attention.ndim != 2:
        return None
    token_attention = token_attention.detach().cpu()
    modality_ids = output.fusion.modality_ids.detach().cpu()
    if modality_ids.ndim != 1 or modality_ids.numel() != token_attention.shape[1]:
        return None
    ordered_ids: list[int] = []
    for modality_id in modality_ids.tolist():
        int_id = int(modality_id)
        if int_id not in ordered_ids:
            ordered_ids.append(int_id)
        if len(ordered_ids) == modality_count:
            break
    if len(ordered_ids) != modality_count:
        return None
    attention_sums = []
    for modality_id in ordered_ids:
        mask = modality_ids == modality_id
        attention_sums.append(token_attention[:, mask].sum(dim=1))
    return torch.stack(attention_sums, dim=1)


def infer_prediction_generator(path: str, class_name: str) -> str:
    if class_name == "real":
        return "real"
    parts = Path(path).parts
    if "fake" not in parts:
        return "unknown"
    index = parts.index("fake")
    if index + 1 >= len(parts):
        return "unknown"
    return parts[index + 1]


def prediction_rows_metrics(rows: Sequence[PredictionRow]) -> BinaryMetrics:
    labels = torch.tensor([row.label for row in rows], dtype=torch.long)
    predictions = torch.tensor([row.prediction for row in rows], dtype=torch.long)
    tp, tn, fp, fn = binary_confusion_from_predictions(predictions, labels)
    return binary_metrics_from_counts(tp, tn, fp, fn)


def build_cached_loader(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    batch_size: int,
    shuffle: bool,
    dataset_root: Path,
    loader_config: CachedLoaderConfig | None = None,
) -> DataLoader[dict[str, Any]]:
    resolved_loader_config = loader_config or CachedLoaderConfig()
    dataset = CachedFeatureDataset(
        examples=examples,
        cache_dir=cache_dir,
        spec_by_modality=specs,
        modalities=modalities,
        strict=True,
        dataset_root=dataset_root,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_cached_feature_batch,
        "num_workers": resolved_loader_config.num_workers,
        "pin_memory": resolved_loader_config.pin_memory,
    }
    if resolved_loader_config.num_workers > 0:
        loader_kwargs["persistent_workers"] = resolved_loader_config.persistent_workers
        if resolved_loader_config.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = resolved_loader_config.prefetch_factor
    return DataLoader(
        dataset,
        **loader_kwargs,
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


def build_scan_progress(
    examples: Sequence[VideoExample], spec: FeatureCacheSpec, enabled: bool, label: str
):
    if not enabled:
        return None
    return tqdm(
        total=len(examples),
        desc=f"scan {label}/{spec.modality}" if label else f"scan {spec.modality}",
        unit="video",
        dynamic_ncols=True,
        leave=False,
    )


def update_scan_progress(
    progress,
    index: int,
    total: int,
    missing_count: int,
    progress_interval: int,
    label: str,
    modality: str,
    cache_from: Path,
    spec_id: str,
) -> None:
    if progress is not None:
        progress.update(1)
        return
    if index == total or index % progress_interval == 0:
        print(
            f"scan {label}: modality={modality} checked={index}/{total} missing={missing_count} "
            f"cache_from={cache_from} spec_id={spec_id}",
            flush=True,
        )


def is_missing_cache_example(
    example: VideoExample,
    cache_dir: Path,
    spec: FeatureCacheSpec,
    dataset_root: Path,
    overwrite: bool,
    root_exists: bool,
    skip_failure_keys: set[tuple[str, str, str]],
    spec_id: str,
) -> bool:
    failure_key = (spec_id, spec.modality, str(example.path))
    if failure_key in skip_failure_keys:
        return False
    if not overwrite and not root_exists:
        return True
    return overwrite or not feature_cache_item_exists(
        cache_dir,
        example,
        spec,
        dataset_root=dataset_root,
    )


def missing_examples_for_modality(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    spec: FeatureCacheSpec,
    dataset_root: Path,
    overwrite: bool,
    skip_failure_keys: set[tuple[str, str, str]],
    progress_bar: bool = False,
    label: str = "",
    progress_every: int = 100,
) -> list[VideoExample]:
    spec_id = feature_cache_spec_id(spec)
    modality_cache_root = feature_cache_spec_dir(cache_dir, spec)
    root_exists = modality_cache_root.exists()
    progress = build_scan_progress(examples, spec, progress_bar, label)
    progress_interval = max(1, progress_every)
    missing: list[VideoExample] = []
    try:
        for index, example in enumerate(examples, start=1):
            if is_missing_cache_example(
                example=example,
                cache_dir=cache_dir,
                spec=spec,
                dataset_root=dataset_root,
                overwrite=overwrite,
                root_exists=root_exists,
                skip_failure_keys=skip_failure_keys,
                spec_id=spec_id,
            ):
                missing.append(example)
            update_scan_progress(
                progress=progress,
                index=index,
                total=len(examples),
                missing_count=len(missing),
                progress_interval=progress_interval,
                label=label,
                modality=spec.modality,
                cache_from=modality_cache_root,
                spec_id=spec_id,
            )
        return missing
    finally:
        if progress is not None:
            progress.close()


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
        modality_cache_root = feature_cache_spec_dir(cache_dir, spec)
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


def cache_score_for_example(
    example: VideoExample,
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
) -> int:
    return sum(
        int(feature_cache_item_exists(cache_dir, example, specs[modality], dataset_root))
        for modality in modalities
    )


def build_cache_score_by_path(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
) -> dict[str, int]:
    return {
        str(example.path): cache_score_for_example(
            example=example,
            cache_dir=cache_dir,
            specs=specs,
            modalities=modalities,
            dataset_root=dataset_root,
        )
        for example in examples
    }


def cache_score_summary(
    examples: Sequence[VideoExample],
    cache_score_by_path: Mapping[str, int],
    modality_count: int,
) -> dict[str, int]:
    full = 0
    partial = 0
    empty = 0
    total_cached_modalities = 0
    for example in examples:
        score = cache_score_by_path.get(str(example.path), 0)
        total_cached_modalities += score
        if score >= modality_count:
            full += 1
        elif score > 0:
            partial += 1
        else:
            empty += 1
    return {
        "examples": len(examples),
        "full": full,
        "partial": partial,
        "empty": empty,
        "cached_modalities": total_cached_modalities,
        "possible_modalities": len(examples) * modality_count,
    }


def cache_key_for_example(example: VideoExample, dataset_root: Path) -> str:
    path = example.path.resolve()
    root = dataset_root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def metadata_matches_example(
    metadata: Mapping[str, Any],
    example: VideoExample,
    spec: FeatureCacheSpec,
    dataset_root: Path,
) -> bool:
    if metadata.get("spec_id") != feature_cache_spec_id(spec):
        return False
    video = metadata.get("video")
    if not isinstance(video, Mapping):
        return False
    if str(video.get("relative_path", "")) != cache_key_for_example(example, dataset_root):
        return False
    try:
        stat = example.path.stat()
    except OSError:
        return False
    return int(video.get("size", -1)) == int(stat.st_size) and int(
        video.get("mtime_ns", -1)
    ) == int(stat.st_mtime_ns)


def cached_example_keys_for_modality(
    examples_by_cache_key: Mapping[str, VideoExample],
    cache_dir: Path,
    spec: FeatureCacheSpec,
    dataset_root: Path,
    progress_every: int | None = None,
) -> set[str]:
    spec_dir = feature_cache_spec_dir(cache_dir, spec)
    if not spec_dir.is_dir():
        return set()
    cached_keys: set[str] = set()
    checked = 0
    start = time.perf_counter()
    for metadata_path in spec_dir.glob("*.json"):
        checked += 1
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, Mapping):
            continue
        video = metadata.get("video")
        if not isinstance(video, Mapping):
            continue
        key = str(video.get("relative_path", ""))
        example = examples_by_cache_key.get(key)
        if example is None:
            continue
        if metadata_matches_example(metadata, example, spec, dataset_root):
            cached_keys.add(key)
        if progress_every is not None and checked % progress_every == 0:
            elapsed = time.perf_counter() - start
            print(
                f"cache selection: modality={spec.modality} "
                f"metadata_checked={checked} matched={len(cached_keys)} "
                f"elapsed={elapsed:.1f}s cache_from={spec_dir} spec_id={feature_cache_spec_id(spec)}",
                flush=True,
            )
    return cached_keys


def select_fully_cached_examples(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    progress_every: int | None = None,
) -> tuple[list[VideoExample], dict[str, int]]:
    examples_by_cache_key = {
        cache_key_for_example(example, dataset_root): example for example in examples
    }
    cached_key_sets: dict[str, set[str]] = {}
    for modality in modalities:
        spec = specs[modality]
        spec_dir = feature_cache_spec_dir(cache_dir, spec)
        spec_id = feature_cache_spec_id(spec)
        print(
            f"cache selection: scan metadata modality={modality} "
            f"cache_from={spec_dir} spec_id={spec_id}",
            flush=True,
        )
        cached_key_sets[modality] = cached_example_keys_for_modality(
            examples_by_cache_key=examples_by_cache_key,
            cache_dir=cache_dir,
            spec=spec,
            dataset_root=dataset_root,
            progress_every=progress_every,
        )
        print(
            f"cache selection: modality={modality} matched={len(cached_key_sets[modality])} "
            f"cache_from={spec_dir} spec_id={spec_id}",
            flush=True,
        )

    if not cached_key_sets:
        cached_keys: set[str] = set()
    else:
        cached_keys = set.intersection(*cached_key_sets.values())
    selected = [
        example
        for example in examples
        if cache_key_for_example(example, dataset_root) in cached_keys
    ]
    counts = {f"{modality}_cached": len(keys) for modality, keys in sorted(cached_key_sets.items())}
    counts.update(
        {
            "dataset_examples": len(examples),
            "fully_cached_examples": len(selected),
            "modalities": len(modalities),
        }
    )
    return selected, counts


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


def rebalance_eval_examples(
    examples: Sequence[VideoExample],
    target_count: int,
    seed: int,
    label: str,
) -> list[VideoExample]:
    balanced = select_balanced_subset(examples, target_count=target_count, seed=seed)
    if len(balanced) != len(examples):
        print(
            f"eval rebalance {label}: input={len(examples)} output={len(balanced)} "
            f"counts={class_counts(balanced)}",
            flush=True,
        )
    return balanced


def chunk_examples(
    examples: Sequence[VideoExample],
    chunk_size: int,
) -> list[list[VideoExample]]:
    size = max(1, chunk_size)
    return [list(examples[index : index + size]) for index in range(0, len(examples), size)]


def group_modalities_by_clip_spec(
    modalities: Sequence[str],
    specs: Mapping[str, FeatureCacheSpec],
) -> dict[tuple[int, int], tuple[str, ...]]:
    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    for modality in modalities:
        spec = specs[modality]
        grouped[(spec.frame_count, spec.image_size)].append(modality)
    return {key: tuple(value) for key, value in grouped.items()}


def extraction_modality_groups(
    modalities: Sequence[str],
    specs: Mapping[str, FeatureCacheSpec],
    group_by_modality: bool = False,
) -> list[tuple[int, int, tuple[str, ...]]]:
    if group_by_modality:
        return [
            (specs[modality].frame_count, specs[modality].image_size, (modality,))
            for modality in modalities
        ]
    return [
        (frame_count, image_size, group_modalities)
        for (frame_count, image_size), group_modalities in group_modalities_by_clip_spec(
            modalities, specs
        ).items()
    ]


def examples_missing_any_modality(
    examples: Sequence[VideoExample],
    modalities: Sequence[str],
    missing_by_modality: Mapping[str, Sequence[VideoExample]],
) -> list[VideoExample]:
    missing_sets = {modality: set(missing_by_modality.get(modality, ())) for modality in modalities}
    return [
        example
        for example in examples
        if any(example in missing_sets[modality] for modality in modalities)
    ]


def build_raw_feature_batch(
    examples: Sequence[VideoExample],
    modalities: Sequence[str],
    frame_count: int,
    image_size: int,
) -> dict[str, Any]:
    dataset = LabeledVideoDataset(
        examples=examples,
        num_frames=dict.fromkeys(modalities, frame_count),
        image_size=image_size,
    )
    return collate_labeled_video_batch([dataset[index] for index in range(len(dataset))])


def append_cache_failure(
    failure_rows: list[dict[str, Any]],
    progress: dict[str, Any],
    spec: FeatureCacheSpec,
    example: VideoExample,
    exc: Exception,
) -> None:
    failure_rows.append(
        {
            "spec_id": feature_cache_spec_id(spec),
            "modality": spec.modality,
            "path": str(example.path),
            "error": str(exc),
        }
    )
    progress[spec.modality]["failed"] += 1


def write_cached_feature_items(
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    examples: Sequence[VideoExample],
    feature_batch: Mapping[str, Any],
    raw_batch: Mapping[str, Any],
    missing_sets: Mapping[str, set[VideoExample]],
    dataset_root: Path,
    progress: dict[str, Any],
) -> None:
    for example, item in zip(examples, split_feature_batch(feature_batch, raw_batch), strict=True):
        for modality in modalities:
            if example not in missing_sets[modality]:
                continue
            write_feature_cache_item(
                cache_dir=cache_dir,
                example=example,
                spec=specs[modality],
                item=item,
                dataset_root=dataset_root,
            )
            progress[modality]["written"] += 1


def cache_single_modality_example(
    example: VideoExample,
    cache_dir: Path,
    spec: FeatureCacheSpec,
    config: Mapping[str, Any],
    dataset_root: Path,
    progress: dict[str, Any],
    failure_rows: list[dict[str, Any]],
    fallback_pipelines: dict[str, Any],
) -> None:
    modality = spec.modality
    build_result = fallback_pipelines.get(modality)
    if build_result is None:
        build_result = build_fusion_pipeline(config=config, modalities=(modality,))
        build_result.pipeline.eval()
        fallback_pipelines[modality] = build_result
    try:
        raw_batch = build_raw_feature_batch(
            examples=[example],
            modalities=(modality,),
            frame_count=spec.frame_count,
            image_size=spec.image_size,
        )
        feature_batch = build_result.pipeline.prepare_features(raw_batch)
        item = split_feature_batch(feature_batch, raw_batch)[0]
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
            f"cache fallback: modality={modality} path={example.path} error={exc}",
            flush=True,
        )
        append_cache_failure(failure_rows, progress, spec, example, exc)


def initialize_feature_cache_progress(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    overwrite: bool,
    skip_failures: bool,
    progress_every: int,
    label: str,
    progress_bar: bool,
) -> tuple[dict[str, list[VideoExample]], dict[str, Any]]:
    skip_failure_keys = read_failure_keys(cache_dir) if skip_failures else set()
    progress: dict[str, Any] = {}
    progress_interval = max(1, progress_every)
    missing_by_modality: dict[str, list[VideoExample]] = {}
    for modality in modalities:
        spec = specs[modality]
        spec_id = feature_cache_spec_id(spec)
        cache_from = feature_cache_spec_dir(cache_dir, spec)
        missing = missing_examples_for_modality(
            examples=examples,
            cache_dir=cache_dir,
            spec=spec,
            dataset_root=dataset_root,
            overwrite=overwrite,
            skip_failure_keys=skip_failure_keys,
            progress_bar=progress_bar,
            label=label,
            progress_every=progress_interval,
        )
        missing_by_modality[modality] = missing
        progress[modality] = {
            "requested": len(examples),
            "missing_before": len(missing),
            "cached_before": 0,
            "skipped_failed": 0,
            "written": 0,
            "failed": 0,
            "cache_from": str(cache_from),
            "spec_id": spec_id,
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
            f"cached={cached_before} skipped_failed={skipped_failed} missing={len(missing)} "
            f"cache_from={cache_from} spec_id={spec_id}",
            flush=True,
        )
    return missing_by_modality, progress


def close_pipeline_result(build_result) -> None:
    build_result.pipeline.close()
    if build_result.device.type == "cuda":
        torch.cuda.empty_cache()


def build_cache_progress_display(
    group_examples: Sequence[VideoExample],
    group_name: str,
    label: str,
    enabled: bool,
):
    if not enabled:
        return None
    return tqdm(
        total=len(group_examples),
        desc=f"cache {label}/{group_name}",
        unit="video",
        dynamic_ncols=True,
        leave=True,
    )


def cache_feature_batch_or_fallback(
    batch_examples: Sequence[VideoExample],
    group_modalities: Sequence[str],
    frame_count: int,
    image_size: int,
    build_result,
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    config: Mapping[str, Any],
    dataset_root: Path,
    progress: dict[str, Any],
    missing_sets: Mapping[str, set[VideoExample]],
    failure_rows: list[dict[str, Any]],
    fallback_pipelines: dict[str, Any],
    chunk_index: int,
    chunk_count: int,
    group_name: str,
    label: str,
) -> None:
    try:
        raw_batch = build_raw_feature_batch(
            examples=batch_examples,
            modalities=group_modalities,
            frame_count=frame_count,
            image_size=image_size,
        )
        feature_batch = build_result.pipeline.prepare_features(raw_batch)
        write_cached_feature_items(
            cache_dir=cache_dir,
            specs=specs,
            modalities=group_modalities,
            examples=batch_examples,
            feature_batch=feature_batch,
            raw_batch=raw_batch,
            missing_sets=missing_sets,
            dataset_root=dataset_root,
            progress=progress,
        )
    except Exception as exc:
        print(
            f"cache {label}: modalities={group_name} batch failed "
            f"{chunk_index}/{chunk_count} size={len(batch_examples)} error={exc}",
            flush=True,
        )
        for example in batch_examples:
            for modality in group_modalities:
                if example not in missing_sets[modality]:
                    continue
                cache_single_modality_example(
                    example=example,
                    cache_dir=cache_dir,
                    spec=specs[modality],
                    config=config,
                    dataset_root=dataset_root,
                    progress=progress,
                    failure_rows=failure_rows,
                    fallback_pipelines=fallback_pipelines,
                )


def update_cache_group_progress(
    progress_display,
    batch_examples: Sequence[VideoExample],
    chunk_index: int,
    batch_size: int,
    group_examples_count: int,
    progress_interval: int,
    progress: Mapping[str, Any],
    group_modalities: Sequence[str],
    start: float,
    label: str,
    group_name: str,
) -> None:
    if progress_display is not None:
        progress_display.update(len(batch_examples))
    done = min(chunk_index * batch_size, group_examples_count)
    if done != group_examples_count and done % progress_interval != 0:
        return
    elapsed = time.perf_counter() - start
    written = sum(int(progress[modality]["written"]) for modality in group_modalities)
    failed = sum(int(progress[modality]["failed"]) for modality in group_modalities)
    rate = 0.0 if elapsed <= 0.0 else done / elapsed
    if progress_display is None:
        print(
            f"cache {label}: modalities={group_name} "
            f"done={done}/{group_examples_count} "
            f"written={written} failed={failed} "
            f"elapsed={elapsed:.1f}s videos_per_s={rate:.2f}",
            flush=True,
        )


def cache_feature_group(
    group_examples: Sequence[VideoExample],
    group_modalities: Sequence[str],
    frame_count: int,
    image_size: int,
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    config: Mapping[str, Any],
    dataset_root: Path,
    batch_size: int,
    progress_interval: int,
    progress: dict[str, Any],
    missing_by_modality: Mapping[str, list[VideoExample]],
    failure_rows: list[dict[str, Any]],
    fallback_pipelines: dict[str, Any],
    label: str,
    progress_bar: bool,
) -> None:
    start = time.perf_counter()
    group_name = ",".join(group_modalities)
    build_result = build_fusion_pipeline(config=config, modalities=group_modalities)
    build_result.pipeline.eval()
    missing_sets = {modality: set(missing_by_modality[modality]) for modality in group_modalities}
    chunks = chunk_examples(group_examples, batch_size)
    progress_display = build_cache_progress_display(
        group_examples=group_examples,
        group_name=group_name,
        label=label,
        enabled=progress_bar,
    )
    try:
        with torch.inference_mode():
            for chunk_index, batch_examples in enumerate(chunks, start=1):
                cache_feature_batch_or_fallback(
                    batch_examples=batch_examples,
                    group_modalities=group_modalities,
                    frame_count=frame_count,
                    image_size=image_size,
                    build_result=build_result,
                    cache_dir=cache_dir,
                    specs=specs,
                    config=config,
                    dataset_root=dataset_root,
                    progress=progress,
                    missing_sets=missing_sets,
                    failure_rows=failure_rows,
                    fallback_pipelines=fallback_pipelines,
                    chunk_index=chunk_index,
                    chunk_count=len(chunks),
                    group_name=group_name,
                    label=label,
                )
                update_cache_group_progress(
                    progress_display=progress_display,
                    batch_examples=batch_examples,
                    chunk_index=chunk_index,
                    batch_size=batch_size,
                    group_examples_count=len(group_examples),
                    progress_interval=progress_interval,
                    progress=progress,
                    group_modalities=group_modalities,
                    start=start,
                    label=label,
                    group_name=group_name,
                )
    finally:
        if progress_display is not None:
            progress_display.close()
        close_pipeline_result(build_result)
    elapsed = time.perf_counter() - start
    print(
        f"cache {label}: modalities={group_name} complete "
        f"written={sum(int(progress[modality]['written']) for modality in group_modalities)} "
        f"failed={sum(int(progress[modality]['failed']) for modality in group_modalities)} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


def cache_missing_feature_groups(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    config: Mapping[str, Any],
    dataset_root: Path,
    batch_size: int,
    progress_interval: int,
    progress: dict[str, Any],
    missing_by_modality: Mapping[str, list[VideoExample]],
    label: str,
    progress_bar: bool,
    group_by_modality: bool,
) -> None:
    failure_rows: list[dict[str, Any]] = []
    fallback_pipelines: dict[str, Any] = {}
    try:
        active_modalities = [
            modality for modality in modalities if missing_by_modality.get(modality)
        ]
        for frame_count, image_size, group_modalities in extraction_modality_groups(
            active_modalities,
            specs,
            group_by_modality=group_by_modality,
        ):
            group_examples = examples_missing_any_modality(
                examples=examples,
                modalities=group_modalities,
                missing_by_modality=missing_by_modality,
            )
            if not group_examples:
                continue
            cache_feature_group(
                group_examples=group_examples,
                group_modalities=group_modalities,
                frame_count=frame_count,
                image_size=image_size,
                cache_dir=cache_dir,
                specs=specs,
                config=config,
                dataset_root=dataset_root,
                batch_size=batch_size,
                progress_interval=progress_interval,
                progress=progress,
                missing_by_modality=missing_by_modality,
                failure_rows=failure_rows,
                fallback_pipelines=fallback_pipelines,
                label=label,
                progress_bar=progress_bar,
            )
    finally:
        for build_result in fallback_pipelines.values():
            close_pipeline_result(build_result)
        append_failure_rows(cache_dir, failure_rows)


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
    progress_bar: bool = False,
    group_by_modality: bool = False,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    progress_bar = progress_bar and sys.stderr.isatty()
    progress_interval = max(1, progress_every)
    missing_by_modality, progress = initialize_feature_cache_progress(
        examples=examples,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        dataset_root=dataset_root,
        overwrite=overwrite,
        skip_failures=skip_failures,
        progress_every=progress_interval,
        label=label,
        progress_bar=progress_bar,
    )
    cache_missing_feature_groups(
        examples=examples,
        cache_dir=cache_dir,
        specs=specs,
        modalities=modalities,
        config=config,
        dataset_root=dataset_root,
        batch_size=max(1, extract_batch_size),
        progress_interval=progress_interval,
        progress=progress,
        missing_by_modality=missing_by_modality,
        label=label,
        progress_bar=progress_bar,
        group_by_modality=group_by_modality,
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
            fieldnames=(
                "epoch",
                "train_loss",
                "train_accuracy",
                *metrics_row_fields("train"),
                "train_elapsed_seconds",
                "val_loss",
                "val_accuracy",
                *metrics_row_fields("val"),
                "val_elapsed_seconds",
                "checkpoint_metric",
                "checkpoint_metric_value",
                "best_checkpoint",
            ),
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


def write_diagnostics(path: Path, rows: Sequence[DiagnosticRow]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "path",
                "class_name",
                "label",
                "prediction",
                "probability",
                "split",
                "generator_id",
                "modality_name",
                "modality_gate_weight",
                "modality_expert_logit",
                "modality_mixed_logit_contribution",
                "token_attention_sum",
            ),
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
                    "generator_id": row.generator_id,
                    "modality_name": row.modality_name,
                    "modality_gate_weight": f"{row.modality_gate_weight:.8f}",
                    "modality_expert_logit": f"{row.modality_expert_logit:.8f}",
                    "modality_mixed_logit_contribution": (
                        f"{row.modality_mixed_logit_contribution:.8f}"
                    ),
                    "token_attention_sum": ""
                    if row.token_attention_sum is None
                    else f"{row.token_attention_sum:.8f}",
                }
            )


def summarize_diagnostic_rows(rows: Sequence[DiagnosticRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        correctness = "correct" if row.label == row.prediction else "incorrect"
        key = (row.split, row.class_name, correctness, row.generator_id, row.modality_name)
        group = groups.setdefault(
            key,
            {
                "split": row.split,
                "class_name": row.class_name,
                "correctness": correctness,
                "generator_id": row.generator_id,
                "modality_name": row.modality_name,
                "count": 0,
                "gate_weight_sum": 0.0,
                "expert_logit_sum": 0.0,
                "mixed_logit_contribution_sum": 0.0,
                "mixed_logit_abs_contribution_sum": 0.0,
                "token_attention_sum": 0.0,
                "token_attention_count": 0,
            },
        )
        group["count"] += 1
        group["gate_weight_sum"] += row.modality_gate_weight
        group["expert_logit_sum"] += row.modality_expert_logit
        group["mixed_logit_contribution_sum"] += row.modality_mixed_logit_contribution
        group["mixed_logit_abs_contribution_sum"] += abs(row.modality_mixed_logit_contribution)
        if row.token_attention_sum is not None:
            group["token_attention_sum"] += row.token_attention_sum
            group["token_attention_count"] += 1

    summaries = []
    for group in groups.values():
        count = int(group["count"])
        token_attention_count = int(group["token_attention_count"])
        summaries.append(
            {
                "split": group["split"],
                "class_name": group["class_name"],
                "correctness": group["correctness"],
                "generator_id": group["generator_id"],
                "modality_name": group["modality_name"],
                "count": count,
                "mean_gate_weight": group["gate_weight_sum"] / count,
                "mean_expert_logit": group["expert_logit_sum"] / count,
                "mean_mixed_logit_contribution": group["mixed_logit_contribution_sum"] / count,
                "mean_abs_mixed_logit_contribution": (
                    group["mixed_logit_abs_contribution_sum"] / count
                ),
                "mean_token_attention_sum": None
                if token_attention_count == 0
                else group["token_attention_sum"] / token_attention_count,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item["split"],
            item["class_name"],
            item["correctness"],
            item["generator_id"],
            -float(item["mean_abs_mixed_logit_contribution"]),
            item["modality_name"],
        ),
    )


def write_diagnostic_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "class_name",
                "correctness",
                "generator_id",
                "modality_name",
                "count",
                "mean_gate_weight",
                "mean_expert_logit",
                "mean_mixed_logit_contribution",
                "mean_abs_mixed_logit_contribution",
                "mean_token_attention_sum",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "mean_gate_weight": f"{float(row['mean_gate_weight']):.8f}",
                    "mean_expert_logit": f"{float(row['mean_expert_logit']):.8f}",
                    "mean_mixed_logit_contribution": (
                        f"{float(row['mean_mixed_logit_contribution']):.8f}"
                    ),
                    "mean_abs_mixed_logit_contribution": (
                        f"{float(row['mean_abs_mixed_logit_contribution']):.8f}"
                    ),
                    "mean_token_attention_sum": ""
                    if row["mean_token_attention_sum"] is None
                    else f"{float(row['mean_token_attention_sum']):.8f}",
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
    warm_start_checkpoint: Path | None = None,
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
    if warm_start_checkpoint is not None:
        state = torch.load(
            warm_start_checkpoint,
            map_location=build_result.device,
            weights_only=False,
        )
        model.load_state_dict(state)
        print(f"loaded warm-start checkpoint: {warm_start_checkpoint}", flush=True)
    modality_lrs = parse_modality_lrs(args.modality_lr)
    loader_config = resolve_cached_loader_config(config)
    regularization_config = resolve_training_regularization_config(config, args)
    optimizer = build_optimizer(
        model,
        base_lr=args.lr,
        modality_lrs=modality_lrs,
        weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_loader = build_cached_loader(
        train_examples,
        cache_dir,
        specs,
        modalities,
        args.batch_size,
        shuffle=True,
        dataset_root=dataset_root,
        loader_config=loader_config,
    )
    val_loader = build_cached_loader(
        val_examples,
        cache_dir,
        specs,
        modalities,
        args.batch_size,
        shuffle=False,
        dataset_root=dataset_root,
        loader_config=loader_config,
    )
    test_loader = build_cached_loader(
        test_examples,
        cache_dir,
        specs,
        modalities,
        args.batch_size,
        shuffle=False,
        dataset_root=dataset_root,
        loader_config=loader_config,
    )

    metrics: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    epochs_without_improvement = 0
    best_path = output_dir / "best.pt"
    print(
        "training: "
        f"modalities={','.join(modalities)} epochs={args.epochs} "
        f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"batch_size={args.batch_size} device={build_result.device} "
        f"loader={asdict(loader_config)} "
        f"regularization={asdict(regularization_config)} "
        f"lr={args.lr} modality_lrs={modality_lrs} "
        f"weight_decay={args.weight_decay} "
        f"checkpoint_metric={args.checkpoint_metric} "
        f"early_stopping_patience={args.early_stopping_patience}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            regularization_config=regularization_config,
        )
        val_result = evaluate_loss_accuracy(model, val_loader, loss_fn)
        metric_value = checkpoint_metric_value(args.checkpoint_metric, result, val_result)
        improved = is_metric_improvement(
            metric_name=args.checkpoint_metric,
            value=metric_value,
            best_value=best_metric_value,
            min_delta=args.early_stopping_min_delta,
        )
        row = {
            "epoch": epoch,
            "train_loss": f"{result.loss:.8f}",
            "train_accuracy": f"{result.accuracy:.8f}",
            **metrics_row_values("train", result.metrics),
            "train_elapsed_seconds": f"{result.elapsed_seconds:.6f}",
            "val_loss": f"{val_result.loss:.8f}",
            "val_accuracy": f"{val_result.accuracy:.8f}",
            **metrics_row_values("val", val_result.metrics),
            "val_elapsed_seconds": f"{val_result.elapsed_seconds:.6f}",
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_metric_value": f"{metric_value:.8f}",
            "best_checkpoint": "1" if improved else "0",
        }
        metrics.append(row)
        if improved:
            best_metric_value = metric_value
            epochs_without_improvement = 0
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={result.loss:.6f} "
            f"train_accuracy={result.accuracy:.4f} "
            f"train_f1={result.metrics.f1:.4f} "
            f"val_loss={val_result.loss:.6f} "
            f"val_accuracy={val_result.accuracy:.4f} "
            f"val_f1={val_result.metrics.f1:.4f} "
            f"val_fp={val_result.metrics.false_positive} "
            f"val_fn={val_result.metrics.false_negative} "
            f"{args.checkpoint_metric}={metric_value:.6f} "
            f"best={1 if improved else 0} "
            f"elapsed={result.elapsed_seconds:.3f}s "
            f"val_elapsed={val_result.elapsed_seconds:.3f}s",
            flush=True,
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "early stopping: "
                f"metric={args.checkpoint_metric} "
                f"best={best_metric_value:.6f} "
                f"epochs_without_improvement={epochs_without_improvement}",
                flush=True,
            )
            break

    best_state = torch.load(best_path, map_location=build_result.device, weights_only=False)
    model.load_state_dict(best_state)
    print(f"loaded best checkpoint for eval: {best_path}", flush=True)
    diagnostic_rows: list[DiagnosticRow] = []
    train_accuracy, train_rows = predict_rows(model, train_loader, diagnostic_rows=diagnostic_rows)
    val_accuracy, val_rows = predict_rows(model, val_loader, diagnostic_rows=diagnostic_rows)
    test_accuracy, test_rows = predict_rows(model, test_loader, diagnostic_rows=diagnostic_rows)
    train_metrics = prediction_rows_metrics(train_rows)
    val_metrics = prediction_rows_metrics(val_rows)
    test_metrics = prediction_rows_metrics(test_rows)
    print(
        f"eval: train_accuracy={train_accuracy:.4f} "
        f"train_f1={train_metrics.f1:.4f} "
        f"val_accuracy={val_accuracy:.4f} val_f1={val_metrics.f1:.4f} "
        f"val_fp={val_metrics.false_positive} val_fn={val_metrics.false_negative} "
        f"test_accuracy={test_accuracy:.4f} test_f1={test_metrics.f1:.4f} "
        f"test_fp={test_metrics.false_positive} test_fn={test_metrics.false_negative}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics(output_dir / "metrics.csv", metrics)
    write_predictions(output_dir / "predictions.csv", [*train_rows, *val_rows, *test_rows])
    diagnostic_summary_rows = summarize_diagnostic_rows(diagnostic_rows)
    write_diagnostics(output_dir / "diagnostics.csv", diagnostic_rows)
    write_diagnostic_summary(output_dir / "diagnostics_summary.csv", diagnostic_summary_rows)
    summary = {
        "modalities": list(modalities),
        "train_count": len(train_examples),
        "val_count": len(val_examples),
        "test_count": len(test_examples),
        "train_class_counts": class_counts(train_examples),
        "val_class_counts": class_counts(val_examples),
        "test_class_counts": class_counts(test_examples),
        "video_metadata": video_metadata_summary([*train_examples, *val_examples, *test_examples]),
        "epochs": args.epochs,
        "epochs_ran": len(metrics),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "modality_lrs": modality_lrs,
        "batch_size": args.batch_size,
        "cached_loader": asdict(loader_config),
        "regularization": asdict(regularization_config),
        "checkpoint_metric": args.checkpoint_metric,
        "best_checkpoint_metric_value": best_metric_value,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopped": len(metrics) < args.epochs,
        "warm_start_checkpoint": None
        if warm_start_checkpoint is None
        else str(warm_start_checkpoint),
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_metrics": asdict(train_metrics),
        "val_metrics": asdict(val_metrics),
        "test_metrics": asdict(test_metrics),
        "diagnostics_csv": None if not diagnostic_rows else str(output_dir / "diagnostics.csv"),
        "diagnostics_summary_csv": None
        if not diagnostic_summary_rows
        else str(output_dir / "diagnostics_summary.csv"),
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
    loader_config = resolve_cached_loader_config(config)
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
            loader_config=loader_config,
        )
        diagnostic_rows: list[DiagnosticRow] = []
        accuracy, rows = predict_rows(model, loader, diagnostic_rows=diagnostic_rows)
        run_dir = output_dir / "sanity_check" / name
        write_predictions(run_dir / "predictions.csv", rows)
        diagnostic_summary_rows = summarize_diagnostic_rows(diagnostic_rows)
        write_diagnostics(run_dir / "diagnostics.csv", diagnostic_rows)
        write_diagnostic_summary(run_dir / "diagnostics_summary.csv", diagnostic_summary_rows)
        result = {
            "round_target": int(summary["round_target"]),
            "modalities": list(modalities),
            "best_checkpoint": str(summary["best_checkpoint"]),
            "sanity_count": len(cached_examples),
            "sanity_accuracy": accuracy,
            "video_metadata": video_metadata_summary(cached_examples),
            "cached_loader": asdict(loader_config),
            "predictions_csv": str(run_dir / "predictions.csv"),
            "diagnostics_csv": None if not diagnostic_rows else str(run_dir / "diagnostics.csv"),
            "diagnostics_summary_csv": None
            if not diagnostic_summary_rows
            else str(run_dir / "diagnostics_summary.csv"),
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
    for line in format_split_audit(examples):
        print(line)


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

    print("dataset selection: loading examples", flush=True)
    examples = build_real_fake_examples(
        real_dir=video_root / "real",
        fake_dir=video_root / "fake",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    dataset_examples = examples
    cached_selection_summary: dict[str, int] | None = None
    if args.prefer_cached_selection:
        examples, cached_selection_summary = select_fully_cached_examples(
            examples=examples,
            cache_dir=cache_dir,
            specs=specs,
            modalities=base_modalities,
            dataset_root=dataset_root,
            progress_every=args.progress_every,
        )
        print(
            "cache selection: "
            f"metadata_only summary={cached_selection_summary} counts={class_counts(examples)}",
            flush=True,
        )
        if not examples:
            raise ValueError(
                "No examples have valid cached features for all requested modalities. "
                "Generate cache first or disable --prefer-cached-selection."
            )
    cache_score_by_path = None
    if args.balanced_total is None:
        split_mode = "dataset_splits"
        train_pool, val_examples, test_examples = split_examples(
            examples,
            eval_count_per_split=args.eval_count_per_split,
            seed=args.seed,
            cache_score_by_path=cache_score_by_path,
        )
    else:
        split_mode = f"balanced_total_{args.balanced_total}"
        train_pool, val_examples, test_examples = split_balanced_total_examples(
            examples=examples,
            balanced_total=args.balanced_total,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            cache_score_by_path=cache_score_by_path,
        )
    train_order = build_balanced_train_order(
        train_pool,
        args.seed + 101,
        cache_score_by_path=cache_score_by_path,
    )
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
    print(f"cached_loader={asdict(resolve_cached_loader_config(config))}", flush=True)
    print(
        f"regularization={asdict(resolve_training_regularization_config(config, args))}",
        flush=True,
    )
    print(
        f"modality_sets={','.join(modality_set_name(item) for item in modality_sets)}", flush=True
    )
    print(
        f"dataset_total={len(dataset_examples)} summary={summarize_examples(dataset_examples)}",
        flush=True,
    )
    if args.prefer_cached_selection:
        print(
            f"cached_selection_total={len(examples)} summary={summarize_examples(examples)}",
            flush=True,
        )
    for line in format_split_audit(dataset_examples):
        print(line, flush=True)
    print(f"split_mode={split_mode}", flush=True)
    print(f"train_pool={len(train_pool)} counts={class_counts(train_pool)}", flush=True)
    print(f"val_fixed={len(val_examples)} counts={class_counts(val_examples)}", flush=True)
    print(f"test_fixed={len(test_examples)} counts={class_counts(test_examples)}", flush=True)
    print(f"round_targets={','.join(str(target) for target in round_targets)}", flush=True)

    if args.dry_run:
        write_dry_run(
            dataset_examples,
            train_pool,
            val_examples,
            test_examples,
            round_targets,
            count_missing_cache(fixed_examples, cache_dir, specs, base_modalities, dataset_root),
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_examples = (
        examples if args.balanced_total is None else [*train_pool, *val_examples, *test_examples]
    )
    write_dataset_manifest(manifest_examples, output_dir / "manifest.csv")
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
            "balanced_total": args.balanced_total,
            "split_mode": split_mode,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "prefer_cached_selection": args.prefer_cached_selection,
            "cached_selection_summary": cached_selection_summary,
            "cached_loader": asdict(resolve_cached_loader_config(config)),
            "regularization": asdict(resolve_training_regularization_config(config, args)),
            "dataset_metadata": video_metadata_summary(manifest_examples),
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
            cached_val_examples = rebalance_eval_examples(
                cached_val_examples,
                target_count=len(val_examples),
                seed=args.seed + 701,
                label=f"val/{name}",
            )
            cached_test_examples = rebalance_eval_examples(
                cached_test_examples,
                target_count=len(test_examples),
                seed=args.seed + 709,
                label=f"test/{name}",
            )
            previous = previous_by_modality_set.get(name)
            warm_start_checkpoint = resolve_warm_start_checkpoint(
                previous,
                enabled=args.warm_start_rounds,
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
                warm_start_checkpoint=warm_start_checkpoint,
            )
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
