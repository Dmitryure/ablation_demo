from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Sampler
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
    ShardedCachedFeatureDataset,
    ShardedFeatureBatchSampler,
    build_feature_cache_specs,
    cache_example_key,
    collate_cached_feature_batch,
    feature_cache_item_exists,
    feature_cache_manifest_path,
    feature_cache_spec_dir,
    feature_cache_spec_id,
    metadata_filename_for_example,
    split_feature_batch,
    write_feature_cache_item,
    write_feature_cache_manifest,
)
from pipeline import build_fusion_pipeline, load_pipeline_yaml
from task_models import BinaryFusionClassifier, build_binary_fusion_classifier

DEFAULT_DATASET_ROOT = Path("/mnt/d/final_dataset")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "registry_fusion.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "iterative_ablation_runs"
FAST_LADDER = (200, 500, 1000, 2000, 4000)
TINY_LADDER = (40, 100, 200, 500)
LARGE_LADDER = (1000, 2500, 5000)
HEAD_TYPES = ("cls_linear", "cls_mlp", "attention_mil", "modality_gated_mil")
TRAIN_BALANCE_MODES = (
    "none",
    "class_weighted_loss",
    "class_balanced_batches",
    "generator_balanced_batches",
)
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

RUN_ARG_DEFAULTS: dict[str, Any] = {
    "dataset_root": DEFAULT_DATASET_ROOT,
    "cache_dir": None,
    "sharded_cache_dir": None,
    "output_dir": DEFAULT_OUTPUT_DIR,
    "modalities": None,
    "modality_permutations": "none",
    "round_ladder": "fast",
    "round_targets": None,
    "eval_count_per_split": 500,
    "full_eval_splits": False,
    "balanced_total": None,
    "train_balance_mode": "none",
    "fake_generator_cap_multiplier": None,
    "fake_generator_cap_exemptions": (),
    "fake_generator_loss_weights": None,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "batch_size": 8,
    "extract_batch_size": 4,
    "progress_every": 25,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "modality_lr": None,
    "modality_dropout": None,
    "depth_dropout": None,
    "gate_entropy_weight": None,
    "device": None,
    "head_type": None,
    "head_hidden_dim": None,
    "head_dropout": None,
    "checkpoint_metric": "val_accuracy",
    "early_stopping_patience": 0,
    "early_stopping_min_delta": 0.0,
    "occlusion_diagnostics": False,
    "occlusion_splits": ("val", "test"),
    "seed": 0,
    "overwrite_cache": False,
    "prefer_cached_selection": False,
    "warm_start_checkpoint": None,
    "warm_start_rounds": False,
    "skip_failures": False,
    "video_decode_mode": "scan",
    "clip_cache_dir": None,
    "enable_clip_cache": False,
    "no_clip_cache": False,
    "sanity_count": 300,
    "no_sanity_check": False,
    "dry_run": False,
}

PATH_RUN_ARGS = {
    "dataset_root",
    "cache_dir",
    "sharded_cache_dir",
    "output_dir",
    "clip_cache_dir",
    "warm_start_checkpoint",
}
SEQUENCE_RUN_ARGS = {
    "modalities",
    "round_targets",
    "occlusion_splits",
    "fake_generator_cap_exemptions",
}


@dataclass(frozen=True)
class PredictionRow:
    path: str
    class_name: str
    generator_id: str
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
class OcclusionRow:
    path: str
    class_name: str
    label: int
    full_prediction: int
    occluded_prediction: int
    full_probability: float
    occluded_probability: float
    split: str
    generator_id: str
    modality_removed: str
    full_logit: float
    occluded_logit: float
    delta_margin: float
    delta_loss: float
    delta_probability: float
    delta_correct_probability: float
    prediction_flipped: bool


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
class ShardedLoaderConfig:
    batch_strategy: str = "mixed_shard_local"
    allow_legacy_shards: bool = False


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
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--sharded-cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument(
        "--modality-permutations",
        choices=("none", "singletons", "singletons-plus-all", "all"),
        default=None,
    )
    parser.add_argument(
        "--round-ladder",
        choices=("fast", "tiny", "large"),
        default=None,
    )
    parser.add_argument(
        "--round-targets",
        type=int,
        nargs="+",
        default=None,
        help="Explicit train video counts. Overrides --round-ladder and does not append full train set.",
    )
    parser.add_argument("--eval-count-per-split", type=int, default=None)
    parser.add_argument(
        "--full-eval-splits",
        action="store_true",
        default=None,
        help="Use every preserved val/test split example instead of balanced eval subsets.",
    )
    parser.add_argument(
        "--balanced-total",
        type=int,
        default=None,
        help=(
            "Select this many videos across the whole dataset with equal real/fake counts, "
            "then derive train/val/test splits from that selected set."
        ),
    )
    parser.add_argument(
        "--train-balance-mode",
        choices=TRAIN_BALANCE_MODES,
        default=None,
        help=(
            "Training-only balancing mode. Keeps validation/test selection unchanged."
        ),
    )
    parser.add_argument(
        "--fake-generator-cap-multiplier",
        type=float,
        default=None,
        help=(
            "Training-pool fake generator cap as median_fake_generator_count * multiplier. "
            "Keeps all real examples and caps overrepresented fake generators before round selection."
        ),
    )
    parser.add_argument(
        "--fake-generator-cap-exemptions",
        nargs="*",
        default=None,
        help="Fake generator ids exempt from --fake-generator-cap-multiplier.",
    )
    parser.add_argument(
        "--fake-generator-loss-weights",
        nargs="*",
        default=None,
        help="Training loss weights for fake generators, e.g. dlc=1.5 visomaster=2.0.",
    )
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--extract-batch-size", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
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
        default=None,
        help="Metric used for best.pt and early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after this many epochs without checkpoint-metric improvement. 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Minimum checkpoint-metric improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--occlusion-diagnostics",
        action="store_true",
        default=None,
        help="After final eval, drop one modality at a time and write modality occlusion diagnostics.",
    )
    parser.add_argument(
        "--occlusion-splits",
        choices=("train", "val", "test"),
        nargs="+",
        default=None,
        help="Splits to evaluate for --occlusion-diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite-cache", action="store_true", default=None)
    parser.add_argument(
        "--prefer-cached-selection",
        action="store_true",
        default=None,
        help=(
            "Prefer examples with existing valid cached features when selecting train/val/test "
            "examples. Class balance and split boundaries are still preserved."
        ),
    )
    parser.add_argument(
        "--warm-start-rounds",
        action="store_true",
        default=None,
        help="Initialize each train-count round from the previous round checkpoint for the same modality set.",
    )
    parser.add_argument(
        "--warm-start-checkpoint",
        type=Path,
        default=None,
        help="Initialize every requested training round from this checkpoint.",
    )
    parser.add_argument("--skip-failures", action="store_true", default=None)
    parser.add_argument(
        "--video-decode-mode",
        choices=("seek", "scan"),
        default=None,
        help="Use random frame seeks or sequential video scan when sampling frames.",
    )
    parser.add_argument(
        "--clip-cache-dir",
        type=Path,
        default=None,
        help="Enable decoded clip cache at this directory.",
    )
    parser.add_argument(
        "--enable-clip-cache",
        action="store_true",
        default=None,
        help="Enable decoded clip cache at <cache-dir>/_clips.",
    )
    parser.add_argument(
        "--no-clip-cache",
        action="store_true",
        default=None,
        help="Deprecated no-op; decoded clip cache is disabled by default.",
    )
    parser.add_argument("--sanity-count", type=int, default=None)
    parser.add_argument("--no-sanity-check", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    return parser.parse_args()


def build_config(config_path: Path, device: str | None) -> dict[str, Any]:
    config = load_pipeline_yaml(config_path)
    if device is not None:
        config["device"] = device
    return config


def training_run_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    training = config.get("training", {})
    if training is None:
        training = {}
    if not isinstance(training, Mapping):
        raise ValueError("Config `training` must be a mapping when provided.")
    run = training.get("run", config.get("run", {}))
    if run is None:
        return {}
    if not isinstance(run, Mapping):
        raise ValueError("Config `training.run` must be a mapping when provided.")
    return run


def _coerce_run_path(value: Any, field_name: str) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    raise ValueError(f"`training.run.{field_name}` must be a non-empty path or null.")


def _coerce_run_sequence(value: Any, field_name: str) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"`training.run.{field_name}` must be a YAML list or null.")
    return list(value)


def _coerce_modality_lr(value: Any) -> dict[str, float] | list[str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        parsed: dict[str, float] = {}
        for modality, lr in value.items():
            if isinstance(lr, bool) or not isinstance(lr, (int, float)) or float(lr) <= 0.0:
                raise ValueError(
                    f"`training.run.modality_lr.{modality}` must be a positive number."
                )
            parsed[str(modality)] = float(lr)
        return parsed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    raise ValueError("`training.run.modality_lr` must be a mapping, list, or null.")


def _coerce_fake_generator_loss_weights(value: Any) -> dict[str, float] | list[str] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        parsed: dict[str, float] = {}
        for generator_id, weight in value.items():
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or float(weight) <= 0.0
            ):
                raise ValueError(
                    "training.run.fake_generator_loss_weights."
                    f"{generator_id} must be a positive number."
                )
            parsed[str(generator_id)] = float(weight)
        return parsed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    raise ValueError(
        "`training.run.fake_generator_loss_weights` must be a mapping, list, or null."
    )


def coerce_training_run_value(field_name: str, value: Any) -> Any:
    if field_name in PATH_RUN_ARGS:
        return _coerce_run_path(value, field_name)
    if field_name in SEQUENCE_RUN_ARGS:
        return _coerce_run_sequence(value, field_name)
    if field_name == "modality_lr":
        return _coerce_modality_lr(value)
    if field_name == "fake_generator_loss_weights":
        return _coerce_fake_generator_loss_weights(value)
    return value


def resolve_training_run_args(
    config: Mapping[str, Any], args: argparse.Namespace
) -> argparse.Namespace:
    run_config = training_run_section(config)
    values = vars(args).copy()
    for field_name, default in RUN_ARG_DEFAULTS.items():
        if values.get(field_name) is not None:
            continue
        if field_name in run_config:
            values[field_name] = coerce_training_run_value(field_name, run_config[field_name])
        else:
            values[field_name] = default
    return argparse.Namespace(**values)


def training_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        field_name: getattr(args, field_name)
        for field_name in RUN_ARG_DEFAULTS
        if hasattr(args, field_name)
    }


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


def resolve_sharded_loader_config(config: Mapping[str, Any]) -> ShardedLoaderConfig:
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
    batch_strategy = str(sharded.get("batch_strategy", "mixed_shard_local"))
    if batch_strategy not in {"mixed_shard_local", "global_shuffle", "shard_local"}:
        raise ValueError(f"Unsupported training.sharded_loader.batch_strategy: {batch_strategy}")
    allow_legacy_shards = _optional_bool(
        sharded.get("allow_legacy_shards"),
        "training.sharded_loader.allow_legacy_shards",
        False,
    )
    return ShardedLoaderConfig(
        batch_strategy=batch_strategy,
        allow_legacy_shards=allow_legacy_shards,
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


def fake_generator_counts(examples: Sequence[VideoExample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example in examples:
        if example.class_name != "fake":
            continue
        counts[str(example.generator_id or "unknown")] += 1
    return dict(sorted(counts.items()))


def median_int(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[midpoint])
    return (float(ordered[midpoint - 1]) + float(ordered[midpoint])) / 2.0


def resolve_fake_generator_cap(
    counts: Mapping[str, int],
    multiplier: float | None,
) -> tuple[int | None, float | None]:
    if multiplier is None:
        return None, None
    if multiplier <= 0.0:
        raise ValueError("`--fake-generator-cap-multiplier` must be positive.")
    median_count = median_int(list(counts.values()))
    cap = max(1, int(median_count * multiplier))
    return cap, median_count


def cap_fake_generators(
    examples: Sequence[VideoExample],
    cap_multiplier: float | None,
    seed: int,
    cap_exemptions: Sequence[str] = (),
    cache_score_by_path: Mapping[str, int] | None = None,
) -> tuple[list[VideoExample], dict[str, Any]]:
    original_counts = fake_generator_counts(examples)
    exemptions = frozenset(str(item) for item in cap_exemptions if str(item))
    cap, median_count = resolve_fake_generator_cap(original_counts, cap_multiplier)
    if cap is None:
        return list(examples), {
            "enabled": False,
            "multiplier": None,
            "exemptions": sorted(exemptions),
            "median_count": None,
            "cap": None,
            "original_fake_generator_counts": original_counts,
            "selected_fake_generator_counts": original_counts,
            "dropped_fake_count": 0,
        }

    real = [example for example in examples if example.class_name == "real"]
    fake_by_generator: dict[str, list[VideoExample]] = defaultdict(list)
    for example in examples:
        if example.class_name == "fake":
            fake_by_generator[str(example.generator_id or "unknown")].append(example)

    selected_fake: list[VideoExample] = []
    for offset, generator_id in enumerate(sorted(fake_by_generator)):
        ordered = _shuffled(
            fake_by_generator[generator_id],
            seed + offset + 1,
            cache_score_by_path=cache_score_by_path,
        )
        selected_fake.extend(ordered if generator_id in exemptions else ordered[:cap])

    capped_examples = [*real, *selected_fake]
    selected_counts = fake_generator_counts(capped_examples)
    return capped_examples, {
        "enabled": True,
        "multiplier": cap_multiplier,
        "exemptions": sorted(exemptions),
        "median_count": median_count,
        "cap": cap,
        "original_fake_generator_counts": original_counts,
        "selected_fake_generator_counts": selected_counts,
        "dropped_fake_count": sum(original_counts.values()) - sum(selected_counts.values()),
    }


def label_index_groups(examples: Sequence[VideoExample]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[int(example.label)].append(index)
    return dict(groups)


def fake_generator_index_groups(examples: Sequence[VideoExample]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    missing: list[str] = []
    for index, example in enumerate(examples):
        if int(example.label) != 1:
            continue
        generator_id = str(example.generator_id or "").strip()
        if not generator_id:
            missing.append(metadata_filename_for_example(example))
            continue
        groups[generator_id].append(index)
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            "generator_balanced_batches requires generator_id for every fake "
            f"training example. Missing examples: {sample}"
        )
    return dict(groups)


def require_binary_train_labels(examples: Sequence[VideoExample]) -> dict[int, list[int]]:
    groups = label_index_groups(examples)
    real = groups.get(0, [])
    fake = groups.get(1, [])
    if not real or not fake:
        raise ValueError(
            "Train balancing requires both real and fake training examples "
            f"(real={len(real)}, fake={len(fake)})."
        )
    return {0: real, 1: fake}


def sample_indices_with_replacement(
    indices: Sequence[int],
    target_count: int,
    rng: random.Random,
) -> list[int]:
    if target_count <= 0:
        return []
    if not indices:
        raise ValueError("Cannot sample from an empty index group.")
    sampled: list[int] = []
    while len(sampled) < target_count:
        chunk = list(indices)
        rng.shuffle(chunk)
        sampled.extend(chunk[: target_count - len(sampled)])
    return sampled


def interleave_label_indices(
    real_indices: Sequence[int],
    fake_indices: Sequence[int],
    rng: random.Random,
) -> list[int]:
    labels = [0, 1]
    rng.shuffle(labels)
    positions = {0: 0, 1: 0}
    by_label = {0: list(real_indices), 1: list(fake_indices)}
    ordered: list[int] = []
    remaining = len(real_indices) + len(fake_indices)
    label_cursor = 0
    while remaining > 0:
        label = labels[label_cursor % len(labels)]
        label_cursor += 1
        position = positions[label]
        if position >= len(by_label[label]):
            continue
        ordered.append(by_label[label][position])
        positions[label] += 1
        remaining -= 1
    return ordered


def generator_balanced_fake_indices(
    generator_groups: Mapping[str, Sequence[int]],
    target_count: int,
    rng: random.Random,
) -> list[int]:
    generators = sorted(generator_groups)
    if not generators:
        raise ValueError("generator_balanced_batches requires at least one fake generator.")
    rng.shuffle(generators)
    base_quota = target_count // len(generators)
    remainder = target_count % len(generators)
    sampled: list[int] = []
    for offset, generator_id in enumerate(generators):
        quota = base_quota + (1 if offset < remainder else 0)
        sampled.extend(sample_indices_with_replacement(generator_groups[generator_id], quota, rng))
    rng.shuffle(sampled)
    return sampled


class BalancedTrainBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        examples: Sequence[VideoExample],
        batch_size: int,
        mode: str,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if mode not in {"class_balanced_batches", "generator_balanced_batches"}:
            raise ValueError(f"Unsupported balanced batch mode: {mode}")
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        self.examples = list(examples)
        self.batch_size = batch_size
        self.mode = mode
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.label_groups = require_binary_train_labels(self.examples)
        self.fake_generator_groups = (
            fake_generator_index_groups(self.examples)
            if mode == "generator_balanced_batches"
            else {}
        )
        self.target_per_label = max(len(self.label_groups[0]), len(self.label_groups[1]))

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        real_indices = sample_indices_with_replacement(
            self.label_groups[0],
            self.target_per_label,
            rng,
        )
        if self.mode == "generator_balanced_batches":
            fake_indices = generator_balanced_fake_indices(
                self.fake_generator_groups,
                self.target_per_label,
                rng,
            )
        else:
            fake_indices = sample_indices_with_replacement(
                self.label_groups[1],
                self.target_per_label,
                rng,
            )
        if not self.shuffle:
            real_indices = sorted(real_indices)
            fake_indices = sorted(fake_indices)
        ordered = interleave_label_indices(real_indices, fake_indices, rng)
        for start in range(0, len(ordered), self.batch_size):
            yield ordered[start : start + self.batch_size]
        self.epoch += 1

    def __len__(self) -> int:
        epoch_samples = self.target_per_label * 2
        return (epoch_samples + self.batch_size - 1) // self.batch_size

    def summary(self) -> dict[str, Any]:
        generator_counts = {
            generator_id: len(indices)
            for generator_id, indices in sorted(self.fake_generator_groups.items())
        }
        return {
            "sampler": self.mode,
            "epoch_samples": self.target_per_label * 2,
            "target_per_label": self.target_per_label,
            "class_counts": class_counts(self.examples),
            "fake_generator_counts": generator_counts,
        }


def train_pos_weight(examples: Sequence[VideoExample]) -> float:
    groups = require_binary_train_labels(examples)
    return len(groups[0]) / len(groups[1])


def build_train_balance_summary(
    examples: Sequence[VideoExample],
    mode: str,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    if mode not in TRAIN_BALANCE_MODES:
        raise ValueError(f"Unsupported train_balance_mode: {mode}")
    summary: dict[str, Any] = {
        "mode": mode,
        "class_counts": class_counts(examples),
        "loss": {"type": "bce_with_logits", "pos_weight": None},
        "sampler": None,
        "epoch_samples": len(examples),
    }
    if mode == "none":
        return summary
    if mode == "class_weighted_loss":
        summary["loss"] = {
            "type": "bce_with_logits",
            "pos_weight": train_pos_weight(examples),
        }
        return summary
    sampler = BalancedTrainBatchSampler(
        examples=examples,
        batch_size=batch_size,
        mode=mode,
        seed=seed,
    )
    sampler_summary = sampler.summary()
    return {**summary, **sampler_summary}


def build_train_batch_sampler(
    examples: Sequence[VideoExample],
    batch_size: int,
    mode: str,
    seed: int,
) -> BalancedTrainBatchSampler | None:
    if mode in {"class_balanced_batches", "generator_balanced_batches"}:
        return BalancedTrainBatchSampler(
            examples=examples,
            batch_size=batch_size,
            mode=mode,
            seed=seed,
        )
    return None


def build_train_loss_fn(
    examples: Sequence[VideoExample],
    mode: str,
    device: torch.device,
) -> torch.nn.Module:
    if mode == "class_weighted_loss":
        pos_weight = torch.tensor([train_pos_weight(examples)], device=device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return torch.nn.BCEWithLogitsLoss()


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
    full_eval_splits: bool = False,
) -> tuple[list[VideoExample], list[VideoExample], list[VideoExample]]:
    train = [example for example in examples if example.split == "train"]
    if full_eval_splits:
        return (
            train,
            [example for example in examples if example.split == "val"],
            [example for example in examples if example.split == "test"],
        )
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


def parse_modality_lrs(values: Mapping[str, Any] | Sequence[str] | None) -> dict[str, float]:
    if not values:
        return {}
    if isinstance(values, Mapping):
        parsed: dict[str, float] = {}
        for modality, raw_lr in values.items():
            lr = float(raw_lr)
            if lr <= 0.0:
                raise ValueError(f"`modality_lr.{modality}` must be positive, got {raw_lr!r}.")
            parsed[str(modality)] = lr
        return parsed
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


def parse_fake_generator_loss_weights(
    values: Mapping[str, Any] | Sequence[str] | None,
) -> dict[str, float]:
    if not values:
        return {}
    if isinstance(values, Mapping):
        parsed: dict[str, float] = {}
        for generator_id, raw_weight in values.items():
            weight = float(raw_weight)
            if weight <= 0.0:
                raise ValueError(
                    f"`fake_generator_loss_weights.{generator_id}` must be positive, "
                    f"got {raw_weight!r}."
                )
            parsed[str(generator_id)] = weight
        return parsed
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"`--fake-generator-loss-weights` must use generator=value, got {value!r}."
            )
        generator_id, raw_weight = value.split("=", 1)
        generator_id = generator_id.strip()
        if not generator_id:
            raise ValueError(
                f"`--fake-generator-loss-weights` has empty generator in {value!r}."
            )
        weight = float(raw_weight)
        if weight <= 0.0:
            raise ValueError(
                f"`--fake-generator-loss-weights` must be positive, got {value!r}."
            )
        parsed[generator_id] = weight
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


def batch_generator_ids(batch: Mapping[str, Any]) -> tuple[str, ...]:
    class_names = tuple(str(item) for item in batch["class_name"])
    generator_ids = tuple(str(item or "") for item in batch.get("generator_id", ()))
    if len(generator_ids) != len(class_names):
        generator_ids = tuple("" for _ in class_names)
    resolved: list[str] = []
    for class_name, generator_id in zip(class_names, generator_ids):
        if class_name == "real":
            resolved.append("real")
        else:
            resolved.append(generator_id or "unknown")
    return tuple(resolved)


def fake_generator_sample_weights(
    batch: Mapping[str, Any],
    weights: Mapping[str, float],
    device: torch.device,
) -> torch.Tensor | None:
    if not weights:
        return None
    class_names = tuple(str(item) for item in batch["class_name"])
    generator_ids = batch_generator_ids(batch)
    values = [
        float(weights.get(generator_id, 1.0)) if class_name == "fake" else 1.0
        for class_name, generator_id in zip(class_names, generator_ids)
    ]
    return torch.tensor(values, dtype=torch.float32, device=device).view(-1, 1)


def compute_weighted_train_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    batch: Mapping[str, Any],
    loss_fn: torch.nn.Module,
    fake_generator_weights: Mapping[str, float],
) -> torch.Tensor:
    sample_weights = fake_generator_sample_weights(batch, fake_generator_weights, logits.device)
    if sample_weights is None:
        return loss_fn(logits, labels)
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=getattr(loss_fn, "pos_weight", None),
        reduction="none",
    )
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(
        torch.finfo(losses.dtype).tiny
    )


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


def should_log_loop_progress(
    label: str | None,
    batch_index: int,
    batch_count: int,
    progress_every: int,
) -> bool:
    return bool(
        label
        and progress_every > 0
        and (batch_index == batch_count or batch_index % progress_every == 0)
    )


def log_loop_progress(
    label: str | None,
    batch_index: int,
    batch_count: int,
    progress_every: int,
    total_count: int,
    total_loss: float,
    true_positive: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
    start: float,
) -> None:
    if not should_log_loop_progress(label, batch_index, batch_count, progress_every):
        return
    elapsed = time.perf_counter() - start
    rate = 0.0 if elapsed <= 0.0 else total_count / elapsed
    metrics = binary_metrics_from_counts(
        true_positive,
        true_negative,
        false_positive,
        false_negative,
    )
    print(
        f"{label}: batch={batch_index}/{batch_count} "
        f"samples={total_count} loss={total_loss / max(total_count, 1):.6f} "
        f"accuracy={metrics.accuracy:.4f} f1={metrics.f1:.4f} "
        f"elapsed={elapsed:.1f}s samples_per_s={rate:.2f}",
        flush=True,
    )


def log_prediction_progress(
    label: str | None,
    batch_index: int,
    batch_count: int,
    progress_every: int,
    total_count: int,
    correct: int,
    start: float,
) -> None:
    if not should_log_loop_progress(label, batch_index, batch_count, progress_every):
        return
    elapsed = time.perf_counter() - start
    rate = 0.0 if elapsed <= 0.0 else total_count / elapsed
    accuracy = correct / max(total_count, 1)
    print(
        f"{label}: batch={batch_index}/{batch_count} "
        f"samples={total_count} accuracy={accuracy:.4f} "
        f"elapsed={elapsed:.1f}s samples_per_s={rate:.2f}",
        flush=True,
    )


def train_one_epoch(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    fake_generator_loss_weights: Mapping[str, float] | None = None,
    regularization_config: TrainingRegularizationConfig | None = None,
    progress_label: str | None = None,
    progress_every: int = 0,
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
    batch_count = len(loader)
    generator_weights = fake_generator_loss_weights or {}
    for batch_index, batch in enumerate(loader, start=1):
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        training_batch = apply_training_modality_dropout(
            batch,
            model.pipeline.enabled_modalities,
            resolved_regularization.modality_dropout,
        )
        output = model(move_tensor_batch_to_device(training_batch, device))
        loss = compute_weighted_train_loss(
            output.logits,
            labels,
            batch,
            loss_fn,
            generator_weights,
        )
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
        log_loop_progress(
            label=progress_label,
            batch_index=batch_index,
            batch_count=batch_count,
            progress_every=progress_every,
            total_count=total_count,
            total_loss=total_loss,
            true_positive=true_positive,
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            start=start,
        )
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
    progress_label: str | None = None,
    progress_every: int = 0,
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
    batch_count = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
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
            log_loop_progress(
                label=progress_label,
                batch_index=batch_index,
                batch_count=batch_count,
                progress_every=progress_every,
                total_count=total_count,
                total_loss=total_loss,
                true_positive=true_positive,
                true_negative=true_negative,
                false_positive=false_positive,
                false_negative=false_negative,
                start=start,
            )
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
    progress_label: str | None = None,
    progress_every: int = 0,
) -> tuple[float, list[PredictionRow]]:
    device = model_device(model)
    correct = 0
    total = 0
    rows: list[PredictionRow] = []
    model.eval()
    start = time.perf_counter()
    batch_count = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            output = model(move_tensor_batch_to_device(batch, device))
            probabilities = output.probabilities.detach().cpu().view(-1)
            labels = batch["label"].view(-1).to(dtype=torch.long)
            predictions = (probabilities >= 0.5).to(dtype=torch.long)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
            log_prediction_progress(
                label=progress_label,
                batch_index=batch_index,
                batch_count=batch_count,
                progress_every=progress_every,
                total_count=total,
                correct=correct,
                start=start,
            )
            if diagnostic_rows is not None:
                append_diagnostic_rows(
                    diagnostic_rows=diagnostic_rows,
                    output=output,
                    batch=batch,
                    labels=labels,
                    probabilities=probabilities,
                    predictions=predictions,
                )
            generator_ids = batch_generator_ids(batch)
            for index, probability in enumerate(probabilities.tolist()):
                rows.append(
                    PredictionRow(
                        path=batch["path"][index],
                        class_name=batch["class_name"][index],
                        generator_id=generator_ids[index],
                        label=int(labels[index].item()),
                        probability=float(probability),
                        prediction=int(predictions[index].item()),
                        split=batch["split"][index],
                    )
                )
    return correct / total, rows


def correct_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    label_sign = labels.to(device=logits.device, dtype=logits.dtype).view(-1, 1) * 2.0 - 1.0
    return logits.view(-1, 1) * label_sign


def correct_class_probability(probabilities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=probabilities.device, dtype=probabilities.dtype).view(-1, 1)
    probabilities = probabilities.view(-1, 1)
    return torch.where(labels >= 0.5, probabilities, 1.0 - probabilities)


def per_example_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=logits.dtype).view_as(logits)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
    )


def build_occlusion_rows(
    batch: Mapping[str, Any],
    modality_removed: str,
    full_logits: torch.Tensor,
    occluded_logits: torch.Tensor,
    split_name: str,
) -> list[OcclusionRow]:
    labels = batch["label"].detach().cpu().view(-1).to(dtype=torch.long)
    full_logits = full_logits.detach().cpu().view(-1, 1)
    occluded_logits = occluded_logits.detach().cpu().view(-1, 1)
    full_probabilities = torch.sigmoid(full_logits)
    occluded_probabilities = torch.sigmoid(occluded_logits)
    full_predictions = (full_probabilities.view(-1) >= 0.5).to(dtype=torch.long)
    occluded_predictions = (occluded_probabilities.view(-1) >= 0.5).to(dtype=torch.long)
    full_margin = correct_class_margin(full_logits, labels)
    occluded_margin = correct_class_margin(occluded_logits, labels)
    full_losses = per_example_bce_loss(full_logits, labels)
    occluded_losses = per_example_bce_loss(occluded_logits, labels)
    full_correct_probabilities = correct_class_probability(full_probabilities, labels)
    occluded_correct_probabilities = correct_class_probability(occluded_probabilities, labels)

    rows: list[OcclusionRow] = []
    for index, label in enumerate(labels.tolist()):
        path = str(batch["path"][index])
        class_name = str(batch["class_name"][index])
        split = str(batch.get("split", [split_name] * len(labels))[index])
        rows.append(
            OcclusionRow(
                path=path,
                class_name=class_name,
                label=int(label),
                full_prediction=int(full_predictions[index].item()),
                occluded_prediction=int(occluded_predictions[index].item()),
                full_probability=float(full_probabilities[index].item()),
                occluded_probability=float(occluded_probabilities[index].item()),
                split=split,
                generator_id=infer_prediction_generator(path, class_name),
                modality_removed=modality_removed,
                full_logit=float(full_logits[index].item()),
                occluded_logit=float(occluded_logits[index].item()),
                delta_margin=float((full_margin[index] - occluded_margin[index]).item()),
                delta_loss=float((occluded_losses[index] - full_losses[index]).item()),
                delta_probability=float(
                    (full_probabilities[index] - occluded_probabilities[index]).item()
                ),
                delta_correct_probability=float(
                    (
                        full_correct_probabilities[index] - occluded_correct_probabilities[index]
                    ).item()
                ),
                prediction_flipped=bool(
                    full_predictions[index].item() != occluded_predictions[index].item()
                ),
            )
        )
    return rows


def run_occlusion_eval(
    model: BinaryFusionClassifier,
    loader: DataLoader[dict[str, Any]],
    modalities: Sequence[str],
    split_name: str,
) -> list[OcclusionRow]:
    device = model_device(model)
    rows: list[OcclusionRow] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            full_output = model(move_tensor_batch_to_device(batch, device))
            full_logits = full_output.logits.detach().cpu()
            for modality in modalities:
                occluded_batch = {**batch, "dropped_modalities": (modality,)}
                occluded_output = model(move_tensor_batch_to_device(occluded_batch, device))
                rows.extend(
                    build_occlusion_rows(
                        batch=batch,
                        modality_removed=modality,
                        full_logits=full_logits,
                        occluded_logits=occluded_output.logits,
                        split_name=split_name,
                    )
                )
    return rows


def occlusion_skip_reason(modalities: Sequence[str], enabled: bool) -> str | None:
    if not enabled:
        return None
    if len(modalities) <= 1:
        return "requires_at_least_two_modalities"
    return None


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


def prediction_generator_metrics(rows: Sequence[PredictionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[PredictionRow]] = defaultdict(list)
    for row in rows:
        groups[(row.split, row.class_name, row.generator_id)].append(row)

    summaries: list[dict[str, Any]] = []
    for (split, class_name, generator_id), group_rows in groups.items():
        metrics = prediction_rows_metrics(group_rows)
        summaries.append(
            {
                "split": split,
                "class_name": class_name,
                "generator_id": generator_id,
                "count": len(group_rows),
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "specificity": metrics.specificity,
                "false_positive": metrics.false_positive,
                "false_negative": metrics.false_negative,
                "true_positive": metrics.true_positive,
                "true_negative": metrics.true_negative,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            str(item["split"]),
            str(item["class_name"]),
            str(item["generator_id"]),
        ),
    )


def write_generator_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "class_name",
                "generator_id",
                "count",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "specificity",
                "false_positive",
                "false_negative",
                "true_positive",
                "true_negative",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "accuracy": f"{float(row['accuracy']):.8f}",
                    "balanced_accuracy": f"{float(row['balanced_accuracy']):.8f}",
                    "precision": f"{float(row['precision']):.8f}",
                    "recall": f"{float(row['recall']):.8f}",
                    "f1": f"{float(row['f1']):.8f}",
                    "specificity": f"{float(row['specificity']):.8f}",
                }
            )


def build_cached_loader(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    batch_size: int,
    shuffle: bool,
    dataset_root: Path,
    loader_config: CachedLoaderConfig | None = None,
    batch_sampler: Sampler[list[int]] | None = None,
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
        "collate_fn": collate_cached_feature_batch,
        "num_workers": resolved_loader_config.num_workers,
        "pin_memory": resolved_loader_config.pin_memory,
    }
    if batch_sampler is None:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
    else:
        loader_kwargs["batch_sampler"] = batch_sampler
    if resolved_loader_config.num_workers > 0:
        loader_kwargs["persistent_workers"] = resolved_loader_config.persistent_workers
        if resolved_loader_config.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = resolved_loader_config.prefetch_factor
    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def build_sharded_cached_loader(
    examples: Sequence[VideoExample],
    all_cache_examples: Sequence[VideoExample],
    sharded_cache_dir: Path,
    batch_size: int,
    shuffle: bool,
    dataset_root: Path,
    loader_config: CachedLoaderConfig | None = None,
    sharded_loader_config: ShardedLoaderConfig | None = None,
    seed: int = 0,
    batch_sampler: Sampler[list[int]] | None = None,
) -> DataLoader[dict[str, Any]]:
    resolved_loader_config = loader_config or CachedLoaderConfig()
    resolved_sharded_config = sharded_loader_config or ShardedLoaderConfig()
    dataset = ShardedCachedFeatureDataset(
        sharded_cache_dir=sharded_cache_dir,
        all_examples=all_cache_examples,
        selected_examples=examples,
        dataset_root=dataset_root,
        allow_legacy_shards=resolved_sharded_config.allow_legacy_shards,
    )
    resolved_batch_sampler = batch_sampler
    if resolved_batch_sampler is None:
        resolved_batch_sampler = ShardedFeatureBatchSampler(
            refs=dataset.refs,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            batch_strategy=resolved_sharded_config.batch_strategy,
        )
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": resolved_batch_sampler,
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
    manifest_minimum_rows: int | None = None,
) -> list[VideoExample]:
    spec_id = feature_cache_spec_id(spec)
    modality_cache_root = feature_cache_spec_dir(cache_dir, spec)
    root_exists = modality_cache_root.exists()
    if not overwrite:
        manifest_keys = read_cached_manifest_keys(
            cache_dir,
            spec,
            minimum_rows=manifest_minimum_rows,
        )
        if manifest_keys is not None:
            return [
                example
                for example in examples
                if cache_key_for_example(example, dataset_root) not in manifest_keys
                and (spec_id, spec.modality, str(example.path)) not in skip_failure_keys
            ]

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
    manifest_minimum_rows: int | None = None,
) -> dict[str, int]:
    missing: dict[str, int] = {}
    for modality in modalities:
        spec = specs[modality]
        modality_cache_root = feature_cache_spec_dir(cache_dir, spec)
        if not modality_cache_root.exists():
            missing[modality] = len(examples)
            continue
        manifest_keys = read_cached_manifest_keys(
            cache_dir,
            spec,
            minimum_rows=manifest_minimum_rows,
        )
        if manifest_keys is not None:
            missing[modality] = sum(
                cache_key_for_example(example, dataset_root) not in manifest_keys
                for example in examples
            )
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
    manifest_minimum_rows: int | None = None,
) -> tuple[int, int]:
    spec_id = feature_cache_spec_id(spec)
    manifest_keys = read_cached_manifest_keys(
        cache_dir,
        spec,
        minimum_rows=manifest_minimum_rows,
    )
    if manifest_keys is not None:
        cached = 0
        skipped_failed = 0
        for example in examples:
            if cache_key_for_example(example, dataset_root) in manifest_keys:
                cached += 1
            elif (spec_id, spec.modality, str(example.path)) in skip_failure_keys:
                skipped_failed += 1
        return cached, skipped_failed

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
    return f"{example.class_name}/{metadata_filename_for_example(example, dataset_root)}"


def read_cached_manifest_keys(
    cache_dir: Path,
    spec: FeatureCacheSpec,
    minimum_rows: int | None = None,
) -> set[str] | None:
    path = feature_cache_manifest_path(cache_dir, spec)
    if not path.is_file():
        return None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"class_name", "filename", "status"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            return None
        cached_keys: set[str] = set()
        row_count = 0
        for row in reader:
            row_count += 1
            class_name = str(row.get("class_name", "")).strip()
            filename = str(row.get("filename", "")).strip()
            if str(row.get("status", "")).strip() == "cached" and class_name and filename:
                cached_keys.add(f"{class_name}/{filename}")
        if minimum_rows is not None and row_count < minimum_rows:
            return None
        return cached_keys


def select_fully_cached_examples(
    examples: Sequence[VideoExample],
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
    dataset_root: Path,
    progress_every: int | None = None,
) -> tuple[list[VideoExample], dict[str, int]]:
    cached_key_sets: dict[str, set[str]] = {}
    manifest_backed_modalities = 0
    for modality in modalities:
        spec = specs[modality]
        spec_dir = feature_cache_spec_dir(cache_dir, spec)
        spec_id = feature_cache_spec_id(spec)
        manifest_keys = read_cached_manifest_keys(cache_dir, spec, minimum_rows=len(examples))
        if manifest_keys is not None:
            cached_key_sets[modality] = manifest_keys
            manifest_backed_modalities += 1
            print(
                f"cache selection: manifest modality={modality} "
                f"matched={len(manifest_keys)} cache_from={spec_dir} spec_id={spec_id}",
                flush=True,
            )
            continue

        print(
            f"cache selection: scan metadata modality={modality} "
            f"cache_from={spec_dir} spec_id={spec_id}",
            flush=True,
        )
        cached_keys: set[str] = set()
        for index, example in enumerate(examples, start=1):
            if feature_cache_item_exists(cache_dir, example, spec, dataset_root):
                cached_keys.add(cache_key_for_example(example, dataset_root))
            if progress_every is not None and index % progress_every == 0:
                print(
                    f"cache selection: modality={modality} checked={index} "
                    f"matched={len(cached_keys)} cache_from={spec_dir} spec_id={spec_id}",
                    flush=True,
                )
        cached_key_sets[modality] = cached_keys
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
            "manifest_backed_modalities": manifest_backed_modalities,
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


def read_sharded_cache_keys(sharded_cache_dir: Path) -> set[str]:
    index_path = sharded_cache_dir / "index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    locations = payload.get("example_locations")
    if not isinstance(locations, Mapping):
        raise ValueError(f"Missing sharded cache example_locations: {index_path}")
    return {str(key) for key in locations}


def filter_examples_with_shards(
    examples: Sequence[VideoExample],
    sharded_cache_keys: set[str],
    dataset_root: Path,
    label: str,
) -> list[VideoExample]:
    kept: list[VideoExample] = []
    dropped: list[VideoExample] = []
    for example in examples:
        if cache_example_key(example, dataset_root) in sharded_cache_keys:
            kept.append(example)
        else:
            dropped.append(example)
    if dropped:
        print(
            f"shard filter {label}: kept={len(kept)} dropped={len(dropped)}",
            flush=True,
        )
        for example in dropped[:5]:
            print(f"shard filter {label}: dropped path={example.path}", flush=True)
        if len(dropped) > 5:
            print(f"shard filter {label}: dropped_more={len(dropped) - 5}", flush=True)
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
) -> dict[tuple[int, int, str | None], tuple[str, ...]]:
    grouped: dict[tuple[int, int, str | None], list[str]] = defaultdict(list)
    for modality in modalities:
        spec = specs[modality]
        grouped[(spec.frame_count, spec.image_size, spec.cache_variant)].append(modality)
    return {key: tuple(value) for key, value in grouped.items()}


def extraction_modality_groups(
    modalities: Sequence[str],
    specs: Mapping[str, FeatureCacheSpec],
    group_by_modality: bool = False,
) -> list[tuple[int, int, str | None, tuple[str, ...]]]:
    if group_by_modality:
        return [
            (
                specs[modality].frame_count,
                specs[modality].image_size,
                specs[modality].cache_variant,
                (modality,),
            )
            for modality in modalities
        ]
    return [
        (frame_count, image_size, cache_variant, group_modalities)
        for (
            frame_count,
            image_size,
            cache_variant,
        ), group_modalities in group_modalities_by_clip_spec(modalities, specs).items()
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
    config: Mapping[str, Any] | None = None,
    video_decode_mode: str = "scan",
    clip_cache_dir: Path | None = None,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    rppg_config = config.get("rppg", {}) if isinstance(config, Mapping) else {}
    modality_configs = {
        modality: config.get(modality, {})
        for modality in modalities
        if isinstance(config, Mapping) and isinstance(config.get(modality, {}), Mapping)
    }
    dataset = LabeledVideoDataset(
        examples=examples,
        num_frames=dict.fromkeys(modalities, frame_count),
        image_size=image_size,
        decode_mode=video_decode_mode,
        clip_cache_dir=clip_cache_dir,
        dataset_root=dataset_root,
        image_size_by_modality=dict.fromkeys(modalities, image_size),
        rppg_config=rppg_config if isinstance(rppg_config, Mapping) else {},
        modality_configs=modality_configs,
        global_config=config if isinstance(config, Mapping) else {},
    )
    return collate_labeled_video_batch([dataset[index] for index in range(len(dataset))])


def append_cache_failure(
    failure_rows: list[dict[str, Any]],
    progress: dict[str, Any],
    spec: FeatureCacheSpec,
    example: VideoExample,
    exc: Exception,
    cache_dir: Path | None = None,
    dataset_root: Path | None = None,
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


def _sum_batch_load_seconds(
    raw_batch: Mapping[str, Any], modalities: Sequence[str]
) -> dict[str, float]:
    timings = raw_batch.get("load_timings_by_modality")
    if not isinstance(timings, Mapping):
        return {}
    load_seconds: dict[str, float] = {}
    for modality in modalities:
        values = timings.get(modality)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            load_seconds[modality] = sum(float(value) for value in values)
    return load_seconds


def update_cache_timing_progress(
    progress: dict[str, Any],
    raw_batch: Mapping[str, Any],
    feature_timings: Mapping[str, float],
    modalities: Sequence[str],
) -> None:
    load_seconds = _sum_batch_load_seconds(raw_batch, modalities)
    status_by_modality = raw_batch.get("face_crop_status_by_modality")
    if not isinstance(status_by_modality, Mapping):
        status_by_modality = raw_batch.get("rppg_face_crop_status_by_modality")
    for modality in modalities:
        if modality in load_seconds:
            progress[modality]["load_seconds"] = (
                float(progress[modality].get("load_seconds", 0.0)) + load_seconds[modality]
            )
        if modality in feature_timings:
            progress[modality]["extract_seconds"] = float(
                progress[modality].get("extract_seconds", 0.0)
            ) + float(feature_timings[modality])
        if isinstance(status_by_modality, Mapping) and modality in status_by_modality:
            counts = dict(progress[modality].get("face_crop_status_counts", {}))
            values = status_by_modality[modality]
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for value in values:
                    if value is None:
                        continue
                    key = str(value)
                    counts[key] = counts.get(key, 0) + 1
            progress[modality]["face_crop_status_counts"] = counts


def cache_single_modality_example(
    example: VideoExample,
    cache_dir: Path,
    spec: FeatureCacheSpec,
    config: Mapping[str, Any],
    dataset_root: Path,
    progress: dict[str, Any],
    failure_rows: list[dict[str, Any]],
    fallback_pipelines: dict[str, Any],
    video_decode_mode: str = "scan",
    clip_cache_dir: Path | None = None,
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
            config=config,
            video_decode_mode=video_decode_mode,
            clip_cache_dir=clip_cache_dir,
            dataset_root=dataset_root,
        )
        feature_batch = build_result.pipeline.prepare_features(raw_batch)
        update_cache_timing_progress(
            progress=progress,
            raw_batch=raw_batch,
            feature_timings=build_result.pipeline.last_feature_timings,
            modalities=(modality,),
        )
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
        append_cache_failure(
            failure_rows,
            progress,
            spec,
            example,
            exc,
            cache_dir=cache_dir,
            dataset_root=dataset_root,
        )


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
    assume_missing_cache: bool = False,
    manifest_minimum_rows: int | None = None,
) -> tuple[dict[str, list[VideoExample]], dict[str, Any]]:
    skip_failure_keys = read_failure_keys(cache_dir) if skip_failures else set()
    progress: dict[str, Any] = {}
    progress_interval = max(1, progress_every)
    missing_by_modality: dict[str, list[VideoExample]] = {}
    for modality in modalities:
        spec = specs[modality]
        spec_id = feature_cache_spec_id(spec)
        cache_from = feature_cache_spec_dir(cache_dir, spec)
        if assume_missing_cache:
            missing = [
                example
                for example in examples
                if (spec_id, modality, str(example.path)) not in skip_failure_keys
            ]
        else:
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
                manifest_minimum_rows=manifest_minimum_rows,
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
            "assume_missing_cache": assume_missing_cache,
            "load_seconds": 0.0,
            "extract_seconds": 0.0,
            "face_crop_status_counts": {},
        }
        if assume_missing_cache:
            cached_before = 0
            skipped_failed = len(examples) - len(missing)
        else:
            cached_before, skipped_failed = count_cached_and_skipped(
                examples=examples,
                cache_dir=cache_dir,
                spec=spec,
                dataset_root=dataset_root,
                skip_failure_keys=skip_failure_keys,
                manifest_minimum_rows=manifest_minimum_rows,
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
    video_decode_mode: str = "scan",
    clip_cache_dir: Path | None = None,
) -> None:
    try:
        raw_batch = build_raw_feature_batch(
            examples=batch_examples,
            modalities=group_modalities,
            frame_count=frame_count,
            image_size=image_size,
            config=config,
            video_decode_mode=video_decode_mode,
            clip_cache_dir=clip_cache_dir,
            dataset_root=dataset_root,
        )
        feature_batch = build_result.pipeline.prepare_features(raw_batch)
        update_cache_timing_progress(
            progress=progress,
            raw_batch=raw_batch,
            feature_timings=build_result.pipeline.last_feature_timings,
            modalities=group_modalities,
        )
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
                    video_decode_mode=video_decode_mode,
                    clip_cache_dir=clip_cache_dir,
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
        timing_parts: list[str] = []
        for modality in group_modalities:
            written_modality = max(1, int(progress[modality]["written"]))
            load_seconds = float(progress[modality].get("load_seconds", 0.0))
            extract_seconds = float(progress[modality].get("extract_seconds", 0.0))
            timing_parts.append(
                f"{modality}:load={load_seconds / written_modality:.3f}s "
                f"extract={extract_seconds / written_modality:.3f}s"
            )
            crop_counts = progress[modality].get("face_crop_status_counts")
            if crop_counts:
                timing_parts[-1] = f"{timing_parts[-1]} crop={dict(crop_counts)}"
        print(
            f"cache {label}: modalities={group_name} "
            f"done={done}/{group_examples_count} "
            f"written={written} failed={failed} "
            f"elapsed={elapsed:.1f}s videos_per_s={rate:.2f} "
            f"timings=({' ; '.join(timing_parts)})",
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
    video_decode_mode: str = "scan",
    clip_cache_dir: Path | None = None,
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
                    video_decode_mode=video_decode_mode,
                    clip_cache_dir=clip_cache_dir,
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
    video_decode_mode: str = "scan",
    clip_cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    failure_rows: list[dict[str, Any]] = []
    fallback_pipelines: dict[str, Any] = {}
    try:
        active_modalities = [
            modality for modality in modalities if missing_by_modality.get(modality)
        ]
        for frame_count, image_size, _cache_variant, group_modalities in extraction_modality_groups(
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
                video_decode_mode=video_decode_mode,
                clip_cache_dir=clip_cache_dir,
            )
    finally:
        for build_result in fallback_pipelines.values():
            close_pipeline_result(build_result)
        append_failure_rows(cache_dir, failure_rows)
    return failure_rows


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
    assume_missing_cache: bool = False,
    video_decode_mode: str = "scan",
    clip_cache_dir: Path | None = None,
    manifest_minimum_rows: int | None = None,
    write_manifests: bool = True,
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
        assume_missing_cache=assume_missing_cache,
        manifest_minimum_rows=manifest_minimum_rows,
    )
    failure_rows = cache_missing_feature_groups(
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
        video_decode_mode=video_decode_mode,
        clip_cache_dir=clip_cache_dir,
    )
    for modality in modalities:
        if not write_manifests:
            continue
        spec = specs[modality]
        errors = {
            str(row["path"]): str(row.get("error", ""))
            for row in failure_rows
            if row.get("modality") == modality
        }
        manifest_path = write_feature_cache_manifest(
            cache_dir=cache_dir,
            examples=examples,
            spec=spec,
            dataset_root=dataset_root,
            errors_by_path=errors,
        )
        print(f"cache {label}: wrote manifest {manifest_path}", flush=True)
    return progress


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def feature_cache_manifest_summary(
    cache_dir: Path,
    specs: Mapping[str, FeatureCacheSpec],
    modalities: Sequence[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for modality in modalities:
        spec = specs[modality]
        spec_dir = feature_cache_spec_dir(cache_dir, spec)
        manifest_path = spec_dir / "manifest.csv"
        statuses: Counter[str] = Counter()
        row_count = 0
        if manifest_path.exists():
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row_count += 1
                    statuses[str(row.get("status", ""))] += 1
        pt_count = sum(1 for _ in spec_dir.rglob("*.pt")) if spec_dir.exists() else 0
        summary[modality] = {
            "spec_id": feature_cache_spec_id(spec),
            "spec_dir": str(spec_dir),
            "manifest": str(manifest_path),
            "manifest_exists": manifest_path.exists(),
            "rows": row_count,
            "statuses": dict(statuses),
            "pt_files": pt_count,
            "pt_matches_cached_rows": pt_count == statuses.get("cached", 0),
        }
    return summary


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
            fieldnames=(
                "path",
                "class_name",
                "generator_id",
                "label",
                "prediction",
                "probability",
                "split",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "class_name": row.class_name,
                    "generator_id": row.generator_id,
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


def write_occlusion_rows(path: Path, rows: Sequence[OcclusionRow]) -> None:
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
                "full_prediction",
                "occluded_prediction",
                "full_probability",
                "occluded_probability",
                "split",
                "generator_id",
                "modality_removed",
                "full_logit",
                "occluded_logit",
                "delta_margin",
                "delta_loss",
                "delta_probability",
                "delta_correct_probability",
                "prediction_flipped",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "class_name": row.class_name,
                    "label": row.label,
                    "full_prediction": row.full_prediction,
                    "occluded_prediction": row.occluded_prediction,
                    "full_probability": f"{row.full_probability:.8f}",
                    "occluded_probability": f"{row.occluded_probability:.8f}",
                    "split": row.split,
                    "generator_id": row.generator_id,
                    "modality_removed": row.modality_removed,
                    "full_logit": f"{row.full_logit:.8f}",
                    "occluded_logit": f"{row.occluded_logit:.8f}",
                    "delta_margin": f"{row.delta_margin:.8f}",
                    "delta_loss": f"{row.delta_loss:.8f}",
                    "delta_probability": f"{row.delta_probability:.8f}",
                    "delta_correct_probability": f"{row.delta_correct_probability:.8f}",
                    "prediction_flipped": int(row.prediction_flipped),
                }
            )


def summarize_occlusion_rows(rows: Sequence[OcclusionRow]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        correctness = "correct" if row.label == row.full_prediction else "incorrect"
        key = (row.split, row.class_name, correctness, row.generator_id, row.modality_removed)
        group = groups.setdefault(
            key,
            {
                "split": row.split,
                "class_name": row.class_name,
                "correctness": correctness,
                "generator_id": row.generator_id,
                "modality_removed": row.modality_removed,
                "count": 0,
                "delta_margin_sum": 0.0,
                "abs_delta_margin_sum": 0.0,
                "delta_loss_sum": 0.0,
                "delta_probability_sum": 0.0,
                "delta_correct_probability_sum": 0.0,
                "prediction_flip_count": 0,
            },
        )
        group["count"] += 1
        group["delta_margin_sum"] += row.delta_margin
        group["abs_delta_margin_sum"] += abs(row.delta_margin)
        group["delta_loss_sum"] += row.delta_loss
        group["delta_probability_sum"] += row.delta_probability
        group["delta_correct_probability_sum"] += row.delta_correct_probability
        group["prediction_flip_count"] += int(row.prediction_flipped)

    summaries = []
    for group in groups.values():
        count = int(group["count"])
        summaries.append(
            {
                "split": group["split"],
                "class_name": group["class_name"],
                "correctness": group["correctness"],
                "generator_id": group["generator_id"],
                "modality_removed": group["modality_removed"],
                "count": count,
                "mean_delta_margin": group["delta_margin_sum"] / count,
                "mean_abs_delta_margin": group["abs_delta_margin_sum"] / count,
                "mean_delta_loss": group["delta_loss_sum"] / count,
                "mean_delta_probability": group["delta_probability_sum"] / count,
                "mean_delta_correct_probability": (group["delta_correct_probability_sum"] / count),
                "prediction_flip_rate": group["prediction_flip_count"] / count,
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item["split"],
            item["class_name"],
            item["correctness"],
            item["generator_id"],
            -float(item["mean_abs_delta_margin"]),
            item["modality_removed"],
        ),
    )


def write_occlusion_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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
                "modality_removed",
                "count",
                "mean_delta_margin",
                "mean_abs_delta_margin",
                "mean_delta_loss",
                "mean_delta_probability",
                "mean_delta_correct_probability",
                "prediction_flip_rate",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "mean_delta_margin": f"{float(row['mean_delta_margin']):.8f}",
                    "mean_abs_delta_margin": f"{float(row['mean_abs_delta_margin']):.8f}",
                    "mean_delta_loss": f"{float(row['mean_delta_loss']):.8f}",
                    "mean_delta_probability": f"{float(row['mean_delta_probability']):.8f}",
                    "mean_delta_correct_probability": (
                        f"{float(row['mean_delta_correct_probability']):.8f}"
                    ),
                    "prediction_flip_rate": f"{float(row['prediction_flip_rate']):.8f}",
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
    sharded_cache_dir: Path | None = None,
    all_cache_examples: Sequence[VideoExample] | None = None,
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
    fake_generator_loss_weights = parse_fake_generator_loss_weights(
        args.fake_generator_loss_weights
    )
    loader_config = resolve_cached_loader_config(config)
    sharded_loader_config = resolve_sharded_loader_config(config)
    regularization_config = resolve_training_regularization_config(config, args)
    train_balance_summary = build_train_balance_summary(
        train_examples,
        mode=args.train_balance_mode,
        batch_size=args.batch_size,
        seed=int(args.seed),
    )
    train_batch_sampler = build_train_batch_sampler(
        train_examples,
        batch_size=args.batch_size,
        mode=args.train_balance_mode,
        seed=int(args.seed),
    )
    optimizer = build_optimizer(
        model,
        base_lr=args.lr,
        modality_lrs=modality_lrs,
        weight_decay=args.weight_decay,
    )
    train_loss_fn = build_train_loss_fn(
        train_examples,
        mode=args.train_balance_mode,
        device=build_result.device,
    )
    eval_loss_fn = torch.nn.BCEWithLogitsLoss()

    if sharded_cache_dir is not None:
        if all_cache_examples is None:
            raise ValueError("`all_cache_examples` is required with `sharded_cache_dir`.")
        train_loader = build_sharded_cached_loader(
            examples=train_examples,
            all_cache_examples=all_cache_examples,
            sharded_cache_dir=sharded_cache_dir,
            batch_size=args.batch_size,
            shuffle=True,
            dataset_root=dataset_root,
            loader_config=loader_config,
            sharded_loader_config=sharded_loader_config,
            seed=int(args.seed),
            batch_sampler=train_batch_sampler,
        )
        val_loader = build_sharded_cached_loader(
            examples=val_examples,
            all_cache_examples=all_cache_examples,
            sharded_cache_dir=sharded_cache_dir,
            batch_size=args.batch_size,
            shuffle=False,
            dataset_root=dataset_root,
            loader_config=loader_config,
            sharded_loader_config=sharded_loader_config,
            seed=int(args.seed),
        )
        test_loader = build_sharded_cached_loader(
            examples=test_examples,
            all_cache_examples=all_cache_examples,
            sharded_cache_dir=sharded_cache_dir,
            batch_size=args.batch_size,
            shuffle=False,
            dataset_root=dataset_root,
            loader_config=loader_config,
            sharded_loader_config=sharded_loader_config,
            seed=int(args.seed),
        )
    else:
        train_loader = build_cached_loader(
            train_examples,
            cache_dir,
            specs,
            modalities,
            args.batch_size,
            shuffle=True,
            dataset_root=dataset_root,
            loader_config=loader_config,
            batch_sampler=train_batch_sampler,
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
        f"train_balance={train_balance_summary} "
        f"regularization={asdict(regularization_config)} "
        f"lr={args.lr} modality_lrs={modality_lrs} "
        f"fake_generator_loss_weights={fake_generator_loss_weights} "
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
            train_loss_fn,
            fake_generator_loss_weights=fake_generator_loss_weights,
            regularization_config=regularization_config,
            progress_label=f"train epoch={epoch}",
            progress_every=int(args.progress_every),
        )
        val_result = evaluate_loss_accuracy(
            model,
            val_loader,
            eval_loss_fn,
            progress_label=f"val epoch={epoch}",
            progress_every=int(args.progress_every),
        )
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
    train_accuracy, train_rows = predict_rows(
        model,
        train_loader,
        diagnostic_rows=diagnostic_rows,
        progress_label="predict train",
        progress_every=int(args.progress_every),
    )
    val_accuracy, val_rows = predict_rows(
        model,
        val_loader,
        diagnostic_rows=diagnostic_rows,
        progress_label="predict val",
        progress_every=int(args.progress_every),
    )
    test_accuracy, test_rows = predict_rows(
        model,
        test_loader,
        diagnostic_rows=diagnostic_rows,
        progress_label="predict test",
        progress_every=int(args.progress_every),
    )
    train_metrics = prediction_rows_metrics(train_rows)
    val_metrics = prediction_rows_metrics(val_rows)
    test_metrics = prediction_rows_metrics(test_rows)
    generator_metrics = prediction_generator_metrics([*train_rows, *val_rows, *test_rows])
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
    write_generator_metrics(output_dir / "generator_metrics.csv", generator_metrics)
    diagnostic_summary_rows = summarize_diagnostic_rows(diagnostic_rows)
    write_diagnostics(output_dir / "diagnostics.csv", diagnostic_rows)
    write_diagnostic_summary(output_dir / "diagnostics_summary.csv", diagnostic_summary_rows)
    occlusion_rows: list[OcclusionRow] = []
    occlusion_summary_rows: list[dict[str, Any]] = []
    occlusion_skipped_reason = occlusion_skip_reason(modalities, args.occlusion_diagnostics)
    if args.occlusion_diagnostics and occlusion_skipped_reason is None:
        split_loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        for split_name in args.occlusion_splits:
            occlusion_rows.extend(
                run_occlusion_eval(
                    model=model,
                    loader=split_loaders[str(split_name)],
                    modalities=modalities,
                    split_name=str(split_name),
                )
            )
        occlusion_summary_rows = summarize_occlusion_rows(occlusion_rows)
        write_occlusion_rows(output_dir / "occlusion.csv", occlusion_rows)
        write_occlusion_summary(
            output_dir / "occlusion_summary.csv",
            occlusion_summary_rows,
        )
        print(
            "occlusion: "
            f"splits={','.join(str(split) for split in args.occlusion_splits)} "
            f"modalities={modality_set_name(modalities)} rows={len(occlusion_rows)}",
            flush=True,
        )
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
        "fake_generator_loss_weights": fake_generator_loss_weights,
        "batch_size": args.batch_size,
        "cached_loader": asdict(loader_config),
        "sharded_cache_dir": None if sharded_cache_dir is None else str(sharded_cache_dir),
        "train_balance": train_balance_summary,
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
        "generator_metrics": generator_metrics,
        "generator_metrics_csv": str(output_dir / "generator_metrics.csv"),
        "diagnostics_csv": None if not diagnostic_rows else str(output_dir / "diagnostics.csv"),
        "diagnostics_summary_csv": None
        if not diagnostic_summary_rows
        else str(output_dir / "diagnostics_summary.csv"),
        "occlusion_csv": None if not occlusion_rows else str(output_dir / "occlusion.csv"),
        "occlusion_summary_csv": None
        if not occlusion_summary_rows
        else str(output_dir / "occlusion_summary.csv"),
        "occlusion_splits": list(args.occlusion_splits) if args.occlusion_diagnostics else [],
        "occlusion_skipped_reason": occlusion_skipped_reason,
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


def sharded_cache_progress(
    *,
    label: str,
    examples: Sequence[VideoExample],
    modalities: Sequence[str],
    sharded_cache_dir: Path,
) -> dict[str, Any]:
    return {
        "label": label,
        "requested": len(examples),
        "cached": len(examples),
        "failed": 0,
        "modalities": list(modalities),
        "sharded_cache_dir": str(sharded_cache_dir),
        "skipped_feature_cache_ensure": True,
        "skip_reason": "sharded_cache_dir is set; using prebuilt shards for training.",
    }


def modality_set_name(modalities: Sequence[str]) -> str:
    return "plus".join(modalities)


def main() -> None:
    args = parse_args()
    config = load_pipeline_yaml(args.config)
    args = resolve_training_run_args(config, args)
    if args.device is not None:
        config["device"] = args.device
    torch.manual_seed(args.seed)
    dataset_root = args.dataset_root
    video_root = resolve_video_root(dataset_root)
    cache_dir = args.cache_dir or (dataset_root / "feature_cache")
    clip_cache_dir = None
    if args.clip_cache_dir is not None:
        clip_cache_dir = args.clip_cache_dir
    elif args.enable_clip_cache and not args.no_clip_cache:
        clip_cache_dir = cache_dir / "_clips"
    output_dir = args.output_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    base_modalities = resolve_base_modalities(config, args.modalities)
    modality_sets = build_modality_sets(base_modalities, args.modality_permutations)
    specs = build_feature_cache_specs(config, base_modalities)
    resolved_training_run = training_run_payload(args)
    sharded_cache_keys = (
        read_sharded_cache_keys(args.sharded_cache_dir)
        if args.sharded_cache_dir is not None
        else None
    )

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
    if args.prefer_cached_selection and sharded_cache_keys is not None:
        examples = [
            example
            for example in examples
            if cache_example_key(example, dataset_root) in sharded_cache_keys
        ]
        cached_selection_summary = {
            "dataset_examples": len(dataset_examples),
            "fully_cached_examples": len(examples),
            "modalities": len(base_modalities),
            "sharded_cache_examples": len(sharded_cache_keys),
        }
        print(
            f"shard selection: summary={cached_selection_summary} counts={class_counts(examples)}",
            flush=True,
        )
        if not examples:
            raise ValueError(
                "No examples are present in the configured sharded cache. "
                "Rebuild shards or disable --prefer-cached-selection."
            )
    elif args.prefer_cached_selection:
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
    if args.full_eval_splits and args.balanced_total is not None:
        raise ValueError(
            "--full-eval-splits requires preserved dataset splits; omit --balanced-total."
        )
    cache_score_by_path = None
    if args.balanced_total is None:
        split_mode = "dataset_splits"
        train_pool, val_examples, test_examples = split_examples(
            examples,
            eval_count_per_split=args.eval_count_per_split,
            seed=args.seed,
            cache_score_by_path=cache_score_by_path,
            full_eval_splits=args.full_eval_splits,
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
    raw_train_pool = train_pool
    train_pool, fake_generator_cap_summary = cap_fake_generators(
        train_pool,
        cap_multiplier=args.fake_generator_cap_multiplier,
        seed=args.seed + 307,
        cap_exemptions=args.fake_generator_cap_exemptions,
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
    print(
        f"sharded_cache_dir={args.sharded_cache_dir if args.sharded_cache_dir is not None else '<disabled>'}",
        flush=True,
    )
    print(
        f"clip_cache_dir={clip_cache_dir if clip_cache_dir is not None else '<disabled>'}",
        flush=True,
    )
    print(f"video_decode_mode={args.video_decode_mode}", flush=True)
    print(f"modalities={','.join(base_modalities)}", flush=True)
    print(f"config_path={args.config}", flush=True)
    print(f"cached_loader={asdict(resolve_cached_loader_config(config))}", flush=True)
    print(
        f"regularization={asdict(resolve_training_regularization_config(config, args))}",
        flush=True,
    )
    print(f"train_balance_mode={args.train_balance_mode}", flush=True)
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
    print(
        f"train_pool={len(train_pool)} raw_train_pool={len(raw_train_pool)} "
        f"counts={class_counts(train_pool)} fake_generator_cap={fake_generator_cap_summary}",
        flush=True,
    )
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
    run_train_balance_summary = build_train_balance_summary(
        max_train_examples,
        mode=args.train_balance_mode,
        batch_size=args.batch_size,
        seed=int(args.seed),
    )
    write_dataset_manifest(manifest_examples, output_dir / "manifest.csv")
    write_json(
        output_dir / "run_config.json",
        {
            "dataset_root": str(dataset_root),
            "cache_dir": str(cache_dir),
            "sharded_cache_dir": None
            if args.sharded_cache_dir is None
            else str(args.sharded_cache_dir),
            "config_path": str(args.config),
            "resolved_training_run": resolved_training_run,
            "pipeline_config": config,
            "base_modalities": list(base_modalities),
            "modality_sets": [list(item) for item in modality_sets],
            "round_targets": list(round_targets),
            "round_targets_source": "explicit"
            if args.round_targets is not None
            else args.round_ladder,
            "eval_count_per_split": args.eval_count_per_split,
            "full_eval_splits": args.full_eval_splits,
            "balanced_total": args.balanced_total,
            "train_balance_mode": args.train_balance_mode,
            "train_balance": run_train_balance_summary,
            "fake_generator_cap": fake_generator_cap_summary,
            "fake_generator_loss_weights": parse_fake_generator_loss_weights(
                args.fake_generator_loss_weights
            ),
            "split_mode": split_mode,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "prefer_cached_selection": args.prefer_cached_selection,
            "cached_selection_summary": cached_selection_summary,
            "cached_loader": asdict(resolve_cached_loader_config(config)),
            "regularization": asdict(resolve_training_regularization_config(config, args)),
            "dataset_metadata": video_metadata_summary(manifest_examples),
            "cache_manifest_summary": feature_cache_manifest_summary(
                cache_dir,
                specs,
                base_modalities,
            ),
            "spec_ids": {modality: feature_cache_spec_id(spec) for modality, spec in specs.items()},
            "video_decode_mode": args.video_decode_mode,
            "clip_cache_dir": None if clip_cache_dir is None else str(clip_cache_dir),
            "clip_cache_enabled": clip_cache_dir is not None,
        },
    )

    if args.sharded_cache_dir is None:
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
            video_decode_mode=args.video_decode_mode,
            clip_cache_dir=clip_cache_dir,
            manifest_minimum_rows=len(dataset_examples),
            write_manifests=False,
        )
    else:
        eval_progress = sharded_cache_progress(
            label="eval",
            examples=[*val_examples, *test_examples],
            modalities=base_modalities,
            sharded_cache_dir=args.sharded_cache_dir,
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
        if args.sharded_cache_dir is None:
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
                video_decode_mode=args.video_decode_mode,
                clip_cache_dir=clip_cache_dir,
                manifest_minimum_rows=len(dataset_examples),
                write_manifests=False,
            )
        else:
            cache_progress = sharded_cache_progress(
                label=f"train_{target}",
                examples=train_examples,
                modalities=base_modalities,
                sharded_cache_dir=args.sharded_cache_dir,
            )
        write_json(round_dir / "cache_progress.json", cache_progress)
        print(f"wrote: {round_dir / 'cache_progress.json'}", flush=True)

        for modalities in modality_sets:
            name = modality_set_name(modalities)
            print(f"round={target} modalities={name}", flush=True)
            if sharded_cache_keys is not None:
                cached_train_examples = filter_examples_with_shards(
                    train_examples,
                    sharded_cache_keys,
                    dataset_root,
                    label=f"train_{target}/{name}",
                )
                cached_val_examples = filter_examples_with_shards(
                    val_examples,
                    sharded_cache_keys,
                    dataset_root,
                    label=f"val/{name}",
                )
                cached_test_examples = filter_examples_with_shards(
                    test_examples,
                    sharded_cache_keys,
                    dataset_root,
                    label=f"test/{name}",
                )
            else:
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
            round_warm_start_checkpoint = resolve_warm_start_checkpoint(
                previous,
                enabled=args.warm_start_rounds and args.warm_start_checkpoint is None,
            )
            warm_start_checkpoint = args.warm_start_checkpoint or round_warm_start_checkpoint
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
                sharded_cache_dir=args.sharded_cache_dir,
                all_cache_examples=dataset_examples,
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
        if args.sharded_cache_dir is None:
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
                video_decode_mode=args.video_decode_mode,
                clip_cache_dir=clip_cache_dir,
                manifest_minimum_rows=len(dataset_examples),
                write_manifests=False,
            )
        else:
            sanity_progress = sharded_cache_progress(
                label="sanity",
                examples=sanity_examples,
                modalities=base_modalities,
                sharded_cache_dir=args.sharded_cache_dir,
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
    write_json(
        output_dir / "summary.json",
        {
            "config_path": str(args.config),
            "resolved_training_run": resolved_training_run,
            "dataset_root": str(dataset_root),
            "cache_dir": str(cache_dir),
            "base_modalities": list(base_modalities),
            "cache_manifest_summary": feature_cache_manifest_summary(
                cache_dir,
                specs,
                base_modalities,
            ),
            "rounds": summaries,
            "sanity": sanity_results,
        },
    )
    print(f"wrote: {output_dir / 'summary.json'}", flush=True)
    print(f"wrote: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
